# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component-major mixed MLA cache views and byte-preserving stores."""

import math

import torch
from vllm.utils.torch_utils import get_dtype_size


def reshape_mixed_mla_cache(
    raw_cache: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    raw_key_shape: tuple[int, int, int, int],
    latent_dtype: torch.dtype,
    raw_key_dtype: torch.dtype,
    page_stride_bytes: int,
    kernel_blocks_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """View each kernel slot as [latent][raw keys][padding] in a shared byte pool."""
    if raw_cache.ndim != 1 or raw_cache.dtype not in (torch.int8, torch.uint8) or not raw_cache.is_contiguous():
        raise ValueError("Mixed MLA views require a contiguous one-dimensional byte tensor.")
    if latent_dtype != torch.float8_e4m3fn or raw_key_dtype != torch.bfloat16:
        raise ValueError("Static mixed MLA cache requires FP8 E4M3 latent and BF16 raw keys.")
    if (
        len(latent_shape) != 4
        or len(raw_key_shape) != 4
        or latent_shape[:3] != raw_key_shape[:3]
        or min(*latent_shape, *raw_key_shape) < 1
    ):
        raise ValueError("Mixed MLA components must have positive, matching block/token/head dimensions.")
    if (
        kernel_blocks_per_page < 1
        or latent_shape[0] % kernel_blocks_per_page
        or page_stride_bytes <= 0
        or page_stride_bytes % kernel_blocks_per_page
    ):
        raise ValueError("Mixed MLA manager pages must contain complete kernel slots.")

    latent_element_bytes = get_dtype_size(latent_dtype)
    raw_key_element_bytes = get_dtype_size(raw_key_dtype)
    slot_stride_bytes = page_stride_bytes // kernel_blocks_per_page
    latent_block_bytes = math.prod(latent_shape[1:]) * latent_element_bytes
    raw_key_block_bytes = math.prod(raw_key_shape[1:]) * raw_key_element_bytes
    base_offset_bytes = raw_cache.storage_offset() * raw_cache.element_size()
    alignment = math.lcm(latent_element_bytes, raw_key_element_bytes)
    if (
        slot_stride_bytes % alignment
        or base_offset_bytes % alignment
        or raw_cache.numel() % alignment
        or latent_block_bytes % raw_key_element_bytes
    ):
        raise ValueError("Mixed MLA component offsets, slice sizes and kernel strides must be dtype-aligned.")
    if latent_block_bytes + raw_key_block_bytes > slot_stride_bytes:
        raise ValueError("Mixed MLA components exceed their kernel slot.")
    if latent_shape[0] * slot_stride_bytes > raw_cache.numel() * raw_cache.element_size():
        raise ValueError("Mixed MLA kernel slots exceed the layer's byte slice.")

    def component_view(
        shape: tuple[int, int, int, int], dtype: torch.dtype, component_offset_bytes: int
    ) -> torch.Tensor:
        element_bytes = get_dtype_size(dtype)
        dense_strides = tuple(math.prod(shape[dimension + 1 :]) for dimension in range(len(shape)))
        return torch.as_strided(
            raw_cache.view(dtype),
            size=shape,
            stride=(slot_stride_bytes // element_bytes, *dense_strides[1:]),
            storage_offset=(base_offset_bytes + component_offset_bytes) // element_bytes,
        )

    return (
        component_view(latent_shape, latent_dtype, 0),
        component_view(raw_key_shape, raw_key_dtype, latent_block_bytes),
    )


def _validate_cache_write_inputs(
    latent: torch.Tensor,
    raw_keys: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    slot_mapping: torch.Tensor,
) -> None:
    if len(kv_cache) != 2 or any(component.ndim != 4 for component in kv_cache):
        raise ValueError("Mixed MLA cache must contain two four-dimensional components.")
    if kv_cache[0].dtype != torch.float8_e4m3fn or kv_cache[1].dtype != torch.bfloat16:
        raise ValueError("Mixed MLA cache must contain FP8 E4M3 latent and BF16 raw keys.")
    if kv_cache[0].shape[:3] != kv_cache[1].shape[:3] or kv_cache[0].shape[2] != 1:
        raise ValueError("Mixed MLA stores require matching single-head cache geometry.")
    if (
        slot_mapping.ndim != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or not slot_mapping.is_contiguous()
    ):
        raise ValueError("Mixed MLA slot mapping must be a contiguous one-dimensional integer tensor.")
    for source, cache in zip((latent, raw_keys), kv_cache):
        component_width = cache.shape[-1]
        if source.dtype != torch.bfloat16 or source.shape != (slot_mapping.numel(), 1, component_width):
            raise ValueError("Mixed MLA current components must be BF16 with one row per slot.")
        if (
            cache.stride(-1) != 1
            or cache.stride(1) != component_width
            or cache.stride(0) < cache.shape[1] * component_width
        ):
            raise ValueError("Mixed MLA stores require component-major PA cache strides.")
        if cache.device.type != "npu" or not (source.device == cache.device == slot_mapping.device):
            raise ValueError("Mixed MLA components, cache and slots must use the same NPU device.")


def validate_mixed_mla_quantization(
    latent_descale: torch.Tensor,
    latent_quant_scale: torch.Tensor,
    key_offset: torch.Tensor,
    value_descale: torch.Tensor,
    value_offset: torch.Tensor,
) -> None:
    """Validate loaded scalar parameters once, before forward or graph capture.

    The shared latent representation uses the K descale for both K and V.
    The separately exported V observer scale is checked for validity without
    requiring equality or substituting it for the shared latent descale.
    """
    scales = (latent_descale, latent_quant_scale, value_descale)
    if any(scale.numel() != 1 for scale in scales):
        raise ValueError("Mixed MLA quantization requires scalar scales.")
    if latent_descale.dtype != torch.float32 or latent_quant_scale.dtype != torch.float32:
        raise ValueError("Mixed MLA latent scales must use FP32.")
    if any(param.device != latent_descale.device for param in (*scales, key_offset, value_offset)):
        raise ValueError("Mixed MLA quantization parameters must share a device.")
    # These transfers happen only during weight processing, never per token.
    values = [scale.detach().float().cpu() for scale in scales]
    if any(not bool(torch.isfinite(value).all() and (value > 0).all()) for value in values):
        raise ValueError("Mixed MLA scales must be finite and strictly positive.")
    if not torch.allclose(values[0] * values[1], torch.ones_like(values[0]), atol=1e-6, rtol=1e-6):
        raise ValueError("Mixed MLA quantization scale must be the latent descale reciprocal.")
    for offset in (key_offset, value_offset):
        if offset.numel() != 1 or not bool((offset.detach().float().cpu() == 0).all()):
            raise ValueError("Mixed MLA FP8 supports only scalar zero offsets.")


def write_mixed_mla_cache(
    latent: torch.Tensor,
    raw_keys: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    slot_mapping: torch.Tensor,
    latent_quant_scale: torch.Tensor,
) -> None:
    """Statically quantize cached latent; preserve raw keys and current BF16 KV.

    ``latent_quant_scale`` is the checkpoint descale reciprocal, already loaded
    as a positive finite FP32 scalar. FIA and prefill context readers apply its
    inverse. The scheduler supplies unique, in-range nonnegative slots;
    negative slots denote padding.
    """
    from vllm_ascend.ops.triton.mla_mixed_cache_store import store_mla_cache_components

    _validate_cache_write_inputs(latent, raw_keys, kv_cache, slot_mapping)
    if slot_mapping.numel() == 0:
        return
    if (
        latent_quant_scale.numel() != 1
        or latent_quant_scale.dtype != torch.float32
        or latent_quant_scale.device != latent.device
    ):
        raise ValueError("Mixed MLA static quantization needs one FP32 scale on the latent device.")
    fp8_max = torch.finfo(kv_cache[0].dtype).max
    cache_latent = (latent.float() * latent_quant_scale).clamp(-fp8_max, fp8_max).to(kv_cache[0].dtype)
    store_mla_cache_components((cache_latent, raw_keys), kv_cache, slot_mapping)
