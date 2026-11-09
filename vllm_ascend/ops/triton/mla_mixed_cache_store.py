# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store current MLA components into their own block-strided cache views."""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_component_bytes(
    source_bytes,
    slot_mapping,
    cache_bytes,
    CACHE_TOKEN_CAPACITY: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_BYTES: tl.constexpr,
    TOKEN_STRIDE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    byte_offset = tl.arange(0, TILE_BYTES)
    slot = tl.load(slot_mapping + row).to(tl.int64)
    valid_slot = (slot >= 0) & (slot < CACHE_TOKEN_CAPACITY)
    value = tl.load(source_bytes + row * ROW_BYTES + byte_offset, byte_offset < ROW_BYTES, other=0)
    cache_offset_bytes = (
        (slot // CACHE_BLOCK_SIZE) * BLOCK_STRIDE_BYTES + (slot % CACHE_BLOCK_SIZE) * TOKEN_STRIDE_BYTES + byte_offset
    )
    tl.store(cache_bytes + cache_offset_bytes, value, valid_slot & (byte_offset < ROW_BYTES))


def store_mla_cache_components(
    current_components: tuple[torch.Tensor, torch.Tensor],
    cache_components: tuple[torch.Tensor, torch.Tensor],
    slot_mapping: torch.Tensor,
) -> None:
    """Copy prevalidated component rows without converting FP8 or BF16 values.

    The caller validates dtype, shape, devices and strides. Valid slots must be
    unique and within the cache capacity; negative slots are padding. The upper
    bound mask prevents unsafe writes and is not a substitute for that scheduler
    invariant. Only current token rows are materialized, never history pages.
    """
    for current_component, cache_component in zip(current_components, cache_components):
        source_bytes = current_component.contiguous().view(torch.uint8)
        cache_bytes = cache_component.view(torch.uint8)
        row_bytes = source_bytes.shape[-1]
        _store_component_bytes[(slot_mapping.numel(),)](
            source_bytes,
            slot_mapping,
            cache_bytes,
            cache_component.shape[0] * cache_component.shape[1],
            cache_component.shape[1],
            cache_bytes.stride(0),
            cache_bytes.stride(1),
            row_bytes,
            triton.next_power_of_2(row_bytes),
        )
