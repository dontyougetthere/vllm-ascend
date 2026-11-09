# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Noncausal C8 attention must preserve storage and match dequantized FP32."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_c8_mxfp import (
    mxfp_k_scale_slot_index,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
    scatter_mxfp_k_scale_cache,
    scatter_mxfp_pa_nz_kv_cache,
)
from vllm_ascend.ops.triton import mxfp_high_precision_attention as hp
from vllm_ascend.ops.triton.mxfp_v_scale import scatter_static_v_scales


def _test_dimensions():
    """Return fresh bounded synthetic-case parameters, not module state."""
    return SimpleNamespace(
        query_heads=64,
        kv_heads=16,
        head_size=64,
        block_size=512,
        prefix_bytes=128,
        padding_bytes=256,
    )


def _mx(tensor):
    quantized, scale = torch_npu.npu_dynamic_mx_quant(tensor.npu(), dst_type=torch.float8_e4m3fn)
    factor = scale.cpu().view(torch.float8_e8m0fnu).float().reshape(*tensor.shape[:-1], 2)
    dequantized = quantized.cpu().float() * factor.repeat_interleave(32, dim=-1)
    return quantized, scale, dequantized


def _reference(query, keys, values, counts, lengths, source_order):
    case_dimensions = _test_dimensions()
    output = torch.zeros_like(query, dtype=torch.float32)
    lse = torch.full(query.shape[:2], -torch.inf)
    start = 0
    for count, length, source in zip(counts, lengths, source_order):
        end = start + count
        if count and length:
            key = keys[source][:length].repeat_interleave(
                case_dimensions.query_heads // case_dimensions.kv_heads, dim=1
            )
            value = values[source][:length].repeat_interleave(
                case_dimensions.query_heads // case_dimensions.kv_heads, dim=1
            )
            scores = torch.einsum("thd,shd->hts", query[start:end], key) / case_dimensions.head_size**0.5
            output[start:end] = torch.einsum("hts,shd->thd", scores.softmax(-1), value)
            lse[start:end] = scores.logsumexp(-1).T
        start = end
    return output, lse


@pytest.mark.parametrize("capacity", [1024, 16384, 32768])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_reader_fp32_reference_and_dynamic_graph(capacity, output_dtype):
    _check_reader_fp32_reference_and_dynamic_graph(capacity, output_dtype, 513)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_reader_streams_long_kv_with_dynamic_graph(output_dtype):
    _check_reader_fp32_reference_and_dynamic_graph(32768, output_dtype, 4097)


def _check_reader_fp32_reference_and_dynamic_graph(capacity, output_dtype, long_length):
    case_dimensions = _test_dimensions()
    generator = torch.Generator(device="cpu").manual_seed(331)
    query = (
        torch.randn((4, case_dimensions.query_heads, case_dimensions.head_size), generator=generator).to(torch.bfloat16)
        * 2
    )
    quantized_query, query_scale, dequantized_query = _mx(query)
    scale_bytes = (torch.arange(case_dimensions.kv_heads * case_dimensions.head_size) % 6 + 124).to(torch.uint8)
    factor = scale_bytes.view(torch.float8_e8m0fnu).float().reshape(case_dimensions.kv_heads, case_dimensions.head_size)
    reciprocal = factor.reciprocal().flatten().to(device="npu", dtype=torch.bfloat16)
    pitch = (
        mxfp_packet_size_bytes(
            case_dimensions.kv_heads, case_dimensions.block_size, case_dimensions.head_size, case_dimensions.head_size
        )
        + case_dimensions.padding_bytes
    )
    raw = torch.full(
        (case_dimensions.prefix_bytes + 12 * pitch + case_dimensions.prefix_bytes,), 93, device="npu", dtype=torch.int8
    )
    cache = mxfp_paged_cache_views(
        raw[case_dimensions.prefix_bytes :],
        12,
        case_dimensions.kv_heads,
        case_dimensions.head_size,
        case_dimensions.head_size,
        case_dimensions.block_size,
        page_stride_bytes=pitch,
    )
    table_cpu = torch.zeros((2, capacity // case_dimensions.block_size), dtype=torch.int32)
    table_cpu[:, :2] = torch.tensor([[1, 4], [3, 2]])
    context_pages = [3, 2, 5, 6, 7, 8, 9, 10, 11][
        : (long_length + case_dimensions.block_size - 1) // case_dimensions.block_size
    ]
    table_cpu[1, : len(context_pages)] = torch.tensor(context_pages)
    table = table_cpu.npu()
    qsl = torch.tensor([0, 3, 4], device="npu", dtype=torch.int32)
    used = torch.tensor([39, long_length], device="npu", dtype=torch.int32)
    dequantized_keys, dequantized_values = [], []
    for request, length in enumerate((39, long_length)):
        key = (
            torch.randn((length, case_dimensions.kv_heads, case_dimensions.head_size), generator=generator).to(
                torch.bfloat16
            )
            * 2
        )
        value = (
            torch.randn((length, case_dimensions.kv_heads, case_dimensions.head_size), generator=generator).to(
                torch.bfloat16
            )
            * 4
        )
        quantized_key, key_scale, dequantized_key = _mx(key)
        quantized_value = torch_npu.npu_quantize(
            value.npu().reshape(length, -1), reciprocal, None, torch.float8_e4m3fn, -1, False
        ).reshape(length, case_dimensions.kv_heads, case_dimensions.head_size)
        positions = torch.arange(length)
        pages = table_cpu[request, positions // case_dimensions.block_size]
        slots = (pages * case_dimensions.block_size + positions % case_dimensions.block_size).to(
            device="npu", dtype=torch.int64
        )
        scatter_mxfp_pa_nz_kv_cache(quantized_key, quantized_value, cache[0], cache[1], slots)
        scatter_mxfp_k_scale_cache(
            key_scale.view(torch.uint8), cache[2], mxfp_k_scale_slot_index(slots, case_dimensions.block_size)
        )
        scatter_static_v_scales(scale_bytes.npu(), cache[3], slots)
        dequantized_value = quantized_value.cpu().float() * factor
        # Distinguish the two 32-token V-scale groups in each 64-token block.
        dequantized_value *= torch.where(positions % 64 >= 32, 2.0, 1.0)[:, None, None]
        dequantized_keys.append(dequantized_key)
        dequantized_values.append(dequantized_value)
    odd_scale = cache[3][..., 1]
    odd_scale.copy_((odd_scale.to(torch.int16) + 1).clamp(0, 254).to(torch.uint8))
    before = raw.cpu()

    def forward():
        case_dimensions = _test_dimensions()
        return hp.read(
            quantized_query,
            query_scale,
            cache,
            table,
            qsl,
            used,
            case_dimensions.head_size**-0.5,
            output_dtype=output_dtype,
        )

    for _ in range(2):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output, graph_lse = forward()
    variants = [
        ([3, 1], [39, long_length], [0, 1]),
        ([1, 3], [long_length, 39], [1, 0]),
        ([0, 3], [0, long_length], [0, 1]),
        ([3, 1], [0, 39], [0, 0]),
    ]
    for counts, lengths, order in variants:
        cumulative = [0, counts[0], sum(counts)]
        qsl.copy_(torch.tensor(cumulative, device="npu", dtype=torch.int32))
        used.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        table.copy_(table_cpu[order].npu())
        eager, eager_lse = forward()
        eager_cpu, eager_lse_cpu = eager.cpu(), eager_lse.cpu()
        graph.replay()
        torch.npu.synchronize()
        actual, actual_lse = graph_output.cpu(), graph_lse.cpu()
        reference, reference_lse = _reference(
            dequantized_query, dequantized_keys, dequantized_values, counts, lengths, order
        )
        torch.testing.assert_close(actual, eager_cpu, atol=0, rtol=0)
        torch.testing.assert_close(actual_lse, eager_lse_cpu, atol=0, rtol=0)
        assert torch.isfinite(actual).all()
        assert not ((actual.float() - reference).abs() > 0.005 + 0.03 * reference.abs()).any()
        assert torch.equal(torch.isneginf(actual_lse), torch.isneginf(reference_lse))
        finite = torch.isfinite(reference_lse)
        assert not (
            (actual_lse[finite] - reference_lse[finite]).abs() > 2e-5 + 1e-5 * reference_lse[finite].abs()
        ).any()
        torch.testing.assert_close(raw.cpu(), before, atol=0, rtol=0)

    if capacity == 1024 and output_dtype == torch.bfloat16:
        bad_scale = cache[2].clone()
        bad_scale.copy_((bad_scale.to(torch.int16) + 3).clamp(0, 254).to(torch.uint8))
        wrong, _ = hp.read(
            quantized_query,
            query_scale,
            (cache[0], cache[1], bad_scale, cache[3]),
            table,
            qsl,
            used,
            case_dimensions.head_size**-0.5,
        )
        assert ((wrong.cpu().float() - reference).abs() > 0.005 + 0.03 * reference.abs()).any()


@pytest.mark.parametrize("invalid", ["causal", "query_start_shape", "table_dtype"])
def test_reader_rejects_unsupported_contract_before_launch(invalid):
    case_dimensions = _test_dimensions()
    raw = torch.zeros(
        2
        * mxfp_packet_size_bytes(
            case_dimensions.kv_heads, case_dimensions.block_size, case_dimensions.head_size, case_dimensions.head_size
        ),
        device="npu",
        dtype=torch.int8,
    )
    cache = mxfp_paged_cache_views(
        raw,
        2,
        case_dimensions.kv_heads,
        case_dimensions.head_size,
        case_dimensions.head_size,
        case_dimensions.block_size,
    )
    query, scale, _ = _mx(
        torch.zeros((1, case_dimensions.query_heads, case_dimensions.head_size), dtype=torch.bfloat16)
    )
    qsl = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    table = torch.zeros((1, 1), device="npu", dtype=torch.int32)
    lengths = torch.ones(1, device="npu", dtype=torch.int32)
    if invalid == "query_start_shape":
        qsl = qsl[:1]
    if invalid == "table_dtype":
        table = table.to(torch.int64)
    exception = NotImplementedError if invalid == "causal" else AssertionError
    with pytest.raises(exception):
        hp.read(query, scale, cache, table, qsl, lengths, case_dimensions.head_size**-0.5, causal=invalid == "causal")
