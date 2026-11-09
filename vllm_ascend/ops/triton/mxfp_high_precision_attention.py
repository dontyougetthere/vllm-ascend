# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""FP32 noncausal GQA attention over packet-packed MXFP8 KV pages.

Q retains dynamic MXFP8 quantization. K/V and their E8M0 scales are read
directly from the existing strided cache, without a full-cache dequantization.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fp8(byte):
    raw = byte.to(tl.int32)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    magnitude = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * (1.0 / 512.0),
        (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0),
    )
    value = tl.where((raw & 128) != 0, -magnitude, magnitude)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)


@triton.jit
def _scale(byte):
    raw = byte.to(tl.int32)
    return tl.where(raw == 255, float("nan"), tl.exp2(raw.to(tl.float32) - 127.0))


@triton.jit
def _partial(
    Q,
    QS,
    K,
    V,
    KS,
    VS,
    TABLE,
    QSL,
    LENGTHS,
    PART,
    STATS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    NR: tl.constexpr,
    RN: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    TB: tl.constexpr,
    KP: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    KT: tl.constexpr,
    VP: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    VT: tl.constexpr,
    KSP: tl.constexpr,
    KSH: tl.constexpr,
    KST: tl.constexpr,
    KSD: tl.constexpr,
    KSI: tl.constexpr,
    VSP: tl.constexpr,
    VSH: tl.constexpr,
    VSD: tl.constexpr,
    VST: tl.constexpr,
    VSI: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    TILE: tl.constexpr,
):
    token_head = tl.program_id(0)
    split = tl.program_id(1)
    token, head = token_head // HQ, token_head % HQ
    kv_head = head // (HQ // HK)
    requests = tl.arange(0, RN)
    starts = tl.load(QSL + requests, requests < NR, other=2147483647)
    request = tl.sum(((token >= starts) & (requests < NR)).to(tl.int32), 0) - 1
    request = tl.maximum(request, 0)
    qstart = tl.load(QSL + request)
    qend = tl.load(QSL + request + 1)
    valid_query = token < tl.load(QSL + NR)
    length = tl.load(LENGTHS + request)
    dims = tl.arange(0, 64)
    query_bits = tl.load(Q + token * QB + head * QH + dims, valid_query, other=0)
    query_scale = tl.load(QS + (token * HQ + head) * 2 + dims // 32, valid_query, other=127)
    query = _fp8(query_bits) * _scale(query_scale)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    result = tl.full((64,), 0.0, tl.float32)
    effective_length = tl.where(valid_query, length, 0)
    for tile_index in tl.range(split, tl.cdiv(effective_length, TILE), SPLITS):
        rows = tile_index * TILE + tl.arange(0, TILE)
        active = valid_query & (rows < length)
        if CAUSAL:
            active = active & (rows <= length - (qend - qstart) + token - qstart)
        page = tl.load(TABLE + request * TB + rows // BLOCK, active, other=0).to(tl.int64)
        position = rows % BLOCK
        key_bits = tl.load(
            K
            + page[:, None] * KP
            + kv_head * KH
            + (dims[None, :] // 32) * KD
            + position[:, None] * KT
            + dims[None, :] % 32,
            active[:, None],
            other=0,
        )
        key_scale = tl.load(
            KS
            + page[:, None] * KSP
            + kv_head * KSH
            + (position[:, None] // 16) * KST
            + (dims[None, :] // 64) * KSD
            + (position[:, None] % 16) * KSI
            + (dims[None, :] % 64) // 32,
            active[:, None],
            other=127,
        )
        keys = _fp8(key_bits) * _scale(key_scale)
        scores = tl.sum(keys * query[None, :], 1) * SOFTMAX_SCALE
        scores = tl.where(active, scores, -float("inf"))
        tile_maximum = tl.max(scores, 0)
        next_maximum = tl.maximum(maximum, tile_maximum)
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        old_weight = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum)
        denominator = denominator * old_weight + tl.sum(probabilities, 0)
        maximum = next_maximum
        value_bits = tl.load(
            V
            + page[:, None] * VP
            + kv_head * VH
            + (dims[None, :] // 32) * VD
            + position[:, None] * VT
            + dims[None, :] % 32,
            active[:, None],
            other=0,
        )
        value_scale = tl.load(
            VS
            + page[:, None] * VSP
            + kv_head * VSH
            + (dims[None, :] // 16) * VSD
            + (position[:, None] // 64) * VST
            + (dims[None, :] % 16) * VSI
            + (position[:, None] % 64) // 32,
            active[:, None],
            other=127,
        )
        values = _fp8(value_bits) * _scale(value_scale)
        result = result * old_weight + tl.sum(probabilities[:, None] * values, 0)
    tl.store(PART + (token_head * SPLITS + split) * 64 + dims, result)
    tl.store(STATS + (token_head * SPLITS + split) * 2, maximum)
    tl.store(STATS + (token_head * SPLITS + split) * 2 + 1, denominator)


@triton.jit
def _merge(PART, STATS, OUT, LSE, SPLITS: tl.constexpr, NS: tl.constexpr, WIDTH: tl.constexpr):
    token_head = tl.program_id(0)
    splits = tl.arange(0, NS)
    dims = tl.program_id(1) * WIDTH + tl.arange(0, WIDTH)
    maximum = tl.load(STATS + (token_head * SPLITS + splits) * 2, splits < SPLITS, other=-float("inf"))
    denominators = tl.load(STATS + (token_head * SPLITS + splits) * 2 + 1, splits < SPLITS, other=0.0)
    global_maximum = tl.max(maximum, 0)
    safe_maximum = tl.where(global_maximum == -float("inf"), 0.0, global_maximum)
    weights = tl.exp(maximum - safe_maximum)
    denominator = tl.sum(weights * denominators, 0)
    partials = tl.load(
        PART + (token_head * SPLITS + splits[:, None]) * 64 + dims[None, :], splits[:, None] < SPLITS, other=0.0
    )
    value = tl.sum(partials * weights[:, None], 0) / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + token_head * 64 + dims, value)
    if tl.program_id(1) == 0:
        tl.store(LSE + token_head, tl.where(denominator > 0, tl.log(denominator) + global_maximum, -float("inf")))


def read(
    quant_query,
    query_scale,
    cache,
    table,
    query_start_loc,
    lengths,
    softmax_scale,
    *,
    causal=False,
    tile=32,
    output_dtype=torch.bfloat16,
):
    """Stream actual device KV lengths with bounded partial storage."""
    if causal:
        raise NotImplementedError("The high-precision MXFP8 reader currently supports noncausal attention only.")
    key, value, key_scale, value_scale = cache
    assert quant_query.ndim == 3 and quant_query.shape[-1] == 64
    assert key.ndim == value.ndim == 5 and key.shape == value.shape
    assert key.shape[2] == 2 and key.shape[-1] == 32
    assert quant_query.dtype == key.dtype == value.dtype == torch.float8_e4m3fn
    assert quant_query.stride(-1) == 1
    assert query_scale.is_contiguous() and query_scale.numel() == quant_query.shape[0] * quant_query.shape[1] * 2
    assert key_scale.ndim == value_scale.ndim == 6 and key_scale.stride(-1) == value_scale.stride(-1) == 1
    assert key.stride(-1) == value.stride(-1) == 1
    assert table.ndim == 2 and table.stride(-1) == 1 and table.dtype == torch.int32
    requests = table.shape[0]
    assert query_start_loc.numel() == requests + 1 and lengths.numel() == requests
    assert query_start_loc.is_contiguous() and lengths.is_contiguous()
    assert tile in (16, 32, 64) and output_dtype in (torch.float32, torch.bfloat16)
    assert all(t.device == quant_query.device for t in (*cache, query_scale, table, query_start_loc, lengths))
    tokens, heads, _ = quant_query.shape
    assert heads % key.shape[1] == 0
    block = key.shape[3]
    if table.shape[1] == 0:
        raise ValueError("MXFP8 attention requires a nonempty page-table capacity.")
    # This reader intentionally performs one streaming FP32 reduction per
    # token/head. Page-table capacity must not introduce extra reductions or
    # change accumulation order. There is no configurable partition budget.
    splits = 1
    merge_splits = splits
    partial = torch.empty((tokens, heads, splits, 64), device=quant_query.device, dtype=torch.float32)
    stats = torch.empty((tokens, heads, splits, 2), device=quant_query.device, dtype=torch.float32)
    output = torch.empty((tokens, heads, 64), device=quant_query.device, dtype=output_dtype)
    lse = torch.empty((tokens, heads), device=quant_query.device, dtype=torch.float32)
    _partial[(tokens * heads, splits)](
        quant_query.view(torch.uint8),
        query_scale.view(torch.uint8),
        key.view(torch.uint8),
        value.view(torch.uint8),
        key_scale.view(torch.uint8),
        value_scale.view(torch.uint8),
        table,
        query_start_loc,
        lengths,
        partial,
        stats,
        heads,
        key.shape[1],
        requests,
        triton.next_power_of_2(requests),
        splits,
        block,
        quant_query.stride(0),
        quant_query.stride(1),
        table.stride(0),
        *key.stride()[:4],
        *value.stride()[:4],
        *key_scale.stride()[:5],
        *value_scale.stride()[:5],
        float(softmax_scale),
        bool(causal),
        tile,
    )
    # Normalize the single accumulator across the actual supported head width.
    merge_width = quant_query.shape[-1]
    _merge[(tokens * heads, 64 // merge_width)](partial, stats, output, lse, splits, merge_splits, merge_width)
    return output, lse
