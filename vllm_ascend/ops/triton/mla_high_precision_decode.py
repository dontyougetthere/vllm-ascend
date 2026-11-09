# SPDX-License-Identifier: Apache-2.0
"""FP8 MLA cache reader with FP32 QK, softmax and PV; no full-cache copy."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_to_fp32(byte):
    u = byte.to(tl.int32)
    exponent = (u >> 3) & 15
    mantissa = u & 7
    magnitude = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * (1.0 / 512.0),
        (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0),
    )
    signed = tl.where((u & 128) != 0, -magnitude, magnitude)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), signed)


@triton.jit
def _partial(
    Q,
    QR,
    K,
    KR,
    QS,
    KS,
    VS,
    TABLE,
    LENGTH,
    PART,
    STATS,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    BT: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    RB: tl.constexpr,
    RH: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KRB: tl.constexpr,
    KRT: tl.constexpr,
    TB: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = split * TILE + tl.arange(0, TILE)
    ds = tl.arange(0, 512)
    rs = tl.arange(0, 64)
    length = tl.load(LENGTH + b)
    page = tl.load(TABLE + b * TB + rows // BT, rows < length, other=0).to(tl.int64)
    kv = _fp8_to_fp32(
        tl.load(K + page[:, None] * KB + (rows % BT)[:, None] * KT + ds[None, :], rows[:, None] < length, other=0)
    )
    rv = tl.load(
        KR + page[:, None] * KRB + (rows % BT)[:, None] * KRT + rs[None, :], rows[:, None] < length, other=0
    ).to(tl.float32)
    q = _fp8_to_fp32(tl.load(Q + b * QB + h * QH + ds))
    qr = tl.load(QR + b * RB + h * RH + rs).to(tl.float32)
    qs = tl.load(QS + bh)
    ks = tl.load(KS)
    vs = tl.load(VS)
    scores = (tl.sum(kv * q[None, :], 1) + tl.sum(rv * qr[None, :], 1)) * (qs * ks * SCALE)
    scores = tl.where(rows < length, scores, -float("inf"))
    mx = tl.max(scores, 0)
    safe_mx = tl.where(split * TILE < length, mx, 0.0)
    p = tl.exp(scores - safe_mx)
    den = tl.sum(p, 0)
    result = tl.sum(p[:, None] * kv, 0) * vs
    tl.store(PART + (bh * SPLITS + split) * 512 + ds, result)
    tl.store(STATS + (bh * SPLITS + split) * 2, mx)
    tl.store(STATS + (bh * SPLITS + split) * 2 + 1, den)


@triton.jit
def _merge(PART, STATS, OUT, LSE, SPLITS: tl.constexpr, NS: tl.constexpr, D: tl.constexpr):
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    ss = tl.arange(0, NS)
    dd = dc * D + tl.arange(0, D)
    mx = tl.load(STATS + (bh * SPLITS + ss) * 2, ss < SPLITS, other=-float("inf"))
    den = tl.load(STATS + (bh * SPLITS + ss) * 2 + 1, ss < SPLITS, other=0.0)
    maximum = tl.max(mx, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    w = tl.exp(mx - safe_max)
    total = tl.sum(w * den, 0)
    partial = tl.load(PART + (bh * SPLITS + ss[:, None]) * 512 + dd[None, :], ss[:, None] < SPLITS, other=0.0)
    value = tl.sum(partial * w[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(OUT + bh * 512 + dd, value)
    if dc == 0:
        tl.store(LSE + bh, tl.where(total > 0, tl.log(total) + maximum, -float("inf")))


def decode(q, qr, k, kr, qs, ks, vs, table, lengths, softmax_scale, tile=32, out_dtype=torch.bfloat16):
    assert q.ndim == 3 and q.shape[-1] == 512 and qr.shape == (*q.shape[:-1], 64)
    assert k.ndim == 4 and k.shape[1] == 1 and k.shape[-1] == 512
    assert kr.shape == (*k.shape[:-1], 64)
    assert q.dtype == k.dtype == torch.float8_e4m3fn and qr.dtype == kr.dtype == torch.bfloat16
    assert q.stride(-1) == qr.stride(-1) == k.stride(-1) == kr.stride(-1) == 1
    assert qs.is_contiguous() and qs.numel() == q.shape[0] * q.shape[1] and ks.numel() == vs.numel() == 1
    assert table.ndim == 2 and table.shape[0] == q.shape[0] and table.stride(1) == 1
    assert lengths.numel() == q.shape[0] and lengths.is_contiguous()
    assert table.dtype == torch.int32 and lengths.dtype in (torch.int32, torch.int64)
    assert tile in (16, 32, 64) and all(t.device == q.device for t in (qr, k, kr, qs, ks, vs, table, lengths))
    b, h, _ = q.shape
    splits = triton.cdiv(table.shape[1] * k.shape[2], tile)
    part = torch.empty((b, h, splits, 512), device=q.device, dtype=torch.float32)
    stats = torch.empty((b, h, splits, 2), device=q.device, dtype=torch.float32)
    assert out_dtype in (torch.bfloat16, torch.float32)
    out = torch.empty_like(q, dtype=out_dtype)
    lse = torch.empty((b, h, 1), device=q.device, dtype=torch.float32)
    _partial[(b * h, splits)](
        q.view(torch.uint8),
        qr,
        k.view(torch.uint8),
        kr,
        qs,
        ks,
        vs,
        table,
        lengths,
        part,
        stats,
        h,
        splits,
        k.shape[2],
        q.stride(0),
        q.stride(1),
        qr.stride(0),
        qr.stride(1),
        k.stride(0),
        k.stride(2),
        kr.stride(0),
        kr.stride(2),
        table.stride(0),
        float(softmax_scale),
        tile,
    )
    merge_splits = triton.next_power_of_2(splits)
    # Bound the split-by-output tile to 16K FP32 values as context grows.
    merge_width = min(64, max(1, 16384 // merge_splits))
    _merge[(b * h, 512 // merge_width)](part, stats, out, lse, splits, merge_splits, merge_width)
    return out, lse
