"""Restore static V scales only for addressed PA_NZ token groups."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_v_scale_groups(
    source,
    cache,
    sorted_groups,
    num_groups,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    groups_per_block: tl.constexpr,
    num_blocks: tl.constexpr,
    block_stride: tl.constexpr,
    head_stride: tl.constexpr,
    channel_fragment_stride: tl.constexpr,
    token_group_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    pair_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.load(sorted_groups + token).to(tl.int64)
    previous = tl.load(sorted_groups + token - 1, mask=token > 0, other=-2).to(tl.int64)
    unique = (group >= 0) & (group < num_blocks * groups_per_block) & (group != previous)
    index = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = unique & (index < num_heads * head_dim * 2)
    head = index // (head_dim * 2)
    channel = index // 2 % head_dim
    pair = index % 2
    offset = (
        (group // groups_per_block) * block_stride
        + head.to(tl.int64) * head_stride
        + (channel // 16).to(tl.int64) * channel_fragment_stride
        + (group % groups_per_block) * token_group_stride
        + (channel % 16).to(tl.int64) * channel_stride
        + pair.to(tl.int64) * pair_stride
    )
    value = tl.load(source + head * head_dim + channel, mask=valid, other=0)
    tl.store(cache + offset, value, mask=valid)


def scatter_static_v_scales(value_scale, value_scale_cache, slot_mapping):
    """Restore only addressed 64-token groups, skipping negative padded slots.

    Sorting group IDs gives each group exactly one writer even for unordered,
    duplicated slots. Shapes stay static; there are no device-to-host reads.
    The static per-channel scale must be shared by all tokens in each group.
    """
    assert value_scale_cache.ndim == 6 and value_scale_cache.dtype == torch.uint8
    assert value_scale.dtype == torch.uint8
    assert value_scale_cache.shape[4:] == (16, 2)
    assert slot_mapping.ndim == 1
    heads = value_scale_cache.shape[1]
    head_dim = value_scale_cache.shape[2] * 16
    assert value_scale.numel() == heads * head_dim
    assert value_scale.device == value_scale_cache.device == slot_mapping.device
    if slot_mapping.numel() == 0:
        return
    groups = torch.div(slot_mapping.to(torch.int64), 64, rounding_mode="floor").sort().values
    source = value_scale.reshape(-1).contiguous()
    block = 256
    _scatter_v_scale_groups[(groups.numel(), triton.cdiv(heads * head_dim * 2, block))](
        source,
        value_scale_cache,
        groups,
        groups.numel(),
        heads,
        head_dim,
        value_scale_cache.shape[3],
        value_scale_cache.shape[0],
        *value_scale_cache.stride(),
        BLOCK_SIZE=block,
        num_warps=4,
    )
