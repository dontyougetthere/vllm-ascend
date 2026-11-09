# SPDX-License-Identifier: Apache-2.0
"""CPU byte-layout checks: owner aliasing, padded slots and fail-closed guards."""

import pytest
import torch

from vllm_ascend.attention.mla_mixed_cache import reshape_mixed_mla_cache


@pytest.mark.parametrize("kernel_blocks", [1, 2, 4])
def test_mixed_views_preserve_byte_owner_and_padding(kernel_blocks):
    pages, tokens, latent_width, rope_width = 2, 3, 4, 2
    blocks = pages * kernel_blocks
    latent_bytes = tokens * latent_width
    rope_bytes = tokens * rope_width * 2
    slot_bytes = latent_bytes + rope_bytes + 8
    prefix = 16
    backing = torch.full((prefix + blocks * slot_bytes + prefix,), 93, dtype=torch.uint8)
    before = backing.clone()
    latent, rope = reshape_mixed_mla_cache(
        backing[prefix:-prefix],
        (blocks, 1, tokens, latent_width),
        (blocks, 1, tokens, rope_width),
        torch.float8_e4m3fn,
        torch.bfloat16,
        slot_bytes * kernel_blocks,
        kernel_blocks,
    )
    for view in (latent, rope):
        assert view.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    latent_data = (
        torch.arange(blocks * latent_bytes, dtype=torch.int64).to(torch.uint8).reshape(blocks, 1, tokens, latent_width)
    )
    rope_data = (
        (torch.arange(blocks * rope_bytes, dtype=torch.int64) + 37)
        .to(torch.uint8)
        .reshape(blocks, 1, tokens, rope_width * 2)
    )
    latent.view(torch.uint8).copy_(latent_data)
    rope.view(torch.uint8).copy_(rope_data)
    expected = before.clone()
    for block in range(blocks):
        offset = prefix + block * slot_bytes
        expected[offset : offset + latent_bytes] = latent_data[block].flatten()
        expected[offset + latent_bytes : offset + latent_bytes + rope_bytes] = rope_data[block].flatten()
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["misaligned_slice", "short_storage", "incomplete_page", "overlap", "wrong_dtype"])
def test_mixed_views_reject_invalid_storage(invalid):
    backing = torch.zeros(160, dtype=torch.uint8)
    raw = backing[:128]
    page_stride = 64
    latent_dtype = torch.float8_e4m3fn
    if invalid == "misaligned_slice":
        raw = backing[1:129]
    elif invalid == "short_storage":
        raw = backing[:64]
    elif invalid == "incomplete_page":
        page_stride = 63
    elif invalid == "overlap":
        page_stride = 40
    elif invalid == "wrong_dtype":
        latent_dtype = torch.bfloat16
    with pytest.raises(ValueError):
        reshape_mixed_mla_cache(raw, (4, 1, 3, 4), (4, 1, 3, 2), latent_dtype, torch.bfloat16, page_stride, 2)
