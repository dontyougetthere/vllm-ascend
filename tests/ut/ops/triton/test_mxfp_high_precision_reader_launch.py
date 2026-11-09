# SPDX-License-Identifier: Apache-2.0
"""CPU launch-contract checks; numeric NPU coverage lives in the PR ST."""

import pytest
import torch

from vllm_ascend.ops.triton import mxfp_high_precision_attention as reader


@pytest.fixture(autouse=True)
def mock_device_math_boundary(monkeypatch):
    # CPU CI uses vLLM's Triton placeholder, which deliberately has no launch
    # utilities. Only supply integer launch math; kernels remain recorders.
    monkeypatch.setattr(reader.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((grid, args))

        return launch


def inputs(capacity):
    query = torch.empty((2, 2, 64), dtype=torch.float8_e4m3fn)
    scales = torch.empty((2, 2, 2), dtype=torch.uint8)
    key = torch.empty((1, 1, 2, 16, 32), dtype=torch.float8_e4m3fn)
    value = torch.empty_like(key)
    kv_scales = torch.empty((1, 1, 1, 16, 2, 1), dtype=torch.uint8)
    table = torch.zeros((1, capacity), dtype=torch.int32)
    starts = torch.tensor([0, 2], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    return query, scales, (key, value, kv_scales, kv_scales.clone()), table, starts, lengths, 0.125


@pytest.mark.parametrize("capacity", [1, 64, 1024])
def test_page_capacity_preserves_single_streaming_reduction(monkeypatch, capacity):
    partial, merge = LaunchRecorder(), LaunchRecorder()
    monkeypatch.setattr(reader, "_partial", partial)
    monkeypatch.setattr(reader, "_merge", merge)
    output, lse = reader.read(*inputs(capacity), output_dtype=torch.float32)
    assert partial.calls[0][0] == (4, 1)
    assert partial.calls[0][1][9].shape == (2, 2, 1, 64)
    assert partial.calls[0][1][10].shape == (2, 2, 1, 2)
    assert merge.calls[0][0] == (4, 1)
    assert merge.calls[0][1][-3:] == (1, 1, 64)
    assert output.shape == (2, 2, 64) and output.dtype == torch.float32
    assert lse.shape == (2, 2)


def test_empty_page_capacity_rejected_before_kernel_launch(monkeypatch):
    partial, merge = LaunchRecorder(), LaunchRecorder()
    monkeypatch.setattr(reader, "_partial", partial)
    monkeypatch.setattr(reader, "_merge", merge)
    with pytest.raises(ValueError, match="nonempty page-table capacity"):
        reader.read(*inputs(0))
    assert partial.calls == merge.calls == []
