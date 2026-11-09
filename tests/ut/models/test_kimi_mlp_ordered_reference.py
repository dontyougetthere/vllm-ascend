"""CPU contracts for the actual Kimi dense-MLP-only ordered BF16 method."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3


class FakeDist:
    def __init__(self):
        self.calls = 0

    def all_gather_into_tensor(self, gathered, partial, group):
        self.calls += 1
        assert partial.is_contiguous()
        size = len(group.partials)
        padded = []
        for x in group.partials:
            pad = (-x.shape[0]) % size
            padded.append(torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x)
        assert torch.equal(partial, padded[group.rank])
        gathered.copy_(torch.cat(padded, dim=0))


def setup(monkeypatch, partials, rank):
    group = SimpleNamespace(
        world_size=len(partials), rank_in_group=rank,
        device_group=SimpleNamespace(partials=partials, rank=rank),
    )
    fake = FakeDist()
    monkeypatch.setattr(kimi_k3, "get_tp_group", lambda: group)
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: len(partials))
    monkeypatch.setattr(kimi_k3, "dist", fake)
    mlp = kimi_k3.AscendKimiMLP.__new__(kimi_k3.AscendKimiMLP)
    nn.Module.__init__(mlp)
    return mlp, fake


def reference(partials, rank):
    size = len(partials)
    padded = [torch.nn.functional.pad(x, (0, 0, 0, (-x.shape[0]) % size)) for x in partials]
    first = rank * padded[0].shape[0] // size
    chunk = padded[0].shape[0] // size
    output = padded[0][first:first + chunk].clone()
    for item in padded[1:]:
        output.add_(item[first:first + chunk])
    return output


@pytest.mark.parametrize("rows", [2, 4, 16])
def test_actual_method_all_ranks_preserves_shape_dtype_input_and_owner(monkeypatch, rows):
    partials = [torch.arange(rows * 7, dtype=torch.float32).reshape(rows, 7).to(torch.bfloat16) * (rank + 1) for rank in range(4)]
    original = [item.clone() for item in partials]
    for rank in range(4):
        mlp, fake = setup(monkeypatch, partials, rank)
        output = mlp._ordered_sp_reduce_scatter(partials[rank])
        assert fake.calls == 1
        assert output.shape == ((rows + 3) // 4, 7) and output.dtype == torch.bfloat16
        assert torch.equal(output, reference(partials, rank))
        output.zero_()
        assert all(torch.equal(x, y) for x, y in zip(partials, original))


def test_identical_logical_token_moves_from_tp2_to_tp1(monkeypatch):
    vectors = [torch.tensor([v, -v, 0.5], dtype=torch.bfloat16) for v in (0.25, 0.125, -0.5, 0.0625)]
    outputs = []
    for rows, token_row in ((16, 8), (4, 1)):
        partials = [torch.ones((rows, 3), dtype=torch.bfloat16) * (rank + 1) for rank in range(4)]
        for rank, vector in enumerate(vectors):
            partials[rank][token_row] = vector
        owner = token_row // (rows // 4)
        mlp, _ = setup(monkeypatch, partials, owner)
        output = mlp._ordered_sp_reduce_scatter(partials[owner])
        outputs.append(output[token_row % output.shape[0]])
    assert torch.equal(outputs[0], outputs[1])


def test_noncontiguous_input_and_existing_nonzero_padding_not_replaced(monkeypatch):
    partials = [torch.arange(20, dtype=torch.float32).reshape(4, 5).t().to(torch.bfloat16) + rank for rank in range(4)]
    assert not partials[0].is_contiguous() and all(torch.count_nonzero(x[2:]) for x in partials)
    for rank in range(4):
        mlp, _ = setup(monkeypatch, partials, rank)
        assert torch.equal(mlp._ordered_sp_reduce_scatter(partials[rank]), reference(partials, rank))


def test_forward_uses_real_ordered_branch_only_for_sp_bf16_multi_tp(monkeypatch):
    partials = [torch.full((4, 3), rank + 1, dtype=torch.bfloat16) for rank in range(4)]
    monkeypatch.setattr(kimi_k3.KimiMLP, "forward", lambda self, x: partials[0])
    monkeypatch.setattr(kimi_k3, "sp_all_gather", lambda x: x)
    monkeypatch.setattr(kimi_k3, "sp_reduce_scatter", lambda x: (_ for _ in ()).throw(AssertionError("unexpected native SP")))
    mlp, fake = setup(monkeypatch, partials, 0)
    mlp.use_sequence_parallel = True
    output = mlp(torch.ones((1, 3), dtype=torch.bfloat16))
    assert torch.equal(output, reference(partials, 0)) and fake.calls == 1


def test_forward_preserves_non_sp_float32_and_tp1_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(kimi_k3.KimiMLP, "forward", lambda self, x: x.clone())
    monkeypatch.setattr(kimi_k3, "sp_all_gather", lambda x: calls.append("gather") or x)
    monkeypatch.setattr(kimi_k3, "sp_reduce_scatter", lambda x: calls.append("native") or x.clone())
    x = torch.ones((2, 3), dtype=torch.float32)
    mlp, fake = setup(monkeypatch, [x] * 4, 0)
    mlp.use_sequence_parallel = False
    assert torch.equal(mlp(x), x) and calls == []
    mlp.use_sequence_parallel = True
    assert torch.equal(mlp(x), x) and calls == ["gather", "native"] and fake.calls == 0
    y = torch.ones((1, 3), dtype=torch.bfloat16)
    mlp, fake = setup(monkeypatch, [y], 0)
    mlp.use_sequence_parallel = True
    assert torch.equal(mlp(y), y) and calls[-2:] == ["gather", "native"] and fake.calls == 0


def test_wrong_direct_dtype_or_shape_rejected(monkeypatch):
    partials = [torch.ones((4, 3), dtype=torch.bfloat16) for _ in range(4)]
    mlp, _ = setup(monkeypatch, partials, 0)
    with pytest.raises(AssertionError):
        mlp._ordered_sp_reduce_scatter(partials[0].float())
    with pytest.raises(AssertionError):
        mlp._ordered_sp_reduce_scatter(partials[0].flatten())
