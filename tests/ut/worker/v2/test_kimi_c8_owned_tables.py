# SPDX-License-Identifier: Apache-2.0
"""CPU regression for the real DSpark entry hook and owned table views.

GPU metadata contexts and downstream sampling are mocked; the production
propose method and draft table-view constructor are executed unchanged.
Native gather kernels and graph numerics remain covered by device/model ST.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.core.kimi_c8_draft_cache import KimiC8DraftFullHistorySpec
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as module
from vllm_ascend.worker.v2.spec_decode.dspark.kimi_c8_cp import kimi_c8_draft_table_view


class RequestTables:
    def __init__(self):
        self.cp_size, self.cp_rank, self.cp_interleave = 2, 1, 2048
        self.block_tables = [torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)]
        self.input_block_tables = [torch.full((3, 2), 99, dtype=torch.int32)]
        self.slot_mappings = torch.full((1, 6), 88, dtype=torch.int64)
        self.events = []

    def init_block_table_layout_tensors(self):
        self.layout_owner = self

    def gather_block_tables(self, mapping, num_reqs_padded):
        # Independent CPU oracle for the native gather's unchanged contract.
        self.events.append(("gather", mapping, num_reqs_padded))
        target = self.input_block_tables[0]
        target.zero_()
        target[: len(mapping)].copy_(self.block_tables[0].index_select(0, mapping.long()))
        return (target[:num_reqs_padded],)


def owned_tables():
    target = RequestTables()
    spec = KimiC8DraftFullHistorySpec(block_size=2048, num_kv_heads=1, head_size=64, dtype=torch.float8_e4m3fn)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["draft.layer"], kv_cache_spec=spec)])
    draft = kimi_c8_draft_table_view(target, config, ["draft.layer"])
    return target, draft


def invoke_propose(monkeypatch, owner, *, qualified, dummy=False):
    spec = object.__new__(module.AscendDSparkSpeculator)
    spec._kimi_c8_replicated_dcp = qualified
    spec.replicated_pcp = False
    spec.enable_adaptive_verification = False
    spec.use_dcp = False
    spec.max_num_reqs = 3
    spec.max_num_tokens = 6
    spec.block_tables = owner
    spec.input_buffers = SimpleNamespace(positions=torch.zeros(6, dtype=torch.int64))
    config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: config))
    events = []

    @contextmanager
    def metadata(*args, **kwargs):
        events.append(("metadata", owner.input_block_tables[0].clone()))
        assert kwargs["parallel_config"] is config.parallel_config
        yield

    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", metadata)
    result = torch.tensor([7], dtype=torch.int64)
    sync = object()

    def parent(self, *args, **kwargs):
        events.append(("parent", owner.input_block_tables[0].clone()))
        assert args[11] is sync
        return result

    monkeypatch.setattr(module.DSparkSpeculator, "propose", parent)

    def call(mapping):
        batch = SimpleNamespace(
            idx_mapping=mapping, is_prefilling_np=np.zeros(len(mapping), dtype=bool), num_reqs=len(mapping)
        )
        token = torch.zeros(1)
        actual = spec.propose(
            input_batch=batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=token,
            aux_hidden_states=None,
            num_sampled=token,
            num_rejected=token,
            last_sampled=token,
            next_prefill_tokens=token,
            temperature=token,
            seeds=token,
            dp_sync=sync,
            dummy_run=dummy,
        )
        assert actual is result and spec.input_batch is batch
        return events[-2:]

    return call


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("dummy", [False, True])
def test_owned_gather_is_only_for_live_qualified_drafts(monkeypatch, qualified, dummy):
    _, draft = owned_tables()
    call = invoke_propose(monkeypatch, draft, qualified=qualified, dummy=dummy)
    mapping = torch.tensor([2, 0], dtype=torch.int32)
    events = call(mapping)
    if qualified and not dummy:
        assert len(draft.events) == 1
        assert draft.events[0][1] is mapping and draft.events[0][2] == 3
        expected = torch.tensor([[30, 31], [10, 11], [0, 0]], dtype=torch.int32)
    else:
        assert draft.events == []
        expected = torch.zeros((3, 2), dtype=torch.int32)
    assert [x[0] for x in events] == ["metadata", "parent"]
    for _, snapshot in events:
        torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)


def test_reordered_reused_requests_refresh_before_metadata_and_keep_addresses(monkeypatch):
    target, draft = owned_tables()
    pointer = draft.input_block_tables[0].data_ptr()
    target_before = target.input_block_tables[0].clone()
    call = invoke_propose(monkeypatch, draft, qualified=True)
    call(torch.tensor([2, 0], dtype=torch.int32))
    target.block_tables[0][1] = torch.tensor([71, 73], dtype=torch.int32)
    events = call(torch.tensor([1], dtype=torch.int32))
    expected = torch.tensor([[71, 73], [0, 0], [0, 0]], dtype=torch.int32)
    for _, snapshot in events:
        torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
    assert draft.input_block_tables[0].data_ptr() == pointer
    torch.testing.assert_close(target.input_block_tables[0], target_before, rtol=0, atol=0)


def test_draft_table_copy_has_independent_batch_views_and_shared_request_pages():
    target, draft = owned_tables()
    assert (target.cp_size, target.cp_rank, target.cp_interleave) == (2, 1, 2048)
    assert (draft.cp_size, draft.cp_rank, draft.cp_interleave) == (1, 0, 1)
    assert draft.layout_owner is draft
    assert draft.block_tables is target.block_tables
    assert draft.input_block_tables[0].data_ptr() != target.input_block_tables[0].data_ptr()
    assert draft.slot_mappings.data_ptr() != target.slot_mappings.data_ptr()
    assert torch.count_nonzero(draft.input_block_tables[0]).item() == 0
    assert torch.all(draft.slot_mappings == -1).item()
    assert draft.kimi_c8_owned_groups == (0,)


@pytest.mark.parametrize("invalid", ["target_cp1", "mixed_group", "ordinary_descriptor", "missing_draft"])
def test_draft_table_view_keeps_fail_closed_ownership_guards(invalid):
    target = RequestTables()
    spec = KimiC8DraftFullHistorySpec(block_size=2048, num_kv_heads=1, head_size=64, dtype=torch.float8_e4m3fn)
    group = SimpleNamespace(layer_names=["draft.layer"], kv_cache_spec=spec)
    if invalid == "target_cp1":
        target.cp_size = 1
    elif invalid == "mixed_group":
        group.layer_names.append("target.layer")
    elif invalid == "ordinary_descriptor":
        group.kv_cache_spec = object()
    elif invalid == "missing_draft":
        group.layer_names = ["target.layer"]
    with pytest.raises(ValueError):
        kimi_c8_draft_table_view(target, SimpleNamespace(kv_cache_groups=[group]), ["draft.layer"])
