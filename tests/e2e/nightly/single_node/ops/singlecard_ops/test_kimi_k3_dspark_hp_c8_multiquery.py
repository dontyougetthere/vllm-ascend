from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.mla_v1 import (
    AscendMLAMetadataBuilder,
    _use_kimi_k3_dspark_hp_c8_multiquery,
)


def config(model="kimi_linear", method="dspark", pp=1, dcp=1, pcp=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model)),
        speculative_config=None if method is None else SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp, decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
        ),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "qwen3"},
        {"model": "deepseek_v2"},
        {"method": None},
        {"method": "mtp"},
        {"pp": 3},
        {"dcp": 2},
        {"pcp": 2},
    ],
)
def test_scope_preserves_other_models_and_topologies(kwargs):
    assert not _use_kimi_k3_dspark_hp_c8_multiquery(config(**kwargs))


def test_scope_selects_only_k3_dspark():
    assert _use_kimi_k3_dspark_hp_c8_multiquery(config())
    assert not _use_kimi_k3_dspark_hp_c8_multiquery(None)


def fixture(query, endpoints, live, tokens, padded=0, cfg=None):
    torch_npu.npu.set_device(0)
    b = object.__new__(AscendMLAMetadataBuilder)
    b.vllm_config = config() if cfg is None else cfg
    b._hp_c8_enabled = True
    b._hp_c8_capacity = 32
    b._hp_c8_actual_reqs = live
    b._hp_c8_lengths = None
    b._hp_c8_pages = None
    b.num_decodes = len(query)
    b.num_decode_tokens = tokens
    b.graph_pad_size = padded
    b.dcp_enabled = False
    b.pcp_enabled = False
    b.block_table = (torch.arange(len(query) * 3, dtype=torch.int32).reshape(-1, 3) + 10).npu()
    q = torch.tensor(query, dtype=torch.int32)
    common = SimpleNamespace(
        query_start_loc_cpu=torch.cat([torch.zeros(1, dtype=torch.int32), q.cumsum(0)]),
        seq_lens=torch.tensor(endpoints, dtype=torch.int32, device="npu:0"),
        # Deliberately disagree with the device endpoints.
        seq_lens_cpu=torch.full((len(query),), 999, dtype=torch.int32),
        dcp_local_seq_lens=None,
    )
    return b, common


@pytest.mark.parametrize(
    "query,endpoints,live,tokens,rows,expected",
    [
        ([1], [66], 1, 1, 1, [66]),
        ([1, 1, 0], [66, 37, 999], 2, 2, 4, [66, 37, 0, 0]),
        ([1, 1, 1, 1], [66, 37, 999, 999], 2, 2, 4, [66, 37, 0, 0]),
    ],
)
def test_ordinary_singlequery_padding_path_is_unchanged(query, endpoints, live, tokens, rows, expected):
    b, common = fixture(query, endpoints, live, tokens)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, rows)
    assert lengths.cpu().tolist() == expected
    assert torch.equal(pages[:live].cpu(), b.block_table[:live].cpu())
    assert not bool(pages[live:].any().item())


@pytest.mark.parametrize(
    "query,endpoints,live,tokens,padded,expected,ids",
    [
        ([8], [73], 1, 8, 0, list(range(66, 74)), [0] * 8),
        ([3, 1, 0], [12, 20, 999], 2, 4, 8, [10, 11, 12, 20, 0, 0, 0, 0], [0, 0, 0, 1]),
        ([8, 8], [73, 999], 1, 8, 16, list(range(66, 74)) + [0] * 8, [0] * 8),
    ],
)
def test_causal_expansion_uses_device_endpoints_and_zeros_padding(
    query, endpoints, live, tokens, padded, expected, ids
):
    b, common = fixture(query, endpoints, live, tokens, padded)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, len(query))
    assert lengths.cpu().tolist() == expected
    assert torch.equal(pages[:tokens].cpu(), b.block_table[ids].cpu())
    assert not bool(pages[tokens:].any().item())


def test_refresh_preserves_addresses_across_single_and_multiquery():
    b, common = fixture([1], [65], 1, 1)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, 1)
    pointers = lengths.data_ptr(), pages.data_ptr()
    b.num_decode_tokens = 8
    common.query_start_loc_cpu = torch.tensor([0, 8], dtype=torch.int32)
    common.seq_lens.fill_(73)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, 1)
    assert lengths.cpu().tolist() == list(range(66, 74))
    assert (lengths.data_ptr(), pages.data_ptr()) == pointers
    b.num_decode_tokens = 1
    common.query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
    common.seq_lens.fill_(80)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, 1)
    assert lengths.cpu().tolist() == [80]
    assert (lengths.data_ptr(), pages.data_ptr()) == pointers


@pytest.mark.parametrize("cfg", [config(model="qwen3"), config(method=None), config(dcp=2), config(pp=3)])
def test_uncovered_multiquery_retains_existing_guard(cfg):
    b, common = fixture([8], [73], 1, 8, cfg=cfg)
    with pytest.raises(NotImplementedError, match="one live query per request"):
        b._prepare_hp_c8_decode_metadata(common, 1)


def test_real_query_token_mismatch_is_rejected():
    b, common = fixture([8], [73], 1, 7)
    with pytest.raises(ValueError, match="real decode token counts"):
        b._prepare_hp_c8_decode_metadata(common, 1)


def test_multiquery_capacity_and_page_width_are_checked():
    b, common = fixture([8], [73], 1, 8, padded=64)
    with pytest.raises(ValueError, match="retained capacity"):
        b._prepare_hp_c8_decode_metadata(common, 1)
    b.graph_pad_size = 0
    b._prepare_hp_c8_decode_metadata(common, 1)
    b.block_table = torch.ones((1, 4), dtype=torch.int32, device="npu:0")
    with pytest.raises(ValueError, match="page-table width"):
        b._prepare_hp_c8_decode_metadata(common, 1)


def test_pp2_multiquery_reuses_causal_metadata_and_fixed_addresses():
    b, common = fixture([1], [65], 1, 1, cfg=config(pp=2))
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, 1)
    pointers = lengths.data_ptr(), pages.data_ptr()
    b.num_decode_tokens = 8
    common.query_start_loc_cpu = torch.tensor([0, 8], dtype=torch.int32)
    common.seq_lens.fill_(73)
    lengths, pages = b._prepare_hp_c8_decode_metadata(common, 1)
    assert lengths.cpu().tolist() == list(range(66, 74))
    assert pages.cpu().tolist() == [b.block_table[0].cpu().tolist()] * 8
    assert (lengths.data_ptr(), pages.data_ptr()) == pointers
