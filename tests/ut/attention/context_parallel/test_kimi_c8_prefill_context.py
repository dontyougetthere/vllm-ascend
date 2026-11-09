# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace as S

import torch

from vllm_ascend.attention.context_parallel.kimi_c8_prefill import needs_kimi_c8_empty_context_neutralization as enabled
from vllm_ascend.attention.context_parallel.kimi_c8_prefill import neutralize_kimi_c8_empty_context as neutralize


def fixture():
    return S(
        vllm_config=S(model_config=S(hf_text_config=S(model_type="kimi_linear")), cache_config=S(cache_dtype="fp8")),
        use_mixed_kv_cache_store=True,
        fa_quant_layer=True,
        support_fp8_attention=True,
        dcp_enabled=True,
        dcp_size=2,
        pcp_enabled=False,
        is_draft_model=False,
    )


def test_exact_scope():
    base = fixture()
    assert enabled(base)
    for name, value in [
        ("use_mixed_kv_cache_store", False),
        ("fa_quant_layer", False),
        ("support_fp8_attention", False),
        ("dcp_enabled", False),
        ("dcp_size", 1),
        ("dcp_size", 4),
        ("pcp_enabled", True),
        ("is_draft_model", True),
    ]:
        obj = copy.deepcopy(base)
        setattr(obj, name, value)
        assert not enabled(obj)
    obj = fixture()
    obj.vllm_config.model_config.hf_text_config.model_type = "other"
    assert not enabled(obj)
    obj = fixture()
    obj.vllm_config.cache_config.cache_dtype = "bfloat16"
    assert not enabled(obj)


def test_empty_global_rows_and_padding():
    for q, kv, total in [([2, 5], [7, 7], 5), ([2, 5], [0, 7], 5), ([2, 2, 5], [0, 0, 3], 8), ([2, 5], [0, 0], 5)]:
        out = torch.arange(total * 2 * 3, dtype=torch.float32).view(total, 2, 3)
        lse = torch.arange(total * 2, dtype=torch.float32).view(2, total).transpose(0, 1).unsqueeze(-1)
        old = out.clone()
        old_lse = lse.clone()
        o, returned_lse = neutralize(out, lse, q, kv)
        assert o is out and returned_lse is lse
        qs = ks = 0
        for qe, ke in zip(q, kv):
            if ke == ks:
                assert torch.equal(out[qs:qe], torch.zeros_like(out[qs:qe])) and torch.isneginf(lse[qs:qe]).all()
            else:
                assert torch.equal(out[qs:qe], old[qs:qe]) and torch.equal(lse[qs:qe], old_lse[qs:qe])
            qs, ks = qe, ke
        assert torch.equal(out[qs:], torch.zeros_like(out[qs:])) and torch.isneginf(lse[qs:]).all()


def test_nonempty_nan_not_hidden():
    out = torch.full((4, 2, 3), float("nan"))
    lse = torch.ones((4, 2, 1))
    neutralize(out, lse, [2, 4], [1, 1])
    assert torch.isnan(out[:2]).all() and torch.equal(lse[:2], torch.ones_like(lse[:2]))
    assert (out[2:] == 0).all() and torch.isneginf(lse[2:]).all()


def test_invalid_metadata_no_partial_mutation():
    for q, kv in [([2, 5], [1]), ([3, 2], [1, 2]), ([2, 5], [3, 2]), ([2, 6], [1, 1]), ([True, 5], [1, 1]), ([], [])]:
        out = torch.ones((5, 2, 3))
        lse = torch.ones((5, 2, 1))
        try:
            neutralize(out, lse, q, kv)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError((q, kv))
        assert (out == 1).all() and (lse == 1).all()
