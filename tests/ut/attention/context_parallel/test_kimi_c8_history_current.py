# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.kimi_c8_history_current import (
    query_layout,
    use_kimi_c8_dcp_current,
)


def guarded_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="kimi_linear")),
        speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype="mxfp8", num_speculative_tokens=7),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2, prefill_context_parallel_size=1, pipeline_parallel_size=1
        ),
    )


def test_exact_feature_guard():
    config = guarded_config()
    assert use_kimi_c8_dcp_current(config)
    pp2 = deepcopy(config)
    pp2.parallel_config.pipeline_parallel_size = 2
    assert use_kimi_c8_dcp_current(pp2)
    for obj, field, value in [
        ("model", "model_type", "deepseek_v3"),
        ("spec", "method", "eagle"),
        ("spec", "kv_cache_dtype", "bfloat16"),
        ("cache", "cache_dtype", "bfloat16"),
        ("parallel", "decode_context_parallel_size", 1),
        ("parallel", "decode_context_parallel_size", 4),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 4),
    ]:
        candidate = deepcopy(config)
        target = {
            "model": candidate.model_config.hf_text_config,
            "spec": candidate.speculative_config,
            "cache": candidate.cache_config,
            "parallel": candidate.parallel_config,
        }[obj]
        setattr(target, field, value)
        assert not use_kimi_c8_dcp_current(candidate), (obj, field, value)
    assert not use_kimi_c8_dcp_current(None)


def test_query_layout_rejects_missing_or_overwide_queries():
    for args in [
        ([7, 2], 2, 8, 8),
        ([7, 0], 2, 7, 8),
        ([9], 1, 9, 8),
        ([1], 2, 1, 8),
        ([True], 1, 1, 8),
        ([1], -1, 0, 8),
    ]:
        with pytest.raises(ValueError):
            query_layout(*args)


def test_query_layout_keeps_unequal_requests_separate_and_ignores_padding():
    live, req, ends, starts = query_layout([7, 2, 0], 2, 9, 8)
    assert live == [7, 2] and req == [0] * 7 + [1] * 2
    assert ends == list(range(1, 8)) + [1, 2]
    assert starts == [0] * 7 + [7] * 2
    assert query_layout([0], 0, 0, 8) == ([], [], [], [])


def test_causal_local_lengths_match_independent_position_owners():
    import torch

    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import causal_local_lengths

    for interleave in (1, 4, 128, 2048):
        histories = [0, interleave - 1, 2 * interleave - 3 if interleave > 1 else 2, 2 * interleave + 1]
        queries = [2, 8, 7, 1]
        requests = torch.tensor([i for i, n in enumerate(queries) for _ in range(n)], dtype=torch.int64)
        counts = torch.tensor([j for n in queries for j in range(1, n + 1)], dtype=torch.int32)
        for rank in (0, 1):
            actual = causal_local_lengths(
                torch.tensor(histories, dtype=torch.int32),
                requests,
                counts,
                dcp_size=2,
                dcp_rank=rank,
                interleave=interleave,
            )
            expected = [
                sum((p // interleave) % 2 == rank for p in range(histories[i] + j))
                for i, n in enumerate(queries)
                for j in range(1, n + 1)
            ]
            assert actual.dtype == torch.int32 and actual.tolist() == expected


def test_causal_endpoints_do_not_include_later_verification_queries():
    import torch

    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import causal_local_lengths

    history = torch.tensor([4093, 4161], dtype=torch.int32)
    req = torch.tensor([0] * 8 + [1] * 2, dtype=torch.int64)
    counts = torch.tensor(list(range(1, 9)) + [1, 2], dtype=torch.int32)
    lengths = [causal_local_lengths(history, req, counts, dcp_size=2, dcp_rank=r, interleave=2048) for r in (0, 1)]
    assert (lengths[0] + lengths[1]).tolist() == list(range(4094, 4102)) + [4162, 4163]
    assert lengths[0][:8].tolist() == [2048, 2048, 2048, 2049, 2050, 2051, 2052, 2053]
    assert lengths[1][:8].tolist() == [2046, 2047, 2048, 2048, 2048, 2048, 2048, 2048]


def test_empty_causal_layout_is_empty():
    import torch

    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import causal_local_lengths

    actual = causal_local_lengths(
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int32),
        dcp_size=2,
        dcp_rank=0,
        interleave=2048,
    )
    assert actual.numel() == 0 and actual.dtype == torch.int32


@pytest.mark.parametrize("latent,rope,ranks", [(512, 64, 2), (256, 32, 1), (128, 16, 4)])
def test_current_workspace_derives_storage_from_cache_layout(latent, rope, ranks):
    import torch

    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import KimiC8CurrentWorkspace

    workspace = KimiC8CurrentWorkspace(3, "cpu", kv_lora_rank=latent, qk_rope_head_dim=rope, dcp_size=ranks)
    expected_bytes = latent * torch.float8_e4m3fn.itemsize + rope * torch.bfloat16.itemsize
    assert workspace.packet_bytes == expected_bytes
    assert workspace.local_packet.shape == (3, expected_bytes)
    assert workspace.gathered_packets.shape == (ranks * 3, expected_bytes)
    assert workspace.gathered_owners.shape == (ranks * 3,)
    assert workspace.packet.data_ptr() != workspace.local_packet.data_ptr()


@pytest.mark.parametrize("field", ["capacity", "kv_lora_rank", "qk_rope_head_dim", "dcp_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_current_workspace_rejects_invalid_dimensions(field, value):
    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import KimiC8CurrentWorkspace

    dimensions = dict(capacity=3, kv_lora_rank=512, qk_rope_head_dim=64, dcp_size=2)
    dimensions[field] = value
    with pytest.raises(ValueError, match="positive integers"):
        KimiC8CurrentWorkspace(device="cpu", **dimensions)
