# SPDX-License-Identifier: Apache-2.0
"""CPU-only dispatch contract; never substitute mocked numerics for NPU ST."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl
from vllm_ascend.ops.triton import mxfp_high_precision_attention as hp


def configured_impl():
    impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
    impl.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype="mxfp8"),
        attention_config=SimpleNamespace(use_non_causal=True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="kimi_linear")),
        cache_config=SimpleNamespace(cache_dtype="mxfp8"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    impl.sliding_window = None
    impl.head_size = 64
    impl.scale = 0.125
    impl.alibi_slopes = None
    impl.sinks = None
    return impl


def run_dispatch(monkeypatch, impl, *, causal=False, dtype=torch.bfloat16):
    metadata = SimpleNamespace(
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_gpu=torch.ones(1, dtype=torch.int32),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        causal=causal,
        max_query_len=1,
    )
    query = torch.zeros((1, 2, 64), dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.full((1, 2, 1, 2), 127, dtype=torch.uint8)
    output = torch.zeros((1, 2, 64), dtype=dtype)
    calls = []

    def high_precision(*args, **kwargs):
        calls.append("high_precision")
        return torch.ones_like(output), None

    def native(*args, **kwargs):
        calls.append("native")
        return kwargs["output"]

    monkeypatch.setattr(hp, "read", high_precision)
    monkeypatch.setattr(impl, "_run_qfa", native)
    monkeypatch.setattr(impl, "_qfa_step_lengths", lambda meta, count: (meta.query_start_loc_gpu, meta.seq_lens_gpu))
    monkeypatch.setattr(impl, "_get_qfa_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(impl, "_qfa_query_scale_for_layout", lambda scales, count: (scales, "TND"))
    return lambda: impl._forward_mxfp8_attention(query, scales, (), metadata, output), calls, output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_owned_cp1_mxfp8_draft_selects_high_precision(monkeypatch, dtype):
    call, calls, output = run_dispatch(monkeypatch, configured_impl(), dtype=dtype)
    assert call() is output
    assert calls == ["high_precision"]
    assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_config.method", "eagle3"),
        ("speculative_config.kv_cache_dtype", "bfloat16"),
        ("model_config.hf_text_config.model_type", "qwen3"),
        ("cache_config.cache_dtype", "fp8"),
        ("attention_config.use_non_causal", False),
        ("parallel_config.decode_context_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("parallel_config.pipeline_parallel_size", 2),
    ],
)
def test_unqualified_noncausal_draft_is_rejected_without_fallback(monkeypatch, field, value):
    impl = configured_impl()
    owner = impl.vllm_config
    *path, key = field.split(".")
    for part in path:
        owner = getattr(owner, part)
    setattr(owner, key, value)
    call, calls, _ = run_dispatch(monkeypatch, impl)
    with pytest.raises(NotImplementedError, match="non-causal"):
        call()
    assert calls == []


@pytest.mark.parametrize(
    "field,value", [("head_size", 128), ("alibi_slopes", torch.zeros(1)), ("sinks", torch.zeros(1))]
)
def test_unsupported_noncausal_geometry_is_rejected(monkeypatch, field, value):
    impl = configured_impl()
    setattr(impl, field, value)
    call, calls, _ = run_dispatch(monkeypatch, impl)
    with pytest.raises(NotImplementedError, match="non-causal"):
        call()
    assert calls == []


def test_unsupported_noncausal_output_dtype_is_rejected(monkeypatch):
    call, calls, _ = run_dispatch(monkeypatch, configured_impl(), dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="non-causal"):
        call()
    assert calls == []


@pytest.mark.parametrize("model_type,method", [("kimi_linear", "dspark"), ("qwen3", "eagle3")])
def test_causal_path_keeps_native_dispatch(monkeypatch, model_type, method):
    impl = configured_impl()
    impl.vllm_config.model_config.hf_text_config.model_type = model_type
    impl.vllm_config.speculative_config.method = method
    call, calls, output = run_dispatch(monkeypatch, impl, causal=True)
    assert call() is output
    assert calls == ["native"]
