# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 draft C8 must retain its scales in a target-fp8 shared worker."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize(
    "model_type,method,layer_name,backend,dtype,cache_dtype,view_count",
    [
        (
            "kimi_linear",
            "dspark",
            "model.layers.93.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            4,
        ),
        (
            "kimi_linear",
            "dspark",
            "model.layers.97.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            4,
        ),
        (
            "kimi_linear",
            "dspark",
            "model.layers.92.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        (
            "kimi_linear",
            "dspark",
            "model.layers.98.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        (
            "kimi_linear",
            "dspark",
            "layers.93.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        (
            "deepseek_v3",
            "dspark",
            "model.layers.93.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        (
            "kimi_linear",
            "eagle",
            "model.layers.93.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        (
            "kimi_linear",
            None,
            "model.layers.93.self_attn.attn",
            AscendC8MXFPAttentionBackend,
            torch.float8_e4m3fn,
            "fp8",
            2,
        ),
        ("kimi_linear", "dspark", "model.layers.93.self_attn.attn", AscendAttentionBackend, torch.bfloat16, "fp8", 2),
        ("qwen2", None, "model.layers.0.self_attn.attn", AscendC8MXFPAttentionBackend, torch.float8_e4m3fn, "mxfp8", 4),
    ],
)
def test_actual_producer_and_reshape_keep_layer_storage_scope(
    monkeypatch, model_type, method, layer_name, backend, dtype, cache_dtype, view_count
):
    draft_model = SimpleNamespace(get_total_num_hidden_layers=lambda: 5)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type), get_total_num_hidden_layers=lambda: 93
        ),
        speculative_config=SimpleNamespace(method=method, draft_model_config=draft_model) if method else None,
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(indexer_kv_dtype="auto"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    spec = FullAttentionSpec(
        block_size=2048, num_kv_heads=2, head_size=64, head_size_v=64, dtype=dtype, page_size_padded=1310720
    )
    layer = SimpleNamespace(
        get_kv_cache_spec=lambda config: spec, attn_backend=backend, kv_sharing_target_layer_name=None
    )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *args: {layer_name: layer})
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: cfg)
    monkeypatch.setattr(attn_utils, "enable_sfa_dcp_replicated_indexer", lambda config: False)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda config: False)
    monkeypatch.setattr(attn_utils, "_is_dsv4_model", lambda config: False)
    monkeypatch.setattr(attn_utils, "kv_cache_dtype_str_to_dtype", lambda *args: dtype)
    produced = attn_utils.get_kv_cache_spec(cfg)[layer_name]
    assert produced.state_content_bytes == (132 if view_count == 4 else spec.state_content_bytes)
    cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=produced)],
    )
    group = AttentionGroup(backend, [layer_name], produced, 0)
    raw = torch.zeros(2 * produced.page_size_bytes, dtype=torch.int8)
    views = attn_utils._reshape_kv_cache_v2([group], {layer_name: raw}, cache_dtype, [512], {}, cache_config)[
        layer_name
    ]
    assert len(views) == view_count
    assert all(view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr() for view in views)
    assert all(view.shape[0] == 8 for view in views)
    payload_elements = 2 * spec.block_size * spec.num_kv_heads * spec.head_size
    assert views[0].numel() == views[1].numel() == payload_elements
    if view_count == 4:
        assert views[2].numel() == views[3].numel() == payload_elements // 32
        assert views[2].dtype == views[3].dtype == torch.uint8
    assert cfg.cache_config.cache_dtype == cache_dtype
