# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.utils import prepare_kernel_block_sizes

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.patch.platform import patch_kv_cache_utils as kv_utils


def _specs(block_size=2048, draft_dtype=torch.float8_e4m3fn):
    page_size = block_size * 640
    target = AscendMLAAttentionSpec(
        block_size=block_size,
        num_heads=1,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.float8_e4m3fn,
        page_size_padded=page_size,
    )
    draft = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=64,
        head_size_v=64,
        dtype=draft_dtype,
        page_size_padded=page_size,
    )
    recurrent = MambaSpec(
        block_size=8192,
        shapes=((10, 4608), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="none",
        num_speculative_blocks=7,
        page_size_padded=page_size,
    )
    return {
        **{f"language_model.model.layers.{i}.self_attn.attn": target for i in range(24)},
        **{f"model.layers.{i}.self_attn.attn": draft for i in range(93, 98)},
        **{f"language_model.model.layers.{i}.self_attn": recurrent for i in range(24, 93)},
    }


@pytest.mark.parametrize("block_size", [2048, 4096])
def test_kimi_k3_c8_draft_selects_distinct_backend_kernel_blocks(block_size):
    specs = _specs(block_size)
    groups = kv_utils._get_kimi_k3_dspark_mixed_kv_cache_groups(specs)
    assert [len(group.layer_names) for group in groups] == [24, 5, 23, 23, 23]
    names = [name for group in groups for name in group.layer_names]
    assert len(names) == len(set(names)) == 98 and set(names) == set(specs)
    handles = [
        [SimpleNamespace(backend=AscendMLABackend)],
        [SimpleNamespace(backend=AscendC8MXFPAttentionBackend)],
        [],
        [],
        [],
    ]
    assert prepare_kernel_block_sizes(SimpleNamespace(kv_cache_groups=groups), handles) == [128, 512, 8192, 8192, 8192]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_other_draft_dtypes_retain_mixed_grouping(dtype):
    groups = kv_utils._get_kimi_k3_dspark_mixed_kv_cache_groups(_specs(draft_dtype=dtype))
    assert [len(group.layer_names) for group in groups] == [29, 23, 23, 23]


def test_unequal_pages_fall_back_to_existing_grouping():
    specs = _specs()
    name = next(name for name in specs if name.startswith("model.layers."))
    specs[name] = replace(specs[name], page_size_padded=specs[name].page_size_bytes + 256)
    assert kv_utils._get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def test_non_k3_names_keep_generic_hook():
    specs = {"other." + name: spec for name, spec in _specs().items()}
    expected = object()
    with patch.object(kv_utils, "_orig_get_kv_cache_groups_uniform_page_size", return_value=expected) as generic:
        assert kv_utils._get_kv_cache_groups_uniform_page_size(specs) is expected
        generic.assert_called_once_with(specs)


def test_target_only_signature_is_unchanged():
    specs = {name: spec for name, spec in _specs().items() if not name.startswith("model.layers.")}
    assert kv_utils._get_kimi_k3_dspark_mixed_kv_cache_groups(specs) is None


def _pp_descriptor_fixture():
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

    specs = _specs(block_size=4096)
    target_name = next(name for name in specs if name.startswith("language_model."))
    draft_names = [name for name in specs if name.startswith("model.layers.")]
    draft_specs = {name: specs[name] for name in draft_names}
    groups = [
        KVCacheGroupSpec(layer_names=[target_name], kv_cache_spec=specs[target_name]),
        KVCacheGroupSpec(layer_names=[], kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(draft_specs)),
    ]
    tensors = [
        KVCacheTensor(
            size=32 * 1024 * 1024, layers=[target_name], offset=256, layer_stride=5242880, block_stride=2621440
        ),
        KVCacheTensor(size=32 * 1024 * 1024, layers=draft_names, offset=0, layer_stride=5242880, block_stride=2621440),
    ]
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="kimi_linear")),
        speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype="mxfp8"),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
    )
    return config, KVCacheConfig(num_blocks=2, kv_cache_tensors=tensors, kv_cache_groups=groups)


def test_pp0_nonlocal_draft_descriptors_preserve_native_layout():
    config, original = _pp_descriptor_fixture()
    fixed = kv_utils._remove_kimi_k3_dspark_nonlocal_draft_tensors(config, original)
    assert fixed is not original
    assert fixed.num_blocks == original.num_blocks
    assert fixed.kv_cache_groups is original.kv_cache_groups
    assert fixed.kv_cache_tensors == [original.kv_cache_tensors[0]]
    assert fixed.kv_cache_tensors[0] is original.kv_cache_tensors[0]
    assert len(original.kv_cache_tensors) == 2
    assert fixed.kv_cache_groups[1].layer_names == []
    assert len(fixed.kv_cache_groups[1].kv_cache_spec.kv_cache_specs) == 5


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("model", "model_type", "qwen2"),
        ("speculative_config", "method", "mtp"),
        ("speculative_config", "kv_cache_dtype", "auto"),
        ("cache_config", "cache_dtype", "auto"),
        ("parallel_config", "pipeline_parallel_size", 1),
        ("parallel_config", "pipeline_parallel_size", 3),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
    ],
)
def test_pp_descriptor_other_configurations_preserve_object(section, field, value):
    config, original = _pp_descriptor_fixture()
    target = config.model_config.hf_text_config if section == "model" else getattr(config, section)
    setattr(target, field, value)
    assert kv_utils._remove_kimi_k3_dspark_nonlocal_draft_tensors(config, original) is original


def test_pp1_local_draft_descriptors_preserve_object():
    config, original = _pp_descriptor_fixture()
    original.kv_cache_groups[1].layer_names = list(original.kv_cache_groups[1].kv_cache_spec.kv_cache_specs)
    assert kv_utils._remove_kimi_k3_dspark_nonlocal_draft_tensors(config, original) is original


@pytest.mark.parametrize("case", ["mixed", "unknown", "target_empty", "bf16"])
def test_pp_descriptor_unrecognized_ownership_preserves_native_check(case):
    config, original = _pp_descriptor_fixture()
    if case == "mixed":
        original.kv_cache_tensors[1].layers.append(original.kv_cache_groups[0].layer_names[0])
    elif case == "unknown":
        original.kv_cache_tensors[1].layers.append("unknown.layer")
    elif case == "target_empty":
        original.kv_cache_groups[0].layer_names = []
    else:
        group = original.kv_cache_groups[1]
        group.kv_cache_spec = replace(
            group.kv_cache_spec,
            kv_cache_specs={
                name: replace(spec, dtype=torch.bfloat16) for name, spec in group.kv_cache_spec.kv_cache_specs.items()
            },
        )
    assert kv_utils._remove_kimi_k3_dspark_nonlocal_draft_tensors(config, original) is original
