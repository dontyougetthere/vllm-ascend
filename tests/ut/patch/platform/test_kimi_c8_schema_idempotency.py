# SPDX-License-Identifier: Apache-2.0
"""Preserve dtype choices without rebuilding already compatible schemas."""

import typing

import pydantic
import pytest
from vllm.config import cache, speculative

from vllm_ascend.patch.platform import patch_kv_cache_dtype as patch


def test_repeated_setup_preserves_validator_identity(monkeypatch):
    validators = (cache.CacheConfig.__pydantic_validator__, speculative.SpeculativeConfig.__pydantic_validator__)

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("compatible schema must not be rebuilt")

    monkeypatch.setattr(patch._pdc, "rebuild_dataclass", unexpected_rebuild)
    for _ in range(3):
        patch._apply_kv_cache_dtype_int8_patch()
        patch._apply_draft_kv_cache_dtype_patch()
    assert cache.CacheConfig.__pydantic_validator__ is validators[0]
    assert speculative.SpeculativeConfig.__pydantic_validator__ is validators[1]


def test_preserves_upstream_members_and_draft_inheritance():
    original = typing.get_args(patch._ORIG_CACHE_DTYPE)
    assert set(original) <= set(typing.get_args(cache.CacheDType))
    adapter = pydantic.TypeAdapter(speculative.SpeculativeConfig.__dataclass_fields__["kv_cache_dtype"].type)
    for value in (*original, "int8", "mxfp8", None):
        assert adapter.validate_python(value) == value
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python("invalid_kimi_cache_dtype")


@pytest.mark.parametrize("value", ["auto", "bfloat16", "fp8", "mxfp8", "int8"])
def test_public_cache_config_retains_supported_values(value):
    assert cache.CacheConfig(cache_dtype=value).cache_dtype == value
