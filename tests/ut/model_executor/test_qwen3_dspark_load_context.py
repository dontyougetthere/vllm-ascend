# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for draft weights loaded under a target config."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config

from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


def _config(dtype, *, kimi=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=dtype),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="kimi_linear" if kimi else "qwen3")),
        speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype="mxfp8"),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
        ),
    )


def _model(draft_config):
    cls = AscendQwen3DSparkForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model._draft_vllm_config = draft_config
    model.config = SimpleNamespace(vocab_size=16, draft_vocab_size=16)
    model.target_vocab_size = 16
    model.model = nn.Module()
    model.model.confidence_head = None
    model.model._build_fused_kv_buffers = lambda: None
    mxfp = draft_config.cache_config.cache_dtype == "mxfp8"
    model.model.quant_config = (
        AscendModelSlimConfig({"kv_cache_type": "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"}) if mxfp else None
    )
    model.model.layers = nn.ModuleList()
    for _ in range(5):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.attn = nn.Module()
        if mxfp:
            layer.self_attn.attn.v_cache_scale = nn.Parameter(torch.zeros(2, 4, dtype=torch.uint8), requires_grad=False)
        else:
            layer.self_attn.attn.weight = nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16), requires_grad=False)
        model.model.layers.append(layer)
    return model


def test_constructor_retains_own_draft_config(monkeypatch):
    parent = AscendQwen3DSparkForCausalLM.__bases__[0]

    def initialize(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        self.config = SimpleNamespace()

    monkeypatch.setattr(parent, "__init__", initialize)
    draft = _config("mxfp8")
    model = AscendQwen3DSparkForCausalLM(vllm_config=draft)
    assert model._draft_vllm_config is draft


def test_unscoped_parent_reproduces_scale_name_failure():
    model = _model(_config("mxfp8"))
    parent = AscendQwen3DSparkForCausalLM.__bases__[0]
    target = _config("fp8")
    with set_current_vllm_config(target):
        with pytest.raises(ValueError, match="fa_v"):
            parent.load_weights(model, [("layers.0.self_attn.fa_v.scale", torch.full((2, 4), 127, dtype=torch.uint8))])
        assert get_current_vllm_config_or_none() is target


@pytest.mark.parametrize("draft_dtype", ["mxfp8"])
@pytest.mark.parametrize("outer_dtype", ["fp8", "auto", "mxfp8", None])
@pytest.mark.parametrize("fail", [False, True])
def test_loads_own_cache_weights_and_restores_context(draft_dtype, outer_dtype, fail):
    draft = _config(draft_dtype)
    model = _model(draft)
    outer = _config(outer_dtype) if outer_dtype else None
    mxfp = draft_dtype == "mxfp8"
    suffix = "fa_v.scale" if mxfp else "attn.weight"
    dtype = torch.uint8 if mxfp else torch.bfloat16
    values = [torch.full((2, 4), 123 + i if mxfp else i + 1, dtype=dtype) for i in range(5)]

    def weights():
        # Even lazy checkpoint iteration must run under the draft context.
        assert get_current_vllm_config_or_none() is draft
        for i, value in enumerate(values):
            yield f"layers.{i}.self_attn.{suffix}", value
        if fail:
            raise RuntimeError("checkpoint iterator failed")

    with set_current_vllm_config(outer):
        if fail:
            with pytest.raises(RuntimeError, match="checkpoint iterator failed"):
                model.load_weights(weights())
        else:
            model.load_weights(weights())
            for layer, value in zip(model.model.layers, values):
                param = layer.self_attn.attn.v_cache_scale if mxfp else layer.self_attn.attn.weight
                torch.testing.assert_close(param, value, rtol=0, atol=0)
        assert get_current_vllm_config_or_none() is outer


@pytest.mark.parametrize("draft_dtype,kimi", [("auto", True), ("auto", False), ("mxfp8", False)])
@pytest.mark.parametrize("outer_dtype", ["fp8", "auto", None])
@pytest.mark.parametrize("fail", [False, True])
def test_other_drafts_preserve_parent_context(monkeypatch, draft_dtype, kimi, outer_dtype, fail):
    draft = _config(draft_dtype, kimi=kimi)
    model = _model(draft)
    outer = _config(outer_dtype) if outer_dtype else None
    parent = AscendQwen3DSparkForCausalLM.__bases__[0]
    marker = object()

    def weights():
        assert get_current_vllm_config_or_none() is outer
        yield "unchanged.weight", marker

    checkpoint = weights()

    def load_weights(self, supplied):
        assert self is model and supplied is checkpoint
        assert get_current_vllm_config_or_none() is outer
        assert list(supplied) == [("unchanged.weight", marker)]
        if fail:
            raise RuntimeError("parent failure")
        return marker

    monkeypatch.setattr(parent, "load_weights", load_weights)
    with set_current_vllm_config(outer):
        if fail:
            with pytest.raises(RuntimeError, match="parent failure"):
                model.load_weights(checkpoint)
        else:
            assert model.load_weights(checkpoint) is marker
        assert get_current_vllm_config_or_none() is outer
