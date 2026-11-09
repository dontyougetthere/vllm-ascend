# SPDX-License-Identifier: Apache-2.0
"""Independent CP1 execution and owned table views for the K3 C8 drafter."""

from copy import copy

from vllm.config import replace


def use_kimi_c8_replicated_draft(config):
    spec = getattr(config, "speculative_config", None)
    par = getattr(config, "parallel_config", None)
    model = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    return (
        getattr(model, "model_type", None) == "kimi_linear"
        and getattr(spec, "method", None) == "dspark"
        and getattr(spec, "kv_cache_dtype", None) == "mxfp8"
        and getattr(getattr(config, "cache_config", None), "cache_dtype", None) == "fp8"
        and getattr(par, "decode_context_parallel_size", 1) == 2
        and getattr(par, "prefill_context_parallel_size", 1) == 1
        and getattr(par, "pipeline_parallel_size", 1) in (1, 2)
    )


def validated_kimi_c8_draft_parallel(config):
    spec = config.speculative_config
    draft = spec.draft_model_config
    if draft is None or draft.use_mla:
        raise ValueError("K3 replicated C8 drafter must own a GQA model")
    par = replace(
        spec.draft_parallel_config,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    draft.verify_with_parallel_config(par)
    return par


def kimi_c8_draft_execution_config(config):
    if not use_kimi_c8_replicated_draft(config):
        return config
    result = copy(config)
    result.parallel_config = replace(
        config.parallel_config,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    result.speculative_config = copy(config.speculative_config)
    result.speculative_config.draft_parallel_config = validated_kimi_c8_draft_parallel(config)
    return result


def kimi_c8_draft_table_view(table, kv_cache_config, draft_names):
    import torch

    from vllm_ascend.core.kimi_c8_draft_cache import is_kimi_c8_draft_full_history

    if table.cp_size != 2:
        raise ValueError("K3 replicated draft view requires target DCP2")
    owned = []
    for i, group in enumerate(kv_cache_config.kv_cache_groups):
        selected = set(group.layer_names) & set(draft_names)
        if selected:
            if selected != set(group.layer_names) or not is_kimi_c8_draft_full_history(group.kv_cache_spec):
                raise ValueError("K3 draft cache group must be dedicated full-history CP1")
            owned.append(i)
    if not owned:
        raise ValueError("K3 draft has no owned cache group")
    result = copy(table)
    result.cp_size = 1
    result.cp_rank = 0
    result.cp_interleave = 1
    # Request-indexed scheduler pages are shared updates, not shared KV storage.
    # Batch-order tables and slots are independently owned by the draft so graph
    # preparation cannot overwrite captured target table/slot buffers.
    result.input_block_tables = [torch.zeros_like(t) for t in table.input_block_tables]
    result.slot_mappings = torch.full_like(table.slot_mappings, -1)
    result.init_block_table_layout_tensors()
    result.kimi_c8_owned_groups = tuple(owned)
    return result


def use_kimi_c8_draft_constructor(config):
    spec = getattr(config, "speculative_config", None)
    par = getattr(config, "parallel_config", None)
    model = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    return (
        getattr(model, "model_type", None) == "kimi_linear"
        and getattr(spec, "method", None) == "dspark"
        and getattr(spec, "kv_cache_dtype", None) == "mxfp8"
        and getattr(getattr(config, "cache_config", None), "cache_dtype", None) == "mxfp8"
        and getattr(par, "decode_context_parallel_size", None) == 1
        and getattr(par, "prefill_context_parallel_size", None) == 1
        and getattr(par, "pipeline_parallel_size", None) == 1
    )


def check_kimi_c8_target_and_draft_cp(config, speculator, kv_cache_config, check):
    """Run native compatibility checks against each real execution owner."""
    from vllm_ascend.core.kimi_c8_draft_cache import is_kimi_c8_draft_full_history

    if not use_kimi_c8_replicated_draft(config):
        raise ValueError("Scoped CP checks require qualified K3 dual-C8 target DCP2")
    draft_config = speculator.attn_vllm_config
    par = draft_config.parallel_config
    if (par.decode_context_parallel_size, par.prefill_context_parallel_size, par.cp_kv_cache_interleave_size) != (
        1,
        1,
        1,
    ):
        raise ValueError("Draft CP compatibility must use actual CP1/interleave1 configuration")
    names = set(speculator.draft_attn_layer_names)
    context = config.compilation_config.static_forward_context
    if not names or not names <= set(context):
        raise ValueError("Draft layer ownership must be nonempty and present in forward context")
    owned = set()
    for group in kv_cache_config.kv_cache_groups:
        selected = names & set(group.layer_names)
        if selected:
            if selected != set(group.layer_names) or not is_kimi_c8_draft_full_history(group.kv_cache_spec):
                raise ValueError("CP1 draft compatibility requires dedicated full-history descriptors")
            owned.update(selected)
    if owned != names:
        raise ValueError("Draft descriptor coverage is incomplete")

    def scoped(base, selected):
        result = copy(base)
        result.compilation_config = copy(base.compilation_config)
        result.compilation_config.static_forward_context = {k: v for k, v in context.items() if k in selected}
        return result

    # Every non-draft layer remains subject to the original target DCP/LSE
    # and interleave checks. No implementation capability is changed.
    target_names = set(context) - names
    check(scoped(config, target_names), target_names)
    check(scoped(draft_config, names), names)
