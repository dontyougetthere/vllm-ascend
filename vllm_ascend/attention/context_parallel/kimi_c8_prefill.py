# SPDX-License-Identifier: Apache-2.0
"""Neutral empty-history contributions for the qualified K3 C8 prefill path."""

from numbers import Integral

import torch


def needs_kimi_c8_empty_context_neutralization(impl):
    config = getattr(impl, "vllm_config", None)
    model = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    return (
        getattr(model, "model_type", None) == "kimi_linear"
        and getattr(getattr(config, "cache_config", None), "cache_dtype", None) == "fp8"
        and bool(getattr(impl, "use_mixed_kv_cache_store", False))
        and bool(getattr(impl, "fa_quant_layer", False))
        and bool(getattr(impl, "support_fp8_attention", False))
        and bool(getattr(impl, "dcp_enabled", False))
        and getattr(impl, "dcp_size", None) == 2
        and not bool(getattr(impl, "pcp_enabled", False))
        and not bool(getattr(impl, "is_draft_model", False))
    )


def neutralize_kimi_c8_empty_context(output, lse, query_ends, global_kv_ends):
    """Mutate only empty global-history query rows (and query padding).

    Ends are the host cumulative lengths passed to this exact FIA call,
    after DCP KV reorganization. A rank-local zero length is NOT sufficient:
    another DCP rank may own valid history. No device-to-host read or stream
    synchronization is used here. Valid contributions remain byte unchanged.
    """
    if output.dtype != torch.float32 or lse.dtype != torch.float32:
        raise TypeError("K3 prefill merge contributions must stay FP32")
    if output.ndim != 3 or lse.shape != (*output.shape[:2], 1) or lse.device != output.device:
        raise ValueError("K3 prefill output/LSE shape or device mismatch")
    if (
        not isinstance(query_ends, (list, tuple))
        or not isinstance(global_kv_ends, (list, tuple))
        or not query_ends
        or len(query_ends) != len(global_kv_ends)
    ):
        raise ValueError("K3 prefill requires matching nonempty host cumulative-length lists")
    if any(not isinstance(x, Integral) or isinstance(x, bool) for x in (*query_ends, *global_kv_ends)):
        raise TypeError("K3 prefill cumulative lengths must be host integers")
    q = [int(x) for x in query_ends]
    kv = [int(x) for x in global_kv_ends]
    if (
        any(b < a for a, b in zip([0] + q[:-1], q))
        or any(b < a for a, b in zip([0] + kv[:-1], kv))
        or q[-1] > output.shape[0]
    ):
        raise ValueError("K3 prefill cumulative lengths are invalid")
    q0 = kv0 = 0
    for q1, kv1 in zip(q, kv):
        if kv1 == kv0 and q1 > q0:
            output[q0:q1].zero_()
            lse[q0:q1].fill_(-float("inf"))
        q0, kv0 = q1, kv1
    if q0 < output.shape[0]:
        output[q0:].zero_()
        lse[q0:].fill_(-float("inf"))
    return output, lse
