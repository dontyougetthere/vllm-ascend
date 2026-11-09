# SPDX-License-Identifier: Apache-2.0
"""Only dedicated K3 full-history draft descriptors can use this CP1 builder."""

from vllm.config import set_current_vllm_config

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackend, AscendC8MXFPMetadataBuilder
from vllm_ascend.core.kimi_c8_draft_cache import is_kimi_c8_draft_full_history
from vllm_ascend.worker.v2.spec_decode.dspark.kimi_c8_cp import kimi_c8_draft_execution_config


class KimiC8DraftMetadataBuilder(AscendC8MXFPMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if not is_kimi_c8_draft_full_history(kv_cache_spec):
            raise ValueError("Private K3 draft builder requires independent full-history cache descriptor")
        config = kimi_c8_draft_execution_config(vllm_config)
        if (
            config.parallel_config.decode_context_parallel_size != 1
            or config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError("Private K3 draft builder must execute CP1")
        with set_current_vllm_config(config):
            super().__init__(kv_cache_spec, layer_names, config, device)


class KimiC8DraftBackend(AscendC8MXFPAttentionBackend):
    @staticmethod
    def get_impl_cls():
        from vllm.config import get_current_vllm_config

        from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl
        from vllm_ascend.worker.v2.spec_decode.dspark.kimi_c8_cp import use_kimi_c8_draft_constructor

        if not use_kimi_c8_draft_constructor(get_current_vllm_config()):
            raise ValueError("Private K3 draft implementation requires validated CP1 constructor config")
        # Do not consult or clear the process-global target enable_dcp cache.
        return AscendC8MXFPAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        return KimiC8DraftMetadataBuilder
