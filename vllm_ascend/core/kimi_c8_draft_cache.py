# SPDX-License-Identifier: Apache-2.0
"""K3 dual-C8 private full-history draft cache; no target CP semantics."""

from dataclasses import dataclass, fields

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class KimiC8DraftFullHistorySpec(FullAttentionSpec):
    def max_num_blocks_per_req(self, vllm_config, max_len):
        return cdiv(max_len, self.block_size)

    def max_memory_usage_bytes(self, vllm_config):
        return cdiv(vllm_config.model_config.max_model_len, self.block_size) * self.page_size_bytes

    def is_uniform_with_collection(self, kv_cache_specs):
        return all(type(s) is type(self) for s in kv_cache_specs.values())


class KimiC8DraftFullHistoryManager(FullAttentionManager):
    def __init__(self, kv_cache_spec, **kwargs):
        if not isinstance(kv_cache_spec, KimiC8DraftFullHistorySpec):
            raise TypeError("K3 full-history manager requires private draft descriptor")
        kwargs.update(dcp_world_size=1, pcp_world_size=1)
        super().__init__(kv_cache_spec, **kwargs)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        if not isinstance(kv_cache_spec, KimiC8DraftFullHistorySpec):
            raise TypeError("K3 full-history prefix lookup requires private descriptor")
        return super().find_longest_cache_hit(
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            1,
            1,
        )


def is_kimi_c8_draft_full_history(spec):
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return bool(spec.kv_cache_specs) and all(
            isinstance(s, KimiC8DraftFullHistorySpec) for s in spec.kv_cache_specs.values()
        )
    return isinstance(spec, KimiC8DraftFullHistorySpec)


def kimi_c8_draft_full_history_spec(spec):
    if type(spec) is not FullAttentionSpec or spec.sliding_window is not None or spec.attention_chunk_size is not None:
        raise ValueError("K3 replicated draft requires dedicated full-attention cache")
    return KimiC8DraftFullHistorySpec(**{f.name: getattr(spec, f.name) for f in fields(FullAttentionSpec)})


def register_kimi_c8_draft_cache_spec():
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=KimiC8DraftFullHistorySpec,
        manager_class=KimiC8DraftFullHistoryManager,
        uniform_type_base_spec=KimiC8DraftFullHistorySpec,
    )
