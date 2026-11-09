"""Scoped CPU contracts; tensor/collective integration is covered by task NPU runs."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def method(relative, cls, name, namespace):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(root / relative), "exec"), namespace)
    return namespace[name]


class MixedC8DCPContracts(unittest.TestCase):
    def gate(self):
        fn = method(
            "vllm_ascend/models/kimi_k3.py",
            "AscendKimiMLAAttention",
            "_configure_mixed_kv_cache_store",
            {"torch": torch, "enable_sfa": lambda cfg: False},
        )
        impl = NS(
            vllm_config=NS(
                model_config=NS(hf_config=NS(model_type="kimi_k3"), dtype=torch.bfloat16), speculative_config=None
            ),
            fa_quant_layer=True,
            support_fp8_attention=True,
            is_draft_model=False,
            enable_kv_nz=False,
            pcp_enabled=False,
            dcp_enabled=True,
            supports_mixed_c8_dcp=True,
            num_kv_heads=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            use_mixed_kv_cache_store=False,
        )
        return fn, NS(impl=impl)

    def test_qualified_dcp_gate(self):
        fn, obj = self.gate()
        fn(obj, False)
        self.assertTrue(obj.impl.use_mixed_kv_cache_store)

    def test_unimplemented_cp_and_speculation_stay_guarded(self):
        for field, value in [("supports_mixed_c8_dcp", False), ("pcp_enabled", True), ("enable_kv_nz", True)]:
            fn, obj = self.gate()
            setattr(obj.impl, field, value)
            with self.assertRaises(NotImplementedError):
                fn(obj, False)
        fn, obj = self.gate()
        obj.impl.vllm_config.speculative_config = NS()
        with self.assertRaises(NotImplementedError):
            fn(obj, False)

    def test_draft_not_claimed(self):
        fn, obj = self.gate()
        obj.impl.is_draft_model = True
        fn(obj, False)
        self.assertFalse(obj.impl.use_mixed_kv_cache_store)

    def test_mixed_prefill_values_are_not_descaled_twice(self):
        fn = method(
            "vllm_ascend/attention/context_parallel/mla_cp.py",
            "AscendMlaDCPImpl",
            "_reorg_kvcache",
            {"torch": torch, "DCPChunkedContextMetadata": NS, "ChunkedContextMetadata": NS},
        )
        kv = torch.arange(4.0).view(4, 1, 1).expand(4, 1, 512).to(torch.float8_e4m3fn)
        raw = (torch.arange(4.0) + 0.25).view(4, 1, 1).expand(4, 1, 64).to(torch.bfloat16)
        peer = torch.cat([(kv.float() + 8).to(torch.bfloat16), (raw.float() + 8).to(torch.bfloat16)], -1)
        obj = NS(
            use_mixed_kv_cache_store=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            _dcp_all_gather=lambda t, dim: torch.cat([t, peer], dim=dim),
        )
        chunk = NS(
            padded_local_chunk_seq_lens=[[2, 2]],
            local_context_lens_allranks=[[2, 1], [1, 2]],
            cu_seq_lens_lst=[[3, 6]],
            max_seq_lens=[3],
            chunk_size=2,
            kv_rank_offsets=[[0, 4]],
            pcp_global_req_indices=None,
        )
        latent, rope = fn(obj, kv, raw, chunk, 0, 4)
        expected = torch.tensor([0.0, 1.0, 8.0, 2.0, 10.0, 11.0]).view(6, 1, 1).expand(6, 1, 512)
        self.assertEqual(latent.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(latent.float(), expected))
        self.assertTrue(torch.equal(rope.float(), expected[:, :, :64] + 0.25))
        self.assertTrue(torch.equal(latent.float() * 0.25, expected * 0.25))


if __name__ == "__main__":
    unittest.main()
