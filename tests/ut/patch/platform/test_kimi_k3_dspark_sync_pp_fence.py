"""CPU-only scoped lifecycle regression; no model or accelerator required."""

import ast
import copy
import sys
import types
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import patch


def load_helpers():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_pp_mtp.py"
    names = {
        "_use_kimi_k3_dspark_sync_pp_fence",
        "_fence_kimi_k3_dspark_output",
        "_release_kimi_k3_dspark_output",
        "_patch_scheduler_update_after_schedule",
        "_patch_scheduler_update_from_output",
    }
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(_PP_IN_FLIGHT_STEP=1 << 60, wraps=wraps, copy=copy, _use_pp_ipc_runtime_patch=lambda *args: False)
    exec(compile(tree, str(source), "exec"), ns)
    return ns


def config():
    return types.SimpleNamespace(
        use_v2_model_runner=True,
        scheduler_config=types.SimpleNamespace(async_scheduling=False),
        model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(model_type="kimi_linear")),
        speculative_config=types.SimpleNamespace(method="dspark", kv_cache_dtype="mxfp8"),
        cache_config=types.SimpleNamespace(cache_dtype="fp8"),
        kv_transfer_config=None,
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=2, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
    )


class FenceTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_helpers()
        self.req = types.SimpleNamespace(is_prefill_chunk=False, next_decode_eligible_step=0)
        self.s = types.SimpleNamespace(requests={"a": self.req}, vllm_config=config())
        self.out = types.SimpleNamespace(num_scheduled_tokens={"a": 65})

    def fence(self, out=None):
        self.ns["_fence_kimi_k3_dspark_output"](self.s, out or self.out)

    def release(self, out=None):
        self.ns["_release_kimi_k3_dspark_output"](self.s, out or self.out)

    def test_qualified_DCP2_scope(self):
        c = config()
        c.parallel_config.decode_context_parallel_size = 2
        c.speculative_config.draft_parallel_config = types.SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        )
        self.assertTrue(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](c))
        c.speculative_config.draft_parallel_config.decode_context_parallel_size = 2
        self.assertFalse(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](c))
        c.speculative_config.draft_parallel_config.decode_context_parallel_size = 1
        c.speculative_config.draft_parallel_config.prefill_context_parallel_size = 2
        self.assertFalse(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](c))

    def test_qualified_DCP2_native_wrapper_order(self):
        original = config

        def qualified():
            c = original()
            c.parallel_config.decode_context_parallel_size = 2
            c.speculative_config.draft_parallel_config = types.SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1
            )
            return c

        with patch.dict(self.test_actual_wrapper_order_and_native_return.__func__.__globals__, {"config": qualified}):
            self.test_actual_wrapper_order_and_native_return()

    def test_qualified_DCP2_stale_output_stays_fenced(self):
        self.s.vllm_config.parallel_config.decode_context_parallel_size = 2
        self.s.vllm_config.speculative_config.draft_parallel_config = types.SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        )
        self.assertTrue(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](self.s.vllm_config))
        self.test_older_output_does_not_release_newer_step()

    def test_scope_positive(self):
        self.assertTrue(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](config()))

    def test_scope_negative_matrix(self):
        cases = [
            ("use_v2_model_runner", False),
            ("scheduler_config.async_scheduling", True),
            ("scheduler_config.async_scheduling", None),
            ("model_config.hf_text_config.model_type", "other"),
            ("speculative_config.method", "mtp"),
            ("speculative_config.kv_cache_dtype", "auto"),
            ("cache_config.cache_dtype", "auto"),
            ("parallel_config.pipeline_parallel_size", 1),
            ("parallel_config.pipeline_parallel_size", 4),
            ("parallel_config.decode_context_parallel_size", 2),
            ("parallel_config.prefill_context_parallel_size", 2),
            ("kv_transfer_config", types.SimpleNamespace()),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                c = config()
                obj = c
                parts = path.split(".")
                for key in parts[:-1]:
                    obj = getattr(obj, key)
                setattr(obj, parts[-1], value)
                self.assertFalse(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](c))
        self.assertFalse(self.ns["_use_kimi_k3_dspark_sync_pp_fence"](None))

    def test_matching_output_releases(self):
        self.fence()
        self.assertEqual(self.req.next_decode_eligible_step, 1 << 60)
        self.release()
        self.assertEqual(self.req.next_decode_eligible_step, 0)
        self.assertFalse(self.s._kimi_k3_dspark_pending_outputs)

    def test_other_request_remains_schedulable(self):
        b = types.SimpleNamespace(is_prefill_chunk=False, next_decode_eligible_step=0)
        self.s.requests["b"] = b
        self.fence()
        self.assertEqual(b.next_decode_eligible_step, 0)

    def test_intermediate_prefill_unfenced(self):
        self.req.is_prefill_chunk = True
        self.fence()
        self.assertEqual(self.req.next_decode_eligible_step, 0)

    def test_older_output_does_not_release_newer_step(self):
        self.fence()
        new = types.SimpleNamespace(num_scheduled_tokens={"a": 8})
        self.fence(new)
        self.release()
        self.assertEqual(self.req.next_decode_eligible_step, 1 << 60)
        self.release(new)
        self.assertEqual(self.req.next_decode_eligible_step, 0)

    def test_equal_but_distinct_output_not_owner(self):
        self.fence()
        self.release(types.SimpleNamespace(num_scheduled_tokens={"a": 65}))
        self.assertEqual(self.req.next_decode_eligible_step, 1 << 60)

    def test_request_slot_reuse_not_released(self):
        self.fence()
        replacement = types.SimpleNamespace(next_decode_eligible_step=99)
        self.s.requests["a"] = replacement
        self.release()
        self.assertEqual(replacement.next_decode_eligible_step, 99)

    def test_finished_request_cleanup(self):
        self.fence()
        del self.s.requests["a"]
        self.release()
        self.assertFalse(self.s._kimi_k3_dspark_pending_outputs)

    def test_empty_output_and_duplicate_delivery(self):
        self.fence()
        self.release(types.SimpleNamespace(num_scheduled_tokens={}))
        self.assertEqual(self.req.next_decode_eligible_step, 1 << 60)
        self.release()
        self.release()
        self.assertEqual(self.req.next_decode_eligible_step, 0)

    def test_actual_wrapper_order_and_native_return(self):
        events = []
        req = self.req
        ns = self.ns

        class Scheduler:
            def _update_after_schedule(self, o):
                events.append("native-schedule")
                req.is_prefill_chunk = False

            def update_from_output(self, o, m):
                events.append(("native-update", req.next_decode_eligible_step))
                return m

        mod = types.ModuleType("vllm.v1.core.sched.scheduler")
        mod.Scheduler = Scheduler
        with patch.dict(sys.modules, {"vllm.v1.core.sched.scheduler": mod}):
            ns["_patch_scheduler_update_after_schedule"]()
            ns["_patch_scheduler_update_from_output"]()
        s = Scheduler()
        s.requests = {"a": req}
        s.vllm_config = config()
        s.use_pp = True
        s._update_after_schedule(self.out)
        marker = object()
        self.assertIs(s.update_from_output(self.out, marker), marker)
        self.assertEqual(events, ["native-schedule", ("native-update", 1 << 60)])
        self.assertEqual(req.next_decode_eligible_step, 0)

    def test_reduced_native_arithmetic_negative_control(self):
        # Exact scheduler arithmetic, reduced from schedule(): new = tokens + drafts - computed.
        # Before the preceding sample is committed: 65 + 7 - 65 = 7, but 8 logits.
        tokens, computed, drafts = 65, 65, 7
        scheduled = tokens + drafts - computed
        indices = list(range(scheduled - (drafts + 1), scheduled))
        self.assertEqual(indices, [-1, 0, 1, 2, 3, 4, 5, 6])
        # After one sample but with the newer 7-row step already scheduled:
        tokens += 1
        computed += scheduled
        scheduled = tokens + drafts - computed
        self.assertEqual(scheduled, 1)
        self.assertLess(scheduled - (drafts + 1), -scheduled)
        # Fencing until the sample is committed yields a complete 8-row verification.
        scheduled = 66 + 7 - 65
        self.assertEqual(list(range(scheduled - (drafts + 1), scheduled)), list(range(8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
