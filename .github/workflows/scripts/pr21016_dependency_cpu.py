# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run existing tensor contracts without importing vLLM or an NPU runtime.

The production wrappers are loaded by the PR's existing AST test loader.
The two parametrized device tests below are extracted unchanged, apart from
removing pytest decorators. Native operators remain mocked by those tests.
This is a targeted CPU diagnostic, not the repository CPU UT or NPU suite.
"""

import argparse
import ast
import io
import itertools
import json
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def extract_function(path, name, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[name]


def run_parameter_grid(function, parameters):
    failures = []
    total = 0
    for values in itertools.product(*parameters):
        total += 1
        try:
            function(*values)
        except Exception as error:
            failures.append({"parameters": values, "error": f"{type(error).__name__}: {error}"})
    return {"total": total, "passed": total - len(failures), "failures": failures}


def run_contracts(root, output, variant):
    loader = runpy.run_path(str(root / "tests/ut/device/test_a3_c8_padding.py"), run_name="contract_tests")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(loader["TestQuantizedSparseAttention"])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output / f"{variant}-a3-c8.log").write_text(stream.getvalue(), encoding="utf-8")
    wrapper = loader["load_sparse_wrapper"]()
    runtime = wrapper._execute_kv_quant_sparse_flash_attention_vllm.__globals__["torch_npu"]
    namespace = {"torch": torch, "mock": mock, "SimpleNamespace": SimpleNamespace, "BaseDeviceAdaptor": wrapper}
    tests = root / "tests/ut/device/test_device_op.py"
    padding = extract_function(tests, "test_sparse_attention_query_group_padding_preserves_outputs", namespace)
    padded = run_parameter_grid(padding, ([12, 64, 65, 96, 128], [1, 2], [False, True], [False, True]))
    native = extract_function(tests, "test_fp8_sparse_attention_native_heads_preserve_attention_and_lse", namespace)
    package = types.ModuleType("vllm_ascend")
    package.__path__ = []
    device = types.ModuleType("vllm_ascend.device")
    device.__path__ = []
    module = types.ModuleType("vllm_ascend.device.device_op")
    module.torch_npu = runtime
    package.device = device
    device.device_op = module
    modules = {"vllm_ascend": package, "vllm_ascend.device": device, "vllm_ascend.device.device_op": module}
    modes = [(False, False), (False, True), (True, False)]

    def native_case(group, heads, mode, rope):
        native(group, heads, *mode, rope)

    with mock.patch.dict(sys.modules, modules):
        fp8 = run_parameter_grid(native_case, ([8, 12, 48, 64, 96, 128], [1, 2], modes, [0, 2]))
    rotary_path = root / "vllm_ascend/ops/rotary_embedding.py"
    rotary_namespace = {"torch": torch, "_cos_mla": torch.ones(4, 1, 1, 16), "_sin_mla": torch.zeros(4, 1, 1, 16)}
    rotary = extract_function(rotary_path, "get_cos_and_sin_mla", rotary_namespace)
    rotary_error = None
    try:
        cos, sin = rotary(torch.tensor([1, 3]), use_rope=False)
        torch.testing.assert_close(cos, torch.ones(2, 1, 1, 16))
        torch.testing.assert_close(sin, torch.zeros(2, 1, 1, 16))
    except Exception as error:
        rotary_error = f"{type(error).__name__}: {error}"
    return {
        "a3_c8": {
            "test_methods": result.testsRun,
            "successful": result.wasSuccessful(),
            "errors": [{"test": case.id(), "trace": trace} for case, trace in result.errors],
            "failures": [{"test": case.id(), "trace": trace} for case, trace in result.failures],
        },
        "nonquant_padding": padded,
        "fp8_native": fp8,
        "nope_helper_error": rotary_error,
    }


def snapshot_revision(revision, root):
    paths = [
        "tests/ut/device/test_a3_c8_padding.py",
        "tests/ut/device/test_device_op.py",
        "vllm_ascend/device/device_op.py",
        "vllm_ascend/ops/rotary_embedding.py",
    ]
    for path in paths:
        content = subprocess.run(["git", "show", f"{revision}:{path}"], check=True, capture_output=True).stdout
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def is_successful(result):
    return (
        result["a3_c8"]["successful"]
        and not result["nonquant_padding"]["failures"]
        and not result["fp8_native"]["failures"]
        and result["nope_helper_error"] is None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="pr21016-baseline-") as directory:
        baseline_root = Path(directory)
        snapshot_revision(args.baseline, baseline_root)
        baseline = run_contracts(baseline_root, args.output, "baseline")
    candidate = run_contracts(Path.cwd(), args.output, "candidate")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    results = {"baseline_sha": args.baseline, "candidate_sha": sha, "torch": torch.__version__}
    results.update(baseline=baseline, candidate=candidate)
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, result in (("baseline", baseline), ("candidate", candidate)):
        summary = {
            "a3_c8_methods": result["a3_c8"]["test_methods"],
            "a3_c8_successful": result["a3_c8"]["successful"],
            "a3_c8_error_subcases": len(result["a3_c8"]["errors"]),
            "padding_passed": result["nonquant_padding"]["passed"],
            "padding_total": result["nonquant_padding"]["total"],
            "fp8_passed": result["fp8_native"]["passed"],
            "fp8_total": result["fp8_native"]["total"],
            "nope_helper_error": result["nope_helper_error"],
        }
        print(name, json.dumps(summary), flush=True)
    expected_baseline_failure = (
        not baseline["a3_c8"]["successful"]
        and bool(baseline["nonquant_padding"]["failures"])
        and baseline["nope_helper_error"] is not None
    )
    if not expected_baseline_failure:
        raise SystemExit("Baseline did not reproduce the expected missing dependencies; inspect results.json.")
    if not is_successful(candidate):
        raise SystemExit("Adapted candidate failed an existing contract; inspect results.json.")
    print("Expected baseline dependency failures reproduced; all targeted candidate contracts passed.")


if __name__ == "__main__":
    main()
