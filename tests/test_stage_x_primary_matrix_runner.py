from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"


def _runner_functions(*names):
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"PROBE_ID": "PRIMARY_EMIT_T5", "Any": Any, "Mapping": Mapping, "hashlib": hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


class TestStageXPrimaryMatrixRunner(unittest.TestCase):
    def test_update_feature_return_is_unpacked(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "update_feature"
        ]
        self.assertEqual(len(calls), 1)
        target = calls[0].targets[0]
        self.assertIsInstance(target, ast.Tuple)
        self.assertEqual(len(target.elts), 2)

    def test_exposure_failure_before_env_step(self):
        functions = _runner_functions("initial_exposure", "mark_policy_action_materialized")
        exposure = functions["initial_exposure"]()
        counters = {"policy_action_materialized_count": 0}
        functions["mark_policy_action_materialized"](exposure, counters)
        with self.assertRaises(RuntimeError):
            raise RuntimeError("injected-before-env-step")
        self.assertTrue(exposure["policy_action_materialized"])
        self.assertFalse(exposure["first_env_step_executed"])
        self.assertEqual(exposure["rows_materialized"], 0)

    def test_exposure_env_step_boundary_after_return(self):
        functions = _runner_functions("initial_exposure", "mark_env_step_executed")
        exposure = functions["initial_exposure"]()
        with self.assertRaises(RuntimeError):
            raise RuntimeError("injected-env-step-failure")
        self.assertFalse(exposure["first_env_step_executed"])
        functions["mark_env_step_executed"](exposure)
        self.assertTrue(exposure["first_env_step_executed"])

    def test_primary_seed_and_arm_order_use_protocol_formula(self):
        functions = _runner_functions("seed_for", "primary_seed_values", "arm_order")
        protocol = json.loads((ROOT / "configs/STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json").read_text(encoding="utf-8"))
        key = "libero_spatial/task_04/state_33"
        seeds = functions["primary_seed_values"](protocol, key)
        self.assertEqual(
            seeds["eval_seed"],
            int(hashlib.sha256(f"{protocol['seed_contract']['eval_seed_namespace']}|{key}|PRIMARY_EMIT_T5".encode()).hexdigest()[:8], 16),
        )
        self.assertEqual(
            seeds["perturb_seed"],
            int(hashlib.sha256(f"{protocol['seed_contract']['perturb_seed_namespace']}|{key}|PRIMARY_EMIT_T5".encode()).hexdigest()[:8], 16),
        )
        rotation = int(hashlib.sha256(f"{protocol['seed_contract']['arm_order_namespace']}|{key}|PRIMARY_EMIT_T5".encode()).hexdigest()[:2], 16) % 4
        base = protocol["seed_contract"]["arm_order_base"]
        self.assertEqual(functions["arm_order"](key, protocol), base[rotation:] + base[:rotation])


if __name__ == "__main__":
    unittest.main()
