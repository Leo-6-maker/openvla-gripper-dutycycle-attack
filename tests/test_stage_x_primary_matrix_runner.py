from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"


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


if __name__ == "__main__":
    unittest.main()
