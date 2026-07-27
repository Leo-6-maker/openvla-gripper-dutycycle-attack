"""Static regression tests for the R5-F lifecycle contract.

These tests intentionally avoid importing MuJoCo/OpenVLA. They parse the runner
AST so indentation regressions cannot silently reduce the 40-identity loop to
one episode, and so staging cleanup must cover collection, sealing, and publish.
"""
import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("run_r5f_full40_materialize.py")


def call_name(node):
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def subtree_calls(node, name):
    return any(call_name(child) == name for child in ast.walk(node))


class TestR5FLifecycleStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_text = SOURCE.read_text(encoding="utf-8")
        compile(cls.source_text, str(SOURCE), "exec")
        cls.tree = ast.parse(cls.source_text, filename=str(SOURCE))
        cls.main = next(
            node for node in cls.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )

    def test_identity_work_is_inside_for_loop(self):
        loops = [
            node for node in ast.walk(self.main)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "ident"
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "identities"
        ]
        self.assertEqual(len(loops), 1)
        loop = loops[0]
        assigned = {
            target.id
            for node in ast.walk(loop)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertTrue(
            {"suite", "task_idx", "state_id", "ep_id", "coll_seed"}.issubset(assigned)
        )
        self.assertTrue(subtree_calls(loop, "capture_one_episode"))
        self.assertTrue(subtree_calls(loop, "_validate_episode_shapes"))

    def test_outer_finally_covers_collect_seal_and_publish(self):
        lifecycle_tries = [
            node for node in self.main.body
            if isinstance(node, ast.Try) and node.finalbody
        ]
        matches = []
        for node in lifecycle_tries:
            body_wrapper = ast.Module(body=node.body, type_ignores=[])
            final_wrapper = ast.Module(body=node.finalbody, type_ignores=[])
            has_loop = any(
                isinstance(child, ast.For)
                and isinstance(child.iter, ast.Name)
                and child.iter.id == "identities"
                for child in ast.walk(body_wrapper)
            )
            if (
                has_loop
                and subtree_calls(body_wrapper, "seal_root")
                and subtree_calls(body_wrapper, "rename")
                and subtree_calls(final_wrapper, "rmtree")
            ):
                matches.append(node)
        self.assertEqual(
            len(matches), 1,
            "one outer try/finally must cover identities, seal_root, rename, and cleanup",
        )

    def test_published_guard_and_no_undefined_cleanup_helper(self):
        self.assertNotIn("_cleanup_staging", self.source_text)
        published_true = any(
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "published" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            for node in ast.walk(self.main)
        )
        self.assertTrue(published_true)
        self.assertIn("if not published and staging.exists()", self.source_text)


if __name__ == "__main__":
    unittest.main()
