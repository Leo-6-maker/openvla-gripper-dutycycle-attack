import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.multisuite_detector.materialize_c2g_multisuite_dataset_bound import (
    build_bound_invocation,
    build_materializer_command,
    preflight,
    run,
)


class R5BoundMaterializationTests(unittest.TestCase):
    def args(
        self,
        root: Path,
        *,
        dry_run: bool = True,
    ):
        input_root = root / "collection"
        input_root.mkdir()
        output_parent = root / "external"
        output_parent.mkdir()
        output_dir = output_parent / "new" / "dataset"
        files = {}
        for name in (
            "r4.json",
            "model_map.json",
            "model_report.json",
            "goal.json",
            "model_verification.json",
        ):
            path = root / name
            path.write_text("{}\n", encoding="utf-8")
            files[name] = path
        return argparse.Namespace(
            input_root=input_root,
            output_dir=output_dir,
            r4_provenance_binding=files["r4.json"],
            audit_head="b" * 40,
            suite_model_map=files["model_map.json"],
            suite_model_report=files["model_report.json"],
            goal_model_manifest=files["goal.json"],
            model_verification_report=files[
                "model_verification.json"
            ],
            backend="openvla_siglip",
            device="cuda:0",
            embedding_dim=128,
            window=16,
            burst_length=10,
            split_mode="within_task",
            held_out_task="",
            held_out_suite="",
            val_fraction=0.15,
            test_fraction=0.15,
            seed=42,
            positive_weight=2.0,
            max_episodes_per_suite=1,
            min_free_bytes=1,
            dry_run=dry_run,
        )

    def configure_pass_mocks(
        self,
        args,
        verify_binding_mock,
        verify_models_mock,
        disk_usage_mock,
    ):
        verify_binding_mock.return_value = {
            "collection_head": "a" * 40,
            "audit_head": args.audit_head,
        }
        suite_models = {
            "libero_object": {"digest": "1"}
        }
        verify_models_mock.return_value = {
            "suite_models": suite_models,
            "status": (
                "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION"
            ),
        }
        args.model_verification_report.write_text(
            json.dumps(
                {
                    "status": (
                        "PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION"
                    ),
                    "frozen_report_sha256": "f" * 64,
                    "suite_models": suite_models,
                }
            ),
            encoding="utf-8",
        )
        disk_usage_mock.return_value = type(
            "Usage",
            (),
            {"free": 100 * 1024**3},
        )()

    def test_materializer_command_binds_audit_head_and_collection(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            command = build_materializer_command(args)
            self.assertIn(
                str(args.input_root.resolve()),
                command,
            )
            self.assertIn(
                str(args.output_dir.resolve()),
                command,
            )
            self.assertEqual(
                command[command.index("--git-commit") + 1],
                args.audit_head,
            )
            self.assertEqual(
                command[
                    command.index(
                        "--max-episodes-per-suite"
                    )
                    + 1
                ],
                "1",
            )

    def test_bound_invocation_contains_complete_provenance(
        self,
    ):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            command = build_bound_invocation(args)
            self.assertIn(
                str(args.r4_provenance_binding.resolve()),
                command,
            )
            self.assertEqual(
                command[command.index("--audit-head") + 1],
                args.audit_head,
            )
            self.assertIn(
                str(args.suite_model_map.resolve()),
                command,
            )
            self.assertIn("--dry-run", command)

    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "sha256_file",
        return_value="f" * 64,
    )
    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "shutil.disk_usage"
    )
    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "verify_models"
    )
    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "verify_binding"
    )
    def test_dry_run_is_read_only_and_passes_preflight(
        self,
        verify_binding_mock,
        verify_models_mock,
        disk_usage_mock,
        _sha_mock,
    ):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(
                Path(td),
                dry_run=True,
            )
            self.configure_pass_mocks(
                args,
                verify_binding_mock,
                verify_models_mock,
                disk_usage_mock,
            )
            with patch(
                "tools.multisuite_detector."
                "materialize_c2g_multisuite_dataset_bound."
                "subprocess.run"
            ) as subprocess_mock:
                report = run(args)
            subprocess_mock.assert_not_called()
            self.assertEqual(
                report["status"],
                "PASS_C2G_R5_BOUND_MATERIALIZATION_DRY_RUN",
            )
            self.assertEqual(
                report["collection_head"],
                "a" * 40,
            )
            self.assertEqual(
                report["audit_head"],
                args.audit_head,
            )
            self.assertIn(
                str(args.r4_provenance_binding.resolve()),
                report["command"],
            )
            self.assertEqual(
                report["command"][
                    report["command"].index("--audit-head")
                    + 1
                ],
                args.audit_head,
            )
            self.assertFalse(args.output_dir.exists())
            disk_usage_mock.assert_called_once_with(
                args.output_dir.parent.parent
            )

    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "verify_binding"
    )
    def test_preflight_rejects_nonempty_output(
        self,
        verify_binding_mock,
    ):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            args.output_dir.mkdir(parents=True)
            (args.output_dir / "stale.txt").write_text(
                "stale",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "must be empty",
            ):
                preflight(args)
            verify_binding_mock.assert_not_called()

    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "verify_binding"
    )
    def test_preflight_rejects_output_inside_collection(
        self,
        verify_binding_mock,
    ):
        with tempfile.TemporaryDirectory() as td:
            args = self.args(Path(td))
            args.output_dir = (
                args.input_root / "materialized_dataset"
            )
            with self.assertRaisesRegex(
                ValueError,
                "disjoint from the frozen collection",
            ):
                preflight(args)
            verify_binding_mock.assert_not_called()

    @patch(
        "tools.multisuite_detector."
        "materialize_c2g_multisuite_dataset_bound."
        "verify_binding"
    )
    def test_preflight_rejects_output_ancestor_of_collection(
        self,
        verify_binding_mock,
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self.args(root)
            args.output_dir = root
            with self.assertRaisesRegex(
                ValueError,
                "disjoint from the frozen collection",
            ):
                preflight(args)
            verify_binding_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
