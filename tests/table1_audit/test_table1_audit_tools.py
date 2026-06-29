from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.table1_audit.build_condition_freeze_bundle import build as build_bundle
from tools.table1_audit.build_true_t10_manifest import build as build_true_t10
from tools.table1_audit.common import load_json, load_jsonl, write_json, write_jsonl
from tools.table1_audit.validate_formal_clean_closure import validate


def make_clean_tree(root: Path, *, mutate=None) -> tuple[Path, Path]:
    condition = root / "CLEAN"
    rows = []
    for fold in range(1, 10):
        for state in range(2):
            for det in range(1, 4):
                for pert in range(3):
                    out = condition / f"fold_{fold:02d}" / f"state_{state}" / f"det_seed_{det}" / f"pert_seed_{pert}"
                    out.mkdir(parents=True, exist_ok=True)
                    row = {
                        "job_key": f"f{fold:02d}_s{state}_d{det}_p{pert}",
                        "fold": f"{fold:02d}",
                        "state_id": state,
                        "detector_seed": det,
                        "perturbation_seed": pert,
                        "checkpoint_sha256": f"ckpt-{fold:02d}-{det}",
                        "output_dir": str(out),
                    }
                    summary = {
                        "task_success": True,
                        "state_id": state,
                        "runner_sha256": "runner-a",
                        "protocol_sha256": "proto-a",
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                    write_json(out / "episode_summary.json", summary)
                    (out / "step_telemetry.csv").write_text("step\n0\n", encoding="utf-8")
                    rows.append(row)
    if mutate:
        mutate(rows, condition)
    manifest = condition / "MANIFEST.jsonl"
    write_jsonl(manifest, rows)
    return manifest, condition


def run_validate(manifest: Path, condition: Path):
    return validate(SimpleNamespace(
        manifest=manifest,
        condition_root=condition,
        expected_rows=162,
        expected_parents=54,
        expected_replicates=3,
    ))


class FormalCleanValidatorTests(unittest.TestCase):
    def test_valid_162_rows_54_parents_3_replicates(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d))
            result = run_validate(manifest, condition)
            self.assertTrue(result["closure_pass"])
            self.assertEqual(result["row_count"], 162)
            self.assertEqual(result["parent_count"], 54)

    def test_missing_summary_fails(self):
        def mutate(rows, condition):
            Path(rows[0]["output_dir"], "episode_summary.json").unlink()

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertEqual(result["row_classes"]["active_or_incomplete"], 1)

    def test_duplicate_output_directory_fails(self):
        def mutate(rows, condition):
            rows[-1]["output_dir"] = rows[0]["output_dir"]

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertIn("duplicate_output", {p["class"] for p in result["problems"]})

    def test_replaced_state_fails(self):
        def mutate(rows, condition):
            p = Path(rows[0]["output_dir"], "episode_summary.json")
            data = load_json(p)
            data["state_id"] = 99
            write_json(p, data)

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertIn("replaced_state", {p["class"] for p in result["problems"]})

    def test_illegal_fourth_replicate_fails(self):
        def mutate(rows, condition):
            extra = dict(rows[0])
            extra["job_key"] = "extra_fourth"
            extra["perturbation_seed"] = 3
            extra_out = condition / "extra_fourth"
            extra_out.mkdir()
            write_json(extra_out / "episode_summary.json", {"task_success": True, "state_id": extra["state_id"], "runner_sha256": "runner-a"})
            extra["output_dir"] = str(extra_out)
            rows.append(extra)

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = validate(SimpleNamespace(manifest=manifest, condition_root=condition, expected_rows=163, expected_parents=54, expected_replicates=3))
            self.assertFalse(result["closure_pass"])
            self.assertIn("replicate_count", {p["class"] for p in result["problems"]})

    def test_legal_terminal_invalid_does_not_require_task_success(self):
        def mutate(rows, condition):
            p = Path(rows[0]["output_dir"], "episode_summary.json")
            write_json(p, {"terminal_status": "SCIENTIFIC_INVALID", "state_id": rows[0]["state_id"], "runner_sha256": "runner-a"})

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertTrue(result["closure_pass"])
            self.assertEqual(result["row_classes"]["terminal_invalid"], 1)

    def test_mixed_runner_sha_fails(self):
        def mutate(rows, condition):
            p = Path(rows[0]["output_dir"], "episode_summary.json")
            data = load_json(p)
            data["runner_sha256"] = "runner-b"
            write_json(p, data)

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertIn("mixed_provenance", {p["class"] for p in result["problems"]})

    def test_malformed_json_fails(self):
        def mutate(rows, condition):
            Path(rows[0]["output_dir"], "episode_summary.json").write_text("{bad", encoding="utf-8")

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertEqual(result["row_classes"]["malformed"], 1)

    def test_orphan_output_fails(self):
        def mutate(rows, condition):
            orphan = condition / "orphan"
            orphan.mkdir()
            write_json(orphan / "episode_summary.json", {"task_success": True})

        with tempfile.TemporaryDirectory() as d:
            manifest, condition = make_clean_tree(Path(d), mutate=mutate)
            result = run_validate(manifest, condition)
            self.assertFalse(result["closure_pass"])
            self.assertIn("orphan_artifact", {p["class"] for p in result["problems"]})


class FreezeAndManifestTests(unittest.TestCase):
    def test_freeze_builder_dry_run_and_true_t10_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition = make_clean_tree(root)
            validation = run_validate(manifest, condition)
            validation_json = root / "validation.json"
            write_json(validation_json, validation)
            dest = root / "bundle"
            preview = build_bundle(SimpleNamespace(
                validator_json=validation_json,
                manifest=manifest,
                condition_root=condition,
                dest=dest,
                condition_id="CLEAN",
                dry_run=True,
            ))
            self.assertTrue(preview["dry_run"])
            self.assertFalse(dest.exists())
            build_bundle(SimpleNamespace(
                validator_json=validation_json,
                manifest=manifest,
                condition_root=condition,
                dest=dest,
                condition_id="CLEAN",
                dry_run=False,
            ))
            out_manifest = root / "TRUE_T10_MANIFEST.jsonl"
            dry = build_true_t10(SimpleNamespace(
                clean_bundle=dest,
                output_root=Path("/mnt/sdc/dty_user/openvla_attack/evidence/true_t10_synthetic"),
                output_manifest=out_manifest,
                condition_id="TRUE_T10",
                runner_sha256="r" * 64,
                config_sha256="c" * 64,
                metric_schema_sha256="m" * 64,
                write=False,
            ))
            self.assertEqual(dry["row_count"], 162)
            build_true_t10(SimpleNamespace(
                clean_bundle=dest,
                output_root=Path("/mnt/sdc/dty_user/openvla_attack/evidence/true_t10_synthetic"),
                output_manifest=out_manifest,
                condition_id="TRUE_T10",
                runner_sha256="r" * 64,
                config_sha256="c" * 64,
                metric_schema_sha256="m" * 64,
                write=True,
            ))
            rows = load_jsonl(out_manifest)
            self.assertEqual(len(rows), 162)
            self.assertEqual(rows[0]["no_emission_policy"], "ITT_RETAIN")

    def test_true_t10_rejects_missing_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest, condition = make_clean_tree(root)
            validation_json = root / "validation.json"
            write_json(validation_json, run_validate(manifest, condition))
            dest = root / "bundle"
            build_bundle(SimpleNamespace(
                validator_json=validation_json,
                manifest=manifest,
                condition_root=condition,
                dest=dest,
                condition_id="CLEAN",
                dry_run=False,
            ))
            with self.assertRaises(SystemExit):
                build_true_t10(SimpleNamespace(
                    clean_bundle=dest,
                    output_root=Path("/mnt/sdc/dty_user/openvla_attack/evidence/true_t10_synthetic"),
                    output_manifest=root / "x.jsonl",
                    condition_id="TRUE_T10",
                    runner_sha256="MISSING",
                    config_sha256="c" * 64,
                    metric_schema_sha256="m" * 64,
                    write=False,
                ))


if __name__ == "__main__":
    unittest.main()
