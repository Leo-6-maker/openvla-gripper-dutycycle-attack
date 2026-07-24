import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gripper_attack.c2g_clean_policy_signals import (
    CLEAN_POLICY_FEATURE_NAMES,
)
from tools.multisuite_detector.audit_c2g_r5_bound_dataset import (
    ENGINEERING_PASS,
    PASS_STATUS,
    SCIENTIFIC_HOLD,
    SCIENTIFIC_PASS,
    R5_BASE_STATUS,
    R5_BOUND_SCHEMA,
    R5_BOUND_STATUS,
    TrainabilityThresholds,
    _assert_external_new_report,
    _validate_feature_contract,
    audit_bound_dataset,
    reconstruct_episodes,
)
from tools.multisuite_detector.materialize_c2g_clean_window_dataset import (
    DATASET_SCHEMA_VERSION,
    HEADS,
    SUITES,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class R6Fixture:
    def __init__(self, root: Path, *, viable: bool):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        if viable:
            specs = []
            for suite_index, suite in enumerate(SUITES):
                for split_index, split in enumerate(("train", "val", "test")):
                    specs.append(
                        (
                            suite,
                            f"{suite}/task_{suite_index}/ep_{split_index}",
                            suite_index,
                            split,
                            (suite_index + split_index) % 2 == 0,
                        )
                    )
        else:
            split_map = {
                "libero_10": "train",
                "libero_goal": "train",
                "libero_object": "test",
                "libero_spatial": "val",
            }
            positive_map = {
                "libero_10": True,
                "libero_goal": False,
                "libero_object": True,
                "libero_spatial": True,
            }
            specs = [
                (
                    suite,
                    f"{suite}/task_{index}/ep_0",
                    index,
                    split_map[suite],
                    positive_map[suite],
                )
                for index, suite in enumerate(SUITES)
            ]
        records = []
        for suite, episode, task_index, split, positive in specs:
            critical = np.asarray(
                [0, 1, 1, 1, 0] if positive else [0, 0, 0, 0, 0],
                dtype=np.float32,
            )
            start = np.asarray(
                [0, 1, 0, 0, 0] if positive else [0, 0, 0, 0, 0],
                dtype=np.float32,
            )
            auxiliary = critical.copy()
            release = np.zeros(5, dtype=np.float32)
            grounding = np.ones(5, dtype=np.float32)
            global_heads = {}
            for head in HEADS:
                if head in {"critical_window", "window_active"}:
                    global_heads[head] = critical
                elif head == "window_start":
                    global_heads[head] = start
                elif head == "release_safe":
                    global_heads[head] = release
                elif head == "grounding_confidence":
                    global_heads[head] = grounding
                else:
                    global_heads[head] = auxiliary
            for offset in range(3):
                records.append(
                    {
                        "suite": suite,
                        "episode": episode,
                        "task_index": task_index,
                        "split": split,
                        "positive": positive,
                        "step": offset + 2,
                        "heads": {
                            head: values[offset : offset + 3]
                            for head, values in global_heads.items()
                        },
                    }
                )

        n = len(records)
        window = 3
        payload = {
            "schema_version": np.asarray(DATASET_SCHEMA_VERSION),
            "X_proprio": np.zeros((n, window, 25), dtype=np.float32),
            "X_policy": np.zeros(
                (n, window, len(CLEAN_POLICY_FEATURE_NAMES)),
                dtype=np.float32,
            ),
            "X_visual": np.zeros((n, 8), dtype=np.float16),
            "X_language": np.zeros((n, 6), dtype=np.float16),
            "suite": np.asarray([row["suite"] for row in records]),
            "task_index": np.asarray(
                [row["task_index"] for row in records], dtype=np.int64
            ),
            "episode_key": np.asarray([row["episode"] for row in records]),
            "step": np.asarray([row["step"] for row in records], dtype=np.int64),
            "split": np.asarray([row["split"] for row in records]),
            "episode_fully_known_negative": np.asarray(
                [not row["positive"] for row in records], dtype=np.bool_
            ),
            "sample_weight": np.ones((n, window), dtype=np.float32),
            "feature_names_policy": np.asarray(CLEAN_POLICY_FEATURE_NAMES),
        }
        for head in HEADS:
            payload[f"y_{head}"] = np.stack(
                [row["heads"][head] for row in records]
            ).astype(np.float32)
            payload[f"m_{head}"] = np.ones((n, window), dtype=np.bool_)

        self.dataset = (
            self.root
            / "c2g_clean_window_w16_openvla_siglip_within_task.npz"
        )
        np.savez_compressed(self.dataset, **payload)
        per_suite = {}
        for suite in SUITES:
            suite_dir = self.root / "per_suite" / suite
            suite_dir.mkdir(parents=True, exist_ok=True)
            indices = np.flatnonzero(payload["suite"].astype(str) == suite)
            suite_payload = {
                key: (
                    value
                    if key in {"schema_version", "feature_names_policy"}
                    else value[indices]
                )
                for key, value in payload.items()
            }
            suite_dataset = (
                suite_dir
                / "c2g_clean_window_w16_openvla_siglip_within_task.npz"
            )
            np.savez_compressed(suite_dataset, **suite_payload)
            manifest = suite_dir / "c2g_clean_window_input_manifest.jsonl"
            manifest.write_text(
                json.dumps({"path": "episode", "sha256": "0" * 64}) + "\n",
                encoding="utf-8",
            )
            errors = (
                suite_dir / "c2g_clean_window_materialization_errors.jsonl"
            )
            errors.write_text("", encoding="utf-8")
            suite_report = (
                suite_dir / "c2g_clean_window_materialization_report.json"
            )
            episode_count = len(
                set(payload["episode_key"][indices].astype(str).tolist())
            )
            suite_report_value = {
                "status": "PASS_MATERIALIZED",
                "n_episode_errors": 0,
                "n_windows": int(indices.size),
                "n_episodes_processed": episode_count,
                "input_manifest_path": str(manifest.resolve()),
                "input_manifest_sha256": sha256_file(manifest),
                "error_ledger_path": str(errors.resolve()),
            }
            suite_report.write_text(
                json.dumps(suite_report_value),
                encoding="utf-8",
            )
            per_suite[suite] = {
                "dataset_path": str(suite_dataset.resolve()),
                "dataset_sha256": sha256_file(suite_dataset),
                "report_path": str(suite_report.resolve()),
                "report_sha256": sha256_file(suite_report),
                "model_path": f"/models/{suite}",
                "n_windows": int(indices.size),
                "n_episodes_processed": episode_count,
            }

        split_counts = {
            split: int(np.sum(payload["split"].astype(str) == split))
            for split in ("train", "val", "test")
        }
        self.base_report = (
            self.root / "c2g_multisuite_materialization_report.json"
        )
        base_value = {
            "status": R5_BASE_STATUS,
            "combined_dataset": str(self.dataset.resolve()),
            "combined_dataset_sha256": sha256_file(self.dataset),
            "combined_samples": n,
            "split_counts": split_counts,
            "per_suite": per_suite,
            "boundaries": {
                "clean_only": True,
                "attack_outcomes_read": False,
                "suite_task_identity_used_as_model_feature": False,
            },
        }
        self.base_report.write_text(json.dumps(base_value), encoding="utf-8")
        self.materialization_head = "a" * 40
        self.r4_binding = self.root.parent / "r4_binding.json"
        self.r4_binding.write_text(
            json.dumps(
                {
                    "status": "PASS_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING",
                    "audit_head": self.materialization_head,
                }
            ),
            encoding="utf-8",
        )
        self.bound_report = (
            self.root / "c2g_r5_bound_materialization_report.json"
        )
        bound_value = {
            "schema": R5_BOUND_SCHEMA,
            "status": R5_BOUND_STATUS,
            "audit_head": self.materialization_head,
            "r4_provenance_binding": str(self.r4_binding.resolve()),
            "r4_provenance_binding_sha256": sha256_file(self.r4_binding),
            "combined_dataset": str(self.dataset.resolve()),
            "combined_dataset_sha256": sha256_file(self.dataset),
            "base_report": str(self.base_report.resolve()),
            "base_report_sha256": sha256_file(self.base_report),
            "combined_samples": n,
            "split_counts": split_counts,
            "per_suite": per_suite,
            "boundaries": {
                "clean_only": True,
                "attack_outcomes_read": False,
                "counterfactual_read": False,
                "suite_task_identity_used_as_model_feature": False,
                "libero_rollouts_launched": 0,
                "attacks_launched": 0,
                "training_epochs": 0,
            },
        }
        self.bound_report.write_text(
            json.dumps(bound_value),
            encoding="utf-8",
        )

    def audit(self, **kwargs):
        return audit_bound_dataset(
            dataset_path=self.dataset,
            bound_report_path=self.bound_report,
            base_report_path=self.base_report,
            expected_dataset_sha256=sha256_file(self.dataset),
            expected_bound_report_sha256=sha256_file(self.bound_report),
            expected_base_report_sha256=sha256_file(self.base_report),
            expected_materialization_head=self.materialization_head,
            audit_head="b" * 40,
            **kwargs,
        )


class C2gR6BoundDatasetAuditTests(unittest.TestCase):
    def test_tiny_four_episode_fixture_passes_integrity_but_holds_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = R6Fixture(Path(temporary) / "dataset", viable=False)
            report = fixture.audit()
            self.assertEqual(report["status"], PASS_STATUS)
            self.assertEqual(
                report["engineering_smoke_status"],
                ENGINEERING_PASS,
            )
            self.assertEqual(
                report["scientific_trainability_status"],
                SCIENTIFIC_HOLD,
            )
            self.assertEqual(report["episode_count"], 4)
            goal = report["episode_support"]["per_suite"]["libero_goal"]
            self.assertEqual(goal["known_positive_step_count"], 0)
            reasons = {
                (row["scope"], row["field"])
                for row in report["scientific_trainability_violations"]
            }
            self.assertIn(("libero_goal", "known_positive_step_count"), reasons)
            self.assertEqual(
                report["training_authorization"],
                "HOLD_INSUFFICIENT_SCIENTIFIC_SUPPORT",
            )

    def test_structurally_viable_fixture_can_pass_configurable_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = R6Fixture(Path(temporary) / "dataset", viable=True)
            report = fixture.audit(
                thresholds=TrainabilityThresholds(
                    min_total_episodes=12,
                    min_total_tasks=4,
                    min_episodes_per_suite=3,
                    min_tasks_per_suite=1,
                    min_splits_per_suite=3,
                    min_train_episodes=4,
                    min_val_episodes=2,
                    min_test_episodes=2,
                    min_train_suites=4,
                    min_val_suites=2,
                    min_test_suites=2,
                )
            )
            self.assertEqual(
                report["scientific_trainability_status"],
                SCIENTIFIC_PASS,
            )
            self.assertEqual(
                report["training_authorization"],
                "HOLD_PENDING_EXPLICIT_TRAINING_AUTHORIZATION",
            )

    def test_expected_dataset_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = R6Fixture(Path(temporary) / "dataset", viable=False)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_bound_dataset(
                    dataset_path=fixture.dataset,
                    bound_report_path=fixture.bound_report,
                    base_report_path=fixture.base_report,
                    expected_dataset_sha256="0" * 64,
                    expected_bound_report_sha256=sha256_file(
                        fixture.bound_report
                    ),
                    expected_base_report_sha256=sha256_file(
                        fixture.base_report
                    ),
                    expected_materialization_head=fixture.materialization_head,
                    audit_head="b" * 40,
                )

    def test_overlap_inconsistency_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = R6Fixture(Path(temporary) / "dataset", viable=False)
            with np.load(fixture.dataset, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            data["y_window_start"] = data["y_window_start"].copy()
            data["y_window_start"][1, 0] = 0.0
            with self.assertRaisesRegex(ValueError, "overlapping target mismatch"):
                reconstruct_episodes(data)

    def test_runtime_report_must_not_modify_r5_output_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "dataset"
            dataset_root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the immutable"):
                _assert_external_new_report(
                    dataset_root / "r6_audit.json",
                    dataset_root,
                )

    def test_forbidden_outcome_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = R6Fixture(Path(temporary) / "dataset", viable=False)
            with np.load(fixture.dataset, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            data["attack_outcome"] = np.zeros(
                data["X_proprio"].shape[0],
                dtype=np.float32,
            )
            with self.assertRaisesRegex(ValueError, "field closure mismatch"):
                _validate_feature_contract(data)


if __name__ == "__main__":
    unittest.main()
