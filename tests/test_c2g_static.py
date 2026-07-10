import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.gripper_attack.c2g_causal_vulnerability_detector import (
    C2gCausalVulnerabilityDetector,
    first_trigger_episode_losses,
    masked_bce,
)
from tools.multisuite_detector.audit_c2f_teacher_v1_labels import audit_teacher_v1
from tools.multisuite_detector.c2g_dataset_scaffold import (
    assign_episode_splits,
    assert_no_episode_leakage,
    assert_split_viability,
    context_feature_names,
    diagnostic_episode_permutation,
    split_label_coverage,
    task_episode_balanced_weights,
)


class C2gStaticTests(unittest.TestCase):
    def test_teacher_v1_audit_reports_grounding_reasons_and_spatial_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
                ep = root / suite / "episode"
                ep.mkdir(parents=True)
                meta = {"suite": suite, "task_index": 0}
                if suite != "libero_spatial":
                    meta["clean_success"] = True
                (ep / "episode_metadata.json").write_text(json.dumps(meta))
                rows = [
                    {"step": 4, "teacher_phase": "stable_carry", "teacher_event_role": "primary_attackable", "teacher_primary_attackable": 1, "features_25d": [0] * 25},
                    {"step": 5, "teacher_phase": "stable_carry", "teacher_event_role": "unsupported_or_abstain", "teacher_primary_attackable": 0, "features_25d": [0] * 25},
                ]
                if suite == "libero_goal":
                    rows[0]["teacher_primary_attackable"] = 0  # explicit field/role disagreement
                if suite == "libero_spatial":
                    rows[1]["features_25d"][5] = 0.9
                    rows[1]["features_25d"][19] = 0.01
                (ep / "step_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            report, by_task, reasons = audit_teacher_v1(root)
            self.assertTrue(report["status"].startswith("PASS_"))
            self.assertTrue(report["all_required_suites_present"])
            self.assertEqual(sum(row["spatial_absolute_z_fallback_candidate_count"] for row in by_task), 1)
            self.assertEqual(sum(row["window_count"] for row in reasons if row["reason_code"] == "V1_NO_GROUNDED_OBJECT"), 4)
            self.assertEqual(report["clean_success_unknown_episode_count"], 1)
            self.assertEqual(report["primary_field_role_disagreement_count"], 1)
            spatial = next(row for row in by_task if row["suite"] == "libero_spatial")
            self.assertEqual(spatial["clean_success_unknown_episode_count"], 1)
            self.assertNotIn("clean_not_success_episode_count", spatial)

    def test_context_modes_keep_legacy_shortcut_diagnostic_only(self):
        columns = ["ctx_suite_libero_10", "ctx_task_hash_01", "feature_0"]
        self.assertEqual(context_feature_names("no_context", columns), [])
        self.assertEqual(context_feature_names("suite_only", columns), ["ctx_suite_libero_10"])
        self.assertEqual(context_feature_names("full_context_legacy", columns), ["ctx_suite_libero_10", "ctx_task_hash_01"])

    def test_split_modes_and_no_episode_leakage(self):
        rows = [
            {"episode_key": f"{suite}/task_{task}/ep_{i}", "suite": suite, "task_index": task}
            for suite in ("libero_10", "libero_goal") for task in (0, 1) for i in range(8)
        ]
        for mode, kwargs in [
            ("within-task", {}),
            ("leave-one-task-out", {"held_out_task": "libero_10:1"}),
            ("leave-one-suite-out", {"held_out_suite": "libero_goal"}),
        ]:
            assignments = assign_episode_splits(rows, mode, **kwargs)
            assigned = [{**row, "split": assignments[row["episode_key"]]} for row in rows]
            assert_no_episode_leakage(assigned)
            if mode == "within-task":
                for suite in ("libero_10", "libero_goal"):
                    for task in (0, 1):
                        self.assertEqual(
                            {assignments[row["episode_key"]] for row in rows if row["suite"] == suite and row["task_index"] == task},
                            {"train", "val", "test"},
                        )
        self.assertTrue(all(assignments[row["episode_key"]] == "test" for row in rows if row["suite"] == "libero_goal"))

    def test_split_viability_preserves_unknown_and_hard_gates_empty_labels(self):
        rows = [
            {"episode_key": "tr-p", "split": "train", "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "tr-n", "split": "train", "label_known_mask": 1, "y_cmdopen_vulnerable": 0},
            {"episode_key": "va-p", "split": "val", "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "va-n", "split": "val", "label_known_mask": 1, "y_cmdopen_vulnerable": 0},
            {"episode_key": "te-p", "split": "test", "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "te-n", "split": "test", "label_known_mask": 1, "y_cmdopen_vulnerable": 0},
            {"episode_key": "te-u", "split": "test", "label_known_mask": 0, "y_cmdopen_vulnerable": 0},
        ]
        coverage = split_label_coverage(rows)
        self.assertEqual(coverage["test"]["unknown"], 1)
        assert_split_viability(coverage)
        broken = {split: dict(values) for split, values in coverage.items()}
        broken["val"]["known_positive"] = 0
        with self.assertRaises(ValueError):
            assert_split_viability(broken)

    def test_task_episode_balanced_weights_and_diagnostic_permutations(self):
        rows = [
            {"episode_key": "a", "suite": "s1", "task_index": 0, "split": "train"},
            {"episode_key": "a", "suite": "s1", "task_index": 0, "split": "train"},
            {"episode_key": "b", "suite": "s1", "task_index": 1, "split": "train"},
            {"episode_key": "c", "suite": "s2", "task_index": 1, "split": "test"},
            {"episode_key": "d", "suite": "s2", "task_index": 2, "split": "test"},
        ]
        weights = task_episode_balanced_weights(rows)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])
        p1 = diagnostic_episode_permutation(rows, seed=7, diagnostic="shuffled-language")
        p2 = diagnostic_episode_permutation(rows, seed=7, diagnostic="shuffled-language")
        self.assertEqual(p1, p2)
        split_by_episode = {row["episode_key"]: row["split"] for row in rows}
        self.assertTrue(all(split_by_episode[src] == split_by_episode[dst] for src, dst in p1.items()))
        wrong = diagnostic_episode_permutation(rows, seed=7, diagnostic="wrong-language-cross-task")
        task_by_episode = {row["episode_key"]: (row["suite"], row["task_index"]) for row in rows}
        self.assertTrue(all(task_by_episode[src] != task_by_episode[dst] for src, dst in wrong.items()))

    def test_c2g_model_is_causal_and_has_no_task_index_input(self):
        model = C2gCausalVulnerabilityDetector(visual_dim=8, language_dim=6, hidden=12, dropout=0.0).eval()
        self.assertEqual(list(inspect.signature(model.forward).parameters)[:4], ["temporal_25d", "siglip_visual", "language", "return_sequence"])
        x1 = torch.zeros(2, 5, 25)
        x2 = x1.clone()
        x2[:, 4] = 10
        visual = torch.zeros(2, 8)
        language = torch.zeros(2, 6)
        with torch.no_grad():
            y1 = model(x1, visual, language, return_sequence=True)
            y2 = model(x2, visual, language, return_sequence=True)
        self.assertEqual(set(y1), {"vulnerability", "release_safe", "contact", "grounding"})
        torch.testing.assert_close(y1["vulnerability"][:, :4], y2["vulnerability"][:, :4])

    def test_weighted_bce_normalizes_by_active_weight_mass(self):
        logits = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.0, 1.0])
        mask = torch.tensor([1, 1], dtype=torch.bool)
        weight = torch.tensor([1.0, 3.0])
        loss = masked_bce(logits, target, mask, weight)
        self.assertAlmostEqual(float(loss), float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target)), places=6)
        self.assertEqual(float(masked_bce(logits, target, torch.zeros_like(mask), weight)), 0.0)

    def test_first_trigger_losses_mask_unknown_windows_and_require_known_negative(self):
        logits = torch.tensor([
            [0.0, 0.0, 2.0, 20.0],   # positive; final high score is unknown
            [-2.0, -2.0, -2.0, 20.0], # partial-known negative; must not be penalized
            [-2.0, -2.0, -2.0, -2.0], # fully-known negative
        ])
        labels = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        known = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.bool)
        fully_negative = torch.tensor([0, 0, 1], dtype=torch.bool)
        losses = first_trigger_episode_losses(logits, labels, known, fully_negative)
        self.assertEqual(set(losses), {"early_emit", "episode_miss", "negative_episode_any_emit"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertAlmostEqual(float(losses["negative_episode_any_emit"]), float(torch.sigmoid(torch.tensor(-2.0))), places=6)


if __name__ == "__main__":
    unittest.main()
