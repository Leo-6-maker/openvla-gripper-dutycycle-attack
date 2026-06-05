import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnostics" / "finalize_phase_response_labels.py"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FinalizePhaseResponseLabelsV2Test(unittest.TestCase):
    def run_builder(self, tmp, rows_by_source, candidate_rows_by_source=None, expect_ok=True):
        tmp = Path(tmp)
        paths = {}
        for name, rows in rows_by_source.items():
            path = tmp / ("%s.csv" % name)
            write_csv(path, rows)
            paths[name] = path
        candidate_paths = {}
        for name, rows in (candidate_rows_by_source or {}).items():
            path = tmp / ("%s_candidates.csv" % name)
            write_csv(path, rows)
            candidate_paths[name] = path
        out_labels = tmp / "labels_v2.csv"
        out_ready = tmp / "readiness.md"
        out_conflicts = tmp / "conflicts.csv"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--batch1-merged", str(paths.get("batch1", tmp / "missing_batch1.csv")),
            "--batch2b-vis", str(paths.get("batch2b", tmp / "missing_batch2b.csv")),
            "--batch3-vis", str(paths.get("batch3", tmp / "missing_batch3.csv")),
            "--batch3b-vis", str(paths.get("batch3b", tmp / "missing_batch3b.csv")),
            "--batch3c-vis", str(paths.get("batch3c", tmp / "missing_batch3c.csv")),
            "--batch2b-candidates", str(candidate_paths.get("batch2b", tmp / "missing_batch2b_candidates.csv")),
            "--batch3-candidates", str(candidate_paths.get("batch3", tmp / "missing_batch3_candidates.csv")),
            "--batch3b-candidates", str(candidate_paths.get("batch3b", tmp / "missing_batch3b_candidates.csv")),
            "--batch3c-candidates", str(candidate_paths.get("batch3c", tmp / "missing_batch3c_candidates.csv")),
            "--output-labels", str(out_labels),
            "--output-readiness", str(out_ready),
            "--output-conflicts", str(out_conflicts),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        if expect_ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return out_labels, out_ready, out_conflicts, result

    def test_role_specific_controls_and_manual_review_not_train(self):
        rows = [
            {
                "task_key": "cream_cheese",
                "state_id": "4",
                "window_start": "28",
                "window_end": "45",
                "claim_usable": "True",
                "denominator_clean": "True",
                "taxonomy_label": "claim_usable",
                "vis_OPEN_mean": "18",
                "vis_qpos_opening_delta_mean": "0.038",
                "vis_done_all_false": "True",
            },
            {
                "task_key": "salad_dressing",
                "state_id": "0",
                "window_start": "7",
                "window_end": "24",
                "candidate_role": "stable_post_lock_control",
                "denominator_clean": "True",
                "vis_OPEN_mean": "18",
                "vis_qpos_opening_delta_mean": "0.038",
                "vis_done_all_false": "True",
            },
            {
                "task_key": "bbq_sauce",
                "state_id": "5",
                "window_start": "27",
                "window_end": "44",
                "candidate_role": "far_too_early_control",
                "denominator_clean": "True",
                "vis_OPEN_mean": "18",
                "vis_qpos_opening_delta_mean": "0.038",
                "vis_done_all_false": "False",
            },
            {
                "task_key": "ketchup",
                "state_id": "1",
                "window_start": "21",
                "window_end": "38",
                "candidate_role": "pre_lock_control",
                "denominator_clean": "True",
                "vis_OPEN_mean": "18",
                "vis_qpos_opening_delta_mean": "0.038",
                "vis_done_all_false": "True",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            labels_path, _, conflicts_path, _ = self.run_builder(tmp, {"batch3b": rows})
            labels = read_csv(labels_path)
            by_task = {row["task_key"]: row for row in labels}
            self.assertEqual(by_task["cream_cheese"]["label_status"], "positive")
            self.assertEqual(by_task["salad_dressing"]["label_status"], "manual_review")
            self.assertEqual(by_task["salad_dressing"]["label_use"], "manual_review")
            self.assertEqual(by_task["bbq_sauce"]["label_status"], "negative")
            self.assertEqual(by_task["ketchup"]["label_status"], "positive")
            self.assertEqual(read_csv(conflicts_path), [])

    def test_duplicate_conflict_hard_fail_and_conflict_csv(self):
        positive = [{
            "task_key": "ketchup",
            "state_id": "1",
            "window_start": "21",
            "window_end": "38",
            "claim_usable": "True",
            "denominator_clean": "True",
            "taxonomy_label": "claim_usable",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "True",
        }]
        negative = [{
            "task_key": "ketchup",
            "state_id": "1",
            "window_start": "21",
            "window_end": "38",
            "claim_usable": "False",
            "denominator_clean": "True",
            "taxonomy_label": "physical_strong_task_negative",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "False",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            _, _, conflicts_path, _ = self.run_builder(
                tmp,
                {"batch3b": positive, "batch3c": negative},
                expect_ok=False,
            )
            conflicts = read_csv(conflicts_path)
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["reason"], "duplicate_label_conflict")

    def test_batch3c_stable_post_lock_role_join_from_candidates(self):
        summary = [{
            "task_key": "salad_dressing",
            "state_id": "0",
            "window_start": "7",
            "window_end": "24",
            "denominator_clean": "True",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "False",
        }]
        candidates = [{
            "task_key": "salad_dressing",
            "state_id": "0",
            "window_start": "7",
            "window_end": "24",
            "candidate_role": "stable_post_lock_control",
            "denominator_type": "late_open_control",
            "reason_selected": "synthetic stable post lock",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            labels_path, _, conflicts_path, _ = self.run_builder(
                tmp,
                {"batch3c": summary},
                {"batch3c": candidates},
            )
            labels = read_csv(labels_path)
            self.assertEqual(labels[0]["candidate_role"], "stable_post_lock_control")
            self.assertEqual(labels[0]["label_status"], "negative")
            self.assertEqual(labels[0]["label_vulnerability_ready"], "0")
            self.assertEqual(labels[0]["reason_selected"], "synthetic stable post lock")
            self.assertEqual(read_csv(conflicts_path), [])

    def test_batch3c_far_too_early_role_join_from_candidates(self):
        summary = [{
            "task_key": "bbq_sauce",
            "state_id": "5",
            "window_start": "27",
            "window_end": "44",
            "denominator_clean": "True",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "False",
        }]
        candidates = [{
            "task_key": "bbq_sauce",
            "state_id": "5",
            "window_start": "27",
            "window_end": "44",
            "candidate_role": "far_too_early_control",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            labels_path, _, _, _ = self.run_builder(
                tmp,
                {"batch3c": summary},
                {"batch3c": candidates},
            )
            label = read_csv(labels_path)[0]
            self.assertEqual(label["candidate_role"], "far_too_early_control")
            self.assertEqual(label["label_status"], "negative")
            self.assertEqual(label["label_vulnerability_ready"], "0")

    def test_summary_candidate_role_conflict_hard_fails(self):
        summary = [{
            "task_key": "bbq_sauce",
            "state_id": "5",
            "window_start": "27",
            "window_end": "44",
            "candidate_role": "stable_post_lock_control",
            "denominator_clean": "True",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "False",
        }]
        candidates = [{
            "task_key": "bbq_sauce",
            "state_id": "5",
            "window_start": "27",
            "window_end": "44",
            "candidate_role": "far_too_early_control",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            _, _, conflicts_path, _ = self.run_builder(
                tmp,
                {"batch3c": summary},
                {"batch3c": candidates},
                expect_ok=False,
            )
            conflicts = read_csv(conflicts_path)
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["reason"], "summary_candidate_role_conflict")

    def test_batch3c_missing_join_role_is_manual_review_not_train(self):
        summary = [{
            "task_key": "salad_dressing",
            "state_id": "0",
            "window_start": "7",
            "window_end": "24",
            "denominator_clean": "True",
            "vis_OPEN_mean": "18",
            "vis_qpos_opening_delta_mean": "0.038",
            "vis_done_all_false": "False",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            labels_path, _, conflicts_path, _ = self.run_builder(tmp, {"batch3c": summary})
            label = read_csv(labels_path)[0]
            self.assertEqual(label["candidate_role"], "")
            self.assertEqual(label["label_status"], "manual_review")
            self.assertEqual(label["label_use"], "manual_review")
            self.assertEqual(label["label_vulnerability_ready"], "")
            self.assertEqual(read_csv(conflicts_path), [])


if __name__ == "__main__":
    unittest.main()
