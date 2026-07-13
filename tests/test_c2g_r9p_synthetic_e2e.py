"""Non-skippable synthetic R9P plan -> smoke -> audit -> train -> CAL -> stream test."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.stageb.calibrate_c2g_r9p_preview_thresholds import run_calibration, run_check_only
from scripts.stageb.run_c2g_r9p_streaming_replay import run_streaming_replay
from scripts.stageb.train_c2g_r9p_preview_detector import train_model
from tools.multisuite_detector.audit_c2g_r9p_materialization import audit_materialization
from tools.multisuite_detector.build_c2g_r9p_preview_plan import build_plan, GATE_PASS
from tools.multisuite_detector.materialize_c2g_r9p_ogs1500 import run_materialization
from tools.multisuite_detector.c2g_r8r_common import sha256_file


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_suite(root: Path, suite: str) -> Path:
    for task in range(10):
        for state in range(30):
            parent = f"{suite}/task_{task}/state_{state}/detector_train/episode_{state:03d}"
            ep = root / "episodes" / suite / parent
            ep.mkdir(parents=True)
            (ep / "derived_episode_metadata.json").write_text(json.dumps({
                "suite": suite, "task_index": task, "state_id": state,
                "parent_key": parent, "cohort": "DETECTOR_TRAIN", "split": "train",
                "task_language": f"{suite} task {task} pick and place",
            }))
            (ep / "source_binding.json").write_text(json.dumps({"parent_key": parent, "clean": True}))
            (ep / "rgb_frame_manifest.json").write_text(json.dumps({"frames": 12}))
            steps = []
            labels = []
            positive = state % 2 == 0
            for i in range(12):
                base = float(task * 10 + state * 0.1 + i * 0.01)
                steps.append({
                    "step": i,
                    "features_25d": [base + j * 0.001 for j in range(25)],
                    "clean_policy_intent_9d": [base + j * 0.002 for j in range(9)],
                })
                labels.append({
                    "step": i, "label_known_mask": True,
                    "y_attack_start_b": bool(positive and i == 2),
                    "y_burst_feasible": bool(positive and 2 <= i < 12),
                    "y_gripper_critical_window": bool(positive),
                    "y_release_safe": False,
                    "y_contact_or_grasp_stable": bool(positive),
                    "grounding_confidence": 0.9 if positive else 0.2,
                })
            (ep / "step_records_prefix.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in steps)
            )
            (ep / "teacher_v2_labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in labels)
            )
    report = root / "suite_report.json"
    report.write_text(json.dumps({"suite": suite, "status": "PASS", "episodes": 300}))
    return report


class SyntheticR9PEndToEndTests(unittest.TestCase):
    def test_plan_smoke_audit_train_cal_stream(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite_roots = {
                "libero_spatial": root / "spatial",
                "libero_object": root / "object",
                "libero_goal": root / "goal",
            }
            reports = {suite: _write_suite(path, suite) for suite, path in suite_roots.items()}
            r8z1 = root / "r8z1.json"
            amendment = root / "amendment.json"
            composite = root / "composite.json"
            ledger = root / "composite_ledger.jsonl"
            sums = root / "composite_SHA256SUMS"
            for path, payload in ((r8z1, {"status": "PASS"}), (amendment, {"status": "PASS"}),
                                  (composite, {"status": "PASS"}), (ledger, {"rows": 1}),
                                  (sums, {"files": 1})):
                path.write_text(json.dumps(payload))
            head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            plan_root = root / "plan"
            plan = build_plan(
                spatial_root=suite_roots["libero_spatial"], object_root=suite_roots["libero_object"],
                goal_root=suite_roots["libero_goal"], output_root=plan_root, git_commit=head,
                expected_spatial_report_sha=_sha(reports["libero_spatial"]),
                expected_object_report_sha=_sha(reports["libero_object"]),
                expected_goal_report_sha=_sha(reports["libero_goal"]),
                expected_r8z1_audit_sha=_sha(r8z1), r8z1_audit_report_path=str(r8z1),
                r8z1_amendment_sha=_sha(amendment), r8z1_amendment_path=str(amendment),
                r8z_composite_report_sha=_sha(composite), r8z_composite_report_path=str(composite),
                r8z_composite_ledger_sha=_sha(ledger), r8z_composite_ledger_path=str(ledger),
                r8z_composite_sums_sha=_sha(sums), r8z_composite_sums_path=str(sums),
            )
            self.assertEqual(plan["status"], GATE_PASS)

            smoke_root = root / "smoke"
            smoke = run_materialization(plan_root, smoke_root, smoke=True, suite_roots=suite_roots)
            self.assertEqual(smoke["status"], "PASS_C2G_R9P_MATERIALIZATION_SMOKE")
            smoke_audit = audit_materialization(plan_root, smoke_root, root / "smoke_audit", smoke=True)
            self.assertEqual(smoke_audit["status"], "PASS_C2G_R9P_MATERIALIZATION_AUDIT")

            full_root = root / "full"
            full = run_materialization(plan_root, full_root, smoke=False, suite_roots=suite_roots)
            self.assertEqual(full["status"], "PASS_C2G_R9P_TRAINONLY_MATERIALIZATION")
            full_audit = audit_materialization(plan_root, full_root, root / "full_audit", smoke=False)
            self.assertEqual(full_audit["status"], "PASS_C2G_R9P_MATERIALIZATION_AUDIT")

            training = train_model(
                materialization_root=full_root, output_root=root / "training",
                model_label="a", seed=42, epochs=1, batch_size=128, device_str="cpu",
            )
            checkpoint = root / "training" / "model_a_seed42" / "checkpoint.pt"
            self.assertTrue(checkpoint.is_file())
            cal = run_calibration(
                full_root, checkpoint, root / "calibration", device_str="cpu",
                grid={"tau_critical": [0.99], "tau_release": [0.99], "tau_ground": [0.99],
                      "persistence": [{"persistence_window": 1, "persistence_required": 1}]},
            )
            self.assertEqual(cal["status"], "PASS_C2G_R9P_CALIBRATION")
            config = root / "calibration" / "preview_detector_config.json"
            check = run_check_only(full_root, checkpoint, config, root / "check", device_str="cpu")
            self.assertEqual(check["check"]["check_consumption_count"], 1)
            stream = run_streaming_replay(
                full_root, checkpoint, root / "stream", detector_config_path=config,
                max_episodes=2, device_str="cpu",
            )
            self.assertTrue(stream["batch_stream_equivalence"])
            self.assertEqual(stream["multi_trigger_count"], 0)
            self.assertEqual(training["epochs_completed"], 1)


if __name__ == "__main__":
    unittest.main()
