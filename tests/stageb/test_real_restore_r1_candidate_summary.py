import csv
import json
import subprocess
import sys


def test_r1_candidate_summary_buckets_zero_emit(tmp_path):
    run_root = tmp_path / "run_root"
    run_dir = run_root / "run"
    run_dir.mkdir(parents=True)
    with (run_dir / "candidate_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "protocol_id",
                "suite",
                "task_idx",
                "state_id",
                "eval_seed",
                "selection_hash",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "protocol_id": "R1",
                "suite": "libero_goal",
                "task_idx": "1",
                "state_id": "20",
                "eval_seed": "0",
                "selection_hash": "a" * 64,
                "status": "INELIGIBLE",
                "reason": "ExactRestoreError:candidate did not produce eligible natural Student emit",
            }
        )
    (run_dir / "single_parent_restore_qualification_summary.json").write_text(
        json.dumps({"result": "NO_ELIGIBLE_GOAL_RESTORE_PARENT"}),
        encoding="utf-8",
    )
    (run_dir / "openvla_model_binding_receipt.json").write_text(
        json.dumps({"suite": "libero_goal", "unnorm_key": "libero_goal"}),
        encoding="utf-8",
    )
    out = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            "scripts/stageb/summarize_real_restore_r1_candidates.py",
            "--run-root",
            str(run_root),
            "--output-dir",
            str(out),
        ],
        check=True,
    )

    summary = json.loads((out / "real_restore_r1_candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"INELIGIBLE": 1}
    assert summary["reason_bucket_counts"] == {"no_natural_student_emit": 1}
    assert summary["run_summary"]["result"] == "NO_ELIGIBLE_GOAL_RESTORE_PARENT"
