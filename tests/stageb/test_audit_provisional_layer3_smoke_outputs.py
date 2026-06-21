import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb import audit_provisional_layer3_smoke_outputs as audit  # noqa: E402


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_job(tmp_path: Path, condition: str = "VIS", parent: str = "libero_goal|1|4|0|CLEAN"):
    out_dir = tmp_path / f"job_{condition.lower()}"
    detector = tmp_path / "model.pt"
    detector.write_bytes(b"detector")
    detector_sha = audit.sha256_file(detector)
    row = {
        "job_id": f"job_{condition.lower()}",
        "parent_key": parent,
        "suite": "libero_goal",
        "task_idx": "1",
        "state_id": "4",
        "teacher_anchor": "42",
        "condition": condition,
        "attack_seed": "81",
        "model_path": "/models/goal",
        "unnorm_key": "libero_goal",
        "detector_path": str(detector),
        "expected_detector_sha256": detector_sha,
        "render_gpu": "5",
        "output_dir": str(out_dir),
    }
    return row


def write_episode(row: dict, *, emit: int = 5, n_steps: int = 12, attack: bool = True, attack_start: int | None = None):
    out_dir = Path(row["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = row["condition"]
    requested = "VIS" if condition == "VIS" else condition
    attack_start = emit if attack_start is None else attack_start
    summary = {
        "suite": row["suite"],
        "task_idx": int(row["task_idx"]),
        "state_id": int(row["state_id"]),
        "teacher_anchor": int(row["teacher_anchor"]),
        "condition": "TRUE_T10" if condition == "VIS" else condition,
        "requested_condition": requested,
        "unnorm_key": row["unnorm_key"],
        "checkpoint_sha256": row["expected_detector_sha256"],
        "dataset_sha256": "dataset-ok",
        "n_steps": n_steps,
        "task_success": True,
        "mlp_triggered": emit >= 0,
        "mlp_emit_step": emit,
        "attack_frames": 0 if not attack else min(10, max(0, n_steps - attack_start)),
        "invalid_feature_steps": 0,
        "privileged_detector_input_used": False,
        "manual_anchor_used": False,
        "arm_action_preservation_mode": "execute_clean_arm_with_attacked_gripper",
    }
    (out_dir / "episode_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    count = 0
    for step in range(n_steps):
        attack_this = attack and step >= attack_start and count < 10
        if attack_this:
            count += 1
        rows.append(
            {
                "step": step,
                "mlp_emit": emit if step >= emit >= 0 else -1,
                "attack_count": count,
                "attack_this": str(attack_this),
                "adv_token": "31744" if attack_this else "",
                "adv_arm": "6" if attack_this else "",
            }
        )
    write_csv(out_dir / "step_telemetry.csv", rows)
    (out_dir / "rollout_raw.mp4").write_bytes(b"raw")
    (out_dir / "rollout_overlay.mp4").write_bytes(b"overlay")


def ledger_for(row: dict):
    command = (
        f"/py scripts/stageb/run_v2_vis_sc5_mlp_bridge.py --condition {row['condition']} "
        f"--suite {row['suite']} --model_path {row['model_path']} --unnorm_key {row['unnorm_key']} "
        f"--task_idx {row['task_idx']} --state_id {row['state_id']} --anchor {row['teacher_anchor']} "
        f"--seed_id {row['attack_seed']} --output_dir {row['output_dir']} --render_gpu {row['render_gpu']} "
        f"--mlp_path {row['detector_path']} --write_video"
    )
    return {
        "job_id": row["job_id"],
        "status": "COMPLETE",
        "returncode": "0",
        "output_dir": row["output_dir"],
        "command": command,
        "log_path": str(Path(row["output_dir"]) / "worker.log"),
        "duration_sec": "1",
    }


def patch_video(monkeypatch, frames: int = 12):
    monkeypatch.setattr(audit, "video_frame_count", lambda path: (True, frames, ""))
    monkeypatch.setattr(audit, "video_full_decode_ok", lambda path: (True, ""))


def test_good_job_passes_runtime_contract(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "VIS")
    write_episode(row)
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["status"] == "COMPLETE"
    assert all(int(result[field]) == 0 for field in audit.FAIL_FIELDS)
    assert result["attack_this_rows"] == 7


def test_identity_mismatch_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "VIS")
    write_episode(row)
    summary_path = Path(row["output_dir"]) / "episode_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["task_idx"] = 99
    summary_path.write_text(json.dumps(summary))
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["manifest_identity_mismatch"] == 1


def test_clean_or_no_emit_with_attack_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "CLEAN")
    write_episode(row, emit=-1, attack=True, attack_start=3)
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["attack_timing_contract_failure"] == 1
    assert result["attack_count_contract_failure"] == 1


def test_attack_before_emit_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "RAND")
    write_episode(row, emit=5, attack=True, attack_start=4)
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["attack_timing_contract_failure"] == 1


def test_video_frame_mismatch_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch, frames=11)
    row = make_job(tmp_path, "VIS")
    write_episode(row, n_steps=12)
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["video_frame_mismatch"] == 1


def test_command_ledger_mismatch_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "VIS")
    write_episode(row)
    bad = ledger_for(row)
    bad["command"] = bad["command"].replace("--seed_id 81", "--seed_id 82")
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: bad})
    assert result["command_ledger_mismatch"] == 1


def test_checkpoint_file_sha_mismatch_fails(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "VIS")
    write_episode(row)
    Path(row["detector_path"]).write_bytes(b"changed")
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    assert result["checkpoint_sha_mismatch"] == 1


def test_parent_condition_set_and_emit_mismatch_fail(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    rows = []
    for condition in ["CLEAN", "VIS", "RAND"]:
        row = make_job(tmp_path, condition)
        write_episode(row, emit=5, attack=(condition != "CLEAN"))
        result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
        rows.append(result)
    rows[1]["mlp_emit_step"] = "6"
    audit.add_group_contracts(rows)
    assert all(r["parent_condition_set_failure"] == 1 for r in rows)
    assert all(r["matched_emit_mismatch"] == 1 for r in rows)


def test_recursive_sha_manifest_includes_episode_and_ledger(monkeypatch, tmp_path):
    patch_video(monkeypatch)
    row = make_job(tmp_path, "VIS")
    write_episode(row)
    result = audit.audit_job(row, "dataset-ok", {row["job_id"]: ledger_for(row)})
    ledger = tmp_path / "ledger.csv"
    write_csv(ledger, [ledger_for(row)])
    out = tmp_path / "seal.csv"
    summary = audit.build_recursive_sha_manifest([row], [result], ledger, out, [])
    assert summary["sealed_file_count"] >= 4
    assert out.exists()
    assert summary["recursive_sha_manifest_sha256"] == audit.sha256_file(out)
