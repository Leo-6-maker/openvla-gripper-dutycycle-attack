from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.repair_l3_vis_handoff_contract import (
    EXPECTED_FRAMES,
    V4_CONDITIONS,
    audit_frame_rows,
    audit_parent_identity,
    emit_configs,
    summarize_gate,
    validate_job_plan,
    validate_selected_frame_set,
)


def _selected_rows(tmp_path: Path) -> list[dict[str, str]]:
    rows = []
    for exp in EXPECTED_FRAMES:
        raw = tmp_path / f"{exp.parent_id}_{exp.step}.npy"
        proc = tmp_path / f"{exp.parent_id}_{exp.step}.pt"
        raw.write_bytes(f"raw-{exp.parent_id}-{exp.step}".encode())
        proc.write_bytes(f"proc-{exp.parent_id}-{exp.step}".encode())
        import hashlib

        rows.append(
            {
                "parent_id": exp.parent_id,
                "task": exp.task,
                "state_id": str(exp.state_id),
                "timing_class": exp.timing_class,
                "clean_success": "1",
                "frame_step": str(exp.step),
                "frame_role": exp.role,
                "inside_teacher_window": str(exp.inside_teacher_window),
                "d5_emit_relation": exp.d5_emit_relation,
                "raw_frame_path": str(raw),
                "raw_frame_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "processor_tensor_path": str(proc),
                "processor_tensor_sha256": hashlib.sha256(proc.read_bytes()).hexdigest(),
                "png_path": "",
                "png_sha256": "",
                "prompt_instruction": "pick up the object",
                "unnorm_key": "libero_object",
                "target_token": "31744",
                "attack_lambda": "2.0",
                "attack_seeds": "81,82",
                "gpu": "1,5",
            }
        )
    return rows


def _job_rows() -> list[dict[str, str]]:
    rows = []
    for exp in EXPECTED_FRAMES:
        for seed in ("81", "82"):
            rows.append(
                {
                    "parent_id": exp.parent_id,
                    "frame_step": str(exp.step),
                    "frame_role": exp.role,
                    "seed": seed,
                    "condition": "TRUE_PGD_TRAJECTORY21_SELECTIVE",
                    "target_token": "31744",
                    "lambda": "2.0",
                    "gpu": "1,5",
                }
            )
    return rows


def test_selected_frame_set_requires_exact_ten_frames(tmp_path):
    rows = _selected_rows(tmp_path)
    ok, problems = validate_selected_frame_set(rows)
    assert ok, problems

    rows = rows[:-1]
    ok, problems = validate_selected_frame_set(rows)
    assert not ok
    assert any("missing frame salad_dressing_s11:128" in p for p in problems)


def test_selected_frame_set_rejects_wrong_gpu_or_seed(tmp_path):
    rows = _selected_rows(tmp_path)
    rows[0]["gpu"] = "2,6"
    rows[1]["attack_seeds"] = "81"
    ok, problems = validate_selected_frame_set(rows)
    assert not ok
    assert any("gpu='2,6'" in p for p in problems)
    assert any("attack_seeds='81'" in p for p in problems)


def test_job_plan_is_v4_trajectory21_only():
    rows = _job_rows()
    ok, problems = validate_job_plan(rows)
    assert ok, problems

    rows[0]["condition"] = "TRUE_PGD_FINAL"
    ok, problems = validate_job_plan(rows)
    assert not ok
    assert any("TRUE_PGD_TRAJECTORY21_SELECTIVE" in p for p in problems)


def test_frame_audit_fails_closed_without_clean_generation_package(tmp_path):
    rows = _selected_rows(tmp_path)
    package_root = tmp_path / "packages"
    audits = audit_frame_rows(rows, package_root=package_root, require_files=True)
    assert len(audits) == len(EXPECTED_FRAMES)
    assert all(row["frame_package_status"] == "FAIL" for row in audits)
    assert all("clean_generation_missing" in row["failures"] or "frame_package_missing" in row["failures"] for row in audits)


def test_primary_frame_requires_clean_close_31872(tmp_path):
    rows = _selected_rows(tmp_path)
    package_root = tmp_path / "packages"
    exp = EXPECTED_FRAMES[0]
    pkg = package_root / exp.parent_id / f"step_{exp.step:04d}"
    pkg.mkdir(parents=True)
    canonical = pkg / "processor_inputs_attack.pt"
    canonical.write_bytes(b"canonical")
    (pkg / "frame_package_manifest.json").write_text(
        json.dumps({"canonical_processor_tensor_sha256": __import__("hashlib").sha256(b"canonical").hexdigest()}),
        encoding="utf-8",
    )
    (pkg / "clean_generation.json").write_text(
        json.dumps(
            {
                "clean_exact_7_tokens": [1, 2, 3, 4, 5, 6, 31744],
                "clean_gripper_token": 31744,
                "prompt_token_ids_sha256": "abc",
                "model_fingerprint": {"model": "mock"},
            }
        ),
        encoding="utf-8",
    )
    audits = audit_frame_rows(rows[:1], package_root=package_root, require_files=True)
    assert audits[0]["frame_package_status"] == "FAIL"
    assert "primary_clean_gripper_not_31872:31744" in audits[0]["failures"]


def test_identity_audit_requires_capture_and_timing_action_match(tmp_path):
    rows = _selected_rows(tmp_path)
    parent_rows = [r for r in rows if r["parent_id"] == "butter_s11"]
    timing_dir = tmp_path / "timing"
    timing_dir.mkdir()
    trace = timing_dir / "step_trace.csv"
    trace.write_text("step\n0\n", encoding="utf-8")
    with (timing_dir / "action_identity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "action_hash_pre", "env_action_hash", "obs_hash"])
        writer.writeheader()
        for step in (58, 60, 68):
            writer.writerow({"step": step, "action_hash_pre": f"a{step}", "env_action_hash": f"e{step}", "obs_hash": f"o{step}"})
    handoff = [{"task": "butter", "state_id": "11", "trace_path": str(trace)}]
    frame_manifest = tmp_path / "frame_manifest.json"
    frame_manifest.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task": "butter",
                        "state_id": "11",
                        "action_identity": [
                            {"step": step, "action_hash": f"a{step}", "env_action_hash": f"e{step}"} for step in (58, 60, 68)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audits = audit_parent_identity(handoff, parent_rows, frame_manifest_path=frame_manifest, require_files=True)
    butter = [r for r in audits if r["parent_id"] == "butter_s11"][0]
    assert butter["exact_bound_status"] == "EXACT_BOUND"
    assert butter["identity_compared_steps"] == 3

    data = json.loads(frame_manifest.read_text())
    data["results"][0]["action_identity"][1]["env_action_hash"] = "bad"
    frame_manifest.write_text(json.dumps(data), encoding="utf-8")
    audits = audit_parent_identity(handoff, parent_rows, frame_manifest_path=frame_manifest, require_files=True)
    butter = [r for r in audits if r["parent_id"] == "butter_s11"][0]
    assert butter["exact_bound_status"] == "NOT_EXACT_BOUND"
    assert "env_action_hash_mismatch" in butter["failures"]


def test_emitted_configs_are_per_frame_and_v4(tmp_path):
    rows = _selected_rows(tmp_path)
    manifest = emit_configs(rows, tmp_path / "configs")
    assert len(manifest) == len(EXPECTED_FRAMES)
    for row in manifest:
        cfg = json.loads(Path(row["config_path"]).read_text(encoding="utf-8"))
        assert cfg["conditions"] == list(V4_CONDITIONS)
        assert cfg["attack_optimizer"]["selection_rule"] == "hard_feasible_official_decode_v4"
        assert cfg["controls"]["rand21_count"] == 21
        assert cfg["input"]["task"] != "tomato_sauce" or "tomato_sauce" in cfg["parent_id"]


def test_gate_summary_blocks_when_any_h0_hard_gate_missing(tmp_path):
    rows = _selected_rows(tmp_path)
    frame_audits = audit_frame_rows(rows, package_root=tmp_path / "missing", require_files=True)
    parent_audits = [
        {"parent_id": "butter_s11", "exact_bound_status": "EXACT_BOUND"},
        {"parent_id": "tomato_sauce_s23", "exact_bound_status": "EXACT_BOUND"},
        {"parent_id": "salad_dressing_s11", "exact_bound_status": "EXACT_BOUND"},
    ]
    summary = summarize_gate(
        frame_audits,
        parent_audits,
        full_inventory_count=71,
        selected_frame_set_ok=True,
        job_plan_ok=True,
        selected_frame_set_failures=[],
        job_plan_failures=[],
    )
    assert summary["status"] == "BLOCKED"
    assert summary["gpu_authorized_for_h1"] is False
    assert any("selected_frame_packages_pass_0_of_10" == f for f in summary["failures"])


def test_gate_summary_blocks_when_full_inventory_count_is_not_71():
    summary = summarize_gate(
        [{"frame_package_status": "PASS", "frame_denominator": "PRIMARY", "clean_gripper_token": "31872"}] * 6
        + [{"frame_package_status": "PASS", "frame_denominator": "DIAGNOSTIC", "clean_gripper_token": ""}] * 4,
        [
            {"parent_id": "butter_s11", "exact_bound_status": "EXACT_BOUND"},
            {"parent_id": "tomato_sauce_s23", "exact_bound_status": "EXACT_BOUND"},
            {"parent_id": "salad_dressing_s11", "exact_bound_status": "EXACT_BOUND"},
        ],
        full_inventory_count=65,
        selected_frame_set_ok=True,
        job_plan_ok=True,
        selected_frame_set_failures=[],
        job_plan_failures=[],
    )
    assert summary["status"] == "BLOCKED"
    assert "full_inventory_65_of_71" in summary["failures"]
