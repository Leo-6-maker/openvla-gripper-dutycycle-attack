#!/usr/bin/env python3
"""Fail-closed audit and seal for the T1-D1 clean screening census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1_SCREENING_CLEAN_PROTOCOL_V1.json"
PARENT_REL = "reports/STAGE_X_X1R_T1D0R2_PARENT_SEED_INVARIANCE_V1.json"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema") != "STAGE_X_X1R_T1D1_SCREENING_CLEAN_PROTOCOL_V1":
        raise RuntimeError("D1_PROTOCOL_SCHEMA_INVALID")
    if protocol.get("status") != "FROZEN_FOR_SCREENING_CLEAN_EXECUTION":
        raise RuntimeError("D1_PROTOCOL_NOT_FROZEN")
    return protocol


def parents(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = REPO / str(protocol["parent_population"]["seed_report"])
    if sha256_file(path) != protocol["parent_population"]["seed_report_sha256"]:
        raise RuntimeError("PARENT_LEDGER_SHA_MISMATCH")
    report = load_json(path)
    if report.get("status") != "PASS_D0R1_INVARIANTS":
        raise RuntimeError("PARENT_LEDGER_NOT_PASS")
    rows = sorted(report["rows"], key=lambda row: int(row["ordinal"]))
    if len(rows) != 39 or len({row["canonical_parent_key"] for row in rows}) != 39:
        raise RuntimeError("PARENT_LEDGER_COUNT_OR_UNIQUENESS_FAIL")
    return rows


def student_artifacts(protocol: Mapping[str, Any]) -> dict[str, Path]:
    config = load_json(REPO / "configs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_V1.json")
    receipt = load_json(Path(str(config["historical_t1_receipt"]["path"])))
    sealed = receipt["sealed_detector"]
    cfg = protocol["student"]
    paths = {"checkpoint": Path(str(cfg["checkpoint"])), "normalization": Path(str(sealed["normalization"]["path"])), "thresholds": Path(str(sealed["thresholds"]["path"]))}
    expected = {"checkpoint": cfg["checkpoint_sha256"], "normalization": cfg["normalization_sha256"], "thresholds": cfg["thresholds_sha256"]}
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise RuntimeError(f"STUDENT_ARTIFACT_INVALID:{name}")
    return paths


def student_recompute(protocol: Mapping[str, Any], features: list[list[float]]) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if not features:
        return [], {"status": "ABSTAIN_EMPTY"}
    paths = student_artifacts(protocol)
    normalization = load_json(paths["normalization"])
    norm = normalization["episode_heldout"]["train"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    thresholds = load_json(paths["thresholds"])
    physical = float(thresholds["physical_criticality"]["threshold"])
    closing = float(thresholds["gripper_closing_state"]["threshold"])
    if (physical, closing) != (0.55, 0.8):
        raise RuntimeError("STUDENT_THRESHOLD_DRIFT")
    import torch

    sys.path.insert(0, str(REPO / "n5/phase3_student"))
    from n5_student_model import N5MultiHeadStudent

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    raw = np.asarray(features, dtype=np.float32)
    x = torch.from_numpy(((raw - mean) / std).astype(np.float32))[None, ...]
    mask = torch.ones((1, len(features)), dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, timestep_mask=mask)
    names = ("physical_criticality", "k10_feasibility", "safe_release", "instability", "gripper_closing_state")
    prediction = [{name: float(torch.sigmoid(logits[name][0, idx]).item()) for name in names} for idx in range(len(features))]
    return prediction, {"status": "PASS", "forward_calls": 1, "thresholds": {"physical_criticality": physical, "gripper_closing_state": closing}}


def recompute_features(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[bool], float]:
    from gripper_attack.v5_r3_features import materialize_fit670_features

    steps = [{"step": row["step"], "raw_action_7d": row["raw_action_7d"], "action_env_7d": row["action_env_7d"]} for row in rows]
    telemetry = [{"step": row["step"], "robot0_gripper_qpos": row["robot0_gripper_qpos"], "robot0_eef_pos": row["robot0_eef_pos"]} for row in rows]
    materialized = materialize_fit670_features({"steps": steps, "telemetry": telemetry})
    max_diff = 0.0
    features: list[list[float]] = []
    candidates: list[bool] = []
    for row, expected in zip(rows, materialized):
        if not row.get("feature_valid"):
            raise RuntimeError(f"FEATURE_INVALID_IN_RECEIPT:{row.get('step')}")
        actual = np.asarray(row["features_25d"], dtype=np.float32)
        wanted = np.asarray(expected["features_25d"], dtype=np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(actual - wanted), initial=0.0)))
        features.append(wanted.tolist())
        candidates.append(bool(expected["candidate_close"]))
    return features, candidates, max_diff


def check_parent(path: Path, parent: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    receipt_path = path / "parent_receipt.json"
    telemetry_path = path / "step_telemetry.jsonl"
    if not receipt_path.is_file() or not telemetry_path.is_file():
        return {"ordinal": int(parent["ordinal"]), "canonical_parent_key": parent["canonical_parent_key"], "status": "MISSING", "errors": ["RECEIPT_OR_TELEMETRY_MISSING"]}
    receipt = load_json(receipt_path)
    lines = [line for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    key = str(parent["canonical_parent_key"])
    suite = key.split("/", 1)[0]
    if receipt.get("status") != "PASS_SCREENING_CLEAN_EPISODE":
        errors.append(f"RECEIPT_STATUS:{receipt.get('status')}")
    for field, expected in (("canonical_parent_key", key), ("ordinal", int(parent["ordinal"])), ("condition", "SCREENING_CLEAN")):
        if receipt.get(field) != expected:
            errors.append(f"RECEIPT_BINDING:{field}")
    if receipt.get("screening_is_not_clean_eval") is not True or receipt.get("manual_clean_contact_review") != "REQUIRED":
        errors.append("SCREENING_OR_MANUAL_REVIEW_FLAG_INVALID")
    if len(rows) != int(receipt.get("policy_steps_executed", -1)) or [row.get("step") for row in rows] != list(range(len(rows))):
        errors.append("TELEMETRY_STEP_CLOSURE_FAIL")
    for row in rows:
        if row.get("condition") != "SCREENING_CLEAN" or row.get("canonical_parent_key") != key:
            errors.append(f"TELEMETRY_IDENTITY_FAIL:{row.get('step')}")
        if any(name in row for name in ("adv_action", "adversarial_image", "pgd", "v_phys", "attack_outcome", "intervention")):
            errors.append(f"FORBIDDEN_FIELD:{row.get('step')}")
        raw = np.asarray(row.get("raw_action_7d", []), dtype=np.float64)
        env = np.asarray(row.get("action_env_7d", []), dtype=np.float64)
        if raw.shape != (7,) or env.shape != (7,) or not np.isfinite(raw).all() or not np.isfinite(env).all():
            errors.append(f"ACTION_SCHEMA_FAIL:{row.get('step')}")
        if raw.shape == (7,) and env.shape == (7,):
            expected_env = 1.0 if raw[6] < 0.5 else (-1.0 if raw[6] > 0.5 else 0.0)
            if abs(float(env[6]) - expected_env) > 1e-6 or not np.allclose(raw[:6], env[:6], atol=1e-6, rtol=0):
                errors.append(f"ACTION_SEMANTICS_FAIL:{row.get('step')}")
    video = receipt.get("video", {})
    video_path = Path(str(video.get("path", "")))
    durable_root = Path(str(protocol["durable_storage"]["root"])).resolve()
    try:
        video_path.resolve().relative_to(durable_root)
        video_under_root = True
    except ValueError:
        video_under_root = False
    if not video_path.is_file() or not video_under_root:
        errors.append("VIDEO_PATH_NOT_DURABLE")
    else:
        actual_sha = sha256_file(video_path)
        if actual_sha != video.get("sha256") or video_path.stat().st_size != int(video.get("bytes", -1)):
            errors.append("VIDEO_HASH_OR_SIZE_MISMATCH")
        if video_path.stat().st_size > int(protocol["durable_storage"]["per_episode_video_budget_bytes"]):
            errors.append("VIDEO_BUDGET_EXCEEDED")
    counters = receipt.get("counters", {})
    mount_gpu = receipt.get("gpu", {}).get("mount_gate", {})
    if int(mount_gpu.get("free_memory_mib", 0)) <= 20480:
        errors.append("GPU_MOUNT_GATE_NOT_STRICTLY_ABOVE_20480_MIB")
    for name in ("pgd_calls", "attack_backward_calls", "adversarial_images", "physical_interventions", "vphys_reads", "attack_outcome_reads", "eval160_reads", "protected_reads", "attacked_env_steps"):
        if int(counters.get(name, -1)) != 0:
            errors.append(f"PROTECTED_COUNTER_NONZERO:{name}")
    try:
        if not all(bool(row.get("feature_valid")) for row in rows):
            feature_diff = None
            student_status = "ABSTAIN_INVALID_FEATURE_STREAM"
            if receipt.get("first_emit_step") is not None:
                errors.append("INVALID_FEATURE_STREAM_EMITTED")
        else:
            features, candidates, feature_diff = recompute_features(rows)
            predictions, student_audit = student_recompute(protocol, features)
            predicted_schedule = schedule(predictions, candidates, HORIZONS[suite], 0.55, 0.8)
            if predicted_schedule["first_emit_step"] != receipt.get("first_emit_step"):
                errors.append("STUDENT_FIRST_EMIT_RECOMPUTE_MISMATCH")
            for row, prediction in zip(rows, predictions):
                recorded = row.get("student_probabilities", {})
                for name, value in prediction.items():
                    if abs(float(recorded.get(name, float("nan"))) - value) > 1e-6:
                        errors.append(f"STUDENT_PROBABILITY_MISMATCH:{row.get('step')}:{name}")
            student_status = student_audit["status"]
    except Exception as exc:
        feature_diff = None
        student_status = f"AUDIT_ERROR:{type(exc).__name__}:{exc}"
        errors.append(student_status)
    return {"ordinal": int(parent["ordinal"]), "canonical_parent_key": key, "suite": suite, "status": "PASS" if not errors else "HOLD", "errors": errors, "receipt_path": str(receipt_path), "telemetry_path": str(telemetry_path), "policy_steps": len(rows), "clean_success": bool(receipt.get("clean_success")), "first_emit_step": receipt.get("first_emit_step"), "video_path": str(video_path), "feature_recompute_max_abs_diff": feature_diff, "student_audit_status": student_status, "runtime_source_pre_evidence": receipt.get("runtime_source_pre_evidence", {}), "counters": counters}


def schedule(predictions: list[dict[str, float]], candidates: list[bool], horizon: int, physical_threshold: float, closing_threshold: float) -> dict[str, Any]:
    emitted = False
    first = None
    for step, (prediction, candidate) in enumerate(zip(predictions, candidates)):
        if not emitted and candidate and step + 5 + 10 <= horizon and prediction["physical_criticality"] >= physical_threshold and prediction["gripper_closing_state"] >= closing_threshold:
            emitted = True
            first = step
    return {"first_emit_step": first}


def seal_root(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    sums = "\n".join(rows) + "\n"
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    seal = {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_ROOT_SEAL_V1", "status": summary["status"], "summary_sha256": sha256_bytes(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default).encode("utf-8")), "sha256sums_sha256": sums_sha, "file_count": len(rows), "screening_clean_only": True, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"}
    write_json(root / "ROOT_SEAL.json", seal)
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-report", type=Path, default=REPO / "reports/STAGE_X_X1R_T1D1_SCREENING_CLEAN_AUDIT_V1.json")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    if git("status", "--porcelain"):
        raise SystemExit("WORKTREE_NOT_CLEAN")
    rows = parents(protocol)
    root = Path(str(protocol["durable_storage"]["root"]))
    preflight_path = root / "preflight" / "D1_DURABLE_STORAGE_PREFLIGHT.json"
    if not preflight_path.is_file() or load_json(preflight_path).get("status") != "PASS_DURABLE_STORAGE":
        raise SystemExit("DURABLE_PREFLIGHT_MISSING_OR_NOT_PASS")
    results: list[dict[str, Any]] = []
    by_ordinal = {int(row["ordinal"]): row for row in rows}
    for ordinal, parent in sorted(by_ordinal.items()):
        parent_dir = root / "parents" / f"{ordinal:03d}_{safe_name(str(parent['canonical_parent_key']))}"
        attempts = sorted(parent_dir.glob("attempt_*/parent_receipt.json")) if parent_dir.is_dir() else []
        if len(attempts) != 1:
            results.append({"ordinal": ordinal, "canonical_parent_key": parent["canonical_parent_key"], "status": "HOLD", "errors": [f"EXPECTED_ONE_ATTEMPT_GOT_{len(attempts)}"]})
            continue
        results.append(check_parent(attempts[0].parent, parent, protocol))
    counts = {suite: sum(result.get("suite") == suite for result in results) for suite in SUITES}
    errors = [error for result in results for error in result.get("errors", [])]
    pass_count = sum(result.get("status") == "PASS" for result in results)
    runtime_commits = sorted({result.get("runtime_source_pre_evidence", {}).get("commit") for result in results if result.get("runtime_source_pre_evidence")})
    status = "PROTOCOL_ATTACK_ELIGIBLE_PRE_MANUAL_REVIEW" if len(results) == 39 and pass_count == 39 and not errors and len(runtime_commits) == 1 else "HOLD_SCREENING_CLEAN_AUDIT"
    summary = {"schema": "STAGE_X_X1R_T1D1_SCREENING_CLEAN_AUDIT_V1", "status": status, "source": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")}, "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol.resolve())}, "parent_population": {"expected": 39, "observed": len(results), "pass": pass_count, "suite_counts": counts, "missing_cell": "libero_goal/task_01", "replacement": False}, "results": results, "errors": errors, "runtime_source_commits": runtime_commits, "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0}, "manual_review": {"clean_contact_videos": "REQUIRED", "status": "PENDING_OWNER_GPT_REVIEW", "clean_success_is_not_physical_contact_verdict": True}, "next_gate": "PROTOCOL_ATTACK_ELIGIBLE_PRE_MANUAL_REVIEW" if status == "PROTOCOL_ATTACK_ELIGIBLE_PRE_MANUAL_REVIEW" else "HOLD_SCREENING_CLEAN_AUDIT"}
    write_json(args.output_report, summary)
    write_json(root / "D1_CENSUS_AGGREGATE.json", summary)
    seal = seal_root(root, summary)
    summary["durable_root_seal"] = {"path": str(root / "ROOT_SEAL.json"), "sha256": sha256_file(root / "ROOT_SEAL.json"), "sha256sums_sha256": seal["sha256sums_sha256"]}
    write_json(args.output_report, summary)
    write_json(root / "D1_CENSUS_AGGREGATE.json", summary)
    print(json.dumps({"status": status, "observed": len(results), "pass": pass_count, "errors": len(errors), "root": str(root)}, sort_keys=True))
    return 0 if status == "PROTOCOL_ATTACK_ELIGIBLE_PRE_MANUAL_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
