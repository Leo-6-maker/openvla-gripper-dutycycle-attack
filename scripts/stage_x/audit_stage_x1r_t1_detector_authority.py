"""T1-B/T1-C clean-only detector authority and shadow audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ENV = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ACTIVE_HEADS = ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
FEATURE_NAMES = (
    "gripper_command", "gripper_qpos", "gripper_opening_proxy", "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz", "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count", "close_onset",
    "time_since_close", "eef_speed", "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
)


def sha256(path: Path, *, normalize_text: bool = False) -> str:
    data = path.read_bytes()
    if normalize_text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_receipt() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    head = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    status = run("status", "--porcelain")
    return {"head": head, "tree": tree, "status_porcelain": status}


def choose_score_reports(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted(root.glob("SCORE_PATH_*_V2.json")):
        report = load_json(path)
        suite = report.get("suite")
        if suite not in SUITES or "_GPU" in path.name:
            continue
        selected.setdefault(suite, path)
    return selected


def clean_shadow_rows(replay_path: Path) -> tuple[list[dict[str, Any]], list[bool]]:
    replay = load_json(replay_path)
    if replay.get("outcomes_read") is not False or replay.get("v_phys_read") is not False:
        raise ValueError(f"outcome access is not closed: {replay_path}")
    if replay.get("intervention_executed") is not False:
        raise ValueError(f"intervention marker is not clean: {replay_path}")
    counters = replay.get("protected_counters", {})
    if any(int(v) != 0 for v in counters.values()):
        raise ValueError(f"protected counter is nonzero: {replay_path}")

    rows = replay.get("replay_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"replay rows missing: {replay_path}")
    steps = []
    telemetry = []
    for expected, row in enumerate(rows):
        if row.get("step") != expected:
            raise ValueError(f"step sequence mismatch: {replay_path}:{expected}")
        steps.append({
            "step": expected,
            "raw_action_7d": row["raw_action_7d"],
            "action_env_7d": row["env_action_7d"],
        })
        telemetry.append({
            "step": expected,
            "robot0_eef_pos": row["robot0_eef_pos"],
            "robot0_gripper_qpos": row["robot0_gripper_qpos"],
        })

    from gripper_attack.v5_r3_features import materialize_fit670_features

    materialized = materialize_fit670_features({
        "episode_id": replay["canonical_parent_key"],
        "steps": steps,
        "telemetry": telemetry,
    })
    features = [np.asarray(row["features_25d"], dtype=np.float32) for row in materialized]
    candidates = [bool(row["candidate_close"]) for row in materialized]
    if any(x.shape != (25,) or not np.isfinite(x).all() for x in features):
        raise ValueError(f"invalid feature row: {replay_path}")
    return [{"feature": x, "candidate_close": c} for x, c in zip(features, candidates)], candidates


def student_predictions(model: torch.nn.Module, features: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> list[dict[str, float]]:
    # The bound N5 encoder trims every convolution to its left-causal prefix;
    # one full pass is therefore the exact online result for every timestep.
    raw = np.stack([row["feature"] for row in features], axis=0)
    x = torch.from_numpy(((raw - mean) / std).astype(np.float32))[None, ...]
    mask = torch.ones((1, len(features)), dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, timestep_mask=mask)
    return [
        {
            "physical_criticality": float(torch.sigmoid(logits["physical_criticality"][0, i]).item()),
            "gripper_closing_state": float(torch.sigmoid(logits["gripper_closing_state"][0, i]).item()),
        }
        for i in range(len(features))
    ]


def student_prediction_at_prefix(model: torch.nn.Module, features: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, end: int) -> dict[str, float]:
    raw = np.stack([row["feature"] for row in features[:end]], axis=0)
    x = torch.from_numpy(((raw - mean) / std).astype(np.float32))[None, ...]
    mask = torch.ones((1, end), dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, timestep_mask=mask)
    return {
        "physical_criticality": float(torch.sigmoid(logits["physical_criticality"][0, -1]).item()),
        "gripper_closing_state": float(torch.sigmoid(logits["gripper_closing_state"][0, -1]).item()),
    }


def schedule(predictions: list[dict[str, float]], candidates: list[bool], *, t5: int, h_phys: int, physical_threshold: float, closing_threshold: float) -> dict[str, Any]:
    length = len(predictions)
    traces = []
    emitted = False
    emit_step = None
    for step, (prediction, candidate) in enumerate(zip(predictions, candidates)):
        legal_horizon = step + t5 + h_phys <= length
        emit = bool(
            not emitted
            and candidate
            and prediction["physical_criticality"] >= physical_threshold
            and prediction["gripper_closing_state"] >= closing_threshold
            and legal_horizon
        )
        if emit:
            emitted = True
            emit_step = step
        traces.append({
            "step": step,
            "candidate_close": candidate,
            "physical_criticality": prediction["physical_criticality"],
            "gripper_closing_state": prediction["gripper_closing_state"],
            "legal_horizon": legal_horizon,
            "emitted_this_step": emit,
        })
    return {"first_emit_step": emit_step, "emitted_count": int(sum(t["emitted_this_step"] for t in traces)), "traces": traces}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_PRIMARY_CLEAN_STUDENT_VALIDATION_FIREWALL_20260813T170000Z/checkpoint.pt"))
    parser.add_argument("--provenance", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CENSOR_AWARE_STUDENT_VPHYS_HELDOUT_F696F582_20260816T031500Z/PROVENANCE.json"))
    parser.add_argument("--final-decision", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_FINAL_DETECTOR_DECISION_F696F582_20260816T034000Z/FINAL_DETECTOR_DECISION.json"))
    parser.add_argument("--scheduler-freeze", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_STUDENT_TIME_PHYSICAL_MATRIX_PROTOCOL_F696F582_20260816T040000Z/SCHEDULER_FREEZE.json"))
    parser.add_argument("--replay-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CLEAN_REPLAY_STUDENT_INPUTS_F696F582_20260816T021500Z"))
    parser.add_argument("--score-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_pre_pgd_20260818_v2"))
    args = parser.parse_args()

    errors: list[str] = []
    provenance = load_json(args.provenance)
    final_decision = load_json(args.final_decision)
    scheduler_freeze = load_json(args.scheduler_freeze)
    current_git = git_receipt()

    thresholds_path = Path(provenance["thresholds_path"])
    normalization_path = Path(provenance["normalization_path"])
    feature_source = ROOT / "src/gripper_attack/v5_r3_features.py"
    adapter_source = ROOT / "src/gripper_attack/d8_streaming_features_v3.py"
    model_source = ROOT / "n5/phase3_student/n5_student_model.py"
    stale_n4 = ROOT / "scripts/fec/n4_detector_adapter_v4.py"

    for path in (args.checkpoint, thresholds_path, normalization_path, feature_source, adapter_source, model_source, stale_n4):
        if not path.is_file():
            errors.append(f"MISSING:{path}")
    if current_git["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN")
    if provenance.get("source_commit") != final_decision.get("source_commit"):
        errors.append("FINAL_SOURCE_COMMIT_MISMATCH")
    if provenance.get("source_tree") != final_decision.get("source_tree"):
        errors.append("FINAL_SOURCE_TREE_MISMATCH")
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", provenance["source_commit"], current_git["head"]], cwd=ROOT)
    except (subprocess.CalledProcessError, KeyError):
        errors.append("SEALED_DETECTOR_SOURCE_NOT_IN_CURRENT_ANCESTRY")

    checkpoint_sha = sha256(args.checkpoint)
    normalization_sha = sha256(normalization_path)
    thresholds_sha = sha256(thresholds_path)
    if checkpoint_sha != provenance.get("checkpoint_sha256"):
        errors.append("CHECKPOINT_SHA_MISMATCH")
    if normalization_sha != provenance.get("normalization_sha256"):
        errors.append("NORMALIZATION_SHA_MISMATCH")
    if thresholds_sha != provenance.get("thresholds_sha256"):
        errors.append("THRESHOLDS_SHA_MISMATCH")
    if sha256(adapter_source, normalize_text=True) != provenance.get("d8_adapter_sha256"):
        errors.append("D8_ADAPTER_SHA_MISMATCH")
    if sha256(feature_source, normalize_text=True) != provenance.get("feature_source_sha256"):
        errors.append("FEATURE_SOURCE_SHA_MISMATCH")

    if final_decision.get("status") != "PASS_VALID_NEGATIVE_CONCLUSION" or final_decision.get("promotion", {}).get("promoted") is not False:
        errors.append("FINAL_DETECTOR_DECISION_NOT_FROZEN_NEGATIVE")
    if final_decision.get("eval160_status") != "UNREAD":
        errors.append("EVAL160_NOT_UNREAD")
    for name, value in {**provenance.get("protected_counters", {}), **final_decision.get("protected_counters", {})}.items():
        if int(value) != 0:
            errors.append(f"PROTECTED_COUNTER_NONZERO:{name}")

    normalization = load_json(normalization_path)
    norm = normalization.get("episode_heldout", {}).get("train", {})
    mean = np.asarray(norm.get("mean", []), dtype=np.float32)
    std = np.asarray(norm.get("std", []), dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        errors.append("NORMALIZATION_SCHEMA_INVALID")

    thresholds = load_json(thresholds_path)
    physical_threshold = float(thresholds.get("physical_criticality", {}).get("threshold", float("nan")))
    closing_threshold = float(thresholds.get("gripper_closing_state", {}).get("threshold", float("nan")))
    if (physical_threshold, closing_threshold) != (0.55, 0.8):
        errors.append("FROZEN_THRESHOLD_BINDING_MISMATCH")
    if scheduler_freeze.get("status") != "FROZEN" or scheduler_freeze.get("attack_enabled") is not False or scheduler_freeze.get("one_shot") is not True:
        errors.append("SCHEDULER_NOT_FROZEN_CLEAN_ONLY")
    if "physical_criticality>=0.55" not in scheduler_freeze.get("emit_rule", "") or "gripper_closing_state>=0.80" not in scheduler_freeze.get("emit_rule", ""):
        errors.append("SCHEDULER_RULE_BINDING_MISMATCH")

    sys.path.insert(0, str(ROOT / "n5/phase3_student"))
    from n5_student_model import N5MultiHeadStudent
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    active_heads = tuple(checkpoint.get("active_heads", ()))
    try:
        model = N5MultiHeadStudent(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
    except Exception as exc:
        errors.append(f"STUDENT_CHECKPOINT_LOAD_FAILED:{type(exc).__name__}:{exc}")
        model = None
    if active_heads != ACTIVE_HEADS:
        errors.append("ACTIVE_HEADS_MISMATCH")
    if provenance.get("feature_order") != list(FEATURE_NAMES):
        errors.append("FEATURE_ORDER_MISMATCH")
    if mean.shape != (25,):
        errors.append("STUDENT_INPUT_DIM_NOT_25")

    score_paths = choose_score_reports(args.score_root)
    score_receipts: dict[str, Any] = {}
    for suite in SUITES:
        path = score_paths.get(suite)
        if path is None:
            errors.append(f"MISSING_SUITE_SCORE_REPORT:{suite}")
            continue
        report = load_json(path)
        counters = report.get("counters", {})
        if report.get("status") != "DIAGNOSTIC_CLEAN_ONLY" or report.get("suite") != suite:
            errors.append(f"SUITE_SCORE_REPORT_INVALID:{suite}")
        if any(int(v) != 0 for v in counters.values()) or report.get("eval160") != "UNREAD" or report.get("protected_evaluation") != "UNREAD":
            errors.append(f"SUITE_SCORE_REPORT_PROTECTED:{suite}")
        if report.get("processor_parity") != {"attention_mask_exact": True, "input_ids_exact": True, "pixel_values_exact": True}:
            errors.append(f"SUITE_PROCESSOR_PARITY_FAIL:{suite}")
        authority = report.get("authority", {})
        if authority.get("suite") != suite or authority.get("legacy_helper_status") != "HISTORICAL_COMPATIBILITY_ONLY":
            errors.append(f"SUITE_NATIVE_AUTHORITY_BINDING_FAIL:{suite}")
        score_receipts[suite] = {"path": str(path), "sha256": sha256(path), "suite": suite, "model_path": report.get("model_path"), "authority": authority}

    shadow_receipts: dict[str, Any] = {}
    if model is not None and mean.shape == (25,) and std.shape == (25,):
        h_phys = int(scheduler_freeze["h_phys"])
        t5 = int(scheduler_freeze["t5_steps"])
        replay_paths = sorted(args.replay_root.glob("parents/*/CLEAN_REPLAY_STUDENT_INPUTS_V1.json"))
        for suite in SUITES:
            candidates = [p for p in replay_paths if load_json(p).get("suite") == suite]
            if not candidates:
                errors.append(f"MISSING_CLEAN_REPLAY:{suite}")
                continue
            replay_path = candidates[0]
            try:
                features, close_flags = clean_shadow_rows(replay_path)
                first = student_predictions(model, features, mean, std)
                second = student_predictions(model, features, mean, std)
                first_schedule = schedule(first, close_flags, t5=t5, h_phys=h_phys, physical_threshold=physical_threshold, closing_threshold=closing_threshold)
                second_schedule = schedule(second, close_flags, t5=t5, h_phys=h_phys, physical_threshold=physical_threshold, closing_threshold=closing_threshold)
                first_array = np.asarray([[p["physical_criticality"], p["gripper_closing_state"]] for p in first])
                second_array = np.asarray([[p["physical_criticality"], p["gripper_closing_state"]] for p in second])
                max_diff = float(np.max(np.abs(first_array - second_array)))
                parity_indices = sorted({0, len(features) // 2, len(features) - 1})
                if first_schedule["first_emit_step"] is not None:
                    parity_indices.append(first_schedule["first_emit_step"])
                parity_indices = sorted(set(parity_indices))
                parity_diffs = []
                for index in parity_indices:
                    prefix = student_prediction_at_prefix(model, features, mean, std, index + 1)
                    full = first[index]
                    parity_diffs.append(max(abs(prefix[name] - full[name]) for name in ("physical_criticality", "gripper_closing_state")))
                max_prefix_parity_diff = float(max(parity_diffs, default=0.0))
                if max_diff != 0.0 or max_prefix_parity_diff != 0.0 or first_schedule["first_emit_step"] != second_schedule["first_emit_step"] or first_schedule["emitted_count"] not in (0, 1):
                    errors.append(f"SHADOW_DETERMINISM_FAIL:{suite}")
                shadow_receipts[suite] = {
                    "replay_path": str(replay_path), "replay_sha256": sha256(replay_path),
                    "canonical_parent_key": load_json(replay_path)["canonical_parent_key"],
                    "row_count": len(features), "feature_dim": 25,
                    "first_emit_step": first_schedule["first_emit_step"],
                    "emitted_count": first_schedule["emitted_count"],
                    "no_emit_retained": first_schedule["first_emit_step"] is None,
                    "repeat_max_probability_abs_diff": max_diff,
                    "prefix_parity_indices": parity_indices,
                    "prefix_max_probability_abs_diff": max_prefix_parity_diff,
                    "repeat_first_emit_equal": first_schedule["first_emit_step"] == second_schedule["first_emit_step"],
                    "scheduler_rule": scheduler_freeze["emit_rule"],
                    "student_heads_used": ["physical_criticality", "gripper_closing_state"],
                }
            except Exception as exc:
                errors.append(f"SHADOW_FAILED:{suite}:{type(exc).__name__}:{exc}")

    receipt = {
        "schema": "DETECTOR_AUTHORITY_RECEIPT_V1",
        "status": "PASS_T1_B_T1_C_CLEAN_ONLY" if not errors and len(shadow_receipts) == 4 else "HOLD_T1_DETECTOR_AUTHORITY",
        "scientific_claim": "frozen clean Teacher->Student detector clean-only runtime closure; no attack or causal outcome claim",
        "official_environment": OFFICIAL_ENV,
        "source": current_git,
        "sealed_detector": {
            "provenance": str(args.provenance), "provenance_sha256": sha256(args.provenance),
            "final_decision": str(args.final_decision), "final_decision_sha256": sha256(args.final_decision),
            "status": final_decision.get("status"), "decision": final_decision.get("decision"),
            "promotion": final_decision.get("promotion"), "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
            "active_heads": list(active_heads), "feature_order": list(FEATURE_NAMES),
            "feature_source_sha256_lf": sha256(feature_source, normalize_text=True),
            "adapter_source_sha256_lf": sha256(adapter_source, normalize_text=True),
            "student_model_source_sha256": sha256(model_source),
            "normalization": {"path": str(normalization_path), "sha256": normalization_sha, "family": provenance.get("normalization_family")},
            "thresholds": {"path": str(thresholds_path), "sha256": thresholds_sha, "physical_criticality": physical_threshold, "gripper_closing_state": closing_threshold},
        },
        "runtime_adapter": {
            "status": "D8_V3_DIRECT_RUNTIME_BOUND",
            "source": "scripts/stage_x/audit_stage_x1r_t1_detector_authority.py clean shadow path",
            "stale_n4": {"status": "REJECTED_STALE_NOT_CONSUMED", "path": str(stale_n4), "sha256": sha256(stale_n4), "reason": "historical 51D adapter and old checkpoint/threshold contract"},
        },
        "scheduler": {"path": str(args.scheduler_freeze), "sha256": sha256(args.scheduler_freeze), "receipt": scheduler_freeze},
        "suite_matched_clean_score_receipts": score_receipts,
        "clean_shadow": shadow_receipts,
        "protected_counters": {"pgd_calls": 0, "env_step_calls": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "vphys_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "eval160": "UNREAD", "protected_evaluation": "UNREAD", "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "errors": errors, "shadow_suites": sorted(shadow_receipts)}, sort_keys=True))
    return 0 if receipt["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
