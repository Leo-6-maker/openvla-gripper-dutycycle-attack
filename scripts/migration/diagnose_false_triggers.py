#!/usr/bin/env python3
"""Phase A: Sticky-arm vs model-selectivity diagnosis for M1B false triggers.

Three groups with matched controls:
  FALSE_TRIGGER (6): no-corridor episodes where detector triggered
  TRUE_POSITIVE (≤6): teacher-valid K10-pass triggered, verified from replay
  CORRECT_ABSTAIN (all eligible): teacher no-corridor + emit=-1, verified from replay

Per-cell hypothesis (mutually exclusive):
  A_STICKY_ARM — evidence broke between arm and emit
  B_SUSTAINED_MODEL_MISCLASSIFICATION — evidence sustained, but teacher says no-corridor

Overall hypothesis:
  A > n/2 → RUNTIME_STATE_MACHINE_STICKINESS
  B > n/2 → MODEL_SELECTIVITY_FAILURE
  else    → MIXED_RUNTIME_AND_MODEL

Output: evidence/m1c/phase_a/
  phase_a_summary.csv / .json
  hypothesis_classification.json
  cells/<episode>/<profile>/{emit_window.csv, state_machine_trace.csv, diagnostic_summary.json}
  artifact_manifest.json

CPU-only. No GPU, no MuJoCo, no model loading. Read-only on M1B evidence.
"""
import os, sys, json, csv, hashlib, time
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/artifact_manifest_complete.json"
CLASSIFICATION_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/final_classification.json"
EVIDENCE_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime_b0_d1"
OUT_BASE = REPO / "evidence/m1c/phase_a"

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]

TAU_C = 0.3
TAU_R = 0.3
GUARD = 5
WINDOW = 20


def _safe_float(val, default=float("nan")):
    """Parse float from CSV, handling empty strings from pre-init steps."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# ── Pre-registered false triggers (from M1B final classification) ──
FALSE_TRIGGER_KEYS = [
    ("butter_s1", "B0"),
    ("chocolate_pudding_s1", "B0"),
    ("cream_cheese_s0", "B0"),
    ("butter_s2", "B0"),
    ("butter_s1", "D1"),
    ("bbq_sauce_s2", "D1"),
]

M1B_CLOSE_COMMIT = "9ab9f26cbb87869e8395f3238f6703b870df43fc"


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_source_hashes(cell, cell_dir):
    """Fail-closed: all source artifacts must exist and match manifest SHA."""
    for fname, sha_key in [("step_telemetry.csv", "telemetry_sha256"),
                           ("episode_summary.json", "episode_summary_sha256"),
                           (".done", "done_sha256")]:
        fpath = cell_dir / fname
        if not fpath.exists():
            return f"SOURCE_MISSING: {fname}"
        actual = sha256_file(fpath)
        expected = cell.get(sha_key, "")
        if actual != expected:
            return (f"HASH_MISMATCH: {fname} "
                    f"expected={expected} actual={actual}")
    return None


def load_telemetry(cell_dir):
    """Load telemetry CSV with step-index assertion."""
    tel_path = cell_dir / "step_telemetry.csv"
    rows = list(csv.DictReader(open(tel_path)))
    # Assert: row["step"] must equal list index (sequential from 0)
    mismatches = []
    for i, r in enumerate(rows):
        csv_step = int(r.get("step", -1))
        if csv_step != i:
            mismatches.append(f"idx={i} csv_step={csv_step}")
    if mismatches:
        return None, f"STEP_INDEX_MISMATCH: {', '.join(mismatches[:5])}"
    return rows, None


def load_teacher_replay(cell):
    """Load privileged Teacher replay results. Returns (summary, labels, error)."""
    replay_dir = EVIDENCE_BASE / "replay_60cell" / cell["episode_key"] / cell["profile"]
    ts_path = replay_dir / "teacher_summary.json"
    tl_path = replay_dir / "teacher_labels.jsonl"

    if not ts_path.exists():
        return None, None, "TEACHER_SUMMARY_MISSING"
    if not tl_path.exists():
        return None, None, "TEACHER_LABELS_MISSING"

    teacher_summary = json.load(open(ts_path))
    teacher_labels = [json.loads(l) for l in open(tl_path) if l.strip()]
    return teacher_summary, teacher_labels, None


def extract_state_machine_trace(rows):
    """Full state machine trace: transitions and evidence breaks.

    Evidence break = arm_step < t < emit_step and keep-condition fails.
    Arm step itself is NOT checked for keep-condition (original SM only
    requires arm conditions, not release).
    Emit step is checked separately for phase validity.
    """
    trace = []
    current_state = "IDLE"
    arm_step = -1
    emit_step = -1
    evidence_breaks = []

    for i, r in enumerate(rows):
        det_state = r.get("detector_state", "")
        cp = _safe_float(r.get("corridor_p", "nan"))
        rp = _safe_float(r.get("release_p", "nan"))
        pp = r.get("pred_phase", "")
        fv = r.get("feat_valid", "") == "True"

        if det_state != current_state:
            trace.append({
                "step": i,
                "transition": f"{current_state}->{det_state}",
                "corridor_p": cp if not np.isnan(cp) else None,
                "release_p": rp if not np.isnan(rp) else None,
                "pred_phase": pp,
                "feat_valid": fv,
            })
            if det_state == "ARMED" and current_state == "IDLE":
                arm_step = i
            if det_state == "EMITTED":
                emit_step = i
            current_state = det_state

        # Evidence break: strictly between arm and emit (exclusive of both)
        if arm_step >= 0 and emit_step < 0 and i > arm_step:
            # Keep condition: feat_valid AND phase==stable_carry AND cp>tau_c AND rp<tau_r
            keep_ok = (fv and pp == "stable_carry"
                       and not np.isnan(cp) and cp > TAU_C
                       and not np.isnan(rp) and rp < TAU_R)
            if not keep_ok:
                evidence_breaks.append({
                    "step": i,
                    "feat_valid": fv,
                    "corridor_p": cp if not np.isnan(cp) else None,
                    "release_p": rp if not np.isnan(rp) else None,
                    "pred_phase": pp,
                    "fail_reason": _evidence_fail_reason(fv, cp, rp, pp),
                })

    # Check emit step phase validity (separate from pre-emit evidence breaks)
    phase_invalid_at_emit = False
    if emit_step >= 0:
        emit_r = rows[emit_step]
        emit_pp = emit_r.get("pred_phase", "")
        phase_invalid_at_emit = (emit_pp != "stable_carry")

    return {
        "trace": trace,
        "arm_step": arm_step,
        "emit_step": emit_step if emit_step >= 0 else None,
        "n_evidence_breaks": len(evidence_breaks),
        "evidence_breaks": evidence_breaks,
        "phase_invalid_at_emit": phase_invalid_at_emit,
        "arm_to_emit": emit_step - arm_step if arm_step >= 0 and emit_step >= 0 else -1,
    }


def _evidence_fail_reason(fv, cp, rp, pp):
    reasons = []
    if not fv:
        reasons.append("feat_invalid")
    if pp != "stable_carry":
        reasons.append(f"phase={pp}")
    if cp is None or np.isnan(cp) or cp <= TAU_C:
        val = f"{cp:.4f}" if (cp is not None and not np.isnan(cp)) else "nan"
        reasons.append(f"cp={val}")
    if rp is None or np.isnan(rp) or rp >= TAU_R:
        val = f"{rp:.4f}" if (rp is not None and not np.isnan(rp)) else "nan"
        reasons.append(f"rp={val}")
    return "+".join(reasons) if reasons else "ok"


def extract_window(rows, center_step):
    """Extract ±WINDOW rows around center_step."""
    n = len(rows)
    lo = max(0, center_step - WINDOW)
    hi = min(n, center_step + WINDOW + 1)

    window_rows = []
    for i in range(lo, hi):
        r = rows[i]
        cp = _safe_float(r.get("corridor_p", "nan"))
        rp = _safe_float(r.get("release_p", "nan"))

        entry = {
            "step": i,
            "rel": i - center_step,
            "detector_state": r.get("detector_state", ""),
            "corridor_p": cp if not np.isnan(cp) else None,
            "release_p": rp if not np.isnan(rp) else None,
            "pred_phase": r.get("pred_phase", ""),
            "mlp_emit": int(r.get("mlp_emit", -1)),
            "feat_valid": r.get("feat_valid", "") == "True",
            "gripper_raw": _safe_float(r.get("raw_gripper", "nan")),
            "gripper_env": _safe_float(r.get("env_gripper", "nan")),
            "qpos_sum": _safe_float(r.get("qpos_sum", "nan")),
            "eef_x": _safe_float(r.get("eef_x", "nan")),
            "eef_y": _safe_float(r.get("eef_y", "nan")),
            "eef_z": _safe_float(r.get("eef_z", "nan")),
            "obj_x": _safe_float(r.get("obj_x", "nan")),
            "obj_y": _safe_float(r.get("obj_y", "nan")),
            "obj_z": _safe_float(r.get("obj_z", "nan")),
        }
        for fn in SC5_FEATURES:
            entry["f_" + fn] = _safe_float(r.get("f_" + fn, "nan"))
        window_rows.append(entry)
    return window_rows


def compute_arm_interval_metrics(rows, arm_step, emit_step):
    """Metrics for the pre-emit interval (arm_step < t < emit_step).

    Arm step itself is NOT evaluated for keep-condition because the
    original state machine only checks arm conditions at that point.
    Emit step is checked separately via phase_invalid_at_emit.
    """
    # Pre-emit: strictly between arm and emit
    pre_emit = [r for r in rows if arm_step < int(r.get("step", -1)) < emit_step]
    emit_r = rows[emit_step] if emit_step < len(rows) else {}

    n_total = len(pre_emit)
    n_phase_not_sc = 0
    n_cp_below = 0
    n_rp_above = 0
    max_consecutive_ok = 0
    current_consecutive = 0
    first_break_step = -1

    for r in pre_emit:
        pp = r.get("pred_phase", "")
        cp = _safe_float(r.get("corridor_p", "nan"))
        rp = _safe_float(r.get("release_p", "nan"))
        fv = r.get("feat_valid", "") == "True"

        phase_ok = (pp == "stable_carry")
        cp_ok = (not np.isnan(cp) and cp > TAU_C)
        rp_ok = (not np.isnan(rp) and rp < TAU_R)
        all_ok = fv and phase_ok and cp_ok and rp_ok

        if not phase_ok:
            n_phase_not_sc += 1
        if not cp_ok:
            n_cp_below += 1
        if not rp_ok:
            n_rp_above += 1

        if all_ok:
            current_consecutive += 1
            max_consecutive_ok = max(max_consecutive_ok, current_consecutive)
        else:
            if first_break_step < 0 and current_consecutive > 0:
                first_break_step = int(r.get("step", -1))
            current_consecutive = 0

    pre_emit_break = (n_cp_below > 0 or n_phase_not_sc > 0 or n_rp_above > 0)

    # Emit step: check phase validity separately
    emit_pp = emit_r.get("pred_phase", "")
    invalid_phase_at_emit = (emit_pp != "stable_carry")

    sticky_arm = pre_emit_break or invalid_phase_at_emit

    return {
        "n_arm_interval_steps": n_total,
        "n_phase_not_stable_carry": n_phase_not_sc,
        "n_corridor_below_tau": n_cp_below,
        "n_release_above_tau": n_rp_above,
        "max_consecutive_valid_evidence": max_consecutive_ok,
        "first_evidence_break_after_arm": first_break_step,
        "pre_emit_evidence_break": pre_emit_break,
        "invalid_phase_at_emit": invalid_phase_at_emit,
        "sticky_arm_counterexample": sticky_arm,
    }


def compute_emit_point_metrics(rows, arm_step, emit_step):
    """Point metrics at arm and emit."""
    arm_r = rows[arm_step] if 0 <= arm_step < len(rows) else {}
    emit_r = rows[emit_step] if 0 <= emit_step < len(rows) else {}

    def _cp(r):
        v = _safe_float(r.get("corridor_p", "nan"))
        return v if not np.isnan(v) else None

    def _rp(r):
        v = _safe_float(r.get("release_p", "nan"))
        return v if not np.isnan(v) else None

    return {
        "phase_at_arm": arm_r.get("pred_phase", ""),
        "phase_at_emit": emit_r.get("pred_phase", ""),
        "corridor_p_at_arm": _cp(arm_r),
        "corridor_p_at_emit": _cp(emit_r),
        "release_p_at_arm": _rp(arm_r),
        "release_p_at_emit": _rp(emit_r),
    }


def classify_cell_hypothesis(arm_metrics):
    """Single-cell classification — mutually exclusive A or B."""
    if not arm_metrics:
        return "N_A_NOT_TRIGGERED"
    if arm_metrics.get("sticky_arm_counterexample", False):
        return "A_STICKY_ARM"
    else:
        return "B_SUSTAINED_MODEL_MISCLASSIFICATION"


def process_cell(cell, group):
    """Process one cell: verify SHAs, load data, compute metrics.

    Returns (result_dict, error_string). If error_string is not None,
    result_dict is a minimal error record.
    """
    cell_dir = EVIDENCE_BASE / cell["relative_path"]

    # SHA verification
    sha_err = verify_source_hashes(cell, cell_dir)
    if sha_err:
        return {"_error": sha_err, "episode_key": cell["episode_key"],
                "profile": cell["profile"], "group": group}, sha_err

    # Load telemetry with step-index assertion
    rows, step_err = load_telemetry(cell_dir)
    if step_err:
        return {"_error": step_err, "episode_key": cell["episode_key"],
                "profile": cell["profile"], "group": group}, step_err

    summary = json.load(open(cell_dir / "episode_summary.json"))
    teacher_summary, teacher_labels, teacher_err = load_teacher_replay(cell)

    n_steps = len(rows)
    manifest_emit = cell["emit_step"]

    # State machine trace
    sm_trace = extract_state_machine_trace(rows)
    emit_step = sm_trace["emit_step"]  # from telemetry state transitions, not manifest

    # Center step
    center = emit_step if emit_step is not None else n_steps // 2

    # Window around center
    window_rows = extract_window(rows, center)

    # Arm-interval metrics (only for triggered episodes)
    arm_metrics = {}
    point_metrics = {}
    arm_step = sm_trace["arm_step"]
    if emit_step is not None and arm_step >= 0:
        arm_metrics = compute_arm_interval_metrics(rows, arm_step, emit_step)
        point_metrics = compute_emit_point_metrics(rows, arm_step, emit_step)

    # Cell hypothesis
    cell_hypothesis = classify_cell_hypothesis(arm_metrics)

    # Teacher labels around center
    teacher_window = []
    if teacher_labels:
        teacher_window = [lab for lab in teacher_labels
                          if center - WINDOW <= lab.get("step_idx", -1) <= center + WINDOW]

    # Full-trajectory summary stats
    first_cp_high = -1
    for r in rows:
        cp = _safe_float(r.get("corridor_p", "nan"))
        if not np.isnan(cp) and cp > TAU_C:
            first_cp_high = int(r.get("step", -1))
            break

    first_sc_step = -1
    for r in rows:
        if r.get("pred_phase", "") == "stable_carry":
            first_sc_step = int(r.get("step", -1))
            break

    n_cp_high_total = sum(1 for r in rows
                          if not np.isnan(_safe_float(r.get("corridor_p", "nan")))
                          and _safe_float(r.get("corridor_p", "nan")) > TAU_C)
    n_sc_total = sum(1 for r in rows if r.get("pred_phase", "") == "stable_carry")
    n_feat_valid = sum(1 for r in rows if r.get("feat_valid", "") == "True")

    result = {
        "group": group,
        "episode_key": cell["episode_key"],
        "profile": cell["profile"],
        "task_idx": cell["task_idx"],
        "state_id": cell["state_id"],
        "success": cell.get("success", False),
        "n_steps": n_steps,
        "n_feat_valid": n_feat_valid,
        "emit_step": emit_step,
        "center_step": center,
        "window": window_rows,
        "sm_trace": sm_trace,
        "arm_metrics": arm_metrics,
        "point_metrics": point_metrics,
        "cell_hypothesis": cell_hypothesis,
        "first_cp_high": first_cp_high,
        "first_sc_step": first_sc_step,
        "n_cp_high_total": n_cp_high_total,
        "n_sc_total": n_sc_total,
        "teacher_summary": teacher_summary,
        "teacher_window": teacher_window,
        "teacher_error": teacher_err,
        "telemetry_sha_verified": sha256_file(cell_dir / "step_telemetry.csv"),
        "summary_sha_verified": sha256_file(cell_dir / "episode_summary.json"),
    }
    return result, None


def build_summary_row(result):
    """One row for phase_a_summary.csv."""
    am = result.get("arm_metrics", {})
    pm = result.get("point_metrics", {})
    ts = result.get("teacher_summary", {}) or {}
    return {
        "group": result["group"],
        "episode_key": result["episode_key"],
        "profile": result["profile"],
        "task_idx": result["task_idx"],
        "state_id": result["state_id"],
        "success": result["success"],
        "n_steps": result["n_steps"],
        "emit_step": result["emit_step"] if result["emit_step"] is not None else -1,
        "center_step": result["center_step"],
        "arm_step": result["sm_trace"]["arm_step"],
        "arm_to_emit": result["sm_trace"]["arm_to_emit"],
        "n_evidence_breaks": result["sm_trace"]["n_evidence_breaks"],
        "phase_invalid_at_emit": result["sm_trace"]["phase_invalid_at_emit"],
        "pre_emit_evidence_break": am.get("pre_emit_evidence_break", None),
        "sticky_arm": am.get("sticky_arm_counterexample", None),
        "n_arm_interval_steps": am.get("n_arm_interval_steps", -1),
        "n_phase_not_sc": am.get("n_phase_not_stable_carry", -1),
        "n_cp_below_tau": am.get("n_corridor_below_tau", -1),
        "n_rp_above_tau": am.get("n_release_above_tau", -1),
        "max_consecutive_valid": am.get("max_consecutive_valid_evidence", -1),
        "first_evidence_break": am.get("first_evidence_break_after_arm", -1),
        "phase_at_arm": pm.get("phase_at_arm", ""),
        "phase_at_emit": pm.get("phase_at_emit", ""),
        "corridor_p_at_arm": pm.get("corridor_p_at_arm", None),
        "corridor_p_at_emit": pm.get("corridor_p_at_emit", None),
        "release_p_at_arm": pm.get("release_p_at_arm", None),
        "release_p_at_emit": pm.get("release_p_at_emit", None),
        "first_cp_high": result["first_cp_high"],
        "first_sc_step": result["first_sc_step"],
        "n_cp_high_total": result["n_cp_high_total"],
        "n_sc_total": result["n_sc_total"],
        "cell_hypothesis": result["cell_hypothesis"],
        "teacher_sc_present": ts.get("stable_carry_present", None),
        "teacher_anchor": ts.get("anchor_candidate", None),
        "teacher_k10_valid": ts.get("full_k10_valid", None),
        "teacher_k10_reason": ts.get("k10_invalid_reason", ""),
    }


def write_cell_outputs(result, out_base):
    """Write per-cell CSV and JSON outputs. Only for non-error results."""
    cell_dir = out_base / "cells" / result["episode_key"] / result["profile"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    # emit_window.csv
    if result.get("window"):
        with open(cell_dir / "emit_window.csv", "w", newline="") as f:
            fieldnames = result["window"][0].keys()
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["window"])

    # state_machine_trace.csv
    sm = result["sm_trace"]
    with open(cell_dir / "state_machine_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "transition", "corridor_p",
                                          "release_p", "pred_phase", "feat_valid"])
        w.writeheader()
        w.writerows(sm["trace"])
    if sm.get("evidence_breaks"):
        with open(cell_dir / "evidence_breaks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "feat_valid", "corridor_p",
                                              "release_p", "pred_phase", "fail_reason"])
            w.writeheader()
            w.writerows(sm["evidence_breaks"])

    # diagnostic_summary.json (strip large arrays)
    small = {k: v for k, v in result.items()
             if k not in ("window", "sm_trace", "teacher_window")}
    # Keep evidence_breaks count but not the full list in summary
    small["sm_trace"] = {k: v for k, v in sm.items() if k != "evidence_breaks"}
    small["sm_trace"]["n_evidence_breaks"] = sm["n_evidence_breaks"]
    with open(cell_dir / "diagnostic_summary.json", "w") as f:
        json.dump(small, f, indent=2, default=str)

    # Record output file SHAs
    output_shas = {}
    for fname in ["emit_window.csv", "state_machine_trace.csv", "diagnostic_summary.json"]:
        fpath = cell_dir / fname
        if fpath.exists():
            output_shas[fname] = {
                "sha256": sha256_file(fpath),
                "size": fpath.stat().st_size,
            }
        if (cell_dir / "evidence_breaks.csv").exists():
            bp = cell_dir / "evidence_breaks.csv"
            output_shas["evidence_breaks.csv"] = {
                "sha256": sha256_file(bp),
                "size": bp.stat().st_size,
            }
    return output_shas


def classify_overall_hypothesis(false_trigger_results):
    """Overall hypothesis from false-trigger episodes only.

    Strict majority: A > n/2 → runtime, B > n/2 → model, else mixed.
    """
    hypotheses = [r["cell_hypothesis"] for r in false_trigger_results
                  if "_error" not in r and r.get("cell_hypothesis", "").startswith(("A_", "B_"))]
    n = len(hypotheses)
    if n == 0:
        return {"primary_failure_mode": "UNKNOWN", "counts": {}}

    counts = {"A_STICKY_ARM": 0, "B_SUSTAINED_MODEL_MISCLASSIFICATION": 0}
    for h in hypotheses:
        if h in counts:
            counts[h] += 1

    a = counts["A_STICKY_ARM"]
    b = counts["B_SUSTAINED_MODEL_MISCLASSIFICATION"]

    if a > n / 2:
        mode = "RUNTIME_STATE_MACHINE_STICKINESS"
    elif b > n / 2:
        mode = "MODEL_SELECTIVITY_FAILURE"
    else:
        mode = "MIXED_RUNTIME_AND_MODEL"

    return {
        "primary_failure_mode": mode,
        "false_trigger_counts": counts,
        "false_trigger_n": n,
        "false_trigger_sticky_rate": a / n if n > 0 else 0,
        "recommendation": _recommendation(mode),
    }


def _recommendation(mode):
    if mode == "RUNTIME_STATE_MACHINE_STICKINESS":
        return ("M1C-R: Implement revocable state machine with IDLE↔CANDIDATE↔ARMED "
                "hysteresis. Retrain only if six gates still not met after repair.")
    elif mode == "MODEL_SELECTIVITY_FAILURE":
        return ("M1C-M: State machine fix alone insufficient. Train SC5-v2 with "
                "explicit abstention head, hard negatives, and temporal consistency loss.")
    else:
        return ("M1C-RM: Repair state machine first, re-evaluate on frozen data. "
                "Train SC5-v2 only for residual errors.")


def verify_teacher_gate(cell, teacher_summary, required_k10, required_emit_negative):
    """Fail-closed teacher verification for control groups.

    required_k10=True: teacher full_k10_valid must be True
    required_k10=False: teacher full_k10_valid must be False
    required_emit_negative=True: cell emit_step must be -1
    """
    if teacher_summary is None:
        return False, "no_teacher_summary"
    k10 = teacher_summary.get("full_k10_valid", None)
    if required_k10 and k10 is not True:
        return False, f"teacher_k10_not_true (got {k10})"
    if not required_k10 and k10 is not False:
        return False, f"teacher_k10_not_false (got {k10})"
    if required_emit_negative and cell.get("emit_step", -1) != -1:
        return False, f"emit_not_negative (got {cell.get('emit_step')})"
    return True, "ok"


def print_summary_table(results, title):
    """Print compact comparison table."""
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"{'='*130}")
    header = (f"{'Group':<20} {'Episode':<25} {'Prof':>4} {'Emit':>5} {'Arm':>5} "
              f"{'A→E':>5} {'Brk':>4} {'Ph@Em':>5} {'Sticky':>6} {'Hyp':<35} "
              f"{'ph@arm':<25} {'ph@emit':<25}")
    print(header)
    print("-" * 130)
    for r in results:
        if "_error" in r:
            print(f"  {'ERROR':<20} {r['episode_key']:<25} {r['profile']:>4}  --- {r['_error'][:80]}")
            continue
        am = r.get("arm_metrics", {})
        pm = r.get("point_metrics", {})
        arm = r["sm_trace"]["arm_step"]
        emit = r["emit_step"] if r["emit_step"] is not None else -1
        ate = r["sm_trace"]["arm_to_emit"]
        brk = r["sm_trace"]["n_evidence_breaks"]
        ph_inv = "INV" if r["sm_trace"].get("phase_invalid_at_emit") else "ok"
        sticky = "YES" if am.get("sticky_arm_counterexample") else ("no" if am else "-")
        hyp = r.get("cell_hypothesis", "?")
        ph_a = pm.get("phase_at_arm", "")[:25] if pm else ""
        ph_e = pm.get("phase_at_emit", "")[:25] if pm else ""
        print(f"  {r['group']:<20} {r['episode_key']:<25} {r['profile']:>4} {emit:>5} {arm:>5} "
              f"{ate:>5} {brk:>4} {ph_inv:>5} {sticky:>6} {hyp:<35} "
              f"{ph_a:<25} {ph_e:<25}")
    print("-" * 130)


def main():
    manifest = json.load(open(MANIFEST_PATH))
    cells_by_key = {}
    for c in manifest["cells"]:
        cells_by_key[(c["episode_key"], c["profile"])] = c

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # ── Build target list with fail-closed teacher verification ─────
    targets = []
    verification_log = []

    # Group 1: FALSE_TRIGGERS (pre-registered)
    for ek, pk in FALSE_TRIGGER_KEYS:
        cell = cells_by_key.get((ek, pk))
        if not cell:
            verification_log.append(f"FALSE_TRIGGER MISSING: {ek}/{pk}")
            continue
        targets.append((cell, "FALSE_TRIGGER"))

    # Group 2: TRUE_POSITIVES — programmatic selection with teacher verification
    # Prefer same tasks as false triggers; strict K10 gate
    ft_tasks = {cells_by_key[(ek, pk)]["task_idx"] for ek, pk in FALSE_TRIGGER_KEYS
                if (ek, pk) in cells_by_key}
    tp_candidates = []
    for c in manifest["cells"]:
        key = (c["episode_key"], c["profile"])
        if key in {(ek, pk) for ek, pk in FALSE_TRIGGER_KEYS}:
            continue  # not a false trigger
        if c["emit_step"] < 0:
            continue  # not triggered
        ts, _, terr = load_teacher_replay(c)
        ok, reason = verify_teacher_gate(c, ts, required_k10=True, required_emit_negative=False)
        if ok:
            same_task = c["task_idx"] in ft_tasks
            tp_candidates.append((c, same_task))
        else:
            verification_log.append(f"TP_CANDIDATE_REJECTED: {c['episode_key']}/{c['profile']} reason={reason}")

    # Sort: same-task first, then by emit_step (prefer mid-range)
    tp_candidates.sort(key=lambda x: (not x[1], abs(x[0]["emit_step"] - 100)))
    tp_selected = tp_candidates[:6]
    for c, _ in tp_selected:
        targets.append((c, "TRUE_POSITIVE"))

    if len(tp_selected) < 6:
        verification_log.append(f"TRUE_POSITIVE_SHORTFALL: only {len(tp_selected)}/6 eligible")

    # Group 3: CORRECT_ABSTAIN — teacher no-corridor + emit=-1
    ft_key_set = {(ek, pk) for ek, pk in FALSE_TRIGGER_KEYS}
    tp_key_set = {(c["episode_key"], c["profile"]) for c, _ in tp_selected}
    ca_candidates = []
    for c in manifest["cells"]:
        key = (c["episode_key"], c["profile"])
        if key in ft_key_set or key in tp_key_set:
            continue
        if c["emit_step"] != -1:
            continue
        ts, _, terr = load_teacher_replay(c)
        ok, reason = verify_teacher_gate(c, ts, required_k10=False, required_emit_negative=True)
        if ok:
            ca_candidates.append(c)
        else:
            verification_log.append(f"CA_CANDIDATE_REJECTED: {c['episode_key']}/{c['profile']} reason={reason}")

    for c in ca_candidates:
        targets.append((c, "CORRECT_ABSTAIN"))

    if len(ca_candidates) < 5:
        verification_log.append(f"CORRECT_ABSTAIN_SHORTFALL: only {len(ca_candidates)} eligible (want ≥5)")

    # Deduplicate
    seen = set()
    unique_targets = []
    for t in targets:
        key = (t[0]["episode_key"], t[0]["profile"], t[1])
        if key not in seen:
            seen.add(key)
            unique_targets.append(t)
    targets = unique_targets

    print(f"Phase A: {len(targets)} total targets")
    groups = defaultdict(int)
    for t in targets:
        groups[t[1]] += 1
    for g, n in groups.items():
        print(f"  {g}: {n}")
    if verification_log:
        print(f"\nVerification log ({len(verification_log)} entries):")
        for vl in verification_log:
            print(f"  {vl}")

    # ── Process all cells ──────────────────────────────────────────
    all_results = []
    errors = []
    cell_output_shas = {}

    for i, (cell, group) in enumerate(targets):
        key = f"{cell['episode_key']}/{cell['profile']}"
        print(f"\n[{i+1}/{len(targets)}] {group} {key} ...", end=" ", flush=True)

        result, err = process_cell(cell, group)
        all_results.append(result)

        if err:
            print(f"ERROR: {err}")
            errors.append(result)
            # Write error record, NOT normal cell outputs
            err_dir = OUT_BASE / "cells" / cell["episode_key"] / cell["profile"]
            err_dir.mkdir(parents=True, exist_ok=True)
            with open(err_dir / "error.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
        else:
            print(f"OK emit={result['emit_step']} arm={result['sm_trace']['arm_step']} "
                  f"hyp={result['cell_hypothesis']}")
            output_shas = write_cell_outputs(result, OUT_BASE)
            cell_output_shas[key] = output_shas

    # ── Summary CSV ────────────────────────────────────────────────
    summary_rows = [build_summary_row(r) for r in all_results if "_error" not in r]
    if summary_rows:
        csv_path = OUT_BASE / "phase_a_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary CSV: {csv_path}")

    # ── Summary JSON ───────────────────────────────────────────────
    json_path = OUT_BASE / "phase_a_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    # ── Overall hypothesis classification ──────────────────────────
    ft_results = [r for r in all_results if r.get("group") == "FALSE_TRIGGER"]
    overall = classify_overall_hypothesis(ft_results)

    # Cross-group comparison
    tp_results = [r for r in all_results if r.get("group") == "TRUE_POSITIVE" and "_error" not in r]
    tp_sticky = sum(1 for r in tp_results
                    if r.get("arm_metrics", {}).get("sticky_arm_counterexample", False))
    overall["true_positive_n"] = len(tp_results)
    overall["true_positive_sticky_rate"] = tp_sticky / len(tp_results) if tp_results else 0

    ca_results = [r for r in all_results if r.get("group") == "CORRECT_ABSTAIN" and "_error" not in r]
    ca_ever_armed = sum(1 for r in ca_results if r["sm_trace"]["arm_step"] >= 0)
    overall["correct_abstain_n"] = len(ca_results)
    overall["correct_abstain_ever_armed"] = f"{ca_ever_armed}/{len(ca_results)}" if ca_results else "0/0"

    hyp_path = OUT_BASE / "hypothesis_classification.json"
    with open(hyp_path, "w") as f:
        json.dump(overall, f, indent=2)

    # ── Output manifest with full provenance ───────────────────────
    m1c_commit = os.popen("git rev-parse HEAD").read().strip() if REPO.joinpath(".git").exists() else "unknown"
    script_sha = sha256_file(__file__)
    class_sha = sha256_file(CLASSIFICATION_PATH)

    output_files = {}
    for fname in ["phase_a_summary.csv", "phase_a_summary.json", "hypothesis_classification.json"]:
        fp = OUT_BASE / fname
        if fp.exists():
            output_files[fname] = {"sha256": sha256_file(fp), "size": fp.stat().st_size}

    manifest_out = {
        "gate": "PHASE_A_DIAGNOSTIC_MANIFEST",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "m1b_close_commit": M1B_CLOSE_COMMIT,
        "m1c_diagnostic_commit": m1c_commit,
        "diagnostic_script_sha256": script_sha,
        "m1b_classification_sha256": class_sha,
        "n_targets": len(targets),
        "n_success": len(all_results) - len(errors),
        "n_errors": len(errors),
        "group_counts": dict(groups),
        "tau_corridor": TAU_C,
        "tau_release": TAU_R,
        "guard": GUARD,
        "window": WINDOW,
        "overall_hypothesis": overall,
        "verification_log": verification_log,
        "output_files": output_files,
        "cell_output_shas": cell_output_shas,
    }
    with open(OUT_BASE / "artifact_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Console output ─────────────────────────────────────────────
    print_summary_table(all_results, "PHASE A — COMPLETE DIAGNOSTIC TABLE")

    print(f"\n{'='*80}")
    print(f"HYPOTHESIS CLASSIFICATION")
    print(f"{'='*80}")
    print(f"  Primary failure mode: {overall['primary_failure_mode']}")
    fc = overall.get("false_trigger_counts", {})
    print(f"  False trigger: A_STICKY={fc.get('A_STICKY_ARM',0)} "
          f"B_MODEL={fc.get('B_SUSTAINED_MODEL_MISCLASSIFICATION',0)} "
          f"sticky_rate={overall['false_trigger_sticky_rate']:.2f}")
    print(f"  True positive sticky rate: {overall['true_positive_sticky_rate']:.2f}")
    print(f"  Correct abstain ever armed: {overall['correct_abstain_ever_armed']}")
    print(f"  Recommendation: {overall['recommendation']}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e['episode_key']}/{e['profile']}: {e['_error']}")

    print(f"\n  Output: {OUT_BASE}")
    print(f"  Hypothesis: {hyp_path}")
    print(f"  Manifest: {OUT_BASE / 'artifact_manifest.json'}")

    # ── Gate check ─────────────────────────────────────────────────
    ft_ok = sum(1 for t in targets if t[1] == "FALSE_TRIGGER")
    tp_ok = sum(1 for t in targets if t[1] == "TRUE_POSITIVE")
    ca_ok = sum(1 for t in targets if t[1] == "CORRECT_ABSTAIN")
    gate_pass = (ft_ok == 6 and tp_ok == 6 and ca_ok >= 5
                 and len(errors) == 0 and verification_log == [])
    print(f"\n  PHASE_A_GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(f"    FALSE_TRIGGER={ft_ok}/6 TRUE_POSITIVE={tp_ok}/6 CORRECT_ABSTAIN={ca_ok}/5"
          f" errors={len(errors)} verify_warnings={len(verification_log)}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
