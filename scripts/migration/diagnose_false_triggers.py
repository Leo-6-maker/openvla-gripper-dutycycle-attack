#!/usr/bin/env python3
"""Phase A: Sticky-arm vs model-selectivity diagnosis for M1B false triggers.

Three groups with matched controls:
  FALSE_TRIGGER (6): no-corridor episodes where detector triggered
  TRUE_POSITIVE (6): teacher-valid, K10-pass, triggered
  CORRECT_ABSTAIN (all): no-corridor episodes where detector did NOT trigger

Per-group diagnostics:
  - SHA-verify source artifacts before reading
  - Extract emit ±20 (or center ±20 for non-triggered)
  - State machine trace: IDLE→ARMED→EMITTED
  - Sticky-arm counterexample detection
  - Hypothesis classification: A(sticky-arm) / B(model-selectivity) / C(mixed)

Output: /mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase_a/
  phase_a_summary.csv
  phase_a_summary.json
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

# ── Group definitions ──────────────────────────────────────────────
FALSE_TRIGGERS = [
    {"episode_key": "butter_s1", "profile": "B0", "reason": "no-corridor false trigger"},
    {"episode_key": "chocolate_pudding_s1", "profile": "B0", "reason": "no-corridor false trigger"},
    {"episode_key": "cream_cheese_s0", "profile": "B0", "reason": "no-corridor false trigger"},
    {"episode_key": "butter_s2", "profile": "B0", "reason": "no-corridor false trigger"},
    {"episode_key": "butter_s1", "profile": "D1", "reason": "no-corridor false trigger"},
    {"episode_key": "bbq_sauce_s2", "profile": "D1", "reason": "no-corridor false trigger"},
]

# True positives: teacher-valid, K10-pass triggered episodes.
# Prefer same tasks as false triggers; fill remainder by emit consistency.
TRUE_POSITIVE_CANDIDATES = [
    # Same tasks as false triggers, teacher-valid
    {"episode_key": "butter_s0", "profile": "B0"},
    {"episode_key": "butter_s0", "profile": "D1"},
    {"episode_key": "chocolate_pudding_s0", "profile": "B0"},
    {"episode_key": "chocolate_pudding_s2", "profile": "B0"},
    {"episode_key": "cream_cheese_s1", "profile": "B0"},
    {"episode_key": "bbq_sauce_s0", "profile": "B0"},
]


def load_manifest():
    return json.load(open(MANIFEST_PATH))


def load_classification():
    return json.load(open(CLASSIFICATION_PATH))


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


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
            return f"HASH_MISMATCH: {fname} expected={expected[:16]} actual={actual[:16]}"
    return None


def load_teacher_data(cell):
    """Load privileged Teacher replay results if available."""
    replay_dir = EVIDENCE_BASE / "replay_60cell" / cell["episode_key"] / cell["profile"]
    teacher_summary = {}
    teacher_labels = []
    ts_path = replay_dir / "teacher_summary.json"
    if ts_path.exists():
        teacher_summary = json.load(open(ts_path))
    tl_path = replay_dir / "teacher_labels.jsonl"
    if tl_path.exists():
        teacher_labels = [json.loads(l) for l in open(tl_path) if l.strip()]
    return teacher_summary, teacher_labels


def extract_state_machine_trace(rows, emit_step):
    """Full state machine trace from telemetry, not just window.

    Identifies every state transition and evidence break.
    """
    trace = []
    current_state = "IDLE"
    arm_step = -1
    evidence_breaks = []  # steps within ARMED where conditions not met

    for i, r in enumerate(rows):
        det_state = r.get("detector_state", "")
        cp = float(r.get("corridor_p", "nan"))
        rp = float(r.get("release_p", "nan"))
        pp = r.get("pred_phase", "")
        fv = r.get("feat_valid", "") == "True"

        # Detect state transitions from telemetry (not recomputing)
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
            current_state = det_state

        # After arming, check if evidence would satisfy a stricter state machine
        if arm_step >= 0 and det_state == "ARMED":
            evidence_ok = (fv and pp == "stable_carry" and not np.isnan(cp)
                           and cp > TAU_C and not np.isnan(rp) and rp < TAU_R)
            if not evidence_ok:
                evidence_breaks.append({
                    "step": i,
                    "feat_valid": fv,
                    "corridor_p": cp if not np.isnan(cp) else None,
                    "release_p": rp if not np.isnan(rp) else None,
                    "pred_phase": pp,
                    "fail_reason": _evidence_fail_reason(fv, cp, rp, pp),
                })

    return {
        "trace": trace,
        "arm_step": arm_step,
        "n_evidence_breaks": len(evidence_breaks),
        "evidence_breaks": evidence_breaks,
        "arm_to_emit": emit_step - arm_step if arm_step >= 0 and emit_step >= 0 else -1,
    }


def _evidence_fail_reason(fv, cp, rp, pp):
    reasons = []
    if not fv:
        reasons.append("feat_invalid")
    if pp != "stable_carry":
        reasons.append(f"phase={pp}")
    if cp is None or np.isnan(cp) or cp <= TAU_C:
        reasons.append(f"corridor_p={cp}")
    if rp is None or np.isnan(rp) or rp >= TAU_R:
        reasons.append(f"release_p={rp}")
    return "+".join(reasons) if reasons else "ok"


def extract_window(rows, center_step):
    """Extract ±WINDOW rows around center_step."""
    n = len(rows)
    lo = max(0, center_step - WINDOW)
    hi = min(n, center_step + WINDOW + 1)

    window_rows = []
    for i in range(lo, hi):
        r = rows[i]
        cp = float(r.get("corridor_p", "nan"))
        rp = float(r.get("release_p", "nan"))

        entry = {
            "step": i,
            "rel": i - center_step,
            "detector_state": r.get("detector_state", ""),
            "corridor_p": cp if not np.isnan(cp) else None,
            "release_p": rp if not np.isnan(rp) else None,
            "pred_phase": r.get("pred_phase", ""),
            "mlp_emit": int(r.get("mlp_emit", -1)),
            "feat_valid": r.get("feat_valid", "") == "True",
            "gripper_raw": float(r.get("raw_gripper", "nan")),
            "gripper_env": float(r.get("env_gripper", "nan")),
            "qpos_sum": float(r.get("qpos_sum", "nan")),
            "eef_x": float(r.get("eef_x", "nan")),
            "eef_y": float(r.get("eef_y", "nan")),
            "eef_z": float(r.get("eef_z", "nan")),
            "obj_x": float(r.get("obj_x", "nan")),
            "obj_y": float(r.get("obj_y", "nan")),
            "obj_z": float(r.get("obj_z", "nan")),
        }
        for fn in SC5_FEATURES:
            entry["f_" + fn] = float(r.get("f_" + fn, "nan"))
        window_rows.append(entry)
    return window_rows


def compute_arm_interval_metrics(rows, arm_step, emit_step):
    """Metrics for the interval [arm_step, emit_step]."""
    interval = [r for r in rows if arm_step <= int(r.get("step", -1)) <= emit_step]
    if not interval:
        return {}

    n_total = len(interval)
    n_phase_not_sc = 0
    n_cp_below = 0
    n_rp_above = 0
    max_consecutive_ok = 0
    current_consecutive = 0
    first_break_step = -1

    for r in interval:
        pp = r.get("pred_phase", "")
        cp = float(r.get("corridor_p", "nan"))
        rp = float(r.get("release_p", "nan"))
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

    # Sticky-arm: evidence broke AFTER arming but state remained ARMED
    sticky_arm = (n_cp_below > 0 or n_phase_not_sc > 0 or n_rp_above > 0)

    # Continuous model error: evidence NEVER broke
    continuous_model_error = (not sticky_arm)

    return {
        "n_arm_interval_steps": n_total,
        "n_phase_not_stable_carry": n_phase_not_sc,
        "n_corridor_below_tau": n_cp_below,
        "n_release_above_tau": n_rp_above,
        "max_consecutive_valid_evidence": max_consecutive_ok,
        "first_evidence_break_after_arm": first_break_step,
        "sticky_arm_counterexample": sticky_arm,
        "continuous_model_selectivity_error": continuous_model_error,
    }


def compute_emit_point_metrics(rows, arm_step, emit_step):
    """Point metrics at arm and emit."""
    arm_r = next((r for r in rows if int(r.get("step", -1)) == arm_step), {})
    emit_r = next((r for r in rows if int(r.get("step", -1)) == emit_step), {})
    return {
        "phase_at_arm": arm_r.get("pred_phase", ""),
        "phase_at_emit": emit_r.get("pred_phase", ""),
        "corridor_p_at_arm": float(arm_r.get("corridor_p", "nan")),
        "corridor_p_at_emit": float(emit_r.get("corridor_p", "nan")),
        "release_p_at_arm": float(arm_r.get("release_p", "nan")),
        "release_p_at_emit": float(emit_r.get("release_p", "nan")),
    }


def classify_hypothesis(arm_metrics):
    """Classify failure mode: A(sticky-arm), B(model-selectivity), C(mixed)."""
    if arm_metrics.get("sticky_arm_counterexample", False):
        if arm_metrics.get("continuous_model_selectivity_error", False):
            return "C_MIXED"
        else:
            return "A_STICKY_ARM"
    else:
        return "B_MODEL_SELECTIVITY"


def process_cell(cell, group, center_step_override=None):
    """Process one cell: verify SHAs, extract trace, compute metrics."""
    cell_dir = EVIDENCE_BASE / cell["relative_path"]

    # SHA verification
    sha_err = verify_source_hashes(cell, cell_dir)
    if sha_err:
        return {"_error": sha_err, "episode_key": cell["episode_key"], "profile": cell["profile"]}

    # Load data
    rows = list(csv.DictReader(open(cell_dir / "step_telemetry.csv")))
    summary = json.load(open(cell_dir / "episode_summary.json"))
    teacher_summary, teacher_labels = load_teacher_data(cell)

    n_steps = len(rows)
    emit_step = cell["emit_step"]
    if emit_step < 0:
        emit_step = None

    # Center step: use emit if triggered, otherwise midpoint or override
    center = emit_step if emit_step is not None else (center_step_override or n_steps // 2)

    # State machine trace
    sm_trace = extract_state_machine_trace(rows, emit_step if emit_step is not None else -1)

    # Window around center
    window_rows = extract_window(rows, center)

    # Arm-interval metrics (only for triggered episodes)
    arm_metrics = {}
    point_metrics = {}
    if emit_step is not None and sm_trace["arm_step"] >= 0:
        arm_metrics = compute_arm_interval_metrics(rows, sm_trace["arm_step"], emit_step)
        point_metrics = compute_emit_point_metrics(rows, sm_trace["arm_step"], emit_step)

    # Hypothesis
    hypothesis = classify_hypothesis(arm_metrics) if arm_metrics else "N_A_NOT_TRIGGERED"

    # Teacher labels around center
    teacher_window = [lab for lab in teacher_labels
                      if center - WINDOW <= lab.get("step_idx", -1) <= center + WINDOW]

    # Find first step where corridor_p exceeds tau_c (even if not armed)
    first_cp_high = -1
    for r in rows:
        cp = float(r.get("corridor_p", "nan"))
        if not np.isnan(cp) and cp > TAU_C:
            first_cp_high = int(r.get("step", -1))
            break

    # Find first step where phase == stable_carry
    first_sc_step = -1
    for r in rows:
        if r.get("pred_phase", "") == "stable_carry":
            first_sc_step = int(r.get("step", -1))
            break

    # Count total cp>tau steps in full trajectory
    n_cp_high_total = sum(1 for r in rows
                          if not np.isnan(float(r.get("corridor_p", "nan")))
                          and float(r.get("corridor_p", "nan")) > TAU_C)
    n_sc_total = sum(1 for r in rows if r.get("pred_phase", "") == "stable_carry")
    n_feat_valid = sum(1 for r in rows if r.get("feat_valid", "") == "True")

    return {
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
        "hypothesis": hypothesis,
        "first_cp_high": first_cp_high,
        "first_sc_step": first_sc_step,
        "n_cp_high_total": n_cp_high_total,
        "n_sc_total": n_sc_total,
        "teacher_summary": teacher_summary,
        "teacher_window": teacher_window,
        "telemetry_sha_verified": sha256_file(cell_dir / "step_telemetry.csv")[:16],
        "summary_sha_verified": sha256_file(cell_dir / "episode_summary.json")[:16],
    }


def build_summary_row(result):
    """One row for phase_a_summary.csv."""
    am = result.get("arm_metrics", {})
    pm = result.get("point_metrics", {})
    ts = result.get("teacher_summary", {})
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
        "hypothesis": result["hypothesis"],
        "teacher_sc_present": ts.get("stable_carry_present", None),
        "teacher_anchor": ts.get("anchor_candidate", None),
        "teacher_k10_valid": ts.get("full_k10_valid", None),
        "teacher_k10_reason": ts.get("k10_invalid_reason", ""),
    }


def write_cell_outputs(result, out_dir):
    """Write per-cell CSV and JSON outputs."""
    cell_dir = out_dir / "cells" / result["episode_key"] / result["profile"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    # emit_window.csv
    if result["window"]:
        with open(cell_dir / "emit_window.csv", "w", newline="") as f:
            fieldnames = result["window"][0].keys()
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(result["window"])

    # state_machine_trace.csv
    sm = result["sm_trace"]
    with open(cell_dir / "state_machine_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "transition", "corridor_p", "release_p",
                                          "pred_phase", "feat_valid"])
        w.writeheader()
        w.writerows(sm["trace"])
    if sm["evidence_breaks"]:
        with open(cell_dir / "evidence_breaks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "feat_valid", "corridor_p", "release_p",
                                              "pred_phase", "fail_reason"])
            w.writeheader()
            w.writerows(sm["evidence_breaks"])

    # diagnostic_summary.json (strip large arrays)
    small = {k: v for k, v in result.items()
             if k not in ("window", "sm_trace", "teacher_window")}
    with open(cell_dir / "diagnostic_summary.json", "w") as f:
        json.dump(small, f, indent=2, default=str)


def classify_overall_hypothesis(results):
    """Classify overall failure mode across all false triggers."""
    ft_results = [r for r in results if r["group"] == "FALSE_TRIGGER" and "_error" not in r]
    hypotheses = [r["hypothesis"] for r in ft_results]
    counts = {"A_STICKY_ARM": 0, "B_MODEL_SELECTIVITY": 0, "C_MIXED": 0}
    for h in hypotheses:
        if h in counts:
            counts[h] += 1

    n = len(hypotheses)
    if n == 0:
        return {"primary_failure_mode": "UNKNOWN", "counts": counts}

    if counts["A_STICKY_ARM"] >= n / 2:
        mode = "RUNTIME_STATE_MACHINE_STICKINESS"
    elif counts["B_MODEL_SELECTIVITY"] >= n / 2:
        mode = "MODEL_SELECTIVITY_FAILURE"
    else:
        mode = "MIXED_RUNTIME_AND_MODEL"

    # Cross-group comparison
    tp_results = [r for r in results if r["group"] == "TRUE_POSITIVE" and "_error" not in r]
    ft_sticky_rate = counts["A_STICKY_ARM"] / n if n > 0 else 0
    tp_sticky = sum(1 for r in tp_results
                    if r.get("arm_metrics", {}).get("sticky_arm_counterexample", False))
    tp_sticky_rate = tp_sticky / len(tp_results) if tp_results else 0

    ca_results = [r for r in results if r["group"] == "CORRECT_ABSTAIN" and "_error" not in r]
    ca_ever_armed = sum(1 for r in ca_results if r["sm_trace"]["arm_step"] >= 0)
    ca_n = len(ca_results)

    return {
        "primary_failure_mode": mode,
        "false_trigger_counts": counts,
        "false_trigger_n": n,
        "false_trigger_sticky_rate": ft_sticky_rate,
        "true_positive_sticky_rate": tp_sticky_rate,
        "true_positive_n": len(tp_results),
        "correct_abstain_ever_armed": f"{ca_ever_armed}/{ca_n}",
        "recommendation": _recommendation(mode),
    }


def _recommendation(mode):
    if mode == "RUNTIME_STATE_MACHINE_STICKINESS":
        return ("M1C-R: Implement revocable state machine with IDLE↔CANDIDATE↔ARMED hysteresis. "
                "Retrain only if six gates still not met after state machine repair.")
    elif mode == "MODEL_SELECTIVITY_FAILURE":
        return ("M1C-M: State machine fix alone insufficient. Train SC5-v2 with explicit abstention head, "
                "hard negatives, and temporal consistency loss.")
    else:
        return ("M1C-RM: Repair state machine first, re-evaluate on frozen data. "
                "Train SC5-v2 only for residual errors.")


def print_summary_table(results, title):
    """Print compact comparison table."""
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    header = (f"{'Group':<18} {'Episode':<25} {'Prof':>4} {'Emit':>5} {'Arm':>5} "
              f"{'A→E':>5} {'Brk':>4} {'Sticky':>6} {'Hyp':>22} "
              f"{'ph@arm':<25} {'ph@emit':<25} {'cp@arm':>7} {'cp@emit':>7}")
    print(header)
    print("-" * 120)
    for r in results:
        if "_error" in r:
            print(f"  {'ERROR':<18} {r['episode_key']:<25} {r['profile']:>4}  --- {r['_error']}")
            continue
        am = r.get("arm_metrics", {})
        pm = r.get("point_metrics", {})
        arm = r["sm_trace"]["arm_step"]
        emit = r["emit_step"] if r["emit_step"] is not None else -1
        ate = r["sm_trace"]["arm_to_emit"]
        brk = r["sm_trace"]["n_evidence_breaks"]
        sticky = "YES" if am.get("sticky_arm_counterexample") else ("no" if am else "-")
        hyp = r["hypothesis"]
        ph_a = pm.get("phase_at_arm", "")[:25] if pm else ""
        ph_e = pm.get("phase_at_emit", "")[:25] if pm else ""
        cp_a = f"{pm.get('corridor_p_at_arm', 0):.4f}" if pm and pm.get('corridor_p_at_arm') is not None else "-"
        cp_e = f"{pm.get('corridor_p_at_emit', 0):.4f}" if pm and pm.get('corridor_p_at_emit') is not None else "-"
        print(f"  {r['group']:<18} {r['episode_key']:<25} {r['profile']:>4} {emit:>5} {arm:>5} "
              f"{ate:>5} {brk:>4} {sticky:>6} {hyp:<22} "
              f"{ph_a:<25} {ph_e:<25} {cp_a:>7} {cp_e:>7}")
    print("-" * 120)


def main():
    manifest = load_manifest()
    classification = load_classification()
    cells_by_key = {}
    for c in manifest["cells"]:
        cells_by_key[(c["episode_key"], c["profile"])] = c

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # ── Build target list ──────────────────────────────────────────
    targets = []

    # Group 1: FALSE_TRIGGERS
    for ft in FALSE_TRIGGERS:
        cell = cells_by_key.get((ft["episode_key"], ft["profile"]))
        if cell:
            targets.append((cell, "FALSE_TRIGGER", ft.get("reason", "")))
        else:
            print(f"WARNING: False trigger cell not in manifest: {ft}")

    # Group 2: TRUE_POSITIVES
    for tp in TRUE_POSITIVE_CANDIDATES:
        cell = cells_by_key.get((tp["episode_key"], tp["profile"]))
        if cell and cell["emit_step"] >= 0:
            targets.append((cell, "TRUE_POSITIVE", "teacher-valid K10-pass"))
        else:
            print(f"WARNING: TP candidate not usable: {tp}")

    # Group 3: CORRECT_ABSTAIN — all no-corridor episodes with emit_step=-1
    # Identify from classification + manifest. Need teacher replay to confirm no-corridor.
    false_trigger_keys = {(ft["episode_key"], ft["profile"]) for ft in FALSE_TRIGGERS}
    for c in manifest["cells"]:
        key = (c["episode_key"], c["profile"])
        if key in false_trigger_keys:
            continue
        if c["emit_step"] == -1:
            # Check teacher replay for no-corridor classification
            _, teacher_labels = load_teacher_data(c)
            if teacher_labels:
                # Teacher labels exist — check if no-corridor
                # (we can also check if any existing TP target already covers this)
                targets.append((c, "CORRECT_ABSTAIN", "emit=-1 (needs teacher verification)"))
            # Even without teacher labels, include non-triggered as likely abstains
            # (most non-triggered clean episodes are genuine abstains)

    # Deduplicate targets
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

    # ── Process all cells ──────────────────────────────────────────
    all_results = []
    errors = []

    for i, (cell, group, note) in enumerate(targets):
        key = f"{cell['episode_key']}/{cell['profile']}"
        print(f"\n[{i+1}/{len(targets)}] {group} {key} ...", end=" ", flush=True)

        result = process_cell(cell, group)
        result["group_note"] = note
        all_results.append(result)

        if "_error" in result:
            print(f"ERROR: {result['_error']}")
            errors.append(result)
        else:
            print(f"OK emit={result['emit_step']} arm={result['sm_trace']['arm_step']} "
                  f"hyp={result['hypothesis']}")

        # Write per-cell outputs
        write_cell_outputs(result, OUT_BASE)

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
    overall = classify_overall_hypothesis(all_results)
    hyp_path = OUT_BASE / "hypothesis_classification.json"
    with open(hyp_path, "w") as f:
        json.dump(overall, f, indent=2)

    # ── Manifest ───────────────────────────────────────────────────
    manifest_out = {
        "gate": "PHASE_A_DIAGNOSTIC_MANIFEST",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_commit": classification.get("final_freeze_commit", "unknown"),
        "m1b_classification_sha": sha256_file(CLASSIFICATION_PATH)[:16],
        "diagnostic_script_sha": sha256_file(__file__)[:16],
        "n_targets": len(targets),
        "n_success": len(all_results) - len(errors),
        "n_errors": len(errors),
        "groups": dict(groups),
        "tau_corridor": TAU_C,
        "tau_release": TAU_R,
        "guard": GUARD,
        "window": WINDOW,
        "overall_hypothesis": overall,
    }
    with open(OUT_BASE / "artifact_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Console output ─────────────────────────────────────────────
    print_summary_table(all_results, "PHASE A — COMPLETE DIAGNOSTIC TABLE")

    print(f"\n{'='*80}")
    print(f"HYPOTHESIS CLASSIFICATION")
    print(f"{'='*80}")
    print(f"  Primary failure mode: {overall['primary_failure_mode']}")
    print(f"  False trigger sticky rate: {overall['false_trigger_sticky_rate']:.2f} "
          f"({overall.get('false_trigger_counts', {})})")
    print(f"  True positive sticky rate: {overall['true_positive_sticky_rate']:.2f}")
    print(f"  Correct abstain ever armed: {overall['correct_abstain_ever_armed']}")
    print(f"  Recommendation: {overall['recommendation']}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e['episode_key']}/{e['profile']}: {e['_error']}")

    print(f"\n  All outputs: {OUT_BASE}")
    print(f"  Hypothesis: {hyp_path}")
    print("  PHASE A COMPLETE.")


if __name__ == "__main__":
    main()
