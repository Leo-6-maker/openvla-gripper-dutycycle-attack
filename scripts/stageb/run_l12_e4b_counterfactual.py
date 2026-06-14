#!/usr/bin/env python3
"""E4B: Causal policy counterfactual + candidate separability audit.

E4B-A: 5 offline causal trigger policies replayed over frozen traces:
  1. first_threshold (current baseline)
  2. bounded_peak_hold (wait K steps, pick max)
  3. local_maximum (trigger on first score decline after threshold)
  4. score_margin (candidate must beat running peak by margin)
  5. close_event_only_peak (same as peak-hold but only among close events)

E4B-B: Per-CLOSE-candidate separability features table.

CPU only. Same 12 frozen traces. No model training.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))

from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.phase_detector import (
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

SCORE_THRESHOLD = 1.5
NEAR_THRESHOLD = 4


def _sha256_file(path):
    import hashlib
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev_parse():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean():
    try:
        return subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=str(REPO_ROOT), text=True).strip() == ""
    except Exception:
        return False


# ── Policy implementations (offline replay, causal step-by-step) ──

def policy_first_threshold(preds, threshold=SCORE_THRESHOLD):
    """Current baseline: trigger on first step with score >= threshold."""
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold:
            return {"trigger_step": p["step"], "delay": 0, "policy": "first_threshold"}
    return {"trigger_step": -1, "delay": -1, "policy": "first_threshold"}


def policy_bounded_peak_hold(preds, threshold=SCORE_THRESHOLD, hold_steps=4):
    """Wait hold_steps after first threshold crossing, then pick max score in window."""
    first_t = None
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold:
            first_t = p["step"]
            break
    if first_t is None:
        return {"trigger_step": -1, "delay": -1, "policy": f"peak_hold_{hold_steps}"}

    window = [p for p in preds if first_t <= p["step"] <= first_t + hold_steps and not p["abstain"]]
    if not window:
        return {"trigger_step": first_t, "delay": 0, "policy": f"peak_hold_{hold_steps}"}
    best = max(window, key=lambda p: p["score"])
    return {"trigger_step": best["step"], "delay": best["step"] - first_t,
            "policy": f"peak_hold_{hold_steps}"}


def policy_local_maximum(preds, threshold=SCORE_THRESHOLD):
    """Trigger when score first declines after crossing threshold."""
    above = False
    for i, p in enumerate(preds):
        if not p["abstain"] and p["score"] >= threshold:
            above = True
        if above and not p["abstain"]:
            # Check if next step declines
            if i + 1 < len(preds) and not preds[i + 1]["abstain"]:
                if preds[i + 1]["score"] < p["score"]:
                    return {"trigger_step": p["step"], "delay": p["step"] - _first_above(preds, threshold),
                            "policy": "local_maximum"}
    # Fallback: last above-threshold step
    for p in reversed(preds):
        if not p["abstain"] and p["score"] >= threshold:
            return {"trigger_step": p["step"], "delay": p["step"] - _first_above(preds, threshold),
                    "policy": "local_maximum"}
    return {"trigger_step": -1, "delay": -1, "policy": "local_maximum"}


def policy_score_margin(preds, threshold=SCORE_THRESHOLD, margin=0.3):
    """Trigger when a new candidate exceeds the running peak by margin."""
    running_peak = 0.0
    running_step = -1
    first_t = None
    for p in preds:
        if p["abstain"]:
            continue
        if first_t is None and p["score"] >= threshold:
            first_t = p["step"]
        if p["score"] > running_peak + margin:
            running_peak = p["score"]
            running_step = p["step"]
    if running_step >= 0 and first_t is not None:
        return {"trigger_step": running_step, "delay": running_step - first_t,
                "policy": f"score_margin_{margin}"}
    return {"trigger_step": -1, "delay": -1, "policy": f"score_margin_{margin}"}


def policy_close_event_peak(preds, threshold=SCORE_THRESHOLD, hold_steps=4):
    """Like peak_hold but only considers is_close_event_candidate steps."""
    first_t = None
    for p in preds:
        if p.get("is_close_event_candidate") and not p["abstain"] and p["score"] >= threshold:
            first_t = p["step"]
            break
    if first_t is None:
        return {"trigger_step": -1, "delay": -1, "policy": f"close_peak_{hold_steps}"}

    window = [p for p in preds
              if first_t <= p["step"] <= first_t + hold_steps
              and p.get("is_close_event_candidate")
              and not p["abstain"]]
    if not window:
        return {"trigger_step": first_t, "delay": 0, "policy": f"close_peak_{hold_steps}"}
    best = max(window, key=lambda p: p["score"])
    return {"trigger_step": best["step"], "delay": best["step"] - first_t,
            "policy": f"close_peak_{hold_steps}"}


def _first_above(preds, threshold):
    for p in preds:
        if not p["abstain"] and p["score"] >= threshold:
            return p["step"]
    return -1


# Noncausal upper bound
def policy_non_causal_oracle(preds):
    """Best close-event candidate across full trajectory (not causal)."""
    close = [p for p in preds if p.get("is_close_event_candidate") and not p["abstain"]]
    if not close:
        return {"trigger_step": -1, "delay": -1, "policy": "noncausal_oracle"}
    best = max(close, key=lambda p: p["score"])
    return {"trigger_step": best["step"], "delay": -1, "policy": "noncausal_oracle"}


POLICIES = [
    ("first_threshold", lambda preds: policy_first_threshold(preds)),
    ("peak_hold_2", lambda preds: policy_bounded_peak_hold(preds, hold_steps=2)),
    ("peak_hold_4", lambda preds: policy_bounded_peak_hold(preds, hold_steps=4)),
    ("peak_hold_8", lambda preds: policy_bounded_peak_hold(preds, hold_steps=8)),
    ("local_maximum", lambda preds: policy_local_maximum(preds)),
    ("score_margin_0.3", lambda preds: policy_score_margin(preds, margin=0.3)),
    ("close_event_peak_4", lambda preds: policy_close_event_peak(preds, hold_steps=4)),
    ("noncausal_oracle", lambda preds: policy_non_causal_oracle(preds)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None)
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse()
    START_TIME = datetime.now(timezone.utc)

    if not _git_is_clean():
        print("FATAL: tracked worktree dirty")
        sys.exit(1)

    out = Path(args.output_dir)
    if out.exists():
        print(f"FATAL: output dir exists: {out}")
        sys.exit(1)
    out.mkdir(parents=True)

    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    for mr in manifest_rows:
        if _sha256_file(mr["trace_path"]) != mr["expected_sha256"]:
            print(f"FATAL: SHA mismatch {mr['task_key']}_s{mr['state_id']}")
            sys.exit(1)

    policy_rows = []
    separability_rows = []
    invariant_total = 0

    for mr in manifest_rows:
        task = mr["task_key"]
        state = int(mr["state_id"])
        print(f"  {task}_s{state}")

        remap_out = str(out / f"remap_{task}_s{state}.csv")
        l12_rows, inv, fi = remap_v4_to_l12(mr["trace_path"], remap_out, raise_on_invariant=False)
        invariant_total += len(inv)

        p_anchor = teacher_privileged_critical_close_anchor(l12_rows)
        r_anchor = teacher_rule_critical_close_anchor(l12_rows)
        p_avail = p_anchor >= 0

        preds = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if p_avail else -1)

        # ── E4B-A: Policy counterfactuals ──
        for pol_name, pol_fn in POLICIES:
            result = pol_fn(preds)
            trigger = result["trigger_step"]
            err = abs(trigger - p_anchor) if p_avail and trigger >= 0 else ""
            near = int(abs(trigger - p_anchor) <= NEAR_THRESHOLD) if p_avail and trigger >= 0 else ""
            policy_rows.append({
                "task_key": task, "state_id": state,
                "teacher_p_available": p_avail,
                "teacher_p_anchor": p_anchor,
                "policy": result["policy"],
                "trigger_step": trigger,
                "delay_from_first": result.get("delay", ""),
                "abs_error_vs_P": err,
                "is_near_P": near,
            })

        # ── E4B-B: Candidate separability ──
        close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
        for idx, c in enumerate(close_cands):
            step = c["step"]
            separability_rows.append({
                "task_key": task, "state_id": state,
                "candidate_step": step,
                "candidate_index": idx,
                "is_teacher_p": int(step == p_anchor) if p_avail else 0,
                "is_teacher_r": int(step == r_anchor),
                "total_score": c["score"],
                "raw_crossing_bonus": c.get("raw_crossing_bonus", ""),
                "close_streak_bonus": c.get("close_streak_bonus", ""),
                "close_onset_qpos_bonus": c.get("close_onset_qpos_bonus", ""),
                "eef_deceleration_bonus": c.get("eef_deceleration_bonus", ""),
                "qpos_ready_bonus": c.get("qpos_ready_bonus", ""),
                "eef_speed_now": c.get("eef_speed_now", ""),
                "eef_speed_prev": c.get("eef_speed_prev", ""),
                "close_onset": c.get("close_onset", ""),
                "close_streak": c.get("close_streak", ""),
                "qpos": str(c.get("qpos", ""))[:10],
                "raw_crossing": int(c.get("raw_open_to_close_crossing", 0)),
                "time_since_prev_close": (step - close_cands[idx - 1]["step"]) if idx > 0 else "",
                "disabled_features": ",".join(c.get("disabled_features", [])),
            })

    # ── Write tables ──
    def _wcsv(rows, name):
        if not rows:
            return
        p = out / name
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _wcsv(policy_rows, "l12_e4b_policy_counterfactuals.csv")
    _wcsv(separability_rows, "l12_e4b_candidate_separability.csv")

    # ── Summary ──
    n_p_avail = len(set((r["task_key"], r["state_id"]) for r in policy_rows if r["teacher_p_available"]))
    print(f"\n=== E4B POLICY COUNTERFACTUAL ===")
    print(f"Traces: {len(manifest_rows)} (P-available: {n_p_avail})")
    for pol_name, _ in POLICIES:
        subset = [r for r in policy_rows if r["policy"] == pol_name and r["teacher_p_available"]]
        n_correct = sum(1 for r in subset if r["is_near_P"] == 1)
        avg_delay = sum(r["delay_from_first"] for r in subset if isinstance(r["delay_from_first"], (int, float)) and r["delay_from_first"] >= 0) / max(1, len(subset))
        print(f"  {pol_name:20s}: correct={n_correct}/{len(subset)} avg_delay={avg_delay:.1f}")

    # ── Run log ──
    log_path = out / "l12_e4b_run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"E4B RUN LOG\nrunner_commit: {RUNNER_COMMIT}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"start: {START_TIME.isoformat()}\n")
        f.write(f"invariant_violations: {invariant_total}\n")
        f.write(f"policies: {len(POLICIES)}\nworktree_clean: {_git_is_clean()}\n")

    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"):
            shutil.copy2(f, tracked / f.name)
        shutil.copy2(log_path, tracked / "l12_e4b_run_log.txt")

    print(f"\nOutput: {out}")
    print("E4B COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
