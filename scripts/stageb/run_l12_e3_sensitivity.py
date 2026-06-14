#!/usr/bin/env python3
"""E3: Frozen sensitivity analysis and failure taxonomy.

Sweeps preregistered Student parameter grid over hash-frozen development
traces. Teacher-P, remapper, and feature definitions are completely frozen.
Reports sensitivity surface and per-trace failure decomposition.

CPU only. No GPU, no attack, no Layer3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))

import yaml
from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.window_contract import WindowProposal, validate_proposals
from gripper_attack.phase_detector import (
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
    check_teacher_p_privilege_capability,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
    _detect_ambiguous_multiple_closes,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

SELECTOR_COMMIT = "0f72fda"


def _git_rev_parse() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=str(REPO_ROOT), text=True).strip()
        return out == ""
    except Exception:
        return False


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_offline_failure(preds, win, teacher_p_anchor, teacher_p_available):
    """Classify offline selector outcome into taxonomy."""
    abstain = win.get("abstain_reason", "")
    anchor = win.get("anchor_step", -1)

    if not teacher_p_available:
        return "teacher_reference_unavailable"
    if abstain == "ambiguous_multiple_close_candidates":
        return "ambiguous_multiple_close"
    if abstain == "all_abstain":
        return "no_candidate"
    if abstain:
        return f"other_abstain_{abstain}"

    # Eligible
    err = abs(anchor - teacher_p_anchor)
    if err <= 4:
        return "correct_or_near_teacher_anchor"
    if anchor < teacher_p_anchor:
        return "early_selection"
    return "late_selection"


def _classify_online_failure(win, teacher_p_anchor, teacher_p_available):
    """Classify online trigger outcome."""
    trigger = win.get("trigger_step", -1)
    abstain = win.get("abstain_reason", "")

    if not teacher_p_available:
        return "teacher_reference_unavailable"
    if abstain == "no_online_trigger":
        return "no_trigger"
    if abstain:
        return f"other_abstain_{abstain}"

    err = abs(trigger - teacher_p_anchor)
    if err <= 4:
        return "correct_or_near_teacher_anchor"
    if trigger < teacher_p_anchor:
        # Check if trigger is on a definitely spurious early close
        if teacher_p_anchor - trigger > 20:
            return "early_spurious_trigger"
        return "early_trigger"
    if trigger > teacher_p_anchor + 4:
        return "late_trigger"
    return "correct_or_near_teacher_anchor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--grid-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None)
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse()
    RUNNER_SHORT = RUNNER_COMMIT[:7]

    # Preflight
    if not _git_is_clean():
        print("FATAL: tracked worktree is dirty")
        sys.exit(1)

    out = Path(args.output_dir)
    if out.exists():
        print(f"FATAL: output directory exists: {out}")
        sys.exit(1)
    out.mkdir(parents=True)

    # Load grid
    with open(args.grid_config) as f:
        grid_config = yaml.safe_load(f)
    grid = grid_config["grid"]
    frozen = grid_config["frozen"]

    tie_tolerances = grid["tie_tolerance"]
    min_separations = grid["min_close_separation"]
    event_floors = grid["event_score_floor"]
    online_thresholds = grid["online_threshold"]

    n_combos = len(tie_tolerances) * len(min_separations) * len(event_floors) * len(online_thresholds)
    print(f"E3 Grid: {len(tie_tolerances)}×{len(min_separations)}×{len(event_floors)}×{len(online_thresholds)} = {n_combos} combinations")

    # Load traces
    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    # Verify hashes
    for mr in manifest_rows:
        actual = _sha256_file(mr["trace_path"])
        if actual != mr["expected_sha256"]:
            print(f"FATAL: SHA mismatch for {mr['task_key']}_s{mr['state_id']}")
            sys.exit(1)

    # Remap all traces once (Teacher-P is frozen, Student params vary)
    trace_data = []
    for mr in manifest_rows:
        remap_out = str(out / f"remap_{mr['task_key']}_s{mr['state_id']}.csv")
        l12_rows, inv_issues, field_issues = remap_v4_to_l12(
            mr["trace_path"], remap_out, raise_on_invariant=False)
        anchor_p = teacher_privileged_critical_close_anchor(l12_rows)
        anchor_r = teacher_rule_critical_close_anchor(l12_rows)
        cap = check_teacher_p_privilege_capability(l12_rows)
        trace_data.append({
            "task_key": mr["task_key"],
            "state_id": mr["state_id"],
            "l12_rows": l12_rows,
            "teacher_p_anchor": anchor_p,
            "teacher_r_anchor": anchor_r,
            "teacher_p_available": anchor_p >= 0,
            "grasp_privilege_valid": cap["grasp_privilege_valid"],
            "n_steps": len(l12_rows),
        })

    # ── Run grid sweep ──
    sensitivity_rows = []
    failure_rows = []
    candidate_audit_rows = []

    combo_idx = 0
    for tie_tol, min_sep, ev_floor, on_thresh in itertools.product(
        tie_tolerances, min_separations, event_floors, online_thresholds
    ):
        combo_idx += 1
        if combo_idx % 20 == 0:
            print(f"  combo {combo_idx}/{n_combos}...")

        combo_stats = {
            "tie_tolerance": tie_tol,
            "min_close_separation": min_sep,
            "event_score_floor": ev_floor,
            "online_threshold": on_thresh,
        }

        n_p_available = 0
        n_off_eligible = 0
        n_off_ambiguous = 0
        n_off_all_abstain = 0
        n_off_correct = 0
        n_off_early = 0
        n_off_late = 0
        n_on_triggered = 0
        n_on_no_trigger = 0
        n_on_correct = 0
        n_on_early_spurious = 0
        n_on_early = 0
        n_on_late = 0
        sum_off_error = 0.0
        sum_on_error = 0.0
        n_off_error_count = 0
        n_on_error_count = 0

        for td in trace_data:
            rows = td["l12_rows"]
            p_anchor = td["teacher_p_anchor"]
            p_avail = td["teacher_p_available"]
            task = td["task_key"]
            state = td["state_id"]

            if p_avail:
                n_p_available += 1

            # Offline — all 3 ambiguity params passed explicitly
            preds_off = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                                    teacher_anchor=p_anchor if p_avail else -1)
            win_off = select_best_window(
                preds_off, WINDOW_LEN, PRE_OFFSET,
                tie_tolerance=tie_tol,
                min_separation=min_sep,
                event_score_floor=ev_floor,
            )

            # Online
            win_on = select_online_trigger(preds_off, score_threshold=on_thresh,
                                           confirmation_steps=1, mode="close_interception")

            # Classify offline
            off_cat = _classify_offline_failure(preds_off, win_off, p_anchor, p_avail)
            if off_cat == "correct_or_near_teacher_anchor":
                n_off_correct += 1
                n_off_eligible += 1
                n_off_error_count += 1
                sum_off_error += abs(win_off["anchor_step"] - p_anchor)
            elif off_cat == "ambiguous_multiple_close":
                n_off_ambiguous += 1
            elif off_cat == "no_candidate":
                n_off_all_abstain += 1
            elif off_cat.startswith("early"):
                n_off_early += 1
                n_off_eligible += 1
                n_off_error_count += 1
                sum_off_error += abs(win_off["anchor_step"] - p_anchor)
            elif off_cat.startswith("late"):
                n_off_late += 1
                n_off_eligible += 1
                n_off_error_count += 1
                sum_off_error += abs(win_off["anchor_step"] - p_anchor)
            elif off_cat == "teacher_reference_unavailable":
                pass  # not counted in P-available stats

            # Classify online
            on_cat = _classify_online_failure(win_on, p_anchor, p_avail)
            if on_cat == "correct_or_near_teacher_anchor":
                n_on_correct += 1
                n_on_triggered += 1
                n_on_error_count += 1
                sum_on_error += abs(win_on.get("trigger_step", -1) - p_anchor)
            elif on_cat == "no_trigger":
                n_on_no_trigger += 1
            elif on_cat == "early_spurious_trigger":
                n_on_early_spurious += 1
                n_on_triggered += 1
            elif on_cat == "early_trigger":
                n_on_early += 1
                n_on_triggered += 1
            elif on_cat == "late_trigger":
                n_on_late += 1
                n_on_triggered += 1
            elif on_cat == "teacher_reference_unavailable":
                pass

            # Per-trace failure row for this combo
            failure_rows.append({
                "tie_tolerance": tie_tol, "min_close_separation": min_sep,
                "event_score_floor": ev_floor, "online_threshold": on_thresh,
                "task_key": task, "state_id": state,
                "teacher_p_available": p_avail,
                "teacher_p_anchor": p_anchor,
                "offline_category": off_cat,
                "offline_anchor": win_off.get("anchor_step", -1),
                "online_category": on_cat,
                "online_trigger": win_on.get("trigger_step", -1),
            })

        n_traces = len(trace_data)
        combo_stats.update({
            "n_traces": n_traces,
            "n_p_available": n_p_available,
            "off_eligible_pct": round(n_off_eligible / max(1, n_p_available) * 100, 1),
            "off_ambiguous_pct": round(n_off_ambiguous / max(1, n_p_available) * 100, 1),
            "off_all_abstain_pct": round(n_off_all_abstain / max(1, n_p_available) * 100, 1),
            "off_correct_pct": round(n_off_correct / max(1, n_p_available) * 100, 1),
            "off_mae": round(sum_off_error / max(1, n_off_error_count), 2),
            "on_triggered_pct": round(n_on_triggered / max(1, n_p_available) * 100, 1),
            "on_correct_pct": round(n_on_correct / max(1, n_p_available) * 100, 1),
            "on_early_spurious_pct": round(n_on_early_spurious / max(1, n_p_available) * 100, 1),
            "on_no_trigger_pct": round(n_on_no_trigger / max(1, n_p_available) * 100, 1),
        })
        sensitivity_rows.append(combo_stats)

    # ── Candidate-level score audit (at default params) ──
    from gripper_attack.critical_close_selector import TIE_TOLERANCE as _DEF_TIE, \
        MIN_CLOSE_SEPARATION as _DEF_SEP, EVENT_SCORE_FLOOR as _DEF_FLOOR
    for td in trace_data:
        rows = td["l12_rows"]
        p_anchor = td["teacher_p_anchor"]
        r_anchor = td["teacher_r_anchor"]
        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=p_anchor if td["teacher_p_available"] else -1)
        win_on_def = select_online_trigger(preds, mode="close_interception")
        first_trigger = win_on_def.get("trigger_step", -1)

        for p in preds:
            if p.get("is_close_event_candidate") and not p.get("abstain"):
                step = p["step"]
                candidate_audit_rows.append({
                    "task_key": td["task_key"],
                    "state_id": td["state_id"],
                    "candidate_step": step,
                    "is_teacher_p_anchor": int(step == p_anchor) if td["teacher_p_available"] else 0,
                    "is_teacher_r_anchor": int(step == r_anchor),
                    "is_online_first_trigger": int(step == first_trigger),
                    "total_score": p["score"],
                    "raw_open_to_close_crossing": int(p.get("raw_open_to_close_crossing", 0)),
                    "close_onset": p.get("close_onset", 0),
                    "close_streak": p.get("close_streak", 0) if p.get("close_streak") != "" else 0,
                    "qpos": str(p.get("qpos", ""))[:8],
                    "disabled_features": ",".join(p.get("disabled_features", [])),
                    "distance_to_teacher_p": step - p_anchor if td["teacher_p_available"] else "",
                    "abstain": p.get("abstain", ""),
                })

    # ── Write outputs ──
    def _write_csv(rows, name):
        if not rows:
            return
        path = out / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(sensitivity_rows, "l12_e3_sensitivity_surface.csv")
    _write_csv(failure_rows, "l12_e3_failure_taxonomy.csv")
    _write_csv(candidate_audit_rows, "l12_e3_candidate_score_audit.csv")

    # Best combo by offline correct
    best_off = max(sensitivity_rows, key=lambda r: (r["off_correct_pct"], -r["off_ambiguous_pct"]))
    best_on = max(sensitivity_rows, key=lambda r: (r["on_correct_pct"], -r["on_early_spurious_pct"]))

    # ── Summary ──
    print(f"\n=== E3 SENSITIVITY SUMMARY ===")
    print(f"Combinations: {n_combos}")
    print(f"Traces: {len(trace_data)} (P-available: {sum(1 for t in trace_data if t['teacher_p_available'])})")
    print(f"\nBest offline: tie={best_off['tie_tolerance']} sep={best_off['min_close_separation']} floor={best_off['event_score_floor']}")
    print(f"  eligible={best_off['off_eligible_pct']}% correct={best_off['off_correct_pct']}% ambiguous={best_off['off_ambiguous_pct']}% MAE={best_off['off_mae']}")
    print(f"\nBest online: tie={best_on['tie_tolerance']} sep={best_on['min_close_separation']} floor={best_on['event_score_floor']} thresh={best_on['online_threshold']}")
    print(f"  triggered={best_on['on_triggered_pct']}% correct={best_on['on_correct_pct']}% early_spurious={best_on['on_early_spurious_pct']}%")

    # Sensitivity analysis: how much does each parameter affect off_correct?
    for param_name, param_values in [
        ("tie_tolerance", tie_tolerances),
        ("min_close_separation", min_separations),
        ("event_score_floor", event_floors),
    ]:
        print(f"\n  {param_name} sweep (other params median):")
        for pv in param_values:
            subset = [r for r in sensitivity_rows if r[param_name] == pv]
            if subset:
                avg_correct = sum(r["off_correct_pct"] for r in subset) / len(subset)
                avg_ambig = sum(r["off_ambiguous_pct"] for r in subset) / len(subset)
                print(f"    {param_name}={pv}: correct={avg_correct:.1f}% ambig={avg_ambig:.1f}%")

    # ── Run log ──
    log_path = out / "l12_e3_run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"E3 RUN LOG\n")
        f.write(f"runner_commit: {RUNNER_COMMIT}\n")
        f.write(f"selector_commit: {SELECTOR_COMMIT}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"grid_config: {args.grid_config}\n")
        f.write(f"n_combinations: {n_combos}\n")
        f.write(f"best_offline: tie={best_off['tie_tolerance']} sep={best_off['min_close_separation']} floor={best_off['event_score_floor']} correct={best_off['off_correct_pct']}%\n")
        f.write(f"best_online: tie={best_on['tie_tolerance']} sep={best_on['min_close_separation']} floor={best_on['event_score_floor']} thresh={best_on['online_threshold']} correct={best_on['on_correct_pct']}%\n")

    # ── Tracked artifacts ──
    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in out.glob("*.csv"):
            shutil.copy2(f, tracked / f.name)
        shutil.copy2(log_path, tracked / "l12_e3_run_log.txt")
        print(f"\nTracked artifacts: {tracked}")

    print(f"\nOutput: {out}")
    print("E3 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
