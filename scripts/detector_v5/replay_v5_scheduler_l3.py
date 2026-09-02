#!/usr/bin/env python3
"""V5 scheduler L3 replay for exact-W32 inner-CV predictions.

Status: TEACHER_EVENT_GATED_DIAGNOSTIC_ONLY

candidate_close is currently approximated because policy action stream
(raw gripper values) is not present in the factorized Student prediction
bundle. Authoritative runtime-causal candidate_close requires:
  - raw_gripper from policy action (action_contract.classify_openvla_raw_gripper)
  - action_intent == CLOSE (gripper < 0.5)

This script does NOT use Teacher event_id for the scheduler gate.
It produces a diagnostic decomposition of scheduler behavior,
NOT an authoritative L3 false-start rate.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import sha256_file
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def load_steps(prediction_dir: Path) -> list[dict]:
    path = prediction_dir / "heldout_step_predictions.jsonl"
    steps = []
    with open(path) as f:
        for line in f:
            steps.append(json.loads(line))
    return steps


def group_by_episode(steps: list[dict]) -> dict[str, list[dict]]:
    eps = defaultdict(list)
    for s in steps:
        eps[s["canonical_parent_key"]].append(s)
    return dict(eps)


def compute_metrics(episodes, scheduler_results, k10_window=8):
    """Compute decomposed L3 metrics per episode.

    Returns dict of metric_name -> value across all episodes.
    """
    n_total = len(episodes)
    n_negative = 0          # no valid opportunity corridor
    n_positive = 0          # has at least one valid opportunity
    n_false_start = 0       # emitted on negative episode OR emit not in any corridor
    n_valid_hit = 0         # emitted and in at least one valid corridor
    n_abstain = 0           # scheduler never emitted (on positive episode)
    n_emit_total = 0        # total episodes where scheduler emitted
    early_emits = 0
    late_emits = 0

    for ep_key, ep_steps in episodes.items():
        result = scheduler_results.get(ep_key, {})
        emitted = result.get("emitted", False)
        emit_step = result.get("emit_step", -1)

        # Identify valid opportunity corridors from K10 labels
        corridors = _find_k10_corridors(ep_steps, k10_window)

        if not corridors:
            n_negative += 1
            if emitted:
                n_false_start += 1
        else:
            n_positive += 1
            if emitted:
                n_emit_total += 1
                hit = any(c["start"] <= emit_step <= c["end"] for c in corridors)
                if hit:
                    n_valid_hit += 1
                else:
                    n_false_start += 1
                    # Check if emit is early (before first corridor) or late (after last)
                    first_start = min(c["start"] for c in corridors)
                    last_end = max(c["end"] for c in corridors)
                    if emit_step < first_start:
                        early_emits += 1
                    elif emit_step > last_end:
                        late_emits += 1
            else:
                n_abstain += 1

    return {
        "total_episodes": n_total,
        "negative_episodes": n_negative,
        "positive_episodes": n_positive,
        "scheduler_emitted_total": n_emit_total,
        "false_start_episodes": n_false_start,
        "valid_opportunity_hits": n_valid_hit,
        "abstention_episodes": n_abstain,
        "early_emits": early_emits,
        "late_emits": late_emits,
        # Rates
        "negative_episode_false_start_rate": n_false_start / max(1, n_negative) if n_negative > 0 else 0.0,
        "valid_opportunity_recall": n_valid_hit / max(1, n_positive) if n_positive > 0 else 0.0,
        "emit_precision": n_valid_hit / max(1, n_emit_total) if n_emit_total > 0 else 0.0,
        "abstention_rate": n_abstain / max(1, n_positive) if n_positive > 0 else 0.0,
        "overall_emit_rate": n_emit_total / max(1, n_total),
    }


def _find_k10_corridors(ep_steps: list[dict], k10_window: int = 8) -> list[dict]:
    """Find valid opportunity corridors from K10 feasible steps.

    A corridor is a contiguous block of K10 feasible steps of length >= k10_window.
    K10 feasibility is from the Teacher label — this is EVALUATION-ONLY,
    not a runtime gate.
    """
    ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
    corridors = []
    i = 0
    while i < len(ep_sorted):
        s = ep_sorted[i]
        # K10 feasible step: route_supported AND k10 label indicates feasibility
        # For factorized predictions, we use the presence of positive event labels
        # as a proxy. The actual K10 field would need to be exported if needed.
        # Here we use: any known release target step within a release event
        if s.get("route_supported") and s.get("event_id", -1) >= 0:
            start = s["step_index"]
            j = i
            while j < len(ep_sorted) and ep_sorted[j].get("route_supported") and ep_sorted[j].get("event_id", -1) >= 0:
                if ep_sorted[j]["step_index"] != start + (j - i):
                    break
                j += 1
            length = j - i
            if length >= k10_window:
                corridors.append({"start": start, "end": ep_sorted[j-1]["step_index"], "length": length})
            i = j
        else:
            i += 1
    return corridors


def replay_scheduler(episodes: dict[str, list[dict]], calibrator=None):
    """Replay V5 scheduler on all episodes.

    candidate_close approximation:
      route_supported AND event_id >= 0  (TEACHER_EVENT_GATED DIAGNOSTIC)
      This is NOT runtime-causal — see module docstring.

    calibrator: optional dict with per-head (a, b) Platt params.
    """
    config = V5SchedulerConfig()
    a_g, b_g = calibrator.get("grasp_a", 1.0), calibrator.get("grasp_b", 0.0) if calibrator else (1.0, 0.0)
    a_r, b_r = calibrator.get("release_a", 1.0), calibrator.get("release_b", 0.0) if calibrator else (1.0, 0.0)

    results = {}
    for ep_key, ep_steps in sorted(episodes.items()):
        ep_sorted = sorted(ep_steps, key=lambda x: x["step_index"])
        scheduler = V5OneShotScheduler(config)
        final_result = {}

        for s in ep_sorted:
            # APPROXIMATE candidate_close — TEACHER_EVENT_GATED DIAGNOSTIC
            candidate_close = s["route_supported"] and s.get("event_id", -1) >= 0

            gp = s.get("grasp_prob", 0)
            rp = s.get("release_prob", 0)
            mp = s.get("manipulation_prob", 0)

            if calibrator:
                if 0 < gp < 1: gp = sigmoid(a_g * np.log(gp / (1 - gp)) + b_g)
                if 0 < rp < 1: rp = sigmoid(a_r * np.log(rp / (1 - rp)) + b_r)

            final_result = scheduler.update(
                step=s["step_index"],
                candidate_close=candidate_close,
                valid=s["route_supported"],
                utility_probability=gp,
                release_probability=rp,
                regrasp_probability=mp,
                uncertainty_probability=0.0,
            )

        results[ep_key] = {
            "emitted": final_result["one_shot_emitted"],
            "emit_step": final_result["emit_step"],
            "final_state": final_result["state"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="V5 scheduler L3 replay (TEACHER_EVENT_GATED_DIAGNOSTIC)")
    parser.add_argument("--prediction-root", type=Path, required=True,
                        help="Directory containing predict_*/ subdirectories")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, default=None,
                        help="Optional Platt calibration params JSON")
    parser.add_argument("--k10-window", type=int, default=8,
                        help="Minimum corridor length for valid opportunity")
    args = parser.parse_args()

    pred_root = args.prediction_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover splits
    splits = sorted(
        d.name for d in pred_root.iterdir()
        if d.is_dir() and d.name.startswith("predict_V2B_EXACT_")
    )

    if not splits:
        raise SystemExit(f"No predict_*/ directories found in {pred_root}")

    # Load calibrator if provided
    calibrator = None
    cal_data = {}
    if args.calibration_json:
        cal_data = json.loads(args.calibration_json.read_text())
        # cal_data is a list of per-split per-head Platt params

    # Script identity
    script_sha = sha256_file(Path(__file__))
    scheduler_sha = sha256_file(ROOT / "src/gripper_attack/v5_scheduler.py")

    print(f"V5 Scheduler L3 Replay")
    print(f"  Status: TEACHER_EVENT_GATED_DIAGNOSTIC_ONLY")
    print(f"  candidate_close: APPROXIMATE — route_supported AND event_id >= 0")
    print(f"  Splits: {len(splits)}")
    print(f"  Script SHA: {script_sha[:16]}")
    print(f"  Scheduler SHA: {scheduler_sha[:16]}")

    all_metrics = {}
    per_split_rows = []

    for sl in splits:
        pred_dir = pred_root / sl
        if not (pred_dir / "heldout_step_predictions.jsonl").is_file():
            continue

        steps = load_steps(pred_dir)
        episodes = group_by_episode(steps)
        short = sl.replace("predict_V2B_EXACT_W32_H64_D0.1_WD1e-4_", "")

        # Get calibrator for this split
        cal = None
        for entry in cal_data:
            if entry.get("split") == short:
                if cal is None: cal = {}
                if entry.get("head") == "grasp":
                    cal["grasp_a"] = entry.get("a", 1.0)
                    cal["grasp_b"] = entry.get("b", 0.0)
                elif entry.get("head") == "release":
                    cal["release_a"] = entry.get("a", 1.0)
                    cal["release_b"] = entry.get("b", 0.0)

        # Raw replay
        raw_results = replay_scheduler(episodes, None)
        raw_metrics = compute_metrics(episodes, raw_results, args.k10_window)

        # Platt replay (if available)
        platt_metrics = None
        if cal:
            platt_results = replay_scheduler(episodes, cal)
            platt_metrics = compute_metrics(episodes, platt_results, args.k10_window)

        all_metrics[short] = {"raw": raw_metrics, "platt": platt_metrics}

        print(f"\n{short}:")
        print(f"  RAW:  neg_ep={raw_metrics['negative_episodes']} pos_ep={raw_metrics['positive_episodes']} "
              f"emitted={raw_metrics['scheduler_emitted_total']} "
              f"false_start={raw_metrics['false_start_episodes']} "
              f"hit={raw_metrics['valid_opportunity_hits']} "
              f"abstain={raw_metrics['abstention_episodes']}")
        print(f"  RAW rates: false_start={raw_metrics['negative_episode_false_start_rate']:.4f} "
              f"recall={raw_metrics['valid_opportunity_recall']:.4f} "
              f"precision={raw_metrics['emit_precision']:.4f} "
              f"abstention={raw_metrics['abstention_rate']:.4f}")

        if platt_metrics:
            print(f"  PLATT: neg_ep={platt_metrics['negative_episodes']} pos_ep={platt_metrics['positive_episodes']} "
                  f"emitted={platt_metrics['scheduler_emitted_total']} "
                  f"false_start={platt_metrics['false_start_episodes']} "
                  f"hit={platt_metrics['valid_opportunity_hits']} "
                  f"abstain={platt_metrics['abstention_episodes']}")
            print(f"  PLATT rates: false_start={platt_metrics['negative_episode_false_start_rate']:.4f} "
                  f"recall={platt_metrics['valid_opportunity_recall']:.4f} "
                  f"precision={platt_metrics['emit_precision']:.4f} "
                  f"abstention={platt_metrics['abstention_rate']:.4f}")

        per_split_rows.append({
            "split": short,
            "negative_episodes": raw_metrics["negative_episodes"],
            "positive_episodes": raw_metrics["positive_episodes"],
            "raw_emitted": raw_metrics["scheduler_emitted_total"],
            "raw_false_start": raw_metrics["false_start_episodes"],
            "raw_hit": raw_metrics["valid_opportunity_hits"],
            "raw_false_start_rate": round(raw_metrics["negative_episode_false_start_rate"], 6),
            "raw_recall": round(raw_metrics["valid_opportunity_recall"], 6),
            "raw_precision": round(raw_metrics["emit_precision"], 6),
            "platt_emitted": platt_metrics["scheduler_emitted_total"] if platt_metrics else None,
            "platt_false_start": platt_metrics["false_start_episodes"] if platt_metrics else None,
            "platt_hit": platt_metrics["valid_opportunity_hits"] if platt_metrics else None,
            "platt_false_start_rate": round(platt_metrics["negative_episode_false_start_rate"], 6) if platt_metrics else None,
            "platt_recall": round(platt_metrics["valid_opportunity_recall"], 6) if platt_metrics else None,
            "platt_precision": round(platt_metrics["emit_precision"], 6) if platt_metrics else None,
        })

    # Write outputs
    manifest = {
        "replay_status": "TEACHER_EVENT_GATED_DIAGNOSTIC_ONLY",
        "authoritative_l3": False,
        "candidate_close_status": "APPROXIMATE_NOT_RUNTIME_CAUSAL",
        "candidate_close_approximation": "route_supported AND event_id >= 0",
        "blocked_reason": "Policy action stream (raw gripper) not in prediction bundle. "
                          "candidate_close requires runtime gripper intent classification.",
        "required_missing_fields": ["raw_gripper", "action_intent", "candidate_close"],
        "script_sha256": script_sha,
        "scheduler_sha256": scheduler_sha,
        "scheduler_source": "src/gripper_attack/v5_scheduler.py",
        "k10_window": args.k10_window,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_splits": len(splits),
    }

    with open(out_root / "v5_scheduler_l3_replay.json", "w") as f:
        json.dump({**manifest, "per_split": all_metrics}, f, indent=2)

    with open(out_root / "v5_scheduler_l3_per_split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_split_rows[0].keys())
        w.writeheader()
        w.writerows(per_split_rows)

    print(f"\nOutputs written to {out_root}")
    print(f"Status: {manifest['replay_status']}")


if __name__ == "__main__":
    main()
