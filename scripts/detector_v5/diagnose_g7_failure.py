"""D7-B: Root-cause diagnostic of G7 test failure.

POST_G7_DIAGNOSTIC_NONSELECTIVE — test is contaminated for future model selection.
Decomposes missed teacher critical events into:
  UNREACHABLE (candidate gate) vs REACHABLE_BUT_MISSED (model/scheduler).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal

THETA_PHYSICAL = 0.30
THETA_SCHEDULER = 0.25
PERSISTENCE = 5
COOLDOWN = True


def load_g7_predictions(g7_root: Path) -> list[dict[str, Any]]:
    verify_seal(g7_root)
    return [json.loads(l) for l in (g7_root / "g7_test_predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_g1_meta(g1_root: Path) -> dict[str, dict[str, Any]]:
    meta = {}
    for sf in ("EPISODE_TEST_MANIFEST.json",):
        rows = json.loads((g1_root / sf).read_text(encoding="utf-8"))
        for r in rows:
            meta[r["episode_id"]] = {"suite": r["suite"], "task_id": r["task_id"]}
    return meta


def diagnose(g7_root: Path, g1_root: Path, output: Path) -> dict[str, Any]:
    predictions = load_g7_predictions(g7_root)
    meta = load_g1_meta(g1_root)

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_episode[row["episode_id"]].append(row)

    # Event-level analysis
    unreachable_events: list[dict[str, Any]] = []
    reachable_missed: list[dict[str, Any]] = []
    detected_events: list[dict[str, Any]] = []

    for eid, rows in sorted(by_episode.items()):
        md = meta.get(eid, {})
        suite = md.get("suite", "unknown")
        task = md.get("task_id", -1)
        steps = len(rows)

        # Teacher critical spans
        tc_spans = []
        start = None
        for i, r in enumerate(rows):
            ph = r.get("physical_criticality", {})
            is_true = isinstance(ph, dict) and ph.get("known") and ph.get("target") == 1
            if is_true and start is None:
                start = i
            if not is_true and start is not None:
                tc_spans.append((start, i - 1))
                start = None
        if start is not None:
            tc_spans.append((start, steps - 1))

        # Candidate spans + has_candidate mask
        has_candidate = np.zeros(steps, dtype=bool)
        cand_spans = []
        start = None
        for i, r in enumerate(rows):
            if r.get("candidate_close"):
                has_candidate[i] = True
                if start is None:
                    start = i
            else:
                if start is not None:
                    cand_spans.append((start, i - 1))
                    start = None
        if start is not None:
            cand_spans.append((start, steps - 1))

        # For each teacher critical event, determine reachability and detection
        for ts, te in tc_spans:
            event_len = te - ts + 1
            prob_in_event = []
            for i in range(ts, te + 1):
                prob = rows[i].get("physical_criticality", {}).get("probability", 0)
                prob_in_event.append(float(prob) if isinstance(prob, (int, float)) else 0.0)
            max_prob = max(prob_in_event) if prob_in_event else 0.0

            # Is this event reachable by candidate_close?
            reachable = has_candidate[ts:te + 1].any()

            base = {
                "episode_id": eid, "suite": suite, "task_id": task,
                "event_start": ts, "event_end": te, "event_length": event_len,
                "max_physical_probability": max_prob,
                "candidate_overlap": bool(reachable),
            }

            if not reachable:
                # Find distance to nearest candidate span
                nearest_dist = steps
                for cs, ce in cand_spans:
                    if ce < ts:
                        nearest_dist = min(nearest_dist, ts - ce)
                    elif cs > te:
                        nearest_dist = min(nearest_dist, cs - te)
                    else:
                        nearest_dist = 0
                base["nearest_candidate_distance"] = nearest_dist
                unreachable_events.append(base)
                continue

            # Reachable: check detection
            detected = False
            miss_reason = "MODEL_SCORE_BELOW_THRESHOLD"

            # Check per-step threshold crossing
            any_cross = any(p >= THETA_PHYSICAL for p in prob_in_event)
            if not any_cross:
                miss_reason = "MODEL_SCORE_BELOW_THRESHOLD"
            else:
                # Check scheduler persistence
                # Find overlapping candidate spans
                scheduler_detected = False
                for cs, ce in cand_spans:
                    overlap_s = max(ts, cs)
                    overlap_e = min(te, ce)
                    if overlap_s > overlap_e:
                        continue
                    prob_above = []
                    for i in range(cs, ce + 1):
                        p = rows[i].get("physical_criticality", {}).get("probability", 0)
                        prob_above.append(isinstance(p, (int, float)) and p >= THETA_SCHEDULER)
                    for i in range(len(prob_above) - PERSISTENCE + 1):
                        if all(prob_above[i:i + PERSISTENCE]):
                            scheduler_detected = True
                            break
                    if scheduler_detected:
                        break

                if scheduler_detected:
                    detected = True
                else:
                    # Determine why scheduler missed
                    if any_cross and not scheduler_detected:
                        # Check if persistence rejected
                        has_persistence_run = False
                        for cs, ce in cand_spans:
                            overlap_s = max(ts, cs)
                            overlap_e = min(te, ce)
                            if overlap_s > overlap_e:
                                continue
                            prob_above = []
                            for i in range(cs, ce + 1):
                                p = rows[i].get("physical_criticality", {}).get("probability", 0)
                                prob_above.append(isinstance(p, (int, float)) and p >= THETA_SCHEDULER)
                            max_consec = 0; curr = 0
                            for v in prob_above:
                                if v: curr += 1; max_consec = max(max_consec, curr)
                                else: curr = 0
                            if max_consec > 0:
                                has_persistence_run = True
                                base["max_consecutive_above"] = max_consec
                                break
                        if has_persistence_run:
                            miss_reason = "PERSISTENCE_REJECT"
                        elif any_cross:
                            miss_reason = "SCORE_ABOVE_0.30_BUT_BELOW_SCHEDULER_0.25"
                        else:
                            miss_reason = "SCORE_NEVER_CROSSES_0.25"
                    else:
                        miss_reason = "MODEL_SCORE_BELOW_THRESHOLD"

            base["miss_reason"] = miss_reason
            if detected:
                detected_events.append(base)
            else:
                reachable_missed.append(base)

    # Summary
    total_tc = len(unreachable_events) + len(reachable_missed) + len(detected_events)
    missed_total = len(unreachable_events) + len(reachable_missed)

    # Categorize reachable misses
    miss_categories = defaultdict(list)
    for ev in reachable_missed:
        miss_categories[ev["miss_reason"]].append(ev["episode_id"])

    # Suite breakdown
    suite_breakdown = defaultdict(lambda: {"tc": 0, "unreachable": 0, "missed": 0, "detected": 0})
    for ev in unreachable_events:
        suite_breakdown[ev["suite"]]["tc"] += 1
        suite_breakdown[ev["suite"]]["unreachable"] += 1
    for ev in reachable_missed:
        suite_breakdown[ev["suite"]]["tc"] += 1
        suite_breakdown[ev["suite"]]["missed"] += 1
    for ev in detected_events:
        suite_breakdown[ev["suite"]]["tc"] += 1
        suite_breakdown[ev["suite"]]["detected"] += 1

    result = {
        "schema": "DETECTOR_V2_G7_DIAGNOSTIC_V1",
        "status": "POST_G7_DIAGNOSTIC_NONSELECTIVE",
        "test_contaminated_for_future_model_selection": True,
        "summary": {
            "teacher_critical_total": total_tc,
            "detected": len(detected_events),
            "missed_total": missed_total,
            "unreachable_by_candidate": len(unreachable_events),
            "reachable_but_missed": len(reachable_missed),
            "unreachable_pct": len(unreachable_events) / total_tc if total_tc else 0,
            "reachable_missed_pct": len(reachable_missed) / total_tc if total_tc else 0,
        },
        "miss_categories": {k: len(v) for k, v in miss_categories.items()},
        "suite_breakdown": dict(suite_breakdown),
        "unreachable_sample": unreachable_events[:10],
        "reachable_missed_sample": reachable_missed[:10],
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g7-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.g7_root, args.g1_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
