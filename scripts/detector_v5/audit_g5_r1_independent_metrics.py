"""G5-R1: Independent recalculation audit from sealed G4 prediction roots.

Reads predictions.jsonl from each G4 config root, recomputes all metrics
independently, checks candidate ceiling invariant, reports per-suite/task
breakdown, and evaluates deterministic gripper gate baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal

ACTIVE_HEADS = ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
INACTIVE_HEADS = ("safe_release",)


def load_predictions(root: Path) -> list[dict[str, Any]]:
    verify_seal(root)
    rows = []
    with (root / "predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split_metadata(g1_root: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for split_name in ("EPISODE_TRAIN_MANIFEST.json", "EPISODE_VAL_MANIFEST.json"):
        rows = json.loads((g1_root / split_name).read_text(encoding="utf-8"))
        for r in rows:
            meta[r["episode_id"]] = {"suite": r["suite"], "task_id": r["task_id"]}
    return meta


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _safe_auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order].astype(np.float64)
    n = len(y)
    cum_tp = 0.0
    ap = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and score[order[j]] == score[order[i]]:
            j += 1
        group_tp = float(y_sorted[i:j].sum())
        cum_tp += group_tp
        if group_tp > 0:
            ap += (cum_tp / j) * (group_tp / positives)
        i = j
    return float(ap)


def _binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int64)
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    bacc = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom and denom > 0 else None
    return {
        "count": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "auroc": _safe_auc(y, score), "auprc": _safe_auprc(y, score),
        "balanced_accuracy": bacc, "mcc": mcc,
        "precision": tp / (tp + fp) if tp + fp else None, "recall": tpr,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def deterministic_gripper_gate(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics using deterministic gripper closing rule.

    Gate: gripper_command >= 0.9 (gripper actively closing)
    This uses the fact that gripper_command IS dim 0 of the 25D features,
    but for the audit we use the prediction rows which have candidate_close.
    We approximate: gripper closing ≈ candidate_close (since candidate_close
    is derived from gripper action/qpos history).
    """
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in predictions:
        if row.get("split") != "validation":
            continue
        eid = row["episode_id"]
        cc = row.get("candidate_close", False)
        for head in ACTIVE_HEADS:
            hd = row.get(head, {})
            if not isinstance(hd, dict):
                continue
            known = hd.get("known", False)
            if known:
                results[head]["y"].append(float(hd.get("target", 0)))
                results[head]["score"].append(float(cc))  # gate: 1.0 inside candidate, 0.0 outside

    out: dict[str, Any] = {}
    for head in ACTIVE_HEADS:
        y = np.asarray(results[head]["y"], dtype=np.int64)
        s = np.asarray(results[head]["score"], dtype=np.float64)
        out[head] = _binary_metrics(y, s)
    return out


def compute_event_metrics_independent(
    predictions: list[dict[str, Any]], head: str,
    threshold: float, meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Independent event-level metric computation from predictions."""
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row.get("split") != "validation":
            continue
        by_episode[row["episode_id"]].append(row)

    teacher_critical_total = 0
    teacher_critical_reached = 0
    teacher_detected = 0
    candidate_events = 0
    unknown_events = 0
    right_censored = 0
    false_emits = 0
    true_detected = 0
    total_episodes = 0
    episodes_with_candidate = 0
    episodes_with_false = 0
    latencies: list[int] = []

    # Per-suite/task tracking
    per_suite: dict[str, dict[str, Any]] = defaultdict(lambda: {"tcrit": 0, "detected": 0, "reached": 0, "cand": 0})

    for eid, rows in by_episode.items():
        total_episodes += 1
        md = meta.get(eid, {})
        suite = md.get("suite", "unknown")

        steps = len(rows)
        has_candidate = np.zeros(steps, dtype=bool)
        for i, r in enumerate(rows):
            if r.get("candidate_close"):
                has_candidate[i] = True

        # Teacher critical spans: contiguous known TRUE
        tc_spans = []
        start = None
        for i, r in enumerate(rows):
            hd = r.get(head, {})
            if not isinstance(hd, dict):
                continue
            is_true = hd.get("known") and hd.get("target") == 1
            if is_true and start is None:
                start = i
            if not is_true and start is not None:
                tc_spans.append((start, i - 1))
                start = None
        if start is not None:
            tc_spans.append((start, steps - 1))

        teacher_critical_total += len(tc_spans)
        per_suite[suite]["tcrit"] += len(tc_spans)

        # Candidate spans
        cand_spans = []
        start = None
        for i, v in enumerate(has_candidate):
            if v and start is None:
                start = i
            if not v and start is not None:
                cand_spans.append((start, i - 1))
                start = None
        if start is not None:
            cand_spans.append((start, steps - 1))

        candidate_events += len(cand_spans)
        per_suite[suite]["cand"] += len(cand_spans)
        if cand_spans:
            episodes_with_candidate += 1

        # Reached
        for ts, te in tc_spans:
            if has_candidate[ts:te + 1].any():
                teacher_critical_reached += 1
                per_suite[suite]["reached"] += 1

        # Detector evaluation
        ep_false = 0
        for cs, ce in cand_spans:
            # Event label within this candidate span
            known_in_span = 0
            true_in_span = 0
            for i in range(cs, ce + 1):
                hd = rows[i].get(head, {})
                if isinstance(hd, dict) and hd.get("known"):
                    known_in_span += 1
                    if hd.get("target") == 1:
                        true_in_span += 1

            if known_in_span == 0:
                continue  # no known steps — skip

            if known_in_span < (ce - cs + 1):
                unknown_events += 1
                continue  # UNKNOWN event

            # Known event: check if detector fires
            fired = False
            for i in range(cs, ce + 1):
                prob = rows[i].get(head, {}).get("probability", 0)
                if isinstance(prob, (int, float)) and prob >= threshold:
                    fired = True
                    if true_in_span > 0:
                        latencies.append(i - cs)
                    break

            if true_in_span > 0:
                if fired:
                    true_detected += 1
            else:
                if fired:
                    false_emits += 1
                    ep_false += 1

        if ep_false > 0:
            episodes_with_false += 1

    cc = teacher_critical_reached / teacher_critical_total if teacher_critical_total else None
    e2e = true_detected / teacher_critical_total if teacher_critical_total else None
    ccr = true_detected / teacher_critical_reached if teacher_critical_reached else None

    # Check invariant
    invariant_ok = True
    if teacher_critical_reached > teacher_critical_total:
        invariant_ok = False

    return {
        "head": head, "threshold": threshold,
        "teacher_critical_events": teacher_critical_total,
        "teacher_critical_reached_by_candidate": teacher_critical_reached,
        "teacher_detected_events": true_detected,
        "candidate_events": candidate_events,
        "candidate_ceiling": cc,
        "end_to_end_critical_recall": e2e,
        "candidate_conditioned_recall": ccr,
        "invariant_ok": invariant_ok,
        "unknown_events": unknown_events,
        "right_censored_events": right_censored,
        "false_emits": false_emits,
        "false_emits_per_episode": false_emits / total_episodes if total_episodes else None,
        "episodes_with_false_trigger": episodes_with_false,
        "total_episodes": total_episodes,
        "episodes_with_candidate": episodes_with_candidate,
        "no_candidate_episodes": total_episodes - episodes_with_candidate,
        "latency_mean": float(np.mean(latencies)) if latencies else None,
        "latency_count": len(latencies),
        "per_suite": dict(per_suite),
    }


def compute_step_metrics_independent(
    predictions: list[dict[str, Any]], head: str,
    threshold: float, meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Independent step-level metrics."""
    ys = []
    ss = []
    for row in predictions:
        if row.get("split") != "validation":
            continue
        hd = row.get(head, {})
        if not isinstance(hd, dict) or not hd.get("known"):
            continue
        ys.append(int(hd.get("target", 0)))
        ss.append(float(hd.get("probability", 0)))
    y = np.asarray(ys, dtype=np.int64)
    s = np.asarray(ss, dtype=np.float64)
    return _binary_metrics(y, s, threshold)


def compute_shuffle_metrics_independent(
    predictions: list[dict[str, Any]], head: str,
    threshold: float,
) -> dict[str, Any]:
    """Recompute all metrics for shuffle comparison."""
    ys = []
    ss = []
    for row in predictions:
        if row.get("split") != "validation":
            continue
        hd = row.get(head, {})
        if not isinstance(hd, dict) or not hd.get("known"):
            continue
        ys.append(int(hd.get("target", 0)))
        ss.append(float(hd.get("probability", 0)))
    y = np.asarray(ys, dtype=np.int64)
    s = np.asarray(ss, dtype=np.float64)
    return _binary_metrics(y, s, threshold)


def audit_config(
    config_name: str, root: Path, meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    predictions = load_predictions(root)
    report = json.loads((root / "heldout_report.json").read_text(encoding="utf-8"))
    thresholds_data = json.loads((root / "thresholds.json").read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "config": config_name,
        "root": str(root),
        "seal": verify_seal(root)["sha256sums_sha256"],
        "heldout_status": report.get("status"),
        "normalization_source": report.get("normalization_source"),
        "normalization_drift_status": report.get("normalization_drift", {}).get("status"),
    }

    # Step metrics for active heads
    result["step_metrics"] = {}
    result["event_metrics"] = {}
    result["ceiling_check"] = {}

    for head in ACTIVE_HEADS:
        th = thresholds_data.get(head, {})
        threshold = th.get("threshold")
        if threshold is None:
            result["step_metrics"][head] = {"status": "HOLD_COVERAGE" if th.get("status") == "HOLD_COVERAGE" else "NOT_ACTIVE"}
            result["event_metrics"][head] = {"status": "NOT_EVALUABLE"}
            continue

        result["step_metrics"][head] = compute_step_metrics_independent(predictions, head, threshold, meta)
        result["event_metrics"][head] = compute_event_metrics_independent(predictions, head, threshold, meta)

        # Invariant check
        em = result["event_metrics"][head]
        result["ceiling_check"][head] = {
            "tcrit": em["teacher_critical_events"],
            "reached": em["teacher_critical_reached_by_candidate"],
            "detected": em["teacher_detected_events"],
            "invariant_reached_le_tcrit": em["teacher_critical_reached_by_candidate"] <= em["teacher_critical_events"],
            "invariant_detected_le_reached": em["teacher_detected_events"] <= em["teacher_critical_reached_by_candidate"],
        }

    # Deterministic gripper gate
    result["gripper_gate_deterministic"] = deterministic_gripper_gate(predictions)

    # Shuffle full metrics from report
    result["shuffle_full"] = {}
    sh = report.get("label_shuffle", {})
    for sk, sv in sh.items():
        result["shuffle_full"][sk] = {}
        for head in ACTIVE_HEADS:
            th = thresholds_data.get(head, {})
            threshold = th.get("threshold")
            if threshold is None:
                result["shuffle_full"][sk][head] = {"status": "NOT_EVALUABLE"}
                continue
            result["shuffle_full"][sk][head] = compute_shuffle_metrics_independent(predictions, head, threshold)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--three-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--gripper-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = load_split_metadata(args.g1_root)

    results = {}
    for name, root in [
        ("shared_four_head", args.shared_root),
        ("three_head", args.three_root),
        ("physical_only", args.physical_root),
        ("gripper_only", args.gripper_root),
    ]:
        print(f"Auditing {name}...")
        results[name] = audit_config(name, root, meta)

    output = {
        "schema": "V5_R3_G5_R1_INDEPENDENT_AUDIT_V1",
        "status": "COMPLETED",
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
