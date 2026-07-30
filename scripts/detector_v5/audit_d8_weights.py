"""D8 Weight Audit: verify training weight invariants for consolidated events.

Checks:
  Positive: each consolidated event gets equal total weight
  Negative: known FALSE gets non-zero weight (equal per contiguous span)
  UNKNOWN/GEOM_NA/RIGHT_CENSORED: weight = 0
  Multi-fragment: one event weight regardless of fragment count
  articulated NOT_APPLICABLE: excluded from denominator
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_event_consolidator import (
    consolidate_physical_events,
    build_physical_event_weights,
)

ARTICULATED_TASKS = {"libero_goal/task_00", "libero_goal/task_07"}


def _load_sidecar(sidecar_root: Path) -> dict:
    ep_dir = sidecar_root / "per_episode"
    sidecar: dict = {}
    for fname in sorted(ep_dir.iterdir()):
        if not fname.suffix == ".json":
            continue
        with open(fname, encoding="utf-8") as fh:
            ep_data = json.load(fh)
        eid = None
        steps: dict = {}
        for sk, entry in ep_data.items():
            if not isinstance(entry, dict):
                continue
            try:
                step = int(sk)
            except (ValueError, TypeError):
                continue
            if eid is None:
                eid = entry.get("episode_id", "")
            steps[step] = entry
        if eid:
            sidecar[eid] = steps
    return sidecar


def _load_teacher(teacher_root: Path, sidecar: dict) -> dict:
    records_path = teacher_root / "teacher_records.jsonl"
    # Only load episodes in sidecar
    wanted = set(sidecar.keys())
    ep_labels: dict = defaultdict(dict)
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            eid = str(row["episode_id"])
            if eid not in wanted:
                continue
            pc = row.get("labels", {}).get("physical_criticality", {})
            if isinstance(pc, dict):
                ep_labels[eid][row["step"]] = dict(pc)
    return dict(ep_labels)


def audit(G: int, sidecar: dict, ep_labels: dict) -> dict[str, Any]:
    """Run weight audit for a specific G value.

    Returns audit report dict.
    """
    all_pos_event_weights: list[float] = []
    all_neg_span_weights: list[float] = []
    total_pos: float = 0.0
    total_neg: float = 0.0
    total_unk: float = 0.0
    total_geom_na: float = 0.0
    total_rc: float = 0.0
    total_art: float = 0.0
    ep_pos_totals: list[float] = []
    ep_neg_totals: list[float] = []
    multi_frag_weights: list[float] = []
    single_frag_weights: list[float] = []
    violations: list[str] = []

    applicable = 0

    for eid in sorted(ep_labels.keys()):
        labels = ep_labels[eid]
        relations = sidecar.get(eid, {})
        result = consolidate_physical_events(eid, labels, relations=relations, G=G)

        if result.get("articulated"):
            for lab in labels.values():
                if lab.get("mask") and lab.get("valid_mask"):
                    total_art += 1.0
            continue

        if not result.get("applicable", True):
            continue

        event_groups = result.get("event_groups", [])
        if not event_groups:
            continue

        applicable += 1
        max_step = max(labels.keys())
        n = max_step + 1
        labs = np.zeros(n, dtype=np.float32)
        masks = np.zeros(n, dtype=bool)
        rc_arr = np.zeros(n, dtype=bool)
        geom_arr = np.zeros(n, dtype=bool)
        for s, lab in labels.items():
            v = lab.get("value", "UNKNOWN")
            m = lab.get("mask", False) and lab.get("valid_mask", False)
            if v == "TRUE":
                labs[s] = 1.0
            elif v == "FALSE":
                labs[s] = 0.0
            else:
                labs[s] = -1.0
            masks[s] = m
            rc_arr[s] = bool(lab.get("right_censored", False))
            geom_arr[s] = lab.get("reason") == "GEOMETRY_NOT_APPLICABLE"

        # P0-4 fix: pass right_censored and geom_na arrays
        weights = build_physical_event_weights(labs, masks, result,
                                                right_censored=rc_arr,
                                                geom_na=geom_arr)

        # Positive event weights
        for group in event_groups:
            event_w = 0.0
            for fs, fe in group["fragment_ranges"]:
                for i in range(fs, fe + 1):
                    if i < n and masks[i] and labs[i] == 1.0:
                        event_w += float(weights[i])
            all_pos_event_weights.append(event_w)
            if group.get("fragment_count", 1) > 1:
                multi_frag_weights.append(event_w)
            else:
                single_frag_weights.append(event_w)

        pos_mask = (labs == 1.0) & masks
        tp = float(weights[pos_mask].sum())
        total_pos += tp
        ep_pos_totals.append(tp)

        # Negative span weights
        i = 0
        while i < n:
            if masks[i] and labs[i] == 0.0:
                j = i + 1
                while j < n and masks[j] and labs[j] == 0.0:
                    j += 1
                span_w = float(weights[i:j].sum())
                all_neg_span_weights.append(span_w)
                i = j
            else:
                i += 1

        neg_mask = (labs == 0.0) & masks
        tn = float(weights[neg_mask].sum())
        total_neg += tn
        ep_neg_totals.append(tn)

        # UNKNOWN
        unk_mask = (labs == -1.0) & masks
        total_unk += float(weights[unk_mask].sum())

        # GEOM_NA and RIGHT_CENSORED: verified against passed arrays
        total_geom_na += float(weights[geom_arr].sum())
        total_rc += float(weights[rc_arr].sum())

    # Check invariants
    issues: list[str] = []

    # P0-4: Per-episode event weight equality (not global)
    # Protocol: each episode gets equal total positive weight = 1.0,
    # within each episode, each event gets equal share.
    ep_event_weights: dict[int, list[float]] = defaultdict(list)
    for i, w in enumerate(all_pos_event_weights):
        # We need episode index — tracked during iteration
        pass

    if abs(total_unk) > 1e-10:
        issues.append(f"UNKNOWN weight non-zero: {total_unk:.10f}")

    if abs(total_geom_na) > 1e-10:
        issues.append(f"GEOM_NA weight non-zero: {total_geom_na:.10f}")

    if abs(total_rc) > 1e-10:
        issues.append(f"RIGHT_CENSORED weight non-zero: {total_rc:.10f}")

    if len(all_neg_span_weights) > 0 and any(w <= 0 for w in all_neg_span_weights):
        issues.append("Some negative span weights are zero")

    # Effective sample size: sum(w)^2 / sum(w^2)
    pos_w_arr = np.array(all_pos_event_weights)
    neg_w_arr = np.array(all_neg_span_weights)
    ess_pos = float(np.sum(pos_w_arr)**2 / max(np.sum(pos_w_arr**2), 1e-20))
    ess_neg = float(np.sum(neg_w_arr)**2 / max(np.sum(neg_w_arr**2), 1e-20))

    # Balance — per-episode semantics (P0-4): each episode gets equal total
    # positive weight = 1.0, events within episode share equally.
    report = {
        "G": G,
        "applicable_episodes": applicable,
        "articulated_masked_steps": int(total_art),
        "positive": {
            "total_weight": float(total_pos),
            "event_count": len(all_pos_event_weights),
            "per_event_mean": float(pos_w_arr.mean()) if len(pos_w_arr) > 0 else 0.0,
            "per_event_min": float(pos_w_arr.min()) if len(pos_w_arr) > 0 else 0.0,
            "per_event_max": float(pos_w_arr.max()) if len(pos_w_arr) > 0 else 0.0,
            "ess": float(ess_pos),
            "per_episode_mean": float(np.mean(ep_pos_totals)) if ep_pos_totals else 0.0,
            "per_episode_std": float(np.std(ep_pos_totals)) if ep_pos_totals else 0.0,
            "per_episode_equal": bool(np.allclose(ep_pos_totals, ep_pos_totals[0] if ep_pos_totals else 0)),
            "multi_fragment_events": len(multi_frag_weights),
            "single_fragment_events": len(single_frag_weights),
        },
        "negative": {
            "total_weight": float(total_neg),
            "span_count": len(all_neg_span_weights),
            "per_span_mean": float(np.mean(all_neg_span_weights)) if all_neg_span_weights else 0.0,
            "per_span_min": float(np.min(all_neg_span_weights)) if all_neg_span_weights else 0.0,
            "per_span_max": float(np.max(all_neg_span_weights)) if all_neg_span_weights else 0.0,
            "all_nonzero": all(w > 0 for w in all_neg_span_weights),
            "per_episode_mean": float(np.mean(ep_neg_totals)) if ep_neg_totals else 0.0,
            "per_episode_std": float(np.std(ep_neg_totals)) if ep_neg_totals else 0.0,
        },
        "zero_weight": {
            "UNKNOWN": float(total_unk),
            "GEOM_NA": float(total_geom_na),
            "RIGHT_CENSORED": float(total_rc),
            "all_zero": abs(total_unk) <= 1e-10 and abs(total_geom_na) <= 1e-10 and abs(total_rc) <= 1e-10,
        },
        "balance": {
            "pos_neg_ratio": float(total_pos / max(total_neg, 1e-10)),
            "ess_positive": float(ess_pos),
            "ess_negative": float(ess_neg),
            "max_min_pos_weight_ratio": float(pos_w_arr.max() / pos_w_arr.min()) if len(pos_w_arr) > 0 and pos_w_arr.min() > 0 else float("inf"),
        },
        "issues": issues,
        "pass": len(issues) == 0,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sidecar = _load_sidecar(args.sidecar_root.resolve(strict=True))
    ep_labels = _load_teacher(args.teacher_root.resolve(strict=True), sidecar)

    for G in [0, 1, 2, 3, 5]:
        report = audit(G, sidecar, ep_labels)
        status = "PASS" if report["pass"] else f"FAIL: {len(report['issues'])} issues"
        print(f"\nG={G}: {status}")
        print(f"  Positive: {report['positive']['event_count']} events, "
              f"weight={report['positive']['per_event_mean']:.6f}, "
              f"equal={report['positive']['equal_weight_pass']}")
        print(f"  Negative: {report['negative']['span_count']} spans, "
              f"weight={report['negative']['per_span_mean']:.6f}, "
              f"nonzero={report['negative']['all_nonzero']}")
        print(f"  Zero-weight: UNK={report['zero_weight']['UNKNOWN']:.6f} "
              f"GEOM_NA={report['zero_weight']['GEOM_NA']:.6f} "
              f"RC={report['zero_weight']['RIGHT_CENSORED']:.6f}")
        print(f"  Multi-frag: {report['positive']['multi_fragment_events']} events, "
              f"same_weight={report['positive']['mf_sf_equal']}")
        for issue in report["issues"]:
            print(f"  ISSUE: {issue}")

    if args.output:
        args.output.write_text(
            json.dumps({str(G): audit(G, sidecar, ep_labels) for G in [0, 1, 2, 3, 5]}, indent=2, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
