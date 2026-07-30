"""D8 Weight Audit: formal training weight invariant verification.

Uses strict formal loaders (load_sidecar_correct, load_teacher_labels).
Produces sealed audit artifact with per-episode weight checks.

Verified invariants:
  - UNKNOWN/GEOM_NA/RIGHT_CENSORED weights = 0
  - Per-episode: all consolidated events have equal total weight
  - All effective negative spans have non-zero weight
  - Per-episode positive total ≈ 1.0 (within tolerance)
  - Effective sample sizes computed via ESS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
    compute_consolidation_digest,
)
from run_d8_formal_g_sensitivity import (
    load_sidecar_correct,
    load_teacher_labels,
)
from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

ARTICULATED_TASKS = {"libero_goal/task_00", "libero_goal/task_07"}


def _write_seal(p: Path) -> str:
    files = sorted(
        x for x in p.rglob("*")
        if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files),
        encoding="utf-8",
    )
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def audit(
    sidecar_root: Path,
    teacher_root: Path,
    G: int,
    output_root: Path,
) -> dict[str, Any]:
    sidecar = load_sidecar_correct(sidecar_root)
    ep_labels, teacher_steps, teacher_ids = load_teacher_labels(teacher_root)

    # Identity closure
    sc_ids = set(sidecar.keys())
    t_ids = set(ep_labels.keys())
    if sc_ids != t_ids:
        raise ValueError(f"identity mismatch: sidecar={len(sc_ids)} teacher={len(t_ids)}")

    # Per-epoch tracking
    all_pos_event_weights: list[float] = []
    all_neg_span_weights: list[float] = []
    per_episode_pos_events: dict[str, list[float]] = {}  # eid -> [event_w, ...]
    total_pos = 0.0
    total_neg = 0.0
    total_unk = 0.0
    total_geom_na = 0.0
    total_rc = 0.0
    total_art = 0
    issues: list[str] = []
    ep_pos_totals: list[float] = []
    ep_neg_totals: list[float] = []
    n_applicable = 0
    n_zero_event = 0

    for eid in sorted(t_ids):
        labels = ep_labels[eid]
        relations = sidecar[eid]

        # Per-episode step closure
        if set(relations.keys()) != set(labels.keys()):
            raise ValueError(f"step set mismatch in {eid}")

        result = consolidate_physical_events(eid, labels, relations=relations, G=G)
        task_key = "/".join(eid.split("/")[:2])
        is_art = task_key in ARTICULATED_TASKS

        if is_art:
            for lab in labels.values():
                if lab.get("mask") and lab.get("valid_mask"):
                    total_art += 1
            continue

        event_groups = result.get("event_groups", [])

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

        weights = build_physical_event_weights(
            labs, masks, result, right_censored=rc_arr, geom_na=geom_arr,
        )

        effective_mask = masks & (~rc_arr) & (~geom_arr)

        # UNKNOWN / zero-weight categories
        unk_mask = (labs == -1.0) & masks
        total_unk += float(weights[unk_mask].sum())
        total_geom_na += float(weights[geom_arr].sum())
        total_rc += float(weights[rc_arr].sum())

        # Positive: per-event weights within this episode
        ep_event_weights = []
        for group in event_groups:
            event_w = 0.0
            for fs, fe in group["fragment_ranges"]:
                for i in range(fs, fe + 1):
                    if i < n and effective_mask[i] and labs[i] == 1.0:
                        event_w += float(weights[i])
            ep_event_weights.append(event_w)
            all_pos_event_weights.append(event_w)

        if ep_event_weights:
            per_episode_pos_events[eid] = ep_event_weights
            n_applicable += 1
            # Check: all events within this episode have equal weight
            # Tolerance: float32 precision (~1e-7 relative) from weight builder
            if len(ep_event_weights) > 1:
                if not np.allclose(ep_event_weights, ep_event_weights[0], rtol=1e-6, atol=1e-7):
                    issues.append(
                        f"{eid}: intra-episode event weights not equal: "
                        f"{ep_event_weights}"
                    )
        else:
            n_zero_event += 1

        pos_mask = (labs == 1.0) & effective_mask
        ep_pos_totals.append(float(weights[pos_mask].sum()))

        # Negative: per-span weights within this episode
        ep_neg_span_weights = []
        i = 0
        while i < n:
            if effective_mask[i] and labs[i] == 0.0:
                j = i + 1
                while j < n and effective_mask[j] and labs[j] == 0.0:
                    j += 1
                span_w = float(weights[i:j].sum())
                all_neg_span_weights.append(span_w)
                ep_neg_span_weights.append(span_w)
                i = j
            else:
                i += 1

        neg_mask = (labs == 0.0) & effective_mask
        tn_ep = float(weights[neg_mask].sum())
        ep_neg_totals.append(tn_ep)

        # Per-episode negative span equality (same as positive event equality)
        if ep_neg_span_weights and len(ep_neg_span_weights) > 1:
            if not np.allclose(ep_neg_span_weights, ep_neg_span_weights[0], rtol=1e-6, atol=1e-7):
                issues.append(
                    f"{eid}: intra-episode negative span weights not equal: "
                    f"{ep_neg_span_weights}"
                )

        # Per-episode positive total check
        if tp > 0 and not np.isclose(tp, 1.0, rtol=1e-6, atol=1e-7):
            issues.append(f"{eid}: positive total != 1.0: {tp:.10f}")

        # Per-episode negative total check
        if tn_ep > 0 and not np.isclose(tn_ep, 1.0, rtol=1e-6, atol=1e-7):
            issues.append(f"{eid}: negative total != 1.0: {tn_ep:.10f}")

    # Invariant checks
    if abs(total_unk) > 1e-10:
        issues.append(f"UNKNOWN weight non-zero: {total_unk:.10f}")
    if abs(total_geom_na) > 1e-10:
        issues.append(f"GEOM_NA weight non-zero: {total_geom_na:.10f}")
    if abs(total_rc) > 1e-10:
        issues.append(f"RIGHT_CENSORED weight non-zero: {total_rc:.10f}")

    if all_neg_span_weights and any(w <= 0 for w in all_neg_span_weights):
        zero_spans = [w for w in all_neg_span_weights if w <= 0]
        issues.append(f"Negative spans with zero weight: {len(zero_spans)}")

    # ESS
    pos_w_arr = np.array(all_pos_event_weights) if all_pos_event_weights else np.array([0.0])
    neg_w_arr = np.array(all_neg_span_weights) if all_neg_span_weights else np.array([0.0])
    ess_pos = float(np.sum(pos_w_arr)**2 / max(np.sum(pos_w_arr**2), 1e-20))
    ess_neg = float(np.sum(neg_w_arr)**2 / max(np.sum(neg_w_arr**2), 1e-20))

    # Per-episode pos total check (only episodes with positive events)
    ep_pos_arr = np.array(ep_pos_totals) if ep_pos_totals else np.array([0.0])
    nonzero_pos = ep_pos_arr[ep_pos_arr > 0]
    ep_pos_close_to_one = bool(
        len(nonzero_pos) > 0 and np.allclose(nonzero_pos, 1.0, rtol=1e-6, atol=1e-7)
    )

    consumer_eligible = len(issues) == 0 and n_applicable > 0

    report = {
        "schema": "DETECTOR_V3_D8_WEIGHT_AUDIT_V1",
        "status": "PASS" if consumer_eligible else "FAIL",
        "consumer_eligible": consumer_eligible,
        "G": G,
        "applicable_episodes": n_applicable,
        "zero_event_episodes": n_zero_event,
        "articulated_masked_steps": total_art,
        "positive": {
            "total_weight": float(np.sum(pos_w_arr)),
            "event_count": len(all_pos_event_weights),
            "per_event_mean": float(pos_w_arr.mean()) if len(pos_w_arr) > 0 else 0.0,
            "per_event_min": float(pos_w_arr.min()) if len(pos_w_arr) > 0 else 0.0,
            "per_event_max": float(pos_w_arr.max()) if len(pos_w_arr) > 0 else 0.0,
            "ess": float(ess_pos),
            "per_episode_total_near_1": ep_pos_close_to_one,
        },
        "negative": {
            "total_weight": float(np.sum(neg_w_arr)),
            "span_count": len(all_neg_span_weights),
            "per_span_mean": float(neg_w_arr.mean()) if len(neg_w_arr) > 0 else 0.0,
            "ess": float(ess_neg),
            "all_nonzero": all(w > 0 for w in all_neg_span_weights) if all_neg_span_weights else True,
        },
        "zero_weight": {
            "UNKNOWN": float(total_unk),
            "GEOM_NA": float(total_geom_na),
            "RIGHT_CENSORED": float(total_rc),
            "all_zero": abs(total_unk) <= 1e-10 and abs(total_geom_na) <= 1e-10 and abs(total_rc) <= 1e-10,
        },
        "intra_episode": {
            "episodes_with_events": len(per_episode_pos_events),
            "all_episodes_events_equal": len(per_episode_pos_events) > 0,
        },
        "issues": issues,
        "pass": len(issues) == 0,
    }

    # Verify intra-episode equality (float32 tolerance)
    for eid, ews in per_episode_pos_events.items():
        if len(ews) > 1 and not np.allclose(ews, ews[0], rtol=1e-6, atol=1e-7):
            report["intra_episode"]["all_episodes_events_equal"] = False
            break

    # Write sealed output
    if output_root.exists():
        raise FileExistsError(str(output_root))
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)

    (staging / "AUDIT_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="D8 Formal Weight Audit")
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--G", type=int, default=3)
    args = parser.parse_args()

    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True
    ).strip():
        return 1  # clean checkout required

    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"), cwd=ROOT, text=True
    ).strip()

    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)
    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)

    print(f"Sidecar seal: {sidecar_seal['sha256sums_sha256'][:20]}...")
    print(f"Teacher seal: {teacher_seal['sha256sums_sha256'][:20]}...")

    report = audit(sidecar_root, teacher_root, args.G, args.output_root)

    # Add provenance bindings
    report["code_snapshot"] = {"commit": commit, "tree": tree}
    report["audit_script_sha256"] = sha256_file(Path(__file__))
    report["consolidator_sha256"] = sha256_file(
        ROOT / "scripts" / "detector_v5" / "d8_event_consolidator.py"
    )
    report["protocol_sha256"] = sha256_file(
        ROOT / "configs" / "DETECTOR_V3_D8_EVENT_CONSOLIDATION_PROTOCOL.json"
    )
    report["sidecar_root"] = str(sidecar_root)
    report["sidecar_seal"] = sidecar_seal["sha256sums_sha256"]
    report["teacher_root"] = str(teacher_root)
    report["teacher_seal"] = teacher_seal["sha256sums_sha256"]
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    # Rewrite report with provenance (preserving seal)
    audit_path = args.output_root / "AUDIT_REPORT.json"
    if audit_path.exists():
        audit_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    # Reseal with provenance
    digest = _write_seal(args.output_root)
    report["sha256sums_sha256"] = digest

    consumer_eligible = report.get("consumer_eligible", False)
    pass_all = report["pass"] and consumer_eligible

    status = "PASS" if pass_all else f"FAIL"
    p = report["positive"]
    n = report["negative"]
    z = report["zero_weight"]
    print(f"\nG={args.G}: {status}")
    print(f"  Consumer eligible: {consumer_eligible}")
    print(f"  Positive: {p['event_count']} events, ESS={p['ess']:.1f}, "
          f"per_ep_near_1={p['per_episode_total_near_1']}")
    print(f"  Negative: {n['span_count']} spans, ESS={n['ess']:.1f}, "
          f"nonzero={n['all_nonzero']}")
    print(f"  Zero-weight: UNK={z['UNKNOWN']:.10f} GEOM_NA={z['GEOM_NA']:.10f} "
          f"RC={z['RIGHT_CENSORED']:.10f}")
    print(f"  Intra-episode events equal: {report['intra_episode']['all_episodes_events_equal']}")
    for issue in report["issues"]:
        print(f"  ISSUE: {issue}")

    print(f"\nSealed: {digest}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
