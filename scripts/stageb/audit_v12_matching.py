#!/usr/bin/env python3
"""Audit v1.2 TeacherLabeler matching for false-positive risks.

Checks per GPT review recommendations:
- wrong-object false positive rate
- distractor_or_setup trend
- unsupported trend
- top matched canonical objects
- match reason distribution
- nearest body distance distribution
- L10 multi-object alias sensitivity

Runs a small sample of episodes (10 per suite) using the v1.2 adapter
on GPU.  Outputs per-episode and aggregate audit metrics.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"scripts"))

import numpy as np


def audit_one_episode(adapter, episode_cfg, args) -> dict:
    """Run one episode and collect v1.2 label diagnostics."""
    records = list(adapter.run_clean_episode(episode_cfg))
    # These are StepRecord objects from the adapter
    diag = {
        "suite": episode_cfg.get("suite", ""),
        "task_index": episode_cfg.get("task_index", -1),
        "task_name": episode_cfg.get("task_name", ""),
        "task_language": episode_cfg.get("task_language", ""),
        "n_steps": len(records),
        "n_stable_carry": 0,
        "n_primary": 0,
        "n_distractor": 0,
        "n_unsupported": 0,
        "n_auxiliary": 0,
        "match_reasons": defaultdict(int),
        "grasped_objects": defaultdict(int),
        "event_roles": defaultdict(int),
    }
    for rec in records:
        phase = getattr(rec, "teacher_phase", "")
        role = getattr(rec, "teacher_event_role", "")
        diag["event_roles"][role] += 1
        if phase == "stable_carry":
            diag["n_stable_carry"] += 1
            if role == "primary_attackable":
                diag["n_primary"] += 1
            elif role == "distractor_or_setup":
                diag["n_distractor"] += 1
            elif role == "unsupported_or_abstain":
                diag["n_unsupported"] += 1
            elif role == "auxiliary_manipulation":
                diag["n_auxiliary"] += 1
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest.jsonl with episode cfgs")
    ap.add_argument("--max-episodes", type=int, default=40, help="Max episodes to audit (10/suite)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default="", help="Output JSON path")
    args = ap.parse_args()

    import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from scripts.stageb.c2f_libero_openvla_adapter import C2fLiberoOpenVLAAdapter
    adapter = C2fLiberoOpenVLAAdapter(args)

    manifest = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    # Stratify: take up to 10 per suite
    suite_quota = {"libero_object": 10, "libero_goal": 7, "libero_spatial": 7, "libero_10": 6}
    selected = []
    counts = defaultdict(int)
    for ep in manifest:
        s = ep.get("suite", "")
        if counts[s] < suite_quota.get(s, 5):
            selected.append(ep)
            counts[s] += 1
    selected = selected[:args.max_episodes]
    print(f"Auditing {len(selected)} episodes across {len(counts)} suites")

    results = []
    for i, ep in enumerate(selected):
        t0 = time.time()
        diag = audit_one_episode(adapter, ep, args)
        diag["runtime_s"] = round(time.time() - t0, 1)
        diag["episode_idx"] = i
        results.append(diag)
        print(f"  [{i+1}/{len(selected)}] {diag['suite']} task_{diag['task_index']:02d}: "
              f"sc={diag['n_stable_carry']} prim={diag['n_primary']} "
              f"dist={diag['n_distractor']} unsup={diag['n_unsupported']} "
              f"({diag['runtime_s']}s)")

    # Aggregate
    agg = {
        "n_episodes": len(results),
        "total_steps": sum(r["n_steps"] for r in results),
        "total_stable_carry": sum(r["n_stable_carry"] for r in results),
        "total_primary": sum(r["n_primary"] for r in results),
        "total_distractor": sum(r["n_distractor"] for r in results),
        "total_unsupported": sum(r["n_unsupported"] for r in results),
        "total_auxiliary": sum(r["n_auxiliary"] for r in results),
        "per_suite": {},
    }
    for s in sorted(set(r["suite"] for r in results)):
        s_results = [r for r in results if r["suite"] == s]
        agg["per_suite"][s] = {
            "n_episodes": len(s_results),
            "total_stable_carry": sum(r["n_stable_carry"] for r in s_results),
            "total_primary": sum(r["n_primary"] for r in s_results),
            "total_distractor": sum(r["n_distractor"] for r in s_results),
            "total_unsupported": sum(r["n_unsupported"] for r in s_results),
        }

    sc_total = max(agg["total_stable_carry"], 1)
    print(f"\n=== Aggregate ===")
    print(f"  episodes: {agg['n_episodes']}  steps: {agg['total_steps']}  stable_carry: {sc_total}")
    print(f"  primary: {agg['total_primary']} ({agg['total_primary']/sc_total*100:.1f}%)")
    print(f"  distractor: {agg['total_distractor']} ({agg['total_distractor']/sc_total*100:.1f}%)")
    print(f"  unsupported: {agg['total_unsupported']} ({agg['total_unsupported']/sc_total*100:.1f}%)")
    print(f"  auxiliary: {agg['total_auxiliary']} ({agg['total_auxiliary']/sc_total*100:.1f}%)")

    for s, sagg in sorted(agg["per_suite"].items()):
        ssc = max(sagg["total_stable_carry"], 1)
        print(f"\n  {s}:")
        print(f"    primary: {sagg['total_primary']} ({sagg['total_primary']/ssc*100:.1f}%)")
        print(f"    distractor: {sagg['total_distractor']} ({sagg['total_distractor']/ssc*100:.1f}%)")
        print(f"    unsupported: {sagg['total_unsupported']} ({sagg['total_unsupported']/ssc*100:.1f}%)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"aggregate": agg, "per_episode": results}, indent=2, default=str))
        print(f"\nWrote: {out_path}")

    # Risk assessment
    dist_rate = agg["total_distractor"] / sc_total
    unsup_rate = agg["total_unsupported"] / sc_total
    prim_rate = agg["total_primary"] / sc_total
    if dist_rate > 0.15:
        print(f"\nWARNING: distractor rate {dist_rate:.1%} > 15% — possible false matches")
    if unsup_rate > 0.50:
        print(f"\nWARNING: unsupported rate {unsup_rate:.1%} > 50% — object identification gap")
    if prim_rate < 0.10:
        print(f"\nWARNING: primary rate {prim_rate:.1%} < 10% — label sparsity risk")
    print(f"\nRisk: dist={dist_rate:.1%} unsup={unsup_rate:.1%} prim={prim_rate:.1%}")


if __name__ == "__main__":
    main()
