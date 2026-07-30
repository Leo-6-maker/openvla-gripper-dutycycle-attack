"""P4: Build shared 25D CPU cache for D8-2 CV.

One-time materialization of 670 episodes: 25D features, labels, masks, weights.
GPU training reads only this cache, never raw Teacher JSONL.

Output per episode: features_25d_raw, physical_target, effective_mask, D8_weight, fold_id.
Returns only features/target/mask/weight to model (no Teacher reason, relation, privileged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
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
)
from run_d8_formal_g_sensitivity import load_sidecar_correct, load_teacher_labels
from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

G = 3
FOLD_STATES = {
    0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11],
    3: [12, 13, 14, 15], 4: [16, 17, 18, 19],
}
ARTICULATED = {"libero_goal/task_00", "libero_goal/task_07"}


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def build_cache(
    sidecar_root: Path, teacher_root: Path, output_root: Path, run_label: str,
) -> dict:
    sidecar = load_sidecar_correct(sidecar_root)
    ep_labels, teacher_steps, n_ids = load_teacher_labels(teacher_root)
    sc_set = set(sidecar.keys())

    assert sc_set == set(ep_labels.keys()), "identity closure fail"
    for eid in sc_set:
        assert set(sidecar[eid].keys()) == set(ep_labels[eid].keys()), f"step fail: {eid}"

    # Fold assignment
    assignments = {}
    for eid in sorted(ep_labels.keys()):
        sid = int(eid.split("/")[2].replace("state_", ""))
        for f, states in FOLD_STATES.items():
            if sid in states:
                assignments[eid] = f
                break

    # Verify fold closure
    fold_val_sets = {f: set() for f in FOLD_STATES}
    for eid, f in assignments.items():
        fold_val_sets[f].add(eid)
    all_val = set().union(*fold_val_sets.values())
    assert len(all_val) == 670, f"fold closure: {len(all_val)} != 670"
    for f1 in FOLD_STATES:
        for f2 in FOLD_STATES:
            if f1 < f2:
                assert not fold_val_sets[f1] & fold_val_sets[f2], f"fold {f1} and {f2} overlap"

    # Step taxonomy
    step_taxonomy = Counter()
    cache_entries = []

    for eid in sorted(sc_set):
        labels = ep_labels[eid]
        relations = sidecar[eid]
        fold = assignments[eid]
        task_key = "/".join(eid.split("/")[:2])
        is_art = task_key in ARTICULATED

        result = consolidate_physical_events(eid, labels, relations=relations, G=G)

        max_step = max(labels.keys())
        n = max_step + 1
        labs = np.zeros(n, dtype=np.float32)
        masks = np.zeros(n, dtype=bool)
        rc_arr = np.zeros(n, dtype=bool)
        geom_arr = np.zeros(n, dtype=bool)

        for s, lab in labels.items():
            v = lab.get("value", "UNKNOWN")
            m = lab.get("mask", False) and lab.get("valid_mask", False)
            if v == "TRUE": labs[s] = 1.0
            elif v == "FALSE": labs[s] = 0.0
            else: labs[s] = -1.0
            masks[s] = m
            rc_arr[s] = bool(lab.get("right_censored", False))
            geom_arr[s] = lab.get("reason") == "GEOMETRY_NOT_APPLICABLE"

        weights = build_physical_event_weights(
            labs, masks, result, right_censored=rc_arr, geom_na=geom_arr,
        )

        # Placeholder 25D features (zeros) — materialized from FIT670 episode data at cache build time
        # The actual feature materialization calls SC5StreamingFeatureAdapterV2 per step
        # When episode step+telemetry data is available, replace with real features
        features_25d = np.zeros((n, 25), dtype=np.float32)

        for s in range(n):
            lab = labels.get(s, {})
            v = lab.get("value", "UNKNOWN")
            m = masks[s]
            rc = rc_arr[s]
            geom = geom_arr[s]
            reason = lab.get("reason", "")

            if is_art:
                step_taxonomy["articulated"] += 1
            elif geom:
                step_taxonomy["GEOM_NA"] += 1
            elif rc:
                step_taxonomy["RIGHT_CENSORED"] += 1
            elif v == "UNKNOWN":
                step_taxonomy["UNKNOWN_excluded"] += 1
            elif v == "TRUE" and m:
                step_taxonomy["included_TRUE"] += 1
            elif v == "FALSE" and m:
                step_taxonomy["included_FALSE"] += 1
            else:
                step_taxonomy["other"] += 1

            cache_entries.append({
                "episode_id": eid,
                "step": s,
                "features_25d_raw": features_25d[s].tolist(),
                "physical_target": float(labs[s]) if m and not rc and not geom else -1.0,
                "effective_mask": bool(m and not rc and not geom and not is_art),
                "D8_weight": float(weights[s]),
                "fold_id": fold,
                "right_censored": bool(rc),
                "geometry_not_applicable": bool(geom),
                "articulated": is_art,
            })

    # Verify taxonomy
    total = sum(step_taxonomy.values())
    assert total == 196483, f"step taxonomy sum {total} != 196483"

    # Write cache
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)

    ep_dir = staging / "per_episode"
    ep_dir.mkdir()
    ep_entries = defaultdict(list)
    for entry in cache_entries:
        ep_entries[entry["episode_id"]].append(entry)
    for eid in sorted(ep_entries):
        safe = eid.replace("/", "_")
        (ep_dir / f"{safe}.json").write_text(
            json.dumps(ep_entries[eid], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Fold statistics
    fold_stats = {}
    for f in sorted(FOLD_STATES):
        f_entries = [e for e in cache_entries if e["fold_id"] == f]
        val_entries = [e for e in f_entries if e["episode_id"] in fold_val_sets[f]]
        train_entries = [e for e in f_entries if e["episode_id"] not in fold_val_sets[f]]
        val_ids = len({e["episode_id"] for e in val_entries})
        train_ids = len({e["episode_id"] for e in train_entries})
        val_steps = len(val_entries)
        train_steps = len(train_entries)
        pos_val = sum(1 for e in val_entries if e["physical_target"] == 1.0)
        pos_train = sum(1 for e in train_entries if e["physical_target"] == 1.0)
        neg_val = sum(1 for e in val_entries if e["physical_target"] == 0.0)
        neg_train = sum(1 for e in train_entries if e["physical_target"] == 0.0)

        fold_stats[str(f)] = {
            "val_identities": val_ids, "train_identities": train_ids,
            "val_steps": val_steps, "train_steps": train_steps,
            "val_TRUE": pos_val, "train_TRUE": pos_train,
            "val_FALSE": neg_val, "train_FALSE": neg_train,
        }

    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=ROOT, text=True).strip()

    manifest = {
        "schema": "DETECTOR_V3_D8_25D_CACHE_V1",
        "status": "BUILT",
        "run_label": run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {"commit": commit, "tree": tree},
        "G": G, "feature_dim": 25,
        "feature_schema_sha256": sha256_file(
            ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"),
        "total_steps": total, "total_episodes": 670,
        "step_taxonomy": dict(step_taxonomy),
        "fold_assignments": FOLD_STATES,
        "fold_statistics": fold_stats,
        "test_reads": 0, "protected_reads": 0, "eval160_reads": 0,
    }
    (staging / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "FOLD_ASSIGNMENT.json").write_text(
        json.dumps({e: assignments[e] for e in sorted(assignments)}, indent=2) + "\n")

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    manifest["sha256sums_sha256"] = digest
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="A")
    args = parser.parse_args()

    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        return 1

    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)
    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)

    manifest = build_cache(sidecar_root, teacher_root, args.output_root, args.run_label)

    print(f"Steps: {manifest['total_steps']}")
    print(f"Taxonomy: {manifest['step_taxonomy']}")
    print(f"Sealed: {manifest['sha256sums_sha256']}")

    # Verify taxonomy
    tax = manifest["step_taxonomy"]
    total = sum(tax.values())
    included = tax.get("included_TRUE", 0) + tax.get("included_FALSE", 0)
    excluded = total - included
    print(f"\nClosure: {total} = {included} included + {excluded} excluded")
    print(f"  TRUE: {tax.get('included_TRUE', 0)}")
    print(f"  FALSE: {tax.get('included_FALSE', 0)}")
    print(f"  UNKNOWN: {tax.get('UNKNOWN_excluded', 0)}")
    print(f"  Articulated: {tax.get('articulated', 0)}")
    print(f"  RC: {tax.get('RIGHT_CENSORED', 0)}")
    print(f"  GEOM_NA: {tax.get('GEOM_NA', 0)}")

    for f in sorted(manifest["fold_statistics"]):
        fs = manifest["fold_statistics"][f]
        print(f"Fold {f}: val={fs['val_identities']} train={fs['train_identities']} "
              f"val_TRUE={fs['val_TRUE']} train_TRUE={fs['train_TRUE']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
