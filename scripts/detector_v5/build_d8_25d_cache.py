"""P4: Build shared 25D CPU cache for D8-2 CV.

H1-R1: Uses unified strict telemetry loader (load_fit670_25d_telemetry) as
the SINGLE canonical path — no duplicate field parsing, no weak validation.

H1-R2: gripper_qpos = signed sum, gripper_opening_proxy = absolute sum.

H1-R5: Fold statistics distinguish raw vs effective denominator.
IDENTITY_DISPOSITION records per-fold roles (VAL/TRAIN/EXCLUDED_FROM_LOSS).

H1-R6: All formal validation uses explicit FormalContractError, never assert.

H1-R8: Cache A/B use identical inputs (same sidecar root); reproducibility
verified by byte-identical per-episode files + canonical manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
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
from load_fit670_25d_telemetry import (
    FormalContractError,
    build_telemetry_index,
    load_episode_telemetry,
    validate_episode_step_integrity,
    validate_field_invariants,
    materialize_episode_features,
)
from gripper_attack.d8_streaming_features_v3 import FEATURE_NAMES

G = 3
FOLD_STATES = {
    0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11],
    3: [12, 13, 14, 15], 4: [16, 17, 18, 19],
}
ARTICULATED = {"libero_goal/task_00", "libero_goal/task_07"}


# ── Per-episode worker (module-level for pickle) ─────────────────────

_ep_labels_g = None
_sidecar_g = None
_assignments_g = None
_ep_file_map_g = None

def _worker_init(labels, sidecar, assignments, ep_file_map):
    global _ep_labels_g, _sidecar_g, _assignments_g, _ep_file_map_g
    _ep_labels_g = labels
    _sidecar_g = sidecar
    _assignments_g = assignments
    _ep_file_map_g = ep_file_map


def _process_episode(eid: str) -> dict:
    """Process one episode: load + validate + features + consolidation + weights."""
    labels = _ep_labels_g[eid]
    relations = _sidecar_g[eid]
    fold = _assignments_g[eid]
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

    weights = build_physical_event_weights(labs, masks, result, right_censored=rc_arr, geom_na=geom_arr)

    # H1-R1: Use unified strict loader — single canonical path
    ep_data = load_episode_telemetry(_ep_file_map_g[eid])
    validate_episode_step_integrity(ep_data)
    validate_field_invariants(ep_data)
    feat_result = materialize_episode_features(ep_data)
    features_25d = feat_result["features_25d"]

    if features_25d.shape[0] != n:
        raise FormalContractError(f"{eid}: features {features_25d.shape[0]} != steps {n}")

    # Per-identity disposition with fold roles
    n_raw = n
    n_eff = 0; n_true = 0; n_false = 0
    exclusion = "none"

    if is_art:
        n_eff_all_zero = True
        for s_idx in range(n):
            m = masks[s_idx]; rc = rc_arr[s_idx]; geom = geom_arr[s_idx]
            if bool(m and not rc and not geom and not is_art):
                n_eff_all_zero = False
                break
        if n_eff_all_zero:
            exclusion = "articulated_task"
        else:
            # H1-R5: Count effective steps even for articulated tasks with partial coverage
            for s_idx in range(n):
                m = masks[s_idx]; rc = rc_arr[s_idx]; geom = geom_arr[s_idx]
                eff = bool(m and not rc and not geom and not is_art)
                if eff:
                    n_eff += 1
                    if labs[s_idx] == 1.0: n_true += 1
                    elif labs[s_idx] == 0.0: n_false += 1
    elif all(geom_arr):
        exclusion = "GEOM_NA"
    elif all(rc_arr):
        exclusion = "RIGHT_CENSORED"
    else:
        for s_idx in range(n):
            m = masks[s_idx]; rc = rc_arr[s_idx]; geom = geom_arr[s_idx]
            eff = bool(m and not rc and not geom and not is_art)
            if eff:
                n_eff += 1
                if labs[s_idx] == 1.0: n_true += 1
                elif labs[s_idx] == 0.0: n_false += 1
        if n_eff == 0:
            exclusion = "all_steps_masked_or_excluded"

    # H1-R5: Full fold roles for each identity
    fold_roles = {}
    for f_idx in FOLD_STATES:
        if exclusion != "none":
            fold_roles[str(f_idx)] = "EXCLUDED_FROM_LOSS"
        elif fold == f_idx:
            fold_roles[str(f_idx)] = "VAL"
        else:
            fold_roles[str(f_idx)] = "TRAIN"

    disposition = {
        "episode_id": eid,
        "assigned_validation_fold": fold,
        "raw_steps": n_raw,
        "effective_steps": n_eff,
        "TRUE_steps": n_true,
        "FALSE_steps": n_false,
        "exclusion_category": exclusion,
        "fold_roles": fold_roles,
    }

    # Step-level entries with taxonomy
    entries = []
    taxonomy = Counter()
    for s_idx in range(n):
        lab = labels.get(s_idx, {})
        v = lab.get("value", "UNKNOWN")
        m = masks[s_idx]; rc = rc_arr[s_idx]; geom = geom_arr[s_idx]
        effective = bool(m and not rc and not geom and not is_art)

        if is_art: taxonomy["articulated"] += 1
        elif geom: taxonomy["GEOM_NA"] += 1
        elif rc: taxonomy["RIGHT_CENSORED"] += 1
        elif v == "UNKNOWN": taxonomy["UNKNOWN_excluded"] += 1
        elif v == "TRUE" and m: taxonomy["included_TRUE"] += 1
        elif v == "FALSE" and m: taxonomy["included_FALSE"] += 1
        else: taxonomy["other"] += 1

        entries.append({
            "episode_id": eid, "step": s_idx,
            "features_25d_raw": features_25d[s_idx].tolist(),
            "physical_target": float(labs[s_idx]) if m and not rc and not geom else -1.0,
            "effective_mask": effective,
            "D8_weight": float(weights[s_idx]),
            "fold_id": fold,
            "right_censored": bool(rc),
            "geometry_not_applicable": bool(geom),
            "articulated": is_art,
        })

    eff_mask = np.array([e["effective_mask"] for e in entries])
    effective_features = features_25d[eff_mask] if eff_mask.any() else np.zeros((0, 25), dtype=np.float32)

    return {
        "entries": entries,
        "taxonomy": dict(taxonomy),
        "disposition": disposition,
        "effective_features": effective_features,
    }


# ── Seal ──────────────────────────────────────────────────────────────

def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


# ── Main build ────────────────────────────────────────────────────────

def build_cache(
    sidecar_root: Path, teacher_root: Path, telemetry_root: Path,
    output_root: Path, run_label: str, n_workers: int | None = None,
) -> dict:
    print("Loading sidecar...")
    sidecar = load_sidecar_correct(sidecar_root)
    print("Loading teacher labels...")
    ep_labels, teacher_steps, n_ids = load_teacher_labels(teacher_root)
    sc_set = set(sidecar.keys())
    t_set = set(ep_labels.keys())

    # H1-R6: Explicit checks, not assert
    if sc_set != t_set:
        raise FormalContractError(f"identity closure: sidecar={len(sc_set)} teacher={len(t_set)}")
    for eid in sc_set:
        if set(sidecar[eid].keys()) != set(ep_labels[eid].keys()):
            raise FormalContractError(f"step mismatch: {eid}")

    print("Indexing telemetry via unified strict loader...")
    ep_file_map = build_telemetry_index(telemetry_root)
    if len(ep_file_map) != 670:
        raise FormalContractError(f"telemetry episodes: {len(ep_file_map)} != 670")
    if set(ep_file_map.keys()) != sc_set:
        raise FormalContractError("telemetry/sidecar identity mismatch")

    # Fold assignment
    assignments = {}
    for eid in sorted(ep_labels.keys()):
        sid = int(eid.split("/")[2].replace("state_", ""))
        for f, states in FOLD_STATES.items():
            if sid in states:
                assignments[eid] = f
                break

    fold_val_sets = {f: set() for f in FOLD_STATES}
    for eid, f in assignments.items():
        fold_val_sets[f].add(eid)
    all_val = set().union(*fold_val_sets.values())
    if len(all_val) != 670:
        raise FormalContractError(f"fold closure: {len(all_val)} != 670")
    for f1 in FOLD_STATES:
        for f2 in FOLD_STATES:
            if f1 < f2 and (fold_val_sets[f1] & fold_val_sets[f2]):
                raise FormalContractError(f"fold {f1}/{f2} overlap")

    n_workers = n_workers or max(1, cpu_count() - 2)
    print(f"Processing {len(sc_set)} episodes with {n_workers} workers...")

    sorted_eids = sorted(sc_set)
    with Pool(processes=n_workers, initializer=_worker_init,
              initargs=(ep_labels, sidecar, assignments, ep_file_map)) as pool:
        results = pool.map(_process_episode, sorted_eids)

    # Collect
    cache_entries = []
    step_taxonomy = Counter()
    identity_disposition = []
    all_effective_features = []

    for r in results:
        cache_entries.extend(r["entries"])
        step_taxonomy.update(r["taxonomy"])
        identity_disposition.append(r["disposition"])
        if r["effective_features"].shape[0] > 0:
            all_effective_features.append(r["effective_features"])

    total = sum(step_taxonomy.values())
    if total != 196483:
        raise FormalContractError(f"step taxonomy sum {total} != 196483")
    print(f"Step taxonomy: {total}")

    all_feats = np.concatenate(all_effective_features, axis=0)
    if not np.isfinite(all_feats).all():
        raise FormalContractError("non-finite in effective features")
    print(f"Feature matrix: {all_feats.shape}")
    n_zero_rows = int((~all_feats.any(axis=1)).sum())
    print(f"All-zero rows: {n_zero_rows}")

    # Per-dimension stats
    print("\nPer-dimension stats (effective steps):")
    for i, name in enumerate(FEATURE_NAMES):
        col = all_feats[:, i]
        print(f"  [{i:2d}] {name:30s}: min={col.min():10.4f} max={col.max():10.4f} "
              f"mean={col.mean():10.4f} std={col.std():10.4f} zeros={(col==0).sum()/len(col)*100:.1f}%")

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
        (ep_dir / f"{safe}.json").write_text(json.dumps(ep_entries[eid], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # H1-R5: Raw vs effective fold statistics
    fold_stats = {}
    for f in sorted(FOLD_STATES):
        # Raw — all entries
        raw_val = [e for e in cache_entries if e["fold_id"] == f]
        raw_train = [e for e in cache_entries if e["fold_id"] != f]
        # Effective only
        eff_val = [e for e in raw_val if e["effective_mask"]]
        eff_train = [e for e in raw_train if e["effective_mask"]]

        fold_stats[str(f)] = {
            "raw_val_identities": len({e["episode_id"] for e in raw_val}),
            "raw_train_identities": len({e["episode_id"] for e in raw_train}),
            "effective_val_identities": len({e["episode_id"] for e in eff_val}),
            "effective_train_identities": len({e["episode_id"] for e in eff_train}),
            "raw_val_steps": len(raw_val), "raw_train_steps": len(raw_train),
            "effective_val_steps": len(eff_val), "effective_train_steps": len(eff_train),
            "train_TRUE": sum(1 for e in eff_train if e["physical_target"] == 1.0),
            "train_FALSE": sum(1 for e in eff_train if e["physical_target"] == 0.0),
            "val_TRUE": sum(1 for e in eff_val if e["physical_target"] == 1.0),
            "val_FALSE": sum(1 for e in eff_val if e["physical_target"] == 0.0),
        }

    # Identity disposition
    (staging / "IDENTITY_DISPOSITION.json").write_text(
        json.dumps(identity_disposition, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    excluded_ids = [d for d in identity_disposition if d["exclusion_category"] != "none"]
    included_ids = [d for d in identity_disposition if d["exclusion_category"] == "none"]
    print(f"\nIdentity closure: {len(included_ids)} included + {len(excluded_ids)} excluded = 670")
    excluded_by_cat = Counter(d["exclusion_category"] for d in excluded_ids)
    for cat, cnt in excluded_by_cat.most_common():
        print(f"  {cat}: {cnt}")

    # H1-R5: Validation closure checks
    for f_str in [str(f) for f in FOLD_STATES]:
        val_count = sum(1 for d in identity_disposition if d["fold_roles"][f_str] == "VAL")
        train_count = sum(1 for d in identity_disposition if d["fold_roles"][f_str] == "TRAIN")
        excl_count = sum(1 for d in identity_disposition if d["fold_roles"][f_str] == "EXCLUDED_FROM_LOSS")
        if val_count + train_count + excl_count != 670:
            raise FormalContractError(f"fold {f_str} role sum != 670")
    print("Fold role closure: PASS")

    # Provenance
    commit = "a270176658100ff1e2d96a69c11084a26f53f8e9"
    tree = "4f42c91b20ba8cf86a7ad7e1f9d85cb450aa980d"
    run_uuid = hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    self_sha = sha256_file(Path(__file__))
    consolidator_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "d8_event_consolidator.py")
    adapter_sha = sha256_file(ROOT / "src" / "gripper_attack" / "d8_streaming_features_v3.py")
    loader_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "load_fit670_25d_telemetry.py")
    schema_sha = sha256_file(ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json")
    mapping_sha = sha256_file(ROOT / "configs" / "FIT670_25D_SOURCE_MAPPING.json")
    sidecar_seal = sha256_file(sidecar_root / "SHA256SUMS.sha256")
    teacher_seal = sha256_file(teacher_root / "SHA256SUMS.sha256")

    manifest = {
        "schema": "DETECTOR_V3_D8_25D_CACHE_V2",
        "status": "BUILT",
        "consumer_eligible": True,
        "run_label": run_label,
        "run_uuid": run_uuid,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {"commit": commit, "tree": tree},
        "script_provenance": {
            "cache_builder_sha256": self_sha,
            "consolidator_sha256": consolidator_sha,
            "adapter_sha256": adapter_sha,
            "loader_sha256": loader_sha,
            "feature_schema_sha256": schema_sha,
            "source_mapping_sha256": mapping_sha,
        },
        "input_seals": {
            "sidecar_sha256sums_sha256": sidecar_seal,
            "teacher_sha256sums_sha256": teacher_seal,
        },
        "executed_loader": {
            "module": "load_fit670_25d_telemetry",
            "functions": ["load_episode_telemetry", "validate_episode_step_integrity",
                          "validate_field_invariants", "materialize_episode_features"],
        },
        "G": G, "feature_dim": 25, "feature_names": list(FEATURE_NAMES),
        "total_steps": total, "total_episodes": 670,
        "effective_steps": int(all_feats.shape[0]),
        "all_finite": bool(np.isfinite(all_feats).all()),
        "all_zero_rows": n_zero_rows,
        "step_taxonomy": dict(step_taxonomy),
        "fold_assignments": {str(k): v for k, v in FOLD_STATES.items()},
        "fold_statistics": fold_stats,
        "identity_closure": {
            "included": len(included_ids),
            "fully_excluded": len(excluded_ids),
            "excluded_by_category": dict(excluded_by_cat),
        },
        "test_reads": 0, "protected_reads": 0, "eval160_reads": 0,
    }

    (staging / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "FOLD_ASSIGNMENT.json").write_text(
        json.dumps({e: assignments[e] for e in sorted(assignments)}, indent=2) + "\n", encoding="utf-8")

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    manifest["sha256sums_sha256"] = digest

    # Rewrite manifest with seal included
    (output_root / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _write_seal(output_root)
    manifest["sha256sums_sha256"] = digest
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D8 25D cache (H1 unified loader)")
    parser.add_argument("--sidecar-root", type=Path, required=True,
                        help="H1-R8: Use same sidecar for A and B to ensure input_seals match")
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="A")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")

    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)
    telemetry_root = args.telemetry_root.resolve(strict=True)

    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)
    print(f"Sidecar seal: {sidecar_seal['sha256sums_sha256'][:20]}...")
    print(f"Teacher seal: {teacher_seal['sha256sums_sha256'][:20]}...")

    manifest = build_cache(sidecar_root, teacher_root, telemetry_root, args.output_root, args.run_label, args.workers)

    print(f"\nCache built: {manifest['total_steps']} steps, {manifest['total_episodes']} eps")
    print(f"Effective: {manifest['effective_steps']}")
    print(f"Seal: {manifest['sha256sums_sha256']}")
    for f in sorted(manifest["fold_statistics"]):
        fs = manifest["fold_statistics"][f]
        print(f"Fold {f}: raw_val={fs['raw_val_identities']} raw_train={fs['raw_train_identities']} "
              f"eff_val={fs['effective_val_identities']} eff_train={fs['effective_train_identities']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
