"""P4: Build shared 25D CPU cache for D8-2 CV.

One-time materialization of 670 episodes: 25D features, labels, masks, weights.
GPU training reads only this cache, never raw Teacher JSONL.

Output per episode: features_25d_raw, physical_target, effective_mask, D8_weight, fold_id.
Returns only features/target/mask/weight to model (no Teacher reason, relation, privileged).

Requires --telemetry-root pointing to fresh670_v5_v2_formal for feature materialization.
All features are materialized via D8StreamingFeatureAdapterV3 using the frozen
FIT670_25D_SOURCE_MAPPING field contract.
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
from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3, FEATURE_NAMES

G = 3
FOLD_STATES = {
    0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11],
    3: [12, 13, 14, 15], 4: [16, 17, 18, 19],
}
ARTICULATED = {"libero_goal/task_00", "libero_goal/task_07"}


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.isfinite(arr).all():
        raise ValueError(f"{name}: expected finite shape ({size},), got shape={arr.shape} finite={np.isfinite(arr).all()}")
    return arr


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def _materialize_episode_25d(ep_path: Path) -> np.ndarray:
    """Load episode.json and run D8StreamingFeatureAdapterV3 for all steps."""
    data = json.loads(ep_path.read_text(encoding="utf-8"))
    eid = data.get("episode_id", "?")
    steps = data["steps"]
    teles = data["telemetry"]
    adapter = D8StreamingFeatureAdapterV3()
    prev_eef = None
    features_list = []
    for i, (s, t) in enumerate(zip(steps, teles)):
        raw = _finite_vector(s.get("raw_action_7d"), 7, f"{eid} step {i} raw_action_7d")
        env = _finite_vector(s.get("action_env_7d"), 7, f"{eid} step {i} action_env_7d")
        eef = _finite_vector(t.get("robot0_eef_pos"), 3, f"{eid} step {i} robot0_eef_pos")
        qpos = _finite_vector(t.get("robot0_gripper_qpos"), 2, f"{eid} step {i} robot0_gripper_qpos")
        vel = np.zeros(3) if prev_eef is None else (eef - prev_eef)
        prev_eef = eef
        qsum = float(abs(qpos[0]) + abs(qpos[1]))
        result = adapter.update(
            step_id=i,
            raw_gripper=float(raw[6]), env_gripper=float(env[6]),
            gripper_qpos=qsum, gripper_opening_proxy=qsum,
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(vel[0]), eef_vy=float(vel[1]), eef_vz=float(vel[2]),
            action_dx=float(raw[0]), action_dy=float(raw[1]), action_dz=float(raw[2]),
            action_gripper=float(env[6]),
        )
        if not result.get("valid"):
            raise ValueError(f"{eid} step {i}: V3 adapter rejected: {result.get('error')}")
        values = result.get("features")
        vec = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
        if vec.shape != (25,) or not np.isfinite(vec).all():
            raise ValueError(f"{eid} step {i}: invalid 25D vector shape={vec.shape} finite={np.isfinite(vec).all()}")
        features_list.append(vec)
    return np.array(features_list, dtype=np.float32)


def _build_telemetry_index(telemetry_root: Path) -> dict[str, Path]:
    """Index all episode.json files by episode_id."""
    ep_root = telemetry_root / "episodes"
    index = {}
    for suite in sorted(os.listdir(ep_root)):
        sd = ep_root / suite
        if not sd.is_dir(): continue
        for task in sorted(os.listdir(sd)):
            td = sd / task
            if not td.is_dir(): continue
            for state in sorted(os.listdir(td)):
                stated = td / state
                if not stated.is_dir(): continue
                epf = stated / "episode.json"
                if epf.is_file():
                    meta = json.loads(epf.read_text(encoding="utf-8"))
                    eid = meta.get("episode_id", "")
                    if eid:
                        index[eid] = epf
    return index


def build_cache(
    sidecar_root: Path,
    teacher_root: Path,
    telemetry_root: Path,
    output_root: Path,
    run_label: str,
) -> dict:
    sidecar = load_sidecar_correct(sidecar_root)
    ep_labels, teacher_steps, n_ids = load_teacher_labels(teacher_root)
    sc_set = set(sidecar.keys())

    assert sc_set == set(ep_labels.keys()), f"identity closure fail: sidecar={len(sc_set)} teacher={len(ep_labels)}"
    for eid in sc_set:
        assert set(sidecar[eid].keys()) == set(ep_labels[eid].keys()), f"step mismatch: {eid}"

    # Telemetry index
    ep_file_map = _build_telemetry_index(telemetry_root)
    assert len(ep_file_map) == 670, f"telemetry episodes: {len(ep_file_map)}"
    assert set(ep_file_map.keys()) == sc_set, f"telemetry/sidecar identity mismatch"

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
                assert not fold_val_sets[f1] & fold_val_sets[f2], f"fold {f1}/{f2} overlap"

    # Build cache
    step_taxonomy = Counter()
    cache_entries = []
    all_effective_features = []
    identity_disposition = []  # per-identity accounting

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

        # Materialize 25D features from telemetry
        ep_file = ep_file_map[eid]
        features_25d = _materialize_episode_25d(ep_file)
        assert features_25d.shape[0] == n, f"{eid}: features {features_25d.shape[0]} != steps {n}"

        # Per-identity disposition
        n_effective = 0
        n_true = 0
        n_false = 0
        exclusion = "none"

        if is_art:
            exclusion = "articulated_task"
        elif all(geom_arr):
            exclusion = "GEOM_NA"
        elif all(rc_arr):
            exclusion = "RIGHT_CENSORED"
        else:
            for s in range(n):
                m = masks[s]; rc = rc_arr[s]; geom = geom_arr[s]
                eff = bool(m and not rc and not geom and not is_art)
                if eff:
                    n_effective += 1
                    if labs[s] == 1.0: n_true += 1
                    elif labs[s] == 0.0: n_false += 1
            if n_effective == 0:
                exclusion = "all_steps_masked_or_excluded"

        identity_disposition.append({
            "episode_id": eid,
            "fold_id": fold,
            "raw_steps": n,
            "effective_steps": n_effective,
            "TRUE_steps": n_true,
            "FALSE_steps": n_false,
            "exclusion_category": exclusion,
            "included_in_train": (fold != 0) if n_effective > 0 else False,
            "included_in_val": (fold == 0) if n_effective > 0 else False,
        })

        for s in range(n):
            lab = labels.get(s, {})
            v = lab.get("value", "UNKNOWN")
            m = masks[s]; rc = rc_arr[s]; geom = geom_arr[s]
            effective = bool(m and not rc and not geom and not is_art)

            if is_art: step_taxonomy["articulated"] += 1
            elif geom: step_taxonomy["GEOM_NA"] += 1
            elif rc: step_taxonomy["RIGHT_CENSORED"] += 1
            elif v == "UNKNOWN": step_taxonomy["UNKNOWN_excluded"] += 1
            elif v == "TRUE" and m: step_taxonomy["included_TRUE"] += 1
            elif v == "FALSE" and m: step_taxonomy["included_FALSE"] += 1
            else: step_taxonomy["other"] += 1

            cache_entries.append({
                "episode_id": eid, "step": s,
                "features_25d_raw": features_25d[s].tolist(),
                "physical_target": float(labs[s]) if m and not rc and not geom else -1.0,
                "effective_mask": effective,
                "D8_weight": float(weights[s]),
                "fold_id": fold,
                "right_censored": bool(rc),
                "geometry_not_applicable": bool(geom),
                "articulated": is_art,
            })
            if effective:
                all_effective_features.append(features_25d[s])

    # Verify taxonomy
    total = sum(step_taxonomy.values())
    assert total == 196483, f"step taxonomy sum {total} != 196483"

    all_feats = np.array(all_effective_features, dtype=np.float32)
    print(f"Feature matrix: {all_feats.shape}")
    print(f"All finite: {np.isfinite(all_feats).all()}")
    print(f"All-zero rows: {(~all_feats.any(axis=1)).sum()}")

    # Per-dimension statistics
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
        (ep_dir / f"{safe}.json").write_text(
            json.dumps(ep_entries[eid], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Fold statistics (val=f, train=!f)
    fold_stats = {}
    for f in sorted(FOLD_STATES):
        val_entries = [e for e in cache_entries if e["fold_id"] == f]
        train_entries = [e for e in cache_entries if e["fold_id"] != f]
        fold_stats[str(f)] = {
            "val_identities": len({e["episode_id"] for e in val_entries}),
            "train_identities": len({e["episode_id"] for e in train_entries}),
            "val_steps": len(val_entries), "train_steps": len(train_entries),
            "val_TRUE": sum(1 for e in val_entries if e["physical_target"] == 1.0),
            "train_TRUE": sum(1 for e in train_entries if e["physical_target"] == 1.0),
            "val_FALSE": sum(1 for e in val_entries if e["physical_target"] == 0.0),
            "train_FALSE": sum(1 for e in train_entries if e["physical_target"] == 0.0),
        }

    # Identity disposition
    disp_path = staging / "IDENTITY_DISPOSITION.json"
    disp_path.write_text(json.dumps(identity_disposition, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Account for fully-excluded identities
    excluded_ids = [d for d in identity_disposition if d["exclusion_category"] != "none"]
    included_ids = [d for d in identity_disposition if d["exclusion_category"] == "none"]
    print(f"\nIdentity closure: {len(included_ids)} included + {len(excluded_ids)} excluded = {len(identity_disposition)}")
    excluded_by_cat = Counter(d["exclusion_category"] for d in excluded_ids)
    for cat, cnt in excluded_by_cat.most_common():
        print(f"  {cat}: {cnt}")

    # Provenance
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=ROOT, text=True).strip()

    self_sha = sha256_file(Path(__file__))
    consolidator_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "d8_event_consolidator.py")
    loader_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "load_fit670_25d_telemetry.py")
    adapter_sha = sha256_file(ROOT / "src" / "gripper_attack" / "d8_streaming_features_v3.py")
    schema_sha = sha256_file(ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json")
    mapping_sha = sha256_file(ROOT / "configs" / "FIT670_25D_SOURCE_MAPPING.json")
    sidecar_seal = sha256_file(sidecar_root / "SHA256SUMS.sha256")
    teacher_seal = sha256_file(teacher_root / "SHA256SUMS.sha256")

    manifest = {
        "schema": "DETECTOR_V3_D8_25D_CACHE_V2",
        "status": "BUILT",
        "consumer_eligible": True,
        "run_label": run_label,
        "run_uuid": hashlib.sha256(os.urandom(32)).hexdigest()[:16],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {"commit": commit, "tree": tree},
        "script_provenance": {
            "cache_builder_sha256": self_sha,
            "consolidator_sha256": consolidator_sha,
            "telemetry_loader_sha256": loader_sha,
            "adapter_sha256": adapter_sha,
            "feature_schema_sha256": schema_sha,
            "source_mapping_sha256": mapping_sha,
        },
        "input_seals": {
            "sidecar_sha256sums_sha256": sidecar_seal,
            "teacher_sha256sums_sha256": teacher_seal,
        },
        "G": G, "feature_dim": 25, "feature_names": list(FEATURE_NAMES),
        "total_steps": total, "total_episodes": 670,
        "effective_steps": int(all_feats.shape[0]),
        "all_finite": bool(np.isfinite(all_feats).all()),
        "all_zero_rows": int((~all_feats.any(axis=1)).sum()),
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
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D8 25D cache from FIT670 telemetry")
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True,
                        help="Path to fresh670_v5_v2_formal containing episodes/")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="A")
    args = parser.parse_args()

    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        print("ERROR: clean checkout required — uncommitted changes present")
        return 1

    sidecar_root = args.sidecar_root.resolve(strict=True)
    teacher_root = args.teacher_root.resolve(strict=True)
    telemetry_root = args.telemetry_root.resolve(strict=True)

    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)
    print(f"Sidecar seal: {sidecar_seal['sha256sums_sha256'][:20]}...")
    print(f"Teacher seal: {teacher_seal['sha256sums_sha256'][:20]}...")

    manifest = build_cache(sidecar_root, teacher_root, telemetry_root, args.output_root, args.run_label)

    print(f"\nCache built: {manifest['total_steps']} steps, {manifest['total_episodes']} episodes")
    print(f"Effective steps: {manifest['effective_steps']}")
    print(f"Seal: {manifest['sha256sums_sha256']}")
    print(f"Commit: {manifest['code_snapshot']['commit']}")
    print(f"Finite: {manifest['all_finite']}, Zero rows: {manifest['all_zero_rows']}")
    print(f"Consumer eligible: {manifest['consumer_eligible']}")

    tax = manifest["step_taxonomy"]
    print(f"Taxonomy: TRUE={tax.get('included_TRUE',0)} FALSE={tax.get('included_FALSE',0)} "
          f"UNK={tax.get('UNKNOWN_excluded',0)} RC={tax.get('RIGHT_CENSORED',0)} "
          f"GEOM={tax.get('GEOM_NA',0)} ART={tax.get('articulated',0)}")

    ic = manifest["identity_closure"]
    print(f"Identities: {ic['included']} included + {ic['fully_excluded']} excluded = 670")
    for cat, cnt in ic["excluded_by_category"].items():
        print(f"  {cat}: {cnt}")

    for f in sorted(manifest["fold_statistics"]):
        fs = manifest["fold_statistics"][f]
        print(f"Fold {f}: val={fs['val_identities']} train={fs['train_identities']} "
              f"val_TRUE={fs['val_TRUE']} train_TRUE={fs['train_TRUE']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
