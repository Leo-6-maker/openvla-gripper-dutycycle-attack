"""Build the formal shared 25D cache for Detector-v3 D8 cross-validation.

The formal builder is fail-closed:
- exact deployed source bytes are bound by an external SOURCE_SNAPSHOT_V2;
- sidecar, Teacher and every consumed telemetry JSON are sealed and verified;
- the strict telemetry loader is the only feature-materialization path;
- UNKNOWN/right-censored/geometry-N/A/articulated steps never enter loss;
- all identity, fold, event, span and weight denominators are explicit;
- output is immutable, non-consumer-eligible, and atomically published.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal
from d8_event_consolidator import consolidate_physical_events, build_physical_event_weights
from d8_source_contract import (
    CACHE_REQUIRED_SOURCE_FILES,
    SourceContractError,
    load_and_validate_source_snapshot,
    verify_sha256_manifest,
)
from gripper_attack.d8_streaming_features_v3 import FEATURE_NAMES
from gripper_attack.seal_utils import rename_noreplace
from load_fit670_25d_telemetry import (
    FormalContractError,
    build_telemetry_index,
    load_episode_telemetry,
    materialize_episode_features,
    validate_episode_step_integrity,
    validate_field_invariants,
)
from run_d8_formal_g_sensitivity import load_sidecar_correct, load_teacher_labels

G = 3
FOLD_STATES = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6, 7],
    2: [8, 9, 10, 11],
    3: [12, 13, 14, 15],
    4: [16, 17, 18, 19],
}
ARTICULATED = {"libero_goal/task_00", "libero_goal/task_07"}
EXPECTED_EPISODES = 670
EXPECTED_RAW_STEPS = 196_483
EXPECTED_EFFECTIVE_STEPS = 179_674
EXPECTED_INCLUDED_IDENTITIES = 643
EXPECTED_ARTICULATED_IDENTITIES = 27

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


def _count_negative_spans(effective: np.ndarray, labels: np.ndarray) -> int:
    spans = 0
    in_span = False
    for is_effective, target in zip(effective.tolist(), labels.tolist()):
        is_negative = bool(is_effective and target == 0.0)
        if is_negative and not in_span:
            spans += 1
        in_span = is_negative
    return spans


def _ess(weights: list[float]) -> float:
    if not weights:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.square(w).sum())
    if denom <= 0.0:
        return 0.0
    return float(np.square(w.sum()) / denom)


def _process_episode(eid: str) -> dict:
    labels = _ep_labels_g[eid]
    relations = _sidecar_g[eid]
    fold = _assignments_g[eid]
    task_key = "/".join(eid.split("/")[:2])
    is_articulated = task_key in ARTICULATED

    event_result = consolidate_physical_events(eid, labels, relations=relations, G=G)

    if not labels:
        raise FormalContractError(f"{eid}: empty Teacher labels")
    max_step = max(labels)
    n_steps = max_step + 1
    if set(labels) != set(range(n_steps)):
        raise FormalContractError(f"{eid}: Teacher steps are not zero-based contiguous")

    targets = np.full(n_steps, -1.0, dtype=np.float32)
    teacher_mask = np.zeros(n_steps, dtype=bool)
    right_censored = np.zeros(n_steps, dtype=bool)
    geometry_na = np.zeros(n_steps, dtype=bool)

    for step, label in labels.items():
        value = label.get("value", "UNKNOWN")
        if value == "TRUE":
            targets[step] = 1.0
        elif value == "FALSE":
            targets[step] = 0.0
        elif value != "UNKNOWN":
            raise FormalContractError(f"{eid}/{step}: invalid Teacher value {value!r}")
        teacher_mask[step] = bool(label.get("mask", False) and label.get("valid_mask", False))
        right_censored[step] = bool(label.get("right_censored", False))
        geometry_na[step] = label.get("reason") == "GEOMETRY_NOT_APPLICABLE"

    effective = teacher_mask & ~right_censored & ~geometry_na & (not is_articulated)
    if np.any(effective & ~np.isin(targets, [0.0, 1.0])):
        raise FormalContractError(f"{eid}: effective entries contain non-binary targets")

    weights = np.asarray(build_physical_event_weights(
        targets,
        teacher_mask,
        event_result,
        right_censored=right_censored,
        geom_na=geometry_na,
    ), dtype=np.float64)
    weights[~effective] = 0.0
    if np.any(effective & (~np.isfinite(weights) | (weights <= 0.0))):
        raise FormalContractError(f"{eid}: effective entries require finite positive weights")
    if np.any(~effective & (weights != 0.0)):
        raise FormalContractError(f"{eid}: excluded entries must have zero weight")

    episode_data = load_episode_telemetry(_ep_file_map_g[eid])
    validate_episode_step_integrity(episode_data)
    validate_field_invariants(episode_data)
    feature_result = materialize_episode_features(episode_data)
    features_25d = feature_result["features_25d"]
    if features_25d.shape != (n_steps, 25):
        raise FormalContractError(
            f"{eid}: expected feature shape {(n_steps, 25)}, got {features_25d.shape}"
        )
    if not np.isfinite(features_25d).all():
        raise FormalContractError(f"{eid}: non-finite feature value")

    if is_articulated:
        exclusion = "articulated_task"
    elif int(effective.sum()) == 0:
        if bool(geometry_na.all()):
            exclusion = "GEOM_NA"
        elif bool(right_censored.all()):
            exclusion = "RIGHT_CENSORED"
        else:
            exclusion = "all_steps_masked_or_excluded"
    else:
        exclusion = "none"

    fold_roles = {}
    for fold_idx in FOLD_STATES:
        if exclusion != "none":
            fold_roles[str(fold_idx)] = "EXCLUDED_FROM_LOSS"
        elif fold == fold_idx:
            fold_roles[str(fold_idx)] = "VAL"
        else:
            fold_roles[str(fold_idx)] = "TRAIN"

    taxonomy = Counter()
    entries = []
    for step in range(n_steps):
        value = labels[step].get("value", "UNKNOWN")
        if is_articulated:
            taxonomy["articulated"] += 1
        elif right_censored[step]:
            taxonomy["RIGHT_CENSORED"] += 1
        elif geometry_na[step]:
            taxonomy["GEOM_NA"] += 1
        elif effective[step] and targets[step] == 1.0:
            taxonomy["included_TRUE"] += 1
        elif effective[step] and targets[step] == 0.0:
            taxonomy["included_FALSE"] += 1
        elif value == "UNKNOWN" or not teacher_mask[step]:
            taxonomy["UNKNOWN_excluded"] += 1
        else:
            taxonomy["other_excluded"] += 1

        entries.append({
            "episode_id": eid,
            "step": step,
            "features_25d_raw": features_25d[step].tolist(),
            "physical_target": float(targets[step]) if effective[step] else -1.0,
            "effective_mask": bool(effective[step]),
            "D8_weight": float(weights[step]),
            "fold_id": fold,
            "right_censored": bool(right_censored[step]),
            "geometry_not_applicable": bool(geometry_na[step]),
            "articulated": is_articulated,
        })

    disposition = {
        "episode_id": eid,
        "assigned_validation_fold": fold,
        "raw_steps": n_steps,
        "effective_steps": int(effective.sum()),
        "TRUE_steps": int(np.sum(effective & (targets == 1.0))),
        "FALSE_steps": int(np.sum(effective & (targets == 0.0))),
        "exclusion_category": exclusion,
        "fold_roles": fold_roles,
    }
    episode_stats = {
        "episode_id": eid,
        "fold_id": fold,
        "included": exclusion == "none",
        "positive_events": int(event_result.get("consolidated_event_count", 0)) if exclusion == "none" else 0,
        "negative_spans": _count_negative_spans(effective, targets),
    }

    return {
        "entries": entries,
        "taxonomy": dict(taxonomy),
        "disposition": disposition,
        "episode_stats": episode_stats,
        "effective_features": features_25d[effective],
    }


def _write_seal(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def _validate_identity_roles(identity_disposition: list[dict]) -> dict:
    if len(identity_disposition) != EXPECTED_EPISODES:
        raise FormalContractError(
            f"identity disposition rows {len(identity_disposition)} != {EXPECTED_EPISODES}"
        )
    ids = [row["episode_id"] for row in identity_disposition]
    if len(set(ids)) != EXPECTED_EPISODES:
        raise FormalContractError("identity disposition has duplicate episode_id")

    included = [row for row in identity_disposition if row["exclusion_category"] == "none"]
    excluded = [row for row in identity_disposition if row["exclusion_category"] != "none"]
    if len(included) != EXPECTED_INCLUDED_IDENTITIES:
        raise FormalContractError(f"included identities {len(included)} != {EXPECTED_INCLUDED_IDENTITIES}")
    excluded_by_category = Counter(row["exclusion_category"] for row in excluded)
    if excluded_by_category != Counter({"articulated_task": EXPECTED_ARTICULATED_IDENTITIES}):
        raise FormalContractError(f"unexpected excluded identity taxonomy: {dict(excluded_by_category)}")

    effective_val_union = set()
    assigned_val_union = set()
    for row in identity_disposition:
        roles = row["fold_roles"]
        if set(roles) != {str(i) for i in FOLD_STATES}:
            raise FormalContractError(f"{row['episode_id']}: incomplete fold_roles")
        assigned_val_union.add((row["episode_id"], row["assigned_validation_fold"]))
        values = list(roles.values())
        if row["exclusion_category"] == "none":
            if values.count("VAL") != 1 or values.count("TRAIN") != 4:
                raise FormalContractError(f"{row['episode_id']}: included role closure failed")
            effective_val_union.add(row["episode_id"])
        elif values.count("EXCLUDED_FROM_LOSS") != 5:
            raise FormalContractError(f"{row['episode_id']}: excluded role closure failed")

    if len(assigned_val_union) != EXPECTED_EPISODES:
        raise FormalContractError("assigned validation identity closure failed")
    if len(effective_val_union) != EXPECTED_INCLUDED_IDENTITIES:
        raise FormalContractError("effective validation union closure failed")

    return {
        "included": len(included),
        "fully_excluded": len(excluded),
        "excluded_by_category": dict(excluded_by_category),
        "assigned_validation_union": len(assigned_val_union),
        "effective_validation_union": len(effective_val_union),
    }


def _fold_statistics(entries: list[dict], episode_stats: list[dict]) -> dict:
    stats = {}
    for fold in sorted(FOLD_STATES):
        raw_val = [entry for entry in entries if entry["fold_id"] == fold]
        raw_train = [entry for entry in entries if entry["fold_id"] != fold]
        effective_val = [entry for entry in raw_val if entry["effective_mask"]]
        effective_train = [entry for entry in raw_train if entry["effective_mask"]]
        val_episode_stats = [row for row in episode_stats if row["fold_id"] == fold and row["included"]]
        train_episode_stats = [row for row in episode_stats if row["fold_id"] != fold and row["included"]]

        train_pos_weights = [entry["D8_weight"] for entry in effective_train if entry["physical_target"] == 1.0]
        train_neg_weights = [entry["D8_weight"] for entry in effective_train if entry["physical_target"] == 0.0]
        val_pos_weights = [entry["D8_weight"] for entry in effective_val if entry["physical_target"] == 1.0]
        val_neg_weights = [entry["D8_weight"] for entry in effective_val if entry["physical_target"] == 0.0]

        stats[str(fold)] = {
            "raw_val_identities": len({entry["episode_id"] for entry in raw_val}),
            "raw_train_identities": len({entry["episode_id"] for entry in raw_train}),
            "effective_val_identities": len({entry["episode_id"] for entry in effective_val}),
            "effective_train_identities": len({entry["episode_id"] for entry in effective_train}),
            "raw_val_steps": len(raw_val),
            "raw_train_steps": len(raw_train),
            "effective_val_steps": len(effective_val),
            "effective_train_steps": len(effective_train),
            "train_TRUE": sum(entry["physical_target"] == 1.0 for entry in effective_train),
            "train_FALSE": sum(entry["physical_target"] == 0.0 for entry in effective_train),
            "val_TRUE": sum(entry["physical_target"] == 1.0 for entry in effective_val),
            "val_FALSE": sum(entry["physical_target"] == 0.0 for entry in effective_val),
            "train_positive_events": sum(row["positive_events"] for row in train_episode_stats),
            "val_positive_events": sum(row["positive_events"] for row in val_episode_stats),
            "train_negative_spans": sum(row["negative_spans"] for row in train_episode_stats),
            "val_negative_spans": sum(row["negative_spans"] for row in val_episode_stats),
            "train_positive_ESS": _ess(train_pos_weights),
            "train_negative_ESS": _ess(train_neg_weights),
            "val_positive_ESS": _ess(val_pos_weights),
            "val_negative_ESS": _ess(val_neg_weights),
        }
    return stats


def build_cache(
    sidecar_root: Path,
    teacher_root: Path,
    telemetry_root: Path,
    source_snapshot_path: Path,
    output_root: Path,
    run_label: str,
    n_workers: int | None = None,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")

    try:
        source = load_and_validate_source_snapshot(
            source_snapshot_path, ROOT, CACHE_REQUIRED_SOURCE_FILES
        )
    except SourceContractError as exc:
        raise FormalContractError(str(exc)) from exc
    print(
        "Source snapshot validated: "
        f"commit={source['executable_source_commit'][:12]} "
        f"files={len(source['file_sha256_map'])}"
    )

    sidecar_receipt = verify_seal(sidecar_root)
    teacher_receipt = verify_seal(teacher_root)

    print("Loading sidecar...")
    sidecar = load_sidecar_correct(sidecar_root)
    print("Loading Teacher labels...")
    episode_labels, teacher_steps, identity_count = load_teacher_labels(teacher_root)
    sidecar_ids = set(sidecar)
    teacher_ids = set(episode_labels)
    if sidecar_ids != teacher_ids:
        raise FormalContractError(
            f"identity closure mismatch: sidecar={len(sidecar_ids)} Teacher={len(teacher_ids)}"
        )
    if identity_count != EXPECTED_EPISODES or len(teacher_ids) != EXPECTED_EPISODES:
        raise FormalContractError(
            f"Teacher identities: reported={identity_count} actual={len(teacher_ids)}"
        )
    if teacher_steps != EXPECTED_RAW_STEPS:
        raise FormalContractError(f"Teacher steps {teacher_steps} != {EXPECTED_RAW_STEPS}")
    for eid in sidecar_ids:
        if set(sidecar[eid]) != set(episode_labels[eid]):
            raise FormalContractError(f"{eid}: sidecar/Teacher step mismatch")

    print("Indexing and verifying telemetry...")
    episode_file_map = build_telemetry_index(telemetry_root)
    if len(episode_file_map) != EXPECTED_EPISODES:
        raise FormalContractError(
            f"telemetry episodes {len(episode_file_map)} != {EXPECTED_EPISODES}"
        )
    if set(episode_file_map) != sidecar_ids:
        raise FormalContractError("telemetry/sidecar identity mismatch")
    try:
        telemetry_receipt = verify_sha256_manifest(
            telemetry_root,
            required_files=episode_file_map.values(),
            require_all_files_listed=False,
        )
    except SourceContractError as exc:
        raise FormalContractError(str(exc)) from exc

    assignments = {}
    for eid in sorted(teacher_ids):
        try:
            state_id = int(eid.split("/")[2].replace("state_", ""))
        except (IndexError, ValueError) as exc:
            raise FormalContractError(f"invalid episode identity: {eid}") from exc
        matched = [fold for fold, states in FOLD_STATES.items() if state_id in states]
        if len(matched) != 1:
            raise FormalContractError(f"{eid}: state maps to {matched} folds")
        assignments[eid] = matched[0]
    if len(assignments) != EXPECTED_EPISODES:
        raise FormalContractError("fold assignment identity closure failed")
    val_sets = {fold: {eid for eid, assigned in assignments.items() if assigned == fold} for fold in FOLD_STATES}
    if len(set().union(*val_sets.values())) != EXPECTED_EPISODES:
        raise FormalContractError("fold validation union failed")
    for left in FOLD_STATES:
        for right in FOLD_STATES:
            if left < right and val_sets[left] & val_sets[right]:
                raise FormalContractError(f"fold {left}/{right} overlap")

    worker_count = n_workers or max(1, cpu_count() - 2)
    print(f"Processing {len(teacher_ids)} episodes with {worker_count} workers...")
    sorted_ids = sorted(teacher_ids)
    with Pool(
        processes=worker_count,
        initializer=_worker_init,
        initargs=(episode_labels, sidecar, assignments, episode_file_map),
    ) as pool:
        results = pool.map(_process_episode, sorted_ids)

    cache_entries: list[dict] = []
    step_taxonomy = Counter()
    identity_disposition: list[dict] = []
    episode_stats: list[dict] = []
    effective_feature_blocks = []
    for result in results:
        cache_entries.extend(result["entries"])
        step_taxonomy.update(result["taxonomy"])
        identity_disposition.append(result["disposition"])
        episode_stats.append(result["episode_stats"])
        if result["effective_features"].shape[0] > 0:
            effective_feature_blocks.append(result["effective_features"])

    if len(cache_entries) != EXPECTED_RAW_STEPS:
        raise FormalContractError(f"cache entries {len(cache_entries)} != {EXPECTED_RAW_STEPS}")
    if sum(step_taxonomy.values()) != EXPECTED_RAW_STEPS:
        raise FormalContractError("step taxonomy does not close")
    if step_taxonomy.get("other_excluded", 0) != 0:
        raise FormalContractError(f"unexpected other_excluded steps: {step_taxonomy['other_excluded']}")

    all_effective_features = np.concatenate(effective_feature_blocks, axis=0)
    if all_effective_features.shape != (EXPECTED_EFFECTIVE_STEPS, 25):
        raise FormalContractError(
            f"effective feature shape {all_effective_features.shape} != {(EXPECTED_EFFECTIVE_STEPS, 25)}"
        )
    if not np.isfinite(all_effective_features).all():
        raise FormalContractError("effective features contain non-finite values")

    identity_closure = _validate_identity_roles(identity_disposition)
    fold_stats = _fold_statistics(cache_entries, episode_stats)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    per_episode = staging / "per_episode"
    per_episode.mkdir()
    grouped_entries = defaultdict(list)
    for entry in cache_entries:
        grouped_entries[entry["episode_id"]].append(entry)
    if len(grouped_entries) != EXPECTED_EPISODES:
        raise FormalContractError("per-episode output identity closure failed")
    for eid in sorted(grouped_entries):
        safe_name = eid.replace("/", "_") + ".json"
        (per_episode / safe_name).write_text(
            json.dumps(grouped_entries[eid], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    identity_disposition.sort(key=lambda row: row["episode_id"])
    (staging / "IDENTITY_DISPOSITION.json").write_text(
        json.dumps(identity_disposition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "FOLD_ASSIGNMENT.json").write_text(
        json.dumps({eid: assignments[eid] for eid in sorted(assignments)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_uuid = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    manifest = {
        "schema": "DETECTOR_V3_D8_25D_CACHE_V3",
        "status": "BUILT_PENDING_H1",
        "consumer_eligible": False,
        "run_label": run_label,
        "run_uuid": run_uuid,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {
            "executable_source_commit": source["executable_source_commit"],
            "executable_source_tree": source["executable_source_tree"],
            "source_snapshot_sha256": source["source_snapshot_sha256"],
            "validated_required_file_count": len(CACHE_REQUIRED_SOURCE_FILES),
        },
        "script_provenance": {
            "cache_builder_sha256": sha256_file(Path(__file__)),
            "source_contract_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_source_contract.py"),
            "consolidator_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_event_consolidator.py"),
            "adapter_sha256": sha256_file(ROOT / "src/gripper_attack/d8_streaming_features_v3.py"),
            "loader_sha256": sha256_file(ROOT / "scripts/detector_v5/load_fit670_25d_telemetry.py"),
            "feature_schema_sha256": sha256_file(ROOT / "configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"),
            "source_mapping_sha256": sha256_file(ROOT / "configs/FIT670_25D_SOURCE_MAPPING.json"),
        },
        "input_seals": {
            "sidecar_sha256sums_sha256": sha256_file(sidecar_root / "SHA256SUMS.sha256"),
            "teacher_sha256sums_sha256": sha256_file(teacher_root / "SHA256SUMS.sha256"),
            "telemetry_sha256sums_sha256": telemetry_receipt["sha256sums_sidecar_sha256"],
            "telemetry_manifest_sha256": telemetry_receipt["sha256sums_sha256"],
        },
        "input_verification": {
            "sidecar": sidecar_receipt,
            "teacher": teacher_receipt,
            "telemetry_listed_file_count": telemetry_receipt["listed_file_count"],
            "telemetry_consumed_episode_files": len(episode_file_map),
        },
        "executed_loader": {
            "module": "load_fit670_25d_telemetry",
            "functions": [
                "load_episode_telemetry",
                "validate_episode_step_integrity",
                "validate_field_invariants",
                "materialize_episode_features",
            ],
        },
        "G": G,
        "feature_dim": 25,
        "feature_names": list(FEATURE_NAMES),
        "total_steps": EXPECTED_RAW_STEPS,
        "total_episodes": EXPECTED_EPISODES,
        "effective_steps": EXPECTED_EFFECTIVE_STEPS,
        "all_finite": True,
        "all_zero_effective_rows": int((~all_effective_features.any(axis=1)).sum()),
        "step_taxonomy": dict(step_taxonomy),
        "fold_assignments": {str(key): value for key, value in FOLD_STATES.items()},
        "fold_statistics": fold_stats,
        "identity_closure": identity_closure,
        "test_reads": 0,
        "protected_reads": 0,
        "eval160_reads": 0,
    }
    (staging / "CACHE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    manifest["sha256sums_sha256"] = digest
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build formal D8 25D cache V3")
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    manifest = build_cache(
        args.sidecar_root.resolve(strict=True),
        args.teacher_root.resolve(strict=True),
        args.telemetry_root.resolve(strict=True),
        args.source_snapshot.resolve(strict=True),
        args.output_root,
        args.run_label,
        args.workers,
    )
    print(f"Cache built: {manifest['total_steps']} steps / {manifest['total_episodes']} episodes")
    print(f"Effective steps: {manifest['effective_steps']}")
    print(f"Seal: {manifest['sha256sums_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
