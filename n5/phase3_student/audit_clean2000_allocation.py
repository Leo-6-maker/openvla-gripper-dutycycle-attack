"""Metadata-only Clean2000/V22 allocation census.

This script reads only campaign and worker manifest JSON plus file names.  It
never opens episode payloads, labels, logs, or shards, and it fails closed when
the V22 identity root is not explicitly supplied and sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: List[Mapping[str, Any]], fallback_fields: List[str]) -> None:
    fields = list(rows[0]) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def walk_identity_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        suite = value.get("suite")
        task = value.get("task_idx", value.get("task_index"))
        state = value.get("state_id")
        if isinstance(suite, str) and isinstance(task, int) and isinstance(state, int):
            yield value
            return
        for child in value.values():
            yield from walk_identity_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_identity_objects(child)


def manifest_files(root: Path) -> List[Path]:
    return sorted(root.glob("shards/*/worker_*/worker_manifest.jsonl"))


def shard_manifest_files(root: Path) -> List[Path]:
    return sorted(root.glob("shards/*/worker_*/manifest.json"))


def read_shard_metadata(c2f_root: Path) -> tuple[Dict[str, int], Dict[str, Any]]:
    """Read only shard manifests and checksum manifests, never payload files."""
    step_by_parent: Dict[str, int] = {}
    step_conflicts: List[Dict[str, Any]] = []
    inventory: List[Dict[str, Any]] = []
    source_commits: set[str] = set()
    git_commits: set[str] = set()
    schemas: set[str] = set()
    metadata_episode_count = 0
    metadata_step_count_sum = 0
    metadata_step_count_missing = 0
    metadata_malformed: List[Dict[str, Any]] = []
    for manifest in shard_manifest_files(c2f_root):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            metadata_malformed.append({"manifest": str(manifest), "reason": str(exc)})
            continue
        if not isinstance(data, Mapping):
            metadata_malformed.append({"manifest": str(manifest), "reason": "manifest_not_object"})
            continue
        suite = manifest.parent.parent.name
        worker = manifest.parent.name
        schema = data.get("schema")
        if isinstance(schema, str):
            schemas.add(schema)
        for field, target in (("source_commit", source_commits), ("git_commit", git_commits)):
            value = data.get(field)
            if isinstance(value, str):
                target.add(value)
        episodes = data.get("episodes")
        if not isinstance(episodes, list):
            metadata_malformed.append({"manifest": str(manifest), "reason": "episodes_not_list"})
            episodes = []
        declared = data.get("n_episodes")
        if type(declared) is not int or declared != len(episodes):
            metadata_malformed.append({"manifest": str(manifest), "reason": "n_episodes_mismatch"})
        for episode in episodes:
            if not isinstance(episode, Mapping):
                metadata_malformed.append({"manifest": str(manifest), "reason": "episode_not_object"})
                continue
            metadata_episode_count += 1
            parent_key = episode.get("parent_key")
            step_count = episode.get("n_steps")
            if type(step_count) is not int or step_count <= 0:
                metadata_step_count_missing += 1
            else:
                metadata_step_count_sum += step_count
                if isinstance(parent_key, str):
                    previous = step_by_parent.get(parent_key)
                    if previous is not None and previous != step_count:
                        step_conflicts.append({"parent_key": parent_key, "previous": previous, "current": step_count, "manifest": str(manifest)})
                    else:
                        step_by_parent[parent_key] = step_count
        sums = manifest.parent / "SHA256SUMS"
        sidecar = manifest.parent / "SHA256SUMS.sha256"
        entries: List[Tuple[str, str]] = []
        if sums.is_file():
            for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
                    entries.append((parts[0].lower(), parts[1]))
        npz_entries = [(digest, name) for digest, name in entries if name.lower().endswith(".npz")]
        duplicate_names = len(entries) - len({name for _, name in entries})
        unsafe_names = sum(1 for _, name in entries if Path(name).is_absolute() or ".." in Path(name).parts)
        missing_names = sum(1 for _, name in entries if unsafe_names == 0 and not (manifest.parent / name).is_file())
        sidecar_matches = False
        if sidecar.is_file() and sums.is_file():
            sidecar_text = sidecar.read_text(encoding="utf-8", errors="replace").strip().split()
            if sidecar_text:
                sidecar_matches = sidecar_text[0] == sha256_file(sums)
        inventory.append({
            "suite": suite,
            "worker": worker,
            "manifest_json": str(manifest),
            "manifest_json_sha256": sha256_file(manifest),
            "worker_manifest_jsonl": str(manifest.parent / "worker_manifest.jsonl"),
            "worker_manifest_jsonl_sha256": sha256_file(manifest.parent / "worker_manifest.jsonl") if (manifest.parent / "worker_manifest.jsonl").is_file() else "MISSING",
            "sha256sums": str(sums),
            "sha256sums_sha256": sha256_file(sums) if sums.is_file() else "MISSING",
            "sha256sums_sidecar": str(sidecar),
            "sha256sums_sidecar_sha256": sha256_file(sidecar) if sidecar.is_file() else "MISSING",
            "sidecar_matches_sha256sums": sidecar_matches,
            "checksum_entry_count": len(entries),
            "checksum_duplicate_name_count": duplicate_names,
            "checksum_unsafe_path_count": unsafe_names,
            "checksum_missing_path_count": missing_names,
            "npz_entry_count": len(npz_entries),
            "npz_sha256_bindings": json.dumps(npz_entries, separators=(",", ":"), ensure_ascii=False),
            "n_episodes": len(episodes),
            "metadata_step_count_sum": sum(e.get("n_steps", 0) for e in episodes if isinstance(e, Mapping) and type(e.get("n_steps")) is int and e.get("n_steps") > 0),
            "metadata_step_count_missing": sum(1 for e in episodes if not isinstance(e, Mapping) or type(e.get("n_steps")) is not int or e.get("n_steps") <= 0),
            "source_commit": data.get("source_commit", "MISSING"),
            "git_commit": data.get("git_commit", "MISSING"),
            "schema": schema or "MISSING",
        })
    return step_by_parent, {
        "shard_manifest_count": len(shard_manifest_files(c2f_root)),
        "metadata_episode_count": metadata_episode_count,
        "metadata_step_count_sum": metadata_step_count_sum,
        "metadata_step_count_missing": metadata_step_count_missing,
        "metadata_step_count_conflict_count": len(step_conflicts),
        "metadata_malformed_count": len(metadata_malformed),
        "metadata_malformed": metadata_malformed,
        "source_commit_values": sorted(source_commits),
        "git_commit_values": sorted(git_commits),
        "schema_values": sorted(schemas),
        "npz_entry_count": sum(int(row["npz_entry_count"]) for row in inventory),
        "shard_checksum_sidecars_all_match": bool(inventory) and all(row["sidecar_matches_sha256sums"] for row in inventory),
        "shard_checksum_path_closure": bool(inventory) and all(
            row["checksum_duplicate_name_count"] == 0
            and row["checksum_unsafe_path_count"] == 0
            and row["checksum_missing_path_count"] == 0
            for row in inventory
        ),
        "shard_inventory": inventory,
        "window_count_363513_recomputed": False,
        "window_count_363513_status": "NOT_PRESENT_IN_ALLOWED_METADATA",
        "payloads_read": False,
    }


def _step_count(obj: Mapping[str, Any]) -> int | None:
    for field in ("step_count", "episode_step_count", "num_steps", "n_steps", "length"):
        value = obj.get(field)
        if type(value) is int and value > 0:
            return value
    return None


def _identity_key(obj: Mapping[str, Any]) -> Tuple[str, int, int] | None:
    suite = obj.get("suite")
    task = obj.get("task_idx", obj.get("task_index"))
    state = obj.get("state_id")
    if not isinstance(suite, str) or type(task) is not int or type(state) is not int:
        return None
    return suite, task, state


def _metadata_fingerprint(obj: Mapping[str, Any]) -> str:
    fields = (
        "suite", "task_idx", "task_index", "state_id", "init_seed", "seed",
        "step_count", "episode_step_count", "num_steps", "n_steps", "length",
        "task_key", "canonical_parent_key", "artifact_sha256", "source_sha256",
        "task_name", "task_language", "split", "official_horizon", "initial_state_sha256",
    )
    return json.dumps({field: obj[field] for field in fields if field in obj}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _base_row(c2f_root: Path, worker_manifest: Path, obj: Mapping[str, Any], key: Tuple[str, int, int], metadata_step_by_parent: Mapping[str, int]) -> Dict[str, Any]:
    suite, task, state = key
    worker = worker_manifest.parent.name
    suite_dir = worker_manifest.parent.parent.name
    return {
        "episode_id": f"{suite}/task_{task:02d}/state_{state}",
        "suite": suite,
        "task_id": task,
        "state_id": state,
        "seed": obj.get("init_seed", obj.get("seed", "UNKNOWN_METADATA_ONLY")),
        "step_count": _step_count(obj) or metadata_step_by_parent.get(str(obj.get("parent_key")), "UNKNOWN_METADATA_ONLY"),
        "source_root": str(c2f_root),
        "worker_manifest": str(worker_manifest),
        "worker_manifest_sha256": sha256_file(worker_manifest),
        "worker_shard": f"{suite_dir}/{worker}",
        "historical_split": "UNKNOWN_METADATA_ONLY",
        "protected_status": "UNVERIFIED_NO_FIT_BINDING",
        "in_v22_800": "UNKNOWN_ROOT_NOT_MOUNTED",
        "has_v1_1_label": "UNKNOWN_PAYLOAD_NOT_READ",
        "has_v22_label": "UNKNOWN_ROOT_NOT_MOUNTED",
        "has_object_pose": "UNKNOWN_PAYLOAD_NOT_READ",
        "has_target_geometry": "UNKNOWN_PAYLOAD_NOT_READ",
        "has_init_state": "UNKNOWN_PAYLOAD_NOT_READ",
        "has_action_trace": "UNKNOWN_PAYLOAD_NOT_READ",
        "replayable": "UNKNOWN_PAYLOAD_NOT_READ",
        "v23_labelable": "UNKNOWN_PROVENANCE",
    }


def _finalize_records(
    records: List[Dict[str, Any]],
    malformed: List[Dict[str, Any]],
    *,
    manifest_count: int,
    extra_observed: Dict[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Finalize explicit identity groups without silently folding duplicates."""
    by_key: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for record in records:
        key = record["key"]
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(record)
    duplicates = []
    conflicts = []
    unique_rows = []
    for key in sorted(by_key):
        group = by_key[key]
        fingerprints = sorted({str(item["fingerprint"]) for item in group})
        if len(group) > 1:
            duplicates.append({"suite": key[0], "task_id": key[1], "state_id": key[2], "record_count": len(group), "workers": sorted({item["row"].get("worker_shard", "official_manifest") for item in group})})
        if len(fingerprints) > 1:
            conflicts.append({"suite": key[0], "task_id": key[1], "state_id": key[2], "fingerprint_count": len(fingerprints), "workers": sorted({item["row"].get("worker_shard", "official_manifest") for item in group})})
        unique_rows.append(group[0]["row"])

    def grouped_counts(selector):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            name = selector(record["row"])
            if name not in groups:
                groups[name] = []
            groups[name].append(record)
        result = []
        for name in sorted(groups):
            group = groups[name]
            keys = {item["key"] for item in group}
            result.append({
                "group": name,
                "raw_record_count": len(group),
                "unique_identity_count": len(keys),
                "duplicate_record_count": sum(1 for item in group if len(by_key[item["key"]]) > 1),
                "conflict_record_count": sum(1 for item in group if len({x["fingerprint"] for x in by_key[item["key"]]}) > 1),
                "step_count_values": sorted({item["row"].get("step_count") for item in group if type(item["row"].get("step_count")) is int}),
                "official_horizon_values": sorted({item["row"].get("official_horizon") for item in group if type(item["row"].get("official_horizon")) is int}),
            })
        return result

    per_worker = grouped_counts(lambda row: row.get("worker_shard", "official_manifest"))
    per_suite = grouped_counts(lambda row: row["suite"])
    per_task = grouped_counts(lambda row: f"{row['suite']}/task_{row['task_id']:02d}")
    raw_count = len(records) + len(malformed)
    unique_count = len(unique_rows)
    closure = raw_count == 2000 and unique_count == 2000 and not duplicates and not conflicts and not malformed
    observed = {
        "worker_manifest_count": manifest_count,
        "raw_record_count": raw_count,
        "valid_identity_record_count": len(records),
        "unique_identity_count": unique_count,
        "duplicate_identity_count": len(duplicates),
        "duplicate_record_count": sum(int(item["record_count"]) - 1 for item in duplicates),
        "conflict_identity_count": len(conflicts),
        "malformed_record_count": len(malformed),
        "identity_closure_2000": closure,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "malformed": malformed,
        "per_worker_counts": per_worker,
        "per_suite_counts": per_suite,
        "per_task_counts": per_task,
        "observed_identity_count": unique_count,
        "identity_extraction": "manifest_metadata_only",
        "payloads_read": False,
    }
    if extra_observed:
        observed.update(extra_observed)
    return unique_rows, observed


def build_rows(c2f_root: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []
    worker_manifests = manifest_files(c2f_root)
    metadata_step_by_parent, shard_metadata = read_shard_metadata(c2f_root)
    for worker_manifest in worker_manifests:
        for line_no, raw in enumerate(worker_manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                malformed.append({"worker_manifest": str(worker_manifest), "line": line_no, "reason": f"JSON:{exc.msg}"})
                continue
            candidates = list(walk_identity_objects(data))
            if len(candidates) != 1:
                malformed.append({"worker_manifest": str(worker_manifest), "line": line_no, "reason": f"identity_candidate_count={len(candidates)}"})
                continue
            obj = candidates[0]
            key = _identity_key(obj)
            if key is None:
                malformed.append({"worker_manifest": str(worker_manifest), "line": line_no, "reason": "identity_fields_missing_or_invalid"})
                continue
            records.append({
                "key": key,
                "fingerprint": _metadata_fingerprint(obj),
                "row": _base_row(c2f_root, worker_manifest, obj, key, metadata_step_by_parent),
                "worker_manifest": str(worker_manifest),
                "line": line_no,
            })
    return _finalize_records(records, malformed, manifest_count=len(worker_manifests), extra_observed={"shard_metadata": shard_metadata})


def build_official_manifest_rows(manifest_path: Path, source_root: Path, source_provenance: Path | None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"suite", "task_idx", "task_name", "task_language", "state_id", "canonical_parent_key", "split", "official_horizon", "initial_state_sha256"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"official identity manifest missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for line_no, row in enumerate(reader, 2):
            try:
                suite = row["suite"]
                task_id = int(row["task_idx"])
                state_id = int(row["state_id"])
                horizon = int(row["official_horizon"])
                if not suite or not 0 <= task_id < 10 or not 0 <= state_id < 50 or horizon <= 0 or len(row["initial_state_sha256"]) != 64:
                    raise ValueError("invalid identity metadata")
            except (KeyError, TypeError, ValueError) as exc:
                malformed.append({"worker_manifest": str(manifest_path), "line": line_no, "reason": str(exc)})
                continue
            split = (
                "FIT_TRAIN" if state_id < 20 else
                "FIT_DEV" if state_id < 24 else
                "CAL" if state_id < 27 else
                "CHECK" if state_id < 30 else
                "FINAL_EVAL_CANDIDATE"
            )
            obj = {
                "suite": suite,
                "task_idx": task_id,
                "state_id": state_id,
                "task_name": row["task_name"],
                "task_language": row["task_language"],
                "canonical_parent_key": row["canonical_parent_key"],
                "split": row["split"],
                "official_horizon": horizon,
                "initial_state_sha256": row["initial_state_sha256"],
            }
            key = (suite, task_id, state_id)
            records.append({
                "key": key,
                "fingerprint": _metadata_fingerprint(obj),
                "row": {
                    "episode_id": row["canonical_parent_key"],
                    "suite": suite,
                    "task_id": task_id,
                    "state_id": state_id,
                    "task_name": row["task_name"],
                    "task_language": row["task_language"],
                    "seed": "UNKNOWN_PAYLOAD_NOT_READ",
                    "step_count": "UNKNOWN_PAYLOAD_NOT_READ",
                    "official_horizon": horizon,
                    "source_root": str(source_root),
                    "source_manifest": str(manifest_path),
                    "source_manifest_sha256": sha256_file(manifest_path),
                    "historical_split": split,
                    "protected_status": "FIT_TRAIN_METADATA" if split == "FIT_TRAIN" else "PROTECTED_OR_FIT_DEV_METADATA_ONLY",
                    "in_v22_800": "UNKNOWN_ROOT_NOT_MOUNTED",
                    "has_v1_1_label": "UNKNOWN_PAYLOAD_NOT_READ",
                    "has_v22_label": "UNKNOWN_ROOT_NOT_MOUNTED",
                    "has_object_pose": "UNKNOWN_PAYLOAD_NOT_READ",
                    "has_target_geometry": "UNKNOWN_PAYLOAD_NOT_READ",
                    "has_init_state": "UNKNOWN_PAYLOAD_NOT_READ",
                    "has_action_trace": "UNKNOWN_PAYLOAD_NOT_READ",
                    "replayable": "UNKNOWN_PAYLOAD_NOT_READ",
                    "v23_labelable": "UNKNOWN_PROVENANCE",
                    "worker_shard": "official_clean_manifest",
                },
                "worker_manifest": str(manifest_path),
                "line": line_no,
            })
    provenance_summary: Dict[str, Any] = {}
    if source_provenance and source_provenance.is_file():
        provenance = json.loads(source_provenance.read_text(encoding="utf-8"))
        if isinstance(provenance, Mapping):
            for field in (
                "schema", "status", "protocol_id", "collector_revision", "v3_code_local_commit",
                "v3_code_remote_commit", "v3_code_tree_source_sha256", "official_action_parity_gate_sha256",
                "relay_config_sha256", "official_config", "openvla_upstream", "libero_upstream", "runtime",
            ):
                if field in provenance:
                    provenance_summary[field] = provenance[field]
            provenance_summary["collector_source_sha256"] = provenance.get("collector_source_sha256", {})
    extra = {
        "shard_metadata": {
            "shard_manifest_count": 0,
            "metadata_episode_count": 0,
            "metadata_step_count_sum": 0,
            "metadata_step_count_missing": 0,
            "window_count_363513_recomputed": False,
            "window_count_363513_status": "NOT_RECOMPUTED_PAYLOAD_NOT_READ",
            "source_provenance_path": str(source_provenance) if source_provenance else "NOT_SUPPLIED",
            "source_provenance_sha256": sha256_file(source_provenance) if source_provenance and source_provenance.is_file() else "NOT_SUPPLIED",
            "source_provenance_summary": provenance_summary,
            "source_commit_values": [],
            "git_commit_values": [],
            "schema_values": [],
            "shard_checksum_sidecars_all_match": False,
            "shard_checksum_path_closure": False,
            "shard_inventory": [],
            "npz_entry_count": 0,
        },
        "official_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "sidecar_path": str(manifest_path.with_name(manifest_path.name + ".sha256")),
            "sidecar_sha256": sha256_file(manifest_path.with_name(manifest_path.name + ".sha256")) if manifest_path.with_name(manifest_path.name + ".sha256").is_file() else "MISSING",
            "sidecar_matches": (
                manifest_path.with_name(manifest_path.name + ".sha256").is_file()
                and manifest_path.with_name(manifest_path.name + ".sha256").read_text(encoding="utf-8").strip()
                == f"{sha256_file(manifest_path)}  {manifest_path.name}"
            ),
        },
    }
    return _finalize_records(records, malformed, manifest_count=1, extra_observed=extra)


def seal(staging: Path, final: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    if final.exists():
        raise FileExistsError(final)
    write_json(staging / "MANIFEST.json", manifest)
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    sums = staging / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256_file(staging / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    sums_sha = sha256_file(sums)
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.replace(staging, final)
    return {"root": str(final), "sha256sums_sha256": sums_sha}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.c2f_root) == bool(args.clean2000_manifest):
        raise ValueError("supply exactly one of --c2f-root or --clean2000-manifest")
    official_mode = bool(args.clean2000_manifest)
    source_root = Path(args.source_root).resolve() if args.source_root else None
    if official_mode:
        manifest_path = Path(args.clean2000_manifest).resolve()
        if not manifest_path.is_file() or source_root is None or not source_root.is_dir():
            raise ValueError("official manifest and --source-root must exist")
        rows, observed = build_official_manifest_rows(manifest_path, source_root, Path(args.source_provenance).resolve() if args.source_provenance else None)
        root_for_summary = source_root
    else:
        c2f_root = Path(args.c2f_root).resolve()
        if not c2f_root.is_dir():
            raise ValueError(f"C2F root missing: {c2f_root}")
        rows, observed = build_rows(c2f_root)
        root_for_summary = c2f_root
    top_level_seal = all((root_for_summary / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256"))
    identity_closure = bool(observed["identity_closure_2000"])
    v22_status = "ROOT_NOT_MOUNTED" if identity_closure and not args.v22_manifest else ("NOT_ENTERED_R1A" if not identity_closure else "NOT_EXECUTED_METADATA_ONLY")
    decision = "HOLD_PROVENANCE" if identity_closure else "FAIL_IDENTITY_CLOSURE"
    summary = {
        "schema": "C3_S3_CLEAN2000_ALLOCATION_LEDGER_V1",
        "status": decision,
        "decision": decision,
        "r1a_status": "PASS" if identity_closure else "FAIL",
        "r1a_stop_after_closure_failure": not identity_closure,
        "source_root": str(root_for_summary),
        "source_kind": "OFFICIAL_V3_IDENTITY_MANIFEST" if official_mode else "C2F_WORKER_MANIFESTS",
        "c2f_root": str(root_for_summary),
        "c2f_top_level_seal_present": top_level_seal,
        "c2f_worker_shard_count": len(sorted(root_for_summary.glob("shards/*/worker_*"))),
        "official_manifest": observed.get("official_manifest", {}),
        "v22_manifest": args.v22_manifest,
        "v22_status": v22_status,
        "clean2000_claimed_total": 2000,
        "clean2000_claim_source": "OFFICIAL_CLEAN_2000_MANIFEST_V3" if official_mode else "user_statement_not_audited",
        "v22_claimed_total": 800,
        "v22_claim_source": "user_statement_not_audited",
        "observed": observed,
        "allocation_counts": {
            "clean2000_total": "UNVERIFIED",
            "v22_overlap": "UNVERIFIED",
            "unupgraded": "UNVERIFIED",
            "direct_v23_relabel": "UNVERIFIED",
            "deterministic_replay": "UNVERIFIED",
            "minimal_collection": "UNVERIFIED",
            "protected": "UNVERIFIED",
            "train": "UNVERIFIED",
            "validation": "UNVERIFIED",
            "test": "UNVERIFIED",
        },
        "source_binding": (
            {
                "status": "DERIVED_OFFICIAL_V3_METADATA_ONLY",
                "original_immutable_provenance": "TOP_LEVEL_SEAL_MISSING",
                "identity_manifest_sha256": observed["official_manifest"]["sha256"],
                "identity_manifest_sidecar_sha256": observed["official_manifest"]["sidecar_sha256"],
                "identity_manifest_sidecar_matches": observed["official_manifest"]["sidecar_matches"],
                "source_provenance_path": observed["shard_metadata"]["source_provenance_path"],
                "source_provenance_sha256": observed["shard_metadata"]["source_provenance_sha256"],
                "source_provenance_summary": observed["shard_metadata"]["source_provenance_summary"],
                "generation_commit": observed["shard_metadata"]["source_provenance_summary"].get("v3_code_remote_commit", "MISSING"),
                "generation_tree": observed["shard_metadata"]["source_provenance_summary"].get("v3_code_tree_source_sha256", "MISSING"),
                "generation_command": "NOT_PRESENT_IN_ALLOWED_METADATA",
                "generation_environment": observed["shard_metadata"]["source_provenance_summary"].get("runtime", "MISSING"),
                "collector_source_sha256": observed["shard_metadata"]["source_provenance_summary"].get("collector_source_sha256", {}),
                "worker_manifests": 0,
                "shard_inventory": "NOT_APPLICABLE_OFFICIAL_MANIFEST_MODE",
                "npz_sha256_bindings": 0,
                "window_count_363513": observed["shard_metadata"]["window_count_363513_status"],
                "metadata_step_count_sum": "NOT_PRESENT_IN_OFFICIAL_IDENTITY_MANIFEST",
            }
            if official_mode else
            {
                "status": "DERIVED",
                "original_immutable_provenance": "NOT_ESTABLISHED",
                "claimed_source_commits": observed["shard_metadata"]["source_commit_values"],
                "generation_tree": "NOT_PRESENT_IN_ALLOWED_METADATA",
                "generation_command": "NOT_PRESENT_IN_ALLOWED_METADATA",
                "generation_environment": "NOT_PRESENT_IN_ALLOWED_METADATA",
                "worker_manifests": observed["worker_manifest_count"],
                "shard_inventory": "shard_inventory.csv",
                "npz_sha256_bindings": observed["shard_metadata"]["npz_entry_count"],
                "shard_checksum_path_closure": observed["shard_metadata"]["shard_checksum_path_closure"],
                "window_count_363513": observed["shard_metadata"]["window_count_363513_status"],
                "metadata_step_count_sum": observed["shard_metadata"]["metadata_step_count_sum"],
            }
        ),
        "protected_reads": [],
        "payloads_read": False,
        "model_inference": False,
        "training": False,
        "rollout": False,
        "attack": False,
    }
    parent = Path(args.out_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (args.output_name or f"clean2000_allocation_{uuid.uuid4().hex[:8]}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        write_rows(staging / "allocation_ledger.csv", rows, ["episode_id", "status"])
        write_rows(staging / "duplicate_identity_rows.csv", observed["duplicates"], ["suite", "task_id", "state_id", "record_count", "workers"])
        write_rows(staging / "conflict_identity_rows.csv", observed["conflicts"], ["suite", "task_id", "state_id", "fingerprint_count", "workers"])
        write_rows(staging / "malformed_rows.csv", observed["malformed"], ["worker_manifest", "line", "reason"])
        write_rows(staging / "per_worker_counts.csv", observed["per_worker_counts"], ["group", "raw_record_count", "unique_identity_count", "duplicate_record_count", "conflict_record_count", "step_count_values"])
        write_rows(staging / "per_suite_counts.csv", observed["per_suite_counts"], ["group", "raw_record_count", "unique_identity_count", "duplicate_record_count", "conflict_record_count", "step_count_values"])
        write_rows(staging / "per_task_counts.csv", observed["per_task_counts"], ["group", "raw_record_count", "unique_identity_count", "duplicate_record_count", "conflict_record_count", "step_count_values"])
        write_rows(staging / "shard_inventory.csv", observed["shard_metadata"]["shard_inventory"], [
            "suite", "worker", "manifest_json", "manifest_json_sha256", "worker_manifest_jsonl", "worker_manifest_jsonl_sha256",
            "sha256sums", "sha256sums_sha256", "sha256sums_sidecar", "sha256sums_sidecar_sha256", "sidecar_matches_sha256sums",
            "checksum_entry_count", "npz_entry_count", "npz_sha256_bindings", "n_episodes", "metadata_step_count_sum",
            "metadata_step_count_missing", "source_commit", "git_commit", "schema",
        ])
        write_json(staging / "summary.json", summary)
        write_json(staging / "source_binding.json", summary["source_binding"])
        seal(staging, final, {"schema": summary["schema"], "status": summary["status"], "payloads_read": False, "decision": decision})
        return summary | {"output_root": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2f-root")
    parser.add_argument("--clean2000-manifest")
    parser.add_argument("--source-root")
    parser.add_argument("--source-provenance")
    parser.add_argument("--v22-manifest")
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
