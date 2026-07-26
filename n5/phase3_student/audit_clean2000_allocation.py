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
    )
    return json.dumps({field: obj[field] for field in fields if field in obj}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _base_row(c2f_root: Path, worker_manifest: Path, obj: Mapping[str, Any], key: Tuple[str, int, int]) -> Dict[str, Any]:
    suite, task, state = key
    worker = worker_manifest.parent.name
    suite_dir = worker_manifest.parent.parent.name
    return {
        "episode_id": f"{suite}/task_{task:02d}/state_{state}",
        "suite": suite,
        "task_id": task,
        "state_id": state,
        "seed": obj.get("init_seed", obj.get("seed", "UNKNOWN_METADATA_ONLY")),
        "step_count": _step_count(obj) or "UNKNOWN_METADATA_ONLY",
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


def build_rows(c2f_root: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []
    worker_manifests = manifest_files(c2f_root)
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
                "row": _base_row(c2f_root, worker_manifest, obj, key),
                "worker_manifest": str(worker_manifest),
                "line": line_no,
            })
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
            duplicates.append({"suite": key[0], "task_id": key[1], "state_id": key[2], "record_count": len(group), "workers": sorted({item["row"]["worker_shard"] for item in group})})
        if len(fingerprints) > 1:
            conflicts.append({"suite": key[0], "task_id": key[1], "state_id": key[2], "fingerprint_count": len(fingerprints), "workers": sorted({item["row"]["worker_shard"] for item in group})})
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
                "step_count_values": sorted({item["row"]["step_count"] for item in group if type(item["row"]["step_count"]) is int}),
            })
        return result

    per_worker = grouped_counts(lambda row: row["worker_shard"])
    per_suite = grouped_counts(lambda row: row["suite"])
    per_task = grouped_counts(lambda row: f"{row['suite']}/task_{row['task_id']:02d}")
    raw_count = len(records) + len(malformed)
    unique_count = len(unique_rows)
    closure = raw_count == 2000 and unique_count == 2000 and not duplicates and not conflicts and not malformed
    return unique_rows, {
        "worker_manifest_count": len(worker_manifests),
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
    c2f_root = Path(args.c2f_root).resolve()
    if not c2f_root.is_dir():
        raise ValueError(f"C2F root missing: {c2f_root}")
    rows, observed = build_rows(c2f_root)
    top_level_seal = all((c2f_root / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256"))
    identity_closure = bool(observed["identity_closure_2000"])
    v22_status = "ROOT_NOT_MOUNTED" if identity_closure and not args.v22_manifest else ("NOT_ENTERED_R1A" if not identity_closure else "NOT_EXECUTED_METADATA_ONLY")
    decision = "HOLD_PROVENANCE" if identity_closure else "FAIL_IDENTITY_CLOSURE"
    summary = {
        "schema": "C3_S3_CLEAN2000_ALLOCATION_LEDGER_V1",
        "status": decision,
        "decision": decision,
        "r1a_status": "PASS" if identity_closure else "FAIL",
        "r1a_stop_after_closure_failure": not identity_closure,
        "c2f_root": str(c2f_root),
        "c2f_top_level_seal_present": top_level_seal,
        "c2f_worker_shard_count": len(sorted(c2f_root.glob("shards/*/worker_*"))),
        "v22_manifest": args.v22_manifest,
        "v22_status": v22_status,
        "clean2000_claimed_total": 2000,
        "clean2000_claim_source": "user_statement_not_audited",
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
        write_json(staging / "summary.json", summary)
        seal(staging, final, {"schema": summary["schema"], "status": summary["status"], "payloads_read": False, "decision": decision})
        return summary | {"output_root": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2f-root", required=True)
    parser.add_argument("--v22-manifest")
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
