"""Metadata-only D0 Clean2000 allocation audit.

The custodian may read protected manifests only to compute aggregate set
statistics.  It never writes identity rows, identity strings, or protected
manifest contents to the receipt.
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
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_field(value: Any, prefixes: tuple[str, ...]) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:" + "|".join(map(re.escape, prefixes)) + r")[_-]?(\d+)", value, re.I)
    return int(match.group(1)) if match else None


def identity_key(value: Mapping[str, Any]) -> tuple[str, int, int] | None:
    suite = value.get("suite") or value.get("suite_name")
    task = next((value.get(name) for name in ("task_idx", "task_index", "task_id", "task") if name in value), None)
    state = next((value.get(name) for name in ("state_id", "state_idx", "state") if name in value), None)
    task_id = _int_field(task, ("task", "state"))
    state_id = _int_field(state, ("state", "step"))
    if task_id is None and type(task) is int:
        task_id = task
    if state_id is None and type(state) is int:
        state_id = state
    if isinstance(suite, str) and task_id is not None and state_id is not None:
        return suite, task_id, state_id
    return None


def walk_identity_maps(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if identity_key(value) is not None:
            yield value
            return
        for child in value.values():
            yield from walk_identity_maps(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_identity_maps(child)


def parse_identity_string(value: Any) -> tuple[str, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([^/]+)/task[_-]?(\d+)/state[_-]?(\d+)", value)
    return (match.group(1), int(match.group(2)), int(match.group(3))) if match else None


def walk_identity_entries(value: Any) -> Iterable[tuple[tuple[str, int, int], str]]:
    """Yield key/fingerprint pairs without retaining or printing identity values."""
    if isinstance(value, str):
        key = parse_identity_string(value)
        if key is not None:
            yield key, hashlib.sha256(value.encode("utf-8")).hexdigest()
    elif isinstance(value, Mapping):
        key = identity_key(value)
        if key is not None:
            yield key, _fingerprint(value)
            return
        for child in value.values():
            yield from walk_identity_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_identity_entries(child)


def _fingerprint(value: Mapping[str, Any]) -> str:
    # Only used internally to detect conflicting duplicate rows.
    fields = ("suite", "suite_name", "task_idx", "task_index", "task_id", "task", "state_id", "state_idx", "state", "seed", "initial_state_sha256", "source_sha256")
    encoded = json.dumps({name: value[name] for name in fields if name in value}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicates)


def extract_keys(path: Path) -> dict[str, Any]:
    raw = 0
    malformed = 0
    fingerprints: dict[tuple[str, int, int], set[str]] = {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw += 1
                try:
                    key = identity_key(row)
                    if key is None:
                        raise ValueError("identity fields missing")
                    if key not in fingerprints:
                        fingerprints[key] = set()
                    fingerprints[key].add(_fingerprint(row))
                except (TypeError, ValueError):
                    malformed += 1
    else:
        try:
            payload = _strict_json(path)
            entries = list(walk_identity_entries(payload))
            raw = len(entries)
            for key, fingerprint in entries:
                if key not in fingerprints:
                    fingerprints[key] = set()
                fingerprints[key].add(fingerprint)
        except (OSError, ValueError, json.JSONDecodeError):
            malformed += 1
    duplicate = sum(1 for values in fingerprints.values() if len(values) > 1)
    duplicate_rows = raw - len(fingerprints) - malformed
    return {
        "raw_records": raw,
        "unique_records": len(fingerprints),
        "duplicate_records": max(0, duplicate_rows),
        "conflict_identities": duplicate,
        "malformed_records": malformed,
        "keys": set(fingerprints),
    }


def seal(staging: Path, final: Path, manifest: dict[str, Any]) -> dict[str, str]:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
    sums = staging / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256_file(staging / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    sums_sha = sha256_file(sums)
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.rename(staging, final)
    return {"root": str(final), "sha256sums_sha256": sums_sha}


def run(args: argparse.Namespace) -> dict[str, Any]:
    clean = Path(args.clean_manifest).resolve()
    protected = [Path(path).resolve() for path in args.protected_manifest]
    if not clean.is_file() or any(not path.is_file() for path in protected):
        raise FileNotFoundError("clean or protected manifest missing")
    clean_audit = extract_keys(clean)
    protected_audits = [extract_keys(path) for path in protected]
    protected_union: set[tuple[str, int, int]] = set()
    for audit in protected_audits:
        protected_union.update(audit["keys"])
    overlap = clean_audit["keys"] & protected_union
    source_root = Path(args.source_root).resolve() if args.source_root else None
    source_entries = sorted(path.name for path in source_root.iterdir()) if source_root and source_root.is_dir() else []
    source_top_seal = bool(source_root and all((source_root / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256")))
    split_counts: dict[str, int] = {"FIT_TRAIN": 0, "FIT_DEV": 0, "CAL": 0, "CHECK": 0, "FINAL_EVAL_CANDIDATE": 0}
    for suite, _task, state in clean_audit["keys"]:
        split = "FIT_TRAIN" if state < 20 else "FIT_DEV" if state < 24 else "CAL" if state < 27 else "CHECK" if state < 30 else "FINAL_EVAL_CANDIDATE"
        split_counts[split] += 1
    closure = clean_audit["unique_records"] == 2000 and clean_audit["duplicate_records"] == 0 and clean_audit["conflict_identities"] == 0 and clean_audit["malformed_records"] == 0
    protected_complete = all(a["unique_records"] > 0 and a["duplicate_records"] == 0 and a["conflict_identities"] == 0 and a["malformed_records"] == 0 for a in protected_audits)
    overlap_zero = not overlap
    capability = "PARTIAL"  # schema-only metadata cannot certify replay or V23 relabelability.
    status = "PASS_METADATA_OVERLAP_ZERO" if closure and protected_complete and overlap_zero else "HOLD_PROVENANCE"
    receipt: dict[str, Any] = {
        "schema": "C3_D0_CLEAN2000_ALLOCATION_CUSTODIAN_V1",
        "status": status,
        "decision": status,
        "clean_manifest": {"path": str(clean), "sha256": sha256_file(clean), **{k: v for k, v in clean_audit.items() if k != "keys"}},
        "protected_manifests": [{"path": str(path), "sha256": sha256_file(path), "root_seal_files_present": sorted(name for name in ("P0_EVIDENCE_SEAL.json", "G6_SEAL_V2.json", "G6_SEAL.json") if (path.parent / name).is_file()), **{k: v for k, v in audit.items() if k != "keys"}} for path, audit in zip(protected, protected_audits)],
        "aggregate": {"protected_union_unique": len(protected_union), "clean_protected_overlap_count": len(overlap), "protected_cross_manifest_overlap_count": sum(len(protected_audits[i]["keys"] & protected_audits[j]["keys"]) for i in range(len(protected_audits)) for j in range(i + 1, len(protected_audits)))},
        "protected_identity_values_emitted": 0,
        "protected_content_emitted": False,
        "identity_closure_2000": closure,
        "split_counts": split_counts,
        "fit_only_status": "DERIVED_FROM_OFFICIAL_V3_SPLIT_METADATA" if split_counts["FIT_TRAIN"] == 800 else "HOLD",
        "capability_decision": capability,
        "source_root": {"path": str(source_root) if source_root else None, "top_level_seal_present": source_top_seal, "top_level_names": source_entries, "payloads_read": False},
        "derived_snapshot_created": False,
        "model_inference": False,
        "training": False,
        "rollout": False,
        "attack": False,
        "protected_reads_scope": "aggregate_identity_set_statistics_only",
    }
    parent = Path(args.out_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (args.output_name or f"d0_clean2000_{uuid.uuid4().hex[:8]}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        if overlap_zero:
            (staging / "V23_SPLIT_PLAN.json").write_text(json.dumps({"schema": "C3_D0_TASK_STRATIFIED_V23_SPLIT_V1", "status": "FROZEN_METADATA_ONLY", "split_counts": split_counts, "identity_manifest_sha256": receipt["clean_manifest"]["sha256"], "protected_overlap_count": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "D0_FEASIBILITY_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return receipt | seal(staging, final, {"schema": receipt["schema"], "status": status, "protected_identity_values_emitted": 0, "payloads_read": False})
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--protected-manifest", action="append", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
