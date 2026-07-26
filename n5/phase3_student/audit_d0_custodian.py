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


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_counts(keys: Iterable[tuple[str, int, int]]) -> tuple[dict[str, int], list[dict[str, int | str]]]:
    per_suite: dict[str, int] = {}
    per_task: dict[tuple[str, int], int] = {}
    for suite, task, _state in keys:
        per_suite[suite] = per_suite.get(suite, 0) + 1
        per_task[(suite, task)] = per_task.get((suite, task), 0) + 1
    rows = [{"suite": suite, "task_id": task, "unique_count": count} for (suite, task), count in sorted(per_task.items())]
    return dict(sorted(per_suite.items())), rows


def schema_only_capability(source_root: Path | None, keys: set[tuple[str, int, int]]) -> dict[str, Any]:
    expected = ("episode_metadata.json", "step_records.jsonl", "runtime_audit.json", "privileged_teacher_sidecar.jsonl", "artifact_sha256.json", "episode_summary.json")
    present = {name: 0 for name in expected}
    root_count = 0
    replay_ready = 0
    if source_root and source_root.is_dir():
        for suite, task, state in keys:
            episode_root = source_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
            if episode_root.is_dir():
                root_count += 1
            for name in expected:
                if (episode_root / name).is_file():
                    present[name] += 1
            if (episode_root / "episode_metadata.json").is_file() and (episode_root / "step_records.jsonl").is_file():
                replay_ready += 1
    if root_count == len(keys) and replay_ready == len(keys):
        decision = "REPLAY"
    elif root_count == 0:
        decision = "INCOMPATIBLE"
    else:
        decision = "PARTIAL"
    return {"decision": decision, "schema_only": True, "dev_identity_count": len(keys), "episode_root_count": root_count, "replay_ready_count": replay_ready, "expected_file_counts": present}


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
    fit_keys = {key for key in clean_audit["keys"] if key[2] < 20}
    dev_keys = clean_audit["keys"] - protected_union
    source_root = Path(args.source_root).resolve() if args.source_root else None
    source_entries = sorted(path.name for path in source_root.iterdir()) if source_root and source_root.is_dir() else []
    source_top_seal = bool(source_root and all((source_root / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256")))
    split_counts: dict[str, int] = {"FIT_TRAIN": 0, "FIT_DEV": 0, "CAL": 0, "CHECK": 0, "FINAL_EVAL_CANDIDATE": 0}
    for suite, _task, state in clean_audit["keys"]:
        split = "FIT_TRAIN" if state < 20 else "FIT_DEV" if state < 24 else "CAL" if state < 27 else "CHECK" if state < 30 else "FINAL_EVAL_CANDIDATE"
        split_counts[split] += 1
    closure = clean_audit["unique_records"] == 2000 and clean_audit["duplicate_records"] == 0 and clean_audit["conflict_identities"] == 0 and clean_audit["malformed_records"] == 0
    # Exact duplicate rows are reported but do not invalidate a set-valued
    # manifest; conflicting or malformed identities do.
    protected_complete = all(a["unique_records"] > 0 and a["conflict_identities"] == 0 and a["malformed_records"] == 0 for a in protected_audits)
    overlap_zero = not overlap
    dev_suite_counts, dev_task_counts = aggregate_counts(dev_keys)
    capability_audit = schema_only_capability(source_root, dev_keys)
    dev_closure = len(dev_keys) == 670 and not (dev_keys & protected_union) and (dev_keys | protected_union) == clean_audit["keys"]
    capability = capability_audit["decision"]
    status = "PASS" if closure and protected_complete and dev_closure and capability in {"DIRECT_RELABEL", "REPLAY"} else "HOLD_PROVENANCE"
    receipt: dict[str, Any] = {
        "schema": "C3_D0_CLEAN2000_ALLOCATION_CUSTODIAN_V1",
        "status": status,
        "decision": status,
        "clean_manifest": {"path": str(clean), "sha256": sha256_file(clean), **{k: v for k, v in clean_audit.items() if k != "keys"}},
        "protected_manifests": [{"path": str(path), "sha256": sha256_file(path), "root_seal_files_present": sorted(name for name in ("P0_EVIDENCE_SEAL.json", "G6_SEAL_V2.json", "G6_SEAL.json") if (path.parent / name).is_file()), **{k: v for k, v in audit.items() if k != "keys"}} for path, audit in zip(protected, protected_audits)],
        "aggregate": {"clean_unique": len(clean_audit["keys"]), "protected_union_unique": len(protected_union), "dev_pool_unique": len(dev_keys), "clean_protected_overlap_count": len(overlap), "fit_train_protected_overlap_count": len(fit_keys & protected_union), "non_fit_clean_protected_overlap_count": len((clean_audit["keys"] - fit_keys) & protected_union), "protected_cross_manifest_overlap_count": sum(len(protected_audits[i]["keys"] & protected_audits[j]["keys"]) for i in range(len(protected_audits)) for j in range(i + 1, len(protected_audits)))},
        "protected_identity_values_emitted": 0,
        "protected_content_emitted": False,
        "identity_closure_2000": closure,
        "dev_pool_closure_670": dev_closure,
        "dev_pool_per_suite_counts": dev_suite_counts,
        "dev_pool_per_task_counts": dev_task_counts,
        "split_counts": split_counts,
        "fit_only_status": "DERIVED_FROM_OFFICIAL_V3_SPLIT_METADATA" if split_counts["FIT_TRAIN"] == 800 else "HOLD",
        "capability_decision": capability,
        "capability_audit": capability_audit,
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
        if dev_closure:
            receipt["derived_snapshot_created"] = True
            dev_manifest_path = staging / "DEV_POOL_IDENTITY_MANIFEST.csv"
            write_csv(dev_manifest_path, ["suite", "task_id", "state_id"], ({"suite": suite, "task_id": task, "state_id": state} for suite, task, state in sorted(dev_keys)))
            dev_manifest_sha = sha256_file(dev_manifest_path)
            receipt["dev_pool_identity_manifest_sha256"] = dev_manifest_sha
            write_csv(staging / "DEV_POOL_PER_SUITE_TASK_COUNTS.csv", ["suite", "task_id", "unique_count"], dev_task_counts)
            (staging / "CAPABILITY_AUDIT.json").write_text(json.dumps(capability_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / "V23_DEV_POOL_SPLIT_PLAN.json").write_text(json.dumps({"schema": "C3_D0_R2_V23_DEV_POOL_SPLIT_V1", "status": "FROZEN_METADATA_ONLY", "dev_pool_unique": len(dev_keys), "per_suite_counts": dev_suite_counts, "per_task_counts": dev_task_counts, "protected_identity_values_emitted": 0, "identity_manifest_sha256": dev_manifest_sha}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
