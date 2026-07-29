"""Read-only reconciliation of a live FIT670 publication snapshot.

This audits names, manifests, and seals only. It never parses episode telemetry
and it refuses protected-looking roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.seal_utils import rename_noreplace


FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_allowed(path: Path) -> None:
    lowered = {part.lower() for part in path.resolve().parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"forbidden/protected-looking path: {path}")


def verify_sealed_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"missing seal: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"symlink in sealed root: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        relative = Path(name)
        if separator != "  " or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"malformed seal row: {line!r}")
        if name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"duplicate/excluded seal row: {name}")
        if relative.is_absolute() or ".." in relative.parts or (root / relative).is_symlink() or not (root / relative).is_file():
            raise ValueError(f"unsafe/missing sealed file: {name}")
        if sha256_file(root / relative) != digest:
            raise ValueError(f"file seal mismatch: {name}")
        listed.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if actual != listed:
        raise ValueError(f"file closure mismatch: missing={sorted(actual - listed)} extra={sorted(listed - actual)}")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": len(listed), "payload_bytes_hashed": True}


def _published_episode_roots(formal_root: Path) -> list[tuple[str, Path]]:
    episodes_root = formal_root / "episodes"
    if not episodes_root.is_dir():
        return []
    found: dict[str, Path] = {}
    for marker in episodes_root.rglob("episode.json"):
        if any(".staging." in part for part in marker.parts):
            continue
        episode_root = marker.parent
        relative = episode_root.relative_to(episodes_root)
        identity = "/".join(relative.parts)
        if not identity or identity in found:
            raise ValueError(f"duplicate published episode identity: {identity}")
        found[identity] = episode_root
    return sorted(found.items())


def _worker_manifests(formal_root: Path) -> list[tuple[str, Path, Path, dict[str, Any]]]:
    rows: list[tuple[str, Path, Path, dict[str, Any]]] = []
    for worker_root in sorted(formal_root.glob("gpu_*")):
        if not worker_root.is_dir() or worker_root.is_symlink():
            continue
        worker_id = worker_root.name
        candidates = [path for path in worker_root.iterdir() if path.is_file() and "manifest" in path.name.lower() and ".staging." not in path.name]
        if len(candidates) != 1:
            if candidates:
                rows.append((worker_id, worker_root, candidates[0], {"_manifest_conflict": [str(item) for item in candidates]}))
            continue
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and any(key in data for key in ("results", "episodes", "completed", "n_success")):
                rows.append((worker_id, worker_root, path, data))
                break
    return rows


def _manifest_identity_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "episodes", "completed"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _identity(row: dict[str, Any]) -> str | None:
    value = row.get("episode_id", row.get("identity"))
    return str(value) if value not in (None, "") else None


def _write_seal(root: Path) -> dict[str, Any]:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded)
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8"
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": digest, "file_count": len(files)}


def audit(formal_root: Path, allowlist_path: Path | None = None, output_root: Path | None = None, *, reported_total: int | None = None, reported_shard_sum: int | None = None, expected_worker_ids: set[str] | None = None) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    _assert_allowed(formal_root)
    if allowlist_path is not None:
        allowlist_path = allowlist_path.resolve()
        _assert_allowed(allowlist_path)
    if output_root is not None:
        output_root = output_root.resolve()
        _assert_allowed(output_root)
        if output_root.exists():
            raise FileExistsError(output_root)
    staging = [path for path in formal_root.rglob("*") if ".staging." in path.name]
    episodes_root = formal_root / "episodes"
    candidate_dirs = {
        path.relative_to(episodes_root).as_posix(): path
        for path in episodes_root.rglob("*")
        if path.is_dir() and not path.is_symlink() and len(path.relative_to(episodes_root).parts) == 3 and ".staging." not in path.name
    } if episodes_root.is_dir() else {}
    published = _published_episode_roots(formal_root)
    published_map = dict(published)
    missing_episode_markers = sorted(set(candidate_dirs) - set(published_map))
    valid_ids: set[str] = set()
    bad_seal: list[dict[str, str]] = []
    for identity, episode_root in published:
        try:
            verify_sealed_root(episode_root)
            valid_ids.add(identity)
        except (OSError, ValueError) as exc:
            bad_seal.append({"identity": identity, "error": str(exc)})

    allowlisted: set[str] | None = None
    allowlist_sha = None
    allowlist_error = None
    if allowlist_path is not None and allowlist_path.is_file():
        try:
            allowlist_sha = sha256_file(allowlist_path)
            data = json.loads(allowlist_path.read_text(encoding="utf-8"))
            identities = data.get("identities") if isinstance(data, dict) else None
            entries = [item for item in identities or [] if isinstance(item, dict) and item.get("episode_id")]
            required = ("episode_id", "suite", "task_id", "state_id", "collection_seed", "initial_state_sha256")
            if any(any(key not in item for key in required) for item in entries):
                raise ValueError("allowlist entries incomplete")
            canonical_entries = [{key: item[key] for key in required} for item in entries]
            computed = hashlib.sha256(json.dumps(canonical_entries, sort_keys=True).encode()).hexdigest()
            if data.get("schema") != "FIT670_IDENTITY_ALLOWLIST_V1" or data.get("protected_overlap") != 0 or len(entries) != len(identities or []) or data.get("identity_set_digest") != computed:
                raise ValueError("allowlist schema/digest/protected overlap invalid")
            ids = [str(item["episode_id"]) for item in entries]
            if len(set(ids)) != len(ids):
                raise ValueError("allowlist duplicate identities")
            allowlisted = set(ids)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            allowlist_error = str(exc)

    if expected_worker_ids is None:
        expected_worker_ids = {f"gpu_{index}" for index in range(8)}
    workers = _worker_manifests(formal_root)
    worker_names = {worker_id for worker_id, _, _, _ in workers}
    missing_workers = sorted((expected_worker_ids or set()) - worker_names)
    extra_workers = sorted(worker_names - expected_worker_ids)
    worker_ids: dict[str, list[str]] = {}
    worker_rows: list[dict[str, Any]] = []
    worker_manifest_conflicts = []
    worker_manifest_mismatches = []
    worker_bad_seals = []
    for worker_id, worker_root, path, data in workers:
        if "_manifest_conflict" in data:
            worker_manifest_conflicts.append({"worker": worker_id, "paths": data["_manifest_conflict"]})
            continue
        try:
            worker_seal = verify_sealed_root(worker_root)
        except (OSError, ValueError) as exc:
            worker_bad_seals.append({"worker": worker_id, "error": str(exc)})
            continue
        raw_rows = _manifest_identity_rows(data)
        if not isinstance(data.get("results", raw_rows), list):
            worker_manifest_mismatches.append({"worker": worker_id, "error": "results is not a list"})
        ids = [identity for row in raw_rows if (identity := _identity(row))]
        worker_ids[worker_id] = ids
        declared = data.get("n_success")
        if not isinstance(declared, int) or isinstance(declared, bool) or declared != len(ids):
            worker_manifest_mismatches.append({"worker": worker_id, "declared_n_success": declared, "parsed_results": len(ids)})
        worker_rows.append({"worker": worker_id, "manifest": str(path), "manifest_sha256": sha256_file(path), "worker_sha256sums_sha256": worker_seal["sha256sums_sha256"], "count": len(ids), "declared_n_success": declared, "ids": ids})
    all_worker_ids = [identity for ids in worker_ids.values() for identity in ids]
    worker_unique = set(all_worker_ids)
    duplicate_worker_ids = sorted({identity for identity in all_worker_ids if all_worker_ids.count(identity) > 1})
    extra = sorted(valid_ids - (allowlisted or valid_ids)) if allowlisted is not None else []
    unallowlisted = sorted(valid_ids - allowlisted) if allowlisted is not None else []
    missing_allowlisted = sorted(allowlisted - valid_ids) if allowlisted is not None else []
    shard_unique_sum = sum(len(set(ids)) for ids in worker_ids.values())
    reported_mismatches = []
    if reported_total is not None and reported_total != len(published):
        reported_mismatches.append({"field": "reported_total", "reported": reported_total, "observed": len(published)})
    if reported_shard_sum is not None and reported_shard_sum != shard_unique_sum:
        reported_mismatches.append({"field": "reported_shard_sum", "reported": reported_shard_sum, "observed": shard_unique_sum})
    closure_pass = bool(
        valid_ids
        and not bad_seal
        and not duplicate_worker_ids
        and not extra
        and not unallowlisted
        and not missing_allowlisted
        and len(valid_ids) == shard_unique_sum == len(worker_unique)
        and not staging
        and not missing_episode_markers
        and not worker_manifest_conflicts
        and not worker_manifest_mismatches
        and not worker_bad_seals
        and allowlisted is not None
        and not missing_workers
        and not extra_workers
        and not reported_mismatches
    )
    report = {
        "schema": "V5_R3_PROGRESS_RECONCILIATION_V1",
        "status": "PASS" if closure_pass else "HOLD_INCOMPLETE_PUBLICATION_OR_WORKER_CLOSURE",
        "metadata_only": True,
        "payload_semantics_read": False,
        "formal_root": str(formal_root),
        "reported_total": reported_total,
        "reported_shard_sum": reported_shard_sum,
        "published_episode_directories": len(published),
        "published_directory_candidates": len(candidate_dirs),
        "missing_episode_marker_count": len(missing_episode_markers),
        "missing_episode_markers": missing_episode_markers,
        "valid_sealed_episodes": len(valid_ids),
        "worker_manifest_count": len(workers),
        "per_shard_unique_sum": shard_unique_sum,
        "unique_worker_id_count": len(worker_unique),
        "duplicate_worker_id_count": len(duplicate_worker_ids),
        "duplicate_worker_ids": duplicate_worker_ids,
        "extra_count": len(extra),
        "extra_ids": extra,
        "unallowlisted_count": len(unallowlisted),
        "unallowlisted_ids": unallowlisted,
        "missing_allowlisted_count": len(missing_allowlisted),
        "missing_allowlisted_ids": missing_allowlisted,
        "bad_seal_count": len(bad_seal),
        "bad_seals": bad_seal,
        "staging_residue_count": len(staging),
        "staging_residues": [str(path) for path in staging],
        "worker_manifest_conflicts": worker_manifest_conflicts,
        "worker_manifest_mismatches": worker_manifest_mismatches,
        "worker_bad_seals": worker_bad_seals,
        "missing_worker_count": len(missing_workers),
        "missing_workers": missing_workers,
        "extra_worker_count": len(extra_workers),
        "extra_workers": extra_workers,
        "reported_count_mismatches": reported_mismatches,
        "allowlist_sha256": allowlist_sha,
        "allowlist_error": allowlist_error,
        "allowlist_available": allowlisted is not None,
        "protected_read_audit": {"status": "PASS", "forbidden_paths_scanned": sorted(FORBIDDEN_PARTS & {part.lower() for part in formal_root.parts})},
        "protected_reads": [],
        "payload_bytes_hashed": True,
        "attack_enabled": False,
        "worker_manifests": worker_rows,
        "closure_equality": {
            "unique_published_equals_shard_unique_sum": len(valid_ids) == shard_unique_sum,
            "unique_published_equals_valid_sealed": len(valid_ids) == len(published) and not bad_seal,
            "unique_published_equals_worker_unique": len(valid_ids) == len(worker_unique),
        },
    }
    if output_root is not None:
        staging_root = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
        if staging_root.exists():
            raise FileExistsError(staging_root)
        staging_root.mkdir(parents=True)
        (staging_root / "R3_1A_RECONCILIATION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging_root / "WORKER_CENSUS.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["worker", "manifest", "manifest_sha256", "count", "declared_n_success"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in worker_rows)
        with (staging_root / "PUBLISHED_CENSUS.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["identity", "valid_seal"])
            writer.writeheader()
            writer.writerows({"identity": identity, "valid_seal": identity in valid_ids} for identity, _ in published)
        _write_seal(staging_root)
        rename_noreplace(staging_root, output_root)
        report["output_root"] = str(output_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reported-total", type=int)
    parser.add_argument("--reported-shard-sum", type=int)
    parser.add_argument("--expected-worker-ids", help="comma-separated exact worker IDs; defaults to gpu_0..gpu_7")
    args = parser.parse_args()
    expected_workers = None if args.expected_worker_ids is None else {item.strip() for item in args.expected_worker_ids.split(",") if item.strip()}
    print(json.dumps(audit(args.formal_root, args.allowlist, args.output_root, reported_total=args.reported_total, reported_shard_sum=args.reported_shard_sum, expected_worker_ids=expected_workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
