"""Build the byte-sealed, one-episode-per-task V23 development pilot input.

Selection happens from the derived DEV_POOL identity manifest before any
episode payload is opened.  Protected manifests and payloads are never read.
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


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS = tuple(range(10))
REQUIRED_FILES = (
    "episode_metadata.json",
    "step_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
)
IDENTITY_FIELDS = ("suite", "task_id", "state_id")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicates)


def _int_value(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _nested_value(data: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    for key in ("identity", "episode", "episode_identity", "initial_state"):
        child = data.get(key)
        if isinstance(child, Mapping):
            found = _nested_value(child, names)
            if found is not None:
                return found
    return None


def _metadata_identity(metadata: Mapping[str, Any]) -> tuple[str | None, int | None, int | None]:
    suite = _nested_value(metadata, ("suite", "suite_name"))
    task = _int_value(_nested_value(metadata, ("task_id", "task_idx", "task_index")))
    state = _int_value(_nested_value(metadata, ("state_id", "state_idx", "state_index")))
    return (suite if isinstance(suite, str) else None, task, state)


def _schema_version(metadata: Mapping[str, Any]) -> str | None:
    value = _nested_value(metadata, ("schema_version", "schema"))
    return value if isinstance(value, str) and value else None


def _seed(metadata: Mapping[str, Any]) -> Any:
    return _nested_value(metadata, ("seed", "rng_seed", "initial_state_seed"))


def canonical_identity(suite: str, task_id: int, state_id: int) -> str:
    return f"{suite}/task_{task_id:02d}/state_{state_id:02d}"


def _safe_episode_root(source_root: Path, suite: str, task_id: int, state_id: int) -> Path:
    if suite not in SUITES or task_id not in TASKS or state_id < 0:
        raise ValueError("identity outside official Clean2000 coordinate space")
    root = source_root.resolve(strict=True)
    candidate = source_root / suite / f"task_{task_id:02d}" / f"state_{state_id:02d}"
    absolute = candidate.absolute()
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError("episode path escapes source root") from exc
    current = root
    for part in absolute.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink path component rejected: {current}")
    if not absolute.is_dir() or absolute.resolve(strict=True) != absolute:
        raise ValueError(f"episode root is not a real directory: {absolute}")
    return absolute


def _load_dev_pool(path: Path, expected_sha256: str | None = None) -> list[tuple[str, int, int]]:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("DEV_POOL identity manifest SHA mismatch")
    rows: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(IDENTITY_FIELDS):
            raise ValueError("DEV_POOL manifest header mismatch")
        for row in reader:
            suite = row.get("suite")
            task = _int_value(row.get("task_id"))
            state = _int_value(row.get("state_id"))
            if not isinstance(suite, str) or task is None or state is None:
                raise ValueError("malformed DEV_POOL identity")
            key = (suite, task, state)
            if key in seen:
                raise ValueError("duplicate DEV_POOL identity")
            seen.add(key)
            rows.append(key)
    if len(rows) != 670:
        raise ValueError(f"DEV_POOL must contain 670 identities, got {len(rows)}")
    return rows


def _select_one_per_task(keys: Iterable[tuple[str, int, int]], seed: str) -> list[tuple[str, int, int]]:
    by_task: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
    for key in keys:
        by_task.setdefault((key[0], key[1]), []).append(key)
    expected = {(suite, task) for suite in SUITES for task in TASKS}
    if set(by_task) != expected:
        missing = sorted(expected - set(by_task))
        extra = sorted(set(by_task) - expected)
        raise ValueError(f"DEV_POOL task closure mismatch: missing={missing}, extra={extra}")
    selected: list[tuple[str, int, int]] = []
    for suite in SUITES:
        for task in TASKS:
            candidates = by_task[(suite, task)]
            ranked = sorted(
                candidates,
                key=lambda key: hashlib.sha256(
                    f"{seed}:{canonical_identity(*key)}".encode("utf-8")
                ).hexdigest(),
            )
            selected.append(ranked[0])
    if len(selected) != 40 or len(set(selected)) != 40:
        raise ValueError("pilot selection is not 40 unique identities")
    return selected


def _load_steps(path: Path, identity: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"step row is not an object at line {line_number}")
            step = row.get("step")
            if type(step) is not int or step < 0 or step in seen:
                raise ValueError(f"duplicate/invalid step at line {line_number}: {step!r}")
            row_identity = row.get("episode_id") or row.get("identity")
            if row_identity is not None and row_identity != identity:
                raise ValueError(f"step identity mismatch at line {line_number}")
            seen.add(step)
            rows.append(row)
    if not rows or seen != set(range(len(rows))):
        raise ValueError(f"step sequence is not exactly 0..T-1 for {identity}")
    return rows, {"count": len(rows), "first": 0, "last": len(rows) - 1}


def _verify_optional_identity_fields(rows: Iterable[Mapping[str, Any]], key: tuple[str, int, int]) -> None:
    suite, task_id, state_id = key
    for row in rows:
        if "suite" in row and row["suite"] != suite:
            raise ValueError("step stream suite mismatch")
        if "task_idx" in row and row["task_idx"] != task_id:
            raise ValueError("step stream task mismatch")
        if "state_id" in row and row["state_id"] != state_id:
            raise ValueError("step stream state mismatch")


def _audit_episode(source_root: Path, key: tuple[str, int, int]) -> dict[str, Any]:
    suite, task_id, state_id = key
    identity = canonical_identity(*key)
    episode_root = _safe_episode_root(source_root, *key)
    source_files: list[dict[str, Any]] = []
    for filename in REQUIRED_FILES:
        path = episode_root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing/non-regular required file: {path}")
        source_files.append({
            "name": filename,
            "path": str(path),
            "relative_to_source": path.relative_to(source_root.resolve()).as_posix(),
            "is_regular_file": True,
            "is_symlink": False,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    metadata_path = episode_root / "episode_metadata.json"
    metadata = strict_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata is not an object: {identity}")
    metadata_identity = _metadata_identity(metadata)
    if any(value is not None for value in metadata_identity) and metadata_identity != key:
        raise ValueError(f"metadata identity mismatch: {identity} vs {metadata_identity}")
    steps, step_summary = _load_steps(episode_root / "step_records.jsonl", identity)
    _verify_optional_identity_fields(steps, key)
    sidecar_steps, sidecar_summary = _load_steps(episode_root / "privileged_teacher_sidecar.jsonl", identity)
    _verify_optional_identity_fields(sidecar_steps, key)
    if sidecar_summary["count"] != step_summary["count"]:
        raise ValueError(f"step/sidecar count mismatch for {identity}")
    return {
        "episode_id": identity,
        "suite": suite,
        "task_id": task_id,
        "state_id": state_id,
        "seed": _seed(metadata),
        "source_episode_root": str(episode_root),
        "source_root_real": str(source_root.resolve()),
        "path_safety": {"root_symlink": False, "file_symlinks": False, "path_escape": False},
        "source_files": source_files,
        "schema_version": _schema_version(metadata) or "UNDECLARED",
        "observed_step_count": step_summary["count"],
        "sidecar_step_count": sidecar_summary["count"],
        "first_step_index": step_summary["first"],
        "last_step_index": step_summary["last"],
        "identity_join_status": "PASS_METADATA_AND_STEP_STREAM",
        "step_indices_contiguous": True,
        "payload_parse_status": "PASS",
    }


def _seal(staging: Path, final: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    (staging / "PILOT_INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "MANIFEST.json").write_text(
        json.dumps({"schema": manifest["schema"], "status": manifest["status"], "episode_count": manifest["episode_count"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    )
    sums = staging / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256_file(staging / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    sums_sha = sha256_file(sums)
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.rename(staging, final)
    return {"root": str(final), "sha256sums_sha256": sums_sha}


def build(dev_pool_manifest: Path, d0_receipt_path: Path, source_root: Path,
          out_parent: Path, output_name: str, selection_seed: str) -> dict[str, Any]:
    d0_receipt = strict_json(d0_receipt_path)
    if d0_receipt.get("status") != "PASS" or d0_receipt.get("dev_pool_closure_670") is not True:
        raise ValueError("D0-R2 receipt is not a passing 670-identity closure")
    if d0_receipt.get("protected_identity_values_emitted") != 0 or d0_receipt.get("protected_content_emitted") is not False:
        raise ValueError("D0-R2 receipt violates protected emission boundary")
    expected_dev_sha = d0_receipt.get("dev_pool_identity_manifest_sha256")
    keys = _load_dev_pool(dev_pool_manifest, expected_dev_sha)
    # Freeze selection from identity metadata before opening selected episode files.
    selected = _select_one_per_task(keys, selection_seed)
    records = [_audit_episode(source_root, key) for key in selected]
    if len(records) != 40 or len({record["episode_id"] for record in records}) != 40:
        raise ValueError("pilot payload audit did not produce 40 unique episodes")
    if any(record["payload_parse_status"] != "PASS" for record in records):
        raise ValueError("pilot payload parse failure")
    manifest: dict[str, Any] = {
        "schema": "V23_DEV_PILOT_V1",
        "status": "FROZEN_INPUT_BYTES_ONLY",
        "selection_algorithm": "sha256(selection_seed + ':' + canonical_episode_identity), lexicographic minimum per suite/task",
        "selection_seed": selection_seed,
        "selection_before_payload_read": True,
        "episode_count": 40,
        "task_count": 40,
        "suite_count": 4,
        "source_root": str(source_root.resolve()),
        "dev_pool_manifest": {"path": str(dev_pool_manifest.resolve()), "sha256": sha256_file(dev_pool_manifest)},
        "d0_receipt": {"path": str(d0_receipt_path.resolve()), "sha256": sha256_file(d0_receipt_path), "dev_pool_unique": 670, "protected_overlap_count": 0},
        "protected_membership_aggregate": {"source": "sealed D0-R2 receipt", "overlap_count": 0, "protected_identity_values_emitted": 0},
        "protected_payload_read": False,
        "records": records,
        "no_model": True,
        "no_rollout": True,
        "no_attack": True,
    }
    out_parent.mkdir(parents=True, exist_ok=True)
    final = out_parent / output_name
    staging = out_parent / f".staging_{output_name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        return manifest | _seal(staging, final, manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-pool-manifest", required=True, type=Path)
    parser.add_argument("--d0-receipt", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-parent", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--selection-seed", default="C3-T1-D0-R2P-V1")
    args = parser.parse_args()
    result = build(
        args.dev_pool_manifest,
        args.d0_receipt,
        args.source_root,
        args.out_parent,
        args.output_name,
        args.selection_seed,
    )
    print(json.dumps({key: result[key] for key in ("schema", "status", "episode_count", "task_count", "suite_count", "protected_payload_read", "root", "sha256sums_sha256")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
