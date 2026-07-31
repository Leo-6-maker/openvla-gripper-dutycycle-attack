"""Fail-closed source and artifact provenance helpers for Detector-v3 D8.

SOURCE_SNAPSHOT_V2 is deliberately external to the Git tree.  It is generated
from the exact committed checkout that will be archived/deployed and is passed
explicitly to cache/P5 commands.  This avoids the self-referential failure mode
of committing a snapshot that hashes files changed by a later metadata commit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")

CACHE_REQUIRED_SOURCE_FILES = (
    "scripts/detector_v5/build_d8_25d_cache.py",
    "scripts/detector_v5/d8_source_contract.py",
    "scripts/detector_v5/load_fit670_25d_telemetry.py",
    "scripts/detector_v5/d8_event_consolidator.py",
    "scripts/detector_v5/run_d8_formal_g_sensitivity.py",
    "scripts/detector_v5/audit_r3_contact_input.py",
    "src/gripper_attack/d8_streaming_features_v3.py",
    "src/gripper_attack/action_contract.py",
    "src/gripper_attack/seal_utils.py",
    "configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json",
    "configs/FIT670_25D_SOURCE_MAPPING.json",
)

P5_REQUIRED_SOURCE_FILES = (
    "scripts/detector_v5/run_d8_p5_25d_gpu_smoke.py",
    "scripts/detector_v5/d8_train_core.py",
    "scripts/detector_v5/d8_source_contract.py",
    "scripts/detector_v5/audit_r3_contact_input.py",
    "src/gripper_attack/seal_utils.py",
    "configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json",
)

REVIEW_REQUIRED_SOURCE_FILES = tuple(sorted(set(
    CACHE_REQUIRED_SOURCE_FILES
    + P5_REQUIRED_SOURCE_FILES
    + (
        "scripts/detector_v5/compare_d8_25d_caches.py",
        "scripts/detector_v5/audit_d8_h1_r9.py",
        "scripts/detector_v5/make_d8_source_snapshot.py",
    )
)))


class SourceContractError(RuntimeError):
    """Raised when source or artifact provenance is not formally closed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relpath(value: str) -> str:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or value in {"", "."}:
        raise SourceContractError(f"unsafe relative path in manifest: {value!r}")
    return p.as_posix()


def load_and_validate_source_snapshot(
    snapshot_path: Path,
    repo_root: Path,
    required_paths: Iterable[str],
) -> dict:
    """Validate an external SOURCE_SNAPSHOT_V2 against local deployed bytes."""
    snapshot_path = snapshot_path.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    try:
        snap = json.loads(snapshot_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceContractError(f"invalid source snapshot: {snapshot_path}: {exc}") from exc

    if snap.get("schema") != "SOURCE_SNAPSHOT_V2":
        raise SourceContractError("source snapshot schema must be SOURCE_SNAPSHOT_V2")

    commit = snap.get("executable_source_commit", "")
    tree = snap.get("executable_source_tree", "")
    if not isinstance(commit, str) or not GIT_OID_RE.fullmatch(commit):
        raise SourceContractError(f"invalid executable_source_commit: {commit!r}")
    if not isinstance(tree, str) or not GIT_OID_RE.fullmatch(tree):
        raise SourceContractError(f"invalid executable_source_tree: {tree!r}")

    file_map = snap.get("file_sha256_map")
    if not isinstance(file_map, dict) or not file_map:
        raise SourceContractError("source snapshot file_sha256_map must be a non-empty object")

    normalized_map: dict[str, str] = {}
    for raw_rel, expected in file_map.items():
        if not isinstance(raw_rel, str) or not isinstance(expected, str):
            raise SourceContractError("source snapshot path/digest must be strings")
        rel = _safe_relpath(raw_rel)
        if rel in normalized_map:
            raise SourceContractError(f"duplicate source path after normalization: {rel}")
        if not SHA256_RE.fullmatch(expected):
            raise SourceContractError(f"invalid SHA256 for {rel}: {expected!r}")
        normalized_map[rel] = expected

    missing_contract_paths = sorted(set(required_paths) - set(normalized_map))
    if missing_contract_paths:
        raise SourceContractError(
            "source snapshot missing required files: " + ", ".join(missing_contract_paths)
        )

    mismatches = []
    for rel, expected in sorted(normalized_map.items()):
        local = repo_root / rel
        if local.is_symlink():
            mismatches.append(f"symlink:{rel}")
            continue
        if not local.is_file():
            mismatches.append(f"missing:{rel}")
            continue
        actual = sha256_file(local)
        if actual != expected:
            mismatches.append(f"sha:{rel}:expected={expected}:actual={actual}")
    if mismatches:
        raise SourceContractError("source byte validation failed:\n" + "\n".join(mismatches))

    return {
        "schema": snap["schema"],
        "executable_source_commit": commit,
        "executable_source_tree": tree,
        "github_branch": snap.get("github_branch", ""),
        "github_remote": snap.get("github_remote", ""),
        "generated_at_utc": snap.get("generated_at_utc", ""),
        "file_sha256_map": normalized_map,
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "source_snapshot_path": str(snapshot_path),
    }


def verify_sha256_manifest(
    root: Path,
    *,
    required_files: Iterable[Path] = (),
    require_all_files_listed: bool,
) -> dict:
    """Verify SHA256SUMS and every file it lists.

    For telemetry, PNGs may intentionally be unsealed, so
    ``require_all_files_listed`` can be false; every telemetry JSON actually
    consumed by the loader must still be passed through ``required_files``.
    """
    root = root.resolve(strict=True)
    sums = root / "SHA256SUMS"
    sums_sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sums_sidecar.is_file():
        raise SourceContractError(f"missing SHA256SUMS seal in {root}")

    actual_manifest_sha = sha256_file(sums)
    sidecar_lines = [line.strip() for line in sums_sidecar.read_text("utf-8").splitlines() if line.strip()]
    if len(sidecar_lines) != 1:
        raise SourceContractError(f"{sums_sidecar}: expected exactly one non-empty line")
    parts = sidecar_lines[0].split(maxsplit=1)
    if len(parts) != 2 or parts[1].strip() != "SHA256SUMS" or parts[0] != actual_manifest_sha:
        raise SourceContractError(f"{sums_sidecar}: SHA256SUMS digest mismatch")

    listed: dict[str, str] = {}
    for line_no, line in enumerate(sums.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise SourceContractError(f"{sums}:{line_no}: malformed line")
        digest, raw_rel = parts[0], parts[1].strip()
        if raw_rel.startswith("*"):
            raw_rel = raw_rel[1:]
        rel = _safe_relpath(raw_rel)
        if not SHA256_RE.fullmatch(digest):
            raise SourceContractError(f"{sums}:{line_no}: invalid SHA256")
        if rel in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise SourceContractError(f"{sums}:{line_no}: seal files must not be self-listed")
        if rel in listed:
            raise SourceContractError(f"{sums}:{line_no}: duplicate path {rel}")
        listed[rel] = digest

    if not listed:
        raise SourceContractError(f"empty SHA256SUMS: {sums}")

    mismatches = []
    for rel, expected in sorted(listed.items()):
        path = root / rel
        if path.is_symlink():
            mismatches.append(f"symlink:{rel}")
        elif not path.is_file():
            mismatches.append(f"missing:{rel}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                mismatches.append(f"sha:{rel}:expected={expected}:actual={actual}")
    if mismatches:
        raise SourceContractError("sealed-file verification failed:\n" + "\n".join(mismatches))

    required_rel = set()
    for required in required_files:
        if required.is_symlink():
            raise SourceContractError(f"required file is a symlink: {required}")
        resolved = required.resolve(strict=True)
        try:
            rel = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise SourceContractError(f"required file escapes sealed root: {resolved}") from exc
        required_rel.add(rel)
    absent_required = sorted(required_rel - set(listed))
    if absent_required:
        raise SourceContractError(
            "files consumed by the formal loader are absent from SHA256SUMS: "
            + ", ".join(absent_required)
        )

    if require_all_files_listed:
        actual_files = set()
        for p in root.rglob("*"):
            if p.is_symlink():
                raise SourceContractError(f"symlink in sealed root: {p}")
            if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
                actual_files.add(p.relative_to(root).as_posix())
        unlisted = sorted(actual_files - set(listed))
        unexpected = sorted(set(listed) - actual_files)
        if unlisted or unexpected:
            raise SourceContractError(
                f"seal closure failed: unlisted={unlisted[:20]} unexpected={unexpected[:20]}"
            )

    return {
        "root": str(root),
        "listed_file_count": len(listed),
        "listed_paths": sorted(listed),
        "sha256sums_sha256": actual_manifest_sha,
        "sha256sums_sidecar_sha256": sha256_file(sums_sidecar),
    }
