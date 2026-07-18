"""Small, shared contracts for the corrected Official V3 Detector V4 line.

This module intentionally has no server paths and no experiment side effects.
It is the single source of truth for the 25D feature names/order and for
read-only evidence-root checks used by the trainer and evaluator.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sc5_detector_runtime import SC5_FEATURES

SC5_FEATURES = tuple(SC5_FEATURES)
FEATURE_INDEX = {name: index for index, name in enumerate(SC5_FEATURES)}
FEATURE_ORDER_SHA256 = hashlib.sha256(
    json.dumps(list(SC5_FEATURES), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()

FEATURE_NAMES_A = SC5_FEATURES
FEATURE_NAMES_B = FEATURE_NAMES_A + (
    "delta_gripper_qpos",
    "delta2_gripper_qpos",
    "gripper_command_qpos_deviation",
    "close_dwell_duration",
    "time_since_close_onset",
    "recent_close_count",
    "opening_trend",
    "recent_command_variance",
)
FEATURE_NAMES_C = FEATURE_NAMES_B + (
    "eef_velocity",
    "eef_acceleration",
    "eef_vertical_velocity",
    "eef_stability",
    "eef_displacement_since_close_onset",
    "action_consistency",
)
VIEW_FEATURE_NAMES = {"A": FEATURE_NAMES_A, "B": FEATURE_NAMES_B, "C": FEATURE_NAMES_C}
VIEW_FEATURE_COUNTS = {name: len(values) for name, values in VIEW_FEATURE_NAMES.items()}

PHASES = (
    "NO_CLOSE",
    "PRE_SUPPORT",
    "VALID_RETENTION",
    "RELEASE_IMMINENT_TAIL",
    "POST_RELEASE",
    "UNSTABLE_TRANSITION",
    "UNKNOWN",
)
PHASE_INDEX = {name: index for index, name in enumerate(PHASES)}

FIT_STATES = frozenset(range(20))
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def json_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Sequence[Path]) -> str:
    return json_sha([sha256_file(path) for path in sorted(paths, key=lambda item: str(item))])


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) not in ("", ".")


def verify_checksum_manifest(root: Path) -> dict[str, Any]:
    """Verify a sealed root and exact file-set closure without changing it."""
    sums_path = root / "SHA256SUMS"
    sidecar_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"missing checksum seal in {root}")

    sidecar_tokens = sidecar_path.read_text(encoding="utf-8").split()
    if len(sidecar_tokens) < 2 or sidecar_tokens[1] != "SHA256SUMS":
        raise ValueError(f"invalid SHA256SUMS sidecar in {root}")
    if sidecar_tokens[0].lower() != sha256_file(sums_path):
        raise ValueError(f"SHA256SUMS sidecar mismatch in {root}")

    listed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64 or not _safe_relative_path(parts[1].lstrip("*")):
            raise ValueError(f"invalid checksum line in {root}: {line!r}")
        rel = parts[1].lstrip("*")
        if rel in listed:
            raise ValueError(f"duplicate checksum path in {root}: {rel}")
        listed[rel] = parts[0].lower()

    mismatches = []
    for rel, expected in listed.items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(rel)
    if mismatches:
        raise ValueError(f"checksum mismatch in {root}: {mismatches[:5]}")

    actual = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if actual != set(listed):
        raise ValueError(
            f"checksum file-set mismatch in {root}: listed-only={sorted(set(listed)-actual)[:5]} "
            f"actual-only={sorted(actual-set(listed))[:5]}"
        )
    return {
        "status": "PASS",
        "root": str(root),
        "sha256sums_sha256": sha256_file(sums_path),
        "file_count": len(actual),
    }


def measured_git_binding(repo: Path, tracked_paths: Sequence[str]) -> dict[str, Any]:
    """Measure the runner binding from the checkout instead of trusting JSON."""
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(f"runner worktree is dirty: {dirty[:300]}")
    blobs = {}
    files = {}
    for path in tracked_paths:
        run("ls-files", "--error-unmatch", path)
        blobs[path] = run("rev-parse", f"HEAD:{path}")
        file_path = repo / path
        if not file_path.is_file():
            raise ValueError(f"tracked runner file is missing: {path}")
        committed = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"HEAD:{path}"], stderr=subprocess.STDOUT
        )
        actual = file_path.read_bytes()
        if hashlib.sha256(actual).hexdigest() != hashlib.sha256(committed).hexdigest():
            raise ValueError(f"runner file differs from HEAD: {path}")
        files[path] = hashlib.sha256(actual).hexdigest()
    payload = {
        "status": "PASS",
        "runner_repo": str(repo),
        "runner_head": head,
        "runner_worktree_clean": True,
        "tracked_blobs": blobs,
        "tracked_file_sha256": files,
    }
    payload["runner_binding_sha256"] = json_sha(payload)
    return payload


def parse_identity(identity: str) -> tuple[str, int, int]:
    parts = identity.split("/")
    if len(parts) != 3 or not parts[0] or not parts[1].startswith("task_") or not parts[2].startswith("state_"):
        raise ValueError(f"invalid canonical identity: {identity}")
    return parts[0], int(parts[1].split("_", 1)[1]), int(parts[2].split("_", 1)[1])


def identity_sha(identities: Sequence[str]) -> str:
    return json_sha(sorted(set(identities)))
