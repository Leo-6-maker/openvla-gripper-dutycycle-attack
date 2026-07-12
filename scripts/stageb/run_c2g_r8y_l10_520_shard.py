#!/usr/bin/env python3
"""Run one hash-bound R8Y L10-520 collection shard.

Delegates to the existing R8W collector with the R8Y manifest.
The collector already reads max_steps from the manifest.

All collector-required arguments are exposed on the CLI so the
scheduler can pass them through without command-line mismatch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.c2g_official_suite_horizons import (
    OFFICIAL_DUMMY_WAIT_STEPS,
    OFFICIAL_MAX_POLICY_STEPS,
)

COLLECTOR = REPO / "scripts" / "stageb" / "collect_c2g_r8w_teacher_v2_clean.py"
TARGET_SUITE = "libero_10"
CANONICAL_MAX_STEPS = OFFICIAL_MAX_POLICY_STEPS[TARGET_SUITE]


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def validate_manifest(episodes: Sequence[Mapping[str, Any]]) -> int:
    steps = {row.get("max_steps") for row in episodes}
    if steps != {CANONICAL_MAX_STEPS}:
        raise ValueError(
            f"R8Y manifest max_steps must be {CANONICAL_MAX_STEPS}, got {steps}"
        )
    suites = {str(row.get("suite", "")) for row in episodes}
    if suites != {TARGET_SUITE}:
        raise ValueError(f"R8Y manifest must be L10 only, got {suites}")
    if len(episodes) != 25:
        raise ValueError(f"R8Y shard must have 25 episodes, got {len(episodes)}")
    return CANONICAL_MAX_STEPS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Required by collector + wrapper
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--model-load-lock-file", type=Path, required=True)
    parser.add_argument("--worker-status-file", type=Path, required=True)
    # Optional
    parser.add_argument("--dummy-wait", type=int, default=OFFICIAL_DUMMY_WAIT_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["preview", "run"], default="preview")
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    if sha256_file(manifest) != args.manifest_sha256:
        raise ValueError("manifest SHA256 mismatch")

    episodes = read_jsonl(manifest)
    validate_manifest(episodes)

    if args.mode == "run":
        output_root = args.output_root.resolve()
        if output_root.exists():
            raise FileExistsError(f"output root already exists: {output_root}")

    collector_args = [
        sys.executable, str(COLLECTOR),
        "--manifest", str(manifest),
        "--manifest-sha256", args.manifest_sha256,
        "--output-root", str(args.output_root),
        "--expected-git-commit", args.expected_git_commit,
        "--suite-model-map", str(args.suite_model_map),
        "--suite-model-report", str(args.suite_model_report),
        "--goal-model-manifest", str(args.goal_model_manifest),
        "--model-verification-report", str(args.model_verification_report),
        "--worker-id", args.worker_id,
        "--shard-id", args.shard_id,
        "--physical-gpu", str(args.physical_gpu),
        "--model-load-lock-file", str(args.model_load_lock_file),
        "--worker-status-file", str(args.worker_status_file),
        "--dummy-wait", str(args.dummy_wait),
    ]

    result = subprocess.run(collector_args, cwd=REPO)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
