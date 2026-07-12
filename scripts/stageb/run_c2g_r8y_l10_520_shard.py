#!/usr/bin/env python3
"""Run one hash-bound R8Y L10-520 collection shard.

Delegates to the existing R8W collector with the R8Y manifest.
The collector already reads max_steps from the manifest, so no
collector modification is needed — 520 is driven by the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
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
SCHEMA = "c2g.r8y.l10_520_shard.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8Y_L10_520_SHARD"
RUN_STATUS = "PASS_C2G_R8Y_L10_520_SHARD_RUN"
RECEIPT_SCHEMA = "c2g.r8y.l10_520_worker_receipt.2026-07-12.v1"
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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_manifest(episodes: Sequence[Mapping[str, Any]]) -> int:
    """Validate that all episodes have max_steps=520 and are L10."""
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--render-gpu-device-id", type=int, required=True)
    parser.add_argument("--worker-id", default="r8y_unknown")
    parser.add_argument("--shard-id", default="r8y_unknown")
    parser.add_argument("--dummy-wait", type=int, default=OFFICIAL_DUMMY_WAIT_STEPS)
    parser.add_argument("--ep-dir-prefix", default="episode_")
    parser.add_argument("--mode", choices=["preview", "run"], default="preview")
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    if sha256_file(manifest) != args.manifest_sha256:
        raise ValueError("manifest SHA256 mismatch")

    episodes = read_jsonl(manifest)
    validate_manifest(episodes)

    output_root = args.output_root.resolve()
    if args.mode == "run":
        if output_root.exists():
            raise FileExistsError(f"output root already exists: {output_root}")

    # Delegate to the existing collector with our manifest
    collector_args = [
        sys.executable, str(COLLECTOR),
        "--manifest", str(manifest),
        "--manifest-sha256", args.manifest_sha256,
        "--output-root", str(output_root),
        "--expected-git-commit", args.expected_git_commit,
        "--suite-model-map", str(args.suite_model_map),
        "--suite-model-report", str(args.suite_model_report),
        "--physical-gpu", str(args.physical_gpu),
        "--render-gpu-device-id", str(args.render_gpu_device_id),
        "--dummy-wait", str(args.dummy_wait),
        "--ep-dir-prefix", args.ep_dir_prefix,
    ]
    if args.mode == "preview":
        collector_args.append("--preview")

    result = subprocess.run(collector_args, cwd=REPO)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
