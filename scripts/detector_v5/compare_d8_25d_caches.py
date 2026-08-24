"""Canonical reproducibility comparator for two formal D8 25D caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file
from d8_source_contract import verify_sha256_manifest
from gripper_attack.seal_utils import rename_noreplace

ALLOWED_MANIFEST_DIFFERENCES = {"run_label", "run_uuid", "timestamp_utc"}
EXACT_FILES = ("FOLD_ASSIGNMENT.json", "IDENTITY_DISPOSITION.json")


def _write_seal(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def _load_cache(root: Path) -> tuple[dict, dict]:
    seal = verify_sha256_manifest(root, require_all_files_listed=True)
    manifest = json.loads((root / "CACHE_MANIFEST.json").read_text("utf-8"))
    if manifest.get("schema") != "DETECTOR_V3_D8_25D_CACHE_V3":
        raise RuntimeError(f"{root}: unsupported cache schema {manifest.get('schema')!r}")
    return manifest, seal


def _canonical_manifest(manifest: dict) -> dict:
    return {
        key: value for key, value in manifest.items()
        if key not in ALLOWED_MANIFEST_DIFFERENCES
    }


def compare_caches(cache_a: Path, cache_b: Path) -> dict:
    cache_a = cache_a.resolve(strict=True)
    cache_b = cache_b.resolve(strict=True)
    manifest_a, seal_a = _load_cache(cache_a)
    manifest_b, seal_b = _load_cache(cache_b)

    failures = []
    for field in ("input_seals", "code_snapshot", "script_provenance", "executed_loader"):
        if manifest_a.get(field) != manifest_b.get(field):
            failures.append(f"manifest field differs: {field}")

    canonical_a = _canonical_manifest(manifest_a)
    canonical_b = _canonical_manifest(manifest_b)
    if canonical_a != canonical_b:
        differing_keys = sorted(
            key for key in set(canonical_a) | set(canonical_b)
            if canonical_a.get(key) != canonical_b.get(key)
        )
        failures.append(f"canonical manifests differ: {differing_keys}")

    exact_file_results = {}
    for name in EXACT_FILES:
        path_a, path_b = cache_a / name, cache_b / name
        same = path_a.is_file() and path_b.is_file() and path_a.read_bytes() == path_b.read_bytes()
        exact_file_results[name] = same
        if not same:
            failures.append(f"byte mismatch: {name}")

    per_a = {path.name: path for path in (cache_a / "per_episode").glob("*.json")}
    per_b = {path.name: path for path in (cache_b / "per_episode").glob("*.json")}
    if len(per_a) != 670 or len(per_b) != 670 or set(per_a) != set(per_b):
        failures.append(
            f"per-episode identity closure failed: A={len(per_a)} B={len(per_b)}"
        )
    mismatched_episode_files = []
    for name in sorted(set(per_a) & set(per_b)):
        if per_a[name].read_bytes() != per_b[name].read_bytes():
            mismatched_episode_files.append(name)
    if mismatched_episode_files:
        failures.append(
            f"per-episode byte mismatches: {mismatched_episode_files[:20]}"
        )

    independent_runs = (
        manifest_a.get("run_uuid") != manifest_b.get("run_uuid")
        and manifest_a.get("run_label") != manifest_b.get("run_label")
    )
    if not independent_runs:
        failures.append("A/B run UUID and run label must differ")

    return {
        "schema": "D8_CACHE_AB_COMPARATOR_V1",
        "status": "PASS" if not failures else "FAIL",
        "cache_a": {
            "root": str(cache_a),
            "package_seal": seal_a["sha256sums_sha256"],
            "run_label": manifest_a.get("run_label"),
            "run_uuid": manifest_a.get("run_uuid"),
        },
        "cache_b": {
            "root": str(cache_b),
            "package_seal": seal_b["sha256sums_sha256"],
            "run_label": manifest_b.get("run_label"),
            "run_uuid": manifest_b.get("run_uuid"),
        },
        "allowed_manifest_differences": sorted(ALLOWED_MANIFEST_DIFFERENCES),
        "independent_runs": independent_runs,
        "input_seals_match": manifest_a.get("input_seals") == manifest_b.get("input_seals"),
        "source_binding_match": manifest_a.get("code_snapshot") == manifest_b.get("code_snapshot"),
        "script_provenance_match": manifest_a.get("script_provenance") == manifest_b.get("script_provenance"),
        "canonical_manifest_match": canonical_a == canonical_b,
        "exact_file_results": exact_file_results,
        "per_episode_file_count_a": len(per_a),
        "per_episode_file_count_b": len(per_b),
        "per_episode_identical": len(mismatched_episode_files) == 0 and set(per_a) == set(per_b),
        "per_episode_mismatches": mismatched_episode_files,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    report = compare_caches(args.cache_a, args.cache_b)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_root.with_name(f".{args.output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (staging / "CACHE_AB_COMPARATOR_RECEIPT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _write_seal(staging)
    rename_noreplace(staging, args.output_root)
    print(f"status={report['status']}")
    print(f"seal={digest}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
