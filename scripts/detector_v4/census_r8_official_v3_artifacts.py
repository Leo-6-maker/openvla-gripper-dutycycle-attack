#!/usr/bin/env python3
"""R8.0.1: Official V3 visual recoverability census.

A. Enumerate all 800 FIT artifact files for binary containers
B. Field-level keyword search for visual data across all streams
"""

from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.b3_training_protocol import (
    load_fit_fold_bundle, verify_sealed_directory, sha256_file,
)

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FIT_STATES = list(range(0, 20))

VISUAL_KEYWORDS = [
    "rgb", "image", "agentview", "pixel", "vision", "visual",
    "patch", "embedding", "hidden", "frame", "camera",
]

BINARY_EXTS = {".npz", ".npy", ".pt", ".pth", ".h5", ".hdf5", ".pkl",
               ".bin", ".zarr", ".lmdb", ".mp4", ".avi", ".mkv", ".webm"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _seal_root(root: Path) -> str:
    exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted([f for f in root.rglob("*") if f.is_file() and f.name not in exclude],
                   key=lambda f: str(f.relative_to(root)))
    lines = []
    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    (root / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
    return sha


def census_artifacts(clean_root: Path, identities: list[str]) -> dict[str, Any]:
    """Enumerate all files across all 800 FIT identities."""
    all_extensions: dict[str, int] = defaultdict(int)
    all_filenames: dict[str, int] = defaultdict(int)
    binary_files: list[dict] = []
    total_files = 0
    identity_count = 0

    for identity in identities:
        parts = identity.split("/")
        ep_dir = clean_root / parts[0] / parts[1] / parts[2]
        if not ep_dir.is_dir():
            continue
        identity_count += 1
        for fp in ep_dir.iterdir():
            if fp.is_file():
                total_files += 1
                ext = fp.suffix.lower() if fp.suffix else "no_ext"
                all_extensions[ext] += 1
                all_filenames[fp.name] += 1
                if ext in BINARY_EXTS:
                    binary_files.append({
                        "identity": identity,
                        "filename": fp.name,
                        "size": fp.stat().st_size,
                    })

    return {
        "identity_count": identity_count,
        "total_files": total_files,
        "extensions": dict(all_extensions),
        "filenames": dict(all_filenames),
        "binary_files": binary_files,
        "n_binary": len(binary_files),
    }


def field_census(clean_root: Path, identities: list[str]) -> dict[str, Any]:
    """Scan key streams for visual-related keywords in field names."""
    streams = ["episode_metadata.json", "episode_summary.json", "runtime_audit.json",
               "condition_config.json", "attack_config.json",
               "step_records.jsonl", "policy_intent_records.jsonl",
               "privileged_teacher_sidecar.jsonl"]

    hits: dict[str, list[dict]] = defaultdict(list)
    identity_count = 0

    for identity in identities[:5]:  # First 5 for sampling
        parts = identity.split("/")
        ep_dir = clean_root / parts[0] / parts[1] / parts[2]
        if not ep_dir.is_dir():
            continue
        identity_count += 1

        for stream_name in streams:
            path = ep_dir / stream_name
            if not path.is_file():
                continue

            if stream_name.endswith(".json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                fields = list(_flatten_keys(data))
            elif stream_name.endswith(".jsonl"):
                with open(path, encoding="utf-8") as fh:
                    first = json.loads(fh.readline())
                fields = list(_flatten_keys(first))
            else:
                continue

            for field in fields:
                field_lower = field.lower()
                for kw in VISUAL_KEYWORDS:
                    if kw in field_lower:
                        hits[kw].append({
                            "identity": identity,
                            "stream": stream_name,
                            "field": field,
                        })

    return {"identities_sampled": identity_count, "keyword_hits": {k: len(v) for k, v in hits.items()},
            "details": {k: v[:3] for k, v in hits.items() if v}}


def _flatten_keys(obj: Any, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            yield full
            yield from _flatten_keys(v, full)
    elif isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            for k in obj[0].keys():
                full = f"{prefix}[].{k}" if prefix else f"[].{k}"
                yield full


def check_privileged_teacher_fields(clean_root: Path, identities: list[str]) -> dict[str, Any]:
    """Enumerate fields in privileged_teacher_sidecar.jsonl for geometry/oracle potential."""
    identity = identities[0]
    parts = identity.split("/")
    path = clean_root / parts[0] / parts[1] / parts[2] / "privileged_teacher_sidecar.jsonl"
    if not path.is_file():
        return {"error": "no sidecar found"}

    records = _jsonl(path)
    if not records:
        return {"error": "empty sidecar"}

    fields = set(records[0].keys())
    geometry_fields = [f for f in fields if any(kw in f.lower() for kw in
                       ["eef", "gripper", "qpos", "object_state", "contact",
                        "mujoco", "pose", "position", "velocity", "robot"])]
    return {
        "total_fields": len(fields),
        "all_fields": sorted(fields),
        "geometry_fields": sorted(geometry_fields),
        "n_steps": len(records),
    }


def main():
    ap = argparse.ArgumentParser(description="R8.0.1 Visual Recoverability Census")
    ap.add_argument("--clean-root", type=Path, required=True,
                    help="Official V3 clean campaign root (c2g_cs200_official_v3_20260716/clean)")
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"output root already exists: {out}")
    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    try:
        git_commit = _git_commit()
        print(f"=== R8.0.1 VISUAL RECOVERABILITY CENSUS ===\nGit: {git_commit}")

        fold = load_fit_fold_bundle(args.fold_root)
        fold0 = next(f for f in fold["folds"] if f["fold_id"] == 0)
        fit_ids = sorted(set(fold0["train_identities"]) | set(fold0["validation_identities"]))
        print(f"FIT identities: {len(fit_ids)}")

        # A. Artifact census
        print("\n--- A. Artifact Container Census ---")
        artifact_census = census_artifacts(args.clean_root, fit_ids)
        print(f"  Identities found: {artifact_census['identity_count']}/800")
        print(f"  Total files: {artifact_census['total_files']}")
        print(f"  Extensions: {artifact_census['extensions']}")
        print(f"  Binary containers: {artifact_census['n_binary']}")
        if artifact_census['binary_files']:
            for bf in artifact_census['binary_files'][:10]:
                print(f"    {bf['identity']}: {bf['filename']} ({bf['size']} bytes)")

        # B. Field census
        print("\n--- B. Field-Level Keyword Census ---")
        field_results = field_census(args.clean_root, fit_ids)
        print(f"  Identities sampled: {field_results['identities_sampled']}")
        print(f"  Keyword hit counts: {field_results['keyword_hits']}")
        for kw, details in field_results.get("details", {}).items():
            print(f"  {kw}:")
            for d in details[:3]:
                print(f"    {d['identity']}/{d['stream']}: {d['field']}")

        # C. Privileged teacher geometry
        print("\n--- C. Privileged Teacher Geometry ---")
        teacher = check_privileged_teacher_fields(args.clean_root, fit_ids)
        print(f"  Total fields: {teacher.get('total_fields', 'N/A')}")
        print(f"  Geometry/contact fields: {teacher.get('geometry_fields', [])}")
        if 'all_fields' in teacher:
            for f in teacher['all_fields']:
                print(f"    {f}")

        # Classification
        has_binary = artifact_census['n_binary'] > 0
        has_rgb_keywords = field_results['keyword_hits'].get('rgb', 0) > 0 or \
                           field_results['keyword_hits'].get('image', 0) > 0
        visual_asset_status = "NO_VISUAL_ASSET"
        if has_binary:
            visual_asset_status = "BINARY_CONTAINERS_PRESENT"
        if has_rgb_keywords:
            visual_asset_status = "VISUAL_KEYWORDS_PRESENT"

        print(f"\n=== CENSUS RESULT: {visual_asset_status} ===")

        # Write outputs
        (staging / "CENSUS_REPORT.json").write_text(json.dumps({
            "schema": "R8_VISUAL_RECOVERABILITY_CENSUS_V1",
            "git_commit": git_commit,
            "clean_root": str(args.clean_root),
            "fit_identities": len(fit_ids),
            "artifact_census": artifact_census,
            "field_census": field_results,
            "teacher_geometry": teacher,
            "visual_asset_status": visual_asset_status,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        (staging / "MANIFEST.json").write_text(json.dumps({
            "schema": "R8_VISUAL_CENSUS_MANIFEST_V1",
            "status": visual_asset_status,
        }, indent=2) + "\n", encoding="utf-8")

        (staging / "commands.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

        root_sha = _seal_root(staging)
        os.replace(staging, out)
        print(f"\nRoot: {out}\nSHA256SUMS: {root_sha}")

    except Exception:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
