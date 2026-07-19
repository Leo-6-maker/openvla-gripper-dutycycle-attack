#!/usr/bin/env python3
"""R8.0.1b: exhaustive Official V3 visual recoverability census.

This is a read-only FIT audit. It recursively enumerates every file under every
frozen FIT identity and scans every JSON object / every JSONL row for visual
asset field names. It never runs a simulator, model, renderer, trainer, or
attack and never reads protected identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.b3_training_protocol import load_fit_fold_bundle

EXPECTED_FILES = {
    "episode_metadata.json",
    "episode_summary.json",
    "runtime_audit.json",
    "condition_config.json",
    "attack_config.json",
    "step_records.jsonl",
    "policy_intent_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
    "artifact_sha256.json",
}

SCANNED_STREAMS = (
    "episode_metadata.json",
    "episode_summary.json",
    "runtime_audit.json",
    "condition_config.json",
    "attack_config.json",
    "step_records.jsonl",
    "policy_intent_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
)

VISUAL_KEYWORDS = (
    "rgb",
    "image",
    "agentview",
    "pixel",
    "vision",
    "visual",
    "patch",
    "embedding",
    "hidden",
    "frame",
    "camera",
)

BINARY_EXTS = {
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".h5",
    ".hdf5",
    ".pkl",
    ".pickle",
    ".bin",
    ".zarr",
    ".lmdb",
    ".mp4",
    ".avi",
    ".mkv",
    ".webm",
    ".mov",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".xz",
}

GEOMETRY_KEYWORDS = (
    "eef",
    "gripper",
    "qpos",
    "object_state",
    "contact",
    "mujoco",
    "pose",
    "position",
    "velocity",
    "robot",
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal_root(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    root_sha = _sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{root_sha}  SHA256SUMS\n", encoding="utf-8"
    )
    return root_sha


def _identity_root(clean_root: Path, identity: str) -> Path:
    parts = identity.split("/")
    if len(parts) != 3:
        raise ValueError(f"invalid canonical identity: {identity}")
    return clean_root / parts[0] / parts[1] / parts[2]


def _keyword_matches(value: str) -> list[str]:
    lowered = value.lower()
    return [keyword for keyword in VISUAL_KEYWORDS if keyword in lowered]


def _flatten_keys(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            yield full
            yield from _flatten_keys(item, full)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                list_prefix = f"{prefix}[]" if prefix else "[]"
                yield from _flatten_keys(item, list_prefix)


def census_artifacts(clean_root: Path, identities: list[str]) -> dict[str, Any]:
    """Recursively enumerate every file under every requested FIT identity."""
    extension_counts: Counter[str] = Counter()
    filename_counts: Counter[str] = Counter()
    binary_files: list[dict[str, Any]] = []
    filename_keyword_hits: Counter[str] = Counter()
    filename_keyword_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    identity_rows: list[dict[str, Any]] = []
    missing_identities: list[str] = []
    total_files = 0

    for identity in identities:
        episode_root = _identity_root(clean_root, identity)
        if not episode_root.is_dir():
            missing_identities.append(identity)
            continue

        files = sorted(
            (path for path in episode_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(episode_root).as_posix(),
        )
        relative_files = [path.relative_to(episode_root).as_posix() for path in files]
        direct_names = {path.name for path in files if path.parent == episode_root}
        nested_files = [name for name in relative_files if "/" in name]
        identity_binary: list[str] = []

        for path, relative in zip(files, relative_files):
            total_files += 1
            suffix = path.suffix.lower() if path.suffix else "<none>"
            extension_counts[suffix] += 1
            filename_counts[path.name] += 1
            if suffix in BINARY_EXTS:
                identity_binary.append(relative)
                binary_files.append(
                    {
                        "identity": identity,
                        "path": relative,
                        "size_bytes": int(path.stat().st_size),
                        "sha256": _sha256_file(path),
                    }
                )
            for keyword in _keyword_matches(relative):
                filename_keyword_hits[keyword] += 1
                if len(filename_keyword_examples[keyword]) < 20:
                    filename_keyword_examples[keyword].append(
                        {"identity": identity, "path": relative}
                    )

        identity_rows.append(
            {
                "identity": identity,
                "file_count": len(files),
                "direct_file_count": len(direct_names),
                "nested_file_count": len(nested_files),
                "nested_files": nested_files,
                "binary_file_count": len(identity_binary),
                "binary_files": identity_binary,
                "missing_expected_files": sorted(EXPECTED_FILES - direct_names),
                "unexpected_direct_files": sorted(direct_names - EXPECTED_FILES),
                "exact_expected_file_set": direct_names == EXPECTED_FILES and not nested_files,
            }
        )

    return {
        "requested_identity_count": len(identities),
        "identity_count": len(identity_rows),
        "missing_identity_count": len(missing_identities),
        "missing_identities": missing_identities,
        "total_files": total_files,
        "extensions": dict(sorted(extension_counts.items())),
        "filenames": dict(sorted(filename_counts.items())),
        "binary_files": binary_files,
        "n_binary": len(binary_files),
        "filename_keyword_hits": dict(sorted(filename_keyword_hits.items())),
        "filename_keyword_examples": dict(filename_keyword_examples),
        "exact_expected_file_set_count": sum(
            bool(row["exact_expected_file_set"]) for row in identity_rows
        ),
        "identity_rows": identity_rows,
    }


def _scan_json_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    yield 1, value


def _scan_jsonl_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required: {path}:{line_number}")
            yield line_number, value


def field_census(clean_root: Path, identities: list[str]) -> dict[str, Any]:
    """Scan all eight semantic streams for all identities and all rows."""
    hit_counts: Counter[str] = Counter()
    hit_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stream_field_unions: dict[str, set[str]] = defaultdict(set)
    stream_rows_scanned: Counter[str] = Counter()
    missing_streams: list[dict[str, str]] = []
    identities_scanned = 0
    files_scanned = 0

    for identity in identities:
        episode_root = _identity_root(clean_root, identity)
        if not episode_root.is_dir():
            continue
        identities_scanned += 1

        for stream_name in SCANNED_STREAMS:
            path = episode_root / stream_name
            if not path.is_file():
                missing_streams.append({"identity": identity, "stream": stream_name})
                continue
            files_scanned += 1
            rows = _scan_json_file(path) if path.suffix == ".json" else _scan_jsonl_file(path)
            for row_number, value in rows:
                stream_rows_scanned[stream_name] += 1
                for field in _flatten_keys(value):
                    stream_field_unions[stream_name].add(field)
                    for keyword in _keyword_matches(field):
                        hit_counts[keyword] += 1
                        if len(hit_examples[keyword]) < 20:
                            hit_examples[keyword].append(
                                {
                                    "identity": identity,
                                    "stream": stream_name,
                                    "row": row_number,
                                    "field": field,
                                }
                            )

    return {
        "requested_identity_count": len(identities),
        "identities_scanned": identities_scanned,
        "files_scanned": files_scanned,
        "rows_scanned": int(sum(stream_rows_scanned.values())),
        "stream_rows_scanned": dict(sorted(stream_rows_scanned.items())),
        "missing_stream_count": len(missing_streams),
        "missing_streams": missing_streams,
        "keyword_hits": dict(sorted(hit_counts.items())),
        "keyword_hit_total": int(sum(hit_counts.values())),
        "details": dict(hit_examples),
        "stream_field_unions": {
            stream: sorted(fields) for stream, fields in sorted(stream_field_unions.items())
        },
    }


def teacher_geometry_summary(field_results: dict[str, Any]) -> dict[str, Any]:
    fields = set(
        field_results.get("stream_field_unions", {}).get(
            "privileged_teacher_sidecar.jsonl", []
        )
    )
    geometry_fields = sorted(
        field
        for field in fields
        if any(keyword in field.lower() for keyword in GEOMETRY_KEYWORDS)
    )
    return {
        "total_field_paths": len(fields),
        "all_field_paths": sorted(fields),
        "geometry_field_paths": geometry_fields,
        "geometry_field_count": len(geometry_fields),
    }


def classify_census(
    identities: list[str], artifact_census: dict[str, Any], field_results: dict[str, Any]
) -> tuple[str, list[str]]:
    errors: list[str] = []
    if len(identities) != 800:
        errors.append(f"fold identity count is {len(identities)}, expected 800")
    if artifact_census["identity_count"] != len(identities):
        errors.append("artifact identity closure failed")
    if artifact_census["exact_expected_file_set_count"] != len(identities):
        errors.append("not every identity has the exact nine-file artifact set")
    if field_results["identities_scanned"] != len(identities):
        errors.append("field census identity closure failed")
    if field_results["missing_stream_count"] != 0:
        errors.append("field census has missing semantic streams")
    if artifact_census["n_binary"] != 0:
        errors.append("binary/image/video/archive carrier files are present")
    if sum(artifact_census["filename_keyword_hits"].values()) != 0:
        errors.append("visual keyword appears in an artifact path")
    if field_results["keyword_hit_total"] != 0:
        errors.append("visual keyword appears in a semantic field path")

    if errors:
        if artifact_census["n_binary"] or field_results["keyword_hit_total"] or sum(
            artifact_census["filename_keyword_hits"].values()
        ):
            return "VISUAL_ASSET_CANDIDATE_PRESENT", errors
        return "HOLD_INCOMPLETE_CENSUS", errors
    return "NO_VISUAL_ASSET", []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output root already exists: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    try:
        git_commit = _git_commit()
        fold = load_fit_fold_bundle(args.fold_root.resolve())
        fold0 = next(item for item in fold["folds"] if item["fold_id"] == 0)
        fit_ids = sorted(
            set(fold0["train_identities"]) | set(fold0["validation_identities"])
        )

        artifact_census = census_artifacts(args.clean_root.resolve(), fit_ids)
        field_results = field_census(args.clean_root.resolve(), fit_ids)
        teacher = teacher_geometry_summary(field_results)
        status, errors = classify_census(fit_ids, artifact_census, field_results)

        identity_rows = artifact_census.pop("identity_rows")
        (staging / "IDENTITY_FILE_CENSUS.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in identity_rows
            ),
            encoding="utf-8",
        )
        report = {
            "schema": "R8_VISUAL_RECOVERABILITY_CENSUS_V2",
            "status": status,
            "errors": errors,
            "git_commit": git_commit,
            "clean_root": str(args.clean_root.resolve()),
            "fold_root": str(args.fold_root.resolve()),
            "fit_identity_count": len(fit_ids),
            "artifact_census": artifact_census,
            "field_census": field_results,
            "teacher_geometry": teacher,
            "protected_identity_reads": 0,
            "simulator_runs": 0,
            "model_inference_runs": 0,
            "source_mutation": 0,
        }
        (staging / "CENSUS_REPORT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema": "R8_VISUAL_CENSUS_MANIFEST_V2",
                    "status": status,
                    "fit_identity_count": len(fit_ids),
                    "total_files": artifact_census["total_files"],
                    "field_rows_scanned": field_results["rows_scanned"],
                    "formal_visual_training_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging / "commands.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
        root_sha = _seal_root(staging)
        os.replace(staging, output)
        print(
            json.dumps(
                {
                    "status": status,
                    "output_root": str(output),
                    "sha256s_sha256": root_sha,
                    "fit_identities": len(fit_ids),
                    "total_files": artifact_census["total_files"],
                    "field_rows_scanned": field_results["rows_scanned"],
                    "errors": errors,
                },
                sort_keys=True,
            )
        )
        return 0 if status == "NO_VISUAL_ASSET" else 2
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
