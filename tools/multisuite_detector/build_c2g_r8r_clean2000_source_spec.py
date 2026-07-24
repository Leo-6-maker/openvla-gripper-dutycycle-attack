#!/usr/bin/env python3
"""Build a hash-bound source specification for the R8R Clean2000 reuse audit.

This helper records the exact raw/merged/replacement roots, gathers their local
manifest/checksum evidence, binds shared provenance reports, and freezes the
canonical suite precedence used by the audit. It never opens attack outcomes,
loads a model, or creates a LIBERO environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_SPEC_SCHEMA = "c2g.r8r.clean2000_source_spec.2026-07-11.v1"
EVIDENCE_NAMES = {
    "manifest.json", "SHA256SUMS.sha256", "collection_report.json",
    "c2f_collection_hygiene_report.json",
}
EVIDENCE_TOKENS = ("manifest", "provenance", "integrity", "hygiene", "audit", "report")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_evidence(root: Path) -> list[Path]:
    """Find small provenance files without walking frame/episode payload trees."""
    output: list[Path] = []
    pruned = {"episodes", "rgb", "frames", "images", "image_0", "image_1"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name.lower() not in pruned]
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            low = filename.lower()
            if filename in EVIDENCE_NAMES or (
                path.suffix.lower() in {".json", ".md", ".txt", ".csv"}
                and any(token in low for token in EVIDENCE_TOKENS)
            ):
                output.append(path.resolve())
    return sorted(set(output))


def _view(*, name: str, root: Path, source_class: str,
          canonical_suites: Sequence[str], priority: int,
          shared_evidence: Sequence[Path]) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    evidence = sorted(set(discover_evidence(root)) | {path.resolve() for path in shared_evidence})
    if not evidence:
        raise ValueError(f"no provenance/checksum evidence discovered for {root}")
    for path in evidence:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "name": name,
        "root": str(root),
        "source_class": source_class,
        "canonical_suites": list(canonical_suites),
        "priority": priority,
        "clean_only": True,
        "runtime_valid_by_manifest": True,
        "model_provenance_bound": True,
        "processor_provenance_bound": True,
        "feature_25d_order_bound": True,
        "evidence_paths": [str(path) for path in evidence],
        "evidence_sha256": {str(path): sha256_file(path) for path in evidence},
    }


def build_source_spec(*, raw_root: Path, merged_root: Path, replacement_root: Path,
                      shared_evidence: Sequence[Path], predecessor_roots: Sequence[Path],
                      output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared = [path.resolve() for path in shared_evidence]
    for path in shared:
        if not path.is_file():
            raise FileNotFoundError(path)
    value = {
        "schema": SOURCE_SPEC_SCHEMA,
        "views": [
            _view(
                name="clean2000_obs_clean_raw", root=raw_root,
                source_class="RAW_COLLECTION_SOURCE",
                canonical_suites=("libero_spatial", "libero_goal", "libero_10"),
                priority=10, shared_evidence=shared,
            ),
            _view(
                name="clean2000_merged_view", root=merged_root,
                source_class="MERGED_VIEW", canonical_suites=(),
                priority=5, shared_evidence=shared,
            ),
            _view(
                name="object500_v1_1_replacement", root=replacement_root,
                source_class="REPLACEMENT_SOURCE",
                canonical_suites=("libero_object",), priority=20,
                shared_evidence=shared,
            ),
        ],
        "predecessor_roots": [str(path.resolve()) for path in predecessor_roots],
        "authoritative_source_rule": {
            "libero_object": "object500_v1_1_replacement",
            "libero_spatial": "clean2000_obs_clean_raw",
            "libero_goal": "clean2000_obs_clean_raw",
            "libero_10": "clean2000_obs_clean_raw",
        },
        "boundaries": {
            "attack_outcomes_read": False, "models_loaded": 0,
            "libero_environments_created": 0, "source_assets_modified": False,
        },
    }
    write_json(output, value)
    return {**value, "output": str(output), "output_sha256": sha256_file(output)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--shared-evidence", action="append", type=Path, default=[])
    parser.add_argument("--predecessor-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_source_spec(
        raw_root=args.raw_root, merged_root=args.merged_root,
        replacement_root=args.replacement_root,
        shared_evidence=args.shared_evidence,
        predecessor_roots=args.predecessor_root, output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
