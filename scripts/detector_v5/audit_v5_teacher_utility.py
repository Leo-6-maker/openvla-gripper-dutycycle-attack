#!/usr/bin/env python3
"""Independent audit for the sealed V5 utility-proxy Teacher root."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import sha256_file, seal_directory
from gripper_attack.v5_protocol import validate_teacher_row


def _verify_recursive_seal(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError("V5 Teacher root is not sealed")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError("V5 Teacher root checksum sidecar mismatch")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid V5 Teacher checksum row: {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"V5 Teacher checksum mismatch: {name}")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != listed | {"SHA256SUMS", "SHA256SUMS.sha256"}:
        raise ValueError("V5 Teacher checksum file-set mismatch")
    return sha256_file(sums)


def audit(root: Path, expected_source_seal: str | None = None) -> dict[str, Any]:
    source_seal = _verify_recursive_seal(root)
    manifest_path = root / "v5_teacher_utility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "DETECTOR_V5_TEACHER_UTILITY_V1_MANIFEST":
        raise ValueError("unexpected V5 Teacher manifest schema")
    if expected_source_seal is not None and manifest.get("source_root_sha256s_sha256") != expected_source_seal:
        raise ValueError("V5 Teacher source seal does not match expected source")
    identities = 0
    steps = 0
    known = 0
    phases: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for labels_path in sorted(root.rglob("v5_teacher_utility.jsonl")):
        identity = "/".join(labels_path.relative_to(root).parts[:-1])
        values = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, row in enumerate(values):
            validate_teacher_row(row)
            if row.get("canonical_parent_key") != identity or int(row.get("step", -1)) != index:
                raise ValueError(f"V5 Teacher identity/step mismatch: {identity}:{index}")
        identities += 1
        steps += len(values)
        known += sum(bool(row["known_mask"]) for row in values)
        phases.update(str(row["phase_name"]) for row in values)
        tiers.update(str(row["utility_tier"]) for row in values if row["utility_tier"] is not None)
        rows.append({"canonical_parent_key": identity, "step_count": len(values), "known_step_count": sum(bool(row["known_mask"]) for row in values)})
    if identities != 800:
        raise ValueError(f"V5 Teacher identity count mismatch: {identities}")
    report = {
        "schema": "DETECTOR_V5_TEACHER_UTILITY_AUDIT_V1",
        "status": "PASS",
        "teacher_root_sha256s_sha256": source_seal,
        "teacher_manifest_sha256": sha256_file(manifest_path),
        "identity_count": identities,
        "step_count": steps,
        "known_step_count": known,
        "phase_counts": dict(sorted(phases.items())),
        "utility_tier_counts": dict(sorted(tiers.items())),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    if manifest.get("identity_count") != identities or manifest.get("step_count") != steps or manifest.get("known_step_count") != known:
        raise ValueError("V5 Teacher manifest aggregate mismatch")
    if manifest.get("utility_tier_counts") != report["utility_tier_counts"]:
        raise ValueError("V5 Teacher utility tier aggregate mismatch")
    return {"report": report, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--expected-source-seal")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite output root: {args.output_root}")
    result = audit(args.teacher_root.resolve(), args.expected_source_seal)
    staging = args.output_root.with_name(f".{args.output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "audit.json").write_text(json.dumps(result["report"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "identity_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["canonical_parent_key", "step_count", "known_step_count"])
            writer.writeheader()
            writer.writerows(result["rows"])
        seal_directory(staging)
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result["report"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
