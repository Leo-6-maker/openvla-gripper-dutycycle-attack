#!/usr/bin/env python3
"""Independent read-only audit for the phase-segmented Teacher V2.1.3 root."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from gripper_attack.v4_contract import sha256_file, verify_checksum_manifest


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"),
    )
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in payloads), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def audit_teacher_v213(teacher_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_seal = verify_checksum_manifest(teacher_root)
    manifest_path = teacher_root / "teacher_v213_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("teacher_v213_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "DETECTOR_V4_TEACHER_V213_V1_MANIFEST":
        raise ValueError("wrong Teacher V2.1.3 manifest schema")
    labels_paths = sorted(teacher_root.rglob("teacher_v213_labels.jsonl"))
    identity_count = len(labels_paths)
    xor_failures = 0
    malformed_windows = 0
    window_count = 0
    for path in labels_paths:
        rows = _jsonl(path)
        seen: dict[int, tuple[int, int, str]] = {}
        for index, row in enumerate(rows):
            if int(row.get("step", index)) != index:
                malformed_windows += 1
            event_id = int(row.get("event_id", -1))
            window_id = int(row.get("window_id", -1))
            segment = int(row.get("phase_segment_index", -1))
            phase = str(row.get("phase_name", row.get("phase", "UNKNOWN")))
            if bool(row.get("quality_valid", False)) and bool(row.get("veto_invalid", False)):
                xor_failures += 1
            if event_id < 0:
                if window_id != -1 or segment != -1:
                    malformed_windows += 1
                continue
            expected = event_id * 1000 + segment
            if segment < 0 or window_id != expected:
                malformed_windows += 1
            prior = seen.get(window_id)
            signature = (event_id, segment, phase)
            if prior is not None and prior != signature:
                malformed_windows += 1
            seen[window_id] = signature
        window_count += len(seen)
    status = "PASS" if (
        identity_count == 800
        and int(manifest.get("identity_count", 0)) == 800
        and xor_failures == 0
        and int(manifest.get("xor_failures", 1)) == 0
        and malformed_windows == 0
        and int(manifest.get("window_count", -1)) == window_count
    ) else "HOLD"
    report = {
        "schema": "DETECTOR_V4_TEACHER_V213_AUDIT_V1",
        "status": status,
        "teacher_root_sha256s_sha256": source_seal["sha256sums_sha256"],
        "teacher_manifest_sha256": sha256_file(manifest_path),
        "identity_count": identity_count,
        "xor_failures": xor_failures,
        "malformed_windows": malformed_windows,
        "window_count": window_count,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "teacher_v213_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "input_binding.json").write_text(json.dumps({"teacher_root_sha256s_sha256": source_seal["sha256sums_sha256"]}, indent=2) + "\n", encoding="utf-8")
        _seal(staging)
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_teacher_v213(args.teacher_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
