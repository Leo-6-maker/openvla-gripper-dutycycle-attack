#!/usr/bin/env python3
"""Build the corrected FIT-only Teacher V2.1.2 derivative root.

The source V2.1.1 root is never modified.  This pass only adds explicit
window/phase and supervision-mask fields required by the corrected trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

from gripper_attack.v4_contract import FIT_STATES, PHASE_INDEX, SUITES, sha256_file, verify_checksum_manifest


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _seal(root: Path) -> None:
    seal_names = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in seal_names),
        key=lambda p: str(p.relative_to(root)),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(p)}  {str(p.relative_to(root)).replace(os.sep, '/')}\n" for p in payloads),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")


def _copy_identity(source: Path, target: Path, identity: str) -> tuple[int, int, Counter]:
    labels_path = source / "teacher_v21_labels.jsonl"
    phases_path = source / "close_phases.json"
    if not labels_path.is_file() or not phases_path.is_file():
        raise ValueError(f"{identity}: missing V2.1 labels or close_phases")
    labels = _jsonl(labels_path)
    phases = json.loads(phases_path.read_text(encoding="utf-8"))
    bounds = {
        int(phase.get("event_id", -1)): (
            int(phase.get("start_step", 0)),
            int(phase.get("end_step", 0)),
        )
        for phase in phases.get("phases", [])
        if int(phase.get("event_id", -1)) >= 0
    }
    out_labels = []
    xor_failures = 0
    supervised_steps = 0
    phase_counts: Counter = Counter()
    for index, row in enumerate(labels):
        if int(row.get("step", index)) != index:
            raise ValueError(f"{identity}: non-contiguous teacher step {index}")
        quality = bool(row.get("quality_valid", False))
        veto = bool(row.get("veto_invalid", False))
        known = bool(row.get("known_mask", False))
        candidate = bool(row.get("candidate_close", False))
        event_id = int(row.get("event_id", -1))
        phase = str(row.get("phase", "UNKNOWN"))
        exclusive = quality ^ veto
        if quality and veto:
            xor_failures += 1
        quality_mask = bool(known and candidate and exclusive)
        release_known = bool(known and event_id >= 0)
        if quality_mask:
            supervised_steps += 1
        phase_counts[phase] += 1
        start, end = bounds.get(event_id, (-1, -1))
        transformed = dict(row)
        transformed.update(
            {
                "schema": "DETECTOR_V4_TEACHER_V212_STEP_V1",
                "phase_id": PHASE_INDEX.get(phase, PHASE_INDEX["UNKNOWN"]),
                "phase_name": phase,
                "window_id": event_id,
                "window_start": start,
                "window_end": end,
                "exclusive_label": exclusive,
                "quality_supervision_mask": quality_mask,
                "release_known": release_known,
            }
        )
        out_labels.append(transformed)

    target.mkdir(parents=True, exist_ok=True)
    (target / "teacher_v212_labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in out_labels),
        encoding="utf-8",
    )
    (target / "close_phases.json").write_text(
        json.dumps(phases, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return xor_failures, supervised_steps, phase_counts


def build_teacher_v212(source_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_seal = verify_checksum_manifest(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        identities = 0
        xor_failures = 0
        supervised_steps = 0
        phases = Counter()
        for suite in SUITES:
            for task in range(10):
                for state in sorted(FIT_STATES):
                    identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                    source = source_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
                    target = staging / suite / f"task_{task:02d}" / f"state_{state:02d}"
                    failures, supervised, counts = _copy_identity(source, target, identity)
                    identities += 1
                    xor_failures += failures
                    supervised_steps += supervised
                    phases.update(counts)
        if identities != 800 or xor_failures:
            raise ValueError(f"Teacher contract failed: identities={identities}, xor_failures={xor_failures}")
        manifest = {
            "schema": "DETECTOR_V4_TEACHER_V212_V1_MANIFEST",
            "source_root_sha256s_sha256": source_seal["sha256sums_sha256"],
            "identity_count": identities,
            "supervised_steps": supervised_steps,
            "xor_failures": xor_failures,
            "phase_counts": dict(sorted(phases.items())),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        (staging / "teacher_v212_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _seal(staging)
        os.replace(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_teacher_v212(args.source_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
