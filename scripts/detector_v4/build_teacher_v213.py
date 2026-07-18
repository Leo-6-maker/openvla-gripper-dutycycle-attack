#!/usr/bin/env python3
"""Build the phase-segmented Teacher V2.1.3 derivative from V2.1.2.

V2.1.2 used ``window_id == event_id`` even when one event contained several
phase segments.  V2.1.3 keeps the event identity for reporting but assigns a
stable integer window to each contiguous ``(event_id, phase_name)`` segment.
The V2.1.2 root is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gripper_attack.v4_contract import FIT_STATES, PHASE_INDEX, SUITES, sha256_file, verify_checksum_manifest


WINDOW_MULTIPLIER = 1000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: str(p.relative_to(root)).replace(os.sep, "/"),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {str(path.relative_to(root)).replace(os.sep, '/')}\n" for path in payloads),
        encoding="utf-8",
    )
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def _transform_identity(source: Path, target: Path, identity: str) -> tuple[int, int, Counter, int]:
    labels_path = source / "teacher_v212_labels.jsonl"
    phases_path = source / "close_phases.json"
    if not labels_path.is_file() or not phases_path.is_file():
        raise ValueError(f"{identity}: missing V2.1.2 labels or close_phases")
    labels = _jsonl(labels_path)
    phases = json.loads(phases_path.read_text(encoding="utf-8"))

    transformed: list[dict[str, Any]] = []
    segment_index_by_event: dict[int, int] = defaultdict(lambda: -1)
    segment_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
    segment_names: dict[tuple[int, int], str] = {}
    last_event = -1
    last_phase = None
    xor_failures = 0
    supervised_steps = 0
    phase_counts: Counter = Counter()

    for index, row in enumerate(labels):
        if int(row.get("step", index)) != index:
            raise ValueError(f"{identity}: non-contiguous Teacher step {index}")
        quality = bool(row.get("quality_valid", False))
        veto = bool(row.get("veto_invalid", False))
        known = bool(row.get("known_mask", False))
        candidate = bool(row.get("candidate_close", False))
        event_id = int(row.get("event_id", -1))
        phase = str(row.get("phase_name", row.get("phase", "UNKNOWN")))
        exclusive = quality ^ veto
        if quality and veto:
            xor_failures += 1
        quality_mask = bool(known and candidate and exclusive)
        if quality_mask:
            supervised_steps += 1
        phase_counts[phase] += 1
        if event_id < 0:
            segment_index = -1
            last_event = -1
            last_phase = None
        else:
            if event_id not in segment_index_by_event:
                segment_index_by_event[event_id] = 0
            elif event_id != last_event or phase != last_phase:
                segment_index_by_event[event_id] += 1
            segment_index = segment_index_by_event[event_id]
            key = (event_id, segment_index)
            segment_rows[key].append(index)
            segment_names[key] = phase
            last_event = event_id
            last_phase = phase
        window_id = -1 if event_id < 0 else event_id * WINDOW_MULTIPLIER + segment_index
        transformed_row = dict(row)
        transformed_row.update({
            "schema": "DETECTOR_V4_TEACHER_V213_STEP_V1",
            "phase_id": PHASE_INDEX.get(phase, PHASE_INDEX["UNKNOWN"]),
            "phase_name": phase,
            "phase_segment_index": segment_index,
            "window_id": window_id,
            "window_start": -1,
            "window_end": -1,
            "exclusive_label": exclusive,
            "quality_supervision_mask": quality_mask,
            "release_known": bool(known and event_id >= 0),
        })
        transformed.append(transformed_row)

    window_segments = []
    for (event_id, segment_index), indices in sorted(segment_rows.items()):
        start, end = min(indices), max(indices)
        window_id = event_id * WINDOW_MULTIPLIER + segment_index
        for index in indices:
            transformed[index]["window_start"] = start
            transformed[index]["window_end"] = end
        window_segments.append({
            "event_id": event_id,
            "phase_segment_index": segment_index,
            "phase_name": segment_names[(event_id, segment_index)],
            "window_id": window_id,
            "start_step": start,
            "end_step": end,
        })

    target.mkdir(parents=True, exist_ok=True)
    (target / "teacher_v213_labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in transformed),
        encoding="utf-8",
    )
    (target / "close_phases.json").write_text(json.dumps(phases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "window_segments.json").write_text(json.dumps(window_segments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return xor_failures, supervised_steps, phase_counts, len(window_segments)


def build_teacher_v213(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_seal = verify_checksum_manifest(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        identities = 0
        xor_failures = 0
        supervised_steps = 0
        window_count = 0
        phases: Counter = Counter()
        for suite in SUITES:
            for task in range(10):
                for state in sorted(FIT_STATES):
                    identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                    source = source_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
                    target = staging / suite / f"task_{task:02d}" / f"state_{state:02d}"
                    failures, supervised, counts, windows = _transform_identity(source, target, identity)
                    identities += 1
                    xor_failures += failures
                    supervised_steps += supervised
                    window_count += windows
                    phases.update(counts)
        if identities != 800 or xor_failures:
            raise ValueError(f"Teacher V2.1.3 contract failed: identities={identities}, xor_failures={xor_failures}")
        manifest = {
            "schema": "DETECTOR_V4_TEACHER_V213_V1_MANIFEST",
            "source_root_sha256s_sha256": source_seal["sha256sums_sha256"],
            "source_teacher_schema": "DETECTOR_V4_TEACHER_V212_V1_MANIFEST",
            "identity_count": identities,
            "supervised_steps": supervised_steps,
            "window_count": window_count,
            "window_id_encoding": "event_id * 1000 + contiguous_phase_segment_index",
            "window_semantics": "contiguous_event_phase_segments",
            "xor_failures": xor_failures,
            "phase_counts": dict(sorted(phases.items())),
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        (staging / "teacher_v213_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    print(json.dumps(build_teacher_v213(args.source_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
