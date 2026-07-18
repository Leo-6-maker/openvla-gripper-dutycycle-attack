#!/usr/bin/env python3
"""Independently audit a sealed FIT-only Physics Teacher V2 root."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import sha256_file
from gripper_attack.v5_physics import PHYSICS_TEACHER_FIELDS, PHYSICS_TEACHER_V21_FIELDS
from build_v5_physics_teacher import verify_sealed_root


EXPECTED_FIELDS = PHYSICS_TEACHER_FIELDS
_FLOAT_FIELDS = {
    "gripper_contact_score", "relative_pose_stability", "object_eef_comotion_score", "lift_score", "target_progress",
    "task_grasp_necessity", "stable_grasp_score", "release_risk", "regrasp_or_instability_risk", "support_removed",
    "utility_score", "teacher_confidence",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_seal(root: Path) -> None:
    payloads = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in payloads), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    teacher_root = args.teacher_root.resolve()
    teacher_seal = verify_sealed_root(teacher_root)
    manifest = _load_json(teacher_root / "physics_teacher_v2_manifest.json")
    if manifest.get("schema") not in {
        "DETECTOR_V5_PHYSICS_TEACHER_V2_MANIFEST",
        "DETECTOR_V5_PHYSICS_TEACHER_V21_MANIFEST",
    }:
        raise ValueError("unexpected Physics Teacher manifest schema")
    v21 = manifest["schema"] == "DETECTOR_V5_PHYSICS_TEACHER_V21_MANIFEST"
    protocol = _load_json(args.protocol.resolve())
    expected_protocol = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21" if v21 else "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V1"
    if protocol.get("schema") != expected_protocol:
        raise ValueError("Physics Teacher protocol/manifest version mismatch")
    if manifest.get("formal_training_authorized") is not False or manifest.get("formal_attack_authorized") is not False:
        raise ValueError("Physics Teacher root contains an authorization claim")
    if manifest.get("counterfactual_attack_label") is not False or manifest.get("student_future_leakage") is not False:
        raise ValueError("Physics Teacher root violates clean-only boundary")
    registry_seal = verify_sealed_root(args.registry_root.resolve())
    decoder_seal = verify_sealed_root(args.decoder_root.resolve())
    physics_seal = verify_sealed_root(args.physics_audit_root.resolve())
    if manifest.get("registry_csv_sha256") != sha256_file(args.registry_csv.resolve()):
        raise ValueError("registry CSV SHA mismatch")
    if manifest.get("protocol_sha256") != sha256_file(args.protocol.resolve()):
        raise ValueError("protocol SHA mismatch")
    if manifest.get("registry_root_sha256s_sha256") != registry_seal["sha256sums_sha256"]:
        raise ValueError("registry root binding mismatch")
    if manifest.get("decoder_root_sha256sums_sha256") != decoder_seal["sha256sums_sha256"]:
        raise ValueError("decoder root binding mismatch")
    if manifest.get("physics_audit_root_sha256sums_sha256") != physics_seal["sha256sums_sha256"]:
        raise ValueError("physics audit root binding mismatch")
    with args.registry_csv.resolve().open(newline="", encoding="utf-8") as handle:
        registry_rows = [row for row in csv.DictReader(handle) if row.get("split") == "FIT_TRAIN"]
    registry_keys = {row["canonical_parent_key"] for row in registry_rows}
    label_name = "physics_teacher_v21.jsonl" if v21 else "physics_teacher_v2.jsonl"
    label_files = sorted(teacher_root.glob(f"labels/*/task_*/state_*/{label_name}"))
    if len(registry_rows) != 800 or len(label_files) != 800 or registry_keys != {
        "/".join(path.relative_to(teacher_root / "labels").parts[:3]) for path in label_files
    }:
        raise ValueError("Teacher identity file closure mismatch")
    with (teacher_root / "task_roles.csv").open(newline="", encoding="utf-8") as handle:
        role_rows = list(csv.DictReader(handle))
    if len(role_rows) != 40 or any(row.get("status") == "ABSTAIN_DECODER_HOLD" for row in role_rows):
        raise ValueError("task role decoder hold remains")
    step_count = 0
    known_steps = 0
    tier_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    window_count = 0
    for path in label_files:
        relative = path.relative_to(teacher_root / "labels")
        identity = "/".join(relative.parts[:3])
        rows = _load_jsonl(path)
        if [int(row.get("step", -1)) for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous Teacher steps: {identity}")
        seen_windows: set[str] = set()
        for row in rows:
            expected_fields = PHYSICS_TEACHER_V21_FIELDS if v21 else EXPECTED_FIELDS
            if set(row) != expected_fields:
                raise ValueError(f"Teacher field whitelist mismatch: {identity}")
            if row["canonical_parent_key"] != identity or int(row["state_id"]) not in range(20):
                raise ValueError(f"Teacher identity mismatch: {identity}")
            if row["physics_protocol_schema"] != expected_protocol:
                raise ValueError(f"Teacher protocol mismatch: {identity}")
            if row["phase_name"] not in {"PRE_SUPPORT", "VALID_RETENTION", "RELEASE_IMMINENT_TAIL", "POST_RELEASE", "UNSTABLE_TRANSITION", "UNKNOWN"}:
                raise ValueError(f"Teacher phase mismatch: {identity}")
            if row["known_mask"] and row["utility_tier"] not in {0, 1, 2, 3}:
                raise ValueError(f"known Teacher row has invalid tier: {identity}")
            if not row["known_mask"] and row["utility_tier"] is not None:
                raise ValueError(f"unknown Teacher row has a tier: {identity}")
            for field in _FLOAT_FIELDS:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite Teacher value: {identity}/{field}")
            if row["candidate_close"]:
                start, end = int(row["window_start"]), int(row["window_end"])
                if int(row["step"]) < start or int(row["step"]) > end or not str(row["window_id"]).startswith("candidate:"):
                    raise ValueError(f"candidate window geometry mismatch: {identity}")
                seen_windows.add(str(row["window_id"]))
            elif not str(row["window_id"]).startswith("none:"):
                raise ValueError(f"non-candidate row has rankable window id: {identity}")
        step_count += len(rows)
        known_steps += sum(bool(row["known_mask"]) for row in rows)
        tier_counts.update(str(row["utility_tier"]) for row in rows if row["utility_tier"] is not None)
        phase_counts.update(row["phase_name"] for row in rows)
        window_count += len(seen_windows)
    if step_count != int(manifest.get("step_count")) or known_steps != int(manifest.get("known_step_count")):
        raise ValueError("manifest step counts do not match Teacher rows")
    if window_count != int(manifest.get("window_count")):
        raise ValueError("manifest window count does not match Teacher rows")
    if dict(sorted(tier_counts.items())) != manifest.get("utility_tier_step_counts"):
        raise ValueError("manifest tier counts do not match Teacher rows")
    if dict(sorted(phase_counts.items())) != manifest.get("phase_step_counts"):
        raise ValueError("manifest phase counts do not match Teacher rows")
    report = {
        "schema": "DETECTOR_V5_PHYSICS_TEACHER_V21_INDEPENDENT_AUDIT_V1" if v21 else "DETECTOR_V5_PHYSICS_TEACHER_V2_INDEPENDENT_AUDIT_V1",
        "status": "PASS",
        "teacher_root_sha256sums_sha256": teacher_seal["sha256sums_sha256"],
        "identity_count": len(label_files),
        "step_count": step_count,
        "known_step_count": known_steps,
        "window_count": window_count,
        "task_role_count": len(role_rows),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "input_binding.json").write_text(json.dumps({
            "teacher_root_sha256sums_sha256": teacher_seal["sha256sums_sha256"],
            "registry_root_sha256sums_sha256": registry_seal["sha256sums_sha256"],
            "decoder_root_sha256sums_sha256": decoder_seal["sha256sums_sha256"],
            "physics_audit_root_sha256sums_sha256": physics_seal["sha256sums_sha256"],
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "protocol_sha256": sha256_file(args.protocol.resolve()),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        os.replace(staging, output)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--decoder-root", type=Path, required=True)
    parser.add_argument("--physics-audit-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
