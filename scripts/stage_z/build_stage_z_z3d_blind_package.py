#!/usr/bin/env python3
"""Freeze the existing Z3-C manual videos into a neutral, blinded package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


MODEL_DIR = {"M0_OPENVLA": "M0_OPENVLA", "M1_OPENVLA_OFT": "M1_OPENVLA_OFT", "M2_PI05_LIBERO": "M2_PI05_LIBERO"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str, failures: list[str]) -> None:
    if not ok:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root-seal", type=Path, required=True)
    parser.add_argument("--copy-reconciliation", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    parser.add_argument("--hidden-mapping", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--git-tree", required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    manifest_path = rooted(args.manifest)
    root_seal_path = rooted(args.root_seal)
    reconciliation_path = rooted(args.copy_reconciliation)
    package_dir = rooted(args.package_dir)
    human_path = rooted(args.human_manifest)
    mapping_path = rooted(args.hidden_mapping)
    manifest, root_seal, reconciliation = load(manifest_path), load(root_seal_path), load(reconciliation_path)
    failures: list[str] = []
    require(reconciliation.get("status") == "STAGE_Z_Z3C_GIT_COPY_RECONCILIATION_PASS", "COPY_RECONCILIATION_STATUS", failures)
    require(manifest.get("status") == "STAGE_Z_Z3_EXECUTION_MANIFEST_FROZEN_NOT_EXECUTED", "MANIFEST_STATUS", failures)
    require(root_seal.get("status") == "PASS_Z3C_FIXED_MATRIX_COMPLETE", "ROOT_SEAL_STATUS", failures)
    jobs = [job for job in manifest.get("jobs", []) if job.get("blinded_video_id")]
    require(len(jobs) == 120, f"SELECTED_VIDEO_COUNT:{len(jobs)}", failures)
    require(len({job["manual_audit_id"] for job in jobs}) == 24, "AUDIT_PARENT_COUNT", failures)
    require(len({job["blinded_video_id"] for job in jobs}) == 120, "VIDEO_ID_DUPLICATE", failures)
    package_dir.mkdir(parents=True, exist_ok=True)
    human_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for job in jobs:
        video_id = str(job["blinded_video_id"])
        source = root / "stage_z_z3c_outputs_v1" / MODEL_DIR[str(job["model_family"])] / str(job["suite"]) / "manual_videos" / f"{video_id}.mp4"
        destination = package_dir / f"{video_id}.mp4"
        require(source.is_file(), f"MISSING_SOURCE:{video_id}", failures)
        if not source.is_file():
            continue
        source_sha, source_bytes = sha(source), source.stat().st_size
        if destination.exists():
            require(sha(destination) == source_sha and destination.stat().st_size == source_bytes, f"PACKAGE_EXISTING_MISMATCH:{video_id}", failures)
        else:
            shutil.copyfile(source, destination)
        package_sha, package_bytes = sha(destination), destination.stat().st_size
        require(package_sha == source_sha and package_bytes == source_bytes, f"PACKAGE_COPY_MISMATCH:{video_id}", failures)
        human_rows.append({"blinded_video_id": video_id, "manual_audit_id": str(job["manual_audit_id"]), "package_filename": f"{video_id}.mp4", "sha256": package_sha, "bytes": package_bytes, "label_status": "NOT_PROVIDED"})
        hidden_rows.append({"blinded_video_id": video_id, "manual_audit_id": str(job["manual_audit_id"]), "branch_id": str(job["branch_id"]), "model_family": str(job["model_family"]), "suite": str(job["suite"]), "arm": str(job["arm"]), "duration": int(job["duration"]), "canonical_parent_key": str(job["canonical_parent_key"]), "source_path": str(source.relative_to(root)).replace("\\", "/"), "source_sha256": source_sha, "source_bytes": source_bytes, "package_path": str(destination.relative_to(root)).replace("\\", "/"), "package_sha256": package_sha, "package_bytes": package_bytes})
    human_rows.sort(key=lambda row: row["blinded_video_id"])
    hidden_rows.sort(key=lambda row: row["blinded_video_id"])
    require(len(human_rows) == 120 and len(hidden_rows) == 120, "MATERIALIZED_VIDEO_COUNT", failures)
    human_doc = {
        "schema": "STAGE_Z_Z3D_BLINDED_VIDEO_MANIFEST_V1",
        "status": "PASS_Z3D_BLINDED_VIDEO_PACKAGE_FROZEN",
        "selection": {"source": "STAGE_Z_Z3_EXECUTION_MANIFEST_V2", "rule": "STAGE_Z_Z3_MANUAL_AUDIT_V1_20260823; up to two lowest hashes per model x suite", "replacement": False, "top_up": False, "regeneration": False, "selected_audit_parents": 24, "selected_videos": 120},
        "reviewer_firewall": {"hide_model_family": True, "hide_suite": True, "hide_arm_and_dose": True, "hide_branch_id": True, "hide_automatic_label": True, "hide_telemetry": True, "codex_must_not_generate_labels": True, "allowed_labels": ["STABLE_GRASP", "PREMATURE_APERTURE", "CONTACT_LOSS", "PREMATURE_RELEASE_OR_DROP", "OBJECT_DISPLACEMENT", "AMBIGUOUS_OR_OCCLUDED", "NOT_IDENTIFIABLE"]},
        "reviewer_instructions": "Review only the neutral package filenames. For each video, record one frozen vocabulary label and the existing visual boolean fields. Use AMBIGUOUS_OR_OCCLUDED or NOT_IDENTIFIABLE instead of guessing. Do not inspect hidden mapping or automatic telemetry before committing labels.",
        "source_manifest_sha256": sha(manifest_path),
        "z3c_root_seal_sha256": sha(root_seal_path),
        "copy_reconciliation_sha256": sha(reconciliation_path),
        "git_binding": {"head_commit": args.git_head, "head_tree": args.git_tree},
        "rows": human_rows,
    }
    hidden_doc = {
        "schema": "STAGE_Z_Z3D_BLINDED_VIDEO_MAPPING_V1",
        "status": "SEALED_HIDDEN_MAPPING_NOT_FOR_REVIEWER",
        "human_manifest_sha256": hashlib.sha256((json.dumps(human_doc, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
        "rows": hidden_rows,
    }
    human_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    human_path.write_text(json.dumps(human_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hidden_doc["human_manifest_sha256"] = sha(human_path)
    mapping_path.write_text(json.dumps(hidden_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(json.dumps({"status": "HOLD_Z3D_BLIND_PACKAGE_VALIDATION_FAILURE", "failures": failures[:20]}, sort_keys=True))
    print(json.dumps({"status": human_doc["status"], "videos": len(human_rows), "audit_parents": len({row["manual_audit_id"] for row in human_rows}), "human_manifest": str(human_path), "hidden_mapping": str(mapping_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
