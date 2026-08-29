#!/usr/bin/env python3
"""Build the AC4 outcome-blinded presentation package from frozen videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


SAMPLE = "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json"
PUBLIC_MANIFEST = "STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json"
PRIVATE_MAPPING = "STAGE_AC_AC4_NEUTRAL_BLIND_MAPPING_PRIVATE_V1.json"
PACKAGE = "STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_V1.zip"
SEAL = "STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json"
ID_SALT = "STAGE_AC_AC4_NEUTRAL_REPRESENTATION_ID_V1_20260828"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fresh_id(source_id: str) -> str:
    return "A4-" + hashlib.sha256(f"{ID_SALT}\0{source_id}".encode()).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    private_dir = args.private_dir.resolve()
    sample_path = (root / SAMPLE).resolve()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = list(sample.get("sample", []))
    if sample.get("status") != "STAGE_AC_AC4_BLIND_SAMPLE_FROZEN_PRE_TREATMENT" or len(rows) != 96:
        raise SystemExit("AC4_FROZEN_SAMPLE_INVALID")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"AC4_OUTPUT_NOT_EMPTY:{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    video_dir = out_dir / "videos"
    video_dir.mkdir()
    private_dir.mkdir(parents=True, exist_ok=True)

    present = {path.stem: path for path in source_dir.glob("*.mp4")}
    sample_ids = {str(row["blinded_video_id"]) for row in rows}
    if set(present) - sample_ids:
        raise SystemExit("AC4_SOURCE_HAS_UNAUTHORISED_VIDEO")
    if len(present) != 91:
        raise SystemExit(f"AC4_PRESENT_VIDEO_COUNT:{len(present)}")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["blinded_video_id"])):
        source_id = str(row["blinded_video_id"])
        neutral_id = fresh_id(source_id)
        source = present.get(source_id)
        public = {"blinded_video_id": neutral_id}
        private = {"blinded_video_id": neutral_id, "source_frozen_video_id": source_id, "frozen_sample_row": row}
        if source is None:
            public["availability"] = "FROZEN_SAMPLE_VIDEO_MISSING_NO_REPLACEMENT"
            private["availability"] = public["availability"]
        else:
            destination = video_dir / f"{neutral_id}.mp4"
            shutil.copyfile(source, destination)
            public.update({
                "availability": "PRESENT",
                "package_filename": f"videos/{neutral_id}.mp4",
                "bytes": destination.stat().st_size,
                "sha256": sha(destination),
                "fps": 10,
                "frames": 20,
                "width": 256,
                "height": 256,
            })
            private.update({
                "availability": "PRESENT",
                "source_filename": source.name,
                "source_bytes": source.stat().st_size,
                "source_sha256": sha(source),
                "package_filename": f"videos/{neutral_id}.mp4",
                "package_bytes": destination.stat().st_size,
                "package_sha256": sha(destination),
            })
        public_rows.append(public)
        private_rows.append(private)

    public_doc = {
        "schema": "STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1",
        "status": "STAGE_AC_AC4_NEUTRAL_PACKAGE_MATERIALIZED",
        "frozen_sample_sha256": sha(sample_path),
        "selection": {"preselected": True, "replacement": False, "top_up": False, "frozen_slots": 96},
        "presentation": {"fps": 10, "frames": 20, "width": 256, "height": 256, "short_source_padding": "clone_last_frame_only", "audio": "removed", "model_suite_parent_condition_hidden": True},
        "reviewer_firewall": {"model_family_hidden": True, "suite_hidden": True, "parent_hidden": True, "condition_hidden": True, "dose_hidden": True, "automatic_label_hidden": True, "telemetry_hidden": True, "source_video_id_hidden": True, "mapping_not_in_package": True},
        "counts": {"frozen_slots": 96, "present_videos": 91, "missing_frozen_videos": 5},
        "rows": public_rows,
    }
    public_path = out_dir / PUBLIC_MANIFEST
    write_json(public_path, public_doc)
    private_doc = {"schema": "STAGE_AC_AC4_NEUTRAL_BLIND_MAPPING_PRIVATE_V1", "status": "SEALED_PRIVATE_NOT_FOR_REVIEWER", "public_manifest_sha256": sha(public_path), "rows": private_rows}
    write_json(private_dir / PRIVATE_MAPPING, private_doc)

    rubric = """AC4 neutral visual audit rubric\n\nReview only visible video evidence. Do not infer model, suite, parent, arm, dose, automatic labels, or telemetry.\nUse one primary label per present clip:\nSTABLE_GRASP, PREMATURE_APERTURE, CONTACT_LOSS, PREMATURE_RELEASE_OR_DROP, OBJECT_DISPLACEMENT, AMBIGUOUS_OR_OCCLUDED, NOT_IDENTIFIABLE.\nUse AMBIGUOUS_OR_OCCLUDED or NOT_IDENTIFIABLE when the video does not establish the event.\n"""
    rubric_path = out_dir / "REVIEW_RUBRIC_V1.txt"
    rubric_path.write_text(rubric, encoding="utf-8")
    package_path = out_dir / PACKAGE
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [rubric_path, public_path, *sorted(video_dir.glob("*.mp4"))]:
            name = path.relative_to(out_dir).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    seal = {
        "schema": "STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1",
        "status": "STAGE_AC_AC4_NEUTRAL_PACKAGE_SEALED",
        "public_manifest": {"path": PUBLIC_MANIFEST, "bytes": public_path.stat().st_size, "sha256": sha(public_path)},
        "package": {"path": PACKAGE, "bytes": package_path.stat().st_size, "sha256": sha(package_path)},
        "review_rubric": {"path": "REVIEW_RUBRIC_V1.txt", "bytes": rubric_path.stat().st_size, "sha256": sha(rubric_path)},
        "counts": {"frozen_slots": 96, "present_videos": 91, "missing_frozen_videos": 5},
        "mapping": {"stored_separately": True, "included_in_package": False, "available_before_labels_sealed": False},
    }
    write_json(out_dir / SEAL, seal)
    print(json.dumps({"status": seal["status"], "present": 91, "missing": 5, "manifest_sha256": seal["public_manifest"]["sha256"], "package_sha256": seal["package"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
