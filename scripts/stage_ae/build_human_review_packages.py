#!/usr/bin/env python3
"""Build the Stage AE human-review packages from the sealed AC4 package.

This builder copies video bytes only.  It does not render, pad, reorder frames,
inspect outcome metadata, or create labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AC4_MANIFEST = "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json"
AC4_PACKAGE_SEAL = "reports/STAGE_AC_AC4_NEUTRAL_BLIND_PACKAGE_SEAL_V1.json"
AC4_RUBRIC = "reports/STAGE_AC_AC4_NEUTRAL_BLIND_REVIEW_RUBRIC_V1.txt"
AC4_SAMPLE = "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json"

AE_PROTOCOL = "configs/STAGE_AE_HUMAN_BLINDED_ENDPOINT_OBSERVABILITY_PROTOCOL_V1.json"
AE_INSTRUCTIONS = "docs/handoffs/STAGE_AE_HUMAN_REVIEWER_INSTRUCTIONS_V1.md"
AE_RUBRIC = "docs/handoffs/STAGE_AE_HUMAN_REVIEWER_RUBRIC_V1.txt"
AE_SCHEMA = "docs/handoffs/STAGE_AE_HUMAN_REVIEW_LABEL_SCHEMA_V1.csv"
AE_ETHICS = "docs/handoffs/STAGE_AE_HUMAN_REVIEW_ETHICS_CHECKLIST_V1.md"
AE_ORDER_DIR = "reports/STAGE_AE_HUMAN_REVIEW_ORDER_MANIFESTS_V1"
AE_MAPPING = "reports/STAGE_AE_HUMAN_BLIND_MAPPING_V1.json"
AE_MAPPING_SEAL = "reports/STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V1.json"
AE_ROOT_SEAL = "reports/STAGE_AE_PRE_HUMAN_REVIEW_ROOT_SEAL_V1.json"
AE_PACKAGE_DIR = "reports/STAGE_AE_HUMAN_REVIEW_PACKAGES"

ORDER_SALT = "STAGE_AE_HUMAN_REVIEW_ORDER_SALT_V1"
REVIEWERS = ("HR1", "HR2", "HR3")
LEGAL_LABELS = (
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    write_bytes(path, data)


def read_json_bytes(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def digest_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha_file(path)}


def package_digest(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)}


def load_and_verify_ac4(root: Path, source_package: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    repo_manifest_path = root / AC4_MANIFEST
    repo_seal_path = root / AC4_PACKAGE_SEAL
    repo_rubric_path = root / AC4_RUBRIC
    package_bytes = source_package.read_bytes()
    seal = json.loads(repo_seal_path.read_text(encoding="utf-8"))
    expected_package = seal["package"]
    if len(package_bytes) != int(expected_package["bytes"]):
        raise SystemExit("AC4_SOURCE_PACKAGE_BYTE_MISMATCH")
    if sha_bytes(package_bytes) != expected_package["sha256"]:
        raise SystemExit("AC4_SOURCE_PACKAGE_SHA_MISMATCH")
    with zipfile.ZipFile(source_package) as archive:
        names = set(archive.namelist())
        required = {"STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json", "REVIEW_RUBRIC_V1.txt"}
        if not required.issubset(names):
            raise SystemExit("AC4_SOURCE_PACKAGE_REQUIRED_ENTRY_MISSING")
        package_manifest_bytes = archive.read("STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json")
        package_rubric_bytes = archive.read("REVIEW_RUBRIC_V1.txt")
        if package_manifest_bytes != repo_manifest_path.read_bytes():
            raise SystemExit("AC4_PACKAGE_MANIFEST_BYTES_MISMATCH")
        if package_rubric_bytes != repo_rubric_path.read_bytes():
            raise SystemExit("AC4_PACKAGE_RUBRIC_BYTES_MISMATCH")
        manifest = read_json_bytes(package_manifest_bytes)
        rows = list(manifest.get("rows", []))
        if manifest.get("schema") != "STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1" or len(rows) != 96:
            raise SystemExit("AC4_MANIFEST_SCHEMA_OR_COUNT_INVALID")
        present = [row for row in rows if row.get("availability") == "PRESENT"]
        missing = [row for row in rows if row.get("availability") != "PRESENT"]
        if len(present) != 91 or len(missing) != 5:
            raise SystemExit("AC4_MANIFEST_PRESENT_MISSING_COUNTS_INVALID")
        files: dict[str, bytes] = {}
        for row in present:
            name = str(row["package_filename"])
            if name not in names:
                raise SystemExit(f"AC4_SOURCE_VIDEO_MISSING:{name}")
            data = archive.read(name)
            if len(data) != int(row["bytes"]) or sha_bytes(data) != row["sha256"]:
                raise SystemExit(f"AC4_SOURCE_VIDEO_DIGEST_MISMATCH:{name}")
            files[str(row["blinded_video_id"])] = data
        if len(files) != 91:
            raise SystemExit("AC4_SOURCE_VIDEO_ID_COUNT_INVALID")
    if sha_file(repo_manifest_path) != seal["public_manifest"]["sha256"]:
        raise SystemExit("AC4_REPO_MANIFEST_SEAL_MISMATCH")
    if sha_file(repo_rubric_path) != seal["review_rubric"]["sha256"]:
        raise SystemExit("AC4_REPO_RUBRIC_SEAL_MISMATCH")
    return manifest, seal, files


def deterministic_order(reviewer: str, source_ids: list[str]) -> list[str]:
    return sorted(
        source_ids,
        key=lambda source_id: (
            sha_bytes(f"{ORDER_SALT}\0{reviewer}\0{source_id}".encode("utf-8")),
            source_id,
        ),
    )


def write_label_template(path: Path, reviewer: str, local_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["reviewer_id", "review_order_index", "reviewer_clip_id", "label", "review_complete"])
        for index, local_id in enumerate(local_ids, start=1):
            writer.writerow([reviewer, index, local_id, "", ""])


def package_zip(
    package_path: Path,
    reviewer: str,
    local_to_source: list[tuple[str, str]],
    video_bytes: dict[str, bytes],
    instructions: bytes,
    rubric: bytes,
    order_manifest: bytes,
) -> None:
    files: dict[str, bytes] = {
        "REVIEW_INSTRUCTIONS_V1.md": instructions,
        "REVIEW_RUBRIC_V1.txt": rubric,
        "ORDER_MANIFEST_V1.json": order_manifest,
    }
    package_rows: list[dict[str, Any]] = []
    for local_id, source_id in local_to_source:
        data = video_bytes[source_id]
        filename = f"videos/{local_id}.mp4"
        files[filename] = data
        package_rows.append({"reviewer_clip_id": local_id, "package_filename": filename, "bytes": len(data), "sha256": sha_bytes(data)})
    package_manifest = {
        "schema": "STAGE_AE_HUMAN_REVIEWER_PACKAGE_MANIFEST_V1",
        "reviewer_id": reviewer,
        "present_clip_count": len(local_to_source),
        "source_video_ids_hidden": True,
        "rows": package_rows,
    }
    files["PACKAGE_MANIFEST_V1.json"] = (json.dumps(package_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    label_template = bytearray()
    # The template is deliberately produced as bytes here so its newline style is stable.
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["reviewer_id", "review_order_index", "reviewer_clip_id", "label", "review_complete"])
    for index, (local_id, _) in enumerate(local_to_source, start=1):
        writer.writerow([reviewer, index, local_id, "", ""])
    files["LABEL_TEMPLATE.csv"] = stream.getvalue().encode("utf-8")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source_package = args.source_package.resolve()
    package_dir = root / AE_PACKAGE_DIR
    if package_dir.exists() and any(package_dir.iterdir()):
        raise SystemExit("AE_PACKAGE_OUTPUT_NOT_EMPTY")
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest, ac4_seal, videos = load_and_verify_ac4(root, source_package)
    source_ids = sorted(videos)
    instructions = (root / AE_INSTRUCTIONS).read_bytes()
    rubric = (root / AE_RUBRIC).read_bytes()

    order_records: dict[str, dict[str, Any]] = {}
    mapping_rows: list[dict[str, Any]] = []
    package_paths: list[Path] = []
    order_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="stage_ae_build_", dir=str(package_dir.parent)) as temp_name:
        temp = Path(temp_name)
        for reviewer in REVIEWERS:
            ordered_sources = deterministic_order(reviewer, source_ids)
            local_to_source = [(f"{reviewer}-C{index:03d}", source_id) for index, source_id in enumerate(ordered_sources, start=1)]
            local_ids = [local_id for local_id, _ in local_to_source]
            order_doc = {
                "schema": "STAGE_AE_HUMAN_REVIEW_ORDER_MANIFEST_V1",
                "status": "SEALED_BEFORE_HUMAN_LABELING",
                "reviewer_id": reviewer,
                "order_salt": ORDER_SALT,
                "present_clip_count": 91,
                "reviewer_clip_ids": local_ids,
            }
            order_path = root / AE_ORDER_DIR / f"{reviewer}_ORDER_MANIFEST_V1.json"
            write_json(order_path, order_doc)
            order_bytes = order_path.read_bytes()
            order_paths.append(order_path)
            order_records[reviewer] = {"path": str(order_path.relative_to(root)).replace("\\", "/"), "bytes": len(order_bytes), "sha256": sha_bytes(order_bytes), "clip_count": len(local_ids)}
            for local_id, source_id in local_to_source:
                row = next(item for item in manifest["rows"] if item["blinded_video_id"] == source_id)
                mapping_rows.append({
                    "reviewer_id": reviewer,
                    "reviewer_clip_id": local_id,
                    "source_ac4_blinded_video_id": source_id,
                    "source_package_filename": row["package_filename"],
                    "source_bytes": row["bytes"],
                    "source_sha256": row["sha256"],
                })
            package_stage = temp / f"stage-ae-human-review-{reviewer}"
            package_stage.mkdir()
            label_path = package_stage / "LABEL_TEMPLATE.csv"
            write_label_template(label_path, reviewer, local_ids)
            package_path = package_dir / f"stage-ae-human-review-{reviewer}.zip"
            package_zip(package_path, reviewer, local_to_source, videos, instructions, rubric, order_bytes)
            package_paths.append(package_path)
    mapping_doc = {
        "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_V1",
        "status": "SEALED_PRIVATE_NOT_FOR_REVIEWER",
        "sealed_before_human_labeling": True,
        "mapping_in_reviewer_packages": False,
        "source_ac4_package": package_digest(source_package),
        "source_ac4_manifest": digest_record(root, AC4_MANIFEST),
        "source_ac4_package_seal": digest_record(root, AC4_PACKAGE_SEAL),
        "counts": {"frozen_slots": 96, "present_videos": 91, "missing_frozen_videos": 5, "reviewers": 3, "future_label_rows": 273},
        "rows": mapping_rows,
    }
    mapping_path = root / AE_MAPPING
    write_json(mapping_path, mapping_doc)
    mapping_seal = {
        "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V1",
        "status": "SEALED_BEFORE_HUMAN_LABELING",
        "mapping": digest_record(root, AE_MAPPING),
        "source_ac4_package": package_digest(source_package),
        "counts": {"frozen_slots": 96, "present_videos": 91, "missing_frozen_videos": 5, "reviewers": ["HR1", "HR2", "HR3"], "future_label_rows": 273},
        "order_manifests": order_records,
        "reviewer_packages": {reviewer: package_digest(path) for reviewer, path in zip(REVIEWERS, package_paths)},
        "firewall": {"human_labels_read": False, "unblind_performed": False, "mapping_in_package": False},
    }
    mapping_seal_path = root / AE_MAPPING_SEAL
    write_json(mapping_seal_path, mapping_seal)
    mapping_seal = json.loads(mapping_seal_path.read_text(encoding="utf-8"))

    artifact_paths = [AE_PROTOCOL, AE_INSTRUCTIONS, AE_RUBRIC, AE_SCHEMA, AE_ETHICS, AE_MAPPING, AE_MAPPING_SEAL]
    artifact_paths.extend(str(Path(AE_ORDER_DIR) / f"{reviewer}_ORDER_MANIFEST_V1.json") for reviewer in REVIEWERS)
    artifact_paths.extend(str(Path(AE_PACKAGE_DIR) / path.name) for path in package_paths)
    artifact_paths.extend(["scripts/stage_ae/build_human_review_packages.py", "scripts/stage_ae/reconcile_human_blinded_reviews.py", "scripts/stage_ae/test_reconcile_human_blinded_reviews.py"])
    root_doc = {
        "schema": "STAGE_AE_PRE_HUMAN_REVIEW_ROOT_SEAL_V1",
        "status": "STAGE_AE_PRE_HUMAN_REVIEW_PACKAGE_COMPLETE_STOP_FOR_PI",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_authority": {
            "experiment_commit": git_value(root, "rev-parse", "HEAD"),
            "experiment_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
            "ac4_manifest": digest_record(root, AC4_MANIFEST),
            "ac4_sample": digest_record(root, AC4_SAMPLE),
            "ac4_package_external": package_digest(source_package),
            "ac4_package_seal": digest_record(root, AC4_PACKAGE_SEAL),
            "ac4_rubric": digest_record(root, AC4_RUBRIC),
        },
        "population": {"frozen_slots": 96, "present_videos": 91, "fixed_missing": 5, "reviewers": list(REVIEWERS), "future_label_rows": 273, "replacement": False, "top_up": False, "rerender": False},
        "artifacts": {relative: digest_record(root, relative) for relative in artifact_paths},
        "firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_pgd": 0,
            "new_simulator_execution": 0,
            "new_identity": 0,
            "relabeling": 0,
            "denominator_modification": 0,
            "protected_or_eval160_read": 0,
            "human_label_read": 0,
            "unblind_performed": 0,
            "human_review_completed": False,
            "human_endpoint_confirmed": False,
        },
        "review_boundary": {"human_labels_exist_in_repo": False, "human_labels_submitted": False, "phase_b_unblind_authorized": False, "ai_or_synthetic_labels_used": False, "terminal_requires_pi_review": True},
        "self_hash": "excluded_from_artifact_list_to_avoid_circularity",
    }
    write_json(root / AE_ROOT_SEAL, root_doc)
    print(json.dumps({"status": root_doc["status"], "present": 91, "missing": 5, "packages": {reviewer: package_digest(path)["sha256"] for reviewer, path in zip(REVIEWERS, package_paths)}}, sort_keys=True))


if __name__ == "__main__":
    main()
