#!/usr/bin/env python3
"""Build the append-only Stage AE V2 reviewer packages.

V1 artifacts remain immutable.  V2 reuses the frozen AC4 video bytes and the
same deterministic local ordering, while replacing only the reviewer rubric
and the five-column return contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_human_review_packages import (
    AC4_MANIFEST,
    AC4_PACKAGE_SEAL,
    AC4_RUBRIC,
    AC4_SAMPLE,
    LEGAL_LABELS,
    ORDER_SALT,
    REVIEWERS,
    deterministic_order,
    digest_record,
    git_value,
    load_and_verify_ac4,
    package_digest,
    sha_bytes,
    sha_file,
    write_json,
)


V2_PROTOCOL = "configs/STAGE_AE_HUMAN_BLINDED_ENDPOINT_OBSERVABILITY_PROTOCOL_V2.json"
V2_INSTRUCTIONS = "docs/handoffs/STAGE_AE_HUMAN_REVIEWER_INSTRUCTIONS_V2.md"
V2_RUBRIC = "docs/handoffs/STAGE_AE_HUMAN_REVIEWER_RUBRIC_V2.txt"
V2_SCHEMA = "docs/handoffs/STAGE_AE_HUMAN_REVIEW_LABEL_SCHEMA_V2.csv"
V1_ETHICS = "docs/handoffs/STAGE_AE_HUMAN_REVIEW_ETHICS_CHECKLIST_V1.md"
V1_ROOT_SEAL = "reports/STAGE_AE_PRE_HUMAN_REVIEW_ROOT_SEAL_V1.json"
V1_MAPPING_SEAL = "reports/STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V1.json"
V2_ORDER_DIR = "reports/STAGE_AE_HUMAN_REVIEW_ORDER_MANIFESTS_V2"
V2_MAPPING = "reports/STAGE_AE_HUMAN_BLIND_MAPPING_V2.json"
V2_MAPPING_SEAL = "reports/STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V2.json"
V2_ROOT_SEAL = "reports/STAGE_AE_PRE_HUMAN_REVIEW_ROOT_SEAL_V2.json"
V2_PACKAGE_DIR = "reports/STAGE_AE_HUMAN_REVIEW_PACKAGES_V2"


def package_zip_v2(
    package_path: Path,
    reviewer: str,
    local_to_source: list[tuple[str, str]],
    video_bytes: dict[str, bytes],
    instructions: bytes,
    rubric: bytes,
    order_manifest: bytes,
) -> None:
    files: dict[str, bytes] = {
        "REVIEW_INSTRUCTIONS_V2.md": instructions,
        "REVIEW_RUBRIC_V2.txt": rubric,
        "ORDER_MANIFEST_V2.json": order_manifest,
    }
    package_rows: list[dict[str, Any]] = []
    for local_id, source_id in local_to_source:
        data = video_bytes[source_id]
        filename = f"videos/{local_id}.mp4"
        files[filename] = data
        package_rows.append(
            {
                "reviewer_clip_id": local_id,
                "package_filename": filename,
                "bytes": len(data),
                "sha256": sha_bytes(data),
            }
        )
    package_manifest = {
        "schema": "STAGE_AE_HUMAN_REVIEWER_PACKAGE_MANIFEST_V2",
        "reviewer_id": reviewer,
        "present_clip_count": len(local_to_source),
        "source_video_ids_hidden": True,
        "source_mapping_in_package": False,
        "rows": package_rows,
    }
    files["PACKAGE_MANIFEST_V2.json"] = (
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    files["LABEL_TEMPLATE.csv"] = (
        "reviewer_id,review_order_index,reviewer_clip_id,label,review_complete\n"
        + "".join(
            f"{reviewer},{index},{local_id},,\n"
            for index, (local_id, _) in enumerate(local_to_source, start=1)
        )
    ).encode("utf-8")
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
    package_dir = root / V2_PACKAGE_DIR
    if package_dir.exists() and any(package_dir.iterdir()):
        raise SystemExit("AE_V2_PACKAGE_OUTPUT_NOT_EMPTY")
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest, _, videos = load_and_verify_ac4(root, source_package)
    source_ids = sorted(videos)
    instructions = (root / V2_INSTRUCTIONS).read_bytes()
    rubric = (root / V2_RUBRIC).read_bytes()
    order_records: dict[str, dict[str, Any]] = {}
    mapping_rows: list[dict[str, Any]] = []
    package_paths: list[Path] = []

    for reviewer in REVIEWERS:
        ordered_sources = deterministic_order(reviewer, source_ids)
        local_to_source = [
            (f"{reviewer}-C{index:03d}", source_id)
            for index, source_id in enumerate(ordered_sources, start=1)
        ]
        order_doc = {
            "schema": "STAGE_AE_HUMAN_REVIEW_ORDER_MANIFEST_V2",
            "status": "SEALED_BEFORE_HUMAN_LABELING",
            "reviewer_id": reviewer,
            "order_salt": ORDER_SALT,
            "present_clip_count": 91,
            "reviewer_clip_ids": [local_id for local_id, _ in local_to_source],
        }
        order_path = root / V2_ORDER_DIR / f"{reviewer}_ORDER_MANIFEST_V2.json"
        write_json(order_path, order_doc)
        order_records[reviewer] = {
            "path": order_path.relative_to(root).as_posix(),
            "bytes": order_path.stat().st_size,
            "sha256": sha_file(order_path),
            "clip_count": len(local_to_source),
        }
        for local_id, source_id in local_to_source:
            source_row = next(
                item for item in manifest["rows"] if item["blinded_video_id"] == source_id
            )
            mapping_rows.append(
                {
                    "reviewer_id": reviewer,
                    "reviewer_clip_id": local_id,
                    "source_ac4_blinded_video_id": source_id,
                    "source_package_filename": source_row["package_filename"],
                    "source_bytes": source_row["bytes"],
                    "source_sha256": source_row["sha256"],
                }
            )
        package_path = package_dir / f"stage-ae-human-review-{reviewer}-v2.zip"
        package_zip_v2(
            package_path,
            reviewer,
            local_to_source,
            videos,
            instructions,
            rubric,
            order_path.read_bytes(),
        )
        package_paths.append(package_path)

    mapping_doc = {
        "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_V2",
        "status": "SEALED_PRIVATE_NOT_FOR_REVIEWER",
        "sealed_before_human_labeling": True,
        "mapping_in_reviewer_packages": False,
        "v1_artifacts_immutable": True,
        "source_ac4_package": package_digest(source_package),
        "source_ac4_manifest": digest_record(root, AC4_MANIFEST),
        "source_ac4_package_seal": digest_record(root, AC4_PACKAGE_SEAL),
        "order_contract": {
            "order_salt": ORDER_SALT,
            "same_deterministic_order_as_v1": True,
            "source_video_bytes_unchanged": True,
        },
        "counts": {
            "frozen_slots": 96,
            "present_videos": 91,
            "missing_frozen_videos": 5,
            "reviewers": 3,
            "future_label_rows": 273,
        },
        "rows": mapping_rows,
    }
    mapping_path = root / V2_MAPPING
    write_json(mapping_path, mapping_doc)
    mapping_seal = {
        "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V2",
        "status": "SEALED_BEFORE_HUMAN_LABELING",
        "supersedes_for_reviewer_package_authority": V1_MAPPING_SEAL,
        "mapping": digest_record(root, V2_MAPPING),
        "source_ac4_package": package_digest(source_package),
        "counts": {
            "frozen_slots": 96,
            "present_videos": 91,
            "missing_frozen_videos": 5,
            "reviewers": list(REVIEWERS),
            "future_label_rows": 273,
        },
        "order_manifests": order_records,
        "reviewer_packages": {
            reviewer: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha_file(path),
            }
            for reviewer, path in zip(REVIEWERS, package_paths)
        },
        "firewall": {
            "human_labels_read": False,
            "unblind_performed": False,
            "mapping_in_package": False,
            "source_video_bytes_changed": False,
        },
    }
    mapping_seal_path = root / V2_MAPPING_SEAL
    write_json(mapping_seal_path, mapping_seal)

    artifact_paths = [
        V2_PROTOCOL,
        V2_INSTRUCTIONS,
        V2_RUBRIC,
        V2_SCHEMA,
        V1_ETHICS,
        V2_MAPPING,
        V2_MAPPING_SEAL,
    ]
    artifact_paths.extend(
        f"{V2_ORDER_DIR}/{reviewer}_ORDER_MANIFEST_V2.json" for reviewer in REVIEWERS
    )
    artifact_paths.extend(f"{V2_PACKAGE_DIR}/{path.name}" for path in package_paths)
    artifact_paths.extend(
        [
            "scripts/stage_ae/build_human_review_packages_v2.py",
            "scripts/stage_ae/reconcile_human_blinded_reviews_v2.py",
            "scripts/stage_ae/test_reconcile_human_blinded_reviews_v2.py",
        ]
    )
    root_doc = {
        "schema": "STAGE_AE_PRE_HUMAN_REVIEW_ROOT_SEAL_V2",
        "status": "STAGE_AE_PRE_HUMAN_REVIEW_PACKAGE_V2_COMPLETE_STOP_FOR_PI",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "supersedes": {
            "v1_root_seal": digest_record(root, V1_ROOT_SEAL),
            "v1_mapping_seal": digest_record(root, V1_MAPPING_SEAL),
            "v1_artifacts_unchanged": True,
        },
        "source_authority": {
            "experiment_commit": git_value(root, "rev-parse", "HEAD"),
            "experiment_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
            "ac4_manifest": digest_record(root, AC4_MANIFEST),
            "ac4_sample": digest_record(root, AC4_SAMPLE),
            "ac4_package_external": package_digest(source_package),
            "ac4_package_seal": digest_record(root, AC4_PACKAGE_SEAL),
            "ac4_rubric": digest_record(root, AC4_RUBRIC),
        },
        "population": {
            "frozen_slots": 96,
            "present_videos": 91,
            "fixed_missing": 5,
            "reviewers": list(REVIEWERS),
            "future_label_rows": 273,
            "replacement": False,
            "top_up": False,
            "rerender": False,
            "source_video_bytes_changed": False,
        },
        "artifacts": {path: digest_record(root, path) for path in artifact_paths},
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
        },
        "review_boundary": {
            "human_labels_exist_in_repo": False,
            "human_labels_submitted": False,
            "phase_b_unblind_authorized": False,
            "ai_or_synthetic_labels_used": False,
            "technical_note_column": False,
            "terminal_requires_pi_review": True,
        },
        "path_normalization": {
            "artifact_paths_use_posix_separators": True,
            "backslash_paths_present": False,
        },
        "self_hash": "excluded_from_artifact_list_to_avoid_circularity",
    }
    write_json(root / V2_ROOT_SEAL, root_doc)
    print(
        json.dumps(
            {
                "status": root_doc["status"],
                "present": 91,
                "missing": 5,
                "packages": {
                    reviewer: package_digest(path)["sha256"]
                    for reviewer, path in zip(REVIEWERS, package_paths)
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
