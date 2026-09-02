#!/usr/bin/env python3
"""Synthetic-only end-to-end tests for the Stage AE V2 contract."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reconcile_human_blinded_reviews_v2 import (
    LEGAL_LABELS,
    REQUIRED_COLUMNS,
    REVIEWERS,
    V2_MAPPING,
    V2_MAPPING_SEAL,
    V2_ORDER_DIR,
    V2_PACKAGE_DIR,
    digest_record,
    fleiss_kappa,
    phase_a,
    phase_b,
    validate_label_rows,
    write_json,
)


def _package(root: Path, reviewer: str, clip_ids: list[str]) -> Path:
    package_path = root / V2_PACKAGE_DIR / f"stage-ae-human-review-{reviewer}-v2.zip"
    files: dict[str, bytes] = {
        "REVIEW_INSTRUCTIONS_V2.md": b"neutral instructions\n",
        "REVIEW_RUBRIC_V2.txt": b"neutral rubric\n",
        "ORDER_MANIFEST_V2.json": b"neutral order\n",
    }
    rows = []
    for index, clip_id in enumerate(clip_ids, start=1):
        name = f"videos/{clip_id}.mp4"
        data = f"synthetic-video-{index}".encode()
        files[name] = data
        rows.append(
            {
                "reviewer_clip_id": clip_id,
                "package_filename": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    files["PACKAGE_MANIFEST_V2.json"] = (
        json.dumps(
            {
                "schema": "STAGE_AE_HUMAN_REVIEWER_PACKAGE_MANIFEST_V2",
                "reviewer_id": reviewer,
                "present_clip_count": 91,
                "source_video_ids_hidden": True,
                "source_mapping_in_package": False,
                "rows": rows,
            },
            sort_keys=True,
        ).encode()
    )
    files["LABEL_TEMPLATE.csv"] = b"reviewer_id,review_order_index,reviewer_clip_id,label,review_complete\n"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return package_path


def _make_authority(root: Path) -> dict[str, list[str]]:
    order_ids: dict[str, list[str]] = {}
    mapping_rows = []
    package_entries = {}
    order_entries = {}
    for reviewer in REVIEWERS:
        clip_ids = [f"{reviewer}-C{index:03d}" for index in range(1, 92)]
        order_ids[reviewer] = clip_ids
        order_path = root / V2_ORDER_DIR / f"{reviewer}_ORDER_MANIFEST_V2.json"
        write_json(
            order_path,
            {
                "schema": "STAGE_AE_HUMAN_REVIEW_ORDER_MANIFEST_V2",
                "status": "SEALED_BEFORE_HUMAN_LABELING",
                "reviewer_id": reviewer,
                "present_clip_count": 91,
                "reviewer_clip_ids": clip_ids,
            },
        )
        order_entries[reviewer] = digest_record(order_path, f"{V2_ORDER_DIR}/{order_path.name}")
        package_path = _package(root, reviewer, clip_ids)
        package_entries[reviewer] = {
            "path": package_path.name,
            "bytes": package_path.stat().st_size,
            "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        }
        for index, clip_id in enumerate(clip_ids, start=1):
            mapping_rows.append(
                {
                    "reviewer_id": reviewer,
                    "reviewer_clip_id": clip_id,
                    "source_ac4_blinded_video_id": f"A4-C{index:03d}",
                }
            )
    mapping_path = root / V2_MAPPING
    write_json(
        mapping_path,
        {
            "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_V2",
            "status": "SEALED_PRIVATE_NOT_FOR_REVIEWER",
            "sealed_before_human_labeling": True,
            "counts": {
                "frozen_slots": 96,
                "present_videos": 91,
                "missing_frozen_videos": 5,
                "reviewers": 3,
                "future_label_rows": 273,
            },
            "rows": mapping_rows,
        },
    )
    write_json(
        root / V2_MAPPING_SEAL,
        {
            "schema": "STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V2",
            "status": "SEALED_BEFORE_HUMAN_LABELING",
            "mapping": digest_record(mapping_path, V2_MAPPING),
            "counts": {
                "frozen_slots": 96,
                "present_videos": 91,
                "missing_frozen_videos": 5,
                "reviewers": list(REVIEWERS),
                "future_label_rows": 273,
            },
            "order_manifests": order_entries,
            "reviewer_packages": package_entries,
        },
    )
    return order_ids


def _write_labels(labels_dir: Path, order_ids: dict[str, list[str]]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in REVIEWERS:
        rows = []
        for index, clip_id in enumerate(order_ids[reviewer], start=1):
            if index <= 30:
                label = "STABLE_GRASP"
            elif index <= 60:
                label = "STABLE_GRASP" if reviewer != "HR3" else "NOT_IDENTIFIABLE"
            elif index <= 90:
                label = {
                    "HR1": "STABLE_GRASP",
                    "HR2": "NOT_IDENTIFIABLE",
                    "HR3": "OBJECT_DISPLACEMENT",
                }[reviewer]
            else:
                label = "OBJECT_DISPLACEMENT"
            rows.append(
                {
                    "reviewer_id": reviewer,
                    "review_order_index": str(index),
                    "reviewer_clip_id": clip_id,
                    "label": label,
                    "review_complete": "true",
                }
            )
        with (labels_dir / f"{reviewer}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


class StageAEV2EndToEndTests(unittest.TestCase):
    def test_three_reviewer_91_clip_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage_ae_v2_test_") as temp_name:
            root = Path(temp_name)
            order_ids = _make_authority(root)
            labels_dir = root / "labels"
            seals_dir = root / "seals"
            preseal_path = root / "preseal.json"
            output_path = root / "reconciliation.json"
            _write_labels(labels_dir, order_ids)

            phase_a(root, labels_dir, seals_dir, preseal_path)
            preseal = json.loads(preseal_path.read_text(encoding="utf-8"))
            self.assertEqual(preseal["status"], "HUMAN_LABELS_SEALED_BEFORE_UNBLIND")
            self.assertFalse(preseal["phase_b_unblind_authorized"])
            self.assertEqual(preseal["counts"]["total_rows"], 273)

            result = phase_b(root, labels_dir, seals_dir, preseal_path, output_path)
            self.assertEqual(result["counts"]["unanimous_3_of_3"], 31)
            self.assertEqual(result["counts"]["majority_2_of_3"], 30)
            self.assertEqual(result["counts"]["all_distinct_1_of_1_of_1"], 30)
            self.assertEqual(sum(result["counts"][key] for key in (
                "unanimous_3_of_3", "majority_2_of_3", "all_distinct_1_of_1_of_1"
            )), 91)
            self.assertEqual(result["pairwise_raw_agreement"]["HR1_vs_HR2"], 61 / 91)
            self.assertEqual(result["pairwise_raw_agreement"]["HR1_vs_HR3"], 31 / 91)
            self.assertEqual(result["pairwise_raw_agreement"]["HR2_vs_HR3"], 31 / 91)
            self.assertAlmostEqual(
                result["fleiss_kappa"],
                fleiss_kappa([[item["labels"][reviewer] for reviewer in REVIEWERS] for item in result["unblinded_rows"]]),
            )
            self.assertEqual(result["binary_observability"]["per_reviewer"]["HR1"]["observable"], 91)
            self.assertEqual(result["binary_observability"]["per_reviewer"]["HR2"]["unobservable"], 30)
            self.assertEqual(result["binary_observability"]["per_reviewer"]["HR3"]["unobservable"], 30)
            self.assertFalse(result["automatic_endpoint_mutated"])
            self.assertFalse(result["denominators_mutated"])

    def test_v2_rejects_technical_note_column(self) -> None:
        rows = [
            {
                "reviewer_id": "HR1",
                "review_order_index": str(index),
                "reviewer_clip_id": f"HR1-C{index:03d}",
                "label": LEGAL_LABELS[0],
                "review_complete": "true",
            }
            for index in range(1, 92)
        ]
        rows[0]["technical_note"] = "not allowed"
        with self.assertRaises(ValueError):
            validate_label_rows("HR1", rows, [f"HR1-C{index:03d}" for index in range(1, 92)])


if __name__ == "__main__":
    unittest.main()
