#!/usr/bin/env python3
"""Fail-closed Stage AE human-review reconciliation.

Phase A seals complete reviewer files without reading the hidden mapping.
Phase B requires an explicit local authorization flag and only then unblinds.
No result from this tool can rewrite the automatic endpoint or denominators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REVIEWERS = ("HR1", "HR2", "HR3")
REQUIRED_COLUMNS = ("reviewer_id", "review_order_index", "reviewer_clip_id", "label", "review_complete")
OPTIONAL_COLUMNS = ("technical_note",)
LEGAL_LABELS = (
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
)
UNOBSERVABLE = {"AMBIGUOUS_OR_OCCLUDED", "NOT_IDENTIFIABLE"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_record(path: Path, display_path: str | None = None) -> dict[str, Any]:
    return {"path": display_path or path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def expected_clip_ids(order_doc: dict[str, Any], reviewer: str) -> list[str]:
    if order_doc.get("schema") != "STAGE_AE_HUMAN_REVIEW_ORDER_MANIFEST_V1":
        raise ValueError("ORDER_SCHEMA_INVALID")
    if order_doc.get("status") != "SEALED_BEFORE_HUMAN_LABELING":
        raise ValueError("ORDER_NOT_SEALED")
    if order_doc.get("reviewer_id") != reviewer:
        raise ValueError("ORDER_REVIEWER_MISMATCH")
    ids = list(order_doc.get("reviewer_clip_ids", []))
    if len(ids) != 91 or len(set(ids)) != 91 or any(not item.startswith(f"{reviewer}-C") for item in ids):
        raise ValueError("ORDER_CLIP_IDS_INVALID")
    return ids


def validate_label_rows(reviewer: str, rows: Iterable[dict[str, str]], clip_ids: list[str]) -> list[dict[str, str]]:
    rows = list(rows)
    if len(rows) != 91:
        raise ValueError(f"LABEL_ROW_COUNT:{reviewer}:{len(rows)}")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"LABEL_REQUIRED_COLUMN_MISSING:{','.join(missing)}")
        extra = [column for column in row if column not in REQUIRED_COLUMNS and column not in OPTIONAL_COLUMNS]
        if extra:
            raise ValueError(f"LABEL_UNEXPECTED_COLUMN:{','.join(extra)}")
        if row["reviewer_id"] != reviewer:
            raise ValueError(f"LABEL_REVIEWER_MISMATCH:{reviewer}")
        try:
            order_index = int(row["review_order_index"])
        except ValueError as exc:
            raise ValueError("LABEL_ORDER_NOT_INTEGER") from exc
        if order_index != index or row["reviewer_clip_id"] != clip_ids[index - 1]:
            raise ValueError(f"LABEL_ORDER_MISMATCH:{reviewer}:{index}")
        local_id = row["reviewer_clip_id"]
        if local_id in seen:
            raise ValueError(f"LABEL_DUPLICATE:{reviewer}:{local_id}")
        seen.add(local_id)
        if row["label"] not in LEGAL_LABELS:
            raise ValueError(f"LABEL_ILLEGAL:{reviewer}:{local_id}")
        if row["review_complete"].strip().lower() != "true":
            raise ValueError(f"LABEL_REVIEW_NOT_COMPLETE:{reviewer}:{local_id}")
        normalized.append({key: value for key, value in row.items() if key in REQUIRED_COLUMNS or key in OPTIONAL_COLUMNS})
    if seen != set(clip_ids):
        raise ValueError(f"LABEL_MISSING_OR_UNEXPECTED_IDS:{reviewer}")
    return normalized


def read_reviewer_csv(path: Path, reviewer: str, clip_ids: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"LABEL_HEADER_MISSING:{reviewer}")
        return validate_label_rows(reviewer, reader, clip_ids)


def verify_package_zip(path: Path, expected: dict[str, Any], reviewer: str) -> None:
    if path.stat().st_size != int(expected["bytes"]) or sha_file(path) != expected["sha256"]:
        raise ValueError(f"PACKAGE_DIGEST_MISMATCH:{reviewer}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        videos = [name for name in names if name.startswith("videos/") and name.endswith(".mp4")]
        if len(videos) != 91 or len(set(videos)) != 91:
            raise ValueError(f"PACKAGE_VIDEO_COUNT:{reviewer}")
        lowered = "\n".join(names).lower()
        forbidden = ("mapping", "telemetry", "automatic", "model", "suite", "condition", "dose", "source_video_id")
        if any(term in lowered for term in forbidden):
            raise ValueError(f"PACKAGE_FIREWALL_NAME_LEAK:{reviewer}")
        for name in names:
            if name.endswith((".json", ".csv", ".md", ".txt")) and b"A4-" in archive.read(name):
                raise ValueError(f"PACKAGE_SOURCE_ID_LEAK:{reviewer}:{name}")
        if "PACKAGE_MANIFEST_V1.json" not in names or "ORDER_MANIFEST_V1.json" not in names:
            raise ValueError(f"PACKAGE_METADATA_MISSING:{reviewer}")
        package_manifest = json.loads(archive.read("PACKAGE_MANIFEST_V1.json").decode("utf-8"))
        rows = package_manifest.get("rows", [])
        if package_manifest.get("reviewer_id") != reviewer or len(rows) != 91:
            raise ValueError(f"PACKAGE_MANIFEST_INVALID:{reviewer}")


def verify_authority(repo_root: Path) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Path]]:
    seal_path = repo_root / "reports/STAGE_AE_HUMAN_BLIND_MAPPING_SEAL_V1.json"
    mapping_path = repo_root / "reports/STAGE_AE_HUMAN_BLIND_MAPPING_V1.json"
    seal = load_json(seal_path)
    mapping_digest = seal.get("mapping", {})
    if mapping_digest.get("bytes") != mapping_path.stat().st_size or mapping_digest.get("sha256") != sha_file(mapping_path):
        raise ValueError("MAPPING_SEAL_MISMATCH")
    mapping = load_json(mapping_path)
    if mapping.get("status") != "SEALED_PRIVATE_NOT_FOR_REVIEWER" or mapping.get("sealed_before_human_labeling") is not True:
        raise ValueError("MAPPING_STATUS_INVALID")
    order_ids: dict[str, list[str]] = {}
    order_paths: dict[str, Path] = {}
    for reviewer in REVIEWERS:
        entry = seal.get("order_manifests", {}).get(reviewer, {})
        order_path = repo_root / entry.get("path", "")
        if not order_path.is_file() or order_path.stat().st_size != int(entry.get("bytes", -1)) or sha_file(order_path) != entry.get("sha256"):
            raise ValueError(f"ORDER_DIGEST_MISMATCH:{reviewer}")
        order_ids[reviewer] = expected_clip_ids(load_json(order_path), reviewer)
        order_paths[reviewer] = order_path
        package_entry = seal.get("reviewer_packages", {}).get(reviewer, {})
        package_path = repo_root / "reports/STAGE_AE_HUMAN_REVIEW_PACKAGES" / package_entry.get("path", "")
        if package_path.name != f"stage-ae-human-review-{reviewer}.zip":
            raise ValueError(f"PACKAGE_PATH_INVALID:{reviewer}")
        verify_package_zip(package_path, package_entry, reviewer)
    if mapping.get("counts", {}).get("future_label_rows") != 273:
        raise ValueError("MAPPING_COUNT_INVALID")
    if len(mapping.get("rows", [])) != 273:
        raise ValueError("MAPPING_ROW_COUNT_INVALID")
    return seal, order_ids, {reviewer: repo_root / "reports/STAGE_AE_HUMAN_REVIEW_PACKAGES" / seal["reviewer_packages"][reviewer]["path"] for reviewer in REVIEWERS}


def phase_a(repo_root: Path, labels_dir: Path, seals_dir: Path, output: Path) -> dict[str, Any]:
    seal, order_ids, _ = verify_authority(repo_root)
    label_records: dict[str, dict[str, Any]] = {}
    for reviewer in REVIEWERS:
        label_path = labels_dir / f"{reviewer}.csv"
        if not label_path.is_file():
            raise ValueError(f"LABEL_FILE_MISSING:{reviewer}")
        rows = read_reviewer_csv(label_path, reviewer, order_ids[reviewer])
        reviewer_seal = {
            "schema": "STAGE_AE_HUMAN_REVIEWER_LABEL_SEAL_V1",
            "status": "SEALED_BEFORE_UNBLIND",
            "reviewer_id": reviewer,
            "labels_read_for_structural_validation": True,
            "label_values_persisted_in_this_seal": False,
            "rows": len(rows),
            "file": digest_record(label_path, f"{reviewer}.csv"),
        }
        reviewer_seal_path = seals_dir / f"{reviewer}_LABEL_SEAL_V1.json"
        write_json(reviewer_seal_path, reviewer_seal)
        label_records[reviewer] = {"rows": len(rows), "file": digest_record(label_path, f"{reviewer}.csv"), "seal": digest_record(reviewer_seal_path, f"{reviewer}_LABEL_SEAL_V1.json")}
    result = {
        "schema": "STAGE_AE_HUMAN_LABELS_SEALED_BEFORE_UNBLIND_V1",
        "status": "HUMAN_LABELS_SEALED_BEFORE_UNBLIND",
        "reviewers": label_records,
        "counts": {"reviewers": 3, "rows_per_reviewer": 91, "total_rows": 273},
        "mapping_sha256": seal["mapping"]["sha256"],
        "phase_b_unblind_authorized": False,
        "labels_summary_persisted": False,
        "automatic_endpoint_mutated": False,
        "denominators_mutated": False,
    }
    write_json(output, result)
    return result


def fleiss_kappa(labels_by_item: list[list[str]]) -> float:
    category_counts = [Counter(item) for item in labels_by_item]
    n = len(labels_by_item)
    r = len(reviewers)
    p_values = {label: sum(counts.get(label, 0) for counts in category_counts) / (n * r) for label in LEGAL_LABELS}
    p_bar = sum((sum(value * value for value in counts.values()) - r) / (r * (r - 1)) for counts in category_counts) / n
    p_e = sum(value * value for value in p_values.values())
    return 1.0 if p_e == 1.0 else (p_bar - p_e) / (1.0 - p_e)


def phase_b(repo_root: Path, labels_dir: Path, seals_dir: Path, preseal: Path, output: Path) -> dict[str, Any]:
    pre = load_json(preseal)
    if pre.get("status") != "HUMAN_LABELS_SEALED_BEFORE_UNBLIND" or pre.get("phase_b_unblind_authorized") is not False:
        raise ValueError("PRE_UNBLIND_SEAL_INVALID")
    seal, order_ids, _ = verify_authority(repo_root)
    label_rows = {reviewer: read_reviewer_csv(labels_dir / f"{reviewer}.csv", reviewer, order_ids[reviewer]) for reviewer in REVIEWERS}
    for reviewer in REVIEWERS:
        expected = pre["reviewers"][reviewer]["file"]
        actual = digest_record(labels_dir / f"{reviewer}.csv", f"{reviewer}.csv")
        if expected != actual:
            raise ValueError(f"LABEL_CHANGED_AFTER_SEAL:{reviewer}")
        expected_seal = seals_dir / f"{reviewer}_LABEL_SEAL_V1.json"
        if not expected_seal.is_file():
            raise ValueError(f"REVIEWER_SEAL_MISSING:{reviewer}")
    mapping = load_json(repo_root / "reports/STAGE_AE_HUMAN_BLIND_MAPPING_V1.json")
    map_rows = {(row["reviewer_id"], row["reviewer_clip_id"]): row["source_ac4_blinded_video_id"] for row in mapping["rows"]}
    local_by_source: dict[str, dict[str, str]] = {}
    for reviewer in REVIEWERS:
        for row in label_rows[reviewer]:
            source_id = map_rows[(reviewer, row["reviewer_clip_id"])]
            local_by_source.setdefault(source_id, {})[reviewer] = row["reviewer_clip_id"]
    if len(local_by_source) != 91 or any(set(local) != set(REVIEWERS) for local in local_by_source.values()):
        raise ValueError("UNBLIND_SOURCE_ALIGNMENT_INVALID")
    label_by_local = {reviewer: {row["reviewer_clip_id"]: row["label"] for row in label_rows[reviewer]} for reviewer in REVIEWERS}
    unblinded = []
    for source_id in sorted(local_by_source):
        item = {"source_ac4_blinded_video_id": source_id, "reviewer_clip_id_by_reviewer": local_by_source[source_id]}
        item["labels"] = {reviewer: label_by_local[reviewer][local_by_source[source_id][reviewer]] for reviewer in REVIEWERS}
        unblinded.append(item)
    labels_by_item = [[item["labels"][reviewer] for reviewer in REVIEWERS] for item in unblinded]
    pattern_counts = Counter("/".join(sorted(item)) for item in labels_by_item)
    unanimous = sum(len(set(item)) == 1 for item in labels_by_item)
    majority = sum(max(Counter(item).values()) == 2 for item in labels_by_item)
    all_distinct = sum(len(set(item)) == 3 for item in labels_by_item)
    pairwise = {}
    for left_index, left in enumerate(REVIEWERS):
        for right in REVIEWERS[left_index + 1:]:
            pairwise[f"{left}_vs_{right}"] = sum(item[left_index] == item[REVIEWERS.index(right)] for item in labels_by_item) / 91
    reviewer_distributions = {reviewer: dict(sorted(Counter(row["label"] for row in label_rows[reviewer]).items())) for reviewer in REVIEWERS}
    majority_labels = [Counter(item).most_common(1)[0][0] for item in labels_by_item if max(Counter(item).values()) == 2]
    observability = {
        reviewer: {"observable": sum(row["label"] not in UNOBSERVABLE for row in label_rows[reviewer]), "unobservable": sum(row["label"] in UNOBSERVABLE for row in label_rows[reviewer]), "total": 91}
        for reviewer in REVIEWERS
    }
    result = {
        "schema": "STAGE_AE_HUMAN_UNBLINDED_RECONCILIATION_V1",
        "status": "UNBLINDED_AFTER_THREE_REVIEWER_SEALS",
        "human_review_completed": True,
        "human_endpoint_confirmed": False,
        "human_endpoint_confirmed_interpretation": "NOT_INFERRED_BY_THIS_AUDIT",
        "counts": {"present_videos": 91, "fixed_missing_slots": 5, "unanimous_3_of_3": unanimous, "majority_2_of_3": majority, "all_distinct_1_of_1_of_1": all_distinct},
        "pairwise_raw_agreement": pairwise,
        "fleiss_kappa": fleiss_kappa(labels_by_item),
        "per_reviewer_label_distribution": reviewer_distributions,
        "majority_label_distribution": dict(sorted(Counter(majority_labels).items())),
        "agreement_pattern_counts": dict(sorted(pattern_counts.items())),
        "binary_observability": {"unobservable_labels": sorted(UNOBSERVABLE), "per_reviewer": observability, "interpretation": "Observability only; does not rewrite automatic endpoint, V_phys, denominators, or unknowns."},
        "source_ac4_mapping_used_after_seals": True,
        "automatic_endpoint_mutated": False,
        "denominators_mutated": False,
        "unblinded_rows": unblinded,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-unblind", "post-unblind"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--seals-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-unblind-seal", type=Path)
    parser.add_argument("--authorize-unblind", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "pre-unblind":
            phase_a(args.repo_root.resolve(), args.labels_dir.resolve(), args.seals_dir.resolve(), args.output.resolve())
        else:
            if not args.authorize_unblind:
                raise ValueError("EXPLICIT_UNBLIND_AUTHORIZATION_REQUIRED")
            if args.pre_unblind_seal is None:
                raise ValueError("PRE_UNBLIND_SEAL_REQUIRED")
            phase_b(args.repo_root.resolve(), args.labels_dir.resolve(), args.seals_dir.resolve(), args.pre_unblind_seal.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"STAGE_AE_RECONCILIATION_HOLD:{exc}") from exc


if __name__ == "__main__":
    main()
