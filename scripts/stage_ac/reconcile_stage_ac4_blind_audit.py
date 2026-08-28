#!/usr/bin/env python3
"""CPU-only reconciliation after AC4 AI-secondary labels are sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = {
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cross(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[str(row.get(left))][str(row.get(right))] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def grouped_counts(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)][str(row["primary_label"])] += 1
    return [
        {**{key: value for key, value in zip(keys, group)}, "labels": dict(sorted(counts.items()))}
        for group, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--package-seal", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-seal", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--branch-index", type=Path, required=True)
    parser.add_argument("--g3-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = load(args.manifest)
    package_seal = load(args.package_seal)
    labels = load(args.labels)
    label_seal = load(args.label_seal)
    private = load(args.private_map)
    index = load(args.branch_index)
    g3 = load(args.g3_stats)
    failures: list[str] = []
    if manifest.get("schema") != "STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1": failures.append("MANIFEST_SCHEMA")
    if package_seal.get("status") != "STAGE_AC_AC4_NEUTRAL_PACKAGE_SEALED": failures.append("PACKAGE_SEAL_STATUS")
    if label_seal.get("labels_sealed_before_unblind") is not True: failures.append("LABELS_NOT_SEALED_BEFORE_UNBLIND")
    if label_seal.get("hidden_mapping_read_before_label_seal") is not False: failures.append("MAPPING_ORDER")
    if label_seal.get("reviewer_type") != "AI_SECONDARY": failures.append("REVIEWER_TYPE")
    if label_seal.get("human_review_gate_satisfied") is not False: failures.append("HUMAN_GATE_FLAG")
    if label_seal.get("label_sha256") != sha(args.labels): failures.append("LABEL_SHA_BINDING")
    if label_seal.get("manifest_sha256") != sha(args.manifest): failures.append("MANIFEST_SHA_BINDING")

    public_rows = manifest.get("rows", [])
    public_by_id = {row.get("blinded_video_id"): row for row in public_rows}
    present = {key: row for key, row in public_by_id.items() if row.get("availability") == "PRESENT"}
    missing = {key: row for key, row in public_by_id.items() if row.get("availability") != "PRESENT"}
    if len(public_rows) != 96 or len(public_by_id) != 96: failures.append("MANIFEST_ROW_COUNT_OR_DUPLICATE")
    if len(present) != 91 or len(missing) != 5: failures.append("MANIFEST_PRESENT_MISSING_COUNT")
    package_root = args.package_root.resolve()
    for key, row in present.items():
        path = package_root / str(row["package_filename"]).replace("/", "\\")
        if not path.is_file():
            failures.append("PACKAGE_VIDEO_MISSING:" + str(key))
            continue
        if path.stat().st_size != row.get("bytes") or sha(path) != row.get("sha256"):
            failures.append("PACKAGE_VIDEO_HASH:" + str(key))

    label_rows = labels.get("rows", [])
    label_by_id = {row.get("blinded_video_id"): row for row in label_rows}
    if len(label_rows) != 91 or len(label_by_id) != 91: failures.append("LABEL_ROW_COUNT_OR_DUPLICATE")
    if set(label_by_id) != set(present): failures.append("LABEL_ID_SET")
    for key, row in label_by_id.items():
        if row.get("primary_label") not in LABELS: failures.append("LABEL_VOCABULARY:" + str(key))

    private_rows = private.get("rows", [])
    private_by_id = {row.get("blinded_video_id"): row for row in private_rows}
    if private.get("status") != "SEALED_PRIVATE_NOT_FOR_REVIEWER": failures.append("PRIVATE_MAP_STATUS")
    if set(private_by_id) != set(public_by_id): failures.append("PRIVATE_PUBLIC_ID_SET")
    branch_by_id = {row.get("branch_id"): row for row in index.get("rows", [])}
    joined: list[dict[str, Any]] = []
    for key, label in sorted(label_by_id.items()):
        private_row = private_by_id.get(key)
        if private_row is None: continue
        source = private_row.get("frozen_sample_row", {})
        branch_id = source.get("branch_id")
        branch = branch_by_id.get(branch_id)
        if branch is None:
            failures.append("BRANCH_JOIN:" + str(key))
            continue
        validation = branch.get("validation", {})
        joined.append({
            "blinded_video_id": key,
            "primary_label": label.get("primary_label"),
            "confidence": label.get("confidence"),
            "visual_note": label.get("visual_note", label.get("note")),
            "model_family": source.get("model_family"),
            "suite": source.get("suite"),
            "condition": source.get("condition"),
            "dose": source.get("dose"),
            "branch_id": branch_id,
            "canonical_parent_key": source.get("canonical_parent_key"),
            "automatic_status": branch.get("status"),
            "automatic_physical_class": validation.get("physical_class"),
            "automatic_v_phys_label": validation.get("v_phys_label"),
        })
    if len(joined) != 91: failures.append("JOINED_ROW_COUNT:" + str(len(joined)))
    for key in missing:
        missing[key]["availability"] = "FROZEN_SAMPLE_VIDEO_MISSING_NO_LABEL"

    label_counts = Counter(row["primary_label"] for row in joined)
    model_counts = {model: dict(sorted(Counter(row["primary_label"] for row in joined if row["model_family"] == model).items())) for model in sorted({row["model_family"] for row in joined})}
    auto_contact = [row for row in joined if row["automatic_physical_class"] == "GRIPPER_CONTACT_LOSS"]
    direct_events = {label: sum(row["primary_label"] == label for row in joined) for label in ["PREMATURE_APERTURE", "CONTACT_LOSS", "PREMATURE_RELEASE_OR_DROP", "OBJECT_DISPLACEMENT"]}
    stats_summary = {model: {"complete_t3_t10_pair_count": value.get("complete_t3_t10_pair_count"), "complete_t3_t5_t10_triplet_count": value.get("complete_t3_t5_t10_triplet_count"), "triplet_patterns": value.get("complete_t3_t5_t10_patterns")} for model, value in g3.get("model_summary", {}).items()}
    source_files = [args.manifest, args.package_seal, args.labels, args.label_seal, args.private_map, args.branch_index, args.g3_stats]
    authority = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in source_files}
    report = {
        "schema": "STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1",
        "status": "STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_COMPLETE_CONTINUE_TO_AC5" if not failures else "HOLD_AC4_RECONCILIATION_VALIDATION_FAILURE",
        "reviewer": {"type": "AI_SECONDARY", "human_review_gate_satisfied": False, "labels_sealed_before_unblind": True},
        "counts": {"frozen_slots": 96, "present_videos": 91, "missing_frozen_videos": 5, "label_rows": len(label_rows), "joined_rows": len(joined)},
        "label_distribution": dict(sorted(label_counts.items())),
        "labels_by_model": model_counts,
        "labels_by_model_condition_dose": grouped_counts(joined, ("model_family", "condition", "dose")),
        "labels_by_auto_physical_class": cross(joined, "automatic_physical_class", "primary_label"),
        "labels_by_auto_v_phys_label": cross(joined, "automatic_v_phys_label", "primary_label"),
        "automatic_contact_loss_audit": {"automatic_rows": len(auto_contact), "labels": dict(sorted(Counter(row["primary_label"] for row in auto_contact).items())), "not_identifiable_or_ambiguous": sum(row["primary_label"] in {"NOT_IDENTIFIABLE", "AMBIGUOUS_OR_OCCLUDED"} for row in auto_contact), "stable_grasp": sum(row["primary_label"] == "STABLE_GRASP" for row in auto_contact)},
        "direct_visual_event_counts": direct_events,
        "missing_slots": sorted(missing),
        "same_parent_g3r1_static_reference": stats_summary,
        "joined_rows": joined,
        "authority": authority,
        "scientific_firewall": {"new_model_inference": 0, "new_env_step": 0, "new_open_intervention": 0, "new_pgd": 0, "new_protected_reads": 0, "automatic_labels_rewritten": 0, "denominator_changed": 0, "replacement_or_top_up": 0},
        "claim_boundary": "AI-secondary visual endpoint-validity reconciliation only; human review gate remains unsatisfied; automatic endpoint labels are never rewritten; no new execution or scientific promotion.",
        "next_legal_action": "CONTINUE_TO_AC5_STATIC_SYNTHESIS" if not failures else "STOP_AND_REPAIR_AC4_RECONCILIATION",
        "validation_failures": failures,
    }
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise SystemExit("AC4_OUTPUT_NOT_EMPTY:" + str(output))
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "STAGE_AC_AC4_AI_SECONDARY_RECONCILIATION_V1.json"
    dump(report_path, report)
    payload = {"schema": report["schema"], "status": report["status"], "report_sha256": sha(report_path), "report_bytes": report_path.stat().st_size, "counts": report["counts"], "scientific_firewall": report["scientific_firewall"]}
    root = {"schema": "STAGE_AC_AC4_AI_SECONDARY_ROOT_SEAL_V1", "status": report["status"], "root_payload": payload, "root_payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "artifacts": {"report": {"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": sha(report_path)}}, "next_legal_action": report["next_legal_action"]}
    dump(output / "STAGE_AC_AC4_AI_SECONDARY_ROOT_SEAL_V1.json", root)
    print(json.dumps({"status": report["status"], "failures": failures, "label_distribution": report["label_distribution"], "automatic_contact_loss_audit": report["automatic_contact_loss_audit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
