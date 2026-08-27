#!/usr/bin/env python3
"""CPU-only unblind reconciliation for the sealed Z3-D AI-secondary labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_LABEL_FILES = {
    "labels_json": (67648, "dfc3484db22c342f598c8e248388615cfdd28d9acb4c9b139501b5f1c3f600cd"),
    "labels_csv": (26749, "ff1411afec9de439a1134fb8f228c77609a2cfb588feccdaae897d4f3923d12c"),
    "label_seal": (721, "1c0d7beddaa6153c746ee541604ea861698289379278036bf634909b4e2878cd"),
}
PRIMARY_LABELS = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}
ARMS = {
    "CLEAN_BRANCH_CRITICAL": ("CLEAN_REFERENCE", 0),
    "COMMAND_OPEN_T3_CRITICAL": ("CRITICAL_OPEN_PRIMARY", 3),
    "COMMAND_OPEN_T5_CRITICAL": ("CRITICAL_OPEN_PRIMARY", 5),
    "COMMAND_OPEN_T10_CRITICAL": ("CRITICAL_OPEN_PRIMARY", 10),
    "COMMAND_OPEN_T5_NONCRITICAL_CONTROL": ("NONCRITICAL_T5_CONTROL", 5),
}
FORBIDDEN_COUNTERS = ("protected_reads", "eval160_reads", "pgd_calls", "attack_outcome_reads")
AI_LABELS = {
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
}


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str, failures: List[str]) -> None:
    if not condition:
        failures.append(message)


def rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bool_text(value: str) -> bool:
    return value.strip().lower() == "true"


def rate(numerator: int, denominator: int) -> Any:
    return numerator / denominator if denominator else None


def sorted_counts(values: Iterable[Any]) -> Dict[str, int]:
    return dict(sorted(Counter("<NULL>" if value is None else str(value) for value in values).items()))


def telemetry(branch: Dict[str, Any]) -> Dict[str, Any]:
    rows = branch.get("rows", [])
    valid = [row for row in rows if row.get("post_contact_telemetry_valid") is True]
    contacts = [row.get("post_object_gripper_contact") for row in valid]
    support = [row.get("post_object_support_contact") for row in valid]
    distances = [float(row["post_object_eef_distance_m"]) for row in valid if finite(row.get("post_object_eef_distance_m"))]
    contact_true = sum(value is True for value in contacts)
    contact_false = sum(value is False for value in contacts)
    support_true = sum(value is True for value in support)
    support_false = sum(value is False for value in support)
    return {
        "rows": len(rows),
        "telemetry_valid_rows": len(valid),
        "telemetry_invalid_rows": len(rows) - len(valid),
        "contact_observed_rows": contact_true + contact_false,
        "contact_true": contact_true,
        "contact_false": contact_false,
        "contact_unknown": len(contacts) - contact_true - contact_false,
        "contact_true_rate_observed": rate(contact_true, contact_true + contact_false),
        "support_observed_rows": support_true + support_false,
        "support_true": support_true,
        "support_false": support_false,
        "support_unknown": len(support) - support_true - support_false,
        "support_true_rate_observed": rate(support_true, support_true + support_false),
        "distance_observed_rows": len(distances),
        "distance_max_m": max(distances) if distances else None,
        "distance_mean_m": statistics.mean(distances) if distances else None,
        "post_telemetry_reasons": sorted_counts(row.get("post_telemetry_reason") for row in rows),
    }


def merge_telemetry(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = (
        "rows", "telemetry_valid_rows", "telemetry_invalid_rows", "contact_observed_rows",
        "contact_true", "contact_false", "contact_unknown", "support_observed_rows",
        "support_true", "support_false", "support_unknown", "distance_observed_rows",
    )
    result = {field: sum(int(record["telemetry"][field]) for record in records) for field in fields}
    distances = [record["telemetry"]["distance_max_m"] for record in records if record["telemetry"]["distance_max_m"] is not None]
    result.update({
        "branch_count": len(records),
        "parent_count": len({record["canonical_parent_key"] for record in records}),
        "contact_true_rate_observed": rate(result["contact_true"], result["contact_observed_rows"]),
        "support_true_rate_observed": rate(result["support_true"], result["support_observed_rows"]),
        "distance_max_m": max(distances) if distances else None,
        "distance_mean_of_branch_maxima_m": statistics.mean(distances) if distances else None,
        "physical_class_counts": sorted_counts(record["auto_physical_class"] for record in records),
        "auto_v_phys_label_counts": sorted_counts(record["auto_v_phys_label"] for record in records),
    })
    return result


def group_summary(records: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in keys)].append(record)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item = {name: value for name, value in zip(keys, key)}
        item.update(merge_telemetry(values))
        output.append(item)
    return output


def cross_tab(records: List[Dict[str, Any]], left: str) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        result[str(record[left])][record["ai_label"]] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--label-seal", required=True)
    parser.add_argument("--r1-map", required=True)
    parser.add_argument("--hidden-map", required=True)
    parser.add_argument("--branch-index", required=True)
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--z3c-root-seal", required=True)
    parser.add_argument("--ingestion-receipt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = rooted(root, str(args.output_dir))
    failures: List[str] = []

    paths = {
        "labels_json": rooted(root, args.labels_json),
        "labels_csv": rooted(root, args.labels_csv),
        "label_seal": rooted(root, args.label_seal),
        "r1_map": rooted(root, args.r1_map),
        "hidden_map": rooted(root, args.hidden_map),
        "branch_index": rooted(root, args.branch_index),
        "terminal": rooted(root, args.terminal),
        "z3c_root_seal": rooted(root, args.z3c_root_seal),
        "ingestion_receipt": rooted(root, args.ingestion_receipt),
    }
    for name, path in paths.items():
        require(path.is_file(), "MISSING_INPUT:" + name + ":" + str(path), failures)
    if failures:
        raise SystemExit(json.dumps({"status": "HOLD_INPUT_MISSING", "failures": failures}, sort_keys=True))

    for name, (expected_bytes, expected_sha) in EXPECTED_LABEL_FILES.items():
        path = paths[name]
        require(path.stat().st_size == expected_bytes, "LABEL_BYTES:" + name, failures)
        require(sha(path) == expected_sha, "LABEL_SHA:" + name, failures)

    labels_doc = load(paths["labels_json"])
    label_seal = load(paths["label_seal"])
    ingestion = load(paths["ingestion_receipt"])
    with paths["labels_csv"].open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = labels_doc.get("rows", [])
    require(len(json_rows) == 120 and len(csv_rows) == 120, "LABEL_ROW_COUNT", failures)
    require(label_seal.get("row_count") == 120, "SEAL_ROW_COUNT", failures)
    require(label_seal.get("label_json_sha256") == sha(paths["labels_json"]), "SEAL_JSON_BINDING", failures)
    require(label_seal.get("label_csv_sha256") == sha(paths["labels_csv"]), "SEAL_CSV_BINDING", failures)
    require(label_seal.get("human_review_gate_satisfied") is False, "SEAL_HUMAN_GATE", failures)
    require(label_seal.get("unblind_permitted_for_ai_secondary_reconciliation") is True, "SEAL_UNBLIND_PERMISSION", failures)
    require(label_seal.get("z4_authorized") is False, "SEAL_Z4_FLAG", failures)
    require(ingestion.get("status") == "STAGE_Z_Z3D_AI_SECONDARY_LABEL_INGESTION_PASS", "INGESTION_STATUS", failures)
    require(ingestion.get("governance", {}).get("hidden_mapping_read_before_ingestion_pass") is False, "INGESTION_ORDER", failures)

    json_by_id = {row.get("blinded_video_id"): row for row in json_rows}
    csv_by_id = {row.get("blinded_video_id"): row for row in csv_rows}
    require(len(json_by_id) == len(json_rows) == 120, "JSON_DUPLICATE_IDS", failures)
    require(len(csv_by_id) == len(csv_rows) == 120, "CSV_DUPLICATE_IDS", failures)
    for blinded_id, row in json_by_id.items():
        other = csv_by_id.get(blinded_id)
        require(other is not None, "CSV_MISSING_LABEL:" + str(blinded_id), failures)
        if other is None:
            continue
        for field in ("manual_audit_id", "label", "blinded_note", "reviewer_confidence"):
            require(str(row.get(field)) == str(other.get(field)), "LABEL_FIELD_MISMATCH:" + field + ":" + str(blinded_id), failures)
        for field in ("stable_grasp_maintained", "premature_aperture", "slip_or_contact_loss", "premature_release_or_drop", "object_displacement_consistent_with_loss"):
            require(row.get(field) is bool_text(other.get(field, "")), "LABEL_BOOL_MISMATCH:" + field + ":" + str(blinded_id), failures)
        require(row.get("label") in AI_LABELS, "LABEL_VOCABULARY:" + str(blinded_id), failures)

    r1 = load(paths["r1_map"])
    hidden = load(paths["hidden_map"])
    index = load(paths["branch_index"])
    terminal = load(paths["terminal"])
    z3c_seal = load(paths["z3c_root_seal"])
    require(r1.get("source_videos_byte_identical") is True, "R1_SOURCE_BYTE_BINDING", failures)
    r1_rows = r1.get("rows", [])
    hidden_rows = hidden.get("rows", [])
    index_rows = index.get("rows", [])
    parent_rows = terminal.get("physical_results", {}).get("parent_rows", [])
    require(len(r1_rows) == 120, "R1_ROW_COUNT", failures)
    require(len(hidden_rows) == 120, "HIDDEN_ROW_COUNT", failures)
    require(len(index_rows) == 460, "BRANCH_INDEX_COUNT", failures)
    require(len(parent_rows) == 92, "PARENT_ROW_COUNT", failures)
    require(sha(paths["branch_index"]) == z3c_seal.get("branch_index", {}).get("sha256"), "Z3C_BRANCH_INDEX_SEAL", failures)
    require(sha(paths["terminal"]) == z3c_seal.get("terminal_synthesis", {}).get("sha256"), "Z3C_TERMINAL_SEAL", failures)
    require(z3c_seal.get("status") == "PASS_Z3C_FIXED_MATRIX_COMPLETE", "Z3C_ROOT_STATUS", failures)

    r1_by_derivative = {row.get("derivative_blinded_video_id"): row for row in r1_rows}
    hidden_by_source = {row.get("blinded_video_id"): row for row in hidden_rows}
    index_by_branch = {row.get("branch_id"): row for row in index_rows}
    parents = {(row.get("model_family"), row.get("suite"), row.get("canonical_parent_key")): row for row in parent_rows}
    require(len(r1_by_derivative) == 120, "R1_DUPLICATE_DERIVATIVE_IDS", failures)
    require(len(hidden_by_source) == 120, "HIDDEN_DUPLICATE_SOURCE_IDS", failures)
    require(len(index_by_branch) == 460, "INDEX_DUPLICATE_BRANCH_IDS", failures)
    require(len(parents) == 92, "PARENT_DUPLICATE_KEYS", failures)

    branches: Dict[str, Dict[str, Any]] = {}
    source_receipt_counters = Counter()
    for job in index_rows:
        branch_path = rooted(root, job["receipt_path"])
        require(branch_path.is_file(), "MISSING_RECEIPT:" + str(job.get("branch_id")), failures)
        if not branch_path.is_file():
            continue
        require(branch_path.stat().st_size == job.get("bytes"), "RECEIPT_BYTES:" + str(job.get("branch_id")), failures)
        require(sha(branch_path) == job.get("receipt_sha256"), "RECEIPT_SHA:" + str(job.get("branch_id")), failures)
        try:
            branch = load(branch_path)
        except Exception as exc:
            failures.append("RECEIPT_JSON:" + str(job.get("branch_id")) + ":" + str(exc))
            continue
        branches[job["branch_id"]] = branch
        for field in ("branch_id", "model_family", "suite", "canonical_parent_key", "arm", "duration"):
            require(branch.get(field) == job.get(field), "RECEIPT_FIELD:" + field + ":" + str(job.get("branch_id")), failures)
        require(branch.get("status") == "PASS", "RECEIPT_STATUS:" + str(job.get("branch_id")), failures)
        require(branch.get("model_inference") is False, "RECEIPT_MODEL_INFERENCE:" + str(job.get("branch_id")), failures)
        require(branch.get("state_restore_exact") is True, "RECEIPT_STATE_RESTORE:" + str(job.get("branch_id")), failures)
        require(branch.get("causal_input_binding_pass") is True, "RECEIPT_CAUSAL_BINDING:" + str(job.get("branch_id")), failures)
        counters = branch.get("runtime_counters", {})
        for counter in FORBIDDEN_COUNTERS:
            require(int(counters.get(counter, 0)) == 0, "RECEIPT_FORBIDDEN_COUNTER:" + counter + ":" + str(job.get("branch_id")), failures)
        for counter, value in counters.items():
            try:
                source_receipt_counters[counter] += int(value or 0)
            except (TypeError, ValueError):
                failures.append("RECEIPT_COUNTER_VALUE:" + str(job.get("branch_id")) + ":" + counter)

    records: List[Dict[str, Any]] = []
    for label in json_rows:
        derivative_id = label["blinded_video_id"]
        r1_row = r1_by_derivative.get(derivative_id)
        require(r1_row is not None, "R1_MISSING_DERIVATIVE:" + derivative_id, failures)
        if r1_row is None:
            continue
        source_id = r1_row["source_blinded_video_id"]
        hidden_row = hidden_by_source.get(source_id)
        require(hidden_row is not None, "HIDDEN_MISSING_SOURCE:" + source_id, failures)
        if hidden_row is None:
            continue
        require(r1_row.get("source_sha256") == hidden_row.get("source_sha256"), "SOURCE_SHA_BINDING:" + source_id, failures)
        require(r1_row.get("source_bytes") == hidden_row.get("source_bytes"), "SOURCE_BYTES_BINDING:" + source_id, failures)
        source_path = rooted(root, hidden_row["source_path"])
        package_path = rooted(root, hidden_row["package_path"])
        for path, expected_bytes, expected_sha, tag in ((source_path, hidden_row.get("source_bytes"), hidden_row.get("source_sha256"), "SOURCE"), (package_path, hidden_row.get("package_bytes"), hidden_row.get("package_sha256"), "PACKAGE")):
            require(path.is_file(), tag + "_VIDEO_MISSING:" + source_id, failures)
            if path.is_file():
                require(path.stat().st_size == expected_bytes, tag + "_VIDEO_BYTES:" + source_id, failures)
                require(sha(path) == expected_sha, tag + "_VIDEO_SHA:" + source_id, failures)
        branch_id = hidden_row["branch_id"]
        job = index_by_branch.get(branch_id)
        branch = branches.get(branch_id)
        require(job is not None and branch is not None, "BRANCH_JOIN:" + source_id, failures)
        if job is None or branch is None:
            continue
        model = hidden_row["model_family"]
        suite = hidden_row["suite"]
        parent_key = hidden_row["canonical_parent_key"]
        arm = hidden_row["arm"]
        role, dose = ARMS.get(arm, ("UNKNOWN", hidden_row.get("duration")))
        require(job.get("model_family") == model and job.get("suite") == suite and job.get("canonical_parent_key") == parent_key and job.get("arm") == arm and job.get("duration") == dose, "HIDDEN_INDEX_BINDING:" + branch_id, failures)
        parent = parents.get((model, suite, parent_key))
        require(parent is not None, "PARENT_JOIN:" + branch_id, failures)
        if parent is None:
            continue
        expected = parent.get("clean_physical_class") if role == "CLEAN_REFERENCE" else (parent.get("noncritical_t5_control", {}).get("physical_class") if role == "NONCRITICAL_T5_CONTROL" else parent.get("critical", {}).get(str(dose), {}).get("physical_class"))
        expected_vphys = None if role == "CLEAN_REFERENCE" else ("NOT_APPLICABLE_NONCRITICAL_CONTROL" if role == "NONCRITICAL_T5_CONTROL" else parent.get("critical", {}).get(str(dose), {}).get("v_phys_label"))
        require(branch.get("physical_class") == expected, "PHYSICAL_CLASS_BINDING:" + branch_id, failures)
        if role == "CRITICAL_OPEN_PRIMARY":
            require(branch.get("v_phys_label") == expected_vphys, "VPHYS_BINDING:" + branch_id, failures)
        auto_valid = role == "CRITICAL_OPEN_PRIMARY" and expected_vphys in PRIMARY_LABELS
        auto_abstain = role == "CRITICAL_OPEN_PRIMARY" and isinstance(expected_vphys, str) and expected_vphys.startswith("CONTROL_")
        records.append({
            "blinded_video_id": derivative_id,
            "source_blinded_video_id": source_id,
            "manual_audit_id": label.get("manual_audit_id"),
            "ai_label": label.get("label"),
            "stable_grasp_maintained": label.get("stable_grasp_maintained"),
            "premature_aperture": label.get("premature_aperture"),
            "slip_or_contact_loss": label.get("slip_or_contact_loss"),
            "premature_release_or_drop": label.get("premature_release_or_drop"),
            "object_displacement_consistent_with_loss": label.get("object_displacement_consistent_with_loss"),
            "reviewer_confidence": label.get("reviewer_confidence"),
            "blinded_note": label.get("blinded_note"),
            "model_family": model,
            "suite": suite,
            "canonical_parent_key": parent_key,
            "branch_id": branch_id,
            "arm": arm,
            "role": role,
            "dose": dose,
            "duration": dose,
            "auto_physical_class": branch.get("physical_class"),
            "auto_v_phys_label": expected_vphys,
            "auto_valid_primary": auto_valid,
            "auto_abstain_primary": auto_abstain,
            "telemetry": telemetry(branch),
        })

    require(len(records) == 120, "UNBLIND_ROW_COUNT:" + str(len(records)), failures)
    records.sort(key=lambda row: row["blinded_video_id"])

    ai_by_model = {model: sorted_counts(row["ai_label"] for row in records if row["model_family"] == model) for model in sorted({row["model_family"] for row in records})}
    ai_by_suite = {suite: sorted_counts(row["ai_label"] for row in records if row["suite"] == suite) for suite in sorted({row["suite"] for row in records})}
    ai_by_model_arm_dose = group_summary(records, ("model_family", "arm", "dose"))
    not_identifiable = [row for row in records if row["ai_label"] == "NOT_IDENTIFIABLE"]
    not_identifiable_by_location = group_summary(not_identifiable, ("model_family", "arm", "dose"))
    notable_locations = [{key: row[key] for key in ("blinded_video_id", "model_family", "suite", "canonical_parent_key", "branch_id", "arm", "dose", "ai_label", "reviewer_confidence", "auto_physical_class", "auto_v_phys_label")} for row in records if row["ai_label"] in {"STABLE_GRASP", "OBJECT_DISPLACEMENT", "AMBIGUOUS_OR_OCCLUDED"}]

    parent_intersection: Dict[str, Any] = {}
    for model in sorted({row["model_family"] for row in parent_rows}):
        model_parents = [row for row in parent_rows if row.get("model_family") == model]
        dose_valid_counts = {}
        pattern_counts = Counter()
        complete_rows = []
        for parent in model_parents:
            labels = [parent.get("critical", {}).get(str(dose), {}).get("v_phys_label") for dose in (3, 5, 10)]
            for dose, value in zip((3, 5, 10), labels):
                if value in PRIMARY_LABELS:
                    dose_valid_counts[str(dose)] = dose_valid_counts.get(str(dose), 0) + 1
            if all(value in PRIMARY_LABELS for value in labels):
                pattern = "".join("1" if value == "V_PHYS" else "0" for value in labels)
                pattern_counts[pattern] += 1
                complete_rows.append({"canonical_parent_key": parent["canonical_parent_key"], "suite": parent["suite"], "pattern_3_5_10": pattern, "labels_3_5_10": labels})
        parent_intersection[model] = {"total_parents": len(model_parents), "dose_valid_parent_counts": dict(sorted(dose_valid_counts.items())), "complete_all_dose_parents": len(complete_rows), "complete_all_dose_pattern_counts": dict(sorted(pattern_counts.items())), "complete_all_dose_rows": complete_rows}

    auto_vs_ai = {
        "auto_physical_class_by_ai_label": cross_tab(records, "auto_physical_class"),
        "auto_v_phys_label_by_ai_label": cross_tab(records, "auto_v_phys_label"),
        "notable_counts": {
            "auto_gripper_contact_loss_ai_stable_grasp": sum(row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" and row["ai_label"] == "STABLE_GRASP" for row in records),
            "auto_gripper_contact_loss_ai_not_identifiable": sum(row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" and row["ai_label"] == "NOT_IDENTIFIABLE" for row in records),
            "auto_gripper_contact_loss_ai_object_displacement": sum(row["auto_physical_class"] == "GRIPPER_CONTACT_LOSS" and row["ai_label"] == "OBJECT_DISPLACEMENT" for row in records),
            "auto_abstain_ai_object_displacement": sum(row["auto_abstain_primary"] and row["ai_label"] == "OBJECT_DISPLACEMENT" for row in records),
            "auto_vphys_ai_object_displacement": sum(row["auto_v_phys_label"] == "V_PHYS" and row["ai_label"] == "OBJECT_DISPLACEMENT" for row in records),
            "auto_vphys_ai_not_identifiable": sum(row["auto_v_phys_label"] == "V_PHYS" and row["ai_label"] == "NOT_IDENTIFIABLE" for row in records),
        },
        "interpretation": "NOT_IDENTIFIABLE is unresolved visual evidence; CONTROL_*_ABSTAIN remains abstention and is never relabeled as V_PHYS.",
    }

    static = {
        "schema": "STAGE_Z_Z3D_STATIC_DIAGNOSTIC_SYNTHESIS_V1",
        "status": "STAGE_Z_Z3D_AI_SECONDARY_RECONCILIATION_COMPLETE_STOP_FOR_PI" if not failures else "HOLD_Z3D_RECONCILIATION_VALIDATION_FAILURE",
        "authority": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in paths.items()
        },
        "label_distribution": sorted_counts(row["ai_label"] for row in records),
        "ai_labels_by_model": ai_by_model,
        "ai_labels_by_suite": ai_by_suite,
        "ai_labels_by_model_arm_dose": ai_by_model_arm_dose,
        "not_identifiable_by_model_arm_dose": not_identifiable_by_location,
        "notable_label_locations": notable_locations,
        "automatic_vs_ai": auto_vs_ai,
        "same_parent_t3_t5_t10": parent_intersection,
        "telemetry_by_model_role_dose": group_summary(records, ("model_family", "role", "dose")),
        "telemetry_by_model_suite_role_dose": group_summary(records, ("model_family", "suite", "role", "dose")),
        "source_z3c_counters": dict(sorted(source_receipt_counters.items())),
        "interpretation_guardrails": {
            "not_identifiable_is_unresolved": True,
            "control_abstain_not_relabelled": True,
            "ai_secondary_not_human_manual_audit": True,
            "no_cross_model_replication_claim": True,
            "z4_not_authorized": True,
        },
    }

    reconciliation = {
        "schema": "STAGE_Z_Z3D_AI_SECONDARY_UNBLIND_RECONCILIATION_V1",
        "status": static["status"],
        "authority": static["authority"],
        "execution": {"label_rows": len(json_rows), "joined_rows": len(records), "branch_receipts_read": len(branches), "z3c_parent_rows_read": len(parent_rows)},
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_identities": 0,
            "branch_reexecution": 0,
            "pgd": 0,
            "eval160_reads": 0,
            "protected_reads": 0,
            "f1": 0,
            "bridge": 0,
            "human_review_gate_satisfied": False,
            "z4_authorized": False,
        },
        "rows": records,
        "static_diagnostics": static,
        "claim_boundary": "AI-secondary unblind reconciliation of sealed Z3-C presentation videos only; no human-review completion, no V_phys relabeling, no new simulator execution, no Z4 or Paper promotion.",
        "next_legal_action": "STOP_FOR_PI_NO_Z4_NO_BRIDGE_NO_NEW_SIMULATOR_EXECUTION",
        "validation_failures": failures,
    }

    if output.exists() and any(output.iterdir()):
        raise SystemExit(json.dumps({"status": "HOLD_OUTPUT_EXISTS", "output": str(output)}, sort_keys=True))
    output.mkdir(parents=True, exist_ok=True)
    reconciliation_path = output / "STAGE_Z_Z3D_AI_SECONDARY_UNBLIND_RECONCILIATION_V1.json"
    static_path = output / "STAGE_Z_Z3D_STATIC_DIAGNOSTIC_SYNTHESIS_V1.json"
    csv_path = output / "STAGE_Z_Z3D_AI_SECONDARY_UNBLIND_RECONCILIATION_V1.csv"
    dump(reconciliation_path, reconciliation)
    dump(static_path, static)
    csv_fields = [
        "blinded_video_id", "source_blinded_video_id", "manual_audit_id", "ai_label", "reviewer_confidence",
        "model_family", "suite", "canonical_parent_key", "branch_id", "arm", "role", "dose", "duration",
        "auto_physical_class", "auto_v_phys_label", "auto_valid_primary", "auto_abstain_primary",
        "telemetry_rows", "telemetry_valid_rows", "contact_true", "contact_false", "support_true", "support_false",
        "distance_max_m", "distance_mean_m", "blinded_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in records:
            flat = {field: row.get(field) for field in csv_fields}
            flat.update({
                "telemetry_rows": row["telemetry"]["rows"],
                "telemetry_valid_rows": row["telemetry"]["telemetry_valid_rows"],
                "contact_true": row["telemetry"]["contact_true"],
                "contact_false": row["telemetry"]["contact_false"],
                "support_true": row["telemetry"]["support_true"],
                "support_false": row["telemetry"]["support_false"],
                "distance_max_m": row["telemetry"]["distance_max_m"],
                "distance_mean_m": row["telemetry"]["distance_mean_m"],
            })
            writer.writerow(flat)

    root_seal = {
        "schema": "STAGE_Z_Z3D_AI_SECONDARY_ROOT_SEAL_V1",
        "status": static["status"],
        "analysis_script": {"path": "scripts/stage_z/reconcile_stage_z_z3d_ai_secondary.py", "sha256": sha(Path(__file__))},
        "artifacts": {
            "reconciliation_json": {"path": str(reconciliation_path.relative_to(root)).replace("\\", "/"), "bytes": reconciliation_path.stat().st_size, "sha256": sha(reconciliation_path)},
            "reconciliation_csv": {"path": str(csv_path.relative_to(root)).replace("\\", "/"), "bytes": csv_path.stat().st_size, "sha256": sha(csv_path)},
            "static_diagnostic_synthesis": {"path": str(static_path.relative_to(root)).replace("\\", "/"), "bytes": static_path.stat().st_size, "sha256": sha(static_path)},
        },
        "source_bindings": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in paths.items()},
        "execution": reconciliation["execution"],
        "scientific_firewall": reconciliation["scientific_firewall"],
        "claim_boundary": reconciliation["claim_boundary"],
        "next_legal_action": reconciliation["next_legal_action"],
        "validation_failures": failures,
    }
    seal_path = output / "STAGE_Z_Z3D_AI_SECONDARY_ROOT_SEAL_V1.json"
    dump(seal_path, root_seal)
    print(json.dumps({"status": root_seal["status"], "joined_rows": len(records), "receipts": len(branches), "output": str(output), "root_seal": sha(seal_path), "failures": failures[:10]}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
