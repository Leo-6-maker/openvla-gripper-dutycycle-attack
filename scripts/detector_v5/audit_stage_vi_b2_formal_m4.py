"""Aggregate and audit only structurally valid Stage VI-B2 formal parents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
ARMS = {"CONTROL", "T3", "T5", "T10"}
BINARY = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    scheduler = args.scheduler_root.resolve()
    authority = load(args.authority.resolve())
    plan_manifest = load(args.plan_root.resolve() / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    status = load(scheduler / "SCHEDULER_STATUS.json")
    queue = load(scheduler / "FORMAL_PARENT_QUEUE.json")
    errors: list[str] = []
    source_commit = authority.get("source_binding", {}).get("runtime_commit")
    source_tree = authority.get("source_binding", {}).get("runtime_tree")
    if authority.get("status") != "PASS" or authority.get("formal_m4_authorized") is not True or authority.get("protected_counters") != COUNTERS:
        errors.append("AUTHORITY")
    if status.get("status") != "PASS_STAGE_VI_B2_FORMAL_M4_SCHEDULER" or status.get("protected_counters") != COUNTERS or queue.get("status") != "PASS":
        errors.append("SCHEDULER_STATUS")
    parents = [dict(row) for row in plan_manifest.get("parents", []) if isinstance(row, Mapping)]
    if len(parents) != 16 or plan_manifest.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN":
        errors.append("PLAN")
    parent_results: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    all_branches: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    for planned in parents:
        key = str(planned["canonical_parent_key"])
        root = scheduler / "parents" / f"{int(planned['ordinal']):02d}_{key.replace('/', '__')}"
        result_path, audit_path = root / "PARENT_RESULT.json", root / "M4_INDEPENDENT_AUDIT.json"
        result, audit = load(result_path), load(audit_path)
        if result.get("status") != "PASS" or result.get("schema") != "STAGE_V_M4_PARENT_RESULT_V1" or result.get("canonical_parent_key") != key or result.get("source_commit") != source_commit or result.get("source_tree") != source_tree or result.get("probe_count") != 24 or result.get("branch_count") != 96 or result.get("treatment_label_count") != 72 or result.get("selection_outcomes_read") is not False or result.get("label_status") != "VALID" or result.get("protected_counters") != COUNTERS:
            errors.append(f"RESULT:{key}")
        if audit.get("status") != "PASS_M4_PARENT_INDEPENDENT" or audit.get("canonical_parent_key") != key or audit.get("source_commit") != source_commit or audit.get("source_tree") != source_tree or audit.get("branch_count") != 96 or audit.get("label_count") != 72 or audit.get("protected_counters") != COUNTERS:
            errors.append(f"AUDIT:{key}")
        branches = jsonl(root / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl")
        labels = jsonl(root / "M4_V_PHYS_LABELS_V1.jsonl")
        observations = jsonl(root / "M4_TREATMENT_OBSERVATIONS_V1.jsonl")
        if len(branches) != 96 or len(labels) != 72 or len(observations) != 72:
            errors.append(f"FILES:{key}")
        if key in seen_parents:
            errors.append(f"DUPLICATE_PARENT:{key}")
        seen_parents.add(key)
        for branch in branches:
            if branch.get("canonical_parent_key") != key or branch.get("arm") not in ARMS or branch.get("protected_counters") != COUNTERS:
                errors.append(f"BRANCH:{key}")
        for label in labels:
            if label.get("canonical_parent_key") != key or label.get("dose") not in {"T3", "T5", "T10"} or label.get("schema") != "STAGE_V_M4_V_PHYS_LABEL_V1" or label.get("protected_counters") != COUNTERS:
                errors.append(f"LABEL:{key}")
            if label.get("binary_label_consumable") is True and label.get("label_class") not in BINARY:
                errors.append(f"BINARY_LABEL:{key}")
        all_branches.extend(branches)
        all_labels.extend(labels)
        all_observations.extend(observations)
        parent_results.append({"ordinal": planned["ordinal"], "canonical_parent_key": key, "parent_result_sha256": sha(result_path), "audit_sha256": sha(audit_path), "binary_label_count": sum(row.get("binary_label_consumable") is True for row in labels), "abstention_label_count": sum(row.get("binary_label_consumable") is not True for row in labels), "censored_label_count": sum(row.get("censoring_class") != "NONE" for row in labels)})
    if len(seen_parents) != 16 or len(all_branches) != 1536 or len(all_labels) != 1152 or len(all_observations) != 1152:
        errors.append("AGGREGATE_ACCOUNTING")
    consumable = [row for row in all_labels if row.get("binary_label_consumable") is True]
    censoring = [{"canonical_parent_key": row.get("canonical_parent_key"), "probe_id": row.get("probe_id"), "dose": row.get("dose"), "label_id": row.get("label_id"), "label_class": row.get("label_class"), "binary_label_consumable": row.get("binary_label_consumable"), "censoring_class": row.get("censoring_class"), "treatment_compliant": row.get("treatment_compliant"), "protected_counters": row.get("protected_counters")} for row in all_labels]
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError(f"REFUSE_OVERWRITE:{output}")
    output.mkdir(parents=True)
    write_jsonl(output / "B2_ALL_LABELS.jsonl", all_labels)
    write_jsonl(output / "B2_VPHYS_CONSUMABLE_LABELS.jsonl", consumable)
    write_jsonl(output / "B2_CENSORING_MAP.jsonl", censoring)
    write_jsonl(output / "B2_ALL_BRANCHES.jsonl", all_branches)
    write_jsonl(output / "B2_ALL_OBSERVATIONS.jsonl", all_observations)
    aggregate = {"schema": "STAGE_VI_B2_FORMAL_M4_AGGREGATE_V1", "status": "PASS_STAGE_VI_B2_FORMAL_M4", "parent_count": len(seen_parents), "branch_count": len(all_branches), "treatment_label_count": len(all_labels), "consumable_binary_label_count": len(consumable), "abstention_label_count": len(all_labels) - len(consumable), "censored_label_count": sum(row.get("censoring_class") != "NONE" for row in all_labels), "counts_by_dose": {dose: {label: sum(row.get("dose") == dose and row.get("label_class") == label for row in all_labels) for label in sorted(BINARY)} for dose in ("T3", "T5", "T10")}, "primary_t5_consumable_count": sum(row.get("dose") == "T5" and row.get("binary_label_consumable") is True for row in all_labels), "source_commit": source_commit, "source_tree": source_tree, "authority_sha256": sha(args.authority.resolve()), "scheduler_status_sha256": sha(scheduler / "SCHEDULER_STATUS.json"), "plan_manifest_sha256": authority.get("exact_plan_manifest_sha256"), "parent_results": parent_results, "selection_outcomes_read": False, "teacher_predictions_read": False, "student_predictions_read": False, "eval160_status": "UNREAD", "protected_counters": COUNTERS, "censoring_policy": "preserve all abstains; only binary_label_consumable true enters V_phys"}
    write(output / "B2_FORMAL_M4_AGGREGATE.json", aggregate)
    audit = {"schema": "STAGE_VI_B2_FORMAL_M4_AGGREGATE_INDEPENDENT_AUDIT_V1", "status": "PASS_STAGE_VI_B2_FORMAL_M4" if not errors else "HOLD_STAGE_VI_B2_FORMAL_M4", "errors": sorted(set(errors)), "parent_count": len(seen_parents), "branch_count": len(all_branches), "treatment_label_count": len(all_labels), "consumable_binary_label_count": len(consumable), "source_commit": source_commit, "source_tree": source_tree, "authority_sha256": sha(args.authority.resolve()), "outcomes_read": False, "protected_counters": COUNTERS}
    write(output / "B2_FORMAL_M4_AGGREGATE_AUDIT.json", audit)
    seal(output)
    result = {"status": audit["status"], "root": str(output), "parent_count": len(seen_parents), "branches": len(all_branches), "labels": len(all_labels), "consumable": len(consumable), "errors": audit["errors"], "protected_counters": COUNTERS}
    write(output / "B2_AGGREGATE_RESULT.json", result)
    seal(output)
    print(json.dumps(result, sort_keys=True))
    return 0 if audit["status"] == "PASS_STAGE_VI_B2_FORMAL_M4" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD_STAGE_VI_B2_FORMAL_M4", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
