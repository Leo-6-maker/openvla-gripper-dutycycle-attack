#!/usr/bin/env python3
"""Read-only X0 mediator availability audit; no environment or outcome analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("CONTROL", "T3", "T5", "T10")
DOSES = ("T3", "T5", "T10")
COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
    "env_steps_with_perturbed_action": 0,
}
TASK_KEYS = (
    "task_success",
    "task_failure",
    "object_release",
    "object_drop",
    "release",
    "drop",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row: {path}:{line_number}")
            rows.append(value)
    return rows


def git_value(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(worktree), *args], text=True).strip()


def nested(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = nested(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = nested(child, key)
            if found is not None:
                return found
    return None


def first_value(record: dict[str, Any], branch: dict[str, Any], key: str) -> Any:
    return record.get(key, branch.get(key))


def branch_object(record: dict[str, Any]) -> dict[str, Any]:
    branch = record.get("branch")
    return branch if isinstance(branch, dict) else record


def derive_parent(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = re.match(r"^\d+_(libero_[^/]+__task_[^/]+__state_[^/]+)$", part)
        if match:
            return match.group(1).replace("__", "/")
    return None


def identity(record: dict[str, Any], branch: dict[str, Any], path: Path) -> tuple[str, str, int]:
    parent = first_value(record, branch, "canonical_parent_key")
    if not isinstance(parent, str):
        parent = nested(record, "canonical_parent_key")
    if not isinstance(parent, str):
        parent = derive_parent(path)
    probe_id = first_value(record, branch, "probe_id")
    if not isinstance(probe_id, str):
        probe_id = nested(record, "probe_id")
    probe_step = first_value(record, branch, "probe_step")
    if probe_step is None:
        probe_step = nested(record, "probe_step")
    if not isinstance(parent, str) or not isinstance(probe_id, str) or not isinstance(probe_step, int):
        raise ValueError(f"identity missing in {path}: {record.keys()}")
    return parent, probe_id, probe_step


def arm(record: dict[str, Any], branch: dict[str, Any]) -> str:
    value = first_value(record, branch, "arm")
    if value not in ARMS:
        raise ValueError(f"invalid arm {value!r}")
    return str(value)


def finite_vector(value: Any, length: int | None = None) -> bool:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        return False
    try:
        return all(float(item) == float(item) for item in value)
    except (TypeError, ValueError):
        return False


def bool_value(value: Any) -> bool:
    return isinstance(value, bool)


def collect_protected(value: Any, found: list[dict[str, Any]], path: str = "") -> None:
    if isinstance(value, dict):
        counters = value.get("protected_counters")
        if isinstance(counters, dict):
            found.append({"path": path, "counters": counters})
        for key, child in value.items():
            collect_protected(child, found, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect_protected(child, found, f"{path}[{index}]")


def audit_branch(record: dict[str, Any], path: Path) -> dict[str, Any]:
    branch = branch_object(record)
    parent, probe_id, probe_step = identity(record, branch, path)
    branch_arm = arm(record, branch)
    rows = branch.get("rows")
    actions = branch.get("actions")
    receipts = branch.get("treatment_receipts")
    if not isinstance(rows, list):
        rows = []
    if not isinstance(actions, list):
        actions = []
    if not isinstance(receipts, list):
        receipts = []

    command_rows = [
        isinstance(action, dict)
        and finite_vector(action.get("env_action"), 7)
        and finite_vector(action.get("normalized_action"), 7)
        and finite_vector(action.get("raw_policy_action"), 7)
        for action in actions
    ]
    aperture_field = None
    for candidate in ("gripper_aperture", "post_aperture"):
        if rows and all(isinstance(row, dict) and isinstance(row.get(candidate), (int, float)) for row in rows):
            aperture_field = candidate
            break
    contact_fields = ("post_contact_telemetry_valid", "post_object_gripper_contact", "post_object_support_contact")
    contact_complete = bool(rows) and all(
        isinstance(row, dict) and all(bool_value(row.get(field)) for field in contact_fields) for row in rows
    )
    object_position_complete = bool(rows) and all(
        isinstance(row, dict) and finite_vector(row.get("post_object_position"), 3) for row in rows
    )
    relative_steps = [row.get("relative_step") if isinstance(row, dict) else None for row in rows]
    relative_step_complete = bool(rows) and all(isinstance(step, int) for step in relative_steps)
    treatment_receipt_complete = branch_arm == "CONTROL" or (
        "treatment_compliance" in branch
        and isinstance(branch.get("treatment_compliance"), dict)
        and "treatment_receipts" in branch
        and isinstance(branch.get("treatment_receipts"), list)
    )
    task_keys = sorted(key for key in TASK_KEYS if nested(record, key) is not None)
    return {
        "stage": None,
        "source_file": str(path),
        "canonical_parent_key": parent,
        "probe_id": probe_id,
        "probe_step": probe_step,
        "arm": branch_arm,
        "row_count": len(rows),
        "action_count": len(actions),
        "command_fields_complete": bool(actions) and all(command_rows),
        "command_field_complete_count": sum(command_rows),
        "treatment_receipt_complete": treatment_receipt_complete,
        "aperture_field": aperture_field,
        "aperture_complete": aperture_field is not None,
        "contact_complete": contact_complete,
        "object_position_complete": object_position_complete,
        "relative_steps": relative_steps,
        "relative_step_complete": relative_step_complete,
        "task_keys_present": task_keys,
        "required_horizon": branch.get("required_physical_steps"),
        "available_horizon": branch.get("available_horizon_steps"),
    }


def source_records(stage: str, paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for path in sorted(paths):
        rows = load_jsonl(path)
        manifest.append({"stage": stage, "path": str(path), "sha256": sha256_file(path), "rows": len(rows)})
        for record in rows:
            item = audit_branch(record, path)
            item["stage"] = stage
            records.append(item)
    return records, manifest


def branch_paths(protocol: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    stage_v = Path(protocol["inputs"]["stage_v_branch_root"]["path"])
    current = sorted(stage_v.rglob("M4_COUNTERFACTUAL_BRANCHES_V1.jsonl"))
    binding = Path(protocol["inputs"]["stage_v_branch_root"]["bridge_parent_binding"])
    bridge = load_json(binding)
    bridge_root = Path(bridge["replacement_parent_root"])
    bridge_files = sorted(bridge_root.rglob("M4_COUNTERFACTUAL_BRANCHES_V1.jsonl"))
    stage_v_paths = [path for path in current if path not in bridge_files]
    stage_v_paths.extend(bridge_files)
    stage_vi = Path(protocol["inputs"]["stage_vi_b2_branches"]["path"])
    if not stage_vi.is_file():
        raise FileNotFoundError(stage_vi)
    return stage_v_paths, [stage_vi]


def audit_labels(protocol: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in (("stage_v", protocol["inputs"]["stage_v_labels"]), ("stage_vi_b2", protocol["inputs"]["stage_vi_b2_labels"])):
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"label SHA256 mismatch: {path}")
        rows = load_jsonl(path)
        protected: list[dict[str, Any]] = []
        for row in rows:
            collect_protected(row, protected)
        result[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "expected_row_count": spec["rows"],
            "schema_values": sorted({str(row.get("schema")) for row in rows}),
            "protected_counter_records": len(protected),
            "protected_counters": sorted({json.dumps(item["counters"], sort_keys=True) for item in protected}),
        }
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"label row count mismatch: {path}")
    return result


def paired_groups(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["stage"], record["canonical_parent_key"], record["probe_id"], int(record["probe_step"]))
        arm_name = record["arm"]
        if arm_name in groups[key]:
            raise ValueError(f"duplicate group arm: {key} {arm_name}")
        groups[key][arm_name] = record
    return groups


def availability(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = paired_groups(records)
    complete_groups = [group for group in groups.values() if all(arm_name in group for arm_name in ARMS)]
    equal_horizon_groups = [
        group for group in complete_groups
        if len({int(group[arm_name]["row_count"]) for arm_name in ARMS}) == 1
        and len({int(group[arm_name]["action_count"]) for arm_name in ARMS}) == 1
    ]
    aligned_overlap_groups = [
        group for group in complete_groups
        if all(group[arm_name]["relative_step_complete"] for arm_name in ARMS)
        and all(
            set(group["CONTROL"]["relative_steps"]).intersection(group[dose]["relative_steps"])
            for dose in DOSES
        )
    ]
    m1_records = [record for record in records if record["command_fields_complete"] and record["treatment_receipt_complete"]]
    m2_records = [record for record in records if record["aperture_complete"]]
    m3_records = [record for record in records if record["contact_complete"]]
    m4_records = [record for record in records if record["object_position_complete"]]
    pair_complete = len(aligned_overlap_groups) == len(groups) and bool(groups)
    task_keys = sorted({key for record in records for key in record["task_keys_present"]})
    return {
        "record_count": len(records),
        "parent_count": len({record["canonical_parent_key"] for record in records}),
        "probe_group_count": len(groups),
        "arm_counts": dict(sorted(Counter(record["arm"] for record in records).items())),
        "row_count_distribution": dict(sorted(Counter(record["row_count"] for record in records).items())),
        "action_count_distribution": dict(sorted(Counter(record["action_count"] for record in records).items())),
        "complete_four_arm_groups": len(complete_groups),
        "equal_horizon_four_arm_groups": len(equal_horizon_groups),
        "aligned_overlap_four_arm_groups": len(aligned_overlap_groups),
        "all_four_arm_groups_have_exact_overlap": pair_complete,
        "m1": {
            "name": "commanded_open_fraction",
            "exact_record_count": len(m1_records),
            "available": len(m1_records) == len(records),
            "required_fields": ["actions.env_action", "actions.normalized_action", "actions.raw_policy_action", "treatment_compliance", "treatment_receipts"],
        },
        "m2": {
            "name": "aperture_excess_auc_vs_control" if pair_complete else "max_aperture_delta_vs_control",
            "aperture_field_counts": dict(sorted(Counter(record["aperture_field"] for record in records).items())),
            "exact_record_count": len(m2_records),
            "available": len(m2_records) == len(records) and pair_complete,
            "fallback_available": len(m2_records) == len(records) and bool(groups),
            "required_fields": ["gripper_aperture or post_aperture", "relative_step", "matched CONTROL", "nonempty exact overlap"],
        },
        "m3": {
            "name": "any_contact_loss",
            "exact_record_count": len(m3_records),
            "available": len(m3_records) == len(records) and pair_complete,
            "required_fields": ["relative_step", "post_contact_telemetry_valid", "post_object_gripper_contact", "post_object_support_contact"],
        },
        "m4": {
            "name": "object_displacement",
            "exact_record_count": len(m4_records),
            "available": len(m4_records) == len(records) and pair_complete,
            "required_fields": ["relative_step", "post_object_position"],
        },
        "task_failure": {
            "name": "frozen_exact_task_failure_taxonomy",
            "available": bool(task_keys) and len(task_keys) > 0,
            "keys_present": task_keys,
            "rule": "do not infer from V_phys or physical class",
        },
    }


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    root_seal = {
        "schema": "STAGE_X_X0_MEDIATOR_AVAILABILITY_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": sha256_file(root / "STAGE_X_X0_MEDIATOR_AVAILABILITY.json"),
        "sha256sums_sha256": sums_sha,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "ROOT_SEAL.json").write_text(json.dumps(root_seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("schema") != "STAGE_X_X0_DUTY_CYCLE_MECHANISM_PROTOCOL_V1":
        raise ValueError("wrong X0 protocol")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    stage_v_paths, stage_vi_paths = branch_paths(protocol)
    if len(stage_v_paths) != 40:
        raise ValueError(f"Stage V branch file count is {len(stage_v_paths)}, expected 40")
    stage_v_records, stage_v_manifest = source_records("STAGE_V", stage_v_paths)
    stage_vi_records, stage_vi_manifest = source_records("STAGE_VI_B2", stage_vi_paths)
    records = stage_v_records + stage_vi_records
    labels = audit_labels(protocol)
    protected: list[dict[str, Any]] = []
    for record in records:
        collect_protected(record, protected)
    counters = [item["counters"] for item in protected]
    protected_ok = all(counters_item == COUNTERS for counters_item in counters) if counters else True
    stage_v_parent_count = len({record["canonical_parent_key"] for record in stage_v_records})
    stage_vi_parent_count = len({record["canonical_parent_key"] for record in stage_vi_records})
    if len(stage_v_records) != 3840 or stage_v_parent_count != 40:
        raise ValueError(f"Stage V branch population mismatch: {len(stage_v_records)} rows/{stage_v_parent_count} parents")
    if len(stage_vi_records) != 1536 or stage_vi_parent_count != 16:
        raise ValueError(f"Stage VI-B2 branch population mismatch: {len(stage_vi_records)} rows/{stage_vi_parent_count} parents")

    summary = {
        "schema": "STAGE_X_X0_MEDIATOR_AVAILABILITY_V1",
        "status": "AUDIT_COMPLETE",
        "x0_authorized": False,
        "source_commit": git_value(args.worktree, "rev-parse", "HEAD"),
        "source_tree": git_value(args.worktree, "rev-parse", "HEAD^{tree}"),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "physical_intervention": False,
        "new_env_steps": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "input_file_count": len(stage_v_manifest) + len(stage_vi_manifest) + 2,
        "input_files": stage_v_manifest + stage_vi_manifest,
        "labels": labels,
        "population": {
            "stage_v_parent_count": stage_v_parent_count,
            "stage_v_branch_count": len(stage_v_records),
            "stage_vi_b2_parent_count": stage_vi_parent_count,
            "stage_vi_b2_branch_count": len(stage_vi_records),
            "combined_branch_count": len(records),
        },
        "protected_counter_records": len(protected),
        "protected_counters_observed": sorted({json.dumps(item, sort_keys=True) for item in counters}),
        "protected_counters_valid": protected_ok,
        "stage_v_availability": availability(stage_v_records),
        "stage_vi_b2_availability": availability(stage_vi_records),
        "combined_availability": availability(records),
        "definitions_selected_by": "field completeness and exact matched-pair coverage only; no outcomes or scores",
        "selection_outcome_used": False,
    }
    (root / "STAGE_X_X0_MEDIATOR_AVAILABILITY.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "EXECUTION_MANIFEST.json").write_text(json.dumps({
        "schema": "STAGE_X_X0_MEDIATOR_AVAILABILITY_EXECUTION_MANIFEST_V1",
        "protocol": str(args.protocol),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script_sha256": summary["source_script_sha256"],
        "read_only": True,
        "physical_intervention": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal(root, summary)
    print(json.dumps({
        "status": summary["status"],
        "combined_branch_count": len(records),
        "protected_counters_valid": protected_ok,
        "output_root": str(root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
