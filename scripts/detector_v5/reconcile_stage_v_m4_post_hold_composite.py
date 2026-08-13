#!/usr/bin/env python3
"""Build the sealed post-HOLD 55-attempt / 40-parent composite inputs.

This is CPU-only and reads existing receipts.  It never launches a rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
QUALIFICATION_SUITES = ("libero_10", "libero_goal", "libero_spatial")
SOURCE_COMMIT = "3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2"
SOURCE_TREE = "2492a075e782a112d1e857248956b2647e751039"
RUNNER_SHA256 = "26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279"
V2_HOLD_SHA256 = "866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9"
V1_INVALID_HOLD_SHA256 = "2e2863ff9c84c2b004df32d258dadc9dba320b2ff9a6ed42f0873d6bc36843e0"

# Frozen mechanical result of the V1.1 queue prefix and the pre-HOLD slot deficit.
EXPECTED_NEW_BY_SUITE = {
    "libero_10": ["libero_10/task_00/state_27"],
    "libero_goal": [
        "libero_goal/task_03/state_36",
        "libero_goal/task_03/state_41",
        "libero_goal/task_02/state_40",
    ],
    "libero_spatial": [
        "libero_spatial/task_09/state_34",
        "libero_spatial/task_06/state_24",
        "libero_spatial/task_06/state_34",
        "libero_spatial/task_04/state_44",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_key(key: str) -> tuple[str, int, int]:
    parts = key.split("/")
    require(len(parts) == 3 and parts[1].startswith("task_") and parts[2].startswith("state_"), f"PARENT_KEY_INVALID:{key}")
    return parts[0], int(parts[1][5:]), int(parts[2][6:])


def sidecar_matches(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").split()
    return len(fields) >= 1 and fields[0] == sha256_file(path)


def input_binding(path: Path) -> dict[str, str]:
    require(path.is_file(), f"INPUT_MISSING:{path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_receipt(
    path: Path,
    expected_sha: str,
    key: str,
    replicate: str,
    expected_status: str | None,
) -> dict[str, Any]:
    require(path.is_file(), f"RECEIPT_MISSING:{key}:{replicate}:{path}")
    actual_sha = sha256_file(path)
    require(actual_sha == expected_sha, f"RECEIPT_SHA_MISMATCH:{key}:{replicate}")
    receipt = load_json(path)
    require(receipt.get("schema") == "STAGE_V_M4_CORRIDOR_PREFLIGHT_V1", f"RECEIPT_SCHEMA_INVALID:{key}:{replicate}")
    require(receipt.get("canonical_parent_key") == key, f"RECEIPT_KEY_MISMATCH:{key}:{replicate}")
    require(receipt.get("replicate") == replicate, f"RECEIPT_REPLICATE_MISMATCH:{key}:{replicate}")
    require(receipt.get("source_commit") == SOURCE_COMMIT, f"RECEIPT_SOURCE_COMMIT_MISMATCH:{key}:{replicate}")
    require(receipt.get("source_tree") == SOURCE_TREE, f"RECEIPT_SOURCE_TREE_MISMATCH:{key}:{replicate}")
    require(receipt.get("outcomes_read") is False, f"RECEIPT_OUTCOME_READ:{key}:{replicate}")
    require(receipt.get("old_artifacts_reused") is False, f"RECEIPT_OLD_ARTIFACT_REUSE:{key}:{replicate}")
    require(receipt.get("source_artifacts_modified") is False, f"RECEIPT_SOURCE_MODIFIED:{key}:{replicate}")
    require(receipt.get("protected_counters") == COUNTERS, f"RECEIPT_COUNTER_INVALID:{key}:{replicate}")
    if expected_status is not None:
        require(receipt.get("status") == expected_status, f"RECEIPT_STATUS_MISMATCH:{key}:{replicate}")
    if receipt.get("status") == "PASS":
        require(receipt.get("clean_success") is True, f"PASS_CLEAN_SUCCESS_INVALID:{key}:{replicate}")
        require(receipt.get("m4_primary_horizon_complete") is True, f"PASS_HORIZON_INVALID:{key}:{replicate}")
        require(receipt.get("m4_probe_eligible") is True, f"PASS_PROBE_ELIGIBILITY_INVALID:{key}:{replicate}")
        require(receipt.get("probe_count") == 24, f"PASS_PROBE_COUNT_INVALID:{key}:{replicate}")
        require(receipt.get("reason") == "M4_CORRIDOR_24_EXACT", f"PASS_REASON_INVALID:{key}:{replicate}")
    return receipt


def terminal_receipts(report: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    audit = report.get("independent_audit", {})
    rows = audit.get("receipt_rows")
    require(isinstance(rows, list) and len(rows) == 80, "V2_RECEIPT_ROWS_NOT_EXACT_80")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        require(isinstance(row, dict), "V2_RECEIPT_ROW_NOT_OBJECT")
        key, replicate = str(row.get("canonical_parent_key", "")), str(row.get("replicate", ""))
        require(replicate in {"A", "B"}, f"V2_REPLICATE_INVALID:{key}:{replicate}")
        require(key not in grouped or replicate not in grouped[key], f"V2_RECEIPT_DUPLICATE:{key}:{replicate}")
        path = Path(str(row.get("path", "")))
        validate_receipt(path, str(row.get("receipt_sha256", "")), key, replicate, str(row.get("status")))
        grouped.setdefault(key, {})[replicate] = dict(row)
    require(len(grouped) == 40 and all(set(pair) == {"A", "B"} for pair in grouped.values()), "V2_PARENT_PAIR_ACCOUNTING_INVALID")
    return grouped


def validate_outer(outer: Mapping[str, Any]) -> None:
    require(outer.get("status") == "FROZEN_RUNTIME_AUTHORIZED" and outer.get("runtime_authorized") is True, "OUTER_PROTOCOL_NOT_FROZEN_AUTHORIZED")
    source = outer.get("source_binding", {})
    require(source.get("science_commit") == SOURCE_COMMIT, "OUTER_SOURCE_COMMIT_INVALID")
    require(source.get("science_tree") == SOURCE_TREE, "OUTER_SOURCE_TREE_INVALID")
    require(source.get("corridor_runner_sha256") == RUNNER_SHA256, "OUTER_RUNNER_SHA_INVALID")
    qualification = outer.get("qualification", {})
    require(qualification.get("no_speculative_launch") is True and qualification.get("no_fallback") is True and qualification.get("no_rerun") is True, "OUTER_SEQUENTIAL_RULE_INVALID")
    require(qualification.get("target_stable_by_suite") == {"libero_10": 1, "libero_goal": 3, "libero_spatial": 4}, "OUTER_TARGET_INVALID")
    require(qualification.get("slot_order") == {"libero_10": ["VAL"], "libero_goal": ["TRAIN", "TRAIN", "TEST"], "libero_spatial": ["TRAIN", "TRAIN", "TRAIN", "VAL"]}, "OUTER_SLOT_ORDER_INVALID")
    require(qualification.get("candidate_parent_count") == 22, "OUTER_CANDIDATE_COUNT_INVALID")


def new_runtime_pairs(runtime: Mapping[str, Any], candidate: Mapping[str, Any], outer: Mapping[str, Any], runtime_sha: str) -> dict[str, dict[str, Any]]:
    require(runtime.get("schema") == "STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION", "RUNTIME_SCHEMA_INVALID")
    require(runtime.get("status") == "PASS_POST_HOLD_CORRIDOR_TARGETS_REACHED", "RUNTIME_NOT_TERMINAL_PASS")
    require(runtime.get("terminal") is True and runtime.get("sealed") is True and runtime.get("immutable") is True, "RUNTIME_NOT_SEALED_IMMUTABLE")
    require(runtime.get("retry_forbidden") is True and runtime.get("intervention_executed") is False and runtime.get("outcomes_read") is False, "RUNTIME_BOUNDARY_INVALID")
    require(runtime.get("protected_counters") == COUNTERS, "RUNTIME_COUNTERS_INVALID")
    require(runtime.get("attempted_identity_count") == 12 and runtime.get("pair_count") == 12, "RUNTIME_PAIR_COUNT_INVALID")
    require(runtime.get("stable_by_suite") == {"libero_10": 1, "libero_goal": 3, "libero_spatial": 4}, "RUNTIME_STABLE_COUNTS_INVALID")
    candidate_rows = {str(row.get("canonical_parent_key")): row for row in candidate.get("parents", [])}
    require(len(candidate_rows) == 22, "CANDIDATE_MANIFEST_NOT_EXACT_22")
    pairs = runtime.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 12, "RUNTIME_PAIRS_NOT_EXACT_12")
    by_key: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        require(isinstance(pair, dict), "RUNTIME_PAIR_NOT_OBJECT")
        key, suite = str(pair.get("canonical_parent_key", "")), str(pair.get("suite", ""))
        require(key not in by_key, f"RUNTIME_DUPLICATE_KEY:{key}")
        require(suite in QUALIFICATION_SUITES and key.startswith(suite + "/"), f"RUNTIME_SUITE_INVALID:{key}")
        rank = int(pair.get("selection_rank", 0))
        require(rank >= 1, f"RUNTIME_RANK_INVALID:{key}")
        candidate_row = candidate_rows.get(key)
        require(candidate_row is not None and candidate_row.get("selection_rank") == rank and candidate_row.get("taxonomy_status") == "SUPPORTED", f"RUNTIME_CANDIDATE_BINDING_INVALID:{key}")
        reps = pair.get("replicates", {})
        require(set(reps) == {"A", "B"}, f"RUNTIME_REPLICATES_INVALID:{key}")
        statuses = []
        for replicate in ("A", "B"):
            item = reps[replicate]
            receipt = validate_receipt(Path(str(item.get("receipt", ""))), str(item.get("receipt_sha256", "")), key, replicate, str(item.get("status")))
            require(receipt.get("trajectory_sha256") == item.get("trajectory_sha256"), f"RUNTIME_TRAJECTORY_BINDING_INVALID:{key}:{replicate}")
            statuses.append(str(item.get("status")))
        status_pair = "/".join(statuses)
        require(pair.get("status_pair") == status_pair, f"RUNTIME_STATUS_PAIR_INVALID:{key}")
        require(pair.get("stable_pass_pass") is (status_pair == "PASS/PASS"), f"RUNTIME_STABLE_FLAG_INVALID:{key}")
        require(pair.get("outcomes_read") is False and pair.get("protected_counters") == COUNTERS, f"RUNTIME_PAIR_BOUNDARY_INVALID:{key}")
        if status_pair == "PASS/PASS":
            require(all(reps[rep].get("probe_count") == 24 and reps[rep].get("reason") == "M4_CORRIDOR_24_EXACT" for rep in ("A", "B")), f"RUNTIME_PASS_FIELDS_INVALID:{key}")
        pair_copy = dict(pair)
        pair_copy["runtime_reconciliation_sha256"] = runtime_sha
        by_key[key] = pair_copy
    for suite in QUALIFICATION_SUITES:
        rows = sorted((row for row in by_key.values() if row["suite"] == suite), key=lambda row: int(row["selection_rank"]))
        require([int(row["selection_rank"]) for row in rows] == list(range(1, len(rows) + 1)), f"RUNTIME_RANK_PREFIX_INVALID:{suite}")
        queue = outer["qualification"]["queues"][suite]
        require([row["canonical_parent_key"] for row in rows] == queue[: len(rows)], f"RUNTIME_QUEUE_PREFIX_INVALID:{suite}")
    selected = {suite: [key for key in EXPECTED_NEW_BY_SUITE[suite]] for suite in QUALIFICATION_SUITES}
    for suite, expected in selected.items():
        actual = [row["canonical_parent_key"] for row in sorted((row for row in by_key.values() if row["suite"] == suite and row["stable_pass_pass"]), key=lambda row: int(row["selection_rank"]))]
        require(actual == expected, f"RUNTIME_SELECTED_PREFIX_INVALID:{suite}")
    return by_key


def build_attempt_registry(
    terminal: Mapping[str, dict[str, dict[str, Any]]],
    old_registry: Mapping[str, Any],
    invalid: Mapping[str, Any],
    runtime_pairs: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, dict[str, str]],
) -> dict[str, Any]:
    old_rows = old_registry.get("attempted_identities")
    require(old_registry.get("status") == "FROZEN_EXACT_43" and isinstance(old_rows, list) and len(old_rows) == 43, "OLD_ATTEMPT_REGISTRY_INVALID")
    old_map = {str(row.get("canonical_parent_key")): row for row in old_rows}
    require(len(old_map) == 43, "OLD_ATTEMPT_REGISTRY_DUPLICATE")
    terminal_keys = set(terminal)
    require(terminal_keys == {key for key, row in old_map.items() if row.get("origin") == "V2_CURRENT_SOURCE_FINAL40"}, "OLD_ATTEMPT_REGISTRY_V2_KEYS_INVALID")
    rows: list[dict[str, Any]] = []
    for key in sorted(terminal):
        pair = terminal[key]
        a, b = pair["A"], pair["B"]
        old = old_map[key]
        require(old.get("A_receipt_sha256") == a.get("receipt_sha256") and old.get("B_receipt_sha256") == b.get("receipt_sha256"), f"OLD_REGISTRY_RECEIPT_BINDING_INVALID:{key}")
        rows.append({
            "canonical_parent_key": key,
            "suite": key.split("/", 1)[0],
            "origin": "V2_CURRENT_SOURCE_FINAL40",
            "status_pair": f"{a.get('status')}/{b.get('status')}",
            "A_receipt_path": a.get("path"),
            "A_receipt_sha256": a.get("receipt_sha256"),
            "B_receipt_path": b.get("path"),
            "B_receipt_sha256": b.get("receipt_sha256"),
        })
    invalid_rows = invalid.get("attempted_identities")
    require(invalid.get("status") == "HOLD_ENGINEERING_INVALID_PRELAUNCH_GOVERNANCE_INCOMPLETE" and invalid.get("consumable") is False and isinstance(invalid_rows, list) and len(invalid_rows) == 3, "INVALID_V1_HOLD_INVALID")
    for item in invalid_rows:
        key = str(item.get("canonical_parent_key"))
        require(key not in terminal and key not in runtime_pairs, f"INVALID_V1_KEY_OVERLAP:{key}")
        rows.append({
            "canonical_parent_key": key,
            "suite": item.get("suite"),
            "origin": "POST_HOLD_V1_ENGINEERING_INVALID",
            "status_pair": item.get("pair_status"),
            "attempt_status": item.get("attempt_status"),
            "A_receipt_sha256": item.get("A", {}).get("receipt_sha256"),
            "B_receipt_sha256": item.get("B", {}).get("receipt_sha256"),
            "B_runtime_binding_receipt_sha256": item.get("B", {}).get("runtime_binding_receipt_sha256"),
            "B_log_sha256": item.get("B", {}).get("log_sha256"),
        })
    for key, pair in sorted(runtime_pairs.items()):
        require(key not in terminal and key not in {str(item.get("canonical_parent_key")) for item in invalid_rows}, f"NEW_ATTEMPT_KEY_OVERLAP:{key}")
        rows.append({
            "canonical_parent_key": key,
            "suite": pair["suite"],
            "origin": "POST_HOLD_V1_1",
            "selection_rank": pair["selection_rank"],
            "status_pair": pair["status_pair"],
            "reconciled_utc": pair.get("reconciled_utc"),
            "runtime_reconciliation_sha256": bindings["runtime_reconciliation"]["sha256"],
            "A_receipt_path": pair["replicates"]["A"]["receipt"],
            "A_receipt_sha256": pair["replicates"]["A"]["receipt_sha256"],
            "B_receipt_path": pair["replicates"]["B"]["receipt"],
            "B_receipt_sha256": pair["replicates"]["B"]["receipt_sha256"],
        })
    require(len(rows) == 55 and len({row["canonical_parent_key"] for row in rows}) == 55, "ATTEMPT_REGISTRY_NOT_EXACT_55")
    origin_counts = {origin: sum(row["origin"] == origin for row in rows) for origin in ("V2_CURRENT_SOURCE_FINAL40", "POST_HOLD_V1_ENGINEERING_INVALID", "POST_HOLD_V1_1")}
    require(origin_counts == {"V2_CURRENT_SOURCE_FINAL40": 40, "POST_HOLD_V1_ENGINEERING_INVALID": 3, "POST_HOLD_V1_1": 12}, "ATTEMPT_REGISTRY_ORIGIN_COUNTS_INVALID")
    return {
        "schema": "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1",
        "status": "FROZEN_EXACT55_CORRIDOR_ATTEMPT_FIREWALL",
        "sealed": True,
        "attempted_identity_count": 55,
        "unique_identity_count": 55,
        "duplicate_count": 0,
        "origin_counts": origin_counts,
        "inputs": {name: value for name, value in bindings.items() if name in {"terminal_report", "old_attempt_registry", "invalid_hold", "runtime_reconciliation"}},
        "attempted_identities": sorted(rows, key=lambda row: (row["origin"], row["canonical_parent_key"])),
        "firewall": "EXCLUDE_ALL_55_FROM_PRIMARY_TEACHER_STUDENT_FIT_CAL_CHECK_THRESHOLD_CHECKPOINT_FEATURE_MODEL_SELECTION",
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
        "source_binding": {"science_commit": SOURCE_COMMIT, "science_tree": SOURCE_TREE, "runner_sha256": RUNNER_SHA256},
    }


def build_final_population(
    inventory: Mapping[str, Any],
    terminal: Mapping[str, dict[str, dict[str, Any]]],
    runtime_pairs: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Any],
    old_split: Mapping[str, Any],
    bindings: Mapping[str, dict[str, str]],
    final_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_split_rows = old_split.get("parents")
    require(isinstance(old_split_rows, list) and len(old_split_rows) == 40, "HISTORICAL_SPLIT_NOT_EXACT_40")
    old_split_map = {str(row.get("canonical_parent_key")): row for row in old_split_rows}
    require(len(old_split_map) == 40, "HISTORICAL_SPLIT_DUPLICATE")
    candidate_rows = {str(row.get("canonical_parent_key")): row for row in candidate.get("parents", [])}
    parents: list[dict[str, Any]] = []
    predecessor_keys = {str(row["canonical_parent_key"]) for row in inventory.get("parents", [])}
    require(len(predecessor_keys) == 32 and predecessor_keys.issubset(terminal), "PREDECESSOR_INVENTORY_INVALID")
    for item in sorted(inventory["parents"], key=lambda row: str(row["canonical_parent_key"])):
        key, suite, task_index, state_index = str(item["canonical_parent_key"]), str(item["suite"]), parse_key(str(item["canonical_parent_key"]))[1], parse_key(str(item["canonical_parent_key"]))[2]
        split_row = old_split_map.get(key)
        require(split_row is not None and split_row.get("split") in {"TRAIN", "VAL", "TEST"}, f"PREDECESSOR_SPLIT_MISSING:{key}")
        pair = terminal[key]
        parents.append({
            "canonical_parent_key": key,
            "suite": suite,
            "task_index": task_index,
            "state_index": state_index,
            "split": split_row["split"],
            "origin": "V2_PREDECESSOR_PASS_PASS",
            "status_pair": "PASS/PASS",
            "science_commit": SOURCE_COMMIT,
            "science_tree": SOURCE_TREE,
            "runner_sha256": RUNNER_SHA256,
            "A_receipt_path": pair["A"]["path"],
            "A_receipt_sha256": pair["A"]["receipt_sha256"],
            "A_trajectory_sha256": pair["A"].get("trajectory_sha256"),
            "B_receipt_path": pair["B"]["path"],
            "B_receipt_sha256": pair["B"]["receipt_sha256"],
            "B_trajectory_sha256": pair["B"].get("trajectory_sha256"),
        })
    for suite in QUALIFICATION_SUITES:
        for key, split in zip(EXPECTED_NEW_BY_SUITE[suite], {"libero_10": ["VAL"], "libero_goal": ["TRAIN", "TRAIN", "TEST"], "libero_spatial": ["TRAIN", "TRAIN", "TRAIN", "VAL"]}[suite]):
            pair = runtime_pairs[key]
            candidate_row = candidate_rows[key]
            parsed_suite, task_index, state_index = parse_key(key)
            require(parsed_suite == suite and pair["status_pair"] == "PASS/PASS", f"NEW_PARENT_NOT_PASS:{key}")
            parents.append({
                "canonical_parent_key": key,
                "suite": suite,
                "task_index": task_index,
                "state_index": state_index,
                "split": split,
                "origin": "POST_HOLD_V1_1_PASS_PASS",
                "status_pair": "PASS/PASS",
                "selection_rank": pair["selection_rank"],
                "taxonomy_status": candidate_row["taxonomy_status"],
                "v7_candidate_sha256": candidate_row["v7_candidate_sha256"],
                "qualification_rank_sha256": candidate_row["qualification_rank_sha256"],
                "candidate_manifest_sha256": bindings["candidate_manifest"]["sha256"],
                "runtime_reconciliation_sha256": bindings["runtime_reconciliation"]["sha256"],
                "science_commit": SOURCE_COMMIT,
                "science_tree": SOURCE_TREE,
                "runner_sha256": RUNNER_SHA256,
                "A_receipt_path": pair["replicates"]["A"]["receipt"],
                "A_receipt_sha256": pair["replicates"]["A"]["receipt_sha256"],
                "A_trajectory_sha256": pair["replicates"]["A"]["trajectory_sha256"],
                "B_receipt_path": pair["replicates"]["B"]["receipt"],
                "B_receipt_sha256": pair["replicates"]["B"]["receipt_sha256"],
                "B_trajectory_sha256": pair["replicates"]["B"]["trajectory_sha256"],
            })
    require(len(parents) == 40 and len({row["canonical_parent_key"] for row in parents}) == 40, "FINAL40_IDENTITY_INVALID")
    suite_counts = {suite: sum(row["suite"] == suite for row in parents) for suite in SUITES}
    require(suite_counts == {suite: 10 for suite in SUITES}, "FINAL40_SUITE_COUNTS_INVALID")
    split_counts = {split: sum(row["split"] == split for row in parents) for split in ("TRAIN", "VAL", "TEST")}
    per_suite_split = {suite: {split: sum(row["suite"] == suite and row["split"] == split for row in parents) for split in ("TRAIN", "VAL", "TEST")} for suite in SUITES}
    require(split_counts == {"TRAIN": 24, "VAL": 8, "TEST": 8}, "FINAL_SPLIT_COUNTS_INVALID")
    require(all(counts == {"TRAIN": 6, "VAL": 2, "TEST": 2} for counts in per_suite_split.values()), "FINAL_PER_SUITE_SPLIT_INVALID")
    final_manifest = {
        "schema": "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2",
        "version": "POST-HOLD-COMPOSITE-V1",
        "status": "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE",
        "formal_m4_authorized": False,
        "parent_count": 40,
        "receipt_count": 80,
        "parents": sorted(parents, key=lambda row: (SUITES.index(row["suite"]), row["split"], row["canonical_parent_key"])),
        "counts_by_suite": suite_counts,
        "split_counts": split_counts,
        "per_suite_split_counts": per_suite_split,
        "selection_rule": "Carry only the 32 immutable current-source PASS/PASS predecessor pairs; append only the frozen eight V1.1 PASS/PASS queue-prefix identities; preserve predecessor split slots and fill only the pre-registered deficit slots.",
        "historical_v2_artifacts_are_not_reused_as_population": True,
        "historical_v2_split_used_only_for_predecessor_slot_mapping": True,
        "inputs": {
            "terminal_report": bindings["terminal_report"],
            "predecessor_inventory": bindings["predecessor_inventory"],
            "runtime_reconciliation": bindings["runtime_reconciliation"],
            "candidate_manifest": bindings["candidate_manifest"],
            "historical_v2_split": bindings["old_split"],
            "historical_v2_formal_manifest": bindings["old_manifest"],
            "outer_protocol": bindings["outer_protocol"],
            "compatibility_audit": bindings["compatibility_audit"],
        },
        "final_split_path": str(final_root / "STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json"),
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
        "source_binding": {"science_commit": SOURCE_COMMIT, "science_tree": SOURCE_TREE, "runner_sha256": RUNNER_SHA256},
    }
    final_split = {
        "schema": "STAGE_V_M4_FINAL_PARENT_SPLIT_V2",
        "version": "POST-HOLD-COMPOSITE-V1",
        "status": "FROZEN",
        "formal_m4_authorized": False,
        "assignment_rule": "Preserve the historical split for the 32 predecessor PASS/PASS rows; assign the eight V1.1 stable queue-prefix rows in suite order and frozen slot order exactly as encoded by the post-HOLD protocol.",
        "final_manifest_path": str(final_root / "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json"),
        "final_manifest_sha256": "BOUND_AFTER_MANIFEST_WRITE",
        "historical_v2_split_path": bindings["old_split"]["path"],
        "historical_v2_split_sha256": bindings["old_split"]["sha256"],
        "parents": [{key: row[key] for key in ("canonical_parent_key", "suite", "split", "origin")} for row in final_manifest["parents"]],
        "counts": split_counts,
        "per_suite_counts": per_suite_split,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    }
    return final_manifest, final_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("terminal-report", "old-attempt-registry", "invalid-hold", "runtime-reconciliation", "predecessor-inventory", "candidate-manifest", "outer-protocol", "compatibility-audit", "old-split", "old-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    final_root = args.output_dir.resolve()
    temporary_root = final_root.with_name(final_root.name + ".incomplete")
    require(not final_root.exists() and not temporary_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{final_root}")

    paths = {name.replace("-", "_"): value.resolve() for name, value in vars(args).items() if name not in {"output_dir"}}
    bindings = {name: input_binding(path) for name, path in paths.items()}
    require(bindings["terminal_report"]["sha256"] == V2_HOLD_SHA256, "V2_HOLD_SHA_MISMATCH")
    require(bindings["invalid_hold"]["sha256"] == V1_INVALID_HOLD_SHA256, "V1_INVALID_HOLD_SHA_MISMATCH")
    outer = load_json(args.outer_protocol.resolve())
    validate_outer(outer)
    report = load_json(args.terminal_report.resolve())
    require(report.get("status") == "HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT" and report.get("terminal") is True and report.get("sealed") is True and report.get("immutable") is True and report.get("retry_forbidden") is True, "V2_HOLD_BOUNDARY_INVALID")
    require(report.get("source_binding", {}).get("science_commit") == SOURCE_COMMIT and report.get("source_binding", {}).get("science_tree") == SOURCE_TREE, "V2_SOURCE_BINDING_INVALID")
    require(report.get("independent_audit", {}).get("protected_counters") == COUNTERS and report.get("independent_audit", {}).get("outcomes_read") is False, "V2_HOLD_COUNTER_BOUNDARY_INVALID")
    compatibility = load_json(args.compatibility_audit.resolve())
    require(compatibility.get("status") == "PASS_PREDECESSOR_32_COMPATIBLE_WITH_POST_HOLD_V1_1" and compatibility.get("failure_count") == 0 and compatibility.get("outcomes_read") is False and compatibility.get("intervention_executed") is False, "COMPATIBILITY_AUDIT_NOT_PASS")
    require(bindings["compatibility_audit"]["sha256"] == "8edf5e048b6dfb448fe0b5381e4cea221bc2a438f322a303ad6310fcc14a94e2", "COMPATIBILITY_AUDIT_SHA_MISMATCH")
    inventory = load_json(args.predecessor_inventory.resolve())
    require(inventory.get("schema") == "STAGE_V_M4_CORRIDOR_PREDECESSOR_PASS_PASS_INVENTORY_V1" and inventory.get("sealed") is True and inventory.get("parent_count") == 32 and inventory.get("receipt_count") == 64 and inventory.get("outcomes_read") is False and inventory.get("protected_counters") == COUNTERS, "PREDECESSOR_INVENTORY_INVALID")
    require(bindings["predecessor_inventory"]["sha256"] == "4020a3f45efefb704c7110fd20e243adf84594a2c0920c413ee44928284baf14", "PREDECESSOR_INVENTORY_SHA_MISMATCH")
    require(bindings["old_attempt_registry"]["sha256"] == "7d5cfd1b3396f6af4ecd6f3de9b9d6ef454bb927c14a6619a90f14b27a273968", "OLD_ATTEMPT_REGISTRY_SHA_MISMATCH")
    terminal = terminal_receipts(report)
    require(sum(1 for pair in terminal.values() if pair["A"].get("status") == "PASS" and pair["B"].get("status") == "PASS") == 32, "V2_PASS_PASS_COUNT_INVALID")
    candidate = load_json(args.candidate_manifest.resolve())
    require(candidate.get("schema") == "STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1_1" and candidate.get("status") == "FROZEN" and candidate.get("candidate_count") == 22 and candidate.get("outcomes_read") is False, "CANDIDATE_MANIFEST_INVALID")
    runtime = load_json(args.runtime_reconciliation.resolve())
    require(bindings["runtime_reconciliation"]["sha256"] == "6fce4411d737f16be7e01b76475d714bbef28006c51e667dec93afca22191a6a", "RUNTIME_RECONCILIATION_SHA_MISMATCH")
    require(sidecar_matches(args.runtime_reconciliation.resolve()), "RUNTIME_RECONCILIATION_SIDECAR_INVALID")
    runtime_pairs = new_runtime_pairs(runtime, candidate, outer, bindings["runtime_reconciliation"]["sha256"])
    old_split = load_json(args.old_split.resolve())
    old_manifest = load_json(args.old_manifest.resolve())
    require(bindings["old_split"]["sha256"] == "f76ababf0750a78ee7adb3c81e0d15b945275d7210e2d0e41de1a623c6549cc4", "HISTORICAL_SPLIT_SHA_MISMATCH")
    require(bindings["old_manifest"]["sha256"] == "f2c142b7c140b8412113e14a52d243eb8d9dafa9f9c4970456a91f2e41a479fa" and isinstance(old_manifest.get("parents"), list), "HISTORICAL_MANIFEST_INVALID")
    attempt_registry = build_attempt_registry(terminal, load_json(args.old_attempt_registry.resolve()), load_json(args.invalid_hold.resolve()), runtime_pairs, bindings)
    final_manifest, final_split = build_final_population(inventory, terminal, runtime_pairs, candidate, old_split, bindings, final_root)
    temporary_root.mkdir(parents=True)
    write_json(temporary_root / "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1.json", attempt_registry)
    write_json(temporary_root / "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json", final_manifest)
    final_manifest_sha = sha256_file(temporary_root / "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json")
    final_split["final_manifest_sha256"] = final_manifest_sha
    write_json(temporary_root / "STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json", final_split)
    composite = {
        "schema": "STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1",
        "status": "PASS_POST_HOLD_COMPOSITE_CORRIDOR_40_40",
        "sealed": False,
        "final40_frozen": True,
        "final_split_frozen": True,
        "predecessor_count": 32,
        "new_pass_pass_count": 8,
        "final_parent_count": 40,
        "counts_by_suite": final_manifest["counts_by_suite"],
        "split_counts": final_manifest["split_counts"],
        "per_suite_split_counts": final_manifest["per_suite_split_counts"],
        "attempted_identity_count": 55,
        "attempt_registry_path": str(final_root / "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1.json"),
        "attempt_registry_sha256": sha256_file(temporary_root / "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1.json"),
        "final_manifest_path": str(final_root / "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json"),
        "final_manifest_sha256": final_manifest_sha,
        "final_split_path": str(final_root / "STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json"),
        "final_split_sha256": sha256_file(temporary_root / "STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json"),
        "inputs": bindings,
        "claim_boundary": "Eligibility is limited to the prospectively defined clean-successful, taxonomy-supported, A/B corridor-stable critical-opportunity population; this is not a V2 repair and is not a formal M4 authorization.",
        "exact_40x24_plan_gate": "NOT_STARTED",
        "formal_teacher": "NOT_STARTED",
        "formal_student": "NOT_STARTED",
        "formal_m4": "NOT_AUTHORIZED",
        "v_phys": "NOT_GENERATED",
        "outcomes_read": False,
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
        "source_binding": {"science_commit": SOURCE_COMMIT, "science_tree": SOURCE_TREE, "runner_sha256": RUNNER_SHA256},
        "next_action": "RUN_INDEPENDENT_COMPOSITE_AUDIT_THEN_SEAL; STOP_BEFORE_EXACT_40X24",
    }
    write_json(temporary_root / "STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1.json", composite)
    os.replace(temporary_root, final_root)
    print(json.dumps({"status": composite["status"], "output_root": str(final_root), "attempted": 55, "final40": 40, "split": final_manifest["split_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
