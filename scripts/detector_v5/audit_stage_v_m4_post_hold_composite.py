#!/usr/bin/env python3
"""Independently audit and seal the post-HOLD composite population.

The audit re-reads the source receipts and does not import the producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
QUALIFICATION_SUITES = ("libero_10", "libero_goal", "libero_spatial")
SOURCE_COMMIT = "3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2"
SOURCE_TREE = "2492a075e782a112d1e857248956b2647e751039"
RUNNER_SHA256 = "26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279"
EXPECTED_NEW = {
    "libero_10": ["libero_10/task_00/state_27"],
    "libero_goal": ["libero_goal/task_03/state_36", "libero_goal/task_03/state_41", "libero_goal/task_02/state_40"],
    "libero_spatial": ["libero_spatial/task_09/state_34", "libero_spatial/task_06/state_24", "libero_spatial/task_06/state_34", "libero_spatial/task_04/state_44"],
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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"ARTIFACT_SIDECAR_MISSING:{path.name}")
    fields = sidecar.read_text(encoding="utf-8").split()
    require(fields and fields[0] == sha256_file(path), f"ARTIFACT_SIDECAR_MISMATCH:{path.name}")


def parse_key(key: str) -> tuple[str, int, int]:
    parts = key.split("/")
    require(len(parts) == 3 and parts[1].startswith("task_") and parts[2].startswith("state_"), f"PARENT_KEY_INVALID:{key}")
    return parts[0], int(parts[1][5:]), int(parts[2][6:])


def receipt(path: Path, expected_sha: str, key: str, replicate: str, status: str) -> dict[str, Any]:
    require(path.is_file() and sha256_file(path) == expected_sha, f"RECEIPT_HASH_INVALID:{key}:{replicate}")
    value = load_json(path)
    require(value.get("schema") == "STAGE_V_M4_CORRIDOR_PREFLIGHT_V1", f"RECEIPT_SCHEMA_INVALID:{key}:{replicate}")
    require(value.get("canonical_parent_key") == key and value.get("replicate") == replicate, f"RECEIPT_ID_INVALID:{key}:{replicate}")
    require(value.get("status") == status, f"RECEIPT_STATUS_INVALID:{key}:{replicate}")
    require(value.get("source_commit") == SOURCE_COMMIT and value.get("source_tree") == SOURCE_TREE, f"RECEIPT_SOURCE_INVALID:{key}:{replicate}")
    require(value.get("outcomes_read") is False and value.get("old_artifacts_reused") is False and value.get("source_artifacts_modified") is False, f"RECEIPT_BOUNDARY_INVALID:{key}:{replicate}")
    require(value.get("protected_counters") == COUNTERS, f"RECEIPT_COUNTERS_INVALID:{key}:{replicate}")
    if status == "PASS":
        require(value.get("clean_success") is True and value.get("m4_primary_horizon_complete") is True and value.get("m4_probe_eligible") is True, f"PASS_ELIGIBILITY_INVALID:{key}:{replicate}")
        require(value.get("probe_count") == 24 and value.get("reason") == "M4_CORRIDOR_24_EXACT", f"PASS_CORRIDOR_FIELDS_INVALID:{key}:{replicate}")
    return value


def terminal_map(report: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    rows = report.get("independent_audit", {}).get("receipt_rows")
    require(isinstance(rows, list) and len(rows) == 80, "V2_RECEIPT_ROWS_INVALID")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key, rep = str(row.get("canonical_parent_key")), str(row.get("replicate"))
        require(key not in grouped or rep not in grouped[key], f"V2_DUPLICATE_RECEIPT:{key}:{rep}")
        receipt(Path(str(row["path"])), str(row["receipt_sha256"]), key, rep, str(row["status"]))
        grouped.setdefault(key, {})[rep] = dict(row)
    require(len(grouped) == 40 and all(set(pair) == {"A", "B"} for pair in grouped.values()), "V2_PAIR_ACCOUNTING_INVALID")
    return grouped


def audit_all(root: Path, args: argparse.Namespace) -> dict[str, bool]:
    names = {
        "composite": "STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1.json",
        "attempts": "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1.json",
        "manifest": "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json",
        "split": "STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json",
    }
    artifacts = {name: root / filename for name, filename in names.items()}
    for path in artifacts.values():
        verify_sidecar(path)
    composite, attempts, manifest, split = (load_json(artifacts[name]) for name in ("composite", "attempts", "manifest", "split"))
    require(composite.get("schema") == "STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1", "COMPOSITE_SCHEMA_INVALID")
    require(composite.get("status") == "PASS_POST_HOLD_COMPOSITE_CORRIDOR_40_40" and composite.get("final40_frozen") is True and composite.get("final_split_frozen") is True, "COMPOSITE_STATUS_INVALID")
    require(composite.get("final_parent_count") == 40 and composite.get("attempted_identity_count") == 55, "COMPOSITE_COUNTS_INVALID")
    require(composite.get("outcomes_read") is False and composite.get("intervention_executed") is False and composite.get("protected_counters") == COUNTERS, "COMPOSITE_BOUNDARY_INVALID")
    require(composite.get("exact_40x24_plan_gate") == "NOT_STARTED" and composite.get("formal_m4") == "NOT_AUTHORIZED", "COMPOSITE_DOWNSTREAM_BOUNDARY_INVALID")
    require(attempts.get("schema") == "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1" and attempts.get("status") == "FROZEN_EXACT55_CORRIDOR_ATTEMPT_FIREWALL", "ATTEMPT_ARTIFACT_INVALID")
    attempt_rows = attempts.get("attempted_identities")
    require(isinstance(attempt_rows, list) and len(attempt_rows) == 55 and len({str(row.get("canonical_parent_key")) for row in attempt_rows}) == 55, "ATTEMPT_EXACT55_INVALID")
    require(attempts.get("origin_counts") == {"V2_CURRENT_SOURCE_FINAL40": 40, "POST_HOLD_V1_ENGINEERING_INVALID": 3, "POST_HOLD_V1_1": 12}, "ATTEMPT_ORIGINS_INVALID")
    require(attempts.get("outcomes_read") is False and attempts.get("protected_counters") == COUNTERS, "ATTEMPT_BOUNDARY_INVALID")

    terminal = terminal_map(load_json(args.terminal_report.resolve()))
    inventory = load_json(args.predecessor_inventory.resolve())
    invalid = load_json(args.invalid_hold.resolve())
    old_registry = load_json(args.old_attempt_registry.resolve())
    runtime = load_json(args.runtime_reconciliation.resolve())
    candidate = load_json(args.candidate_manifest.resolve())
    outer = load_json(args.outer_protocol.resolve())
    old_split = load_json(args.old_split.resolve())
    require(sha256_file(args.terminal_report.resolve()) == "866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9", "V2_HOLD_SHA_INVALID")
    require(sha256_file(args.invalid_hold.resolve()) == "2e2863ff9c84c2b004df32d258dadc9dba320b2ff9a6ed42f0873d6bc36843e0", "V1_INVALID_HOLD_SHA_INVALID")
    require(sha256_file(args.old_attempt_registry.resolve()) == "7d5cfd1b3396f6af4ecd6f3de9b9d6ef454bb927c14a6619a90f14b27a273968", "OLD_ATTEMPT_REGISTRY_SHA_INVALID")
    require(sha256_file(args.runtime_reconciliation.resolve()) == "6fce4411d737f16be7e01b76475d714bbef28006c51e667dec93afca22191a6a", "RUNTIME_SHA_INVALID")
    require(sha256_file(args.predecessor_inventory.resolve()) == "4020a3f45efefb704c7110fd20e243adf84594a2c0920c413ee44928284baf14", "INVENTORY_SHA_INVALID")
    require(sha256_file(args.old_split.resolve()) == "f76ababf0750a78ee7adb3c81e0d15b945275d7210e2d0e41de1a623c6549cc4", "OLD_SPLIT_SHA_INVALID")
    require(outer.get("source_binding", {}).get("science_commit") == SOURCE_COMMIT and outer.get("source_binding", {}).get("science_tree") == SOURCE_TREE and outer.get("source_binding", {}).get("corridor_runner_sha256") == RUNNER_SHA256, "OUTER_SOURCE_INVALID")
    require(runtime.get("schema") == "STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION_V1" and runtime.get("status") == "PASS_POST_HOLD_CORRIDOR_TARGETS_REACHED" and runtime.get("sealed") is True and runtime.get("outcomes_read") is False and runtime.get("intervention_executed") is False and runtime.get("protected_counters") == COUNTERS, "RUNTIME_BOUNDARY_INVALID")
    require(candidate.get("schema") == "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1" and candidate.get("candidate_count") == 22 and candidate.get("status") == "FROZEN", "CANDIDATE_SOURCE_INVALID")
    candidate_map = {str(row.get("canonical_parent_key")): row for row in candidate.get("parents", [])}
    require(len(candidate_map) == 22, "CANDIDATE_KEY_SET_INVALID")
    require(invalid.get("status") == "HOLD_ENGINEERING_INVALID_PRELAUNCH_GOVERNANCE_INCOMPLETE" and invalid.get("consumable") is False and len(invalid.get("attempted_identities", [])) == 3, "INVALID_V1_SOURCE_INVALID")

    old_keys = set(terminal)
    invalid_keys = {str(row.get("canonical_parent_key")) for row in invalid.get("attempted_identities", [])}
    runtime_pairs = runtime.get("pairs")
    require(isinstance(runtime_pairs, list) and len(runtime_pairs) == 12, "RUNTIME_PAIR_COUNT_INVALID")
    runtime_map = {str(row.get("canonical_parent_key")): row for row in runtime_pairs}
    require(len(runtime_map) == 12 and not (old_keys | invalid_keys) & set(runtime_map), "ATTEMPT_UNION_OVERLAP")
    attempt_map = {str(row.get("canonical_parent_key")): row for row in attempt_rows}
    require(set(attempt_map) == old_keys | invalid_keys | set(runtime_map), "ATTEMPT_UNION_KEY_SET_INVALID")
    for key in old_keys:
        require(attempt_map[key].get("origin") == "V2_CURRENT_SOURCE_FINAL40", f"ATTEMPT_OLD_ORIGIN_INVALID:{key}")
        require(attempt_map[key].get("A_receipt_sha256") == terminal[key]["A"].get("receipt_sha256") and attempt_map[key].get("B_receipt_sha256") == terminal[key]["B"].get("receipt_sha256"), f"ATTEMPT_OLD_RECEIPT_BINDING_INVALID:{key}")
    for key in invalid_keys:
        require(attempt_map[key].get("origin") == "POST_HOLD_V1_ENGINEERING_INVALID", f"ATTEMPT_INVALID_ORIGIN_INVALID:{key}")
    for key, pair in runtime_map.items():
        require(attempt_map[key].get("origin") == "POST_HOLD_V1_1" and attempt_map[key].get("runtime_reconciliation_sha256") == sha256_file(args.runtime_reconciliation.resolve()), f"ATTEMPT_NEW_BINDING_INVALID:{key}")
    for pair in runtime_pairs:
        key, suite = str(pair["canonical_parent_key"]), str(pair["suite"])
        reps = pair["replicates"]
        statuses = []
        for rep in ("A", "B"):
            item = reps[rep]
            data = receipt(Path(str(item["receipt"])), str(item["receipt_sha256"]), key, rep, str(item["status"]))
            require(data.get("trajectory_sha256") == item.get("trajectory_sha256"), f"RUNTIME_TRAJECTORY_INVALID:{key}:{rep}")
            statuses.append(str(item["status"]))
        require(pair.get("status_pair") == "/".join(statuses), f"RUNTIME_PAIR_STATUS_INVALID:{key}")
        require(pair.get("stable_pass_pass") is (pair.get("status_pair") == "PASS/PASS"), f"RUNTIME_STABLE_FLAG_INVALID:{key}")
        require(pair.get("outcomes_read") is False and pair.get("protected_counters") == COUNTERS, f"RUNTIME_PAIR_BOUNDARY_INVALID:{key}")
    for suite in QUALIFICATION_SUITES:
        rows = sorted((row for row in runtime_pairs if row["suite"] == suite), key=lambda row: int(row["selection_rank"]))
        require([int(row["selection_rank"]) for row in rows] == list(range(1, len(rows) + 1)), f"RUNTIME_RANK_INVALID:{suite}")
        require([row["canonical_parent_key"] for row in rows] == outer["qualification"]["queues"][suite][: len(rows)], f"RUNTIME_QUEUE_INVALID:{suite}")
        selected = [row["canonical_parent_key"] for row in rows if row.get("stable_pass_pass")]
        require(selected == EXPECTED_NEW[suite], f"RUNTIME_SELECTED_INVALID:{suite}")

    inventory_rows = inventory.get("parents")
    require(isinstance(inventory_rows, list) and len(inventory_rows) == 32, "INVENTORY_PARENT_COUNT_INVALID")
    predecessor_keys = {str(row["canonical_parent_key"]) for row in inventory_rows}
    require(len(predecessor_keys) == 32 and predecessor_keys.issubset(old_keys), "INVENTORY_KEY_SET_INVALID")
    for item in inventory_rows:
        key = str(item["canonical_parent_key"])
        require(item.get("status_pair") == "PASS/PASS", f"INVENTORY_STATUS_INVALID:{key}")
        for rep in ("A", "B"):
            require(item[rep].get("path") == terminal[key][rep].get("path") and item[rep].get("receipt_sha256") == terminal[key][rep].get("receipt_sha256"), f"INVENTORY_RECEIPT_BINDING_INVALID:{key}:{rep}")
    old_split_rows = old_split.get("parents")
    old_split_map = {str(row["canonical_parent_key"]): row for row in old_split_rows}
    require(len(old_split_map) == 40, "OLD_SPLIT_PARENT_COUNT_INVALID")
    final_rows = manifest.get("parents")
    require(manifest.get("schema") == "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" and manifest.get("status") == "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" and manifest.get("formal_m4_authorized") is False, "FINAL_MANIFEST_STATUS_INVALID")
    require(isinstance(final_rows, list) and len(final_rows) == 40 and len({str(row.get("canonical_parent_key")) for row in final_rows}) == 40, "FINAL_MANIFEST_IDENTITY_INVALID")
    final_map = {str(row["canonical_parent_key"]): row for row in final_rows}
    require(set(final_map) == predecessor_keys | set(EXPECTED_NEW["libero_10"] + EXPECTED_NEW["libero_goal"] + EXPECTED_NEW["libero_spatial"]), "FINAL40_KEY_SET_INVALID")
    for row in final_rows:
        key, suite, task, state = parse_key(str(row["canonical_parent_key"]))
        require(row.get("suite") == suite and row.get("task_index") == task and row.get("state_index") == state and row.get("status_pair") == "PASS/PASS", f"FINAL_ROW_ID_INVALID:{key}")
        for rep in ("A", "B"):
            path = Path(str(row[f"{rep}_receipt_path"]))
            data = receipt(path, str(row[f"{rep}_receipt_sha256"]), key, rep, "PASS")
            require(data.get("trajectory_sha256") == row.get(f"{rep}_trajectory_sha256"), f"FINAL_TRAJECTORY_INVALID:{key}:{rep}")
        if row.get("origin") == "V2_PREDECESSOR_PASS_PASS":
            require(key in predecessor_keys and row.get("split") == old_split_map[key].get("split"), f"PREDECESSOR_SLOT_CHANGED:{key}")
        else:
            require(row.get("origin") == "POST_HOLD_V1_1_PASS_PASS" and key in runtime_map and row.get("split") in {"TRAIN", "VAL", "TEST"}, f"NEW_FINAL_ROW_INVALID:{key}")
            candidate_row = candidate_map.get(key)
            require(candidate_row is not None and candidate_row.get("taxonomy_status") == "SUPPORTED", f"NEW_CANDIDATE_MISSING:{key}")
            require(row.get("v7_candidate_sha256") == candidate_row.get("v7_candidate_sha256") and row.get("qualification_rank_sha256") == candidate_row.get("qualification_rank_sha256"), f"NEW_CANDIDATE_BINDING_INVALID:{key}")
    require({suite: sum(row["suite"] == suite for row in final_rows) for suite in SUITES} == {suite: 10 for suite in SUITES}, "FINAL_SUITE_COUNTS_INVALID")
    require({split_name: sum(row["split"] == split_name for row in final_rows) for split_name in ("TRAIN", "VAL", "TEST")} == {"TRAIN": 24, "VAL": 8, "TEST": 8}, "FINAL_SPLIT_COUNTS_INVALID")
    require(all({split_name: sum(row["suite"] == suite and row["split"] == split_name for row in final_rows) for split_name in ("TRAIN", "VAL", "TEST")} == {"TRAIN": 6, "VAL": 2, "TEST": 2} for suite in SUITES), "FINAL_PER_SUITE_SPLIT_INVALID")

    split_rows = split.get("parents")
    require(split.get("schema") == "STAGE_V_M4_FINAL_PARENT_SPLIT_V2" and split.get("status") == "FROZEN" and isinstance(split_rows, list) and len(split_rows) == 40, "FINAL_SPLIT_ARTIFACT_INVALID")
    split_map = {str(row["canonical_parent_key"]): row for row in split_rows}
    require(len(split_map) == 40 and split.get("final_manifest_sha256") == sha256_file(artifacts["manifest"]), "FINAL_SPLIT_MANIFEST_BINDING_INVALID")
    require(all(split_map[key].get("split") == row.get("split") and split_map[key].get("origin") == row.get("origin") for key, row in final_map.items()), "FINAL_SPLIT_ROW_BINDING_INVALID")
    require(composite.get("final_manifest_sha256") == sha256_file(artifacts["manifest"]) and composite.get("final_split_sha256") == sha256_file(artifacts["split"]) and composite.get("attempt_registry_sha256") == sha256_file(artifacts["attempts"]), "COMPOSITE_OUTPUT_HASH_BINDING_INVALID")
    return {
        "source_receipts_reverified": True,
        "exact55_reverified": True,
        "final40_reverified": True,
        "final_split_reverified": True,
        "queue_prefix_reverified": True,
        "protected_boundary_reverified": True,
    }


def finalize_root(root: Path) -> None:
    sums = root / "SHA256SUMS"
    root_seal = root / "ROOT_SEAL.sha256"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.sha256"})
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    root_seal.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    expected = root_seal.read_text(encoding="utf-8").strip()
    require(expected == f"{sha256_file(sums)}  SHA256SUMS", "ROOT_SEAL_INVALID")
    for raw in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = raw.split("  ", 1)
        require(sha256_file(root / relative) == digest, f"ROOT_FILE_HASH_INVALID:{relative}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("terminal-report", "old-attempt-registry", "invalid-hold", "runtime-reconciliation", "predecessor-inventory", "candidate-manifest", "outer-protocol", "old-split"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    require(root.is_dir(), f"OUTPUT_ROOT_MISSING:{root}")
    require(not (root / "SHA256SUMS").exists() and not (root / "ROOT_SEAL.sha256").exists(), "OUTPUT_ROOT_ALREADY_SEALED")
    audit_path = root / "STAGE_V_M4_POST_HOLD_COMPOSITE_INDEPENDENT_AUDIT_V1.json"
    errors: list[str] = []
    checks: dict[str, bool] = {}
    try:
        checks = audit_all(root, args)
    except Exception as exc:
        errors.append(str(exc))
    status = "PASS_INDEPENDENT_EXACT55_FINAL40_SPLIT" if not errors else "HOLD_SEALED"
    audit = {
        "schema": "STAGE_V_M4_POST_HOLD_COMPOSITE_INDEPENDENT_AUDIT_V1",
        "status": status,
        "sealed": True,
        "independent_of_producer": True,
        "checks": checks,
        "failure_count": len(errors),
        "errors": errors,
        "final40": "FROZEN" if not errors else "NOT_FROZEN",
        "final_split": "FROZEN" if not errors else "NOT_FROZEN",
        "exact55_firewall": "FROZEN" if not errors else "NOT_FROZEN",
        "root_seal": "REQUIRED_AFTER_AUDIT_WRITE",
        "exact_40x24_plan_gate": "NOT_STARTED",
        "formal_m4": "NOT_AUTHORIZED",
        "outcomes_read": False,
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
        "next_action": "STOP_BEFORE_EXACT_40X24" if not errors else "HOLD_SEALED_NO_RETRY_NO_RESERVE",
    }
    write_json(audit_path, audit)
    finalize_root(root)
    print(json.dumps({"status": status, "output_root": str(root), "errors": errors, "root_sealed": True}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
