"""Producer-free audit for the Stage VI-B2 zero-treatment plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, source_commit: str, source_tree: str) -> dict[str, Any]:
    errors: list[str] = []
    try:
        protocol = load(root / "B2_PLAN_PROTOCOL.json")
        selection = load(root / "B2_SELECTION_MANIFEST.json")
        split = load(root / "B2_PARENT_SPLIT.json")
        registry = load(root / "PARENT_RUN_REGISTRY.json")
        manifest = load(root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    except Exception as exc:
        errors.append(f"INPUT:{type(exc).__name__}:{exc}")
        protocol = selection = split = registry = manifest = {}
    if protocol.get("schema") != "STAGE_VI_B2_ZERO_TREATMENT_PLAN_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("source_binding") != {"runtime_commit": source_commit, "runtime_tree": source_tree}: errors.append("PROTOCOL")
    if protocol.get("operation", {}).get("intervention_executed") is not False or protocol.get("operation", {}).get("outcomes_read") is not False or protocol.get("operation", {}).get("fresh_render_primary_consumption") != "HARD_STOP": errors.append("PROTOCOL_BOUNDARY")
    if protocol.get("protected_counters") != COUNTERS: errors.append("PROTOCOL_COUNTERS")
    parents = selection.get("selected_parents") if isinstance(selection.get("selected_parents"), list) else []
    keys = [str(row.get("canonical_parent_key")) for row in parents if isinstance(row, Mapping)]
    if selection.get("status") != "FROZEN_BEFORE_PLAN_RUNTIME" or len(keys) != 16 or len(set(keys)) != 16 or selection.get("outcomes_read") is not False or selection.get("protected_counters") != COUNTERS: errors.append("SELECTION")
    split_keys = [str(row.get("canonical_parent_key")) for row in split.get("parents", []) if isinstance(row, Mapping)]
    if split.get("schema") != "STAGE_VI_B2_FORMAL_PARENT_SPLIT_V1" or split.get("status") != "FROZEN" or split.get("parent_count") != 16 or split.get("counts") != {"TRAIN": 0, "VAL": 0, "TEST": 16} or split_keys != keys or split.get("formal_m4_authorized") is not False or split.get("outcomes_read") is not False: errors.append("SPLIT")
    rows = registry.get("results") if isinstance(registry.get("results"), list) else []
    if registry.get("parent_count") != 16 or len(rows) != 16 or any(row.get("return_code") != 0 or row.get("outcomes_read") is not False or row.get("intervention_executed") is not False or row.get("protected_counters") != COUNTERS for row in rows): errors.append("RUN_REGISTRY")
    if manifest.get("schema") != "STAGE_VI_B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1" or manifest.get("downstream_source") is None or manifest.get("downstream_source", {}).get("commit") != source_commit or manifest.get("downstream_source", {}).get("tree") != source_tree or manifest.get("parent_count") != 16 or manifest.get("probe_count_per_parent") != 24 or manifest.get("probe_count_total") != 384 or manifest.get("planned_branch_authority_count") != 1536 or manifest.get("planned_branch_authority_expected") != 1536: errors.append("EXACT_MANIFEST_HEADER")
    if manifest.get("selection_outcomes_read") is not False or manifest.get("intervention_executed") is not False or manifest.get("v_phys_generated") is not False or manifest.get("protected_counters") != COUNTERS: errors.append("EXACT_MANIFEST_BOUNDARY")
    manifest_parents = manifest.get("parents") if isinstance(manifest.get("parents"), list) else []
    if len(manifest_parents) != 16 or [str(row.get("canonical_parent_key")) for row in manifest_parents] != keys: errors.append("EXACT_PARENT_KEYS")
    probe_count = branch_count = 0
    for row in manifest_parents:
        if row.get("status") != "PASS" or row.get("probe_count") != 24 or row.get("outcomes_read") is not False or row.get("intervention_executed") is not False: errors.append(f"PARENT:{row.get('canonical_parent_key')}")
        parent_root = root / str(row.get("output_dir", ""))
        receipt_path, audit_path = parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json", parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
        receipt, gate_audit = load(receipt_path), load(audit_path)
        if receipt.get("status") != "PASS" or receipt.get("snapshot_count") != 24 or receipt.get("intervention_executed") is not False or receipt.get("outcomes_read") is not False or receipt.get("protected_counters") != COUNTERS or gate_audit.get("status") != "PASS": errors.append(f"GATE_A:{row.get('canonical_parent_key')}")
        if (parent_root / "M4_V_PHYS_LABELS_V1.jsonl").exists() or (parent_root / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl").exists(): errors.append(f"INTERVENTION_ARTIFACT:{row.get('canonical_parent_key')}")
        probe_count += int(row.get("probe_count", 0))
    branch_count = len(manifest.get("branch_authorities", [])) if isinstance(manifest.get("branch_authorities"), list) else 0
    if probe_count != 384 or branch_count != 1536: errors.append("ACCOUNTING")
    status = "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" if not errors else "HOLD_SEALED_STAGE_VI_B2_ZERO_TREATMENT_PLAN"
    result = {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PLAN_INDEPENDENT_AUDIT_V1", "status": status, "root": str(root), "source_commit": source_commit, "source_tree": source_tree, "parent_count": len(manifest_parents), "probe_count": probe_count, "planned_branch_authority_count": branch_count, "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS, "errors": sorted(set(errors))}
    (root / "B2_PLAN_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    try:
        result = audit(args.root.resolve(), args.source_commit, args.source_tree)
    except Exception as exc:
        result = {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PLAN_INDEPENDENT_AUDIT_V1", "status": "HOLD_SEALED_STAGE_VI_B2_ZERO_TREATMENT_PLAN", "errors": [f"AUDITOR:{type(exc).__name__}:{exc}"], "protected_counters": COUNTERS}
        args.root.resolve().mkdir(parents=True, exist_ok=True)
        (args.root.resolve() / "B2_PLAN_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
