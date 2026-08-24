#!/usr/bin/env python3
"""Freeze and seal the single authorized F1-C4 fresh canary namespace."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
NAMESPACE = "STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822"
OUT = ROOT / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822"
CLASSIFICATION = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_EXPOSURE_CLASSIFICATION_V3.json"
SPLIT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_SPLIT_LEDGER_V3.json"
F1A3_ROOT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json"
F1C_METHOD = ROOT / "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_METHOD_SPEC_V3.json"
F1C_CONFIG = ROOT / "configs/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_PROTOCOL_V3.json"
F1C_REPAIR = ROOT / "reports/STAGE_X_X1R2_F1C_REPAIR_STATIC_AUDIT_V1_20260822/F1C_REPAIR_STATIC_AUDIT_V1.json"
F1B_DECISION = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json"
F1B_RESULT_ROOT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json"
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r2_f1c_t5_canary.py"
DEV_RUNNER = ROOT / "scripts/stage_x/run_stage_x1r2_f1b_dev.py"
SUITE_CONTRACT = ROOT / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"
PROTOCOL_PATH = ROOT / "configs/STAGE_X_X1R2_F1C4_FRESH_CANARY_PROTOCOL_V1.json"
SALT = NAMESPACE
POST_A3_SOURCE_COMMIT = "86c5c26f193a243051b30385d24d6c45abba9a96"
POST_A3_SCAN_DIRS = (
    ROOT / "f1c_remote_receipts_20260822",
    ROOT / "remote_f1c_official_reports_20260821",
    ROOT / "remote_f1c_official_reports_r4_20260821",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_receipt() -> dict[str, Any]:
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("show", "-s", "--format=%T", "HEAD"),
        "status_porcelain": git("status", "--porcelain", "--untracked-files=no"),
    }


def current_post_a3_paths() -> list[Path]:
    names = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--name-only", f"{POST_A3_SOURCE_COMMIT}..HEAD"],
        text=True,
    ).splitlines()
    source_authority_prefix = "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/"
    return [
        ROOT / name
        for name in names
        if (ROOT / name).is_file() and not name.startswith(source_authority_prefix)
    ]


def scan_for_keys(keys: list[str], paths: list[Path]) -> dict[str, list[str]]:
    hits = {key: [] for key in keys}
    unique_paths = sorted({path.resolve() for path in paths if path.is_file()})
    for path in unique_paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            continue
        for key in keys:
            if key.encode("utf-8") in data:
                hits[key].append(rel(path))
    return hits


def method_contract(method_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method_spec["method"],
        "temporal_arms": method_spec["temporal_arms"],
        "temporal_selection": method_spec["temporal_selection"],
        "execution": method_spec["execution"],
        "method_fields": method_spec["method"],
        "probe_fields": {
            key: value for key, value in method_spec["probe"].items() if key != "selection_salt"
        },
    }


def main() -> int:
    if OUT.exists() or PROTOCOL_PATH.exists():
        raise SystemExit("F1C4_OUTPUT_ALREADY_EXISTS")
    source = source_receipt()
    if source["status_porcelain"]:
        raise SystemExit(f"F1C4_STATIC_WORKTREE_NOT_CLEAN:{source['status_porcelain']}")

    classification = load(CLASSIFICATION)
    split = load(SPLIT)
    old_method_spec = load(F1C_METHOD)
    old_protocol = load(F1C_CONFIG)
    repair = load(F1C_REPAIR)
    old_roles = {str(row["canonical_parent_key"]) for row in split["rows"]}
    rows = list(classification["rows"])

    candidate_rows = [
        row for row in rows
        if row.get("source_class") == "V3_PRISTINE_NO_RELEVANT_IDENTITY_EXPOSURE"
        and row.get("eligible_for_v3_split") is True
        and row.get("canonical_parent_key") not in old_roles
        and row.get("protected_or_eval160_read") is False
        and row.get("historical_outcome_values_read") is False
        and row.get("attack_physical_runtime_read") is False
    ]
    ranked = sorted(
        candidate_rows,
        key=lambda row: sha256_bytes(f"{SALT}|{row['canonical_parent_key']}".encode("utf-8")),
    )
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        suite_rows = [row for row in ranked if row["suite"] == suite]
        selected.extend(suite_rows[:2])
    selected = sorted(selected, key=lambda row: (SUITES.index(row["suite"]), sha256_bytes(f"{SALT}|{row['canonical_parent_key']}".encode("utf-8"))))
    selected_keys = [str(row["canonical_parent_key"]) for row in selected]

    scan_paths = current_post_a3_paths()
    for directory in POST_A3_SCAN_DIRS:
        if directory.is_dir():
            scan_paths.extend(path for path in directory.rglob("*") if path.is_file())
    current_hits = scan_for_keys(selected_keys, scan_paths)
    if any(current_hits.values()):
        raise SystemExit(f"F1C4_CURRENT_IDENTITY_HIT:{current_hits}")

    counts = {suite: sum(row["suite"] == suite for row in selected) for suite in SUITES}
    if len(selected) != 8 or counts != {suite: 2 for suite in SUITES}:
        raise SystemExit(f"F1C4_COUNTS_INVALID:{counts}")
    if old_roles.intersection(selected_keys):
        raise SystemExit("F1C4_ROLE_INTERSECTION_NONZERO")

    ledger_rows = []
    for ordinal, row in enumerate(selected, start=1):
        key = str(row["canonical_parent_key"])
        suite, task, state = key.split("/")
        ledger_rows.append({
            "canonical_parent_key": key,
            "contamination_tier": "PRISTINE",
            "fixture_id": f"F1C4_{suite}_{task}_{state}",
            "identity_only_at_freeze": True,
            "outcome_read": False,
            "permanent_exclusion": True,
            "rank_hash": sha256_bytes(f"{SALT}|{key}".encode("utf-8")),
            "role": "F1C4_FRESH_CANARY_V1",
            "role_ordinal": ordinal,
            "source_class": row["source_class"],
            "source_domain": "F1C4_FRESH_CANARY_NAMESPACE_V1",
            "suite": suite,
            "task": task,
            "state": state,
            "runtime_read": False,
            "protected_or_eval160_read": False,
            "attack_physical_runtime_read": False,
            "historical_outcome_values_read": False,
        })

    ledger = {
        "schema": "STAGE_X1R2_F1C4_FRESH_CANARY_LEDGER_V1",
        "status": "PASS_F1C4_FRESH_NAMESPACE_FREEZE",
        "gate": "STAGE_X_X1R2_F1C4_FRESH_CANARY_REQUALIFICATION_V1",
        "namespace": NAMESPACE,
        "source_authority": {
            "f1a3_classification_path": rel(CLASSIFICATION),
            "f1a3_classification_sha256": sha256_file(CLASSIFICATION),
            "f1a3_split_path": rel(SPLIT),
            "f1a3_split_sha256": sha256_file(SPLIT),
            "f1a3_root_path": rel(F1A3_ROOT),
            "f1a3_root_sha256": sha256_file(F1A3_ROOT),
            "source_commit_at_freeze": source["commit"],
            "source_tree_at_freeze": source["tree"],
            "post_a3_source_commit": POST_A3_SOURCE_COMMIT,
            "post_a3_scan_paths": [rel(path) for path in scan_paths],
        },
        "selection": {
            "salt": SALT,
            "rank_rule": "sha256(namespace_salt|canonical_parent_key)",
            "candidate_class": "V3_PRISTINE_NO_RELEVANT_IDENTITY_EXPOSURE",
            "no_detector_train_only_when_pristine_capacity_exists": True,
            "no_replacement": True,
            "no_top_up": True,
        },
        "row_count": len(ledger_rows),
        "per_suite_count": counts,
        "role_intersections": {"DEV_V3": 0, "C_CANARY_V3": 0, "BRIDGE_V3": 0},
        "rows": ledger_rows,
        "protected_boundary": {
            "bridge_runtime": 0,
            "bridge_outcome_read": 0,
            "eval160": "UNREAD",
            "protected": "UNREAD",
            "vphys": 0,
            "physical_outcome": 0,
        },
    }

    exposure = {
        "schema": "STAGE_X1R2_F1C4_EXPOSURE_REAUDIT_V1",
        "status": "PASS_F1C4_FRESH_NAMESPACE_EXPOSURE_REAUDIT",
        "namespace": NAMESPACE,
        "source_class_counts": classification["class_counts"],
        "source_universe_row_count": len(rows),
        "old_f1a3_role_union_count": len(old_roles),
        "fresh_candidate_pool_by_suite": {suite: sum(row["suite"] == suite for row in candidate_rows) for suite in SUITES},
        "selected_by_suite": counts,
        "selected_current_head_identity_hits": current_hits,
        "current_head_scan": {
            "base_commit": POST_A3_SOURCE_COMMIT,
            "head_commit": source["commit"],
            "path_count": len(scan_paths),
            "outcome_values_not_read": True,
        },
        "intersection_audit": {"DEV_V3": 0, "C_CANARY_V3": 0, "BRIDGE_V3": 0},
        "protected_boundary": ledger["protected_boundary"],
    }

    new_method = method_contract(old_method_spec)
    method_audit = {
        "schema": "STAGE_X1R2_F1C4_METHOD_EQUIVALENCE_AUDIT_V1",
        "status": "PASS_F1C4_FROZEN_METHOD_EQUIVALENCE",
        "namespace": NAMESPACE,
        "historical_f1c_method_spec_sha256": sha256_file(F1C_METHOD),
        "historical_f1c_protocol_sha256": old_method_spec["protocol_sha256"],
        "f1b_decision_sha256": sha256_file(F1B_DECISION),
        "selected_method": "M1",
        "method_contract": new_method,
        "required_f1c4_method": {
            "method": "M1",
            "objective": "autoregressive_prefix_gripper_native_open_logratio_v4",
            "iterations": 10,
            "epsilon_processor_pixel_values": 0.03,
            "step_size": 0.003,
            "random_start": False,
            "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
            "candidate_order": ["delta0", *(f"pgd_iteration_{i}" for i in range(1, 11))],
            "target_execution_class": "NATIVE_OPEN",
            "target_token_id_secondary": 31745,
            "exact_arm_dimensions": [0, 1, 2, 3, 4, 5],
            "direct_action_token_count": 7,
            "strict_route": True,
            "allow_fallback": False,
            "no_decode_reencode": True,
            "no_actuator_overwrite": True,
        },
        "semantic_changes": [],
        "repair_binding": {
            "repair_audit_path": rel(F1C_REPAIR),
            "repair_audit_sha256": sha256_file(F1C_REPAIR),
            "repair_status": repair["qualification_status_unchanged"],
        },
        "current_source": {
            "commit": source["commit"],
            "tree": source["tree"],
            "runner_sha256": sha256_file(RUNNER),
            "dev_runner_sha256": sha256_file(DEV_RUNNER),
        },
        "protected_boundary": ledger["protected_boundary"],
    }
    if method_audit["method_contract"]["method_fields"] != method_audit["required_f1c4_method"]:
        raise SystemExit("F1C4_METHOD_EQUIVALENCE_FAIL")

    protocol = {
        "schema": "STAGE_X1R2_F1C4_FRESH_CANARY_PROTOCOL_V1",
        "status": "FROZEN_F1C4_T5_CANARY_V1",
        "scientific_authority": False,
        "gate": "STAGE_X_X1R2_F1C4_FRESH_CANARY_REQUALIFICATION_V1",
        "population": {
            "role": "F1C4_FRESH_CANARY_V1",
            "path": rel(OUT / "F1C4_FRESH_CANARY_LEDGER_V1.json"),
            "row_count": 8,
            "per_suite_count": 2,
            "f1a3_root_seal_path": rel(F1A3_ROOT),
            "f1a3_root_seal_sha256": sha256_file(F1A3_ROOT),
            "canary_ledger_sha256": "PENDING",
            "source_split_salt": split["split_salt"],
        },
        "method": old_method_spec["method"],
        "temporal_arms": old_method_spec["temporal_arms"],
        "temporal_selection": old_method_spec["temporal_selection"],
        "probe": {
            **old_method_spec["probe"],
            "selection_salt": "STAGE_X1R2_F1C4_PROBE_V1_20260822",
        },
        "execution": old_method_spec["execution"],
        "upstream": {
            "f1b_decision_path": rel(F1B_DECISION),
            "f1b_decision_sha256": sha256_file(F1B_DECISION),
            "f1b_result_root_path": rel(F1B_RESULT_ROOT),
            "f1b_result_root_sha256": sha256_file(F1B_RESULT_ROOT),
            "f1b_selected_method": "M1",
            "f1b_selected_iterations": 10,
            "historical_f1c_method_spec_path": rel(F1C_METHOD),
            "historical_f1c_method_spec_sha256": sha256_file(F1C_METHOD),
        },
        "freeze": {
            "method_spec_path": rel(OUT / "F1C4_METHOD_SPEC_V1.json"),
            "pre_gpu_audit_path": rel(OUT / "F1C4_PRE_GPU_AUDIT_V1.json"),
            "root_seal_path": rel(OUT / "F1C4_ROOT_SEAL_V1.json"),
            "root_sidecar_path": rel(OUT / "F1C4_ROOT_SEAL_V1.sha256"),
        },
        "validation": {
            "protocol_statuses": ["FROZEN_F1C4_T5_CANARY_V1"],
            "ledger_status": "PASS_F1C4_FRESH_NAMESPACE_FREEZE",
            "method_spec_status": "PASS_F1C4_METHOD_SPEC_SEALED",
            "pre_gpu_status": "PASS_F1C4_PRE_GPU_STATIC_CONTRACT",
            "root_seal_status": "PASS_F1C4_PRE_GPU_STATIC_CONTRACT",
        },
        "runtime": {
            "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "durable_output_root": "/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_f1c4_fresh_canary_v1_20260822",
            "attack_seed_salt": "STAGE_X1R2_F1C4_ATTACK_V1_20260822",
            "source_receipt": "bind_current_exact_commit_tree_and_runtime_file_blobs_before_first_model_load",
        },
        "resource": {
            "free_memory_mib_strictly_greater_than": 20480,
            "max_project_workers": 8,
            "one_project_worker_per_physical_gpu": True,
            "foreign_processes_untouched": True,
        },
        "protected_boundary": ledger["protected_boundary"] | {
            "attack_outcome_reads": 0,
            "attacked_env_steps": 0,
            "physical_interventions": 0,
        },
    }

    # Write the ledger first, then bind its digest into the protocol.
    OUT.mkdir(parents=True, exist_ok=False)
    ledger_path = OUT / "F1C4_FRESH_CANARY_LEDGER_V1.json"
    exposure_path = OUT / "F1C4_EXPOSURE_REAUDIT_V1.json"
    method_audit_path = OUT / "F1C4_METHOD_EQUIVALENCE_AUDIT_V1.json"
    write(ledger_path, ledger)
    protocol["population"]["canary_ledger_sha256"] = sha256_file(ledger_path)
    write(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    method_spec = {
        "schema": "STAGE_X1R2_F1C4_METHOD_SPEC_V1",
        "status": "PASS_F1C4_METHOD_SPEC_SEALED",
        "protocol_sha256": protocol_sha,
        "method": old_method_spec["method"],
        "temporal_arms": old_method_spec["temporal_arms"],
        "temporal_selection": old_method_spec["temporal_selection"],
        "execution": old_method_spec["execution"],
        "probe": {**old_method_spec["probe"], "selection_salt": protocol["probe"]["selection_salt"]},
        "population": protocol["population"],
        "equivalent_to_historical_f1c_method_spec_sha256": sha256_file(F1C_METHOD),
        "source": source,
        "protected_boundary": protocol["protected_boundary"],
    }
    method_spec_path = OUT / "F1C4_METHOD_SPEC_V1.json"
    write(method_spec_path, method_spec)
    method_audit["protocol_sha256"] = protocol_sha
    method_audit["method_spec_sha256"] = sha256_file(method_spec_path)
    write(exposure_path, exposure)
    write(method_audit_path, method_audit)

    pre_gpu = {
        "schema": "STAGE_X1R2_F1C4_PRE_GPU_AUDIT_V1",
        "status": "PASS_F1C4_PRE_GPU_STATIC_CONTRACT",
        "gate": protocol["gate"],
        "protocol_path": rel(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "ledger_path": rel(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "exposure_reaudit_path": rel(exposure_path),
        "exposure_reaudit_sha256": sha256_file(exposure_path),
        "method_equivalence_path": rel(method_audit_path),
        "method_equivalence_sha256": sha256_file(method_audit_path),
        "method_spec_path": rel(method_spec_path),
        "method_spec_sha256": sha256_file(method_spec_path),
        "source": source,
        "counts": counts,
        "intersections": ledger["role_intersections"],
        "gpu": 0,
        "model_inference": 0,
        "simulator": 0,
        "pgd": 0,
        "attacked_env_steps": 0,
        "physical_interventions": 0,
        "vphys": 0,
        "bridge_runtime": 0,
        "bridge_outcome_read": 0,
        "eval160": "UNREAD",
        "protected": "UNREAD",
        "protected_boundary": protocol["protected_boundary"],
    }
    pre_gpu_path = OUT / "F1C4_PRE_GPU_AUDIT_V1.json"
    write(pre_gpu_path, pre_gpu)

    artifact_paths = [
        rel(PROTOCOL_PATH), rel(ledger_path), rel(exposure_path), rel(method_audit_path), rel(method_spec_path), rel(pre_gpu_path),
        rel(F1A3_ROOT), rel(F1C_METHOD), rel(F1C_REPAIR), rel(F1B_DECISION), rel(F1B_RESULT_ROOT),
        rel(RUNNER), rel(DEV_RUNNER), rel(SUITE_CONTRACT),
    ]
    seal = {
        "schema": "STAGE_X1R2_F1C4_ROOT_SEAL_V1",
        "status": "PASS_F1C4_PRE_GPU_STATIC_CONTRACT",
        "gate": protocol["gate"],
        "namespace": NAMESPACE,
        "protocol_sha256": protocol_sha,
        "method_spec_sha256": sha256_file(method_spec_path),
        "pre_gpu_audit_sha256": sha256_file(pre_gpu_path),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "artifact_hashes": {path: sha256_file(ROOT / path) for path in artifact_paths},
        "role_counts": counts,
        "role_intersections": ledger["role_intersections"],
        "protected_boundary": protocol["protected_boundary"],
        "seal_scope_excludes_sidecar": True,
    }
    root_path = OUT / "F1C4_ROOT_SEAL_V1.json"
    write(root_path, seal)
    (OUT / "F1C4_ROOT_SEAL_V1.sha256").write_text(f"{sha256_file(root_path)}  {root_path.name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": seal["status"], "namespace": NAMESPACE, "counts": counts, "protocol_sha256": protocol_sha, "root_sha256": sha256_file(root_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
