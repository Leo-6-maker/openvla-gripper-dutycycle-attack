#!/usr/bin/env python3
"""Seal the AA2R2 Phase-B 324-cell census and, when possible, AA2 common N."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
RECOVERY_IDS = {"AA2-0004", "AA2-0085", "AA2-0233", "AA2-0117"}
RECOVERY_STATUSES = {"RUNNING", "AA2_ENGINEERING_HOLD_RUNTIME_ERROR"}
FORBIDDEN_COUNTERS = (
    "open_intervention_steps",
    "attacked_env_steps",
    "pgd_calls",
    "aa_v_phys_reads",
    "attack_outcome_reads",
    "task_success_reads",
    "eval160_reads",
    "protected_reads",
)
OUTPUTS = (
    "reports/STAGE_AA_AA2R2_PHASE_B_RECEIPT_INDEX_V1.json",
    "reports/STAGE_AA_AA2R2_PHASE_B_CENSUS_TERMINAL_V1.json",
    "reports/STAGE_AA_AA2R2_COMMON_DENOMINATOR_V1.json",
    "reports/STAGE_AA_AA2R2_PHASE_B_ROOT_SEAL_V1.json",
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def bool_field(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise RuntimeError(f"BOOLEAN_FIELD_INVALID:{field}:{value!r}")


def strip_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_evidence(item) for key, item in value.items() if key != "evidence_rows"}
    if isinstance(value, list):
        return [strip_evidence(item) for item in value]
    return value


def rank_parent(salt: str, parent_key: str) -> str:
    return sha256_bytes(f"{salt}|{parent_key}".encode("utf-8"))


def validate_authority(root: Path, manifest: dict[str, Any], protocol: dict[str, Any], source: dict[str, Any], capacity: dict[str, Any], aa0: dict[str, Any]) -> None:
    require(manifest.get("schema") == "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1", "MANIFEST_SCHEMA")
    require(manifest.get("status") == "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE", "MANIFEST_STATUS")
    require(manifest.get("cell_count") == 324 and len(manifest.get("cells", [])) == 324, "MANIFEST_COUNT")
    require(protocol.get("schema") == "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1", "PROTOCOL_SCHEMA")
    require(protocol.get("status") == "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE", "PROTOCOL_STATUS")
    require(protocol.get("clean_only") is True and protocol.get("open_intervention_allowed") is False, "PROTOCOL_CLEAN_ONLY")
    require(protocol.get("attack_or_pgd_allowed") is False and protocol.get("protected_or_eval160_allowed") is False, "PROTOCOL_FIREWALL")
    require(source.get("status") == "STAGE_AA_AA2R2_PHASE_B_RUNTIME_SOURCE_AUTHORITY_FROZEN", "PHASE_B_SOURCE_STATUS")
    require(source.get("phase_b", {}).get("authorized") is True and source.get("phase_b", {}).get("scientific_only_clean") is True, "PHASE_B_AUTHORITY")
    require(aa0.get("status") == "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI", "AA0_STATUS")
    pool = set(capacity["analysis_pool_after_aa1_reservation"]["keys"])
    require(len(pool) == 108, "AA2_POOL_COUNT")
    require(source.get("original_manifest_sha256") == sha256_file(root / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"), "SOURCE_MANIFEST_BINDING")
    require(source.get("original_cell_count") == 324, "SOURCE_CELL_COUNT")
    for name, binding in source.get("versioned_runtime_files", {}).items():
        path = Path(str(binding["path"]))
        if not path.is_absolute():
            path = root / path
        require(path.is_file(), f"SOURCE_FILE_MISSING:{name}")
        require(path.stat().st_size == int(binding["bytes"]), f"SOURCE_FILE_BYTES:{name}")
        require(sha256_file(path) == binding.get("sha256"), f"SOURCE_FILE_SHA:{name}")


def resolve_receipts(root: Path, manifest: dict[str, Any], protocol: dict[str, Any], capacity: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    old_dir = root / "reports/server_evidence/STAGE_AA_AA2/receipts"
    phase_dir = root / "reports/server_evidence/STAGE_AA_AA2R2/phase_b"
    normal_dir = phase_dir / "receipts"
    recovery_dir = phase_dir / "recovery"
    pool = set(capacity["analysis_pool_after_aa1_reservation"]["keys"])
    manifest_cells = manifest["cells"]
    require(len({row["cell_id"] for row in manifest_cells}) == 324, "MANIFEST_CELL_IDS")
    old_ids = set()
    expected_normal: set[str] = set()
    expected_recovery: set[str] = set()
    receipt_index: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for cell in manifest_cells:
        cell_id = cell["cell_id"]
        old_path = old_dir / f"{cell_id}.json"
        old = load_json(old_path) if old_path.is_file() else None
        if old is not None:
            old_ids.add(cell_id)
            old_status = old.get("status")
            if old_status == "AA2_CLEAN_CELL_COMPLETE":
                source_kind = "HISTORICAL_AA2_COMPLETE"
                path = old_path
                require(not (normal_dir / f"{cell_id}.json").exists(), f"HISTORICAL_RERUN_NORMAL:{cell_id}")
                require(not (recovery_dir / f"{cell_id}.recovery.json").exists(), f"HISTORICAL_RERUN_RECOVERY:{cell_id}")
            elif old_status in RECOVERY_STATUSES:
                require(cell_id in RECOVERY_IDS, f"UNAUTHORIZED_RECOVERY:{cell_id}:{old_status}")
                source_kind = "PHASE_B_RECOVERY"
                expected_recovery.add(cell_id)
                path = recovery_dir / f"{cell_id}.recovery.json"
            else:
                raise RuntimeError(f"UNEXPECTED_HISTORICAL_STATUS:{cell_id}:{old_status}")
        else:
            source_kind = "PHASE_B_NORMAL"
            expected_normal.add(cell_id)
            path = normal_dir / f"{cell_id}.json"
        require(path.is_file(), f"RECEIPT_MISSING:{cell_id}:{path}")
        receipt = load_json(path)
        require(receipt.get("status") in {"AA2_CLEAN_CELL_COMPLETE", "AA2R2_PHASE_B_CLEAN_CELL_COMPLETE"}, f"RECEIPT_NOT_COMPLETE:{cell_id}:{receipt.get('status')}")
        for field in ("cell_id", "model_family", "suite", "canonical_parent_key", "seed"):
            require(receipt.get(field) == cell.get(field), f"RECEIPT_BINDING:{cell_id}:{field}")
        if source_kind == "HISTORICAL_AA2_COMPLETE":
            require(receipt.get("gate") == protocol["gate"], f"RECEIPT_GATE:{cell_id}")
        else:
            require(receipt.get("gate") == source["gate"], f"RECEIPT_GATE:{cell_id}")
        require(receipt.get("clean_only") is True, f"RECEIPT_NOT_CLEAN_ONLY:{cell_id}")
        clean = receipt.get("clean", {})
        require(clean.get("status") in {"PASS_AA2_CLEAN_TRAJECTORY_CAPTURED", "PASS_AA2R2_CLEAN_TRAJECTORY_CAPTURED"}, f"RECEIPT_CLEAN_STATUS:{cell_id}")
        complete = bool_field(clean.get("complete_trajectory"), f"complete_trajectory:{cell_id}")
        steps = int(clean.get("steps_captured", -1))
        horizon = int(clean.get("horizon", -2))
        require(0 < steps <= horizon, f"RECEIPT_HORIZON:{cell_id}")
        require(int(clean.get("telemetry_valid_rows", -1)) == steps, f"RECEIPT_TELEMETRY_ROWS:{cell_id}")
        eligibility = receipt.get("eligibility", {})
        critical = bool_field(eligibility.get("critical"), f"critical:{cell_id}")
        noncritical = bool_field(eligibility.get("noncritical"), f"noncritical:{cell_id}")
        counters = receipt.get("runtime_counters", {})
        for key in FORBIDDEN_COUNTERS:
            require(key in counters and int(counters[key]) == 0, f"RECEIPT_FIREWALL:{cell_id}:{key}")
        require(cell.get("canonical_parent_key") in pool, f"RECEIPT_PARENT_POOL:{cell_id}")
        if source_kind == "PHASE_B_RECOVERY":
            require(receipt.get("attempt_kind") == "RECOVERY" and receipt.get("recovery_of") == cell_id, f"RECOVERY_BINDING:{cell_id}")
        elif source_kind == "PHASE_B_NORMAL":
            require(receipt.get("attempt_kind") == "NORMAL", f"NORMAL_BINDING:{cell_id}")
        else:
            require(receipt.get("status") == "AA2_CLEAN_CELL_COMPLETE", f"HISTORICAL_STATUS:{cell_id}")
        source_counts[source_kind] += 1
        receipts[cell_id] = receipt
        receipt_index.append({
            "cell_id": cell_id,
            "model_family": cell["model_family"],
            "suite": cell["suite"],
            "canonical_parent_key": cell["canonical_parent_key"],
            "seed": cell["seed"],
            "source_kind": source_kind,
            "path": rel(path, root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "status": receipt["status"],
            "critical_eligible": critical,
            "noncritical_eligible": noncritical,
            "runtime_counters": counters,
            "clean_trajectory_sha256": clean.get("clean_trajectory_digest"),
            "complete_trajectory": complete,
            "steps_captured": steps,
            "horizon": horizon,
        })
    require(source_counts["HISTORICAL_AA2_COMPLETE"] == 32, f"HISTORICAL_COMPLETE_COUNT:{source_counts}")
    require(expected_recovery == RECOVERY_IDS, f"RECOVERY_SET:{sorted(expected_recovery)}")
    require(len(expected_normal) == 288, f"NORMAL_SET_COUNT:{len(expected_normal)}")
    actual_normal = {p.stem for p in normal_dir.glob("AA2-*.json")}
    actual_recovery = {p.name.removesuffix(".recovery.json") for p in recovery_dir.glob("AA2-*.recovery.json")}
    require(actual_normal == expected_normal, f"UNEXPECTED_NORMAL_RECEIPTS:{sorted(actual_normal ^ expected_normal)}")
    require(actual_recovery == expected_recovery, f"UNEXPECTED_RECOVERY_RECEIPTS:{sorted(actual_recovery ^ expected_recovery)}")
    require(len(receipt_index) == 324 and len(receipts) == 324, "FULL_CENSUS_COUNT")
    return receipt_index, receipts, source_counts


def build(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = root / "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    protocol_path = root / "configs/STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_V1.json"
    source_path = root / "reports/STAGE_AA_AA2R2_PHASE_B_RUNTIME_SOURCE_AUTHORITY_V1.json"
    capacity_path = root / "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json"
    aa0_path = root / "configs/STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_V1.json"
    manifest, protocol, source, capacity, aa0 = (load_json(path) for path in (manifest_path, protocol_path, source_path, capacity_path, aa0_path))
    validate_authority(root, manifest, protocol, source, capacity, aa0)
    index, receipts, source_counts = resolve_receipts(root, manifest, protocol, capacity)
    forbidden = Counter()
    by_model: dict[str, dict[str, Any]] = {}
    by_suite: dict[str, dict[str, Any]] = {}
    eligible: dict[str, set[str]] = defaultdict(set)
    parent_matrix: dict[str, dict[str, Any]] = {}
    for item in index:
        family = item["model_family"]
        suite = item["suite"]
        receipt = receipts[item["cell_id"]]
        forbidden.update({key: int(value) for key, value in item["runtime_counters"].items() if key in FORBIDDEN_COUNTERS})
        if item["critical_eligible"]:
            eligible[family].add(item["canonical_parent_key"])
        parent_matrix.setdefault(item["canonical_parent_key"], {"canonical_parent_key": item["canonical_parent_key"]})[family] = {
            "critical_eligible": item["critical_eligible"],
            "noncritical_eligible": item["noncritical_eligible"],
            "critical_anchor": strip_evidence(receipt.get("eligibility", {}).get("critical_anchor")),
            "noncritical_anchor": strip_evidence(receipt.get("eligibility", {}).get("noncritical_anchor")),
            "clean_trajectory_sha256": item["clean_trajectory_sha256"],
            "receipt_sha256": item["sha256"],
            "source_kind": item["source_kind"],
            "critical_reason_counts": receipt.get("clean", {}).get("critical_reason_counts", {}),
        }
    for family in MODELS:
        rows = [item for item in index if item["model_family"] == family]
        by_model[family] = {
            "cell_count": len(rows),
            "critical_eligible_count": sum(item["critical_eligible"] for item in rows),
            "noncritical_eligible_count": sum(item["noncritical_eligible"] for item in rows),
            "eligible_parent_keys": sorted(eligible[family]),
            "source_counts": dict(sorted(Counter(item["source_kind"] for item in rows).items())),
        }
    for suite in SUITES:
        rows = [item for item in index if item["suite"] == suite]
        by_suite[suite] = {
            "cell_count": len(rows),
            "critical_eligible_count": sum(item["critical_eligible"] for item in rows),
            "noncritical_eligible_count": sum(item["noncritical_eligible"] for item in rows),
        }
    salt = protocol["literal_salts"]["common_parent_selection"]
    common = sorted(eligible[MODELS[0]] & eligible[MODELS[1]] & eligible[MODELS[2]])
    ranked = sorted(common, key=lambda key: (rank_parent(salt, key), key))
    if len(common) >= 32:
        frozen = ranked[:32]
        denominator_status = "N_COMMON_AT_LEAST_32_SELECT_32"
    elif len(common) >= 24:
        frozen = ranked
        denominator_status = "N_COMMON_24_TO_31_USE_ALL"
    else:
        frozen = []
        denominator_status = "N_COMMON_BELOW_24_CAPACITY_LIMIT"
    arms = (("CLEAN_REFERENCE", 0, "critical"), ("OPEN_T3_CRITICAL", 3, "critical"), ("OPEN_T5_CRITICAL", 5, "critical"), ("OPEN_T10_CRITICAL", 10, "critical"))
    aa3_jobs = []
    for family in MODELS:
        for parent in frozen:
            model_parent_rank = rank_parent(salt, f"{family}|{parent}")
            for arm, dose, anchor in arms:
                aa3_jobs.append({"branch_id": f"AA3-{model_parent_rank[:20]}-{arm}", "model_family": family, "canonical_parent_key": parent, "arm": arm, "dose": dose, "anchor": anchor})
            if parent_matrix[parent].get(family, {}).get("noncritical_eligible"):
                aa3_jobs.append({"branch_id": f"AA3-{model_parent_rank[:20]}-OPEN_T5_NONCRITICAL_CONTROL", "model_family": family, "canonical_parent_key": parent, "arm": "OPEN_T5_NONCRITICAL_CONTROL", "dose": 5, "anchor": "noncritical_control"})
    require(len({job["branch_id"] for job in aa3_jobs}) == len(aa3_jobs), "AA3_BRANCH_IDS")
    status = "STAGE_AA_AA2_COMMON_DENOMINATOR_FROZEN_AUTONOMOUS_CONTINUE" if len(common) >= 24 else "STAGE_AA_AA2_CAPACITY_LIMIT_STOP_FOR_PI"
    terminal = {
        "schema": "STAGE_AA_AA2R2_PHASE_B_CENSUS_TERMINAL_V1",
        "status": status,
        "gate": protocol["gate"],
        "authorization_pi_comment_id": protocol["authorization_pi_comment_id"],
        "claim_boundary": "AA2R2 clean-only full census and common-denominator freeze; no AA3 outcome",
        "authority": {
            "manifest": {"path": rel(manifest_path, root), "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)},
            "protocol": {"path": rel(protocol_path, root), "bytes": protocol_path.stat().st_size, "sha256": sha256_file(protocol_path)},
            "source": {"path": rel(source_path, root), "bytes": source_path.stat().st_size, "sha256": sha256_file(source_path)},
            "phase_a_root_sha256": source.get("phase_a_root", {}).get("sha256"),
        },
        "census": {"manifest_cell_count": 324, "receipt_count": len(index), "source_counts": dict(sorted(source_counts.items())), "by_model": by_model, "by_suite": by_suite},
        "eligibility_sets": {f"E_{family.split('_', 1)[0]}": sorted(eligible[family]) for family in MODELS},
        "common_denominator": {
            "salt": salt,
            "n_common": len(common),
            "denominator_status": denominator_status,
            "ranked_common": [{"canonical_parent_key": key, "rank_sha256": rank_parent(salt, key)} for key in ranked],
            "frozen_primary_parent_keys": frozen,
            "primary_n": len(frozen),
            "floor_n": 24,
            "target_n": 32,
        },
        "parent_matrix": [parent_matrix[key] for key in sorted(parent_matrix)],
        "aa3_branch_manifest": {"primary_branch_count": 3 * len(frozen) * 4, "secondary_branch_count": len(aa3_jobs) - 3 * len(frozen) * 4, "jobs": aa3_jobs},
        "aggregate_runtime_counters": dict(sorted(forbidden.items())),
        "scientific_firewall": {key: forbidden[key] for key in FORBIDDEN_COUNTERS},
        "next_legal_action": "STAGE_AA_AUTONOMOUS_AA3_EXECUTION" if len(common) >= 24 else "STOP_FOR_PI",
    }
    receipt_index = {
        "schema": "STAGE_AA_AA2R2_PHASE_B_RECEIPT_INDEX_V1",
        "status": status,
        "gate": protocol["gate"],
        "receipt_count": len(index),
        "receipts": sorted(index, key=lambda item: item["cell_id"]),
        "terminal_sha256_after_write": None,
    }
    common_artifact = {
        "schema": "STAGE_AA_AA2R2_COMMON_DENOMINATOR_V1",
        "status": status,
        "source_terminal_schema": terminal["schema"],
        "source_terminal_sha256_after_write": None,
        "n_common": len(common),
        "primary_n": len(frozen),
        "frozen_primary_parent_keys": frozen,
        "per_model_critical_anchors": {key: {family: parent_matrix[key].get(family, {}).get("critical_anchor") for family in MODELS} for key in frozen},
        "per_model_noncritical_anchors": {key: {family: parent_matrix[key].get(family, {}).get("noncritical_anchor") for family in MODELS} for key in frozen},
        "aa3_branch_manifest_sha256_after_write": None,
        "next_legal_action": terminal["next_legal_action"],
    }
    root_seal = {
        "schema": "STAGE_AA_AA2R2_PHASE_B_ROOT_SEAL_V1",
        "status": status,
        "gate": protocol["gate"],
        "authorization_pi_comment_id": protocol["authorization_pi_comment_id"],
        "phase_a_root_seal_sha256": source.get("phase_a_root", {}).get("sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "source_authority_sha256": sha256_file(source_path),
        "receipt_count": len(index),
        "terminal_sha256": None,
        "receipt_index_sha256": None,
        "common_denominator_sha256": None,
        "scientific_firewall": terminal["scientific_firewall"],
        "root_payload_sha256": None,
        "next_legal_action": terminal["next_legal_action"],
    }
    return receipt_index, terminal, common_artifact, root_seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    receipt_index, terminal, common_artifact, root_seal = build(args.root)
    if args.write:
        paths = {name: args.root / name for name in OUTPUTS}
        require(not any(path.exists() for path in paths.values()), "APPEND_ONLY_OUTPUT_ALREADY_EXISTS")
        terminal_path = paths[OUTPUTS[1]]
        index_path = paths[OUTPUTS[0]]
        common_path = paths[OUTPUTS[2]]
        root_path = paths[OUTPUTS[3]]
        write_json(terminal_path, terminal)
        receipt_index["terminal_sha256_after_write"] = sha256_file(terminal_path)
        write_json(index_path, receipt_index)
        common_artifact["source_terminal_sha256_after_write"] = sha256_file(terminal_path)
        common_artifact["aa3_branch_manifest_sha256_after_write"] = sha256_bytes(canonical(terminal["aa3_branch_manifest"]))
        write_json(common_path, common_artifact)
        root_seal["terminal_sha256"] = sha256_file(terminal_path)
        root_seal["receipt_index_sha256"] = sha256_file(index_path)
        root_seal["common_denominator_sha256"] = sha256_file(common_path)
        root_seal["root_payload_sha256"] = sha256_bytes(canonical(root_seal))
        write_json(root_path, root_seal)
        root_path.with_suffix(".sha256").write_text(f"{sha256_file(root_path)}  {root_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": "AA2R2_PHASE_B_CENSUS_BUILD_PASS", "receipt_count": receipt_index["receipt_count"], "n_common": terminal["common_denominator"]["n_common"], "terminal_status": terminal["status"], "write": args.write}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
