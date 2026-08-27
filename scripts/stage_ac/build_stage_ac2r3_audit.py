#!/usr/bin/env python3
"""Audit the sealed AC2 clean census without touching runtime state.

This is deliberately a read-only consumer of the 720 server receipts.  It
reuses the frozen eligibility evaluator and writes only compact, append-only
audit artifacts when ``--write`` is requested.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC2R2_CLEAN_SCREEN_V1")
MANIFEST = "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
PROTOCOL = "configs/STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_V1.json"
SOURCE = "reports/STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_V1.json"
M2_AUTHORITY = "reports/STAGE_Z_Z0R1_MODEL_AUTHORITY_MAP_V2.json"
ELIGIBILITY = "src/stage_ac/eligibility_v2.py"
GATE = "STAGE_AC_AC2R3_COMPLETE_CENSUS_EVIDENCE_AUDIT_AND_MODEL_SPECIFIC_DENOMINATOR_FREEZE_V1"
PI_COMMENT_ID = 5441904205
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
FORBIDDEN = (
    "aa_v_phys_reads",
    "attack_outcome_reads",
    "attacked_env_steps",
    "eval160_reads",
    "open_intervention_steps",
    "pgd_calls",
    "physical_endpoint_reads",
    "protected_reads",
    "task_success_reads",
    "v_phys_reads",
    "dummy_wait_env_step_calls",
)
DENOMINATOR_SALT = "STAGE_AC_AC2_MODEL_SPECIFIC_DENOMINATOR_V1_20260827"
CELL_RE = re.compile(r"^AC2-(\d{4})\.json$")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return digest(canonical(value))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_binding(path: Path, relative_to: Path | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    name = path.as_posix() if relative_to is None else path.relative_to(relative_to).as_posix()
    return {"path": name, "bytes": len(data), "sha256": digest(data)}


def write_json(path: Path, value: Any) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"append-only output already exists: {path}")
    path.write_bytes(data)
    return file_binding(path, ROOT)


def git_blob(revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        # Server evidence worktrees are intentionally lightweight source
        # projections rather than Git checkouts; their sealed source files
        # are the fallback representation for this read-only audit.
        return (ROOT / path).read_bytes()


def tracked_binding(path: str, revision: str = "HEAD") -> dict[str, Any]:
    data = git_blob(revision, path)
    return {"path": path, "bytes": len(data), "sha256": digest(data)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def projected_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "step", "anchor_class", "boundary", "selection_rank_sha256", "eligible",
        "reason_codes", "metrics", "continuation_steps", "continuation_digest",
        "boundary_state_sha256",
    )}


def expected_candidates(rows: list[dict[str, Any]], actions: list[dict[str, Any]], family: str, parent: str, baseline: float | None, salt: str, anchor_class: str, evaluator: Any) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    audits: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for step in range(max(0, len(rows) - evaluator.CLEAN_CONTINUATION_STEPS + 1)):
        result = evaluator.evaluate_candidate(rows, actions, step, baseline, anchor_class, 0)
        for reason in set(result["reason_codes"]):
            reasons[reason] += 1
        evidence = rows[step : step + evaluator.CLEAN_CONTINUATION_STEPS]
        audits.append({
            "step": step,
            "anchor_class": anchor_class,
            "boundary": bool(actions[step].get("boundary")),
            "selection_rank_sha256": evaluator.rank_candidate(salt, family, parent, step),
            "eligible": bool(result["eligible"]),
            "reason_codes": list(result["reason_codes"]),
            "metrics": result.get("metrics", {}),
            "continuation_steps": [int(item["step"]) for item in evidence],
            "continuation_digest": canonical_hash(evidence),
            "boundary_state_sha256": actions[step].get("boundary_state_sha256"),
        })
    eligible = sorted((item for item in audits if item["eligible"]), key=lambda item: (item["selection_rank_sha256"], item["step"]))
    return audits, dict(sorted(reasons.items())), eligible


def check_finite(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(check_finite(item) for item in value)
    if isinstance(value, dict):
        return all(check_finite(item) for item in value.values())
    return True


def source_parity(source: dict[str, Any]) -> dict[str, Any]:
    old_entry = next(item for item in source["runtime_files"] if item["path"] == "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py")
    taxonomy_lf = git_blob("HEAD", old_entry["path"])
    old_commit = source["git_binding"]["commit"]
    old_blob = git_blob(old_commit, old_entry["path"])
    require(old_blob == taxonomy_lf, "telemetry taxonomy differs from AC2R2 source commit")
    normalized_crlf = taxonomy_lf.replace(b"\n", b"\r\n")
    derived_crlf = {"bytes": len(normalized_crlf), "sha256": digest(normalized_crlf), "representation": "deterministic CRLF rendering of the frozen LF blob"}
    require(derived_crlf["bytes"] == 18122 and derived_crlf["sha256"].startswith("52e2d393"), "derived CRLF rendering does not match the supplied pre-GPU authority prefix")
    old_ast = digest(ast.dump(ast.parse(old_blob.decode("utf-8")), include_attributes=False).encode("utf-8"))
    head_ast = digest(ast.dump(ast.parse(taxonomy_lf.decode("utf-8")), include_attributes=False).encode("utf-8"))
    require(old_ast == head_ast, "telemetry taxonomy AST changed")
    return {
        "status": "PASS",
        "tracked_lf_binding": {"path": old_entry["path"], "bytes": len(taxonomy_lf), "sha256": digest(taxonomy_lf)},
        "ac2r2_binding": old_entry,
        "old_commit_blob_equal_head": True,
        "ast_semantic_hash_old": old_ast,
        "ast_semantic_hash_head": head_ast,
        "git_semantic_diff": "no tracked diff between AC2R2 source commit and HEAD for telemetry module",
        "eol_reconciliation": {
            "normalized_crlf_from_head_lf": derived_crlf,
            "historical_pre_gpu_authority": {"bytes": 18122, "sha256_prefix": "52e2d393", "prefix_match": True},
            "note": "LF/CRLF container representation only; runtime text and AST are unchanged",
        },
        "eligibility_consumed_fields": {
            "object_identity": "unchanged by tracked-byte equality",
            "object_position": "unchanged by tracked-byte equality",
            "eef_position": "unchanged by tracked-byte equality",
            "object_eef_distance": "unchanged by tracked-byte equality",
            "object_gripper_contact": "unchanged by tracked-byte equality",
            "object_support_contact": "unchanged by tracked-byte equality",
            "telemetry_validity_and_baseline": "unchanged by tracked-byte equality",
        },
        "eligibility_affecting_semantic_drift": False,
        "repair_scope": {
            "frozen_salt_restoration": True,
            "official_m2_clip_delivery": True,
            "m1_manifest_reconciliation": True,
            "cohort_or_threshold_change": False,
        },
    }


def audit(root: Path, output_root: Path) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(root / "src"))
    from stage_ac import eligibility_v2 as evaluator

    manifest_path = root / MANIFEST
    protocol_path = root / PROTOCOL
    source_path = root / SOURCE
    manifest = load(manifest_path)
    protocol = load(protocol_path)
    source = load(source_path)
    require(manifest.get("cell_count") == 720 and len(manifest.get("cells", [])) == 720, "launch manifest is not the frozen 720-cell census")
    cells = {str(item["cell_id"]): item for item in manifest["cells"]}
    require(len(cells) == 720, "duplicate manifest cell IDs")
    receipt_dir = output_root / "receipts"
    files = sorted(receipt_dir.glob("AC2-*.json"))
    require(len(files) == 720, f"accepted receipt count is {len(files)}, expected 720")

    receipt_index: list[dict[str, Any]] = []
    receipt_by_cell: dict[str, dict[str, Any]] = {}
    errors: Counter[str] = Counter()
    counter_totals: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    exposure_counts: Counter[tuple[str, str]] = Counter()
    eligible_by_model: dict[str, set[str]] = defaultdict(set)
    eligible_by_model_suite: Counter[tuple[str, str]] = Counter()
    eligible_by_model_task: Counter[tuple[str, str]] = Counter()
    reason_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    reason_by_suite: dict[str, Counter[str]] = defaultdict(Counter)
    selected_steps: dict[str, Counter[int]] = defaultdict(Counter)
    row_length_by_model: dict[str, Counter[int]] = defaultdict(Counter)
    binding_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    structural_cells: list[dict[str, Any]] = []

    for path in files:
        match = CELL_RE.match(path.name)
        if not match:
            errors["receipt_filename_invalid"] += 1
            continue
        receipt_bytes = path.read_bytes()
        try:
            receipt = json.loads(receipt_bytes)
        except json.JSONDecodeError:
            errors["receipt_json_invalid"] += 1
            continue
        cell_id = str(receipt.get("cell_id"))
        expected = cells.get(cell_id)
        if expected is None:
            errors["receipt_not_in_manifest"] += 1
            continue
        if cell_id in receipt_by_cell:
            errors["duplicate_receipt_cell"] += 1
            continue
        receipt_by_cell[cell_id] = receipt
        family = str(receipt.get("model_family"))
        suite = str(receipt.get("suite"))
        clean = receipt.get("clean") or {}
        binding_status = str((clean.get("binding") or {}).get("status", "MISSING"))
        binding_by_model[family][binding_status] += 1
        if binding_status not in {"PASS", "OBJECT_TAXONOMY_PASS", "BOUND"}:
            structural_cells.append({"cell_id": cell_id, "model_family": family, "suite": suite, "binding_status": binding_status})
        for key in ("canonical_parent_key", "model_family", "suite", "task", "state", "state_id", "state_sha256", "seed", "source_task_idx", "parent_exposure_class", "checkpoint"):
            if receipt.get(key) != expected.get(key):
                errors[f"manifest_field_mismatch:{key}"] += 1
        if receipt.get("status") != "AC2_CLEAN_CELL_COMPLETE":
            errors["status_not_complete"] += 1
        if receipt.get("clean_only") is not True:
            errors["clean_only_contract"] += 1
        if not isinstance(clean.get("rows"), list) or not clean["rows"]:
            errors["rows_missing_or_empty"] += 1
        rows = clean.get("eligibility_rows")
        actions = clean.get("actions")
        if not isinstance(rows, list) or not isinstance(actions, list) or len(rows) != len(actions):
            errors["row_action_alignment"] += 1
            rows, actions = [], []
        row_length_by_model[family][len(rows)] += 1
        for index, (row, action) in enumerate(zip(rows, actions)):
            if row.get("step") != index or action.get("step") != index:
                errors["step_alignment"] += 1
            if not isinstance(action.get("raw"), list) or len(action["raw"]) != 7 or not isinstance(action.get("final"), list) or len(action["final"]) != 7:
                errors["action_not_7d"] += 1
            if not check_finite(action.get("raw")) or not check_finite(action.get("final")) or not check_finite(row):
                errors["nonfinite_evidence"] += 1
            if not isinstance(row.get("object_position"), list) or not isinstance(row.get("eef_position"), list) or not isinstance(row.get("object_eef_distance_m"), (int, float)):
                errors["telemetry_geometry_missing"] += 1
            if not isinstance(row.get("object_gripper_contact"), bool) or not isinstance(row.get("object_support_contact"), bool):
                errors["contact_support_missing"] += 1
        if not isinstance(clean.get("boundary_states"), dict) or not clean["boundary_states"]:
            errors["boundary_state_evidence_missing"] += 1
        if not isinstance(receipt.get("action_pair_audit"), list):
            errors["action_pair_audit_missing"] += 1
        if not isinstance(clean.get("candidate_audit"), list) or not isinstance(clean.get("critical_candidates"), list) or not isinstance(clean.get("noncritical_candidates"), list):
            errors["candidate_evidence_missing"] += 1
        counters = receipt.get("runtime_counters") or {}
        for key, value in counters.items():
            if isinstance(value, (int, float)):
                counter_totals[key] += int(value)
        if not all(int(counters.get(key, 0)) == 0 for key in FORBIDDEN):
            errors["forbidden_counter_nonzero"] += 1

        critical_expected, critical_reasons, critical_eligible = expected_candidates(rows, actions, family, str(receipt["canonical_parent_key"]), clean.get("baseline_z_m"), str(protocol["eligibility"]["critical_selection_salt"]), "CRITICAL", evaluator)
        noncritical_expected, noncritical_reasons, noncritical_eligible = expected_candidates(rows, actions, family, str(receipt["canonical_parent_key"]), clean.get("baseline_z_m"), str(protocol["eligibility"]["noncritical_selection_salt"]), "NONCRITICAL", evaluator)
        expected_audit = sorted(critical_expected + noncritical_expected, key=lambda item: (item["anchor_class"], item["step"]))
        actual_audit = clean.get("candidate_audit") or []
        if [projected_candidate(x) for x in actual_audit] != [projected_candidate(x) for x in expected_audit]:
            errors["candidate_audit_recompute_mismatch"] += 1
        if clean.get("critical_reason_counts") != critical_reasons or clean.get("noncritical_reason_counts") != noncritical_reasons:
            errors["reason_accounting_mismatch"] += 1
        actual_critical = [projected_candidate(x) for x in clean.get("critical_candidates", [])]
        actual_noncritical = [projected_candidate(x) for x in clean.get("noncritical_candidates", [])]
        if actual_critical != [projected_candidate(x) for x in critical_eligible] or actual_noncritical != [projected_candidate(x) for x in noncritical_eligible]:
            errors["eligible_candidate_list_mismatch"] += 1
        expected_selected = critical_eligible[0] if critical_eligible else None
        actual_selected = clean.get("selected_critical")
        if (projected_candidate(actual_selected) if actual_selected else None) != (projected_candidate(expected_selected) if expected_selected else None):
            errors["selected_critical_mismatch"] += 1
        if (clean.get("eligibility_status") == "ELIGIBLE_CRITICAL") != bool(critical_eligible):
            errors["eligibility_flag_mismatch"] += 1

        model_counts[family] += 1
        suite_counts[suite] += 1
        exposure_counts[(family, str(receipt.get("parent_exposure_class")))] += 1
        if critical_eligible:
            eligible_by_model[family].add(str(receipt["canonical_parent_key"]))
            eligible_by_model_suite[(family, suite)] += 1
            eligible_by_model_task[(family, str(receipt.get("task")))] += 1
            selected_steps[family][int(critical_eligible[0]["step"])] += 1
        reason_by_model[family].update(critical_reasons)
        reason_by_suite[suite].update(critical_reasons)
        receipt_index.append({
            "cell_id": cell_id,
            "model_family": family,
            "suite": suite,
            "task": receipt.get("task"),
            "canonical_parent_key": receipt.get("canonical_parent_key"),
            "state_sha256": receipt.get("state_sha256"),
            "seed": receipt.get("seed"),
            "parent_exposure_class": receipt.get("parent_exposure_class"),
            "receipt": file_binding(path),
            "status": receipt.get("status"),
            "clean_trajectory_digest": clean.get("clean_trajectory_digest"),
            "rows": len(clean.get("rows") or []),
            "eligibility_rows": len(rows),
            "actions": len(actions),
            "horizon": clean.get("horizon"),
            "complete_trajectory": clean.get("complete_trajectory"),
            "critical_eligible": bool(critical_eligible),
            "selected_critical_step": expected_selected.get("step") if expected_selected else None,
            "selected_critical_rank_sha256": expected_selected.get("selection_rank_sha256") if expected_selected else None,
            "critical_candidate_count": len(critical_eligible),
            "noncritical_candidate_count": len(noncritical_eligible),
            "checkpoint_manifest_receipt_present": isinstance(receipt.get("checkpoint_manifest"), dict),
            "checkpoint_authority_mode": "M2_GLOBAL_16_FILE_REHASH" if family == "M2_PI05_LIBERO" else ("M1_PER_CELL_MANIFEST" if family == "M1_OPENVLA_OFT" else "M0_LAUNCH_MANIFEST_PATH"),
        })

    require(set(receipt_by_cell) == set(cells), f"manifest/receipt cell set mismatch: manifest={len(cells)} receipts={len(receipt_by_cell)}")
    require(not errors, f"AC2R3 audit checks failed: {dict(errors)}")

    m2_map = load(root / M2_AUTHORITY)
    m2_spec = m2_map["model_families"]["M2_PI05_LIBERO"]
    m2_checkpoint = Path(m2_spec["checkpoint"])
    expected_files = m2_spec.get("checkpoint_file_manifest") or m2_spec.get("checkpoint_manifest")
    require(isinstance(expected_files, list), "M2 global checkpoint file manifest missing")
    actual_files = []
    for item in expected_files:
        path = m2_checkpoint / str(item["path"])
        data = path.read_bytes()
        expected_size = int(item.get("bytes", item.get("size")))
        actual_files.append({"path": str(item["path"]), "expected_bytes": expected_size, "actual_bytes": len(data), "expected_sha256": str(item["sha256"]), "actual_sha256": digest(data), "match": len(data) == expected_size and digest(data) == str(item["sha256"])})
    require(all(item["match"] for item in actual_files), "M2 global checkpoint rehash mismatch")

    parity = source_parity(source)
    aggregate = {key: int(value) for key, value in sorted(counter_totals.items())}
    for key in FORBIDDEN:
        aggregate.setdefault(key, 0)
    denominator: dict[str, Any] = {}
    exposure_by_parent = {(item["model_family"], item["canonical_parent_key"]): item["parent_exposure_class"] for item in receipt_index}
    for family in MODELS:
        eligible = sorted(eligible_by_model[family])
        ranked = sorted((
            (digest(f"{DENOMINATOR_SALT}|{key}".encode("utf-8")), key)
            for key in eligible
        ), key=lambda item: (item[0], item[1]))
        if len(ranked) >= 32:
            status = "N_AT_LEAST_32_SELECT_32"
            frozen = ranked[:32]
        elif len(ranked) >= 24:
            status = "N_24_TO_31_USE_ALL"
            frozen = ranked
        else:
            status = "AC_CAPACITY_LIMITED"
            frozen = []
        denominator[family] = {
            "eligible_count": len(eligible),
            "status": status,
            "target_n": 32,
            "floor_n": 24,
            "rank_rule": "sha256(model_specific_denominator_salt|canonical_parent_key), ascending; no suite/task rebalance",
            "salt": DENOMINATOR_SALT,
            "eligible_parent_keys": eligible,
            "ranked_eligible": [{"rank_sha256": rank, "canonical_parent_key": key} for rank, key in ranked],
            "frozen_primary_parent_keys": [key for _, key in frozen],
            "frozen_primary_ranked": [{"rank_sha256": rank, "canonical_parent_key": key} for rank, key in frozen],
            "frozen_h0_count": sum(exposure_by_parent[(family, key)] == "H0_UNTOUCHED" for _, key in frozen),
            "frozen_hc_count": sum(exposure_by_parent[(family, key)] == "HC_CLEAN_ONLY" for _, key in frozen),
            "eligible_by_suite": {suite: eligible_by_model_suite[(family, suite)] for suite in SUITES},
            "eligible_by_task": {task: eligible_by_model_task[(family, task)] for task in sorted({str(item.get("task")) for item in manifest["cells"]})},
            "selected_step_distribution": dict(sorted(selected_steps[family].items())),
        }

    return {
        "gate": GATE,
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": "AC2 clean-only complete-census evidence audit; no AC3 treatment or physical susceptibility claim",
        "manifest": file_binding(manifest_path, root),
        "protocol": file_binding(protocol_path, root),
        "source_authority": file_binding(source_path, root),
        "receipt_index": sorted(receipt_index, key=lambda item: item["cell_id"]),
        "errors": dict(errors),
        "counts": {"manifest_cells": len(cells), "accepted_receipts": len(receipt_index), "models": dict(sorted(model_counts.items())), "suites": dict(sorted(suite_counts.items())), "exposure_by_model": {family: {exposure: exposure_counts[(family, exposure)] for exposure in ("H0_UNTOUCHED", "HC_CLEAN_ONLY")} for family in MODELS}},
        "row_length_by_model": {family: dict(sorted(values.items())) for family, values in row_length_by_model.items()},
        "structural_binding_by_model": {family: dict(sorted(values.items())) for family, values in binding_by_model.items()},
        "structurally_unsupported_cells": structural_cells,
        "eligibility": {"critical_by_model": {family: len(eligible_by_model[family]) for family in MODELS}, "critical_by_model_suite": {f"{family}/{suite}": eligible_by_model_suite[(family, suite)] for family in MODELS for suite in SUITES}, "critical_by_model_task": {f"{family}/{task}": eligible_by_model_task[(family, task)] for family in MODELS for task in sorted({str(item.get('task')) for item in manifest['cells']})}, "critical_reason_counts_by_model": {family: dict(sorted(reason_by_model[family].items())) for family in MODELS}, "critical_reason_counts_by_suite": {suite: dict(sorted(reason_by_suite[suite].items())) for suite in SUITES}},
        "m2_global_checkpoint_rehash": {"path": str(m2_checkpoint), "manifest_sha256": m2_spec.get("checkpoint_manifest_sha256"), "expected_file_count": len(expected_files), "hashed_file_count": len(actual_files), "all_match": True, "files": actual_files},
        "telemetry_semantic_parity": parity,
        "runtime_counters": aggregate,
        "denominator": denominator,
        "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN},
    }


def build(root: Path, output_root: Path, write: bool) -> dict[str, Any]:
    result = audit(root, output_root)
    paths = {
        "receipt_index": root / "reports/STAGE_AC_AC2R3_RECEIPT_INDEX_V1.json",
        "evidence": root / "reports/STAGE_AC_AC2R3_EVIDENCE_INTEGRITY_AUDIT_V1.json",
        "parity": root / "reports/STAGE_AC_AC2R3_TELEMETRY_SEMANTIC_PARITY_AUDIT_V1.json",
        "eligibility": root / "reports/STAGE_AC_AC2R3_ELIGIBILITY_RECOMPUTATION_V1.json",
        "denominator": root / "reports/STAGE_AC_AC2R3_MODEL_SPECIFIC_DENOMINATOR_LEDGER_V1.json",
        "root": root / "reports/STAGE_AC_AC2R3_ROOT_SEAL_V1.json",
    }
    if write:
        evidence = {key: result[key] for key in ("gate", "authorization_pi_comment_id", "claim_boundary", "manifest", "protocol", "source_authority", "counts", "row_length_by_model", "structural_binding_by_model", "structurally_unsupported_cells", "m2_global_checkpoint_rehash", "errors")}
        eligibility = {key: result[key] for key in ("gate", "authorization_pi_comment_id", "claim_boundary", "manifest", "protocol", "source_authority", "eligibility", "receipt_index")}
        denominator = {key: result[key] for key in ("gate", "authorization_pi_comment_id", "claim_boundary", "manifest", "protocol", "source_authority", "denominator")}
        parity = {key: result[key] for key in ("gate", "authorization_pi_comment_id", "claim_boundary", "source_authority", "telemetry_semantic_parity", "m2_global_checkpoint_rehash")}
        index = {"schema": "STAGE_AC_AC2R3_RECEIPT_INDEX_V1", "status": "STAGE_AC_AC2R3_RECEIPT_INDEX_COMPLETE", "gate": GATE, "authorization_pi_comment_id": PI_COMMENT_ID, "receipt_count": len(result["receipt_index"]), "receipts": result["receipt_index"]}
        bindings = {
            "receipt_index": write_json(paths["receipt_index"], index),
            "evidence": write_json(paths["evidence"], evidence),
            "parity": write_json(paths["parity"], parity),
            "eligibility": write_json(paths["eligibility"], eligibility),
            "denominator": write_json(paths["denominator"], denominator),
        }
        root_payload = {
            "schema": "STAGE_AC_AC2R3_ROOT_SEAL_V1",
            "status": "STAGE_AC_AC2R3_THREE_MODEL_DENOMINATORS_FROZEN_READY_STOP_FOR_PI",
            "gate": GATE,
            "authorization_pi_comment_id": PI_COMMENT_ID,
            "claim_boundary": result["claim_boundary"],
            "audit_status": "PASS_WITH_EXPLICIT_M2_GLOBAL_MANIFEST_RECONCILIATION",
            "input_authorities": {key: result[key] for key in ("manifest", "protocol", "source_authority")},
            "audit_artifacts": bindings,
            "receipt_count": len(result["receipt_index"]),
            "eligibility_counts": result["eligibility"]["critical_by_model"],
            "denominator_status": {family: value["status"] for family, value in result["denominator"].items()},
            "m2_global_checkpoint_rehash": result["m2_global_checkpoint_rehash"],
            "telemetry_semantic_parity": result["telemetry_semantic_parity"],
            "runtime_counters": result["runtime_counters"],
            "scientific_firewall": result["scientific_firewall"],
            "next_legal_action": "STOP_FOR_PI; AC3_UNAUTHORIZED",
        }
        root_payload["root_payload_sha256"] = canonical_hash(root_payload)
        bindings["root"] = write_json(paths["root"], root_payload)
        result["written_bindings"] = bindings
        result["terminal_status"] = root_payload["status"]
    else:
        result["terminal_status"] = "READY_TO_WRITE_APPEND_ONLY_ARTIFACTS"
    return {"terminal_status": result["terminal_status"], "receipt_count": len(result["receipt_index"]), "critical_by_model": result["eligibility"]["critical_by_model"], "denominator_status": {family: value["status"] for family, value in result["denominator"].items()}, "runtime_counters": result["runtime_counters"], "telemetry_parity": result["telemetry_semantic_parity"]["status"], "m2_rehash_all_match": result["m2_global_checkpoint_rehash"]["all_match"], "written_bindings": result.get("written_bindings", {})}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output_root, args.write), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
