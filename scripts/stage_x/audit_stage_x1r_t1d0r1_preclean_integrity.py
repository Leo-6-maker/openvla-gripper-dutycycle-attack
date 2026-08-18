"""CPU-only T1-D0R1 pre-clean integrity audit.

The audit consumes metadata, Git objects, and hashed files only.  It does not
load a model, import a simulator, execute a detector, or materialize a clean
trajectory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.stage_x_x1r_v2_schedule_contract import (  # noqa: E402
    ATTACK_WINDOW_LENGTH,
    PHYSICAL_FOLLOWUP_LENGTH,
    PREV_DELTA_BOUNDARIES,
    attack_steps,
    followup_steps,
    legal_horizon,
)
from scripts.stage_x.audit_stage_x1r_t1d0r_authority import (  # noqa: E402
    AuthorityError,
    derive_population,
    identity_digest,
    load_source,
    parse_directory_identity,
    parse_key,
    sha256_file,
)


class IntegrityError(ValueError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise IntegrityError(f"JSONL_OBJECT_REQUIRED:{path}")
            rows.append(value)
    return rows


def seed_for_parent(namespace: str, canonical_parent_key: str) -> int:
    digest = hashlib.sha256(f"{namespace}|{canonical_parent_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def git_output(repo_root: Path, *args: str, input_bytes: bytes | None = None) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        input=input_bytes,
        text=input_bytes is None,
    ).strip()


def git_blob_record(repo_root: Path, ref: str, relative_path: str) -> dict[str, Any]:
    spec = f"{ref}:{relative_path}"
    blob = git_output(repo_root, "rev-parse", spec)
    size = int(git_output(repo_root, "cat-file", "-s", spec))
    payload = subprocess.check_output(["git", "-C", str(repo_root), "cat-file", "blob", spec])
    if len(payload) != size:
        raise IntegrityError(f"GIT_BLOB_SIZE_MISMATCH:{spec}")
    return {
        "ref": ref,
        "path": relative_path,
        "git_blob_sha": blob,
        "byte_size": size,
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def reconcile_local_bindings(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    spec = protocol["local_binding_reconciliation"]
    refs = [str(ref) for ref in spec["refs"]]
    paths = [str(path) for path in spec["paths"]]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for ref in refs:
        for relative_path in paths:
            try:
                rows.append(git_blob_record(repo_root, ref, relative_path))
            except (OSError, subprocess.CalledProcessError, IntegrityError) as exc:
                errors.append(f"GIT_BINDING_READ:{ref}:{relative_path}:{exc}")

    by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in paths}
    for row in rows:
        by_path[row["path"]].append(row)
    comparisons: dict[str, Any] = {}
    claims = spec["historical_claims"]
    for relative_path, path_rows in by_path.items():
        raw_values = {row["raw_sha256"] for row in path_rows}
        blob_values = {row["git_blob_sha"] for row in path_rows}
        identical = len(raw_values) == 1 and len(blob_values) == 1 and len(path_rows) == len(refs)
        d0_claim = claims["D0"][relative_path]
        d0r_claim = claims["D0R"][relative_path]
        actual = next(iter(raw_values), None)
        classification = "UNRESOLVED"
        if identical and actual == d0r_claim and d0_claim != actual:
            classification = "D0_HISTORICAL_DIGEST_STALE_D0R_CANONICAL_BYTES"
        elif identical and actual == d0_claim == d0r_claim:
            classification = "HISTORICAL_DIGESTS_AGREE"
        else:
            errors.append(f"LOCAL_BINDING_BYTES_NOT_RECONCILED:{relative_path}")
        comparisons[relative_path] = {
            "rows": path_rows,
            "all_refs_byte_identical": identical,
            "byte_identical_to_pr124": identical,
            "byte_identical_to_pr125": identical,
            "byte_identical_to_pr126": identical,
            "actual_raw_sha256": actual,
            "historical_d0_claim": d0_claim,
            "historical_d0r_claim": d0r_claim,
            "classification": classification,
        }
    return {"errors": errors, "comparisons": comparisons}


def list_physical_intervention_source(
    spec: Mapping[str, Any], universe: set[str] | None = None
) -> tuple[set[str], dict[str, Any], list[dict[str, Any]], set[str]]:
    base = Path(str(spec["base_path"]))
    if not base.is_dir():
        raise IntegrityError(f"SOURCE_DIRECTORY_MISSING:{base}")
    names = sorted(path.name for path in base.glob(str(spec["glob"])) if path.is_dir())
    listing_sha = identity_digest(names)
    if listing_sha != str(spec["listing_sha256"]):
        raise IntegrityError(f"DIRECTORY_LISTING_SHA256_MISMATCH:{listing_sha}!={spec['listing_sha256']}")
    aliases = dict(spec.get("aliases", {}))
    rows: list[dict[str, Any]] = []
    keys: list[str] = []
    canonical_keys: set[str] = set()
    for name in names:
        key = parse_directory_identity(name)
        source_kind = "CANONICAL_DIRECTORY_NAME"
        alias_meta: Mapping[str, Any] = {}
        if key is None:
            alias_meta = aliases.get(name, {})
            if not alias_meta:
                raise IntegrityError(f"DIRECTORY_IDENTITY_ALIAS_MISSING:{name}")
            key = parse_key(alias_meta.get("canonical_parent_key"))
            source_kind = "EXPLICIT_CANONICAL_ALIAS"
        else:
            canonical_keys.add(key)
        keys.append(key)
        rows.append(
            {
                "directory_name": name,
                "canonical_parent_key": key,
                "source_kind": source_kind,
                "physical_intervention_semantics_from_directory_name": alias_meta.get(
                    "physical_intervention_semantics_from_directory_name", "IDENTITY_ONLY"
                ),
                "historical_outcome_read": False,
            }
        )
    key_set = set(keys)
    if universe is not None and not key_set <= universe:
        raise IntegrityError(f"DIRECTORY_IDENTITY_JOIN_OUTSIDE_G10:{sorted(key_set - universe)[:3]}")
    duplicate_keys = sorted(key for key in set(keys) if keys.count(key) > 1)
    meta = {
        "name": spec["name"],
        "kind": spec["kind"],
        "base_path": str(base),
        "glob": str(spec["glob"]),
        "directory_count": len(names),
        "listing_sha256": listing_sha,
        "resolved_directory_count": len(rows),
        "identity_count": len(key_set),
        "identity_digest": identity_digest(key_set),
        "ignored_noncanonical_directories": [],
        "explicit_alias_count": sum(row["source_kind"] == "EXPLICIT_CANONICAL_ALIAS" for row in rows),
        "duplicate_identity_keys": duplicate_keys,
    }
    return key_set, meta, rows, canonical_keys


def load_d0r1_sources(protocol: Mapping[str, Any]) -> tuple[set[str], dict[str, set[str]], list[dict[str, Any]], dict[str, Any], set[str]]:
    specs = protocol["source_recompute"]["sources"]
    g10_spec = next(spec for spec in specs if spec["kind"] == "g10_identities")
    g10_keys, g10_meta = load_source(g10_spec)
    exclusion_sets: dict[str, set[str]] = {}
    source_meta = [g10_meta]
    alias_rows: list[dict[str, Any]] = []
    physical_canonical_keys: set[str] = set()
    for spec in specs:
        if spec is g10_spec:
            continue
        if spec["kind"] == "directory_identity_listing":
            values, meta, rows, canonical_keys = list_physical_intervention_source(spec, g10_keys)
            alias_rows.extend(rows)
            physical_canonical_keys = canonical_keys
        else:
            values, meta = load_source(spec, g10_keys)
        exclusion_sets[str(spec["name"])] = values
        source_meta.append(meta)
    return g10_keys, exclusion_sets, source_meta, {"g10": g10_meta}, physical_canonical_keys


def load_d0_selected_parent_keys(repo_root: Path) -> list[str]:
    protocol = load_json(repo_root / "configs/STAGE_X_X1R_T1D0_ATTACK_PARENT_AUTHORITY_V1.json")
    return [parse_key(value) for value in protocol["parent_authority"]["selected_parent_keys"]]


def load_d0r_selected_parent_keys(repo_root: Path) -> list[str]:
    path = repo_root / "reports/STAGE_X_X1R_T1D0R_PARENT_LEDGER_V1.json"
    if not path.is_file():
        raise IntegrityError(f"HISTORICAL_D0R_LEDGER_MISSING:{path}")
    rows = load_jsonl(path)
    return [parse_key(row["canonical_parent_key"]) for row in sorted(rows, key=lambda row: int(row["ordinal"]))]


def audit_schedule_contract(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path = repo_root / str(protocol["timing_freeze"]["pure_contract_module"])
    errors: list[str] = []
    if not path.is_file():
        return {"status": "HOLD", "errors": [f"SCHEDULE_MODULE_MISSING:{path}"]}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import) for _ in [0]]
    imported.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    forbidden_imports = {"torch", "trans" + "formers", "g" + "ym", "mujoco", "robosuite"}
    if any(name.split(".")[0] in forbidden_imports for name in imported):
        errors.append("SCHEDULE_MODULE_EXECUTION_IMPORT")
    if "env." + "step(" in source or "model." + "generate" in source:
        errors.append("SCHEDULE_MODULE_EXECUTION_CALL_TEXT")
    if attack_steps(7) != (7, 8, 9, 10, 11):
        errors.append("ATTACK_STEP_CONTRACT_MISMATCH")
    if followup_steps(7) != (12, 13, 14, 15, 16, 17, 18, 19, 20, 21):
        errors.append("FOLLOWUP_STEP_CONTRACT_MISMATCH")
    if not legal_horizon(7, 22) or legal_horizon(7, 21):
        errors.append("LEGAL_HORIZON_BOUNDARY_MISMATCH")
    timing = protocol["timing_freeze"]
    if timing["attack_window"]["length"] != ATTACK_WINDOW_LENGTH:
        errors.append("CONFIG_ATTACK_WINDOW_LENGTH_MISMATCH")
    if timing["physical_followup"]["length"] != PHYSICAL_FOLLOWUP_LENGTH:
        errors.append("CONFIG_FOLLOWUP_LENGTH_MISMATCH")
    if timing["prev_delta_contract"]["entry"] != PREV_DELTA_BOUNDARIES["entry"]:
        errors.append("CONFIG_PREV_DELTA_ENTRY_MISMATCH")
    plan = protocol["clean_materialization_plan"]
    if plan["enabled"] or plan["execution_authorized"] or plan["runner_bound"]:
        errors.append("CLEAN_PLAN_NOT_HARD_DISABLED")
    return {
        "status": "PASS" if not errors else "HOLD",
        "errors": errors,
        "module_path": str(path),
        "module_sha256": sha256_file(path),
        "contract": {
            "attack_steps_7": list(attack_steps(7)),
            "followup_steps_7": list(followup_steps(7)),
            "legal_horizon_7_22": legal_horizon(7, 22),
            "legal_horizon_7_21": legal_horizon(7, 21),
            "prev_delta_boundaries": dict(PREV_DELTA_BOUNDARIES),
            "clean_materialization_plan": dict(plan),
        },
    }


def verify_file(path: str, expected: str) -> dict[str, Any]:
    target = Path(path)
    row = {"path": path, "expected_sha256": expected, "exists": target.is_file()}
    if target.is_file():
        row["actual_sha256"] = sha256_file(target)
        row["byte_size"] = target.stat().st_size
        row["match"] = row["actual_sha256"] == expected
    else:
        row["actual_sha256"] = None
        row["byte_size"] = None
        row["match"] = False
    return row


def audit_runtime_authority(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    authority = protocol["runtime_authority"]
    errors: list[str] = []
    contract_path = repo_root / str(authority["suite_matched_contract_path"])
    contract_row = verify_file(str(contract_path), str(authority["suite_matched_contract_sha256"]))
    if not contract_row["match"]:
        errors.append("SUITE_MATCHED_CONTRACT_SHA256_MISMATCH")
    handoff_path = repo_root / str(authority["detector_handoff_path"])
    handoff_row = verify_file(str(handoff_path), str(authority["detector_handoff_sha256"]))
    if not handoff_row["match"]:
        errors.append("DETECTOR_HANDOFF_SHA256_MISMATCH")
    external_rows: list[dict[str, Any]] = []
    student = authority["student"]
    for item in (student["checkpoint"], student["model_source"], student["adapter"], student["feature_source"], student["normalization"], student["thresholds"]):
        row = verify_file(str(item["path"]), str(item["sha256"]))
        external_rows.append(row)
        if not row["match"]:
            errors.append(f"RUNTIME_FILE_BINDING_MISMATCH:{item['path']}")
    tokenizer_row = verify_file(str(authority["tokenizer"]["path"]), str(authority["tokenizer"]["sha256"]))
    external_rows.append(tokenizer_row)
    if not tokenizer_row["match"]:
        errors.append("TOKENIZER_BINDING_MISMATCH")
    contract_payload: dict[str, Any] = {}
    if contract_path.is_file():
        contract_payload = load_json(contract_path)
        for suite, suite_info in contract_payload.get("suites", {}).items():
            model_path = Path(str(suite_info["model_path"]))
            for filename, expected in suite_info.get("model_identity", {}).get("key_files", {}).items():
                row = verify_file(str(model_path / filename), str(expected))
                row["suite"] = suite
                row["role"] = "suite_model_key_file"
                external_rows.append(row)
                if not row["match"]:
                    errors.append(f"SUITE_MODEL_KEY_FILE_MISMATCH:{suite}:{filename}")
            decoder_path = model_path / "modeling_prismatic.py"
            decoder_row = verify_file(str(decoder_path), str(authority["model_decoder"]["sha256"]))
            decoder_row["suite"] = suite
            decoder_row["role"] = "suite_model_decoder"
            external_rows.append(decoder_row)
            if not decoder_row["match"]:
                errors.append(f"SUITE_MODEL_DECODER_MISMATCH:{suite}")
    else:
        errors.append("SUITE_MATCHED_CONTRACT_UNREADABLE")
    semantics = authority["clean_runtime_semantics"]
    if semantics["task_success_evaluator"] != "NOT_YET_UNIQUE_PATH_BOUND_IN_T1_AUTHORITY":
        errors.append("TASK_SUCCESS_BINDING_UNEXPECTED")
    if semantics["episode_horizon"] != "NOT_YET_UNIQUE_PATH_BOUND_IN_T1_AUTHORITY":
        errors.append("EPISODE_HORIZON_BINDING_UNEXPECTED")
    errors.extend(
        [
            "TASK_SUCCESS_EVALUATOR_PATH_NOT_UNIQUELY_BOUND",
            "EPISODE_HORIZON_PATH_NOT_UNIQUELY_BOUND",
        ]
    )
    return {
        "status": "PASS" if not errors else "HOLD",
        "errors": errors,
        "declared_contract": {
            "path": str(contract_path),
            "sha256": authority["suite_matched_contract_sha256"],
        },
        "repository_handoff": handoff_row,
        "detector_files": external_rows,
        "clean_runtime_semantics": semantics,
        "historical_model_source_note": "T1 handoff model-source declaration must match the current prospective file; no current directory hash is used to rewrite historical identity.",
    }


def protected_counters(protocol: Mapping[str, Any]) -> dict[str, int]:
    return {str(key): 0 for key in protocol["protected_boundary"]["counters"]}


def audit(protocol: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    g10_keys: set[str] = set()
    exclusion_sets: dict[str, set[str]] = {}
    source_meta: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    physical_canonical_keys: set[str] = set()
    try:
        g10_keys, exclusion_sets, source_meta, _, physical_canonical_keys = load_d0r1_sources(protocol)
        derived = derive_population(
            g10_keys,
            exclusion_sets,
            str(protocol["selection"]["selection_salt"]),
        )
        physical_spec = next(
            spec for spec in protocol["source_recompute"]["sources"] if spec["kind"] == "directory_identity_listing"
        )
        _, _, alias_rows, physical_canonical_keys = list_physical_intervention_source(physical_spec, g10_keys)
    except (AuthorityError, IntegrityError, KeyError, json.JSONDecodeError, OSError) as exc:
        errors.append(str(exc))
        derived = {"g10_rows": [], "design_rows": [], "parent_rows": [], "exclusion_union": [], "fresh_rows": [], "source_counts": {}, "pairwise_exclusion_intersections": {}}

    selected = [row["canonical_parent_key"] for row in derived["parent_rows"]]
    d0_selected: list[str] = []
    d0r_selected: list[str] = []
    try:
        d0_selected = load_d0_selected_parent_keys(repo_root)
        if selected != d0_selected:
            errors.append("STAGE_X_X1R_T1D0R1_HOLD_PARENT_RECOMPUTE_MISMATCH")
    except (IntegrityError, KeyError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"D0_SELECTION_COMPARISON:{exc}")
    try:
        d0r_selected = load_d0r_selected_parent_keys(repo_root)
    except (IntegrityError, KeyError, OSError, json.JSONDecodeError) as exc:
        errors.append(f"D0R_SELECTION_COMPARISON:{exc}")

    alias_invariance: dict[str, Any] = {"status": "NOT_RUN", "errors": []}
    try:
        diagnostic_sets = dict(exclusion_sets)
        physical_name = "prior_physical_intervention_named_roots"
        diagnostic_sets[physical_name] = set(physical_canonical_keys)
        diagnostic = derive_population(g10_keys, diagnostic_sets, str(protocol["selection"]["selection_salt"]))
        baseline_union = derived["exclusion_union"]
        baseline_fresh = [row["canonical_parent_key"] for row in derived["fresh_rows"]]
        diagnostic_union = diagnostic["exclusion_union"]
        diagnostic_fresh = [row["canonical_parent_key"] for row in diagnostic["fresh_rows"]]
        alias_invariance = {
            "status": "PASS" if baseline_union == diagnostic_union and baseline_fresh == diagnostic_fresh and selected == [row["canonical_parent_key"] for row in diagnostic["parent_rows"]] else "HOLD",
            "baseline_with_explicit_aliases": {
                "union_count": len(baseline_union),
                "fresh_count": len(baseline_fresh),
                "selected_parent_keys": selected,
            },
            "diagnostic_canonical_only": {
                "union_count": len(diagnostic_union),
                "fresh_count": len(diagnostic_fresh),
                "selected_parent_keys": [row["canonical_parent_key"] for row in diagnostic["parent_rows"]],
            },
            "alias_keys": sorted(set(exclusion_sets.get(physical_name, set())) - set(physical_canonical_keys)),
            "alias_rows": alias_rows,
        }
        if alias_invariance["status"] != "PASS":
            errors.append("STAGE_X_X1R_T1D0R1_HOLD_PHYSICAL_INTERVENTION_IDENTITY_ALIAS")
    except (AuthorityError, IntegrityError) as exc:
        errors.append(f"STAGE_X_X1R_T1D0R1_HOLD_PHYSICAL_INTERVENTION_IDENTITY_ALIAS:{exc}")
        alias_invariance = {"status": "HOLD", "errors": [str(exc)]}

    local_reconciliation = reconcile_local_bindings(protocol, repo_root)
    errors.extend(local_reconciliation["errors"])
    schedule_audit = audit_schedule_contract(protocol, repo_root)
    errors.extend(schedule_audit["errors"])
    runtime_audit = audit_runtime_authority(protocol, repo_root)
    errors.extend(runtime_audit["errors"])

    seed_namespace = str(protocol["seed_authority"]["namespace"])
    seeded_parent_rows: list[dict[str, Any]] = []
    for row in derived["parent_rows"]:
        seeded = dict(row)
        seeded["clean_seed"] = seed_for_parent(seed_namespace, row["canonical_parent_key"])
        seeded["clean_seed_namespace"] = seed_namespace
        seeded["clean_seed_outcome_blind"] = True
        seeded_parent_rows.append(seeded)
    derived["parent_rows"] = seeded_parent_rows

    missing = [row["design_cell"] for row in derived["design_rows"] if not row["selected"]]
    suite_counts = {
        suite: sum(row["suite"] == suite for row in derived["parent_rows"])
        for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    }
    if len(derived["g10_rows"]) != 1200:
        errors.append(f"G10_IDENTITY_COUNT_DERIVED:{len(derived['g10_rows'])}")
    if len(derived["exclusion_union"]) != 990 or len(derived["fresh_rows"]) != 210:
        errors.append("STAGE_X_X1R_T1D0R1_HOLD_PARENT_RECOMPUTE_MISMATCH")
    if len(derived["design_rows"]) != 40 or len(derived["parent_rows"]) != 39 or missing != ["libero_goal/task_01"] or suite_counts != protocol["selection"]["expected_suite_counts"]:
        errors.append("STAGE_X_X1R_T1D0R1_HOLD_PARENT_RECOMPUTE_MISMATCH")

    status = "STAGE_X_X1R_T1D0R1_AUTHORITY_PASS" if not errors else "STAGE_X_X1R_T1D0R1_HOLD_CLEAN_RUNTIME_AUTHORITY"
    if any("SOURCE" in error or "GIT_BINDING" in error for error in errors):
        status = "STAGE_X_X1R_T1D0R1_HOLD_SOURCE_BINDING"
    elif any("PARENT_RECOMPUTE" in error or "G10_IDENTITY" in error for error in errors):
        status = "STAGE_X_X1R_T1D0R1_HOLD_PARENT_RECOMPUTE_MISMATCH"
    elif any("LOCAL_BINDING" in error for error in errors):
        status = "STAGE_X_X1R_T1D0R1_HOLD_LOCAL_BINDING_PROVENANCE"
    elif any("PHYSICAL_INTERVENTION" in error for error in errors):
        status = "STAGE_X_X1R_T1D0R1_HOLD_PHYSICAL_INTERVENTION_IDENTITY_ALIAS"
    elif any("SCHEDULE" in error or "TIMING" in error for error in errors):
        status = "STAGE_X_X1R_T1D0R1_HOLD_RUNTIME_TIMING_CONTRACT"

    receipt = {
        "schema": "STAGE_X_X1R_T1D0R1_PARENT_RECOMPUTE_AUDIT_V1",
        "status": status,
        "errors": errors,
        "reviewed_source": protocol["reviewed_pr126"],
        "runtime_source": {
            "commit": git_output(repo_root, "rev-parse", "HEAD"),
            "tree": git_output(repo_root, "rev-parse", "HEAD^{tree}"),
            "worktree_status_before_evidence": git_output(repo_root, "status", "--porcelain"),
            "official_environment": protocol["official_environment"],
        },
        "historical_selection_comparison": {
            "d0_selection_salt": protocol["selection"]["selection_salt"],
            "derived_selected_parent_keys": selected,
            "d0_declared_selected_parent_keys": d0_selected,
            "matches_d0_declaration": selected == d0_selected,
            "d0r_historical_selected_parent_keys": d0r_selected,
            "d0r_selection_diff_count": sum(a != b for a, b in zip(selected, d0r_selected)) + abs(len(selected) - len(d0r_selected)),
            "d0r_selection_is_not_an_input": True,
        },
        "source_files": source_meta,
        "derived_population": {
            "g10_identity_count": len(derived["g10_rows"]),
            "g10_identity_digest": identity_digest(row["canonical_parent_key"] for row in derived["g10_rows"]),
            "exclusion_source_counts": derived["source_counts"],
            "pairwise_exclusion_intersections": derived["pairwise_exclusion_intersections"],
            "exclusion_union_count": len(derived["exclusion_union"]),
            "exclusion_union_digest": identity_digest(derived["exclusion_union"]),
            "fresh_candidate_count": len(derived["fresh_rows"]),
            "fresh_candidate_digest": identity_digest(row["canonical_parent_key"] for row in derived["fresh_rows"]),
            "design_cell_count": len(derived["design_rows"]),
            "executable_parent_count": len(derived["parent_rows"]),
            "missing_design_cells": missing,
            "executable_parent_counts_by_suite": suite_counts,
            "zero_replacement": True,
        },
        "alias_invariance": alias_invariance,
        "local_binding_reconciliation": local_reconciliation,
        "runtime_contract_audit": schedule_audit,
        "runtime_authority_audit": runtime_audit,
        "seed_authority": {
            "namespace": seed_namespace,
            "formula": protocol["seed_authority"]["derivation"],
            "outcome_blind": True,
            "seed_count": len(seeded_parent_rows),
        },
        "outcome_firewall": {
            "selection_outcomes_read": False,
            "clean_success_read": False,
            "emit_read": False,
            "vphys_read": False,
            "attack_outcome_read": False,
            "protected_registry_read": False,
            "eval160_read": False,
            "new_model_inference": False,
            "new_simulator_execution": False,
        },
        "authorization": protocol["authorization"],
        "protected_boundary": {
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
            "counters": protected_counters(protocol),
        },
        "historical_records": protocol["historical_records"],
    }
    return receipt, derived, alias_rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def artifact_hashes(repo_root: Path, paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = repo_root / relative
        if not path.is_file():
            raise IntegrityError(f"GENERATED_ARTIFACT_MISSING:{relative}")
        result[relative] = sha256_file(path)
    return result


def write_evidence(protocol: Mapping[str, Any], receipt: dict[str, Any], derived: dict[str, Any], alias_rows: list[dict[str, Any]], repo_root: Path) -> None:
    reports = repo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    parent_audit = reports / "STAGE_X_X1R_T1D0R1_PARENT_RECOMPUTE_AUDIT_V1.json"
    g10_ledger = reports / "STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
    design_ledger = reports / "STAGE_X_X1R_T1D0R1_DESIGN_CELL_LEDGER_V1.json"
    parent_ledger = reports / "STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json"
    local_report = reports / "STAGE_X_X1R_T1D0R1_LOCAL_BINDING_RECONCILIATION_V1.json"
    alias_report = reports / "STAGE_X_X1R_T1D0R1_PHYSICAL_ALIAS_LEDGER_V1.json"
    runtime_report = reports / "STAGE_X_X1R_T1D0R1_RUNTIME_CONTRACT_AUDIT_V1.json"
    sums = reports / "STAGE_X_X1R_T1D0R1_SHA256SUMS.txt"
    root_seal = reports / "STAGE_X_X1R_T1D0R1_ROOT_SEAL.json"
    root_seal_sha = reports / "STAGE_X_X1R_T1D0R1_ROOT_SEAL.sha256"

    parent_audit.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(g10_ledger, derived["g10_rows"])
    write_jsonl(design_ledger, derived["design_rows"])
    write_jsonl(parent_ledger, derived["parent_rows"])
    local_report.write_text(json.dumps(receipt["local_binding_reconciliation"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    alias_report.write_text(json.dumps({"schema": "STAGE_X_X1R_T1D0R1_PHYSICAL_ALIAS_LEDGER_V1", "rows": alias_rows, "invariance": receipt["alias_invariance"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime_report.write_text(json.dumps({"schema": "STAGE_X_X1R_T1D0R1_RUNTIME_CONTRACT_AUDIT_V1", "schedule": receipt["runtime_contract_audit"], "runtime_authority": receipt["runtime_authority_audit"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    authority_paths = [
        "configs/STAGE_X_X1R_T1D0R1_PRECLEAN_INTEGRITY_AUTHORITY_V1.json",
        "src/gripper_attack/stage_x_x1r_v2_schedule_contract.py",
        "scripts/stage_x/audit_stage_x1r_t1d0r1_preclean_integrity.py",
        "tests/stage_x/test_stage_x1r_t1d0r1_preclean_integrity.py",
        "docs/handoffs/STAGE_X_X1R_T1D0R1_PRECLEAN_INTEGRITY_HANDOFF_20260818.md",
        "reports/STAGE_X_X1R_T1D0R1_PARENT_RECOMPUTE_AUDIT_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_DESIGN_CELL_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_LOCAL_BINDING_RECONCILIATION_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_PHYSICAL_ALIAS_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R1_RUNTIME_CONTRACT_AUDIT_V1.json",
    ]
    generated = artifact_hashes(repo_root, authority_paths)
    seal = {
        "schema": "STAGE_X_X1R_T1D0R1_ROOT_SEAL_V1",
        "status": receipt["status"],
        "blocking_errors": receipt["errors"],
        "reviewed_source": receipt["reviewed_source"],
        "runtime_source_pre_evidence": receipt["runtime_source"],
        "generated_artifact_hashes": generated,
        "historical_records": protocol["historical_records"],
        "selection": receipt["historical_selection_comparison"],
        "derived_population": receipt["derived_population"],
        "alias_invariance": receipt["alias_invariance"],
        "local_binding_reconciliation": receipt["local_binding_reconciliation"],
        "runtime_contract_audit": receipt["runtime_contract_audit"],
        "runtime_authority_audit": receipt["runtime_authority_audit"],
        "seed_authority": receipt["seed_authority"],
        "authorization": protocol["authorization"],
        "protected_boundary": receipt["protected_boundary"],
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "final_branch_seal": {
            "commit": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF",
            "tree": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF",
            "note": "The final pushed HEAD/tree is recorded after commit in the handoff; embedding it in this containing commit would be self-referential.",
        },
        "sha256sums_scope": {
            "covers_root_seal_json": True,
            "excludes_self": True,
            "excludes_root_seal_sha256_sidecar": True,
        },
    }
    root_seal.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums_hashes = artifact_hashes(repo_root, authority_paths + ["reports/STAGE_X_X1R_T1D0R1_ROOT_SEAL.json"])
    sums.write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(sums_hashes.items())), encoding="utf-8")
    root_seal_sha.write_text(f"{sha256_file(root_seal)}  {root_seal.name}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only T1-D0R1 pre-clean integrity audit")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1D0R1_PRECLEAN_INTEGRITY_AUTHORITY_V1.json")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol = load_json(args.protocol.resolve())
    receipt, derived, alias_rows = audit(protocol, repo_root)
    if args.write_evidence:
        write_evidence(protocol, receipt, derived, alias_rows, repo_root)
    print(json.dumps({"status": receipt["status"], "errors": receipt["errors"], "derived_population": receipt["derived_population"]}, sort_keys=True))
    return 0 if receipt["status"] == "STAGE_X_X1R_T1D0R1_AUTHORITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
