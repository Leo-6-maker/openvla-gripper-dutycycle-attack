"""CPU-only F1-A2 source/leakage and population audit.

This is intentionally a static audit.  It never loads a model, opens a
simulator, reads a protected registry, or touches a GPU.  The V2 bridge
union includes the older 15-source audit plus stronger current identity
authorities that were not in that diagnostic subset.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE_V2_20260821"
G10 = ROOT / "reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
PI_COMMENT_ID = 5368241704
PI_ATTACHMENT_SHA256 = "64a8b4a966b2579d61eecc672525187bfeef815c727b73ff76e7546428ec7831"

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEV_SALT = "STAGE_X_X1R2_F1A2_DEV_SOURCE_V2_SALT_20260821"
BRIDGE_SALT = "STAGE_X_X1R2_F1A2_BRIDGE_SOURCE_V2_SALT_20260821"
DEV_KEY_RE = re.compile(r"libero_(?:10|goal|object|spatial)/task_\d{2}/state_(?:0\d|1\d)")
BRIDGE_KEY_RE = re.compile(r"libero_(?:10|goal|object|spatial)/task_\d{2}/state_[2-4]\d")

# This is the exact 15-source baseline used by the previous F1-A diagnostic.
BASE_BRIDGE_SOURCES = (
    "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json",
    "reports/STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1.json",
    "reports/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_ENGINEERING_POOL_V1.json",
    "reports/STAGE_X_X1R2_Q3_ENGINEERING_FIXTURES_V1.json",
    "configs/STAGE_VI_B2_FRESH_PARENT_MANIFEST_V3.json",
    "reports/server_evidence/STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json",
    "configs/STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1_1.json",
    "reports/STAGE_X_X1R2_Q3R3_E0_CANDIDATE_MATRIX_V1.csv",
    "reports/STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS_V1.json",
    "reports/STAGE_X1R2_Q3R3_VISUAL_DIVERGENCE_AUDIT_V1.json",
    "reports/STAGE_X_X1R2_Q3R3_E2_GOAL_NO_LEGAL_EMIT_DIAGNOSTIC_V1.json",
    "reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_DECISION_TABLE_V1.json",
    "reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/"
    "STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json",
    "paper/data/PAPER_V1_FIGURE5_E3_E4_PARENT_REALIZABILITY.csv",
    "paper/tables/PAPER_V1_E3_E4_PARENT_REALIZABILITY.csv",
)

# These are stronger than a mere text mention: one is a frozen prior
# engineering population and the other contains identity-level runtime
# attempt receipts.  Both are excluded conservatively from BRIDGE.
STRONGER_BRIDGE_SOURCES = (
    "reports/STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json",
    "configs/STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json_keys(value: Any, pattern: re.Pattern[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        key = value.get("canonical_parent_key")
        if isinstance(key, str) and pattern.fullmatch(key):
            found.add(key)
        for item in value.values():
            found |= canonical_json_keys(item, pattern)
    elif isinstance(value, list):
        for item in value:
            found |= canonical_json_keys(item, pattern)
    return found


def source_keys(path: Path, pattern: re.Pattern[str], *, jsonl: bool = False) -> set[str]:
    if jsonl:
        return {
            key
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for key in canonical_json_keys(json.loads(line), pattern)
        }
    if path.suffix.lower() == ".json":
        return canonical_json_keys(load_json(path), pattern)
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def suite_counts(keys: set[str]) -> dict[str, int]:
    return {suite: sum(key.startswith(f"{suite}/") for key in keys) for suite in SUITES}


def rank_hash(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()


def tracked_scan_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "configs", "reports", "docs/handoffs", "paper"],
        text=True,
    )
    return [path for path in output.splitlines() if path]


def skip_dev_scan(path: str) -> bool:
    lower = path.lower()
    if "eval160" in lower or "protected" in lower:
        return True
    # These are audits of mentions, not historical exposure authorities.
    return any(
        marker in path
        for marker in (
            "STAGE_X_X1R2_F1A_SOURCE_OPTIONS_AUDIT",
            "STAGE_X_X1R2_F1A_STATIC_FEASIBILITY_AUDIT",
            "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE",
        )
    )


def classify_dev_source(path: str) -> tuple[str, bool, bool]:
    lower = path.lower()
    if "g2_canary" in lower or "fec_" in lower or "/fec_" in lower:
        return "HARD_EXCLUDE_FEC_G2_CANARY_OR_RUNTIME", False, True
    if "c3_t1d" in lower or "g_rec" in lower or "protocol_amendment_v5" in lower:
        return "HARD_EXCLUDE_C3_CANARY_OR_RUNTIME", False, True
    if "manual" in lower:
        return "HARD_EXCLUDE_MANUAL_OUTCOME_ADJUDICATION", False, True
    if "official_v3_detector_v5_takeover" in lower:
        return "UNRESOLVED_DETECTOR_SOURCE_AUDIT_CONSERVATIVE_EXCLUDE", False, True
    if any(marker in lower for marker in ("q3", "e2", "e3", "e4", "attack", "pgd", "physical", "v_phys")):
        return "HARD_EXCLUDE_PRIOR_ENGINEERING_OR_ATTACK_EXPOSURE", False, True
    if "detector" in lower or "student" in lower:
        return "LEGACY_DETECTOR_TRAIN_ONLY_ALLOWED_FOR_F1_DEV", True, False
    return "UNRESOLVED_EXPOSURE_CONSERVATIVE_EXCLUDE", False, True


def build_dev_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hits: dict[str, list[dict[str, Any]]] = {}
    scanned_files = 0
    hit_files: set[str] = set()
    for relative in tracked_scan_paths():
        if skip_dev_scan(relative):
            continue
        path = ROOT / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        keys = sorted(set(DEV_KEY_RE.findall(text)))
        if not keys:
            continue
        scanned_files += 1
        hit_files.add(relative)
        classification, train_only, hard = classify_dev_source(relative)
        record = {
            "path": relative,
            "sha256": sha256(path),
            "source_classification": classification,
            "detector_train_only": train_only,
            "hard_exclusion_source": hard,
        }
        for key in keys:
            hits.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                key = f"{suite}/task_{task:02d}/state_{state:02d}"
                source_hits = sorted(hits.get(key, []), key=lambda item: item["path"])
                hard_hits = [item for item in source_hits if item["hard_exclusion_source"]]
                train_hits = [item for item in source_hits if item["detector_train_only"]]
                if hard_hits:
                    exposure = sorted({item["source_classification"] for item in hard_hits})
                    eligible = False
                elif train_hits:
                    exposure = ["LEGACY_DETECTOR_TRAIN_ONLY_ALLOWED_FOR_F1_DEV"]
                    eligible = True
                else:
                    exposure = ["NO_HARD_EXPOSURE_FOUND_IN_SCANNED_AUTHORITY"]
                    eligible = not source_hits
                rows.append(
                    {
                        "canonical_parent_key": key,
                        "suite": suite,
                        "task": f"task_{task:02d}",
                        "state": state,
                        "source_domain": "DEV_SOURCE_V2",
                        "deterministic_rank_hash": rank_hash(DEV_SALT, key),
                        "historical_exposure_source_hits": source_hits,
                        "exposure_classification": exposure,
                        "detector_train_only_overlap": bool(train_hits) and not hard_hits,
                        "no_hard_exclusion_exposure_confirmed": not hard_hits,
                        "dev_eligible": eligible,
                    }
                )
    eligible = [row for row in rows if row["dev_eligible"]]
    proposed: dict[str, list[str]] = {}
    for suite in SUITES:
        ranked = sorted(
            (row for row in eligible if row["suite"] == suite),
            key=lambda row: (row["deterministic_rank_hash"], row["canonical_parent_key"]),
        )
        proposed[suite] = [row["canonical_parent_key"] for row in ranked[:6]]
    audit = {
        "schema": "STAGE_X_X1R2_F1A2_DEV_EXPOSURE_CLASSIFICATION_V2",
        "source_domain": "DEV_SOURCE_V2",
        "state_range": "0..19",
        "deterministic_salt": DEV_SALT,
        "rank_rule": "sha256(salt|canonical_parent_key), then canonical_parent_key",
        "tracked_files_scanned": len(tracked_scan_paths()),
        "files_with_dev_identity_hits": sorted(hit_files),
        "files_with_dev_identity_hit_count": scanned_files,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "eligible_by_suite": suite_counts({row["canonical_parent_key"] for row in eligible}),
        "hard_excluded_count": sum(not row["dev_eligible"] for row in rows),
        "proposed_dev_by_suite": proposed,
        "rows": rows,
    }
    return rows, audit


def bridge_source_record(relative: str, pattern: re.Pattern[str], *, jsonl: bool, role: str) -> tuple[dict[str, Any], set[str]]:
    path = ROOT / relative
    if not path.is_file():
        return {"path": relative, "role": role, "missing": True}, set()
    keys = source_keys(path, pattern, jsonl=jsonl)
    return {
        "path": relative,
        "role": role,
        "sha256": sha256(path),
        "parser": "canonical_parent_key_recursive" if path.suffix.lower() == ".json" and not jsonl else "canonical_parent_key_jsonl" if jsonl else "identity_regex",
        "key_count": len(keys),
    }, keys


def bridge_inventory_classification(relative: str) -> tuple[str, bool]:
    """Return (reason, include_in_union) for every tracked identity mention."""
    if relative in BASE_BRIDGE_SOURCES:
        return "EXPLICIT_BASE_F1A_15_SOURCE_UNION", True
    if relative in STRONGER_BRIDGE_SOURCES:
        return "EXPLICIT_STRONGER_CURRENT_IDENTITY_AUTHORITY", True
    lower = relative.lower()
    if "f1a_source_options_audit" in lower or "f1a_static_feasibility_audit" in lower:
        return "AUDIT_OF_MENTIONS_NOT_IDENTITY_EXPOSURE", False
    if "protocol" in lower or "static_taxonomy" in lower:
        return "STATIC_PROTOCOL_OR_TAXONOMY_NOT_EXECUTION", False
    if any(marker in lower for marker in ("root_seal", "recompute_audit", "design_cell_ledger", "physical_alias_ledger")):
        return "DERIVED_ROOT_OR_AUDIT_NOT_INDEPENDENT_EXPOSURE", False
    # An unclassified identity-bearing report is conservatively included.
    return "UNRESOLVED_IDENTITY_SOURCE_CONSERVATIVE_UNION", True


def scan_bridge_inventory(fresh_keys: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    inventory: list[dict[str, Any]] = []
    included: set[str] = set()
    for relative in tracked_scan_paths():
        lower = relative.lower()
        if "eval160" in lower or "protected" in lower:
            continue
        if "_g10_identity_exclusion_ledger" in lower:
            continue
        path = ROOT / relative
        text = path.read_text(encoding="utf-8", errors="replace")
        keys = set(BRIDGE_KEY_RE.findall(text)) & fresh_keys
        if not keys:
            continue
        reason, include = bridge_inventory_classification(relative)
        inventory.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "fresh_overlap_count": len(keys),
                "fresh_overlap_by_suite": suite_counts(keys),
                "classification": reason,
                "included_in_complete_union": include,
            }
        )
        if include:
            included |= keys
    return sorted(inventory, key=lambda item: item["path"]), included


def build_population_audit(dev_rows: list[dict[str, Any]], dev_audit: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    g10_rows = [json.loads(line) for line in G10.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(g10_rows) != 1200:
        errors.append(f"G10_ROW_COUNT:{len(g10_rows)}")
    fresh_rows = [row for row in g10_rows if row.get("fresh_after_exclusion") is True]
    fresh_keys = {row["canonical_parent_key"] for row in fresh_rows}
    if len(fresh_rows) != 210:
        errors.append(f"G10_FRESH_COUNT:{len(fresh_rows)}")

    source_bindings: list[dict[str, Any]] = []
    consumed_union: set[str] = set()
    for relative in BASE_BRIDGE_SOURCES:
        record, keys = bridge_source_record(relative, BRIDGE_KEY_RE, jsonl=False, role="BASE_F1A_15_SOURCE_UNION")
        if record.get("missing"):
            errors.append(f"MISSING_BRIDGE_SOURCE:{relative}")
        record["fresh_overlap_count"] = len(keys & fresh_keys)
        record["fresh_overlap_by_suite"] = suite_counts(keys & fresh_keys)
        source_bindings.append(record)
        consumed_union |= keys
    baseline_consumed_union = set(consumed_union)
    for relative in STRONGER_BRIDGE_SOURCES:
        record, keys = bridge_source_record(
            relative,
            BRIDGE_KEY_RE,
            jsonl=relative.endswith("PARENT_LEDGER_V1.json"),
            role="STRONGER_CURRENT_IDENTITY_AUTHORITY",
        )
        if record.get("missing"):
            errors.append(f"MISSING_STRONGER_BRIDGE_SOURCE:{relative}")
        record["fresh_overlap_count"] = len(keys & fresh_keys)
        record["fresh_overlap_by_suite"] = suite_counts(keys & fresh_keys)
        source_bindings.append(record)
        consumed_union |= keys
    explicit_stronger_consumed_union = set(consumed_union)

    # Reconcile every tracked identity-bearing authority, not just the
    # hand-listed sources.  Static protocols and derived seals are retained
    # in the inventory but cannot create an exposure by themselves.
    bridge_inventory, inventory_union = scan_bridge_inventory(fresh_keys)
    consumed_union |= inventory_union

    remaining = fresh_keys - consumed_union
    remaining_by_suite = suite_counts(remaining)
    bridge_capacity_deficit = {
        suite: max(0, 5 - remaining_by_suite[suite]) for suite in SUITES
    }
    bridge_proposed = {
        suite: [
            key
            for _, key in sorted(
                (rank_hash(BRIDGE_SALT, key), key)
                for key in remaining
                if key.startswith(f"{suite}/")
            )[:5]
        ]
        for suite in SUITES
    }
    dev_proposed = {
        suite: dev_audit["proposed_dev_by_suite"][suite]
        for suite in SUITES
    }
    dev_keys = set().union(*(set(values) for values in dev_proposed.values()))
    bridge_keys = set().union(*(set(values) for values in bridge_proposed.values()))
    dev_pass = all(len(dev_proposed[suite]) == 6 for suite in SUITES) and all(
        row["no_hard_exclusion_exposure_confirmed"] for row in dev_rows if row["canonical_parent_key"] in dev_keys
    )
    bridge_pass = all(len(bridge_proposed[suite]) == 5 for suite in SUITES)
    a2_pass = not errors and dev_pass and bridge_pass and not (dev_keys & bridge_keys)
    status = (
        "PASS_F1A2_SOURCE_CONTRACT_ESTABLISHED"
        if a2_pass
        else "HOLD_F1A2_SOURCE_CONTRACT_NOT_ESTABLISHED"
    )
    union_digest = hashlib.sha256(("\n".join(sorted(consumed_union)) + "\n").encode()).hexdigest()
    remaining_digest = hashlib.sha256(("\n".join(sorted(remaining)) + "\n").encode()).hexdigest()
    return {
        "schema": "STAGE_X_X1R2_F1A2_POPULATION_AUDIT_V2",
        "gate": "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE_V2",
        "status": status,
        "scope": "CPU/static/offline only; no GPU, model, inference, simulator, env.step, PGD, V_phys, physical, or protected read",
        "source": {
            "branch": git("branch", "--show-current"),
            "input_commit": git("rev-parse", "HEAD"),
            "input_tree": git("rev-parse", "HEAD^{tree}"),
            "g10_path": G10.relative_to(ROOT).as_posix(),
            "g10_sha256": sha256(G10),
            "g10_rows": len(g10_rows),
            "g10_fresh_rows": len(fresh_rows),
            "g10_fresh_by_suite": suite_counts(fresh_keys),
        },
        "pi_authority": {
            "pull_request": 135,
            "latest_v2_comment_id": PI_COMMENT_ID,
            "latest_v2_attachment_sha256": PI_ATTACHMENT_SHA256,
            "superseded_attachment_sha256": "e9ba28d2f45382f24a08d343d58fe914e059dfb392021f774cefd84bfc7d0445",
        },
        "dev": {
            "source_domain": "DEV_SOURCE_V2",
            "requested_count": 24,
            "requested_per_suite": 6,
            "salt": DEV_SALT,
            "proposed_by_suite": dev_proposed,
            "proposed_count": len(dev_keys),
            "permanent_exclusion_applied": False,
            "freeze_status": "NOT_FROZEN_UNTIL_BOTH_LEDGER_PASS" if not a2_pass else "FROZEN",
        },
        "bridge": {
            "source_domain": "BRIDGE_SOURCE_V2",
            "g10_state_range": "20..49",
            "requested_count": 20,
            "requested_per_suite": 5,
            "salt": BRIDGE_SALT,
            "base_source_count": len(BASE_BRIDGE_SOURCES),
            "stronger_source_count": len(STRONGER_BRIDGE_SOURCES),
            "source_bindings": source_bindings,
            "complete_identity_source_inventory": bridge_inventory,
            "complete_identity_source_inventory_count": len(bridge_inventory),
            "complete_current_exclusion_union_digest": union_digest,
            "complete_current_exclusion_union_key_count": len(consumed_union),
            "capacity_reconciliation": {
                "after_previous_15_source_subset": suite_counts(fresh_keys - baseline_consumed_union),
                "after_named_stronger_sources": suite_counts(fresh_keys - explicit_stronger_consumed_union),
                "after_complete_identity_source_inventory": remaining_by_suite,
            },
            "remaining_fresh_digest": remaining_digest,
            "remaining_fresh_count": len(remaining),
            "remaining_fresh_by_suite": remaining_by_suite,
            "capacity_deficit_by_suite": bridge_capacity_deficit,
            "proposed_by_suite": bridge_proposed,
            "identity_only_at_freeze": True,
            "outcome_read": False,
            "runtime_read": False,
            "replacement_top_up_or_suite_substitution": False,
            "freeze_status": "NOT_FROZEN_UNTIL_EXACT_5_PER_SUITE" if not a2_pass else "FROZEN",
        },
        "disjointness": {
            "dev_bridge_key_intersection_count": len(dev_keys & bridge_keys),
            "source_range_intersection_count": 0,
            "pass": not (dev_keys & bridge_keys),
        },
        "gate_checks": {
            "dev_exact_6_per_suite": all(len(dev_proposed[suite]) == 6 for suite in SUITES),
            "dev_no_hard_exposure": dev_pass,
            "bridge_exact_5_per_suite": bridge_pass,
            "bridge_capacity_deficit_by_suite": bridge_capacity_deficit,
            "bridge_strict_g10_fresh": bridge_pass and not errors,
            "intersection_zero": not (dev_keys & bridge_keys),
            "bridge_outcome_read_counter": 0,
            "eval160": "UNREAD",
            "protected": "UNREAD",
            "errors": errors,
        },
        "protected_boundary": {
            "gpu": 0,
            "openvla_inference": 0,
            "simulator": 0,
            "env_step": 0,
            "pgd": 0,
            "vphys": 0,
            "physical_intervention": 0,
            "eval160": "UNREAD",
            "protected": "UNREAD",
        },
        "stop_rule": None if a2_pass else "HOLD_F1A2_SOURCE_CONTRACT_NOT_ESTABLISHED",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev_rows, dev_audit = build_dev_audit()
    population = build_population_audit(dev_rows, dev_audit)
    status = population["status"]

    contract = {
        "schema": "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_CONTRACT_V2",
        "gate": "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE_V2",
        "status": status,
        "pi_authority": population["pi_authority"],
        "source_binding": population["source"],
        "dev_source": {
            "name": "DEV_SOURCE_V2",
            "state_range": "0..19",
            "count": 24,
            "per_suite": 6,
            "ranking": "sha256(STAGE_X_X1R2_F1A2_DEV_SOURCE_V2_SALT_20260821|canonical_parent_key)",
            "held_out_claim": False,
            "permanent_exclusion": True,
        },
        "bridge_source": {
            "name": "BRIDGE_SOURCE_V2",
            "state_range": "20..49",
            "count": 20,
            "per_suite": 5,
            "ranking": "sha256(STAGE_X_X1R2_F1A2_BRIDGE_SOURCE_V2_SALT_20260821|canonical_parent_key)",
            "identity_only_until_f1_c": True,
            "no_replacement_top_up_or_suite_substitution": True,
        },
        "dev_hard_exclusion_policy": [
            "prior_visual_attack_or_pgd_or_attack_result",
            "prior_physical_intervention_or_vphys_or_command_open",
            "prior_q3_q3ar_q3r2_q3r3_e2_e3_engineering_exposure",
            "prior_fec_g2_canary_or_runtime_qualification",
            "prior_manual_outcome_adjudication",
            "protected_or_eval160_membership",
            "unresolved_exposure_category",
        ],
        "allowed_dev_contamination_label": "LEGACY_DETECTOR_TRAIN_ONLY_ALLOWED_FOR_F1_DEV",
        "protected_boundary": population["protected_boundary"],
        "method_or_scientific_counters_changed": False,
        "model_simulator_pgd_physical_protected_reads_authorized": False,
    }
    write_json(OUT / "F1A2_SOURCE_LEAKAGE_CONTRACT_V2.json", contract)
    write_json(OUT / "F1A2_DEV_EXPOSURE_CLASSIFICATION_V2.json", dev_audit)
    write_json(OUT / "F1A2_POPULATION_AUDIT_V2.json", population)

    dev_keys = set().union(*(set(values) for values in population["dev"]["proposed_by_suite"].values()))
    bridge_keys = set().union(*(set(values) for values in population["bridge"]["proposed_by_suite"].values()))
    dev_ledger = {
        "schema": "STAGE_X_X1R2_F1A2_DEV_LEDGER_V2",
        "status": "FROZEN" if status.startswith("PASS") else "NOT_FROZEN_A2_HOLD",
        "freeze_status": "FROZEN" if status.startswith("PASS") else "NOT_FROZEN_A2_HOLD",
        "source_domain": "DEV_SOURCE_V2",
        "salt": DEV_SALT,
        "permanent_exclusion_applied": status.startswith("PASS"),
        "rows": [row for row in dev_rows if row["canonical_parent_key"] in dev_keys],
        "row_count": len(dev_keys),
        "outcome_read": False,
        "protected": "UNREAD",
    }
    bridge_ledger = {
        "schema": "STAGE_X_X1R2_F1A2_BRIDGE_LEDGER_V2",
        "status": "FROZEN" if status.startswith("PASS") else "NOT_FROZEN_A2_HOLD",
        "freeze_status": "FROZEN" if status.startswith("PASS") else "NOT_FROZEN_A2_HOLD",
        "source_domain": "BRIDGE_SOURCE_V2",
        "salt": BRIDGE_SALT,
        "g10_state_range": "20..49",
        "permanent_exclusion_applied": status.startswith("PASS"),
        "rows": [
            {"canonical_parent_key": key, "source_domain": "BRIDGE_SOURCE_V2", "identity_only": True}
            for key in sorted(bridge_keys)
        ],
        "row_count": len(bridge_keys),
        "outcome_read": False,
        "runtime_read": False,
        "protected": "UNREAD",
    }
    write_json(OUT / "F1A2_DEV_LEDGER_V2.json", dev_ledger)
    write_json(OUT / "F1A2_BRIDGE_LEDGER_V2.json", bridge_ledger)

    seal_paths = [
        "scripts/stage_x/audit_stage_x1r2_f1a2_source_leakage_v2.py",
        *(f"reports/{OUT.relative_to(ROOT).as_posix().split('/', 1)[1]}/{name}" for name in (
            "F1A2_SOURCE_LEAKAGE_CONTRACT_V2.json",
            "F1A2_DEV_EXPOSURE_CLASSIFICATION_V2.json",
            "F1A2_POPULATION_AUDIT_V2.json",
            "F1A2_DEV_LEDGER_V2.json",
            "F1A2_BRIDGE_LEDGER_V2.json",
        )),
    ]
    # The first item is a source path; all other items are output paths.
    artifact_hashes = {path: sha256(ROOT / path) for path in seal_paths}
    root_seal = {
        "schema": "STAGE_X_X1R2_F1A2_ROOT_SEAL_V2",
        "status": status,
        "gate": "STAGE_X_X1R2_F1A2_SOURCE_LEAKAGE_AND_POPULATION_FREEZE_V2",
        "input_commit": population["source"]["input_commit"],
        "input_tree": population["source"]["input_tree"],
        "pi_comment_id": PI_COMMENT_ID,
        "pi_attachment_sha256": PI_ATTACHMENT_SHA256,
        "artifact_hashes": artifact_hashes,
        "complete_current_exclusion_union_digest": population["bridge"]["complete_current_exclusion_union_digest"],
        "protected_boundary": population["protected_boundary"],
        "bridge_outcome_read_counter": 0,
        "seal_scope_excludes_sidecar": True,
    }
    root_path = OUT / "F1A2_ROOT_SEAL_V2.json"
    sidecar_path = OUT / "F1A2_ROOT_SEAL_V2.sha256"
    write_json(root_path, root_seal)
    sidecar_path.write_text(f"{sha256(root_path)}  {root_path.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "dev_eligible_by_suite": dev_audit["eligible_by_suite"],
        "bridge_remaining_fresh_by_suite": population["bridge"]["remaining_fresh_by_suite"],
        "bridge_proposed_count": bridge_ledger["row_count"],
        "root_seal": str(root_path.relative_to(ROOT)).replace("\\", "/"),
    }, sort_keys=True))
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
