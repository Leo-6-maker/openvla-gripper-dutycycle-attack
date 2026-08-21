"""CPU-only F1-A population and Y1 evidence audit.

This audit never loads a model, starts a simulator, or reads protected data.
It uses the sealed G10 held-out identity ledger and already-published
engineering manifests to fail closed before any F1-B GPU work.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/STAGE_X_X1R2_F1A_STATIC_FEASIBILITY_AUDIT_V1.json"
G10 = ROOT / "reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
E4_ROWS = ROOT / (
    "reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/"
    "STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json"
)

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
KEY_RE = re.compile(r"libero_(?:10|goal|object|spatial)/task_\d{2}/state_\d{2}")

# These are the published identity-bearing sources that can add consumed
# engineering identities after the immutable G10 exclusion union was sealed.
EXCLUSION_SOURCES = (
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

F1_REQUIRED = {"libero_10": 14, "libero_goal": 14, "libero_object": 14, "libero_spatial": 14}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        key = value.get("canonical_parent_key")
        if isinstance(key, str) and KEY_RE.fullmatch(key):
            found.add(key)
        for item in value.values():
            found |= canonical_keys(item)
    elif isinstance(value, list):
        for item in value:
            found |= canonical_keys(item)
    return found


def source_keys(path: Path) -> set[str]:
    if path.suffix.lower() == ".json":
        return canonical_keys(load_json(path))
    return set(KEY_RE.findall(path.read_text(encoding="utf-8")))


def suite_counts(keys: set[str]) -> dict[str, int]:
    return {suite: sum(key.startswith(f"{suite}/") for key in keys) for suite in SUITES}


def candidate_y1() -> dict[str, Any]:
    data = load_json(E4_ROWS)
    rows: list[dict[str, Any]] = []
    for parent in data.get("parent_rows", []):
        summary = parent.get("summary", {})
        for candidate in parent.get("candidates", []):
            row = dict(candidate)
            row["suite"] = summary.get("suite")
            rows.append(row)

    gripper_ids_by_suite: dict[str, list[int]] = {}
    native_ids_by_suite: dict[str, list[int]] = {}
    strict_by_suite: dict[str, int] = {}
    for suite in SUITES:
        suite_rows = [row for row in rows if row.get("suite") == suite]
        gripper_ids_by_suite[suite] = sorted(
            {int(row["direct_generated_gripper_token_id"]) for row in suite_rows}
        )
        native_ids_by_suite[suite] = sorted(
            {
                int(row["direct_generated_gripper_token_id"])
                for row in suite_rows
                if row.get("native_open") is True
            }
        )
        strict_by_suite[suite] = sum(
            row.get("classification") == "ARM_EXACT_AND_NATIVE_OPEN" for row in suite_rows
        )

    expected_margin_fields = {
        "target_token_score",
        "target_minus_best_competitor_margin",
        "target_minus_competitor_logsumexp_margin",
        "competitor_logsumexp_score",
        "best_native_open_score",
        "best_non_open_competitor_score",
    }
    observed_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    return {
        "status": "Y1_STATIC_AUDIT_COMPLETE",
        "source_path": str(E4_ROWS.relative_to(ROOT).as_posix()),
        "source_sha256": sha256(E4_ROWS),
        "candidate_rows": len(rows),
        "candidate_evidence_complete": all(not row.get("evidence_missing_fields") for row in rows),
        "native_open_token_ids_observed_by_suite": native_ids_by_suite,
        "all_native_open_token_ids_observed": sorted(
            {token for tokens in native_ids_by_suite.values() for token in tokens}
        ),
        "direct_generated_gripper_token_ids_by_suite": gripper_ids_by_suite,
        "strict_valid_candidate_rows_by_suite": strict_by_suite,
        "native_open_rows": sum(row.get("native_open") is True for row in rows),
        "strict_valid_rows": sum(
            row.get("classification") == "ARM_EXACT_AND_NATIVE_OPEN" for row in rows
        ),
        "objective_vs_acceptance": {
            "historical_m0_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            "historical_m0_target_token_id": 31745,
            "acceptance_gate": "checkpoint-local native OPEN token-set membership plus exact direct arm-token equality",
            "acceptance_gate_source": "src/gripper_attack/attack_adapter.py::_select_strict_arm_candidate",
            "open_region_comparator_path": "configs/fec_attack_v5_open_region.yaml",
            "open_region_comparator_sha256": sha256(ROOT / "configs/fec_attack_v5_open_region.yaml"),
            "historical_surrogate_score_path": "cached_autoregressive_generate_v1",
            "target_token_only_vs_set_alignment": "SINGLE_TOKEN_OBJECTIVE_MISALIGNED_WITH_NATIVE_OPEN_SET",
        },
        "sealed_score_evidence": {
            "expected_margin_fields": sorted(expected_margin_fields),
            "observed_candidate_fields": sorted(observed_fields),
            "margin_fields_present": sorted(expected_margin_fields & observed_fields),
            "margin_fields_missing": sorted(expected_margin_fields - observed_fields),
            "checkpoint_local_open_set_reconstructible_without_model_or_runtime": False,
            "classification": "NOT_IDENTIFIABLE_FROM_SEALED_EVIDENCE",
        },
        "predeclared_y1_classifications": [
            "SINGLE_TOKEN_OBJECTIVE_MISALIGNED_WITH_NATIVE_OPEN_SET",
            "NOT_IDENTIFIABLE_FROM_SEALED_EVIDENCE",
        ],
        "not_established": [
            "SURROGATE_PATH_MISMATCH",
            "OPTIMIZATION_BUDGET_LIMITED",
            "TRUE_MODEL_TARGETABILITY_LIMIT",
        ],
    }


def main() -> int:
    errors: list[str] = []
    g10_rows = [json.loads(line) for line in G10.read_text(encoding="utf-8").splitlines() if line.strip()]
    fresh_rows = [row for row in g10_rows if row.get("fresh_after_exclusion") is True]
    fresh_keys = {row["canonical_parent_key"] for row in fresh_rows}
    if len(g10_rows) != 1200:
        errors.append(f"G10_ROW_COUNT:{len(g10_rows)}")
    if len(fresh_rows) != 210:
        errors.append(f"G10_FRESH_COUNT:{len(fresh_rows)}")

    consumed_by_source: dict[str, dict[str, Any]] = {}
    consumed_union: set[str] = set()
    missing_sources: list[str] = []
    for relative in EXCLUSION_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            missing_sources.append(relative)
            continue
        keys = source_keys(path)
        consumed_by_source[relative] = {
            "sha256": sha256(path),
            "key_count": len(keys),
            "fresh_overlap_count": len(keys & fresh_keys),
            "fresh_overlap_by_suite": suite_counts(keys & fresh_keys),
        }
        consumed_union |= keys
    if missing_sources:
        errors.extend(f"MISSING_SOURCE:{path}" for path in missing_sources)

    remaining = fresh_keys - consumed_union
    available = suite_counts(remaining)
    deficits = {suite: max(0, F1_REQUIRED[suite] - available[suite]) for suite in SUITES}
    status = (
        "HOLD_F1A_FRESH_POPULATION_INSUFFICIENT"
        if any(deficits.values()) or errors
        else "PASS_F1A_POPULATIONS_FROZEN"
    )

    report = {
        "schema": "STAGE_X_X1R2_F1A_STATIC_FEASIBILITY_AUDIT_V1",
        "status": status,
        "scope": "CPU/static/offline only; no GPU, model load, inference, simulator, env.step, PGD, V_phys, or protected read",
        "authority": {
            "pull_request": 135,
            "controlling_comment_id": 5367840153,
            "clarification_comment_id": 5367846534,
            "gate": "STAGE_X_X1R2_F1_MATCHED_CVE_BRIDGE",
            "subgate": "F1-A",
            "paper_v1_immutable_root_sha256": "830c55dee96477e87c36437a33760a9dced0d1217ea01b3e72905f26fd142336",
        },
        "source": {
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "g10_path": G10.relative_to(ROOT).as_posix(),
            "g10_sha256": sha256(G10),
            "g10_state_range": "20..49",
            "g10_source_is_jsonl": True,
        },
        "g10_population": {
            "rows": len(g10_rows),
            "excluded_union_rows": sum(row.get("excluded_union") is True for row in g10_rows),
            "fresh_rows": len(fresh_rows),
            "fresh_by_suite": suite_counts(fresh_keys),
            "g10_authority_reconciliation": "1200 -> 990 exclusion union -> 210 fresh",
        },
        "post_g10_consumption": {
            "source_count": len(consumed_by_source),
            "source_bindings": consumed_by_source,
            "consumed_union_key_count": len(consumed_union),
            "fresh_consumed_union_overlap_count": len(consumed_union & fresh_keys),
            "remaining_fresh_count": len(remaining),
            "remaining_fresh_by_suite": available,
            "remaining_key_sha256": hashlib.sha256(
                ("\n".join(sorted(remaining)) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "requested_populations": {
            "dev": {"count": 24, "per_suite": 6},
            "bridge": {"count": 32, "per_suite": 8},
            "combined_required_by_suite": F1_REQUIRED,
            "disjoint_required": True,
            "no_replacement_or_top_up": True,
        },
        "feasibility": {
            "available_by_suite": available,
            "required_by_suite": F1_REQUIRED,
            "deficit_by_suite": deficits,
            "exact_f1a_freeze_possible": not any(deficits.values()) and not errors,
            "reason": (
                "libero_goal has only 5 remaining fresh G10 identities but requires 14 "
                "disjoint DEV+BRIDGE identities; exact F1-A cannot be frozen."
            )
            if deficits.get("libero_goal")
            else "all exact per-suite population requirements are met",
        },
        "forbidden_repairs": [
            "reuse_any_previous_engineering_identity",
            "replacement_or_top_up",
            "suite_substitution",
            "state_range_expansion_to_0..19_without_new_scientific_authority",
            "running_F1_B_before_exact_F1_A_freeze",
        ],
        "y1": candidate_y1(),
        "errors": errors,
        "protected_boundary": {
            "gpu": 0,
            "openvla_inference": 0,
            "simulator": 0,
            "env_step": 0,
            "pgd": 0,
            "backward": 0,
            "physical_intervention": 0,
            "vphys": 0,
            "eval160": "UNREAD",
            "protected": "UNREAD",
        },
        "next_legal_action": (
            "PI review required for a new identity source or changed exact cohort sizes; do not run F1-B GPU."
            if status != "PASS_F1A_POPULATIONS_FROZEN"
            else "F1-B remains separately gated after this sealed F1-A artifact is published."
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "available_by_suite": available, "deficit_by_suite": deficits}, sort_keys=True))
    return 0 if status == "PASS_F1A_POPULATIONS_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
