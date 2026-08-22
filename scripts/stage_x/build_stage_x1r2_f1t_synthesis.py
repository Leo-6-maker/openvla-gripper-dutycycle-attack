#!/usr/bin/env python3
"""Build the sealed, offline F1T synthesis without running any experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
F1A3_ROOT = ROOT / "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json"
F1B_DECISION = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_DECISION_V3.json"
F1B_COMPARISON = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_METHOD_COMPARISON_V3.json"
F1B_ROOT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_RESULT_AGGREGATION_V3_20260821/F1B_DEV_ROOT_SEAL_V3.json"
F1B_FREEZE_ROOT = ROOT / "reports/STAGE_X_X1R2_F1B_DEV_METHOD_FREEZE_V3_20260821/F1B_ROOT_SEAL_V3.json"
F1C_DECISION = ROOT / "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821/F1C_T5_CANARY_DECISION_V3.json"
F1C_ROOT = ROOT / "reports/STAGE_X_X1R2_F1C_T5_CANARY_RESULT_AGGREGATION_V3_20260821/F1C_T5_CANARY_ROOT_SEAL_V3.json"
F1C_FREEZE_ROOT = ROOT / "reports/STAGE_X_X1R2_F1C_METHOD_FREEZE_T5_CANARY_V3_20260821/F1C_ROOT_SEAL_V3.json"
F1C_REPAIR = ROOT / "reports/STAGE_X_X1R2_F1C_REPAIR_STATIC_AUDIT_V1_20260822/F1C_REPAIR_STATIC_AUDIT_V1.json"
F1C4_STATIC_ROOT = ROOT / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_ROOT_SEAL_V1.json"
F1C4_AUDIT = ROOT / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_RUNTIME_AUDIT_V1.json"
F1C4_DECISION = ROOT / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_TERMINAL_DECISION_V1.json"
F1C4_RESULT_ROOT = ROOT / "reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_RESULT_ROOT_SEAL_V1.json"
PAPER_V1_ROOT = ROOT / "paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json"
PAPER_V1_CLAIMS = ROOT / "paper/PAPER_V1_CLAIM_LEDGER_V1.json"
F1T_GATE = "STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_AND_PAPER_V2_DELTA_V1"
F1T_STATUS = "F1T_TERMINAL_SYNTHESIS_SEALED_FOR_PI"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def paper_tree_listing_sha() -> str:
    lines = [
        line
        for line in git("ls-tree", "-r", "HEAD", "--", "paper").splitlines()
        if line.split("\t", 1)[-1].startswith("paper/PAPER_V1_")
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def authority_entry(path: Path, *, scope: str, immutable: bool = False) -> dict[str, Any]:
    data = load(path) if path.suffix == ".json" else {}
    return {
        "path": rel(path),
        "sha256": sha(path),
        "schema": data.get("schema"),
        "status": data.get("status"),
        "source_commit": data.get("source_commit"),
        "source_tree": data.get("source_tree"),
        "scope": scope,
        "immutable": immutable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()

    output_paths = {
        "synthesis": output_root / "reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json",
        "claims": output_root / "reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json",
        "authority": output_root / "reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json",
        "summary": output_root / "reports/STAGE_X_X1R2_F1T_DEV_C4_SUMMARY_V1.json",
        "handoff": output_root / "docs/handoffs/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_20260822.md",
        "paper_delta": output_root / "paper/PAPER_V2_F1_DELTA_FROM_V1.md",
        "root": output_root / "reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json",
        "sidecar": output_root / "reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.sha256",
    }
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("F1T_OUTPUT_ALREADY_EXISTS")

    f1a3 = load(F1A3_ROOT)
    f1b_decision = load(F1B_DECISION)
    f1b_comparison = load(F1B_COMPARISON)
    f1b_root = load(F1B_ROOT)
    f1b_freeze_root = load(F1B_FREEZE_ROOT)
    f1c_decision = load(F1C_DECISION)
    f1c_root = load(F1C_ROOT)
    f1c_freeze_root = load(F1C_FREEZE_ROOT)
    f1c_repair = load(F1C_REPAIR)
    f1c4_static = load(F1C4_STATIC_ROOT)
    f1c4_audit = load(F1C4_AUDIT)
    f1c4_decision = load(F1C4_DECISION)
    f1c4_result_root = load(F1C4_RESULT_ROOT)
    paper_v1_root = load(PAPER_V1_ROOT)
    paper_v1_claims = load(PAPER_V1_CLAIMS)

    paper_binding = f1a3["paper_v1_binding"]
    current_paper_tree_sha = paper_tree_listing_sha()
    paper_v1_diff = [
        line for line in git("diff", "--name-only", "HEAD", "--", "paper").splitlines()
        if line.startswith("paper/PAPER_V1_")
    ]
    paper_v1_staged_diff = [
        line for line in git("diff", "--cached", "--name-only", "HEAD", "--", "paper").splitlines()
        if line.startswith("paper/PAPER_V1_")
    ]
    paper_v1_unchanged = (
        current_paper_tree_sha == paper_binding["paper_v1_tree_listing_sha256"]
        and not paper_v1_diff
        and not paper_v1_staged_diff
        and len(paper_binding["tracked_files"]) == paper_binding["tracked_file_count"]
    )
    if not paper_v1_unchanged:
        raise SystemExit("PAPER_V1_IMMUTABILITY_AUDIT_FAILED")

    # Bind BRIDGE only through the F1A3 root metadata. Do not open the BRIDGE ledger.
    bridge_paths = [
        path for path in f1a3.get("artifact_hashes", {})
        if "BRIDGE_V3" in path and "LEDGER" in path
    ]
    if len(bridge_paths) != 1:
        raise SystemExit(f"BRIDGE_AUTHORITY_BINDING_INVALID:{bridge_paths}")
    bridge_path = bridge_paths[0]

    best = f1b_comparison["best_by_method"]
    method_rows = []
    for method in ("M0", "M1", "M2"):
        row = best[method]
        method_rows.append({
            "method": method,
            "iterations": row["iterations"],
            "min_per_suite_successful_parent_count": row["min_per_suite_parent_success"],
            "total_successful_dev_parent_count": row["parent_success_count"],
            "strict_valid_probe_count_secondary": row["strict_valid_probe_count"],
            "mean_selected_linf_secondary": row["mean_selected_linf"],
            "complexity_rank": row["complexity_rank"],
            "per_suite_successful_parent_count": row["per_suite_parent_success_count"],
            "status_counts": row["status_counts"],
        })
    m1, m2 = best["M1"], best["M2"]
    f1b_summary = {
        "unit": "parent",
        "dev_parent_count": f1b_comparison["population"]["parent_count"],
        "per_suite_parent_count": f1b_comparison["population"]["per_suite_count"],
        "methods": method_rows,
        "advancement_over_m0_first_two_criteria": f1b_comparison["strict_improvement_over_m0_on_first_two_criteria"],
        "selected_method": "M1",
        "selection_reason": [
            "M1-10 and M2-10 both strictly improved over M0-10 on the first two preregistered parent-level criteria.",
            "M1-10 and M2-10 tied on minimum per-suite successful parents (1) and total successful DEV parents (5).",
            f"M1-10 then won the frozen lexicographic tie-break with lower mean selected Linf ({m1['mean_selected_linf']}) than M2-10 ({m2['mean_selected_linf']}) and the lower complexity rank ({m1['complexity_rank']} vs {m2['complexity_rank']}).",
        ],
        "scientific_scope": "engineering/model-side method development only",
        "source_decision_sha256": sha(F1B_DECISION),
    }

    parents = f1c4_audit["parents"]
    strict_parent_keys = []
    completed_zero_strict_keys = []
    hold_parent_keys = []
    for parent in parents:
        strict_steps = sum(int(parent["arms"].get(arm, {}).get("strict_valid_steps", 0)) for arm in ("none", "prev_delta"))
        if parent["status"] == "HOLD_F1C_PARENT":
            hold_parent_keys.append(parent["canonical_parent_key"])
        elif strict_steps > 0:
            strict_parent_keys.append(parent["canonical_parent_key"])
        else:
            completed_zero_strict_keys.append(parent["canonical_parent_key"])
    f1c4_totals = f1c4_decision["totals"]
    f1c4_summary = {
        "unit_policy": "parent primary; candidate and step rows are repeated-within-parent engineering diagnostics, not iid observations",
        "parent_denominator": len(parents),
        "parent_with_at_least_one_strict_valid_executed_step": len(strict_parent_keys),
        "parent_with_at_least_one_strict_valid_executed_step_keys": strict_parent_keys,
        "completed_parent_with_zero_strict_valid_executed_steps": len(completed_zero_strict_keys),
        "replay_hold_parent_count": len(hold_parent_keys),
        "replay_hold_parent_keys": hold_parent_keys,
        "completed_temporal_arm_count": f1c4_decision["completed_arm_count"],
        "temporal_arm_denominator": 16,
        "attempted_step_count": f1c4_totals["attempted_step_count"],
        "candidate_audit_complete_step_count": f1c4_totals["candidate_evidence_complete_steps"],
        "candidate_row_count_diagnostic_only": f1c4_totals["candidate_evidence_rows"],
        "clean_fallback_step_count": f1c4_totals["clean_fallback_steps"],
        "strict_valid_attacked_step_count": f1c4_totals["strict_valid_steps"],
        "pgd_call_count": f1c4_totals["pgd_calls"],
        "attacked_env_step_count": f1c4_totals["attacked_env_steps"],
        "vphys_reads": f1c4_totals["vphys_reads"],
        "physical_interventions": f1c4_totals["physical_interventions"],
        "attack_outcome_reads": f1c4_decision["no_physical_or_protected_use"]["attack_outcome_reads"],
        "temporal_selection": "NOT_APPLICABLE_TERMINAL_HOLD",
        "terminal_status": f1c4_decision["status"],
        "result_root_sha256": sha(F1C4_RESULT_ROOT),
    }

    authority_entries = [
        authority_entry(PAPER_V1_ROOT, scope="immutable Paper V1 final root seal", immutable=True),
        authority_entry(PAPER_V1_CLAIMS, scope="immutable Paper V1 claim ledger", immutable=True),
        authority_entry(F1A3_ROOT, scope="F1-A3 source split, role authority, and Paper V1 binding", immutable=True),
        authority_entry(F1B_FREEZE_ROOT, scope="F1-B selected-method static freeze", immutable=True),
        authority_entry(F1B_COMPARISON, scope="F1-B preregistered DEV method comparison", immutable=True),
        authority_entry(F1B_ROOT, scope="F1-B DEV parent-level result root", immutable=True),
        authority_entry(F1B_DECISION, scope="F1-B method selection decision", immutable=True),
        authority_entry(F1C_FREEZE_ROOT, scope="historical F1-C V3 method freeze", immutable=True),
        authority_entry(F1C_ROOT, scope="historical F1-C V3 executable-evidence HOLD", immutable=True),
        authority_entry(F1C_DECISION, scope="historical F1-C V3 decision", immutable=True),
        authority_entry(F1C_REPAIR, scope="post-HOLD contract-preserving repair audit", immutable=True),
        authority_entry(F1C4_STATIC_ROOT, scope="fresh F1-C4 static namespace/root seal", immutable=True),
        authority_entry(F1C4_AUDIT, scope="fresh F1-C4 runtime audit", immutable=True),
        authority_entry(F1C4_DECISION, scope="fresh F1-C4 terminal decision", immutable=True),
        authority_entry(F1C4_RESULT_ROOT, scope="fresh F1-C4 runtime/result root seal", immutable=True),
    ]
    authority_map = {
        "schema": "STAGE_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1",
        "status": F1T_STATUS,
        "gate": F1T_GATE,
        "live_source": {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}")},
        "paper_v1": {
            "immutable": True,
            "final_root_path": rel(PAPER_V1_ROOT),
            "final_root_sha256": sha(PAPER_V1_ROOT),
            "final_root_status": paper_v1_root["status"],
            "claim_ledger_path": rel(PAPER_V1_CLAIMS),
            "claim_ledger_sha256": sha(PAPER_V1_CLAIMS),
            "claim_count": paper_v1_claims["claim_count"],
            "tree_listing_sha256_current": current_paper_tree_sha,
            "tree_listing_sha256_f1a3_binding": paper_binding["paper_v1_tree_listing_sha256"],
            "tracked_file_count": paper_binding["tracked_file_count"],
            "working_tree_unchanged": paper_v1_unchanged,
        },
        "bridge_v3": {
            "content_read": False,
            "authority_path": bridge_path,
            "authority_sha256_from_f1a3_root": f1a3["artifact_hashes"][bridge_path],
            "role_count_by_suite": {suite: counts["BRIDGE_V3"] for suite, counts in f1a3["role_counts"].items()},
            "bridge_runtime": f1a3["protected_boundary"]["bridge_runtime"],
            "bridge_outcome_read": f1a3["protected_boundary"]["bridge_outcome_read"],
            "disposition": "sealed_and_unopened; separate PI authorization required",
        },
        "f1c4_runtime_manifest": {
            "local_audit_root": f1c4_audit["runtime"]["local_audit_root"],
            "remote_root": f1c4_audit["runtime"]["remote_root"],
            "file_count": f1c4_audit["runtime"]["file_count"],
            "manifest_sha256": f1c4_audit["runtime"]["manifest_sha256"],
            "attested_by": rel(F1C4_AUDIT),
            "disposition": "runtime manifest attestation sealed in F1-C4 audit; no new runtime read",
        },
        "entries": authority_entries,
        "source_bindings": {
            "f1a3": {"commit": f1a3["source_commit"], "tree": f1a3["source_tree"]},
            "f1b_dev": {"commit": f1b_root["source_commit"], "tree": f1b_root["source_tree"]},
            "f1c_historical_runtime": {"commit": f1c_root["source_commit"], "tree": f1c_root["source_tree"]},
            "f1c4_static": {"commit": f1c4_static["source_commit"], "tree": f1c4_static["source_tree"]},
            "f1c4_runtime_workers": {
                "commit": f1c4_audit["source"]["commit"],
                "tree": f1c4_audit["source"]["tree"],
                "all_workers_same_source": f1c4_audit["source"]["all_workers_same_source"],
            },
            "f1c4_result_seal": {"commit": f1c4_result_root["source_commit"], "tree": f1c4_result_root["source_tree"]},
        },
        "protected_boundary": {
            "new_openvla_inference": 0,
            "new_simulator_or_env_step": 0,
            "new_pgd_or_backward": 0,
            "new_adversarial_image_generation": 0,
            "new_physical_intervention": 0,
            "new_vphys_read": 0,
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
        },
    }

    claim_delta = {
        "schema": "STAGE_X1R2_F1T_CLAIM_LEDGER_DELTA_V1",
        "status": F1T_STATUS,
        "gate": F1T_GATE,
        "paper_v1_immutable": True,
        "promotable_claims": [
            {
                "id": "F1T-P01",
                "claim": "M1-10 improved over M0-10 on the preregistered first two DEV parent-level advancement criteria.",
                "scope": "engineering/model-side method development",
                "evidence": [rel(F1B_DECISION), rel(F1B_ROOT)],
                "restrictions": ["not physical efficacy", "not protected evaluation"],
            },
            {
                "id": "F1T-P02",
                "claim": "Under the frozen M1-10 direct-token contract, strict selective visual execution was observed at least once in the fresh F1-C4 canaries.",
                "scope": "bounded E_t(single) execution-layer evidence",
                "evidence": [rel(F1C4_AUDIT), rel(F1C4_RESULT_ROOT)],
                "restrictions": ["one step only", "engineering canary diagnostic", "no physical efficacy"],
            },
            {
                "id": "F1T-P03",
                "claim": "Full T5 executable qualification was not established under the finite F1 track.",
                "scope": "terminal executable qualification result",
                "evidence": [rel(F1C4_DECISION), rel(F1C4_RESULT_ROOT)],
                "restrictions": ["do not reinterpret as targetability failure"],
            },
            {
                "id": "F1T-P04",
                "claim": "Fail-closed clean fallback operated on the other 69 attempted steps, with no invalid candidate promoted to the attack path.",
                "scope": "F1-C4 runtime contract behavior",
                "evidence": [rel(F1C4_AUDIT)],
                "restrictions": ["candidate/step rows are repeated engineering diagnostics, not iid observations"],
            },
            {
                "id": "F1T-P05",
                "claim": "BRIDGE/F1-D was never opened because the executable qualification gate did not pass.",
                "scope": "governance and protected-boundary disposition",
                "evidence": [rel(F1A3_ROOT), rel(F1C4_DECISION)],
                "restrictions": ["future bridge use requires separate PI authorization"],
            },
        ],
        "not_promotable_claims": [
            "reliable or generalizable T5 selective delivery",
            "cross-suite visual attack capability",
            "F1 physical attack efficacy",
            "prevalence estimates from the 8 engineering canaries",
            "temporal-init superiority",
            "a claim that the one attacked env.step caused physical failure",
            "any V_phys, task-outcome, Eval160, or protected conclusion",
            "a causal claim that M1 resolves the Paper V1 factorization gap",
        ],
        "factorization_layers": {
            "C_t": "clean timing/opportunity",
            "V_t(d)": "physical command-OPEN vulnerability",
            "E_t(single)": "single-state strict model-side realization",
            "E_t(T5)": "strict selective delivery across five attempted steps",
            "Y_t(vis)": "physical response of the full visual pipeline",
        },
        "f1t_interpretation": "single-step strict realizability != sustained T5 selective delivery != matched physical efficacy",
        "paper_v1_root_sha256": sha(PAPER_V1_ROOT),
    }

    summary = {
        "schema": "STAGE_X1R2_F1T_DEV_C4_SUMMARY_V1",
        "status": F1T_STATUS,
        "gate": F1T_GATE,
        "unit_policy": "parent-level primary; repeated candidate/step rows diagnostic only",
        "f1b_dev": f1b_summary,
        "f1c4_canary": f1c4_summary,
        "temporal_selection": "not selected; terminal HOLD prevents superiority claim",
        "protected_boundary": authority_map["protected_boundary"],
    }

    synthesis = {
        "schema": "STAGE_X1R2_F1T_TERMINAL_SYNTHESIS_V1",
        "status": F1T_STATUS,
        "gate": F1T_GATE,
        "scientific_disposition": "F1 experimental track closed under the finite namespace and stop-loss",
        "paper_v1_conclusion_preserved": True,
        "headline": "DEV-level targetability improved under M1; isolated strict execution exists; reliable full-T5 executable delivery was not established before terminal stop.",
        "f1b_dev": f1b_summary,
        "f1c4_canary": f1c4_summary,
        "claim_boundary": claim_delta,
        "authority_map_path": rel(output_paths["authority"]),
        "protected_boundary": authority_map["protected_boundary"],
        "mandatory_stop": {
            "f1c5": False,
            "identity_recycle_or_top_up": False,
            "method_tuning": False,
            "bridge_v3_or_f1d": False,
            "r0_r1_r2": False,
            "vphys_eval160_protected": False,
        },
    }

    handoff = f"""# F1T terminal synthesis — 2026-08-22

Gate: `{F1T_GATE}`
Status: `{F1T_STATUS}`

## Decision

The F1 experimental track is closed under its finite namespace and predeclared stop-loss. Paper V1 remains immutable. The result is not a physical negative: F1-C4 produced one strict-valid executed step, but reliable sustained T5 delivery was not established because one fresh parent hit replay observation-hash failure in both temporal arms.

## Quantitative summary

- F1-B DEV denominator: 24 parents, 6 per suite.
- M1-10 and M2-10 both improved over M0-10 on the first two parent-level criteria.
- M1-10 was selected by the frozen lexicographic rule after the M1/M2 tie: lower mean selected L-infinity and lower complexity rank.
- F1-C4: 8 parents; 7 completed, 1 replay-HOLD; 14/16 arms completed.
- One completed parent had a strict-valid executed step; six completed parents had zero strict-valid steps.
- 70 attempted steps, 70 complete candidate audits, 770 candidate rows, 69 clean fallbacks, 1 strict-valid/attacked step.
- No V_phys, physical intervention, attack-outcome, Eval160, or protected read.

## Claim boundary

The promotable execution-layer statement is `E_t(single)` evidence only. It does not establish `E_t(T5)`, `V_t(5)`, or `Y_t(vis)`. Candidate and step rows are repeated within-parent engineering diagnostics, not iid observations; no significance test is appropriate.

## Governance

BRIDGE_V3 remains sealed and unopened. No F1-C5, canary recycling/top-up, tuning, F1-D, BRIDGE execution, R0/R1/R2, V_phys, Eval160, or protected evaluation is authorized by this gate.

Authoritative machine-readable files:

- `{rel(output_paths['synthesis'])}`
- `{rel(output_paths['claims'])}`
- `{rel(output_paths['authority'])}`
- `{rel(output_paths['root'])}`
"""

    paper_delta = f"""# Paper V2 F1 delta from Paper V1

This is an append-only delta. Every `paper/PAPER_V1_*` file and its final root seal remain immutable.

## What the prospective F1 follow-up added

Paper V1's mechanism-first / factorization-gap conclusion remains intact. The prospective F1 follow-up improved model-side targetability on a development cohort and demonstrated isolated strict selective execution, but the finite fresh-canary qualification did not establish reliable sustained T5 delivery, so the matched physical bridge was never opened.

The execution-layer separation is:

`single-step strict realizability` != `sustained T5 selective delivery` != `matched physical efficacy`

## F1-B development result

On the frozen 24-parent DEV denominator, M1-10 and M2-10 both strictly improved over M0-10 on the first two preregistered parent-level criteria. M1-10 and M2-10 tied on those two criteria; M1-10 was selected by the frozen lexicographic rule because its mean selected L-infinity was lower and its complexity rank was lower.

This is engineering/model-side method-development evidence only.

## F1-C4 terminal result

The fresh canary denominator was 8 parents. Seven parents completed; one parent stopped in both temporal arms at `F1C_REPLAY_OBSERVATION_HASH_MISMATCH` before interpretable T5 qualification. One completed parent produced one strict-valid executed step. Six completed parents produced zero strict-valid executed steps. Across the completed arms there were 70 attempted steps, 70 complete candidate audits, 770 diagnostic candidate rows, 69 clean fallbacks, and one strict-valid attacked step.

The one attacked `env.step` is not a physical efficacy result. No V_phys, physical outcome, task outcome, Eval160, or protected evaluation was read. F1-C4 therefore remains a terminal executable-evidence HOLD, not a claim that M1 is physically ineffective and not a qualification PASS.

## Claim boundary

F1 adds bounded evidence at the `E_t(single)` / execution layer. It does not establish `E_t(T5)`, `V_t(5)`, or `Y_t(vis)`, does not establish temporal-init superiority, and does not resolve the Paper V1 factorization gap causally. Candidate and step rows are repeated engineering diagnostics, not iid scientific observations.

## Disposition

F1 is closed under the finite stop-loss. BRIDGE_V3 remains sealed and unopened. Any future physical bridge or visual-method proposal requires separate prospective PI authorization.
"""

    write_json(output_paths["authority"], authority_map)
    write_json(output_paths["claims"], claim_delta)
    write_json(output_paths["summary"], summary)
    write_json(output_paths["synthesis"], synthesis)
    write_text(output_paths["handoff"], handoff)
    write_text(output_paths["paper_delta"], paper_delta)

    artifact_paths = [
        output_paths["synthesis"],
        output_paths["claims"],
        output_paths["authority"],
        output_paths["summary"],
        output_paths["handoff"],
        output_paths["paper_delta"],
        ROOT / "scripts/stage_x/build_stage_x1r2_f1t_synthesis.py",
        PAPER_V1_ROOT,
        F1A3_ROOT,
        F1B_DECISION,
        F1B_ROOT,
        F1C_ROOT,
        F1C_REPAIR,
        F1C4_STATIC_ROOT,
        F1C4_AUDIT,
        F1C4_DECISION,
        F1C4_RESULT_ROOT,
    ]
    root_seal = {
        "schema": "STAGE_X1R2_F1T_ROOT_SEAL_V1",
        "status": F1T_STATUS,
        "gate": F1T_GATE,
        "artifact_hashes": dict(sorted({rel(path): sha(path) for path in artifact_paths}.items())),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "paper_v1_tree_listing_sha256": current_paper_tree_sha,
        "paper_v1_root_sha256": sha(PAPER_V1_ROOT),
        "f1c4_result_root_sha256": sha(F1C4_RESULT_ROOT),
        "f1c4_runtime_manifest_sha256": f1c4_audit["runtime"]["manifest_sha256"],
        "protected_boundary": authority_map["protected_boundary"],
        "mandatory_stop": synthesis["mandatory_stop"],
        "seal_scope_excludes_sidecar": True,
    }
    write_json(output_paths["root"], root_seal)
    output_paths["sidecar"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["sidecar"].write_text(f"{sha(output_paths['root'])}  {output_paths['root'].name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": F1T_STATUS, "root_seal_sha256": sha(output_paths["root"]), "paper_v1_unchanged": paper_v1_unchanged, "f1c4_terminal_status": f1c4_decision["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
