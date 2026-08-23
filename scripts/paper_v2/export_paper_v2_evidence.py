#!/usr/bin/env python3
"""Export deterministic Paper V2 JSON from committed sealed authorities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "exports/paper_v2"
GENERATOR = "scripts/paper_v2/export_paper_v2_evidence.py"
REPOSITORY = "Leo-6-maker/openvla-gripper-dutycycle-attack"
MANIFEST = "PAPER_V2_EXPORT_MANIFEST_V1.json"

PAPER_MAP = "paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json"
PAPER_CLAIMS = "paper/PAPER_V1_CLAIM_LEDGER_V1.json"
PAPER_SUPPLEMENT = "paper/PAPER_V1_SUPPLEMENT_BINDINGS_V1.json"
PAPER_FIGURE_MANIFEST = "paper/PAPER_V1_FIGURE_TABLE_MANIFEST_V1.json"
PAPER_ROOT = "paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json"
FIGURE2 = "paper/data/PAPER_V1_FIGURE2_X0_DOSE_RESPONSE.csv"
FIGURE3 = "paper/data/PAPER_V1_FIGURE3_TIMING_NEGATIVE_CASCADE.csv"
FIGURE4 = "paper/data/PAPER_V1_FIGURE4_FACTORIZATION_GAP.csv"
FIGURE5 = "paper/data/PAPER_V1_FIGURE5_E3_E4_PARENT_REALIZABILITY.csv"
F1T_MAP = "reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json"
F1T_SUMMARY = "reports/STAGE_X_X1R2_F1T_DEV_C4_SUMMARY_V1.json"
F1T_CLAIMS = "reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json"
F1T_ROOT = "reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json"
STAGE_Z_ROOT = "reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json"

AUTHORITY_INPUTS = (
    PAPER_MAP,
    PAPER_CLAIMS,
    PAPER_SUPPLEMENT,
    PAPER_FIGURE_MANIFEST,
    PAPER_ROOT,
    FIGURE2,
    FIGURE3,
    FIGURE4,
    FIGURE5,
    F1T_MAP,
    F1T_SUMMARY,
    F1T_CLAIMS,
    F1T_ROOT,
    STAGE_Z_ROOT,
)
OUTPUT_NAMES = (
    "evidence_hierarchy.json",
    "x0_mechanism.json",
    "timing_generalization_cascade.json",
    "stage_ix_factorization_gap.json",
    "execution_layer_evidence.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"PAPER_V2_EXPORT_FAIL:{message}")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def git_bytes(*args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        error = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"PAPER_V2_EXPORT_FAIL:git:{' '.join(args)}:{error}")
    return process.stdout


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8").strip()


class Snapshot:
    """Read canonical bytes from one committed Git snapshot."""

    def __init__(self, source_ref: str):
        self.commit = git_text("rev-parse", f"{source_ref}^{{commit}}")
        self.tree = git_text("rev-parse", f"{self.commit}^{{tree}}")
        self.timestamp = git_text("show", "-s", "--format=%cI", self.commit)
        self._bytes: dict[str, bytes] = {}

    def bytes(self, path: str) -> bytes:
        if path not in self._bytes:
            self._bytes[path] = git_bytes("show", f"{self.commit}:{path}")
        return self._bytes[path]

    def json(self, path: str) -> dict[str, Any]:
        value = json.loads(self.bytes(path).decode("utf-8"))
        require(isinstance(value, dict), f"json_root:{path}")
        return value

    def csv(self, path: str) -> list[dict[str, str]]:
        text = self.bytes(path).decode("utf-8")
        return list(csv.DictReader(io.StringIO(text, newline="")))

    def blob(self, path: str) -> str:
        return git_text("rev-parse", f"{self.commit}:{path}")

    def binding(self, path: str) -> dict[str, str]:
        return {
            "path": path,
            "sha256_git_blob_bytes": sha256(self.bytes(path)),
            "git_blob": self.blob(path),
        }


def declared_basis(value: bytes, expected: str) -> str:
    canonical = sha256(value)
    crlf = sha256(
        value.decode("utf-8")
        .replace("\r\n", "\n")
        .replace("\n", "\r\n")
        .encode("utf-8")
    )
    if expected == canonical:
        return "canonical_git_blob_bytes"
    if expected == crlf:
        return "historical_crlf_materialization"
    raise RuntimeError(
        f"PAPER_V2_EXPORT_FAIL:declared_digest:{expected}:canonical={canonical}:crlf={crlf}"
    )


def add_declared_check(
    checks: dict[str, list[dict[str, str]]],
    snapshot: Snapshot,
    path: str,
    declared_by: str,
    expected: str,
) -> None:
    checks.setdefault(path, []).append(
        {
            "declared_by": declared_by,
            "sha256": expected,
            "digest_basis": declared_basis(snapshot.bytes(path), expected),
        }
    )


def boundary_is_closed(boundary: Mapping[str, Any]) -> bool:
    numeric = [
        value
        for key, value in boundary.items()
        if key not in {"eval160", "protected_evaluation"}
    ]
    return (
        boundary["eval160"] == "UNREAD"
        and boundary["protected_evaluation"] == "UNREAD"
        and all(value == 0 for value in numeric)
    )


def validate(
    snapshot: Snapshot, values: Mapping[str, dict[str, Any]]
) -> dict[str, list[dict[str, str]]]:
    checks: dict[str, list[dict[str, str]]] = {}
    paper_map = values[PAPER_MAP]
    claims = values[PAPER_CLAIMS]
    supplement = values[PAPER_SUPPLEMENT]
    figure_manifest = values[PAPER_FIGURE_MANIFEST]
    paper_root = values[PAPER_ROOT]
    f1_map = values[F1T_MAP]
    f1_summary = values[F1T_SUMMARY]
    f1_claims = values[F1T_CLAIMS]
    f1_root = values[F1T_ROOT]
    stage_z = values[STAGE_Z_ROOT]

    require(paper_map["status"] == "PAPER_V1_EVIDENCE_AUTHORITY_MAP_PASS", "paper_map")
    require(claims["status"] == "PAPER_V1_CLAIM_AUDIT_PASS", "paper_claims")
    require(
        supplement["status"] == "PAPER_V1_SUPPLEMENT_REPRODUCIBILITY_PASS",
        "paper_supplement",
    )
    require(
        figure_manifest["status"] == "PAPER_V1_FIGURE_TABLE_PACKAGE_PASS",
        "paper_figure_manifest",
    )
    require(
        paper_root["status"] == "PAPER_V1_MECHANISM_FACTORIZATION_DRAFT_BUNDLE_READY_FOR_PI",
        "paper_root",
    )
    require(
        boundary_is_closed(paper_map["canonicalization"]["protected_boundary"]),
        "paper_map_boundary",
    )
    require(boundary_is_closed(claims["protected_boundary"]), "paper_claim_boundary")
    require(
        boundary_is_closed(supplement["protected_boundary"]), "paper_supplement_boundary"
    )

    figure_entries = {row["path"]: row for row in figure_manifest["files"]}
    for path in (FIGURE2, FIGURE3, FIGURE4, FIGURE5):
        require(path in figure_entries, f"figure_unbound:{path}")
        add_declared_check(
            checks,
            snapshot,
            path,
            PAPER_FIGURE_MANIFEST,
            str(figure_entries[path]["sha256"]),
        )

    for value, label in (
        (f1_map, "f1_map"),
        (f1_summary, "f1_summary"),
        (f1_claims, "f1_claims"),
        (f1_root, "f1_root"),
    ):
        require(value["status"] == "F1T_TERMINAL_SYNTHESIS_SEALED_FOR_PI", label)
    require(boundary_is_closed(f1_map["protected_boundary"]), "f1_map_boundary")
    require(boundary_is_closed(f1_summary["protected_boundary"]), "f1_summary_boundary")
    root_hashes = f1_root["artifact_hashes"]
    for path in (F1T_MAP, F1T_SUMMARY, F1T_CLAIMS, PAPER_ROOT):
        require(path in root_hashes, f"f1_root_unbound:{path}")
        add_declared_check(checks, snapshot, path, F1T_ROOT, str(root_hashes[path]))
    require(
        f1_summary["f1c4_canary"]["result_root_sha256"]
        == f1_root["f1c4_result_root_sha256"],
        "f1c4_root_binding",
    )

    require(
        stage_z["status"] == "HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED",
        "stage_z_status",
    )
    require(stage_z["scientific_rollout_started"] is False, "stage_z_rollout")
    require(all(value == 0 for value in stage_z["counters"].values()), "stage_z_counters")
    return checks


def source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    binding = {
        key: source[key]
        for key in (
            "source_commit",
            "source_tree",
            "local_artifact_sha256",
            "sealed_artifact_sha256",
        )
    }
    if "immutable_git_blob_sha256" in source:
        binding["immutable_git_blob_sha256"] = source["immutable_git_blob_sha256"]
    return binding


def f1_claim_binding(
    snapshot: Snapshot,
    entries: Mapping[str, Mapping[str, Any]],
    claims: Mapping[str, Mapping[str, Any]],
    claim_ids: Sequence[str],
) -> dict[str, Any]:
    paths: list[str] = []
    bindings: list[dict[str, Any]] = []
    for claim_id in claim_ids:
        require(claim_id in claims, f"f1_claim:{claim_id}")
        for path in claims[claim_id]["evidence"]:
            if path in paths:
                continue
            paths.append(path)
            require(path in entries, f"f1_entry:{path}")
            entry = entries[path]
            bindings.append(
                {
                    **snapshot.binding(path),
                    "declared_sha256": entry["sha256"],
                    "status": entry["status"],
                    "source_commit": entry["source_commit"],
                    "source_tree": entry["source_tree"],
                }
            )
    return {"paths": paths, "bindings": bindings}


def build_hierarchy(
    snapshot: Snapshot,
    paper_map: Mapping[str, Any],
    claims: Mapping[str, Any],
    supplement: Mapping[str, Any],
    f1_map: Mapping[str, Any],
    f1_summary: Mapping[str, Any],
    f1_claim_delta: Mapping[str, Any],
    stage_z: Mapping[str, Any],
) -> dict[str, Any]:
    sources = {row["id"]: row for row in paper_map["sources"]}
    units = supplement["primary_units"]
    metadata = {
        "X0": (
            "X0",
            "Does command-OPEN exposure show a dose/phase physical mechanism?",
            units["x0"],
            "SEALED_PHYSICAL_COMMAND_OPEN_COUNTERFACTUAL",
            ("C201", "C202"),
        ),
        "VI_B2": (
            "VI-B2",
            "Does frozen Student timing generalize on fresh held-out parents?",
            units["vi_b2"],
            "SEALED_HELDOUT_TIMING_EVALUATION",
            ("C203",),
        ),
        "VII": (
            "VII",
            "Does a frozen context detector pass cross-suite promotion gates?",
            units["vii"],
            "SEALED_DEVELOPMENT_TIMING_EVALUATION",
            ("C204",),
        ),
        "VIII": (
            "VIII",
            "Does a relative selector generalize under parent/LOSO gates?",
            units["viii"],
            "SEALED_RELATIVE_SELECTOR_EVALUATION",
            ("C205",),
        ),
        "IX": (
            "IX",
            "Do model-side targetability scores retain factorized timing utility?",
            units["ix"],
            "NO_ENVIRONMENT_MODEL_SIDE",
            ("C206",),
        ),
        "E3_E4": (
            "E3/E4",
            "Can the frozen timing-decoupled method realize strict selective OPEN?",
            units["e3_e4"],
            "NO_ENVIRONMENT_MODEL_SIDE_STRUCTURAL",
            ("C207", "C208"),
        ),
    }
    rows = []
    for source_id in metadata:
        label, question, unit, exposure, claim_ids = metadata[source_id]
        source = sources[source_id]
        for claim_id in claim_ids:
            claim_by_id(claims, claim_id, source_id)
        rows.append(
            {
                "stage": label,
                "question": question,
                "evidence_class": source["promotional_status"],
                "primary_unit": unit,
                "denominator_censoring": source["population_or_denominator"],
                "environment_exposure": exposure,
                "status": source["status"],
                "promotable_wording_key": list(claim_ids),
                "authority_paths": [PAPER_CLAIMS, *source["source_paths"]],
                "authority_binding": {
                    "claim_ledger": snapshot.binding(PAPER_CLAIMS),
                    "source": source_binding(source),
                },
            }
        )

    f1_entries = {row["path"]: row for row in f1_map["entries"]}
    f1_claims = {row["id"]: row for row in f1_claim_delta["promotable_claims"]}
    f1b = f1_summary["f1b_dev"]
    f1b_authority = f1_claim_binding(
        snapshot, f1_entries, f1_claims, ("F1T-P01",)
    )
    f1b_decision = next(
        row for row in f1_entries.values() if row["path"].endswith("F1B_DEV_DECISION_V3.json")
    )
    rows.append(
        {
            "stage": "F1-B",
            "question": "Does a frozen DEV method improve preregistered parent-level targetability criteria?",
            "evidence_class": "engineering_model_side_method_development",
            "primary_unit": f1b["unit"],
            "denominator_censoring": {
                "dev_parents": f1b["dev_parent_count"],
                "parents_per_suite": f1b["per_suite_parent_count"],
                "methods_kept_separate": True,
            },
            "environment_exposure": "NO_ENVIRONMENT_MODEL_SIDE_DEV",
            "status": f1b_decision["status"],
            "selected_method": f1b["selected_method"],
            "promotable_wording_key": ["F1T-P01"],
            "authority_paths": f1b_authority["paths"],
            "authority_binding": f1b_authority["bindings"],
        }
    )

    f1c_ids = ("F1T-P02", "F1T-P03", "F1T-P04", "F1T-P05")
    f1c = f1_summary["f1c4_canary"]
    f1c_authority = f1_claim_binding(snapshot, f1_entries, f1_claims, f1c_ids)
    rows.append(
        {
            "stage": "F1-C4/F1T",
            "question": "Does the frozen method sustain interpretable T5 selective delivery in fresh canaries?",
            "evidence_class": "terminal_bounded_execution_qualification",
            "primary_unit": f1c["unit_policy"],
            "denominator_censoring": {
                "parents": f1c["parent_denominator"],
                "completed_temporal_arms": f1c["completed_temporal_arm_count"],
                "temporal_arms": f1c["temporal_arm_denominator"],
                "replay_hold_parents": f1c["replay_hold_parent_count"],
            },
            "environment_exposure": {
                "attacked_env_steps": f1c["attacked_env_step_count"],
                "physical_interventions": f1c["physical_interventions"],
                "attack_outcome_reads": f1c["attack_outcome_reads"],
            },
            "status": f1c["terminal_status"],
            "promotable_wording_key": list(f1c_ids),
            "authority_paths": f1c_authority["paths"],
            "authority_binding": f1c_authority["bindings"],
        }
    )

    population = stage_z["population"]
    rows.append(
        {
            "stage": "Stage Z pending",
            "question": "Can the frozen physical mechanism replicate across the prospective model panel?",
            "evidence_class": "engineering_authority_hold_pending",
            "primary_unit": "prospective model-parent pair; no scientific observation",
            "denominator_censoring": {
                "shared_fresh_identities": population["shared_fresh_identities"],
                "nominal_task_cells": population["nominal_task_cells"],
                "structural_missing_cells": population["structural_missing_cells"],
            },
            "environment_exposure": {
                "scientific_rollout_started": stage_z["scientific_rollout_started"],
                "counters": stage_z["counters"],
            },
            "status": stage_z["status"],
            "promotable_wording_key": None,
            "authority_paths": [STAGE_Z_ROOT],
            "authority_binding": {
                **snapshot.binding(STAGE_Z_ROOT),
                "source_commit": stage_z["git_binding"]["head_commit"],
                "source_tree": stage_z["git_binding"]["head_tree"],
            },
        }
    )
    return {
        "schema": "PAPER_V2_EVIDENCE_HIERARCHY_V1",
        "status": "PAPER_V2_EVIDENCE_HIERARCHY_EXPORT_PASS",
        "rows": rows,
    }


def claim_by_id(
    claims: Mapping[str, Any], claim_id: str, source_id: str
) -> Mapping[str, Any]:
    matches = [row for row in claims["claims"] if row["claim_id"] == claim_id]
    require(len(matches) == 1, f"claim_id:{claim_id}:{len(matches)}")
    claim = matches[0]
    require(claim["direct_support"] is True, f"claim_support:{claim_id}")
    require(source_id in claim["source_artifacts"], f"claim_source:{claim_id}:{source_id}")
    return claim


def build_x0(
    snapshot: Snapshot,
    source_rows: Sequence[Mapping[str, str]],
    paper_map: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> dict[str, Any]:
    x0 = next(row for row in paper_map["sources"] if row["id"] == "X0")
    claim = claim_by_id(claims, "C201", "X0")
    claim_by_id(claims, "C202", "X0")
    match = re.search(
        r"Of the ([0-9,]+) complete three-dose patterns, all are monotone and fall in (.+?); no non-monotone pattern is observed",
        claim["exact_wording"],
    )
    require(match is not None, "x0_monotone_parse")
    complete_count = int(match.group(1).replace(",", ""))
    patterns = re.findall(r"`([01]{3})`", match.group(2))
    require(patterns, "x0_patterns")
    t10 = next(row for row in source_rows if row["dose"] == "T10")
    require(int(t10["consumable_rows"]) == complete_count, "x0_count_binding")
    return {
        "schema": "PAPER_V2_X0_MECHANISM_V1",
        "status": x0["status"],
        "claim_ids": ["C201", "C202"],
        "doses": [
            {
                "dose": row["dose"],
                "source_defined_consumable_rows": int(row["consumable_rows"]),
                "raw_positive_rate": format(Decimal(row["raw_positive_rate"]), ".5f"),
            }
            for row in source_rows
        ],
        "complete_three_dose_patterns": {
            "count": complete_count,
            "all_monotone": True,
            "observed_patterns": patterns,
            "per_pattern_counts": None,
            "per_pattern_counts_note": "not present in sealed Paper V1 data",
            "nonmonotone_count": 0,
        },
        "mechanism_consistent_telemetry": x0["claim_boundary"],
        "uncertainty_policy": "no iid uncertainty exported",
        "authority_paths": [FIGURE2, PAPER_CLAIMS, *x0["source_paths"]],
        "authority_binding": {
            "figure_data": snapshot.binding(FIGURE2),
            "claim_ledger": snapshot.binding(PAPER_CLAIMS),
            "source": source_binding(x0),
        },
    }


def build_cascade(
    snapshot: Snapshot,
    source_rows: Sequence[Mapping[str, str]],
    paper_map: Mapping[str, Any],
    supplement: Mapping[str, Any],
) -> dict[str, Any]:
    sources = {row["id"]: row for row in paper_map["sources"]}
    source_ids = {"VI-B2": "VI_B2", "VII": "VII", "VIII": "VIII"}
    unit_keys = {"VI-B2": "vi_b2", "VII": "vii", "VIII": "viii"}
    rows = []
    for row in source_rows:
        source = sources[source_ids[row["stage"]]]
        rows.append(
            {
                **row,
                "primary_unit": supplement["primary_units"][unit_keys[row["stage"]]],
                "denominator_censoring": source["population_or_denominator"],
                "missing_value_policy": "retain censoring/non-identifiable; no imputation",
                "authority_paths": source["source_paths"],
                "authority_binding": source_binding(source),
            }
        )
    return {
        "schema": "PAPER_V2_TIMING_GENERALIZATION_CASCADE_V1",
        "status": "PAPER_V2_TIMING_CASCADE_EXPORT_PASS",
        "rows": rows,
        "source_data": snapshot.binding(FIGURE3),
    }


def build_ix(
    snapshot: Snapshot,
    source_rows: Sequence[Mapping[str, str]],
    paper_map: Mapping[str, Any],
    supplement: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> dict[str, Any]:
    ix = next(row for row in paper_map["sources"] if row["id"] == "IX")
    gate_claim = claim_by_id(claims, "C206", "IX")
    require("remain unsatisfied" in gate_claim["exact_wording"], "ix_gate_status")
    return {
        "schema": "PAPER_V2_STAGE_IX_FACTORIZATION_GAP_V1",
        "status": ix["status"],
        "claim_id": gate_claim["claim_id"],
        "primary_unit": supplement["primary_units"]["ix"],
        "denominator_censoring": ix["population_or_denominator"],
        "environment_exposure": "NO_ENVIRONMENT",
        "rows": [
            {
                "score": row["score"],
                "model_side_auroc": format(Decimal(row["model_side_AUROC"]), ".6f"),
                "factorized_parent_macro_auc": format(
                    Decimal(row["factorized_parent_macro_AUC"]), ".6f"
                ),
                "top_k_loso_gate_status": "UNSATISFIED",
                "top_k_loso_numeric_values": None,
                "top_k_loso_note": "numeric values not present in sealed Paper V1 figure data",
            }
            for row in source_rows
        ],
        "authority_paths": [FIGURE4, *ix["source_paths"]],
        "authority_binding": {
            "figure_data": snapshot.binding(FIGURE4),
            "source": source_binding(ix),
        },
    }


def build_execution(
    snapshot: Snapshot,
    source_rows: Sequence[Mapping[str, str]],
    paper_map: Mapping[str, Any],
    f1_summary: Mapping[str, Any],
    f1_root: Mapping[str, Any],
) -> dict[str, Any]:
    e3 = next(row for row in paper_map["sources"] if row["id"] == "E3_E4")
    parent_rows = [
        row
        for row in source_rows
        if not row["category"].startswith(("libero_", "candidate_"))
    ]
    suite_rows = [row for row in source_rows if row["category"].startswith("libero_")]
    candidate_rows = [
        row for row in source_rows if row["category"].startswith("candidate_")
    ]
    parent_denominators = {int(row["parent_denominator"]) for row in parent_rows}
    candidate_denominators = {int(row["parent_denominator"]) for row in candidate_rows}
    require(len(parent_denominators) == 1, "e3_denominator")
    require(len(candidate_denominators) == 1, "e4_denominator")
    parent_denominator = next(iter(parent_denominators))
    candidate_denominator = next(iter(candidate_denominators))
    parent_counts = {row["category"]: int(row["parent_count"]) for row in parent_rows}
    candidate_counts = {
        row["category"].removeprefix("candidate_"): int(row["parent_count"])
        for row in candidate_rows
    }
    require(sum(parent_counts.values()) == parent_denominator, "e3_count_sum")
    require(sum(candidate_counts.values()) == candidate_denominator, "e4_count_sum")
    require(candidate_counts == e3["candidate_diagnostic_counts"], "e4_map_binding")
    suite_summary = {
        row["category"].removesuffix("_strict_valid"): {
            "strict_valid_parents": int(row["parent_count"]),
            "parent_denominator": int(row["parent_denominator"]),
        }
        for row in suite_rows
    }
    require(
        sum(row["strict_valid_parents"] for row in suite_summary.values())
        == parent_counts["STRICT_REALIZABLE"],
        "e3_suite_binding",
    )
    return {
        "schema": "PAPER_V2_EXECUTION_LAYER_EVIDENCE_V1",
        "status": "PAPER_V2_EXECUTION_LAYER_EXPORT_PASS",
        "no_pooled_funnel_rate": True,
        "population_policy": "E3, E4 diagnostics, F1-B DEV, and F1-C4 remain separate",
        "populations": {
            "e3_strict_single_state_realizability": {
                "primary_unit": "engineering parent",
                "parent_denominator": parent_denominator,
                "parent_categories": parent_counts,
                "suite_summary": suite_summary,
                "status": e3["status"].split(" / ", 1)[0],
                "authority_paths": e3["source_paths"],
                "authority_binding": source_binding(e3),
            },
            "e4_candidate_diagnostics": {
                "diagnostic_only": True,
                "iid": False,
                "candidate_denominator": candidate_denominator,
                "candidate_counts": candidate_counts,
                "status": e3["status"].split(" / ", 1)[1],
                "authority_paths": e3["source_paths"],
                "authority_binding": source_binding(e3),
            },
            "f1b_dev_method_development": {
                **f1_summary["f1b_dev"],
                "population_id": "F1B_DEV_V3",
                "pooled_with_other_populations": False,
                "authority_path": F1T_SUMMARY,
                "authority_sha256": f1_root["artifact_hashes"][F1T_SUMMARY],
            },
            "f1c4_fresh_executable_qualification": {
                **f1_summary["f1c4_canary"],
                "population_id": "F1C4_FRESH_CANARY_V1",
                "pooled_with_other_populations": False,
                "authority_path": F1T_SUMMARY,
                "authority_sha256": f1_root["artifact_hashes"][F1T_SUMMARY],
            },
        },
        "source_data": {
            "e3_e4": snapshot.binding(FIGURE5),
            "f1t": snapshot.binding(F1T_SUMMARY),
        },
    }


def build_bundle(source_ref: str) -> dict[str, bytes]:
    snapshot = Snapshot(source_ref)
    snapshot.bytes(GENERATOR)
    values = {
        path: snapshot.json(path)
        for path in AUTHORITY_INPUTS
        if path.endswith(".json")
    }
    declared_checks = validate(snapshot, values)
    paper_map = values[PAPER_MAP]
    claims = values[PAPER_CLAIMS]
    supplement = values[PAPER_SUPPLEMENT]
    f1_map = values[F1T_MAP]
    f1_summary = values[F1T_SUMMARY]
    f1_claims = values[F1T_CLAIMS]
    f1_root = values[F1T_ROOT]
    stage_z = values[STAGE_Z_ROOT]

    payloads = {
        "evidence_hierarchy.json": build_hierarchy(
            snapshot,
            paper_map,
            claims,
            supplement,
            f1_map,
            f1_summary,
            f1_claims,
            stage_z,
        ),
        "x0_mechanism.json": build_x0(
            snapshot, snapshot.csv(FIGURE2), paper_map, claims
        ),
        "timing_generalization_cascade.json": build_cascade(
            snapshot, snapshot.csv(FIGURE3), paper_map, supplement
        ),
        "stage_ix_factorization_gap.json": build_ix(
            snapshot, snapshot.csv(FIGURE4), paper_map, supplement, claims
        ),
        "execution_layer_evidence.json": build_execution(
            snapshot, snapshot.csv(FIGURE5), paper_map, f1_summary, f1_root
        ),
    }
    require(tuple(payloads) == OUTPUT_NAMES, "output_order")
    generated = {name: json_bytes(value) for name, value in payloads.items()}

    authority_inputs = []
    for path in sorted(AUTHORITY_INPUTS):
        authority_inputs.append(
            {
                **snapshot.binding(path),
                "declared_bindings": declared_checks.get(path, []),
            }
        )
    manifest = {
        "schema": "PAPER_V2_EXPORT_MANIFEST_V1",
        "status": "PAPER_V2_DETERMINISTIC_EXPORT_PASS",
        "source_repository": {
            "repository": REPOSITORY,
            "head": snapshot.commit,
            "tree": snapshot.tree,
            "commit_timestamp": snapshot.timestamp,
        },
        "authority_inputs": authority_inputs,
        "generated_files": [
            {
                "path": f"exports/paper_v2/{name}",
                "sha256": sha256(value),
                "bytes": len(value),
            }
            for name, value in generated.items()
        ],
        "generator": snapshot.binding(GENERATOR),
        "generated_at": snapshot.timestamp,
        "timestamp_policy": "source commit timestamp; no wall-clock input",
        "protected_boundary": {
            "new_openvla_inference": 0,
            "new_simulator_or_env_step": 0,
            "new_pgd_or_backward": 0,
            "new_physical_intervention": 0,
            "new_vphys_read": 0,
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
        },
        "statement": "Eval160/protected were not read; committed sealed static authorities only.",
        "paper_repository_role": "presentation consumer; not scientific source-of-truth",
        "no_manual_scientific_number_transcription": True,
    }
    generated[MANIFEST] = json_bytes(manifest)
    return generated


def write_bundle(source_ref: str) -> int:
    generated = build_bundle(source_ref)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name not in generated
    )
    require(not unexpected, f"unexpected_outputs:{unexpected}")
    for name, value in generated.items():
        (OUTPUT_DIR / name).write_bytes(value)
    manifest = json.loads(generated[MANIFEST])
    print(
        "PAPER_V2_EXPORT_WRITE_PASS "
        f"source_head={manifest['source_repository']['head']} files={len(generated)}"
    )
    return 0


def check_bundle(source_ref: str | None) -> int:
    manifest_path = OUTPUT_DIR / MANIFEST
    require(manifest_path.is_file(), "manifest_missing")
    committed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_ref = source_ref or committed_manifest["source_repository"]["head"]
    generated = build_bundle(effective_ref)
    actual_names = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    require(actual_names == set(generated), f"output_set:{sorted(actual_names)}")
    mismatches = [
        name for name, expected in generated.items() if (OUTPUT_DIR / name).read_bytes() != expected
    ]
    require(not mismatches, f"stale_or_nondeterministic:{mismatches}")
    manifest = json.loads(generated[MANIFEST])
    print(
        "PAPER_V2_EXPORT_CHECK_PASS "
        f"source_head={manifest['source_repository']['head']} files={len(generated)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-ref",
        help="committed source ref; --check defaults to the manifest binding",
    )
    args = parser.parse_args()
    if args.write:
        return write_bundle(args.source_ref or "HEAD")
    return check_bundle(args.source_ref)


if __name__ == "__main__":
    raise SystemExit(main())
