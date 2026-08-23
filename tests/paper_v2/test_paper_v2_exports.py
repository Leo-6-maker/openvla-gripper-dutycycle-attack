from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "exports/paper_v2"
MANIFEST_NAME = "PAPER_V2_EXPORT_MANIFEST_V1.json"


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8").strip()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_export_bundle_rebuilds_byte_for_byte() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/paper_v2/export_paper_v2_evidence.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PAPER_V2_EXPORT_CHECK_PASS" in process.stdout


def test_manifest_binds_git_inputs_generator_and_outputs() -> None:
    manifest = load_json(f"exports/paper_v2/{MANIFEST_NAME}")
    source = manifest["source_repository"]
    head = source["head"]
    assert git_text("rev-parse", f"{head}^{{commit}}") == head
    assert git_text("rev-parse", f"{head}^{{tree}}") == source["tree"]
    assert git_text("show", "-s", "--format=%cI", head) == source["commit_timestamp"]
    assert manifest["generated_at"] == source["commit_timestamp"]

    for row in manifest["authority_inputs"]:
        value = git_bytes("show", f"{head}:{row['path']}")
        assert digest(value) == row["sha256_git_blob_bytes"]
        assert git_text("rev-parse", f"{head}:{row['path']}") == row["git_blob"]

    listed = {row["path"] for row in manifest["generated_files"]}
    actual = {
        f"exports/paper_v2/{path.name}"
        for path in EXPORT_DIR.iterdir()
        if path.is_file()
        if path.name != MANIFEST_NAME
    }
    assert listed == actual
    assert {Path(path).suffix for path in listed} == {".csv", ".json", ".tex"}
    for row in manifest["generated_files"]:
        value = (ROOT / row["path"]).read_bytes()
        assert digest(value) == row["sha256"]
        assert len(value) == row["bytes"]

    generator = manifest["generator"]
    generator_bytes = git_bytes("show", f"{head}:{generator['path']}")
    assert digest(generator_bytes) == generator["sha256_git_blob_bytes"]
    assert git_text("rev-parse", f"{head}:{generator['path']}") == generator["git_blob"]

    boundary = manifest["protected_boundary"]
    assert boundary["eval160"] == "UNREAD"
    assert boundary["protected_evaluation"] == "UNREAD"
    assert all(
        value == 0
        for key, value in boundary.items()
        if key not in {"eval160", "protected_evaluation"}
    )


def test_hierarchy_uses_sealed_claim_ids_and_keeps_stage_z_pending() -> None:
    hierarchy = load_json("exports/paper_v2/evidence_hierarchy.json")
    rows = hierarchy["rows"]
    assert [row["stage"] for row in rows] == [
        "X0",
        "VI-B2",
        "VII",
        "VIII",
        "IX",
        "E3/E4",
        "F1-B",
        "F1-C4/F1T",
        "Stage Z pending",
    ]

    paper_claims = {
        row["claim_id"]
        for row in load_json("paper/PAPER_V1_CLAIM_LEDGER_V1.json")["claims"]
    }
    f1_claims = {
        row["id"]
        for row in load_json(
            "reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json"
        )["promotable_claims"]
    }
    for row in rows[:-1]:
        assert row["primary_unit"]
        assert row["denominator_censoring"]
        assert row["authority_paths"]
        assert row["authority_binding"]
        claim_ids = row["promotable_wording_key"]
        assert claim_ids
        assert all(
            claim_id in (paper_claims | f1_claims) for claim_id in claim_ids
        )

    stage_z = load_json("reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json")
    pending = rows[-1]
    assert pending["status"] == stage_z["status"]
    assert pending["promotable_wording_key"] is None
    assert pending["environment_exposure"]["scientific_rollout_started"] is False
    assert pending["environment_exposure"]["counters"] == stage_z["counters"]
    assert all(value == 0 for value in stage_z["counters"].values())


def test_quantitative_exports_match_sealed_tables_without_pooling() -> None:
    x0_source = load_csv("paper/data/PAPER_V1_FIGURE2_X0_DOSE_RESPONSE.csv")
    x0 = load_json("exports/paper_v2/x0_mechanism.json")
    assert [
        (row["dose"], row["source_defined_consumable_rows"], row["raw_positive_rate"])
        for row in x0["doses"]
    ] == [
        (
            row["dose"],
            int(row["consumable_rows"]),
            format(Decimal(row["raw_positive_rate"]), ".5f"),
        )
        for row in x0_source
    ]
    assert x0["complete_three_dose_patterns"]["count"] == int(
        next(row for row in x0_source if row["dose"] == "T10")["consumable_rows"]
    )
    assert x0["uncertainty_policy"] == "no iid uncertainty exported"

    cascade_source = load_csv(
        "paper/data/PAPER_V1_FIGURE3_TIMING_NEGATIVE_CASCADE.csv"
    )
    cascade = load_json("exports/paper_v2/timing_generalization_cascade.json")
    assert len(cascade["rows"]) == len(cascade_source)
    for emitted, sealed in zip(cascade["rows"], cascade_source, strict=True):
        assert {key: emitted[key] for key in sealed} == sealed
        assert emitted["missing_value_policy"].endswith("no imputation")

    ix_source = load_csv("paper/data/PAPER_V1_FIGURE4_FACTORIZATION_GAP.csv")
    ix = load_json("exports/paper_v2/stage_ix_factorization_gap.json")
    assert [
        (row["score"], row["model_side_auroc"], row["factorized_parent_macro_auc"])
        for row in ix["rows"]
    ] == [
        (
            row["score"],
            format(Decimal(row["model_side_AUROC"]), ".6f"),
            format(Decimal(row["factorized_parent_macro_AUC"]), ".6f"),
        )
        for row in ix_source
    ]
    assert all(row["top_k_loso_gate_status"] == "UNSATISFIED" for row in ix["rows"])
    assert all(row["top_k_loso_numeric_values"] is None for row in ix["rows"])

    plot_rows = load_csv("exports/paper_v2/plot_quantitative.csv")
    assert {row["section"] for row in plot_rows} == {
        "factorization_gap",
        "timing_generalization",
        "x0_mechanism",
    }
    assert all(
        row["denominator"] and row["primary_unit"] and row["claim_id"]
        for row in plot_rows
    )
    assert [
        (row["item"], row["value"], row["denominator"])
        for row in plot_rows
        if row["section"] == "x0_mechanism"
    ] == [
        (
            row["dose"],
            row["raw_positive_rate"],
            str(row["source_defined_consumable_rows"]),
        )
        for row in x0["doses"]
    ]
    assert [
        (row["stage"], row["metric"], row["value"])
        for row in plot_rows
        if row["section"] == "timing_generalization"
    ] == [
        (row["stage"], row["primary_metric"], row["value"])
        for row in cascade["rows"]
    ]
    assert [
        (row["item"], row["metric"], row["value"])
        for row in plot_rows
        if row["section"] == "factorization_gap"
    ] == [
        (row["score"], metric, row[metric])
        for row in ix["rows"]
        for metric in ("model_side_auroc", "factorized_parent_macro_auc")
    ]

    execution = load_json("exports/paper_v2/execution_layer_evidence.json")
    populations = execution["populations"]
    assert execution["no_pooled_funnel_rate"] is True
    assert set(populations) == {
        "e3_strict_single_state_realizability",
        "e4_candidate_diagnostics",
        "f1b_dev_method_development",
        "f1c4_fresh_executable_qualification",
    }
    figure5 = load_csv("paper/data/PAPER_V1_FIGURE5_E3_E4_PARENT_REALIZABILITY.csv")
    parent_rows = [
        row
        for row in figure5
        if not row["category"].startswith(("libero_", "candidate_"))
    ]
    candidate_rows = [row for row in figure5 if row["category"].startswith("candidate_")]
    assert populations["e3_strict_single_state_realizability"]["parent_categories"] == {
        row["category"]: int(row["parent_count"]) for row in parent_rows
    }
    assert populations["e4_candidate_diagnostics"]["candidate_counts"] == {
        row["category"].removeprefix("candidate_"): int(row["parent_count"])
        for row in candidate_rows
    }

    f1 = load_json("reports/STAGE_X_X1R2_F1T_DEV_C4_SUMMARY_V1.json")
    assert populations["f1b_dev_method_development"]["dev_parent_count"] == f1[
        "f1b_dev"
    ]["dev_parent_count"]
    assert populations["f1c4_fresh_executable_qualification"]["parent_denominator"] == f1[
        "f1c4_canary"
    ]["parent_denominator"]
    assert populations["f1b_dev_method_development"]["pooled_with_other_populations"] is False
    assert (
        populations["f1c4_fresh_executable_qualification"][
            "pooled_with_other_populations"
        ]
        is False
    )

    tex = (EXPORT_DIR / "core_numbers.tex").read_text(encoding="ascii")
    manifest = load_json(f"exports/paper_v2/{MANIFEST_NAME}")
    expected_macros = {
        "PaperVTwoSourceHead": manifest["source_repository"]["head"],
        "PaperVTwoXZeroTThreeRate": x0["doses"][0]["raw_positive_rate"],
        "PaperVTwoIXEZeroModelAUROC": ix["rows"][0]["model_side_auroc"],
        "PaperVTwoEThreeParentDenominator": populations[
            "e3_strict_single_state_realizability"
        ]["parent_denominator"],
        "PaperVTwoEFourCandidateDenominator": populations[
            "e4_candidate_diagnostics"
        ]["candidate_denominator"],
        "PaperVTwoFOneBDevParentDenominator": populations[
            "f1b_dev_method_development"
        ]["dev_parent_count"],
        "PaperVTwoFOneCParentDenominator": populations[
            "f1c4_fresh_executable_qualification"
        ]["parent_denominator"],
    }
    for name, value in expected_macros.items():
        assert f"\\newcommand{{\\{name}}}{{{value}}}" in tex
