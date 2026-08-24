#!/usr/bin/env python3
"""Build the paper figure/table package from sealed, already-published evidence.

This is intentionally dependency-free: the figures are simple deterministic
SVGs, and every plotted value is also emitted as CSV.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
DATA = PAPER / "data"
TABLES = PAPER / "tables"
E4_DIR = ROOT / "reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821"
X0 = ROOT / "docs/handoffs/STAGE_X_X0_RESULT_20260817.md"
VI_B2 = ROOT / "docs/handoffs/STAGE_VI_B2_FRESH_M4_AND_NEGATIVE_CAUSAL_HANDOFF_20260816.md"
VII = ROOT / "docs/handoffs/STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md"
VIII_COMMIT = "34b8d435264737734f7d4e4ecc9a3343e57d7c1"
VIII_PATH = "docs/handoffs/STAGE_VIII_R1_RELATIVE_SELECTOR_NEGATIVE_HANDOFF_20260817.md"
IX = ROOT / "docs/handoffs/STAGE_IX_F0_RESULT_20260817.md"

EXPECTED = {
    X0: "7626e020ec524ce124be9b93685a3861b85176ad4bfd749bfc495f89482d3c55",
    VI_B2: "2c550b1ec8f906212bc1f92223b801bcb0efe07381caf94d1b116e4f5e66d006",
    VII: "9ee6782530f730e7bf5eaee90ced94134ec6a569d5c9713b7b19f85662a1c33f",
    IX: "2c445ea1ae5434661d327a348995c412083b191b382cbd78d485a13064268103",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_start(title: str, width: int = 1000, height: int = 560) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#202124} .title{font-size:24px;font-weight:700} .subtitle{font-size:15px;fill:#5f6368} .axis{stroke:#5f6368;stroke-width:1} .grid{stroke:#dadce0;stroke-width:1} .label{font-size:14px} .small{font-size:12px} .box{stroke:#202124;stroke-width:2;rx:12} .arrow{stroke:#5f6368;stroke-width:2;fill:none;marker-end:url(#arrow)} .dash{stroke-dasharray:7 6}</style>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#5f6368"/></marker></defs>',
        f'<text x="40" y="42" class="title">{esc(title)}</text>',
    ]


def svg_end(lines: list[str], note: str, height: int = 560) -> str:
    lines.append(f'<text x="40" y="{height - 24}" class="subtitle">{esc(note)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def bar_chart(lines: list[str], values: list[tuple[str, float]], x: int, y: int, width: int, height: int, maximum: float, colors: list[str] | None = None, suffix: str = "") -> None:
    colors = colors or ["#4e79a7"] * len(values)
    for tick in range(6):
        value = maximum * tick / 5
        yy = y + height - height * value / maximum
        lines.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" class="grid"/>')
        lines.append(f'<text x="{x - 10}" y="{yy + 5:.1f}" text-anchor="end" class="small">{value:.1f}</text>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>')
    lines.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/>')
    slot = width / max(len(values), 1)
    bar_width = min(80, slot * 0.62)
    for index, (label, value) in enumerate(values):
        bx = x + slot * index + (slot - bar_width) / 2
        bh = height * value / maximum
        by = y + height - bh
        lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" height="{bh:.1f}" fill="{colors[index]}"/>')
        lines.append(f'<text x="{bx + bar_width / 2:.1f}" y="{by - 8:.1f}" text-anchor="middle" class="label">{value:.3f}{esc(suffix)}</text>')
        lines.append(f'<text x="{bx + bar_width / 2:.1f}" y="{y + height + 24}" text-anchor="middle" class="label">{esc(label)}</text>')


def figure_1() -> str:
    lines = svg_start("Figure 1. Factorization of timing, physics, and visual exploitability")
    boxes = [(80, 180, 230, 150, "C_t", "clean criticality / opportunity", "timing selector"), (385, 180, 230, 150, "V_t(d)", "physical vulnerability", "duration-d OPEN counterfactual"), (690, 180, 230, 150, "E_t", "visual exploitability", "strict selective realization")]
    for x, y, w, h, symbol, label, detail in boxes:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="box" fill="#eef4fb"/>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 54}" text-anchor="middle" style="font-size:34px;font-weight:700">{symbol}</text>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 91}" text-anchor="middle" class="label">{esc(label)}</text>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 119}" text-anchor="middle" class="small">{esc(detail)}</text>')
    lines.append('<path d="M310 255 L385 255" class="arrow dash"/>')
    lines.append('<path d="M615 255 L690 255" class="arrow dash"/>')
    lines.append('<text x="347" y="236" text-anchor="middle" class="small">not assumed causal</text>')
    lines.append('<text x="652" y="236" text-anchor="middle" class="small">not assumed causal</text>')
    lines.append('<text x="500" y="410" text-anchor="middle" class="label">Paper result: partial alignment, not one latent vulnerability score</text>')
    lines.append('<text x="500" y="445" text-anchor="middle" class="small">X0 informs V; VI–IX constrain C and timing utility; E3/E4 test E without Student timing.</text>')
    return svg_end(lines, "Conceptual separation; dashed arrows are not demonstrated causal mediation.")


def figure_2() -> str:
    lines = svg_start("Figure 2. X0 dose response and OPEN duty-cycle mechanism")
    bar_chart(lines, [("T3", 0.39438), ("T5", 0.67758), ("T10", 0.87300)], 90, 100, 350, 300, 1.0, ["#9ecae1", "#6baed6", "#2171b5"])
    lines.append('<text x="265" y="455" text-anchor="middle" class="label">raw consumable V_phys positive rate</text>')
    chain = [(505, "command delivery", "exact for eligible rows"), (625, "aperture excess", "increases with dose"), (745, "contact loss", "incidence increases"), (865, "displacement", "increases")]
    for x, label, detail in chain:
        lines.append(f'<rect x="{x}" y="205" width="105" height="105" class="box" fill="#fff4e6"/>')
        lines.append(f'<text x="{x + 52}" y="245" text-anchor="middle" class="small">{esc(label)}</text>')
        lines.append(f'<text x="{x + 52}" y="275" text-anchor="middle" class="small">{esc(detail)}</text>')
    for x in [610, 730, 850]:
        lines.append(f'<path d="M{x} 257 L{x + 15} 257" class="arrow"/>')
    lines.append('<text x="710" y="365" text-anchor="middle" class="label">mechanism-consistent descriptive chain</text>')
    lines.append('<text x="710" y="395" text-anchor="middle" class="small">1,126 complete three-dose patterns were all monotone: 000/001/011/111.</text>')
    return svg_end(lines, "X0 is descriptive/mechanistic and bounded-counterfactual evidence; it is not formal mediation.")


def figure_3() -> str:
    lines = svg_start("Figure 3. Timing-selector negative cascade")
    stages = [
        ("VI-B2", "held-out Student", "AUROC 0.624643", "ECE-10 0.460636", "causal/actionable gate not established", "#fdd0a2"),
        ("VII", "three frozen candidates", "S7-A/B/C all fail", "none promoted", "cross-suite gates fail", "#fdae6b"),
        ("VIII", "relative selector R1", "AUC 0.577615/0.658572", "both gates FAIL", "generalizable deployment selector absent", "#e6550d"),
    ]
    for idx, (stage, label, metric, metric2, outcome, color) in enumerate(stages):
        x = 75 + idx * 300
        lines.append(f'<rect x="{x}" y="155" width="245" height="250" class="box" fill="{color}" fill-opacity="0.25"/>')
        lines.append(f'<text x="{x + 122}" y="198" text-anchor="middle" style="font-size:25px;font-weight:700">{stage}</text>')
        lines.append(f'<text x="{x + 122}" y="238" text-anchor="middle" class="label">{esc(label)}</text>')
        lines.append(f'<text x="{x + 122}" y="285" text-anchor="middle" class="label">{esc(metric)}</text>')
        lines.append(f'<text x="{x + 122}" y="315" text-anchor="middle" class="label">{esc(metric2)}</text>')
        lines.append(f'<text x="{x + 122}" y="365" text-anchor="middle" class="small">{esc(outcome)}</text>')
        if idx < 2:
            lines.append(f'<path d="M{x + 245} 280 L{x + 295} 280" class="arrow"/>')
    lines.append('<text x="500" y="465" text-anchor="middle" class="label">overall discrimination ≠ cross-suite generalization ≠ within-parent actionable timing</text>')
    return svg_end(lines, "Negative scientific evidence is stage-specific; it is not a claim that every feature is uninformative.")


def figure_4() -> str:
    lines = svg_start("Figure 4. Stage IX model-to-physics factorization gap")
    labels = ["E0", "E1", "E3"]
    model = [0.870743, 0.900510, 0.897157]
    factor = [0.483698, 0.521112, 0.523390]
    x, y, width, height = 100, 100, 700, 300
    for tick in range(6):
        value = tick / 5
        yy = y + height - height * value
        lines.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" class="grid"/>')
        lines.append(f'<text x="{x - 10}" y="{yy + 5:.1f}" text-anchor="end" class="small">{value:.1f}</text>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>')
    lines.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/>')
    slot = width / 3
    for idx, label in enumerate(labels):
        center = x + slot * idx + slot / 2
        for offset, value, color, name in [(-28, model[idx], "#4e79a7", "model-side AUROC"), (28, factor[idx], "#e15759", "factorized parent-macro AUC")]:
            bx = center + offset - 18
            bh = height * value
            by = y + height - bh
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="36" height="{bh:.1f}" fill="{color}"/>')
            lines.append(f'<text x="{bx + 18:.1f}" y="{by - 8:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
        lines.append(f'<text x="{center}" y="{y + height + 28}" text-anchor="middle" class="label">{label}</text>')
    lines.append('<rect x="860" y="145" width="18" height="18" fill="#4e79a7"/><text x="888" y="159" class="small">model-side AUROC</text>')
    lines.append('<rect x="860" y="180" width="18" height="18" fill="#e15759"/><text x="888" y="194" class="small">parent-macro AUC</text>')
    lines.append('<text x="500" y="470" text-anchor="middle" class="label">high targetability scores did not provide reliable factorized timing utility</text>')
    return svg_end(lines, "Stage IX is no-environment model-side evidence; it is not physical attack efficacy.")


def figure_5(e4: dict) -> str:
    summary = e4["e3_structural_summary"]
    categories = [("targetability limited", 9, "#9ecae1"), ("joint limited", 1, "#fdae6b"), ("strict realizable", 2, "#31a354")]
    lines = svg_start("Figure 5. E3/E4 strict selective realizability decomposition")
    lines.append('<text x="270" y="88" text-anchor="middle" class="label">primary: parent-level categories (12 engineering parents)</text>')
    bar_chart(lines, [(label, value) for label, value, _ in categories], 80, 120, 380, 260, 12, [color for _, _, color in categories])
    lines.append('<text x="270" y="420" text-anchor="middle" class="label">parent count</text>')
    suites = [("L10", 1), ("Goal", 0), ("Object", 0), ("Spatial", 1)]
    lines.append('<text x="700" y="88" text-anchor="middle" class="label">strict-valid parents by suite (denominator 3 each)</text>')
    bar_chart(lines, suites, 520, 120, 360, 260, 3, ["#31a354", "#d9d9d9", "#d9d9d9", "#31a354"])
    lines.append('<text x="700" y="420" text-anchor="middle" class="label">strict-valid parent count</text>')
    lines.append('<text x="500" y="470" text-anchor="middle" class="small">Candidate slots 4/29/1/38 over 72 are diagnostic secondary counts, not iid evidence.</text>')
    assert summary["parents_with_strict_valid_candidate"] == 2
    return svg_end(lines, "E3/E4 establish bounded model-side realizability only; no physical efficacy or impossibility claim.")


def build() -> dict[str, object]:
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, (path, sha256(path), expected)
    e4 = json.loads((E4_DIR / "STAGE_X1R2_E4_FINAL_CLAIM_LEDGER_V1.json").read_text(encoding="utf-8"))
    decomposition = json.loads((E4_DIR / "STAGE_X1R2_E4_E3_CANDIDATE_FAILURE_DECOMPOSITION_V1.json").read_text(encoding="utf-8"))
    assert e4["status"] == "STAGE_X_X1R2_E4_PAPER_LOCK_READY"
    assert e4["attack_efficacy"] is False
    assert decomposition["status"] == "STAGE_X_X1R2_E4_DECOMPOSITION_PASS"

    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    write_csv(DATA / "PAPER_V1_FIGURE2_X0_DOSE_RESPONSE.csv", [
        {"dose": "T3", "raw_positive_rate": 0.39438, "consumable_rows": 1245},
        {"dose": "T5", "raw_positive_rate": 0.67758, "consumable_rows": 1191},
        {"dose": "T10", "raw_positive_rate": 0.87300, "consumable_rows": 1126},
    ])
    write_csv(DATA / "PAPER_V1_FIGURE3_TIMING_NEGATIVE_CASCADE.csv", [
        {"stage": "VI-B2", "primary_metric": "overall_AUROC", "value": 0.6246432939, "gate_or_status": "held-out causal/actionable gate not established"},
        {"stage": "VI-B2", "primary_metric": "ECE10", "value": 0.4606357016, "gate_or_status": "above frozen 0.25 limit"},
        {"stage": "VII", "primary_metric": "promoted_candidates", "value": 0, "gate_or_status": "S7-A/S7-B/S7-C all failed at least one gate"},
        {"stage": "VIII", "primary_metric": "R1-A_parent_macro_AUC", "value": 0.577615, "gate_or_status": "FAIL"},
        {"stage": "VIII", "primary_metric": "R1-B_parent_macro_AUC", "value": 0.658572, "gate_or_status": "FAIL"},
    ])
    write_csv(DATA / "PAPER_V1_FIGURE4_FACTORIZATION_GAP.csv", [
        {"score": "E0", "model_side_AUROC": 0.870743, "factorized_parent_macro_AUC": 0.483698},
        {"score": "E1", "model_side_AUROC": 0.900510, "factorized_parent_macro_AUC": 0.521112},
        {"score": "E3", "model_side_AUROC": 0.897157, "factorized_parent_macro_AUC": 0.523390},
    ])
    write_csv(DATA / "PAPER_V1_FIGURE5_E3_E4_PARENT_REALIZABILITY.csv", [
        {"category": "TARGETABILITY_LIMITED", "parent_count": 9, "parent_denominator": 12},
        {"category": "JOINT_LIMITED", "parent_count": 1, "parent_denominator": 12},
        {"category": "STRICT_REALIZABLE", "parent_count": 2, "parent_denominator": 12},
        {"category": "libero_10_strict_valid", "parent_count": 1, "parent_denominator": 3},
        {"category": "libero_goal_strict_valid", "parent_count": 0, "parent_denominator": 3},
        {"category": "libero_object_strict_valid", "parent_count": 0, "parent_denominator": 3},
        {"category": "libero_spatial_strict_valid", "parent_count": 1, "parent_denominator": 3},
        {"category": "candidate_ARM_EXACT_AND_NATIVE_OPEN", "parent_count": 4, "parent_denominator": 72},
        {"category": "candidate_ARM_EXACT_BUT_NOT_NATIVE_OPEN", "parent_count": 29, "parent_denominator": 72},
        {"category": "candidate_NATIVE_OPEN_BUT_ARM_DRIFT", "parent_count": 1, "parent_denominator": 72},
        {"category": "candidate_NEITHER_OPEN_NOR_ARM_EXACT", "parent_count": 38, "parent_denominator": 72},
    ])

    hierarchy = [
        {"stage": "X0", "role": "primary_bounded", "unit": "parent/probe as sealed", "claim": "dose- and phase-dependent physical OPEN mechanism", "not_claim": "formal mediation or attack efficacy"},
        {"stage": "VI-B2/VII/VIII", "role": "negative_scientific", "unit": "parent-grouped source unit", "claim": "timing-selector generalization not established", "not_claim": "all features or all attacks fail"},
        {"stage": "IX", "role": "primary_negative_scientific", "unit": "no-environment row with parent-macro aggregation", "claim": "model-to-physics timing factorization gap", "not_claim": "physical efficacy"},
        {"stage": "E2", "role": "diagnostic_only", "unit": "three bounded Goal successor identities", "claim": "no legal scheduler emit", "not_claim": "strict visual-method negative"},
        {"stage": "E3/E4", "role": "primary_bounded_model_side_only", "unit": "12 engineering parents", "claim": "sparse suite/state-dependent strict realizability", "not_claim": "physical efficacy or impossibility"},
    ]
    write_csv(TABLES / "PAPER_V1_EVIDENCE_HIERARCHY.csv", hierarchy)
    boundary = [
        {"claim_id": "X0_MECHANISM", "allowed": "descriptive mechanism and bounded OPEN counterfactual", "source": "X0 sealed root", "forbidden": "formal mediation; universal detector; physical efficacy"},
        {"claim_id": "TIMING_NEGATIVE_CASCADE", "allowed": "frozen timing generalization gates did not pass", "source": "VI-B2/VII/VIII", "forbidden": "every feature uninformative; attack failure"},
        {"claim_id": "IX_FACTORIZATION_GAP", "allowed": "model-side targetability separated from factorized timing utility", "source": "IX F0 sealed root", "forbidden": "physical attack efficacy"},
        {"claim_id": "E2_TIMING_HOLD", "allowed": "bounded Goal no-emit diagnostic", "source": "E2 sealed root", "forbidden": "strict visual-method negative; E3 explanation"},
        {"claim_id": "E3_E4_REALIZABILITY", "allowed": "2/12 engineering parents strict-valid, suite/state dependent", "source": "E3/E4 sealed roots", "forbidden": "physical efficacy; Goal/Object impossibility; iid candidate inference"},
        {"claim_id": "PROTECTED_BOUNDARY", "allowed": "Eval160/protected remain unread", "source": "all current handoffs", "forbidden": "any protected validation claim"},
    ]
    write_csv(TABLES / "PAPER_V1_CLAIM_BOUNDARY.csv", boundary)
    write_csv(TABLES / "PAPER_V1_E3_E4_PARENT_REALIZABILITY.csv", [
        {"suite": "libero_10", "parents": 3, "strict_valid_parents": 1, "strict_valid_fraction": "1/3", "failure_categories": "STRICT_REALIZABLE=1; TARGETABILITY_LIMITED=2"},
        {"suite": "libero_goal", "parents": 3, "strict_valid_parents": 0, "strict_valid_fraction": "0/3", "failure_categories": "JOINT_LIMITED=1; TARGETABILITY_LIMITED=2"},
        {"suite": "libero_object", "parents": 3, "strict_valid_parents": 0, "strict_valid_fraction": "0/3", "failure_categories": "TARGETABILITY_LIMITED=3"},
        {"suite": "libero_spatial", "parents": 3, "strict_valid_parents": 1, "strict_valid_fraction": "1/3", "failure_categories": "STRICT_REALIZABLE=1; TARGETABILITY_LIMITED=2"},
    ])

    write_text(FIGURES / "figure1_factorization_conceptual.svg", figure_1())
    write_text(FIGURES / "figure2_x0_dose_response_mechanism.svg", figure_2())
    write_text(FIGURES / "figure3_timing_negative_cascade.svg", figure_3())
    write_text(FIGURES / "figure4_stage_ix_factorization_gap.svg", figure_4())
    write_text(FIGURES / "figure5_e3_e4_realizability_decomposition.svg", figure_5(e4))

    captions = """# Paper V1 figure/table package

Status: `PAPER_V1_FIGURE_TABLE_PACKAGE_PASS`

All values below are generated from already-sealed evidence. SVG is used so
the package remains dependency-free and reviewable in a text diff.

## Main figures

1. **Figure 1 — Factorization.** `C_t`, `V_t(d)`, and `E_t` are distinct
   quantities. Dashed arrows are conceptual only; the paper does not claim a
   demonstrated causal chain.
2. **Figure 2 — X0 mechanism.** The T3/T5/T10 positive rates and monotone
   three-dose patterns support a dose- and phase-dependent OPEN duty-cycle
   mechanism. The chain is descriptive/mechanistic, not formal mediation.
3. **Figure 3 — Timing negatives.** VI-B2, VII, and VIII fail their frozen
   held-out/generalization gates at their own source-declared units.
4. **Figure 4 — Factorization gap.** Stage IX model-side AUROC is high while
   factorized parent-macro AUC is near chance; this is no-environment
   model-side evidence, not physical attack efficacy.
5. **Figure 5 — E3/E4 decomposition.** Parent-level aggregation is primary:
   9 targetability-limited, 1 joint-limited, and 2 strict-realizable parents.
   Candidate-slot counts are diagnostic and explicitly non-iid.

## Tables

- `tables/PAPER_V1_EVIDENCE_HIERARCHY.csv` — stage roles and denominators.
- `tables/PAPER_V1_CLAIM_BOUNDARY.csv` — allowed and forbidden wording.
- `tables/PAPER_V1_E3_E4_PARENT_REALIZABILITY.csv` — suite-level primary E3/E4
  result.

## Reproduction

Run `python scripts/paper/build_paper_v1_figures_tables.py`. The script checks
the hashes of its source handoffs and the E4 status before writing any output.
No model, simulator, GPU, PGD, physical, Eval160, or protected operation is
performed.
"""
    write_text(PAPER / "PAPER_V1_FIGURE_TABLE_PACKAGE_V1.md", captions)

    files = [
        *(DATA.glob("*.csv")),
        *(TABLES.glob("*.csv")),
        *(FIGURES.glob("*.svg")),
        PAPER / "PAPER_V1_FIGURE_TABLE_PACKAGE_V1.md",
    ]
    manifest = {
        "schema": "PAPER_V1_FIGURE_TABLE_PACKAGE_MANIFEST_V1",
        "status": "PAPER_V1_FIGURE_TABLE_PACKAGE_PASS",
        "source_policy": "sealed_artifact_static_only",
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in [X0, VI_B2, VII, IX]},
        "files": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(files)],
        "protected_boundary": {"gpu": 0, "openvla_inference": 0, "env_step": 0, "pgd": 0, "physical_intervention": 0, "eval160": "UNREAD", "protected": "UNREAD"},
    }
    manifest_path = PAPER / "PAPER_V1_FIGURE_TABLE_MANIFEST_V1.json"
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    manifest_sha = sha256(manifest_path)
    write_text(PAPER / "PAPER_V1_FIGURE_TABLE_MANIFEST_V1.sha256", manifest_sha)
    root_seal = {
        "schema": "PAPER_V1_FIGURE_TABLE_ROOT_SEAL_V1",
        "status": "SEALED_PAPER_V1_FIGURE_TABLE_PACKAGE",
        "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": manifest_sha,
        "file_count": len(files),
        "protected_boundary": manifest["protected_boundary"],
    }
    root_seal_path = PAPER / "PAPER_V1_FIGURE_TABLE_ROOT_SEAL_V1.json"
    write_text(root_seal_path, json.dumps(root_seal, indent=2, sort_keys=True))
    write_text(PAPER / "PAPER_V1_FIGURE_TABLE_ROOT_SEAL_V1.sha256", sha256(root_seal_path))
    result = dict(manifest)
    result["root_seal_sha256"] = sha256(root_seal_path)
    return result


if __name__ == "__main__":
    result = build()
    print(f"PAPER_V1_FIGURE_TABLE_PACKAGE_PASS files={len(result['files'])} root_seal_sha256={result['root_seal_sha256']}")
