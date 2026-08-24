# Paper V1 figure/table package

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
