# F1T terminal synthesis — 2026-08-22

Gate: `STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_AND_PAPER_V2_DELTA_V1`
Status: `F1T_TERMINAL_SYNTHESIS_SEALED_FOR_PI`

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

- `reports/STAGE_X_X1R2_F1T_TERMINAL_SYNTHESIS_V1.json`
- `reports/STAGE_X_X1R2_F1T_CLAIM_LEDGER_DELTA_V1.json`
- `reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json`
- `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json`
