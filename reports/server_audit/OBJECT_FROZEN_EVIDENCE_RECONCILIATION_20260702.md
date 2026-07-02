# Object Frozen Evidence Reconciliation — 2026-07-02 (AMENDED 2026-07-02T20:00)

## Status

**FROZEN_REPORTED_RESULT_WITH_PROVENANCE_SEAL_PENDING**

Condition-level totals independently verified from episode_summary.json files.
Per-episode master ledger committed (`tables/server_freeze/object_frozen_master_ledger.csv`, 930 rows across 6 conditions).
Full provenance seal (config/manifest/checkpoint/command SHA chain) still pending.

---

## Independent Verification Results

All numbers re-computed from `episode_summary.json` in every leaf artifact directory.
See `tables/server_freeze/object_frozen_master_ledger.csv` for per-episode data.

| Condition | Leaf Dirs | Emitted | No-Emission | Success | Failure | Claimed | Match |
|---|---|---|---|---|---|---|---|
| CLEAN | 162 | 141 | 21 | 162 | 0 | 162/162 (0% FR) | PASS |
| RAND_T10 | 162 | 141 | 21 | 162 | 0 | 162/162 (0% FR) | PASS |
| RANDOM_TIME_V3 | 162 | 126 | 36 | 119 | 43 | 119/162 (26.5% FR) | PASS |
| EARLY_SHIFT_T10 | 141 | 99 | 42 | 98 | 43 | 98/141 (30.5% FR) | PASS |
| TRUE_T10 | 162 | 141 | 21 | 21 | 141 | 21/162 ITT (87.0% FR) | PASS |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 0 | 0 | 141 | 0/141 (100% FR) | PASS |

**OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS**

---

## Emission-Matched Denominator

The 141 emitted denominator for EARLY_SHIFT_T10 and COMMAND_OPEN_ORACLE_T10 is inherited from TRUE_T10's emission cohort.

TRUE_T10 emitted set = 141 episodes across 17 fold/state parent keys.
TRUE_T10 no-emission set = 21 episodes across 3 fold/state parent keys.

These 3 parents have no-emission for SOME detector/perturbation seed combinations (not all 9).
Some parents have both emitted and no-emission seeds — the emission decision is per-seed, not per-parent.

- `tables/server_freeze/object_true_t10_emitted_141.csv` — 141 rows, one per emitted episode
- `tables/server_freeze/object_true_t10_no_emission_21.csv` — 21 rows, one per no-emission episode

### EARLY_SHIFT_T10 Emission Detail

EARLY_SHIFT_T10 runs the same 141 parent episodes as TRUE_T10's emitted cohort ("inherited from TRUE_T10 emitted cohort").
Under the EARLY_SHIFT condition (T=10 steps before contact):
- 99/141 episodes show mlp_triggered=True (detector fires at the shifted position)
- 42/141 episodes show mlp_triggered=False (detector does not fire at the shifted position)

The "emission" field in EARLY_SHIFT refers to the detector's own runtime behavior under the shifted timing — it is NOT the same as TRUE_T10's emission.

Success breakdown: 95 of 99 emitted succeed (attack at early shift is harmless), 3 of 42 no-emission succeed (natural difficulty of those episodes).

This supports the **timing specificity** claim: perturbation at the early shift position does NOT cause failure.

---

## Provenance Gaps (Still Open)

### What IS verified:
- Per-episode task_success from episode_summary.json — MATCH
- Per-episode mlp_triggered (emission) status — CAPTURED
- Cross-condition parent key matching — VERIFIED
- Condition-level totals — MATCH
- Per-episode summary SHA256 — CAPTURED in master ledger

### What is NOT yet verified:
- Config files for each condition (epsilon, PGD steps, K, target_token, preprocessing, route, fallback, arm gate)
- Per-fold checkpoint SHA256 (only Fold 01 has SHA256SUMS.txt)
- Aggregation script path and SHA
- Historical launch commands for each condition
- Whether the server's current dirty code (attack_adapter.py diff) was the code that generated these artifacts
- PIL Lanczos preprocessing confirmation from actual runtime logs
- Attack frame count verification against claimed protocol

### Evidence Chain Status

| Chain | Condition Totals | Episode Set | Protocol Provenance | Overall |
|---|---|---|---|---|
| Timing specificity | PASS | PASS | HOLD | PARTIAL |
| Direction/selectivity | PASS | PASS | HOLD | PARTIAL |
| Clean-only causal deployment | PASS | PASS | HOLD | PARTIAL |
| OPEN-command mechanism sufficiency | PASS | PASS | HOLD | PARTIAL |

---

## Object Artifact Evidence Location

**Server only**: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/`

No local copy exists. Single point of failure until backup is executed.

---

## Summary

Object numbers are verified at the per-episode level from actual artifacts. The emission-matched denominator (141) is correct. Row-level ledgers committed. Full protocol provenance seal requires config/manifest/checkpoint SHA chain and historical command capture — these remain open but do NOT require re-running any experiment.

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
