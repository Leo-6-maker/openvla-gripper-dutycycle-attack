# Object Frozen Evidence Reconciliation — 2026-07-02 (REVISION 4)

## Status

**FROZEN_EMPIRICAL_RESULTS_UNDER_LEGACY_PROTOCOL_DRIFT**

All condition totals independently verified from episode_summary.json.
Cross-condition selection-mask equality confirmed (141 selected, 21 excluded).
Actual preprocessing confirmed as `upstream_tf_jpeg` (not claimed `official_pil_lanczos`).
Full protocol (epsilon/PGD/K/token/route/fallback/arm_gate) provenance is HOLD.

---

## Verified Results

### Condition Summary

| Condition | Total | Attack Applied | mlp_triggered | AA+T+ | AA+T- | AA-T+ | AA-T- | Success | Failure | Claimed | Match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CLEAN | 162 | 0 | 141 | 0 | 0 | 141 | 21 | 162 | 0 | 162/162 | PASS |
| RAND_T10 | 162 | 141 | 141 | 141 | 0 | 0 | 21 | 162 | 0 | 162/162 | PASS |
| RANDOM_TIME_V3 | 162 | 162 | 126 | 126 | 36 | 0 | 0 | 119 | 43 | 119/162 | PASS |
| EARLY_SHIFT_T10 | 141 | 141 | 99 | 99 | 42 | 0 | 0 | 98 | 43 | 98/141 | PASS |
| TRUE_T10 | 162 | 141 | 141 | 141 | 0 | 0 | 21 | 21 | 141 | 21/162 ITT, 0/141 emitted | PASS |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 141 | 141 | 0 | 0 | 0 | 0 | 141 | 0/141 | PASS |

Legend: AA=attack_applied, T=mlp_triggered, +=True, -=False.
Source: `tables/server_freeze/object_condition_summary.csv` (computed from v2 master ledger).

**OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS**

### Key Corrections (Revision 4)

- **RAND_T10**: attack_applied=True for all 141 emitted episodes (attack_frames=10). All 141 attacked episodes succeed. Previous report incorrectly stated attack_applied=0.
- **CLEAN**: attack_applied=False for all 162 episodes. Detector fires on 141 but no attack is applied. All succeed. Confirmed.
- **TRUE_T10**: attack_applied=True for 141 emitted, attack_applied=False for 21 no-emission. 0/141 emitted succeed, 21/21 no-emission succeed.
- **EARLY_SHIFT**: All 141 have attack_applied=True. 99 detector-triggered, 42 detector-silent. `mlp_triggered` is a post-treatment diagnostic, not an indicator of attack application.

---

## Cross-Condition Selection-Mask Reconciliation

`tables/server_freeze/object_cross_condition_key_reconciliation.csv` (162 rows):

- **141 selected keys**: TRUE_T10 emitted = EARLY_SHIFT_T10 = COMMAND_OPEN_ORACLE_T10
- **21 excluded keys**: present in TRUE_T10 no-emission set, absent from all three conditions
- TRUE selected minus EARLY = 0; EARLY minus TRUE selected = 0
- TRUE selected minus ORACLE = 0; ORACLE minus TRUE selected = 0

The verification confirms **selection-mask equality**: all three conditions operate on the identical 141-episode subset of the 162-episode universe. The 21 excluded episodes are the TRUE_T10 no-emission cohort (detector never triggers, attack never applied in TRUE_T10).

**TRUE_SELECTION_MASK_RECONCILIATION = PASS**

---

## Actual Attack Protocol (from artifact evidence)

### Confirmed

| Parameter | Value | Source |
|---|---|---|
| preprocessing | `upstream_tf_jpeg` | episode_summary.json field `preprocess_backend_resolved` |
| jpeg_roundtrip | True | episode_summary.json field `preprocess_uses_jpeg_roundtrip` |
| checkpoint_sha256 | `d15dab4d56b449e1ff89577ba198e1f683cd67c66aa31a2dc0767c7268165967` | episode_summary.json (all conditions) |
| dataset_sha256 | `b8944b23afaf5718bc2fcab32ce355002f5ec071698d12f5beb7316c51a77b11` | episode_summary.json (all conditions) |
| actual_attn | `eager` | episode_summary.json |
| actual_dtype | `bfloat16` | episode_summary.json |
| attack_frames (emitted) | 10 | episode_summary.json (when attack_applied=True) |

### UNKNOWN (not recorded in episode_summary.json)

| Parameter | Value | Recovery Method |
|---|---|---|
| epsilon | UNKNOWN | Historical config file or launch command |
| PGD steps | UNKNOWN | Historical config file or launch command |
| K (perturbation budget) | UNKNOWN | Historical config file or launch command |
| target_token | UNKNOWN | Historical config file or launch command |
| route (strict/fallback) | UNKNOWN | Historical config file or launch command |
| fallback | UNKNOWN | Historical config file or launch command |
| arm_gate | UNKNOWN | Historical config file or launch command |

### Protocol Drift

| Element | Claimed Frozen Protocol | Actual (from artifacts) | Status |
|---|---|---|---|
| Preprocessing | official_pil_lanczos | upstream_tf_jpeg | **MISMATCH** |
| JPEG roundtrip | No | Yes | **MISMATCH** |
| Epsilon | 6/255 | UNKNOWN | UNVERIFIED |
| PGD steps | 20 | UNKNOWN | UNVERIFIED |
| K | 10 | UNKNOWN | UNVERIFIED |
| Target token | 31744 | UNKNOWN | UNVERIFIED |
| Route | strict | UNKNOWN | UNVERIFIED |
| Fallback | none | UNKNOWN | UNVERIFIED |
| Arm gate | 5/6 | UNKNOWN | UNVERIFIED |

**CRITICAL**: The preprocessing mismatch (`upstream_tf_jpeg` vs `official_pil_lanczos`) does NOT imply epsilon=2/255. The `upstream_tf_jpeg` preprocessing was used in the PR #43 draft (which also proposed epsilon=2/255), but epsilon is an independent parameter not stored in episode_summary.json. The actual epsilon used for these artifacts is UNKNOWN until the historical config is recovered.

**Object results are FROZEN_EMPIRICAL_RESULTS_UNDER_LEGACY_PROTOCOL_DRIFT — valid numerical evidence, but the protocol description must be corrected to match what was actually run.**

---

## Evidence Files

| File | Rows | Fields | Content |
|---|---|---|---|
| `object_frozen_master_ledger.csv` | 930 | 29 | Full per-episode ledger with 64-char SHA256 |
| `object_condition_summary.csv` | 6 | 14 | Condition-level summary with attack-accounting cross-tabs |
| `object_cross_condition_key_reconciliation.csv` | 162 | 21 | Per-key cross-condition membership |
| `object_true_t10_emitted_141.csv` | 141 | 29 | TRUE_T10 emitted subset (from v2 ledger) |
| `object_true_t10_no_emission_21.csv` | 21 | 29 | TRUE_T10 no-emission subset (from v2 ledger) |

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
