# Object Frozen Evidence Reconciliation — 2026-07-02 (REVISION 3)

## Status

**FROZEN_REPORTED_RESULT_WITH_PROVENANCE_SEAL_PENDING**

Condition totals and episode sets independently verified from episode_summary.json.
Cross-condition key matching confirmed: 162 keys, 0 mismatches.
Full protocol provenance seal is HOLD due to preprocessing deviation and incomplete config/manifest SHA chain.

---

## Independent Verification

All numbers re-computed from `episode_summary.json` via `tables/server_freeze/object_frozen_master_ledger.csv` (930 rows, 29 fields).

| Condition | Leaves | Emitted | No-Emission | Attack Applied | Success | Failure | Claimed | Match |
|---|---|---|---|---|---|---|---|---|
| CLEAN | 162 | 141 | 21 | 0 | 162 | 0 | 162/162 | PASS |
| RAND_T10 | 162 | 141 | 21 | 0 | 162 | 0 | 162/162 | PASS |
| RANDOM_TIME_V3 | 162 | 126 | 36 | 162 | 119 | 43 | 119/162 | PASS |
| EARLY_SHIFT_T10 | 141 | 99 | 42 | 141 | 98 | 43 | 98/141 | PASS |
| TRUE_T10 | 162 | 141 | 21 | 141 | 21 | 141 | 21/162 ITT, 0/141 emitted | PASS |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 0 | 141 | 0 | 141 | 0/141 | PASS |

**OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS**

---

## Cross-Condition Key Reconciliation

`tables/server_freeze/object_cross_condition_key_reconciliation.csv` (162 rows) proves:

```
MATCH=162 MISMATCH=0
TRUE_MINUS_EARLY=0 EARLY_MINUS_TRUE=0
TRUE_MINUS_ORACLE=0 ORACLE_MINUS_TRUE=0
```

All 162 (fold, state, detector_seed, perturbation_seed) tuples in TRUE_T10 emitted set are present in EARLY_SHIFT_T10 and COMMAND_OPEN_ORACLE_T10, and vice versa.

**OBJECT_CROSS_CONDITION_KEY_MATCH = PASS**

---

## Emission Semantics (CORRECTED)

### TRUE_T10
- `mlp_triggered=True` means the detector fired and the attack was applied (141 episodes)
- `mlp_triggered=False` means the detector did NOT fire and the attack was NOT applied (21 episodes)
- `attack_applied` = `attack_frames > 0` — in TRUE_T10, this matches `mlp_triggered` exactly
- 21 no-emission episodes: attack NOT applied, all 21 succeed

### EARLY_SHIFT_T10 (CORRECTED)
- ALL 141 episodes have `attack_applied=True` (attack_frames=10 for all)
- `mlp_triggered` in EARLY_SHIFT is a diagnostic/post-treatment variable — it records whether the detector fires at the shifted timing position, NOT whether the attack is applied
- 99 episodes: detector fires at shifted position → 95 succeed, 4 fail
- 42 episodes: detector does NOT fire at shifted position → 3 succeed, 39 fail
- **The 42 "no-emission" episodes are actually "attack applied but detector silent"**
- **Previous claim that failure in these 42 is "natural difficulty" is INCORRECT** — the attack IS applied to all 141

Correct framing:
> EARLY_SHIFT_T10 applies the perturbation at T=10 steps before contact to all 141 TRUE-emitted episodes. The detector fires on 99 and is silent on 42. Overall FR is 30.5% (43/141), significantly lower than TRUE_T10 emitted-only FR of 100%, consistent with timing specificity. However, the 39/42 failure rate in the detector-silent subgroup means early-shift is not "harmless" and the causal pathway requires further analysis.

---

## Preprocessing Protocol Deviation (CRITICAL)

All 930 episodes across all 6 conditions used:

```
preprocess_backend_requested  = upstream_tf_jpeg
preprocess_backend_resolved   = upstream_tf_jpeg
preprocess_uses_jpeg_roundtrip = True
```

This is the **PR #43 draft protocol** (epsilon=2/255 + upstream_tf_jpeg), NOT the claimed frozen protocol (epsilon=6/255 + official_pil_lanczos).

| Protocol Element | Claimed (Frozen) | Actual (from artifact) | Match |
|---|---|---|---|
| Preprocessing | official_pil_lanczos | upstream_tf_jpeg | **MISMATCH** |
| JPEG roundtrip | No | Yes (preprocess_uses_jpeg=True) | **MISMATCH** |
| Epsilon | 6/255 (claimed) | Unknown (not in episode_summary) | UNVERIFIED |
| PGD steps | 20 (claimed) | Unknown | UNVERIFIED |
| K | 10 (claimed) | Unknown | UNVERIFIED |
| Target token | 31744 (claimed) | Unknown | UNVERIFIED |
| Route | strict (claimed) | Unknown | UNVERIFIED |
| Fallback | none (claimed) | Unknown | UNVERIFIED |
| Arm gate | 5/6 (claimed) | Unknown | UNVERIFIED |

The preprocessing field is recorded in every episode_summary.json. The mismatch between claimed `official_pil_lanczos` and actual `upstream_tf_jpeg` means the frozen attack protocol description is incorrect for the actual generated artifacts.

---

## Remaining Provenance Gaps

1. **Preprocessing**: artifacts use `upstream_tf_jpeg`, not `official_pil_lanczos` — protocol description must be corrected OR artifacts were generated under a different protocol version
2. **Epsilon**: not recorded in episode_summary.json — needs config file or launch command recovery
3. **PGD steps / K / target_token**: not recorded in episode_summary.json
4. **Route / fallback / arm gate**: not recorded in episode_summary.json
5. **Config file SHA per condition**: not captured
6. **Manifest SHA per condition**: 32 manifests hashed but not individually listed with condition mapping
7. **Checkpoint SHA**: present in episode_summary (`d15dab4d56b4...`) but not cross-referenced against model files
8. **Aggregation script**: path and SHA not captured
9. **Historical launch commands**: not captured

---

## Evidence Location

Server only: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/`

No local copy. Single point of failure.

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
