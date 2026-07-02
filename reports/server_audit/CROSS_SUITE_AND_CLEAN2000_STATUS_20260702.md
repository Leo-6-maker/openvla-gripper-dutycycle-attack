# Cross-Suite and CLEAN2000 Status — 2026-07-02

## CLEAN300 Status — ARCHIVED AND SUPERSEDED

**AMENDED 2026-07-02:** CLEAN300 is officially archived and superseded. It has NO scientific authority for current planning.

| Field | Value |
|---|---|
| CLEAN300_STATUS | ARCHIVED_SUPERSEDED_DATA |
| CLEAN300_SCIENTIFIC_AUTHORITY | NONE |
| CLEAN300_FUTURE_USE | HISTORICAL_REFERENCE_ONLY |
| Superseded by | dty-server CLEAN2000 (2000 episodes, 4 suites) |
| Historical note | ARCHIVED_ACCEPTED_AT_THE_TIME_BUT_SUPERSEDED_BY_DTY_CLEAN2000 |

The 162 Object episodes previously attributed to CLEAN300 were the LIBERO-Object subset now fully incorporated into CLEAN2000's 500 Object episodes. Do not use CLEAN300 numbers for any current analysis.

---

## CLEAN2000 Status — AUTHORITATIVE CROSS-SUITE CORPUS

### Data Inventory (verified 2026-07-02)

| File | Count | Description |
|---|---|---|
| CLEAN2000_INDEX_DRAFT.jsonl | 2000 | Full episode index |
| CLEAN2000_ATTEMPT_LEDGER.jsonl | 2000 | All collection attempts |
| CLEAN2000_PRIMARY.jsonl | 1043 | Validated primary episodes |
| CLEAN2000_FEATURES_25D_ALL_STEPS.csv | — | 25D features for all steps |
| CLEAN2000_FEATURES_25D_VALID_ONLY.csv | — | 25D features for valid episodes |
| CLEAN2000_TEACHER_LABEL_INDEX.jsonl | — | Teacher mechanism labels |
| CLEAN2000_TEACHER_CROSS_VALIDATION.json | — | Teacher label validation results |
| CLEAN2000_FEATURE_GOLDEN_PARITY.json | — | Feature extraction parity check |
| SHA256SUMS.txt | — | Content integrity hashes |
| FREEZE_ENVELOPE.json | — | Freeze metadata |

### Data Quality

| Check | Status |
|---|---|
| INDEX completeness | 2000/2000 (100%) |
| ATTEMPT LEDGER completeness | 2000/2000 (100%) |
| PRIMARY validation rate | 1043/2000 (52.15%) |
| Feature 25D availability | YES (all steps + valid only) |
| Teacher labels present | YES |
| Teacher cross-validation done | YES |
| SHA256 content integrity | VERIFIED |
| Server adapter scripts | PRESENT (dirty — see divergence report) |
| Formal train/val/test split | CLEAN2000_SPLITS_V1 exists |
| Training release package | CLEAN2000_TRAINING_RELEASE_V1 exists |

### Why Only 1043/2000 Primary?

The 957 episodes not in PRIMARY.jsonl fall into these categories (based on ATTEMPT_LEDGER analysis — needs detailed audit):
- Schema validation failures (missing required fields)
- Infrastructure failures during collection
- Telemetry incomplete (truncated rollouts)
- Supplementary events (not primary mechanism events)
- Multi-event episodes (ambiguous mechanism)
- Unsupported mechanism types

Exact breakdown requires reading ATTEMPT_LEDGER.jsonl in detail.

### Suite Distribution (Verified Exact)

| Suite | Total | Success | Failure | Teacher Eligible | Primary | Primary % |
|---|---|---|---|---|---|---|
| libero_spatial | 500 | 411 | 89 | 500 (100%) | 411 | 82.2% |
| libero_object | 500 | 367 | 133 | 500 (100%) | 367 | 73.4% |
| libero_goal | 500 | 383 | 117 | 300 (60%) | 234 | 46.8% |
| libero_10 | 500 | 231 | 269 | 50 (10%) | 31 | 6.2% |
| **TOTAL** | **2000** | **1392** | **608** | **1350 (67.5%)** | **1043** | **52.15%** |

### Exact Episode Classification (Sum = 2000)

| Category | Count | % |
|---|---|---|
| PRIMARY_ELIGIBLE | 1043 | 52.15% |
| CLEAN_FAILURE_SAFETY | 307 | 15.35% |
| SUPPLEMENTARY (multi_object_two_items) | 200 | 10.00% |
| SUPPLEMENTARY (articulated_task_no_pick_place) | 100 | 5.00% |
| SUPPLEMENTARY (multi_object_two_different_targets) | 100 | 5.00% |
| SUPPLEMENTARY (multi_stage_open_drawer_then_place) | 50 | 2.50% |
| SUPPLEMENTARY (multi_stage_place_then_close) | 50 | 2.50% |
| SUPPLEMENTARY (multi_stage_place_then_close_drawer) | 50 | 2.50% |
| SUPPLEMENTARY (push_task_not_pick_place) | 50 | 2.50% |
| SUPPLEMENTARY (multi_stage_turn_on_then_place) | 50 | 2.50% |
| **TOTAL** | **2000** | **100.00%** |

---

## Teacher Label / Mechanism Resolver Status

### What Exists

- `CLEAN2000_TEACHER_LABEL_INDEX.jsonl` — teacher-assigned mechanism labels
- `CLEAN2000_TEACHER_CROSS_VALIDATION.json` — cross-validation audit of teacher labels
- Per-fold teacher labels in `sc5_object_privileged_loto_v1/fold_0*/FOLD0*_teacher_labels_heldout.jsonl`
- Per-fold teacher labels in `FOLD0*_teacher_labels_train_val.jsonl`

### What's Missing

- Cross-suite teacher labels (Spatial, Goal, 10)
- Mechanism resolver for ambiguous episodes
- Primary/secondary event classification for multi-event episodes
- Inter-rater reliability metrics for teacher labels
- Hard case audit trail

---

## Detector Training Status

### Object-Only Detector (LOTO 10-Fold)

| Fold | Features | Labels | Training | Phase B Eval |
|---|---|---|---|---|
| 01 | YES | YES | NOT RUN | NOT RUN |
| 02 | YES | YES | YES (v3) | YES |
| 03 | YES | YES | YES (v3) | YES |
| 04 | YES | YES | YES (v3) | YES |
| 05 | YES | YES | YES (v3) | YES |
| 06 | YES | YES | YES (v3) | YES |
| 07 | YES | YES | YES (v3) | YES |
| 08 | YES | YES | YES (v3) | YES |
| 09 | YES | YES | YES (v3) | YES |

8/9 folds have completed training (v3). Fold 01 has data prepped but no training artifacts found.

### Pooled Detector

**NOT TRAINED.** Training data (CLEAN2000_TRAINING_RELEASE_V1) and splits (CLEAN2000_SPLITS_V1) exist on server but no training run has been executed.

### LOSO Detector (Leave-One-Suite-Out)

**NOT TRAINED.** No training manifests, launch scripts, or output directories found for cross-suite LOSO.

### Detector Checkpoints

No detector model checkpoints found at expected paths. The LOTO fold training outputs may exist in `fold_0*/training_v3/` but were not inspected for .pt/.safetensors files.

---

## Layer 1 / Layer 2 / Layer 3 Smoke Status

| Layer | Description | Status |
|---|---|---|
| Layer 1 | Proprioceptive-only (no vision) | ENGINEERING_SMOKE — config exists, no formal results |
| Layer 2 | Vision+proprioceptive fusion | NOT STARTED |
| Layer 3 | Full VLA features | DATA COLLECTED (CLEAN2000 features), no training |

Smoke tests were run to validate pipeline but no formal cross-suite detector results exist.

---

## Cross-Suite Attack Pilot

**NOT EXECUTED.** The `cross_suite/clean_qualify` directory exists on server but contains only qualification data, no attack rollout results. Cross-suite attack pilot was planned but blocked by:
1. Project pivot (Issue #46)
2. CLEAN2000 labeling incomplete (52% primary)
3. Detector training not done
4. Attack protocol freeze not finalized for cross-suite

---

## Model Checkpoints (Server)

| Path | Description |
|---|---|
| `/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object` | LIBERO-Object fine-tuned |
| `/mnt/sdc/dty_user/openvla_attack/models/libero-spatial` | LIBERO-Spatial |
| `/mnt/sdc/dty_user/openvla_attack/models/libero-goal` | LIBERO-Goal |
| `/mnt/sdc/dty_user/openvla_attack/models/libero-10` | LIBERO-10 |

---

## Summary

| Component | State | Ready for Training | Ready for Paper |
|---|---|---|---|
| CLEAN300 | Object-only | NO (no cross-suite) | NO |
| CLEAN2000 Data | 2000 collected, 1043 primary | CONDITIONAL | NO |
| Teacher Labels | Object complete, cross-suite missing | NO | NO |
| Object Detector (LOTO) | 8/9 folds trained | CONDITIONAL | NO |
| Pooled Detector | Not trained | NO | NO |
| LOSO Detector | Not trained | NO | NO |
| Layer 1-3 Smoke | Engineering only | NO | NO |
| Cross-Suite Attack Pilot | Not executed | NO | NO |

**CLEAN2000 is DATA_COLLECTED, NOT READY_FOR_TRAINING.** The primary bottleneck is the 52% primary validation rate (1043/2000) and missing cross-suite teacher labels. Before any training:
1. Audit the 957 non-primary episodes
2. Complete teacher labeling for cross-suite data
3. Resolve mechanism ambiguity for multi-event episodes
4. Validate train/val/test split integrity
5. Freeze feature extraction pipeline version

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
