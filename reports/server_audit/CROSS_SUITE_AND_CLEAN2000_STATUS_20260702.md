# Cross-Suite and CLEAN2000 Status — 2026-07-02

## CLEAN300 Status

| Item | Value |
|---|---|
| Planned | 300 episodes across multi-suite |
| Physically Discovered | 162 episodes (LIBERO-Object only) |
| Valid | All 162 from Object suite |
| Clean Success | 162/162 (100%) |
| Clean Failure | 0 |
| Duplicates | None detected |
| Missing | 138 (Spatial/Goal/10 not yet processed) |
| Schema Fail | 0 |
| Telemetry Completeness | Full for Object, missing for other suites |
| Raw Object Detector Emits | Available via LOTO folds |

**Status**: Object-only data complete. Cross-suite expansion not executed. LOTO 10-fold detector trained on Object only. Zero-shot replay on Spatial/Goal/10 data NOT performed.

---

## CLEAN2000 Status

### Data Inventory

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

### Suite Distribution (Estimated)

| Suite | Episodes (approx) | Clean Success | Notes |
|---|---|---|---|
| LIBERO-Object | ~500 | TBD | Most complete |
| LIBERO-Spatial | ~500 | TBD | Partial |
| LIBERO-Goal | ~500 | TBD | Partial |
| LIBERO-10 | ~500 | TBD | Partial |

Exact per-suite counts require parsing INDEX_DRAFT.jsonl.

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
