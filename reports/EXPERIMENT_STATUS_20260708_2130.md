# Experiment Status Report — 2026-07-08 21:30 CST

**Commit**: `36712cc` | **Branch**: `plan/codex-gated-experiment-v1-c2e0`

## 1. D7 Table1 — Main Experiment (COMPLETE)

| Gate | Status | Detail |
|---|---|---|
| D7B2 Rollout | **716/716** | 32 workers (4/GPU), C2e3 GRU detector |
| D7C Postrun Audit | **PASS** | 0 missing, 0 unpaired, 0 contract violations |
| D7D Aggregate | **COMPLETE** | Panel A + O/G/S pooled + all-suite |
| D7E Render | **COMPLETE** | Markdown with corrected labels |
| Paired McNemar | **COMPLETE** | O/G/S TRUE vs RAND p<0.001 |

### Table 1 — O/G/S Pooled (Main Evidence, N=129)

| Condition | Success/N | SR | 95% CI |
|---|---|---|---|
| CLEAN | 115/129 | 89.1% | [82.6, 93.4] |
| TRUE_T10 | 74/129 | 57.4% | [48.8, 65.5] |
| RAND_T10 | 114/129 | 88.4% | [81.8, 92.8] |
| COMMAND_OPEN_ORACLE | 92/129 | 71.3% | [63.0, 78.4] |

**Status**: `D7_TABLE1_MAIN_RESULT = PASS_AUDITED_WITH_L10_STRATIFIED_CAVEAT`

### Evidence Roots

| Artifact | Path | Key SHA256 |
|---|---|---|
| Rollout | `/mnt/sdc/.../d7b2_table1_normalized_rollout/` | 716 summaries |
| Audit | `/mnt/sdc/.../d7b2_audit/` | `5a34a455...` |
| Aggregate | `/mnt/sdc/.../d7b2_aggregate/` | `9d0b572e...` |
| Render | `/mnt/sdc/.../d7b2_render/` | `e803fe63...` |
| Paired Stats | `/mnt/sdc/.../d7b2_paired_stats/` | `d08c09b0...` |

## 2. D8F — 25D-Only Detector Route (CLOSED)

| Experiment | Result |
|---|---|
| D8F1 Selective Abstention | Object/Goal/Spatial ~95% recall, L10=1.1% |
| D8F2 Suite-Balanced Abstention | L10 recall collapsed to 1.1% |
| D8A Ceiling Closeout | All 7 25D variants converge to FP 31-39%, L10 recall <55% |

**Conclusion**: 25D-only route closed. Next: C2f observation/language detector.

## 3. C2f — Observation/Language Detector Pipeline

### C2f Smoke3 (PASS)

| Check | Result |
|---|---|
| Collection | 3 episodes, 732 steps |
| RGB → agentview_image | 732 PNG = 732 steps (perfect parity) |
| features_25d | 732/732 OK, length=25, 0 NaN |
| task_language | Non-empty via robust resolver |
| Stats materialization | 687 windows, PASS_MATERIALIZED |
| Label signal | INCONCLUSIVE (primary=0, expected for L10 smoke) |

**Status**: `C2F_SMOKE3 = PASS_WITH_LABEL_SIGNAL_INCONCLUSIVE`

### C2f Clean2000 Collection (IN PROGRESS)

```
12 workers (3/GPU × GPUs 4-7), 36 CPU threads
2000 episodes: L10=500, Object=500, Goal=500, Spatial=500
~167 episodes/worker
```

| GPU | Suite | Workers | Memory | Status |
|---|---|---|---|---|
| 4 | libero_10 | ×3 | 72.7 GB | Loading |
| 5 | libero_object | ×3 | 76.0 GB | Running |
| 6 | libero_goal | ×3 | 75.9 GB | Running |
| 7 | libero_spatial | ×3 | 76.2 GB | Running |

### C2f Infrastructure (READY)

| Component | File | Status |
|---|---|---|
| Data Spec | `docs/detectors/C2F_OBSERVATION_LANGUAGE_DATA_SPEC.md` | ✅ |
| Collector | `scripts/stageb/collect_c2f_observation_clean_rollouts.py` | ✅ |
| Adapter (D7-aligned) | `scripts/stageb/c2f_libero_openvla_adapter.py` | ✅ |
| Materializer | `tools/multisuite_detector/materialize_c2f_frozen_embeddings.py` | ✅ |
| Trainer (A/B/C/D) | `tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py` | ✅ |
| Ablation Dataset Maker | `tools/multisuite_detector/make_c2f_ablation_datasets.py` | ✅ |
| Hygiene Checker | `scripts/stageb/check_c2f_collection_hygiene.py` | ✅ |
| Collection Audit | `scripts/stageb/audit_c2f_observation_collection.py` | ✅ |
| Sharded Launcher | `scripts/stageb/launch_c2f_clean2000_sharded.py` | ✅ |
| Shard Merger | `scripts/stageb/merge_c2f_sharded_collection.py` | ✅ |
| D8F Ceiling Closeout | `docs/detectors/D8F_25D_ONLY_CEILING_CLOSEOUT.md` | ✅ |
| D7 Closeout | `docs/results/D7_TABLE1_CLOSEOUT_20260708.md` | ✅ |
| D7 Paper-Ready | `docs/results/D7_TABLE1_PAPER_READY_SUMMARY.md` | ✅ |

## 4. Detector Configuration

| Parameter | Value |
|---|---|
| Detector | C2e3 GRU W=16 H=128 |
| Checkpoint SHA | `3283f9492902f8cb...` |
| τ_emit / τ_suppress | 0.33 / 0.67 |
| ε | 6/255 |
| K | 10 |
| MAX_STEPS | 300 |
| Input | 25D proprio/action + 108D context |
| Training data | C2e1 clean-only temporal dataset |
| NO: RGB, language, attack outcome, privileged state in student input |

## 5. Next Steps

```
Clean2000 complete →
  merge shards →
  hygiene check →
  observation audit →
  stats materialization (full) →
  CLIP materialization (GPU) →
  A/B/C/D ablation training →
  C2f gate decision
```

**Priority**: D7 frozen evidence > Clean2000 collection > C2f training

## 6. Boundaries

- D7 Table1 is frozen and audited
- C2f does not modify D7
- C2f student input excludes privileged state
- Clean2000 is clean-only, no attack
- D8F route is permanently closed
