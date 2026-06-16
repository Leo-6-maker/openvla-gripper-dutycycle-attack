# L12 Post-Reboot Freeze Check

**Date:** 2026-06-16
**Operator:** Yuliu / Claude

## Repository State

| Field | Value |
|-------|-------|
| HEAD | `52bdc33507a584e158581f9eded6dc657ac71029` |
| Branch | `exp/l12-production-streaming-adapter-20260615` |
| Remote origin | `https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git` |
| local == remote | YES |
| Working tree tracked files | CLEAN (no modified or deleted tracked files) |
| Working tree untracked | Explainable: docs/, ops/, patches, temp scripts, bundles, reports_temp/ |

## Frozen Artifact Manifest

All artifacts verified present and SHA256 computed on server (klfy-SYS-4028GR-TR2).

| # | Artifact | Path | SHA256 | Lines | Status |
|---|----------|------|--------|-------|--------|
| 1 | Frozen config | `/data/liuyu/outputs/d5_training/d5_frozen_config.json` | `d6f6af61...` | 8 | OK |
| 2 | Checkpoint | `/data/liuyu/outputs/d5_training/d5_candidate_best.pt` | `7eea609f...` | binary | OK |
| 3 | Teacher-P labels v2 | `/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv` | `e731c273...` | 121 | OK |
| 4 | Accepted episode manifest | `/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv` | `fe125a55...` | 121 | OK |
| 5 | Independent Teacher-P audit | `/data/liuyu/outputs/d5_label_generation/d5_teacher_p_independent_audit.csv` | `fb7a8aac...` | 121 | OK |
| 6 | D5 evaluation readout | `/data/liuyu/outputs/d5_training/d5_evaluation_readout.csv` | `76dd307b...` | 121 | OK |
| 7 | 34-state external eval readout | `/data/liuyu/outputs/d5_training/d5_34eval_readout.csv` | `0ba7c63b...` | 35 | OK |

## Frozen Config (d5_frozen_config.json)

```json
{
  "tau": 0.05,
  "checkpoint_path": "/data/liuyu/outputs/d5_training/d5_candidate_best.pt",
  "model_architecture": "CandidateRanker_MLP_128_64_32",
  "feature_schema_version": "d1b_v1",
  "n_features": 16,
  "selection_metric": "max_in_window_on_val",
  "tie_break": "prefer_higher_tau"
}
```

## Checkpoint Internals

| Key | Value |
|-----|-------|
| n_features | 16 |
| feature_names | total_score, raw_crossing_bonus, close_streak_bonus, close_onset_qpos_bonus, eef_deceleration_bonus, qpos_ready_bonus, eef_speed_now, eef_speed_prev, eef_deceleration_delta, close_streak, raw_crossing, close_onset, qpos, time_since_prev_close, time_since_last_open, candidate_index |
| means | 16 |
| stdevs | 16 |
| impute | 16 arrays |
| best_val_acc | present |

## Frozen Performance (for reference — DO NOT RETRAIN)

| Metric | Value |
|--------|-------|
| Internal labeled | 99 |
| Internal abstain | 21 |
| Internal in-window | 75/99 (75.8%) |
| Internal emit | 95/99 (96.0%) |
| Internal test in-window | 11/15 (73.3%) |
| External labeled | 27 |
| External abstain | 7 |
| External in-window | 16/27 (59.3%) |
| D5 causal replay | 76% |
| First-CLOSE | 74% |

## Phase 0 Gates

| Gate | Status |
|------|--------|
| HEAD == 52bdc335... | PASS |
| local == remote HEAD | PASS |
| Working tree clean (tracked files) | PASS |
| Untracked files explainable | PASS |
| All 7 frozen artifacts present | PASS |
| tau == 0.050 | PASS |
| Checkpoint loadable | PASS |
| Teacher-P auditor 120/120 | PASS (previously verified) |

**Phase 0 RESULT: ALL GATES PASS**
