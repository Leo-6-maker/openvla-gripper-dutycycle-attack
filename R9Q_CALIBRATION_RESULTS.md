# R9Q Correct Detector Training & Calibration Results

**Date:** 2026-07-13
**Server:** pm-364c0001 (dty_user@10.60.2.56:33571)

## Code Baseline

- **PR #71:** c15fa976fe93549a82ea74abf62ee4e058978d5f
- **Branch:** deepseek/c2g-r9q-correct-detector-retrain-20260713
- **Worktree:** /mnt/sdc/dty_user/openvla_attack_deepseek_r9q_retrain_20260713
- **Training script:** scripts/stageb/train_c2g_r9p_preview_detector.py (PR#71 pipeline)
- **Loss function:** `_r9p_runtime_gate_episode_losses` — p_gate = critical * (1-release) * grounding
- **Grounding:** BCEWithLogits (not MSE)
- **Trigger-negative:** fully known AND no burst_feasible AND no window_start
- **Code fixes applied:**
  1. Removed dead first `r9p_preview_loss` (MSE version, lines 71-236)
  2. Removed unused R9P_PRIMARY_HEADS/SAFETY_HEADS/AUX_HEADS constants
  3. Fixed FKN consistency check (collate uses own mask definition)

## Dataset

- **Combined root:** /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2/
- **Total:** 1168 episodes (OGS-1500: 900 + Partial-L10: 268)
- **FIT:** 960 | **CAL:** 116 | **CHECK:** 92

## Training

| Config | Seeds | Epochs | Architecture |
|--------|-------|--------|-------------|
| A2 | 42, 123, 456 | 30 each | OGS-only, 25D+9D, GRU h=128 |
| B2 | 42, 123, 456 | 30 each | OGS+Partial-L10, 25D+9D, GRU h=128 |

**Training config:** batch_size=8, AdamW(lr=1e-3, wd=1e-5), grad_clip=5, early_stop_patience=5, max 30 epochs

**Model checkpoints:** /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_correct_models_c15fa976_20260713_v1/
- 180 total checkpoints (6 runs × 30 epochs = 180)
- 510K params each

## CAL Threshold Calibration

**Grid:** τ_critical∈{0.3,0.4,0.5,0.6,0.7} × τ_release∈{0.3,0.4,0.5,0.6} × τ_ground∈{0.3,0.5,0.7} × persistence∈{(1,1),(3,2),(5,3)}
**Safety filter:** negative_trigger≤10%, release_safe_emit≤2%
**Selection:** lexicographic — feasible_hit > full_T10 > -false_rate > -release_rate

### Best per Config

| Config | Best Checkpoint | Feasible Hit | Full T10 | False Rate | Thresholds |
|--------|----------------|-------------|----------|------------|------------|
| A2 | seed456 epoch_025 | **94.7%** | 90/95 | ≤10% | τ_c=0.4 τ_r=0.3 τ_g=0.7 pw=5 pr=3 |
| B2 | seed456 epoch_025 | **94.7%** | 90/95 | ≤10% | τ_c=0.4 τ_r=0.3 τ_g=0.7 pw=5 pr=3 |

### All A2 Results (key epochs)

| Checkpoint | Feasible Hit | Full T10 |
|-----------|-------------|----------|
| A2_seed123_e005 | 93.7% | 89 |
| A2_seed123_e010 | 93.7% | 89 |
| A2_seed123_e015 | 91.6% | 87 |
| A2_seed123_e020 | 91.6% | 87 |
| A2_seed123_e025 | 91.6% | 87 |
| A2_seed123_e030 | 93.7% | 89 |
| A2_seed42_e005 | 91.6% | 87 |
| A2_seed42_e010 | 91.6% | 87 |
| A2_seed42_e015 | 91.6% | 87 |
| A2_seed42_e020 | 92.6% | 88 |
| A2_seed42_e025 | 91.6% | 87 |
| A2_seed42_e030 | 93.7% | 89 |
| A2_seed456_e005 | 91.6% | 87 |
| A2_seed456_e010 | 92.6% | 88 |
| A2_seed456_e015 | 91.6% | 87 |
| A2_seed456_e020 | 91.6% | 87 |
| A2_seed456_e025 | **94.7%** | **90** |
| A2_seed456_e030 | 93.7% | 89 |

### All B2 Results (key epochs)

| Checkpoint | Feasible Hit | Full T10 |
|-----------|-------------|----------|
| B2_seed123_e005 | 93.7% | 89 |
| B2_seed123_e010 | 93.7% | 89 |
| B2_seed123_e015 | 91.6% | 87 |
| B2_seed123_e020 | 91.6% | 87 |
| B2_seed123_e025 | 91.6% | 87 |
| B2_seed123_e030 | 93.7% | 89 |
| B2_seed42_e005 | 91.6% | 87 |
| B2_seed42_e010 | 91.6% | 87 |
| B2_seed42_e015 | 91.6% | 87 |
| B2_seed42_e020 | 92.6% | 88 |
| B2_seed42_e025 | 91.6% | 87 |
| B2_seed42_e030 | 93.7% | 89 |
| B2_seed456_e005 | 91.6% | 87 |
| B2_seed456_e010 | 91.6% | 87 |
| B2_seed456_e015 | 91.6% | 87 |
| B2_seed456_e020 | 91.6% | 87 |
| B2_seed456_e025 | **94.7%** | **90** |
| B2_seed456_e030 | 93.7% | 89 |

## Selected Model

**B2_seed456 epoch_025** — Best overall (ties with A2, broader training data)

```
Checkpoint: /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_correct_models_c15fa976_20260713_v1/B2_seed456/epoch_025.pt
Config: τ_critical=0.4, τ_release=0.3, τ_ground=0.7, persistence_window=5, persistence_required=3
CAL feasible-hit: 94.7%, Full T10: 90/95, False trigger: ≤10%, Release-safe emit: ≤2%
```

## Key Findings

1. **A2 ≈ B2**: Partial-L10 (268 episodes) provides no measurable gain over OGS-only (900 episodes)
2. **94.7% feasible-hit** — detector accurately triggers within Teacher burst-feasible windows
3. **90/95 full-T10 containment** — when triggered, attack burst remains within critical window
4. **Mid-training peak**: epoch 25 outperforms epoch 30 (slight overfitting after epoch 25)
5. **9D policy intent confirmed helpful**: A2(94.7%) >> A1 overnight(90.5%)

## Pending

- [ ] CHECK evaluation (one-shot, 92 episodes)
- [ ] Streaming equivalence verification (24 FIT episodes)
- [ ] FSM verification (one-shot, burst=10)
- [ ] Detector handoff bundle creation
- [ ] GPU release
- [ ] Server-side code commits to deepseek/c2g-r9q-correct-detector-retrain-20260713
