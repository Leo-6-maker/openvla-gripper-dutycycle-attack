# V2 Stage-1 Formal Closure + Corrected Exact-W32 Confirmation — 2026-07-22

## 1. Stage-1 Grid: FORMALLY CLOSED

```
Status:      HOLD_NO_OBSERVED_ELIGIBLE_CONFIG
Reason:      frozen raw any-head union gate (background_false_emit_rate ≤ 0.10)
Coverage:    416/864 full pipeline (train→predict→evaluate→audit)
Observed:    67 configs irreversibly fail worst-split safety gate
             5 V2C configs unobserved (non-random timeout censoring)
             V2C systemic: per-step GRU for-loop exceeds 3600s under GPU contention
Candidates:  0/72 pass safety gate
```

**What happened**: the preregistered gate required uncalibrated raw model output at p=0.5 to satisfy background_false_emit_rate ≤ 0.10 with any-head union. All 67 observed configs fail this gate. The failure is NOT a release discrimination failure — release head background emit averages 0.013, and release overlap / unsupported route gates pass universally. The failure is from grasp head raw emit (~0.128 mean), which may or may not produce actual FSM false triggers after CAL-selected thresholds and persistence filtering.

**What does NOT follow from this**: "the Student architecture cannot learn to detect gripper vulnerabilities" or "the V2 approach is a dead end." A raw-emit metric on uncalibrated logits before the deployment decision layer (threshold, persistence, FSM transitions) is not a valid deployment safety verdict.

## 2. Three Implementation Defects in Original Frozen Code

These defects mean the 864 grid was NOT a valid architecture comparison experiment, even if completed:

| Defect | Impact |
|--------|--------|
| TCN `receptive_field` is a minimum, not exact: W=16→RF=31, W=32→RF=63, W=64→RF=127 | Context argument invalid |
| `class_weights` computed per-route but may not reach loss for all heads | Weighted loss claim unverified |
| Temporal jitter produces invalid prefix steps not excluded from supervision via valid_mask intersection | Supervision leakage |

These are fixed in the corrected exact-W32 implementation (`v5_factorized_student_v2_recommended.py`):
- `ExactCausalTCNEncoder`: kernel_size=2, dilations 1,2,...,W/2, RF exactly W
- `RecommendedEventBalancedLoss`: valid_mask intersected with every Teacher known mask
- Route-specific class weights explicitly consumed per head

## 3. 864 Grid: What It IS Still Useful For

- Architecture trend analysis (V2A vs V2B vs V2C)
- Release discrimination superiority over LR (mean ΔAUPRC ≈ +0.07)
- Evidence that V2C is computationally infeasible for production
- Documentation of raw-operating-point failure mode
- Input to CAL/FSM design (which heads need threshold tuning)

It must NOT be cited as:
- "72-config architecture comparison" (5 V2C unobserved, non-random censoring)
- "V2B is worse than V2A" (implementation defects, class weights may not have reached loss)
- "No Student can pass safety" (gate was at wrong stage of the pipeline)

## 4. Corrected Exact-W32 Confirmation: Go Decision

Frozen configuration (pre-registered, NOT selected from grid):

```
Candidate:       V2B_RECOMMENDED_EXACT
Encoder:         ExactCausalTCNEncoder, W=32, H=64
Input:           25D runtime-only, causal
Dropout:         0.1
Weight decay:    1e-4
Optimizer:       AdamW, lr=1e-3
Batch:           8
Epochs:          30
Canonical seed:  42
formal_selection_eligible: false
```

Basis for this config (NOT a claim of global optimality):
- Exact-W32 is logically clean (RF equals context)
- Three known implementation defects are fixed
- Single-split sidecar AUPRC ≈ 0.854, short-AUPRC ≈ 0.959 (comparable to best V2A)
- Small model, causal, low online cost
- W32 provides reasonable margin around known ~8-10 step vulnerable window

## 5. Confirmation Path (12-split → Full-FIT → CAL → CHECK → Shadow → Smoke → Matrix)

1. **Complete 12-split inner-CV** (1/12 done, 11 remaining) — `launch_factorized_v2_recommended_canary.py`
2. **Full-FIT retrain** all FIT identities, seed=42 canonical, seeds 123/456 parallel for stability
3. **CAL** selects: head-specific thresholds, persistence, FSM transitions, one-shot rule, attack duration (NOT model architecture)
4. **CHECK** once: does threshold/FSM generalize? If fail → stop, no recalibration loop
5. **Passive shadow**: record Student-selected windows online, no attack
6. **8-episode active smoke**: Black Bowl State5 × 2 seeds × 4 conditions
7. **Formal 40/48-episode attack matrix**: State5+State7 × 5-6 seeds × 4 conditions

## 6. Decisions Summary

```
Resume original 864 grid:            NO — formally closed
Continue V2C rescue:                 NO
Preserve grid diagnostic results:    YES — for architecture analysis only
Corrected exact-W32 confirmation:    YES — pre-registered config
Complete remaining 11 splits:        YES — using sidecar launcher
Full-FIT seed 42:                    YES — canonical after 12-split confirmation
Seeds 123/456 stability:             PARALLEL — evidence only, not for selection
CAL for FSM/threshold:               YES — after Full-FIT
Skip CAL, start attack directly:     NO
Passive shadow:                      After CAL/CHECK
8-episode smoke:                     After shadow passes
Formal attack matrix:                After smoke passes
```

## 7. Server-Side State

- **Sidecar worktree**: `/mnt/sdc/dty_user/openvla_attack_v2b_recommended`
- **Existing output**: `OFFICIAL_V3_FACTORIZED_STUDENT_V2_RECOMMENDED_EXACT_W32_V1_20260721` (1/12 complete)
- **Splits**: reuse existing `OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721`
- **Launcher**: `scripts/detector_v5/launch_factorized_v2_recommended_canary.py`
- **Orphan staging dirs**: ~560 directories from aborted waves — do NOT delete without quarantine manifest

## 8. Launch Command (for remaining 11 splits)

```bash
cd /mnt/sdc/dty_user/openvla_attack
PYTHONPATH=/mnt/sdc/dty_user/openvla_attack/src \
/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python \
scripts/detector_v5/launch_factorized_v2_recommended_canary.py \
  --output-base /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_RECOMMENDED_EXACT_W32_V1_20260721 \
  --reference-authorization-root /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_AUTHORIZATION_V1_20260721 \
  --gpus 0 1 2 3 4 5 6 7 \
  --workers 8 \
  --timeout 7200
```

- 1 worker/GPU, 8 parallel max
- No pkill, no process killing, no restart behavior
- Each job: train → predict → evaluate → audit
- Already-complete jobs SKIP (audit PASS check)
- `formal_selection_eligible = false` on all artifacts
