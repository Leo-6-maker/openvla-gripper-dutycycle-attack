# D8A — Detector History and Negative Results Ledger

Purpose: prevent repeating experiments. Record what was tried, what failed, and the ceiling each approach hit.

## Generation 0: Fixed-Window Controlled Probe

- **What**: Hardcoded State7 75-84, State5 78-87, pregrasp 35-45 windows.
- **Result**: Proved mechanism (gripper-targeted VIS/command-open causes duty-cycle failure in known contact-critical windows).
- **Limitation**: Windows manually defined per state_id. Not an automatic detector.

## Generation 1: Clean-Only Generic Autowindow

- **What**: Clean trajectory autowindow detecting grasp, lift, carry, near-target, EEF descent, release intent phase cues. Priority-based pre-place window selection.
- **Guardrails**: No attack outcome, no matched-run result, low-confidence windows diagnostic only, detector_config_hash per output.
- **Result**: Eliminated hardcoded state_id→window table.
- **Limitation**: Offline clean-trajectory only. Not online streaming.

## Generation 2: Privileged Teacher (V2)

- **What**: Clean simulation privileged state (object/target/EEF/gripper), fail-closed. Phase order: approach→grasp_close→stable_grasp→first_lift→stable_carry→pre_place_unsupported→release_safe.
- **Result**: Failure-critical phases = stable_carry + pre_place_unsupported. SC5 anchor: earliest stable_carry_start + guard, K=10 corridor, no cross-release_safe.
- **Key insight**: Windows defined by clean privileged phase annotation, not attack outcome.

## Generation 3: Milestone 2C Proprio Causal Student

- **What**: Proprio-only causal student predicting teacher phase/hazard/release-safe from low-signal deployment features. Input: gripper, qpos, width, EEF, action history, normalized_step. Forbidden: object pose, target pose, teacher window, task_id, state_id, attack outcome.
- **Result**: Hazard F1 0.6743, AUROC 0.9874, coverage 0.8706, false early 0.0116.
- **Limitation**: Offline replay only.

## Generation 3.5: Anti-Timing Ablation (2C.1)

- **What**: Compared full proprio vs rule proxy vs time-only vs no-normalized-step vs label-shuffle.
- **Result**: 
  - full_proprio coverage = 0.8706
  - time_only coverage = 0.4265
  - no_normalized_step coverage = 0.7529
  - label_shuffle = 0 (collapse)
- **Key insight**: Gripper/EEF/action history carries genuine phase signal. Not purely learning "which step to trigger."

## Generation 4: SC5 Canonical 25D MLP + FSM

- **What**: 25D causal features, phase/corridor/release heads, IDLE→ARMED→EMITTED FSM. Per-step streaming, fail-closed.
- **Features**: gripper command/qpos/opening proxy, EEF pose/velocity, action xyz/gripper, recent streak, close onset, time since close, deltas, variance.
- **Result**: C16 closeout Layer 1/2 replay 6/6 pass: coverage 0.873, false early 0.025, post-release 0, median abs error 2.7, K10 containment 0.974, no-corridor abstain 0.954. Layer 3 POC: Butter s0/s2 clean success, VIS_SC5 fail, RAND_T10 success.

## Generation 5: Multi-Suite / LIBERO-10 Event-Role

- **What**: Extended to multi-suite with event-role/abstain heads. Added roles: primary_attackable, auxiliary_manipulation, distractor_or_setup, unsupported_or_abstain.
- **Design**: Clean online proprio/action only, no timestep, no privileged state, no attack outcome, one-shot primary emission FSM by default.
- **Result**: Runtime only arms/emits on stable_carry + corridor + primary_ok.

## Generation 6: C2e0–C2e3 25D Baseline / GRU (Current)

- **What**: Six 25D temporal detectors trained on C2e1 dataset:
  1. Pooling MLP
  2. GRU (W=16, H=128)
  3. Causal TCN
  4. FP-aware GRU
  5. Multi-window GRU
  6. Ensemble (GRU+TCN)
- **C2e3 Frozen Baseline**: GRU W=16 H=128, recall=75.6%, FP=31.8%, L10 recall=45.6%, τ_emit=0.33, τ_suppress=0.67.
- **Result**: All 25D models converge to FP floor 31-39%. GRU/TCN ensemble, FP-aware loss, multi-window GRU do not resolve L10 FP limitation.
- **Blocked**: C2f observation-enhanced detectors (no RGB/language in current clean rollout artifacts).

## Ceiling Confirmed

```
All 25D proprio/action-only detectors converge to FP ~31-39%.
L10 recall <55% across all variants.
No architectural change within 25D-only breaks this ceiling.
```

## What NOT to Repeat

- Hidden size tuning (128→256)
- TCN dilation grid search
- Per-suite / per-task threshold calibration
- FP-aware loss v2
- Multi-window v2
- GRU+TCN score fusion
- Any 25D-only model variant

## Forward Path

1. Event-level resettable multi-event FSM (D8C)
2. Observation/language-enhanced C2f detector (requires new RGB rollout collection)
3. L10 no-emit taxonomy analysis (D8B)
4. Detector contract regression tests (D8E)
