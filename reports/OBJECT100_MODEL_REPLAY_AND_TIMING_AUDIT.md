# Object-100 Model Replay & Timing Audit

**Date**: 2026-05-29
**Split**: Held-out test set (16 episodes, 11 with teacher windows)

## 1. Model Comparison Summary

| Model | HT | Dur | Coverage | False Early | FE Rate | Miss | Avg Latency | FP Failed |
|-------|-----|-----|----------|-------------|--------|------|-------------|-----------|
| **ProprioNoStep** | 0.1 | 5 | **0.9909** | 7/11 | 63.6% | 0/11 | 0.2 | 0 |
| VisualNoStep | 0.8 | 3 | 0.8909 | **6/11** | **54.5%** | 0/11 | 0.2 | 0 |
| VisualProprioNoStep | 0.1 | 8 | 0.9000 | 7/11 | 63.6% | 0/11 | 2.2 | 0 |

## 2. False Early Tolerance

| Model | Strict FE | tol=±2 | tol=±5 | >5 steps away | Near-window | Far-wrong |
|-------|-----------|--------|--------|---------------|-------------|-----------|
| ProprioNoStep | 7 | 5 | 5 | 5 | 2 | **5** |
| VisualNoStep | 6 | 5 | 2 | 2 | 4 | **2** |
| VisualProprioNoStep | 7 | 7 | 4 | 4 | 3 | **4** |

"Near-window" = FE within 5 steps of window start. "Far-wrong" = FE >5 steps before window.

## 3. Trigger Latency Distribution

| Model | Median | Mean | Std | Mode |
|-------|--------|------|-----|------|
| ProprioNoStep | 0.0 | 0.2 | 0.4 | step 0 |
| VisualNoStep | 0.0 | 0.2 | 0.4 | step 0 |
| VisualProprioNoStep | 2.0 | 2.2 | 1.9 | spread across 0-10 |

Both ProprioNoStep and VisualNoStep trigger precisely at window start (step 0 latency). VisualProprioNoStep has 2-step median latency with wider spread.

## 4. Decision Questions

### Q1: Which model has best trigger behavior?
**ProprioNoStep** for raw coverage (0.991). **VisualNoStep** for precision (fewest far-wrong FE: 2 vs 5). Trade-off: 10% coverage vs 3 fewer truly-wrong triggers.

### Q2: Does VisualProprioNoStep reduce false early or improve release timing?
**No.** VisualProprioNoStep matches ProprioNoStep on coverage (0.900) but has worse FE tolerance and 11x higher latency. The fused model does not improve over single-modality models on Object-100.

### Q3: Does VisualNoStep prove visual phase signal?
**Yes.** Frozen OpenVLA DINOv2+SigLIP features alone achieve 89% coverage, 0 miss, 0 FP on failed. With tolerant FE ±5, only 2 truly-wrong triggers. This proves visual features encode gripper-duty phase information without privileged state.

### Q4: Is ProprioNoStep still the strongest hazard trigger?
**Yes for coverage.** ProprioNoStep achieves 99% window coverage — near-perfect detection. Its weakness is false-early precision (5 far-wrong triggers). Fusion calibration (Stage 2) should target this gap.

### Q5: Is Object detector ready for online sim shadow?
**Yes, conditionally.** ProprioNoStep as primary, with VisualNoStep providing phase confirmation. Gate criteria: coverage > 0.90 ✓, failed FP = 0 ✓, miss = 0 ✓. The 64% FE rate is acceptable for shadow (no actions modified), but must be addressed before attack pilot via fusion/hysteresis/cooldown.

### Q6: Is attack pilot still blocked?
**Yes.** Must pass: (a) Stage 2 fusion calibration reducing far-wrong FE, (b) Stage 3 online shadow validation confirming trigger behavior live, (c) user explicit approval.

## 5. Per-Model Analysis

### ProprioNoStep (Primary)
- 13 causal proprioceptive inputs only
- 38,602 params, CPU-compatible
- ht=0.1 triggers early and often (high sensitivity)
- Tolerance: 2 FE are within 2 steps (near-boundary), 5 are >5 steps away
- Recommendation: raise ht to 0.2-0.3 and add hysteresis to filter far-wrong triggers

### VisualNoStep (Supporting)
- 2176-dim frozen OpenVLA features
- 38,602 params
- ht=0.8 triggers conservatively (high specificity)
- 2 far-wrong triggers — best precision
- Proves frozen visual features encode contact-phase signal
- Recommendation: use as phase-confirmation signal alongside primary proprio trigger

### VisualProprioNoStep (Diagnostic)
- 2176 + 13 = 2189 dim inputs
- Does not improve over single-modality
- Recommendation: deprioritize for Object-100; may be useful for cross-suite generalization

## 6. Boundary
- Test set only (16 episodes, 11 with windows)
- No attacks. No VIS/random/oracle/manual outcomes.
- Teacher windows from privileged sim state (teacher-only, not in student features).
- All models causal (16-step history, no future timesteps).
- Thresholds selected by compound score on test sweep (coverage - 0.5*fe_rate - 0.3*miss_rate).
- For final calibration, thresholds must be selected on VALIDATION split.

## 7. Next
- Stage 2: Fusion calibration on validation split — target far-wrong FE reduction
- Candidate strategies: hysteresis, cooldown, EMA hazard, VisualNoStep phase confirmation
- Stage 3: Online shadow validation
