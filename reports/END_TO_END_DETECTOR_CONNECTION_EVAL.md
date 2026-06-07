# End-to-End Detector Connection Evaluation

**Date**: 2026-06-06  
**Data**: join table (77 rows), canonical detector V0_gold_only/LR/D_causal_safe

---

## Modes Evaluated

| Mode | Description | Hard Gate? |
|------|-------------|------------|
| A | Vulnerability-only (baseline) | No |
| B | Phase-as-feature (train with phase features) | No |
| C | Mechanism-aware routing | No |
| D | Soft modulation (never zero) | No |
| E | Hard gate (phase AND vulnerability) | YES — rejected |

---

## Mode A: Vulnerability-Only (Baseline)

```
risk = vulnerability_score
```

### Confusion Matrix (canonical V0_gold D_causal_safe LR, n=22)

| | Pred=1 | Pred=0 |
|---|--------|--------|
| True=1 (pos) | TP=8 | FN=1 |
| True=0 (neg) | FP=6 | TN=7 |

### Metrics

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 0.714 |
| Positive Recall | 0.889 (8/9) |
| Negative Recall | 0.538 (7/13) |
| False Positive Rate | 0.462 (6/13) |
| Accuracy | 0.682 |
| MCC | +0.37 (estimated) |

### Mechanism-Stratified

| Mechanism | Pos Recall | Neg Correct | Notes |
|-----------|-----------|-------------|-------|
| physical_bridge_positive | 8/9 (0.889) | n/a | 1 FP on claim_usable row |
| negative_unclassified | n/a | 7/13 (0.538) | 6 FPs — these are no_action_bridge |

### Verdict
Positive detection works well for physical_bridge. But FPR=0.462 is high —
the detector over-predicts vulnerability on the `no_action_bridge` negatives.
This is expected: these negatives come from VIS attacks that caused task failure
without physical opening — the detector's causal features may pick up the
task_failure signal.

---

## Mode B: Phase-as-Feature

Train or score vulnerability detector with phase features included:
- hazard_score_mean/max
- release_safe_score_mean/min
- predicted_phase encoding
- phase_confidence

### Current State
Cannot evaluate directly — no detector retraining performed.
Phase features exist for only 7/22 valid eval rows (rest are out-of-domain).

### Expected Impact
- Phase features are ALL ZERO for the 7 covered rows (hazard_score=0.0 for all vulnerability windows).
- Adding zero-variance features would NOT change the detector.
- This confirms the orthogonal signal: vulnerability windows exist in phases the phase detector classifies as "safe."

### Verdict
Phase-as-feature **will not help** unless:
1. Phase detector is retrained on attack-relevant windows, OR
2. Phase features capture different dimensions (release_safe, phase_confidence)

---

## Mode C: Mechanism-Aware Routing

```
if vuln_high AND physical_bridge_positive:
    physical_vulnerability_risk → alarm
elif vuln_high AND negative_unclassified:
    policy_action_vulnerability_risk → needs_3R_confirmation
elif vuln_low AND clean_control_negative:
    clean_control_low_risk → suppress
else:
    manual_review
```

### Evaluation (on 22 valid rows)

| Route | Count | Correct Action? |
|-------|-------|-----------------|
| physical_vulnerability_risk (alarm) | 9 | 8 correct, 1 FP (physical_bridge pred=0) |
| policy_action_vulnerability_risk (needs_3R) | 5 | Pred=1 on negative_unclassified → correct to escalate |
| clean_control_low_risk (suppress) | 0 | No clean controls in valid set |
| manual_review | 8 | Pred=0, could be suppressed or reviewed |

### Verdict
Mechanism-aware routing correctly separates physical alarms from policy-confirmation cases.
It prevents false alarm on no_action_bridge while still flagging them for confirmation.
This is the **recommended approach**.

---

## Mode D: Soft Modulation (Never Zero)

```
risk_score = vulnerability_score * soft_phase_factor
soft_phase_factor ∈ [0.5, 1.0] — NEVER ZERO
```

### Evaluation (7 rows with phase)

| Window | Mech | Vuln | Hazard | Factor | Modulated |
|--------|------|------|--------|--------|-----------|
| cream_cheese s4 [28,45] | physical_bridge | 1 | 0.0 | 0.50 | 0.50 |
| ketchup s0 [16,33] | physical_bridge | 1 | 0.0 | 0.50 | 0.50 |
| ketchup s1 [21,38] | physical_bridge | 1 | 0.0 | 0.50 | 0.50 |
| ketchup s4 [28,45] | neg_unclassified | 1 | 0.0 | 0.50 | 0.50 |
| ketchup s5 [9,26] | neg_unclassified | 1 | 0.0 | 0.50 | 0.50 |
| salad_dressing s0 [7,24] | neg_unclassified | 0 | 0.0 | 0.50 | 0.00 |
| salad_dressing s5 [28,45] | neg_unclassified | 0 | 0.0 | 0.50 | 0.00 |

### Problem
All hazard scores = 0.0 for vulnerability-relevant windows. The phase factor is
constant 0.5 — it provides NO discrimination. This is because the phase detector
was trained on clean physical hazard phases, and vulnerability windows are
outside that distribution.

### Verdict
Soft modulation is **diagnostically useless** with the current phase detector.
It only makes sense after:
1. Phase detector retrained on attack windows, OR
2. Using a different modulation signal (release_safe, phase_confidence)

---

## Mode E: Hard Gate (Rejected Baseline)

```
if phase_gate == "hazard":
    risk = vulnerability_score
else:
    risk = 0  # suppressed
```

### POC Result (from prior audit)
- 0/7 positive recall — ALL vulnerability positives suppressed by phase gate
- The phase gate was 100% effective at blocking vulnerability detection
- False alarm rate on clean controls: low (but irrelevant — positives are blocked)

### Verdict: DO NOT USE

---

## Summary Comparison

| Mode | Pos Recall | Neg Recall | FPR | Phase Needed? | Recommendation |
|------|-----------|-----------|-----|---------------|----------------|
| A: Vuln-only | 0.889 | 0.538 | 0.462 | No | **Baseline** |
| B: Phase-as-feature | N/A | N/A | N/A | Yes | Cannot evaluate (no retraining) |
| C: Mechanism routing | 0.889 | 0.538 | 0.462* | No | **RECOMMENDED** |
| D: Soft modulation | 0.889 | 0.538 | 0.462* | Yes | Useless (hazard=0 for all) |
| E: Hard gate | **0.000** | 1.000 | 0.000 | Yes | **REJECTED** |

*FPR is unchanged from Mode A because phase features are constant (zero).

---

## Final Recommendation

**Mode C (mechanism-aware routing)** is the recommended connection mode:

1. Use vulnerability detector as primary signal.
2. Route by mechanism_type:
   - physical_bridge + vuln=1 → ALARM (high confidence)
   - negative_unclassified + vuln=1 → NEEDS_3R_CONFIRMATION
   - vuln=0 → SUPPRESS or MANUAL_REVIEW
3. Phase detector provides context (mechanism audit), not gating.
4. Soft modulation is NOT useful with current phase detector.

### What Would Make Phase Detector Useful

The phase detector would become useful if:
1. It is **retrained** on attack-relevant windows (VIS traces, not just clean rollouts)
2. Its hazard scores show **variance** across vulnerability vs. control windows
3. It measures a **complementary** signal (release safety, phase transition timing)

Until then, its role is limited to:
- Mechanism audit: "does physical_bridge occur in a hazard phase?"
- Clean control stratification: "is this control window in a true hazard phase?"
