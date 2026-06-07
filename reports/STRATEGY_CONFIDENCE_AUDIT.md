# Strategy Confidence Audit

**Date**: 2026-06-06
**Purpose**: Falsification loop — assess confidence in every current strategy claim

---

## Claim A: ProprioNoStep Hard Gate Rejected

**Claim**: Phase detector must NOT be used as hard pre-vulnerability gate.

**Evidence**:
- POC: 0/7 positive recall when phase gate applied before vulnerability detector
- Root cause: phase detector trained on clean hazard phases; vulnerability windows are orthogonal
- All hazard_score=0 for vulnerability windows in covered subset

**Confidence**: **SUPPORTED**

**Action**: Phase detector is permanently restricted to context/audit role. No further hard gate experiments needed.

---

## Claim B: Vulnerability-First Detector Is Primary

**Claim**: Vulnerability detector (LR on features) is the primary risk scorer.

**Evidence**:
- V0_gold/LR/D_causal_safe: posRecall=0.889, BalAcc=0.714 on 22 rows
- BUT: D_causal_safe features use OFFLINE-ONLY columns (qpos_delta, vis_open_count, etc.)
- This detector is an attack outcome classifier, NOT a deployable vulnerability predictor

**Confidence**: **PARTIALLY_SUPPORTED — BLOCKED on feature leakage**

**Blockers**:
1. D_causal_safe features are offline-only → BLOCKED_LEAKAGE_RISK
2. Only A_task_key_only and B_phase_bin_only are online-safe
3. Online-safe detector performance is unknown (likely worse than 0.714)
4. Must rebuild training with online-safe features before claiming readiness

**Action**: Before calling vulnerability-first "ready," rebuild with online-safe features (task_key + clean rollout stats + phase context) and measure the online-safe BalAcc.

---

## Claim C: Mechanism-Aware Offline Routing Useful

**Claim**: After VIS attack, mechanism_type can route results (alarm vs confirmation vs suppress).

**Evidence**:
- 9/9 gold positives are all physical_bridge_positive
- Mechanism routing correctly separates physical alarms (8/8 correct) from policy escalations
- Offline routing uses mechanism_type which is a label/oracle

**Confidence**: **SUPPORTED (offline only)**

**Note**: This is offline routing — it uses VIS outcomes to classify mechanism. It is valid for label building and experiment management, but NOT for online deployment.

**Action**: Keep mechanism-aware routing for offline experiment management. Do not claim it works for online deployment.

---

## Claim D: Mechanism-Aware Online Routing Ready

**Claim**: mechanism_type can be used for online routing at deployment.

**Evidence**:
- mechanism_type is an oracle label (Category C)
- It requires VIS trace analysis to determine
- No model exists to PREDICT mechanism_type from clean rollout only
- phase detector is NOT a mechanism predictor (orthogonal signal)

**Confidence**: **REJECTED**

**Reason**: mechanism_type is an oracle label. It is NOT available at deployment. No mechanism prediction model exists. Phase detector cannot substitute for it.

**Action**: If online mechanism routing is desired, train a mechanism predictor model on clean rollout features. Until then, online routing must use only vulnerability_score thresholds.

---

## Claim E: 1R Promotion Ready

**Claim**: 1R VIS results can be promoted to silver labels for detector training.

**Evidence**:
- Calibration v1: 60% agreement (6/10), 4 disagreements (3R found OPEN where 1R did not)
- Calibration v1 had config mismatch (3R traces from batch1/3, 1R from calibration chain)
- Calibration v2: NOT YET RUN (10 candidates, matched 1R vs 3R)
- Current 32 1R screening results = `uncalibrated_1r_screening` — NOT usable for training

**Confidence**: **BLOCKED — waiting for calibration v2**

**Blockers**:
1. Calibration v2 not yet launched
2. Need >=80% agreement (8/10) to promote 1R
3. Current agreement is only 60% (6/10) with config-mismatched data

**Action**: Launch calibration v2 on GPU 0,1. If agreement >=80%, promote 1R to provisional_silver with sample_weight=0.5.

---

## Claim F: v3 Training Ready

**Claim**: Detector v3 training can begin.

**Evidence**:
| Gate | Status |
|------|--------|
| Config-matched calibration v2 PASS | NOT YET RUN |
| Confirmed hard/control negatives >=6 | 4 physical_task_neg + 0 confirmed controls |
| Clean controls audit PASS | NOT YET RUN (12 candidates need 3R) |
| No pending_negative_1r in train | 32 1R results = uncalibrated |
| No infra/manual/polluted in train | 8 polluted, 8 precheck_failed excluded |
| Phase detector NOT used as hard gate | Confirmed |
| Online-safe feature set defined | NOW DEFINED (F1 audit) |
| LOTO evaluation not worse than v2 | Not applicable (v2 uses leaked features) |
| Mechanism-pure labels | 9 physical_bridge (ready) + 0 no_action_bridge pos |

**Confidence**: **BLOCKED — 5 gates not yet passed**

**Blockers**:
1. Calibration v2 → must run (needs GPU)
2. Clean-control 3R confirmation → must run (needs GPU)
3. Online-safe feature set → must rebuild dataset
4. Labels must be mechanism-pure → partially done (P0 audit complete)
5. 1R promotion not ready → waiting for calibration v2

**Action**: DO NOT train v3. Complete P1+P2 first.

---

## Summary Matrix

| Claim | Confidence | Blocker |
|-------|-----------|---------|
| A. Hard gate rejected | **SUPPORTED** | — |
| B. Vuln-first primary | **PARTIALLY_SUPPORTED** | Feature leakage (D_causal_safe = offline-only) |
| C. Mechanism routing (offline) | **SUPPORTED** | Offline only; not for deployment |
| D. Mechanism routing (online) | **REJECTED** | mechanism_type is oracle, no predictor exists |
| E. 1R promotion ready | **BLOCKED** | Calibration v2 not run |
| F. v3 training ready | **BLOCKED** | 5 gates not passed |

---

## Biggest Falsification Finding

**The current "best" detector (V0_gold/LR/D_causal_safe, BalAcc=0.714) is trained on features that encode the attack outcome itself.** This is a fundamental experimental design error:

```
Problem:  f(qpos_opening_delta, vis_open_count, ...) → vulnerable?
Reality:  qpos_opening_delta and vis_open_count MEASURE the attack outcome.
          The model learns "if attack caused opening → vulnerable" which is tautological.
```

The online-safe detector can only use features available BEFORE the attack:
- Clean rollout statistics
- Phase context
- Task identity

The performance gap between offline-leaked and online-safe features IS the true
information content of the pre-attack vulnerability signal.

---

## Do NOT Claim

- "Detector works" — it uses leaked features
- "Phase detector can route" — it's orthogonal to vulnerability
- "1R results are usable" — calibration v2 not done
- "v3 can begin training" — 5 gates blocked
- "Pipeline is deployment-ready" — online features not yet used

## CAN Claim

- "9 gold positives are physical_bridge with strong evidence" — verified
- "Hard gate rejected with clear evidence" — verified
- "Mechanism taxonomy correctly classifies outcomes" — verified
- "Offline audit infrastructure is functional" — join table, scripts, reports all work
- "Online-safe feature whitelist is defined" — F1 audit complete
