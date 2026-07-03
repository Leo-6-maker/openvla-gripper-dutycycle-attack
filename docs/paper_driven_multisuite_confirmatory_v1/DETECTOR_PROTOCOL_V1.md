# Detector Protocol V1

Status: PLANNING_ONLY

This phase runs detector-only experiments. It does not run attacks.

## Main Model Rule

Freeze the actual production detector implementation before training. If the
runtime implementation is the 13D ProprioNoStep causal TCN, it is the main
model. If the runtime implementation is the 25D MLP, it is the main model and
the TCN is an ablation.

The freeze must record feature order, history length, hidden size, layer count,
heads, normalization, threshold, and guard duration.

## Baselines

- fixed normalized-time;
- close-onset heuristic;
- rule-based proprio;
- logistic regression or shallow MLP;
- one other lightweight temporal model;
- privileged teacher upper bound for offline evaluation only.

## Training Regimes

```text
Object-only x 3 seeds
Pooled x 3 seeds
LOSO x 4 held-out suites x 3 seeds
Object leave-one-task-out x 10 folds x 3 seeds
```

Total: 48 lightweight detector runs, no OpenVLA rollout.

## Threshold Rule

Thresholds are selected on validation only, for example: maximize event F1
subject to false-trigger rate <= 0.20. Object-only zero-shot uses no
suite-specific threshold.

## Gate C

```text
Object held-out event recall >= 0.70
Object held-out +/-10 recall >= 0.65
Object held-out false-trigger rate <= 0.20
Object held-out median timing error <= 10 steps
Cross-suite macro event recall >= 0.60
Each eligible suite recall >= 0.50
Cross-suite false-trigger rate <= 0.30
```

A suite that fails Gate C must not start formal detector-triggered attack. It
may only run separately authorized teacher-timing mechanism smoke.
