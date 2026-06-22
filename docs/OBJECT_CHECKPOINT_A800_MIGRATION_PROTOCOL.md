# Object Checkpoint A800 Migration Protocol

## Goal

Verify that the original Object-suite ProprioNoStep detector (C16 freeze, tag `freeze/layer123-poc-20260618`) transfers correctly to A800 hardware, then isolate each variable change.

## Order

```
F0: Freeze recovered artifact contract
M0-R: Offline replay on frozen Object traces
M0-E: E2E POC reproduction (Butter s0/s2)
Runner parity canary
M1: Runtime profile swap (BF16-Eager → Flash2 → FP32)
M2: Preprocessing swap (PIL → upstream TF/JPEG)
M3: Suite swap (Object → Spatial zero-shot)
```

## Frozen Detector Contract

- Checkpoint SHA256: `66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628`
- Dataset SHA256: `f942f4b0856d3449`
- Architecture: 25→64→64→{9,1,1,1}, 6604 params
- tau_corridor=0.3, tau_release=0.3, guard=5, K=10

## C16 Baseline

- Layer 1/2: coverage=0.873, K10=0.974, false-early=0.025, post-release=0.000
- Layer 3 E2E: 2/2 detector-triggered VIS POC PASS

## Stop Rules

- Any stage fails → STOP, no automatic progression
- No threshold recalibration
- No checkpoint modification
- No retraining
