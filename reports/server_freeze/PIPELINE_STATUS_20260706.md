# CLEAN2000 Detector Pipeline Status — 2026-07-06 20:45 CST

## Pipeline Summary

```
D2A→D2B→D2C4→D2D→D2E→D2F→D4A→D4→D5  PASS
D6A  HOLD (sparse replay: 0 emit)
D6B  RUNNING (dense temporal replay, 2h25m elapsed)
```

## D2-D2F Dataset Freeze

| Gate | Status | Key Metric |
|---|---|---|
| D2A manifest audit | PASS | 3806 rows |
| D2B binding audit | PASS | 3717 ready + 89 excluded |
| D2C4 feature probe | PASS | 3717/3717 pass (184 imputed) |
| D2D materialization | PASS | 3717 rows, 25 features |
| D2E split audit | PASS | 0 leakage, train/val/test=2496/628/593 |
| D2F freeze | PASS | SHA256: 9b9a3f9f... |

## D4 CPU Training

| Metric | Train | Val | Test |
|---|---|---|---|
| corridor_f1 | 0.980 | 0.986 | 0.950 |
| phase_acc | 0.969 | 0.971 | 0.926 |
| event_role_acc | 0.785 | 0.761 | 0.722 |
| release_f1 | 0 | 0 | 0 |

Per-suite test: libero_10=0.970, libero_goal=0.881, libero_object=0.889, libero_spatial=0.718

## D5 Post-Training Audit

- Status: PASS
- Hard violations: 0
- Warning: RELEASE_HEAD_NO_POSITIVE_LABELS
- Imputed (184) corridor_f1: 0.997 vs non-imputed: 0.975 (gap=0.022)
- Phase head collapsed: only predicts stable_carry + abstain_unsupported (2/9 classes)

## D6A Sparse FSM Replay

- Status: HOLD
- primary_recall=0.0, tp_emit=0, fn_no_emit=1303, fp_emit=0
- Root cause: sparse label rows (many groups have 1 row) cannot satisfy FSM guard-delayed emission (IDLE→ARMED→guard→EMITTED)
- NOT a training failure — a replay granularity limitation

## D6B Dense Temporal Replay

- Status: RUNNING (PID 2611470, CPU 87%, ~521MB, 2h25m elapsed)
- 1911 groups × full step_telemetry.csv dense streams
- Estimated completion: ~30 min (within 2026-07-06 21:15 CST)

## Key Caveats

1. Phase head collapsed to 2/9 classes — FSM arm gate (requires stable_carry + corridor + primary_ok) may work but cannot distinguish phases
2. Release head has zero positive labels — cannot serve as safety gate
3. 184 imputed rows (all libero_object VALID_PRIMARY) require separate audit
4. libero_spatial is the weak suite (test corridor_acc=0.718)

## Decision Points After D6B

| D6B Result | Next Step |
|---|---|
| Dense emit > 0 | D6C threshold tuning or proceed to artifact replay |
| Dense emit = 0 | D4B retrain: binary runtime gate, phase head simplification, release head fix |
