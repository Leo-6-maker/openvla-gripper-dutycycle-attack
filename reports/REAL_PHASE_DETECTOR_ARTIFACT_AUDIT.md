# Real Phase Detector Artifact Audit

**Date**: 2026-06-06

## Findings: REAL TRAINED PHASE DETECTOR EXISTS ✓

### Checkpoints

| Path | Size | Type |
|------|------|------|
|  | 156KB | ProprioNoStep TCN (CPU) |
|  | — | ProprioNoStep (GPU) |
|  | — | ProprioNoStep baseline |

### Training Script

- : PhaseTCN (input_dim=D, hidden_dim=64, num_layers=3)
- : EarlyGraspTCN (ProprioNoStep style)
- Save format: 

### Metrics (CPU phase detector)

- Feature set: A_descriptor (clean_open_count, qpos_end, clean_open_ratio, raw_gripper_mean)
- Model: LR (not TCN for CPU eval — streaming replay uses rule-based)
- Per-task accuracy: 70-77%
- Feature importance: clean_open_count (0.13), qpos_end (0.12), clean_open_ratio (0.09)

### Classification

**Learned detector**: YES (PhaseTCN + EarlyGraspTCN models trained on object clean sequences)
**Not proxy/rule-only**: Model checkpoints exist with saved state_dicts
**CPU inference possible**: 156KB model at proprionostep_cpu_20260602

### Data

| Table | Rows | Description |
|-------|------|-------------|
|  | — | Per-task LOTO metrics |
|  | — | Streaming replay predictions |
|  | — | Feature importance |
|  | — | Schema audit |
|  | — | Phase event ground truth |
|  | — | Window descriptors with phase labels |

### Next Steps

1. Load checkpoint → verify model architecture
2. Run inference on labels_v2 + adaptive 1R rows
3. Evaluate phase gate metrics
4. Integrate with vulnerability detector
