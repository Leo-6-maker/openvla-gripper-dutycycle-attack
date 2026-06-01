# VIS Gradient Smoke Status

**Date**: 2026-05-31 | **Status**: BLOCKED — requires runner integration

## Attempt

Tried to run standalone VIS PGD smoke on GPU4,5 with OpenVLA 7B.

## Blockers

1. **OpenVLA requires KV-cache**: Model forward expects `past_key_values` during cached generation. Cannot run simple `model(pixel_values)` for clean forward.

2. **TokenPrefixPGDAttacker requires full runner context**:
   - Action tokenization (via `model.bin_centers` and `model.norm_stats`)
   - KV-cache setup for autoregressive decode
   - Specific image preprocessing pipeline
   - `unnorm_key` resolution

3. **Model loaded successfully** on GPU4,5 (distributed across both GPUs):
   - Vision backbone + layers 0-13 on GPU4 (cuda:0)
   - Layers 14-31 + lm_head on GPU5 (cuda:1)
   - ~9GB per GPU — backprop feasible

## What Works

- OpenVLA 7B loads on GPU4,5 pair
- Memory distributed: ~9GB each GPU
- PGD backprop should be feasible (needs ~12-15GB total for gradients)

## Path Forward

VIS gradient smoke should be done within the runner framework:
1. Integrate `OpenVLAVisualAttacker` into `run_official_eval_artifact_rich.py`
2. Add `attack_condition: "vis_gripper_targeted"` mode
3. Run single-episode smoke with ProprioNoStep trigger windows
4. Measure gripper action delta, arm drift, perturbation norm

## Code Location

- Attacker: `src/gripper_attack/attack_adapter.py:62` (`TokenPrefixPGDAttacker`)
- Factory: `src/gripper_attack/attack_adapter.py:464` (`OpenVLAVisualAttacker`)

## Recommendation

Defer VIS gradient smoke to a dedicated branch where the attacker is integrated into the runner. Current priority: cross-suite ProprioNoStep validation.
