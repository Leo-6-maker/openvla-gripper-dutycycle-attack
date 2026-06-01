# VIS Code and Threat Model Audit

**Date**: 2026-05-31 | **Branch**: `exp/crosssuite-proprio-vis-night-20260531`

## 1. Is VIS Implemented?

**YES** — fully implemented in `src/gripper_attack/attack_adapter.py`.

### Implementation Details

| Component | File | Status |
|-----------|------|--------|
| `TokenPrefixPGDAttacker` | `src/gripper_attack/attack_adapter.py:62` | IMPLEMENTED |
| `OpenVLAVisualAttacker` (factory) | `src/gripper_attack/attack_adapter.py:464` | IMPLEMENTED |
| Visual perturbation tracking | `src/gripper_attack/types.py:51-52` | IMPLEMENTED |
| Integration into production runner | `scripts/run_official_eval_artifact_rich.py` | **NOT INTEGRATED** |

### Threat Model

`TokenPrefixPGDAttacker` implements white-box PGD on OpenVLA:

1. **Target**: Action token prefix cross-entropy loss
2. **Gradient**: Backprop through OpenVLA vision encoder → action head
3. **Perturbation**: Pixel-space PGD on input image, bounded by linf norm
4. **Gripper targeting**: Supports `force_gripper_open` — maximizes gripper-open action token probability while constraining arm drift
5. **Controls**: `untargeted_arm_only` — perturbs arm action only (for comparison)
6. **Method tags**: `token_prefix_pgd_pixel_values_gripper_only`, `token_prefix_pgd_pixel_values_untargeted_clean_ce`

### Attack Configuration Parameters

- `eps`: linf norm bound (e.g., 4/255)
- `alpha`: step size (e.g., 1/255)
- `n_iters`: PGD iterations (e.g., 5, 10, 20)
- `force_gripper_open`: Boolean — target gripper-open dimension
- `arm_penalty_weight`: Weight for arm-drift penalty

## 2. Is VIS Integrated into Production Runner?

**NO.** `run_official_eval_artifact_rich.py` only has command-layer attacks:
- `sustained_command_open_proxy` — gripper action override (production)
- `gripper_inversion_proxy` — command-layer inversion + noise
- `oracle_open` — open gripper unconditionally
- `random_control` — random gripper toggle

The runner explicitly notes: "This is NOT visual PGD... True VIS PGD requires OpenVLAVisualAttacker."

## 3. Integration Path

To integrate VIS into the production runner:
1. Import `OpenVLAVisualAttacker` from `src/gripper_attack/attack_adapter.py`
2. Add `attack_condition: "vis_gripper_targeted"` to argparse choices
3. Wire VIS attacker into the policy step loop after `get_model_action()`
4. Apply perturbation to input image before action inference
5. Track perturbation norms and action deltas

## 4. Gradient Smoke Gate

Before integration, verify:
- OpenVLA backprop works on available GPUs (needs ~16GB for 7B model)
- Gripper action changes in intended direction
- Arm drift is controllable
- PGD runtime is feasible (estimated 2-5s per step with 10 iterations)

**GPU requirement**: Need GPU1,3 or GPU2,6 pair for OpenVLA + backprop.
GPU7 (11GB) alone insufficient. Will test when rollout GPUs free up.

## 5. Production Status

- Command-layer sus30 proxy: **production**
- VIS inference-time perturbation: **implemented, not integrated, not validated**
- VIS is a separate track — not a replacement for command-layer proxy
