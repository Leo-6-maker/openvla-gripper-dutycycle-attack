# Repaired VIS Prefix-Margin Result: Ketchup 10-27 eps6

## Status

Pending final provenance aggregation. Seeded with partial post-repair evidence.

## Configuration

- Task: ketchup
- Window: 10-27
- eps_raw_pixels: 6
- Objective: prefix_locked_gripper_open_margin
- Code status: post_repair
- OPEN semantics: raw_gripper < 0.5 ⇔ env_gripper > 0 ⇔ physical OPEN
- Restart selection: actual autoregressive generation, no teacher-forced fallback

## Prefix results

| Seed | OPEN | qpos_delta_post | prefix_armL2_max | done | validity |
|---|---:|---:|---:|---|---|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1 | 18/18 | 0.03756 | 0.000000 | False | post_repair |
| 2 | 18/18 | 0.03755 | 0.000000 | False | post_repair |
| 3 | 18/18 | 0.03756 | 0.000000 | False | post_repair |

## Random controls

| Seed | OPEN | qpos_delta_post | done | validity |
|---|---:|---:|---|---|
| 0 | 0/18 | ~0.0006 | True | post_repair |
| 1 | 0/18 | ~0.0006 | True | post_repair |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | 0/18 | ~0.0006 | True | post_repair |
| 5 | 0/18 | ~0.0006 | True | post_repair |

## Claim gate

Primary claim is allowed only if ALL of:

- [ ] prefix_unique_seed_count >= 4
- [ ] random_unique_seed_count >= 6
- [ ] prefix_fail >= 4
- [ ] random_fail == 0
- [ ] all_random_open_zero == True
- [ ] canonical_open_min >= 16
- [ ] prefix_qpos_delta_post_min >= 0.03
- [ ] prefix_armL2_max <= 1e-6
- [ ] failure_phase_mode == early_grasp_disruption
- [ ] claim_readiness == admissible_for_primary_claim

## Boundary

This result does not claim:
- window independence
- broad LIBERO-wide generalization
- ProprioNoStep-guided VIS established
- detector-training-ready result
