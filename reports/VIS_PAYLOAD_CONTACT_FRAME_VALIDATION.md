# VIS Payload Contact-Frame Validation

Date: 2026-06-01
Branch: `exp/vis-payload-upgrade-validation-20260601`

## Scope

- Environment: `openvla_official_libero_20260525` (server env name; this is the corrected official LIBERO env).
- OpenVLA diagnostics used physical GPU4/5 via `CUDA_VISIBLE_DEVICES=4,5`, `OPENVLA_ATTN_IMPLEMENTATION=eager`, and `model_gpu_device_id=-1` for visible-slot device map.
- Clean-only frame collection and no-rollout diagnostics only. No VIS rollout, detector-triggered rollout, sus30, or CrossSuite training.
- A prior interrupted attempt with `openvla_compat` is marked invalid under `/data/liuyu/outputs/vis_payload_contact_frame_collection_20260601/INVALID_ENV_PARTIAL_OUTPUT.md` and is not used.

## Clean-Only Contact Frame Collection

Official-env clean artifact-rich episodes were collected for state0:

| Task | Result | Target frame |
| --- | --- | --- |
| ketchup | success=True, 201 steps | `step_0098.png` available |
| tomato_sauce | success=True, 173 steps | `step_0134.png` available |
| cream_cheese | success=True, 155 steps | `step_0143.png` available |

Manifest: `tables/vis_payload_verified_contact_frames.csv`.

## No-Rollout Objective Sweep

Limited threshold sweep on verified contact frames:

- Objectives: `target_action_ce`, `gripper_open_region_ce`, `gripper_logit_margin_cw`.
- Budget: `eps=4/255`; observed Linf stayed `0.015625`, budget_ok=true.
- Steps: 4 and 20.
- Rows tested: 18.
- Token-flip rows: 6.
- Meaningful decoded gripper-action-change rows: 2.

Key positive rows:

| Frame | Objective | Steps | Clean -> adv gripper | Arm L2 |
| --- | --- | ---: | --- | ---: |
| ketchup_s0 | gripper_open_region_ce | 20 | 0.0 -> 0.996078 | 0.03335 |
| tomato_s0 | gripper_open_region_ce | 20 | 0.0 -> 0.996078 | 0.32056 |

Cream-cheese token flips stayed within already-open gripper action (`0.996078 -> 0.996078`), so they are token-only and not behavior-level payload success.

## Random Same-Linf Follow-Up

Follow-up arm-drift/random baseline for `gripper_open_region_ce`, `eps=4/255`, `steps=20`:

| Frame | Targeted result | Random same-Linf | Gate |
| --- | --- | --- | --- |
| ketchup_s0 | did not reproduce action flip on rerun | no flip | fail / nondeterministic |
| tomato_s0 | token/action flip, gripper delta 0.996078, arm L2 0.31263 | no flip | partial pass |

## Gate Decision

M1 PARTIAL PASS / rollout remains blocked.

Reason: one verified contact frame (`tomato_s0`) passed the no-rollout action-change + random-baseline check under strict budget, but ketchup did not reproduce its threshold-sweep flip under the same arm-drift/random harness, and arm drift for tomato is nontrivial. This is promising payload evidence, not strong rollout evidence.

## Valid Claims

- Official-env clean contact frames were collected and verified available for ketchup, tomato_sauce, and cream_cheese state0.
- `gripper_open_region_ce` at `eps=4/255`, `steps=20` can produce decoded gripper action change on at least one verified tomato_sauce contact frame while random same-Linf does not.
- Reproducibility is not yet sufficient for rollout; VIS rollout remains blocked.

## Next Step

Repeat the open-region payload gate on additional verified contact/pre-place frames and add a deterministic repeat count per frame before considering a tiny forced-window VIS micro proposal.
