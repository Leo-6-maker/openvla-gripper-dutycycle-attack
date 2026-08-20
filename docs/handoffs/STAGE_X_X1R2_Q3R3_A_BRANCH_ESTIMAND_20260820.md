# Stage X1R2 Q3R3-A handoff

Status: `STAGE_X1R2_Q3R3_BRANCH_ESTIMAND_FREEZE_PASS`

This is a static/CPU/offline result only. Q3R2-C remains the immutable
`OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED` HOLD.

## Provenance

- Live PR #135 source: `e353522ed323ee4289cb2a76060ed562ead7e4b1`
- Live PR #135 tree: `cde0d92ab42d110f12a0660856b50b03c76034a3`
- Audit script source commit: `6601beebfa2d20b34f5637a2546f1dc2bc15243e`
- Audit script source tree: `c5b7e93808054d06b885030682537be41b851d85`
- Frozen runtime authority: `85fa8e678ca599f21f5a69d180c7179f9ef99478` / `f6555a5d49dda45f29ef64ca8ae4b65b7b08d3f9`
- Durable audit root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_a_20260820`
- Audit report SHA256: `df71fce77552ee95663a1583565159dc0d14eee2ee70f78edf56d17794d36ec0`

## Visual-divergence characterization

The Q3R2 telemetry stores only raw-agentview and processor-pixel SHA256
values. The durable Q3R2 root contains no PNG/JPEG/NPY/NPZ/video frame
payloads. Therefore changed-pixel count, fraction, channel deltas, RMSE,
and processor-space L-infinity are recorded as `NOT_AVAILABLE`, not inferred
from hashes.

- `libero_goal`, `libero_goal/task_02/state_37`: first visual hash divergence
  at step 13; both raw-agentview and processor-pixel hashes changed. Action,
  EEF state, gripper qpos, and direct generated tokens matched at that step.
  Emit remained 63/63.
- `libero_spatial`, `libero_spatial/task_09/state_29`: first visual hash
  divergence at step 14 with the same isolated hash fields. The comparable
  clean repeats had first emit 68/65.
- `libero_object`, `libero_object/task_01/state_34`: sealed report says full
  trace and prefix exact.

The audit does not call the visual mismatch benign and does not localize the
mismatch to pre- versus post-preprocessing without raw bytes.

## Runtime surface

The official environment is `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
The surface seal records the mixed interpreter/package roots explicitly:
Python resolves through `/home/sz/miniconda3/envs/hallo`, while several
packages resolve from the official prefix. MuJoCo/robosuite versions and
initializer hashes, camera/preprocess settings, driver `550.90.12`, and
private-macro absence are sealed. `MUJOCO_GL` was unset in the static shell;
EGL/OpenGL vendor/version was not queried without creating a display context.
Future runtime must bind the selected backend or fail closed.

## Frozen prospective branch estimand

Each parent has exactly one realized `REFERENCE_CLEAN` trajectory. Student
timing is computed once. Branches reset to the same initial state and replay
only the sealed reference `env_action_7d` prefix for steps `[0,t_emit)`;
the branch point is immediately before the action at `t_emit`.

All first branch decisions use the same captured reference observation bytes.
No OpenVLA or Student inference is allowed before the branch point. Branch
state equality binds model/task identity, seed/dummy wait, wrapper step,
`qpos`, `qvel`, `act`, `ctrl`, time, mocap, task/object, and controller state.
Stable fields require exact serialization; float comparison is frozen at
`atol=1e-12`, `rtol=0`, with no later widening.

This is a new branch-controlled estimand, not a Q3R2-C PASS and not a claim
of bitwise-reproducible independent online rollouts.

## Boundary

- Q3R3-B: next authorized gate.
- Q3R3-C/D: not authorized yet.
- Scientific population and PGD: not authorized.
- `Eval160`: `UNREAD`.
- Protected evaluation: `UNREAD`.
- Q3R2-C exposed fixtures: never rerun or replaced.
