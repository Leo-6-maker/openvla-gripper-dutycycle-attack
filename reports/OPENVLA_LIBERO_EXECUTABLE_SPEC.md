# OpenVLA-LIBERO Executable Spec

**Version**: `openvla_libero_exec_spec_v2_official_boundary_20260607`
**Executable module**: `src/gripper_attack/openvla_libero_exec_spec.py`
**Status**: frozen standard for Stage-B semantic repair

## Authority

This spec is derived from the official OpenVLA LIBERO evaluation path:

- OpenVLA README: `predict_action(...)` returns a 7-DoF action.
- `experiments/robot/libero/run_libero_eval.py`: model action is executed as
  `normalize_gripper_action(action, binarize=True)`, then
  `invert_gripper_action(action)`, then `env.step(action.tolist())`.
- `experiments/robot/robot_utils.py`: raw gripper is normalized from `[0, 1]`
  to `[-1, +1]`, then inverted because LIBERO env action convention is
  `-1=open`, `+1=close`.
- `experiments/robot/libero/libero_utils.py`: `obs["agentview_image"]` is
  rotated by 180 degrees before OpenVLA preprocessing.
- `run_libero_eval.py`: robot state includes `obs["robot0_gripper_qpos"]`.

All current open/close classification must import the executable module or its
backward-compatible wrapper `src/gripper_attack/gripper_semantics.py`.

## Official Action Chain

```text
raw OpenVLA action
-> normalize_gripper_action(action, binarize=True)
-> invert_gripper_action(action)
-> env.step(action.tolist())
```

The raw/decoded gripper dimension follows OpenVLA dataloader convention:

```text
raw_gripper = 0.0 -> close
raw_gripper = 1.0 -> open
```

After official normalize + invert, LIBERO receives:

```text
raw_gripper <  0.5 -> env_action_6 = +1.0 -> physical CLOSE
raw_gripper >  0.5 -> env_action_6 = -1.0 -> physical OPEN
raw_gripper == 0.5 -> env_action_6 = 0.0 -> BOUNDARY / NEUTRAL
```

Boundary rule: official OpenVLA uses `np.sign` during binarization, so
`raw_gripper == 0.5` stays zero after normalization and is not OPEN/CLOSE.

## Truth Table

| raw_gripper | normalized | binarized | env_action_6 | physical |
|---:|---:|---:|---:|---|
| 0.000 | -1.000 | -1.0 | +1.0 | CLOSE |
| 0.499 | -0.002 | -1.0 | +1.0 | CLOSE |
| 0.500 |  0.000 |  0.0 |  0.0 | BOUNDARY |
| 0.996 | +0.992 | +1.0 | -1.0 | OPEN |
| 1.000 | +1.000 | +1.0 | -1.0 | OPEN |

## Canonical Rules

Raw/decoded action space:

```text
OPEN  = raw_gripper >  0.5
CLOSE = raw_gripper <  0.5
BOUNDARY = raw_gripper == 0.5
```

LIBERO env action space:

```text
OPEN  = env_action_6 < -0.5
CLOSE = env_action_6 > +0.5
```

Token regions:

```text
open_token_ids  = {token_id | decoded raw_gripper >  0.5}
close_token_ids = {token_id | decoded raw_gripper <  0.5}
boundary_token_ids = {token_id | decoded raw_gripper == 0.5}
```

Saturation tokens such as `31744` / `31745` must be classified by decoded raw
gripper and post-transform env sign, not by old comments.

## Required Runtime Standards

Prompt:

```text
In: What action should the robot take to {instruction}?
Out:
```

Unnormalization key:

```text
libero_object
```

Image preprocessing:

```text
obs["agentview_image"][::-1, ::-1]
```

Official Octo-style resize remains a runner responsibility. Any runner that
skips the official rotation/resize path must mark itself `legacy_preprocess` or
`nonstandard_preprocess`.

Qpos source:

```text
obs["robot0_gripper_qpos"]
```

Do not use `sim.data.qpos[-2:]` as gripper qpos in new labels. For physical
response, use the vector or `abs(q0)+abs(q1)` and compute shifted response:
action at step `s` affects qpos at step `s+1`.

## Executable API

```python
from gripper_attack.openvla_libero_exec_spec import (
    OPEN_THRESHOLD_RAW,
    RAW_GRIPPER_CLOSE_VALUE,
    RAW_GRIPPER_OPEN_VALUE,
    ENV_GRIPPER_OPEN_VALUE,
    ENV_GRIPPER_CLOSE_VALUE,
    OPENVLA_LIBERO_EXEC_SPEC_VERSION,
    OFFICIAL_PROMPT_STYLE,
    OFFICIAL_UNNORM_KEY_LIBERO_OBJECT,
    OFFICIAL_QPOS_SOURCE,
    OFFICIAL_IMAGE_PREPROCESSING,
    official_prompt,
    normalize_gripper_raw,
    raw_gripper_to_env_gripper,
    decoded_action_to_env_gripper,
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_is_boundary,
    env_gripper_is_open,
    env_gripper_is_close,
    classify_raw_gripper,
    classify_env_gripper,
    open_token_ids_from_decoded_action,
    close_token_ids_from_decoded_action,
    boundary_token_ids_from_decoded_action,
    validate_open_close_token_sets,
    get_libero_image_official,
)
```

## Invalid Old Conventions

These conventions are invalid for current Stage-B evidence:

```text
raw_gripper < 0.5 means OPEN
decoded_action < 0.5 means OPEN
env_action_6 > 0.5 means OPEN
env_gripper > 0 means OPEN
oracle_open writes env_action_6 = +1.0
qpos = sim.data.qpos[-2:] as trusted gripper qpos
```

## Quarantine Rule

Any Stage-B labels or physical-response claims generated before this executable
spec passed audit must be treated as:

```text
QUARANTINED_OPEN_SEMANTICS_INVERTED_OR_UNVERIFIED
```

The 44-row patched rerun is retained only as inverted-objective diagnostic /
bug-discovery evidence until a corrected-objective smoke and rerun are produced
under this spec.
