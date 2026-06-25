# C3R Renderer Determinism Protocol

Preregistered: 2026-06-25

## Scope

```text
PARENT: libero_goal|4|1|0|CLEAN
CANDIDATE_MANIFEST_SHA256:
  3db7e7a8071c11710946d6944a6b9e26f8fc373ee4345e7467ad6d922aa5a76a
STATIC_SAME_PROCESS_RENDERS: 100
STATIC_FRESH_PROCESSES: 10
PREFIX_CALIBRATION_REPETITIONS: 20
PREFIX_HELDOUT_FRESH_PROCESSES: 10
ATTACKS: FORBIDDEN
```

This qualification uses only the frozen clean Goal C3 parent. No attack
outcome may influence a tolerance, parent, repetition, or acceptance rule.

## Render Stages

| Stage | Definition |
|---|---|
| R0 | direct `sim.render()` framebuffer readback |
| R1 | `agentview_image` returned by the environment observation path |
| R2 | project OpenVLA image transform output |
| R3 | exact uint8 RGB array passed to the processor |
| R4 | processor `pixel_values` tensor |

For each stage the runner records shape, dtype, SHA-256, differing-element
count, maximum absolute difference, mean absolute difference, first 32
differing indices, per-channel counts where applicable, and a difference
bounding box.

## Frozen Comparison Rules

The first same-process clean repetition supplies the R0 baseline. R1-R4 are
compared with the frozen clean reference observation captured while locating
the natural Student emit.

Policy and state acceptance is exact:

```text
7/7 generated action tokens
raw 7D action bytes
postprocessed environment 7D action bytes
gripper semantic
Student state and feature history
qpos, qvel, and flattened simulator state
```

The visual tolerance is derived only from the 20 calibration repetitions:

```text
bound(metric) = maximum observed calibration value
```

The preregistered metrics are:

```text
R1 differing-element count
R1 maximum absolute difference
R1 mean absolute difference
R2 differing-element count
R2 maximum absolute difference
R2 mean absolute difference
R4 differing-element count
R4 maximum absolute difference
R4 mean absolute difference
```

No padding or post-hoc margin is added.

## Route Decision

`C3_RENDER_TOLERANT_POLICY_EQUIVALENCE` passes only if all ten held-out fresh
processes satisfy every frozen visual bound and every exact policy, Student,
action, and nonvisual-state requirement.

Any held-out policy/action/state mismatch selects:

```text
CANONICAL_BOUNDARY_OBSERVATION_REQUIRED
```

Strict RGB byte parity remains independently reported and is not renamed or
overridden.

## Fresh Seal

The result directory is sealed only after summaries are finalized. An
independent auditor must recompute every listed file hash with zero mismatch.

## Forbidden Claims

This protocol cannot establish attack effectiveness, VIS superiority, Table 1
readiness, renderer hardware failure, or general LIBERO determinism.
