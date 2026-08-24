# Stage V M3.5 V1.4 gripper-state repair handoff — 2026-08-12

## Current boundary

V1.3.4 remains immutable and sealed. Its M3.5 measurement contract produced
zero consumable labels because independently reconstructed policy observations
were not causally bound. This is an execution/measurement blocker, not a
scientific refutation; V7, M4, Teacher, Student, scheduler, and timing remain
blocked until M3.5 passes.

## Root cause

The exact simulator state was restored, but the Robosuite Panda gripper's
mutable `current_action` was not part of the controller/wrapper runtime
snapshot. `PandaGripper.format_action()` integrates the next gripper command
from that hidden value. A fresh wrapper could therefore move the fingers in a
different direction even when simulator arrays matched. The old V1.4 Gate B
failure at `goal/task_04 Q06/T3` (`APERTURE_RESPONSE_NOT_SATISFIED`) is
consistent with this hidden-state omission.

## Repair and gates

- Runtime snapshot schema is now `STAGE_V_CONTROLLER_WRAPPER_RUNTIME_STATE_V2`.
- Every robot snapshot binds/restores `gripper.current_action`, `speed`, and
  `dof`; legacy runtime states fail closed.
- Gate A and Gate B independent auditors require the gripper runtime binding.
- Exact A800 regression: 221/221 passed, CUDA visibility empty, protected
  counters zero.
- New Gate A protocol: `configs/STAGE_V_M3_5_V1_4_GATE_A_PROTOCOL_FROZEN_GRIPPER_STATE_20260812.json`.
- New Gate B protocol binds each Gate A receipt and independent-audit SHA:
  `configs/STAGE_V_M3_5_V1_4_GATE_B_PROTOCOL_FROZEN_GRIPPER_STATE_20260812.json`.
- New Gate A run: 8/8 parents PASS, 24/24 snapshots and independent audit PASS
  per parent; runtime payloads contain all three gripper fields; protected,
  Eval160, attack, and VIS counters are zero.

## Active Gate B

The formal Gate B matrix is running on GPU0–7 under new sealed-root prefixes.
GPU admission requires more than 20 GiB free; external processes are not
terminated or modified. One object/task_04 launch at the first timestamp was
rejected before runtime because of a mistyped model path; it has no receipt or
label and is not reused. The corrected worker uses a new root and the bound
model path.

No final aggregator is run until every parent has a valid Gate B receipt and
independent audit. Any branch with horizon censoring or treatment
noncompliance remains an abstain; no tolerance or retry-to-pass is allowed.
