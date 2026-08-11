# Stage V M3.5 V1.4 causal observation snapshot design — 2026-08-11

## Decision

`V1.3.4` remains sealed and non-consumable. The next prospective protocol is
`M3.5 V1.4 CAUSAL_OBSERVATION_SNAPSHOT_REPLAY`.

The causal branch state is no longer treated as simulator state alone:

```text
X_t = simulator state
    + exact canonical observation
    + exact processed policy inputs
    + exact clean reference actions
    + every mutable runtime state that can affect execution
```

This is design-only. No GPU experiment, intervention, V7/M4 run, Teacher,
Student, timing, VIS, or Eval160 read is authorized by these files.

## Live audit result

The formal V1.3.4 source remains `d104713027a82eeb858ba9036200d7ab010959cc`
with tree `3f22ea412975f294b59bc569ef9fb896eff8d410`. Its runner restores a
fresh environment with `set_init_state`, replays the clean action prefix, then
re-renders policy RGB and re-runs the policy. The server LIBERO wrapper shows
that `set_init_state` only writes the flattened MuJoCo state, forwards the sim,
and refreshes observables.

The live runtime sources also show mutable state outside that flat vector:

- wrapper `timestep`, `cur_time`, `done`, horizon and timing state;
- observable sampling timers, delay, cache and current values;
- OSC goals, orientation references, interpolators and robot recent buffers;
- Python/NumPy/Torch RNG state;
- the exact model inputs after the pinned EOS/attention-mask binding.

The machine-readable inventory and sufficiency audit record each disposition
and keep runtime authorization closed until the five implementation bindings
are independently verified:

- [component inventory](STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_COMPONENT_INVENTORY.json)
- [sufficiency audit](STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_SUFFICIENCY_AUDIT.json)
- [model statelessness receipt](STAGE_V_M3_5_V1_4_MODEL_STEP_STATELESS_RECEIPT.json)

## Implemented CPU foundation

`src/gripper_attack/stage_v_causal_observation_snapshot.py` now provides:

- exact raw array sidecars and dtype/shape/byte SHA descriptors;
- simulator data plus registered flat-state capture;
- manifest and sidecar checksum verification with fail-closed tamper errors;
- runtime-state capture scaffolding for wrappers, observables, controllers,
  robot buffers, model/adapter state and RNGs;
- contiguous clean reference-action validation;
- surgical matched-action construction whose arm delta is exactly zero.

The focused module compiles and its bundled-Python self-checks pass. The local
Python installation does not include pytest, so the repository pytest file is
present but has not been executed locally; no test dependency was added.

`scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py` is the prospective
runtime binding. It refuses `DRAFT_NOT_FROZEN`, requires a frozen source
binding and exposed-parent manifest, captures snapshots during the canonical
clean path, and replays only the clean action prefix. Its canary consumes
loaded frozen bytes and restores the recorded RNG state; it does not call the
renderer or policy in the primary replay path. The CPU-only static audit is
recorded in:

- [static audit](STAGE_V_M3_5_V1_4_CAUSAL_SNAPSHOT_STATIC_AUDIT.json)

The static audit is `PASS_STATIC_DESIGN_ONLY`; it is not runtime authorization.

The matched-action producer and independent audits are now also wired:

- `scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py` runs CONTROL/T3/T5/T10
  from the Gate-A packages, keeps CONTROL through the longest 20-step primary
  window, and uses no native policy decode or fresh-render input there.
- `scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_a.py` independently checks
  snapshot sidecars, exact payload fields, reference windows and zero treatment.
- `scripts/detector_v5/audit_stage_v_m3_5_v1_4_gate_b.py` independently
  recomputes branch accounting, treatment compliance, arm isolation, physical
  classes and 3/3 repeatability for one parent.
- `scripts/detector_v5/audit_stage_v_m3_5_v1_4_final.py` requires two valid
  parent roots per suite before it can emit `M3_5_LABEL_VALIDATION=PASS`.

These paths are code-complete enough for preflight but remain unrun and
unauthorized until a new frozen protocol, exact regression, runtime receipt,
smoke root and Gate-A PASS are bound.

## Gate drafts

- [Gate A draft](../../configs/STAGE_V_M3_5_V1_4_GATE_A_PROTOCOL_DRAFT.json)
  is zero-treatment only. It requires exact loaded frozen bytes and forbids
  fresh-render consumption by the primary path.
- [Gate B draft](../../configs/STAGE_V_M3_5_V1_4_GATE_B_PROTOCOL_DRAFT.json)
  requires Gate A PASS and uses matched clean action replay for the primary
  physical window. Native closed-loop is secondary only after `V_phys` is
  frozen.

Neither draft is frozen or runtime-authorized. A future freeze must bind a new
source commit/tree, protocol SHA, exact regression, independent audit,
authorization, smoke root, and fresh Gate A root.
