# M3 Arm-V5 Infrastructure Final Seal Handoff - 2026-06-16

```text
STAGE: M3_ARM_V5_1R_CAPTURE_INFRA_FINAL_SEAL
GPU_EXECUTION: NOT_RUN
V5_1_FRESH_CAPTURE: NOT_RUN
V5_2_ATTACK: NOT_RUN
SEED428198: NOT_RUN
LIBERO_ATTACK_ROLLOUT: NOT_RUN
```

## Summary

This handoff seals the Layer3 V5.1/V5.2 software and audit chain while Layer2
GPU jobs occupy the healthy GPU pairs. It does not run new scientific GPU work.

## V5.1 Capture Repairs

Implemented:

- truly independent clean-capture auditor:
  - no import from the producer capture runner;
  - no import from the shared event selector;
  - independently recomputes the 20-state pool from the prior Layer3 ledger;
  - independently validates attempt markers and retry legality;
  - independently recomputes earliest clean CLOSE events;
  - independently freezes the first eight events by state hash;
  - independently verifies exact input artifact bindings.
- ordered GPU binding:
  - physical `CUDA_VISIBLE_DEVICES` index order must match ordered UUID list;
  - unordered UUID set membership is no longer sufficient.
- atomic artifact writes for producer CSV/JSON/phase markers using temp files
  and `os.replace`.
- model bundle exact-set audit in the independent auditor:
  - actual model directory is re-enumerated;
  - manifest rows must exactly equal actual config/tokenizer/processor/remote
    code/weight shard set.
- capture commit is read from capture-root provenance, not from the audit-time
  checkout HEAD.
- authorized runtime environment is documented as:

```text
/home/liuyu/.conda/envs/openvla_official_libero_20260525
```

Fallback to the similarly named `/data/aviary/...` environment remains
forbidden.

## Old Partial Root External Audit

Read-only external audit was run on:

```text
/data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039
```

Result:

```text
audit_status: FAIL
failure_reason: V5_CAPTURE_POOL_INSUFFICIENT
ledger_present: false
captured_count: 5
post_action_interrupted_count: 1
not_started_count: 14
selected_count: 5
```

The old root remains:

```text
V5_CAPTURE_PARTIAL_POST_ACTION_TERMINATION
INVALID_FOR_V5_1_DENOMINATOR
```

Server evidence:

```text
/data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039_external_audit_2a57ffe
```

Git summary:

```text
tables/m3_arm_v5_partial_root_external_audit_2a57ffe_20260616.csv
```

## Fresh V5.1 Restart Amendment

Added:

```text
reports/M3_ARM_V5_1_FRESH_CAPTURE_RESTART_AMENDMENT_20260616.md
```

The amendment freezes:

- old root cannot be resumed;
- old five captured states cannot be reused;
- `butter_s23` cannot be filled inside the old root;
- fresh V5.1 requires a new output root;
- all 20 frozen states must run from scratch;
- state pool and event-selection rule remain unchanged.

## V5.2 CPU/Mock Harness Seal

Added CPU-only artifact contract:

```text
src/gripper_attack/m3_v5_attack_harness.py
scripts/stageb/run_m3_arm_v5_frame_group.py
scripts/stageb/audit_m3_arm_v5_frame_group.py
tests/stageb/test_m3_arm_v5_attack_harness.py
```

This contract validates:

- exact frozen input SHA binding;
- frozen seed `428198`;
- forbidden legacy seeds `85` and `86`;
- 21 candidates per condition;
- condition-specific artifact directories;
- TRUE/RAND/shuffled cross-contamination rejection;
- epsilon/L∞ gate;
- exact 7-token official decode fields;
- target token `31744`;
- arm-prefix gate `>=5/6`;
- independent frame-group audit.

It does not run OpenVLA, PGD, RAND, shuffled-gradient, or LIBERO.

## Future Wrapper Scripts

Generated fail-closed wrappers:

```text
scripts/stageb/run_v5_1_fresh_capture.sh
scripts/stageb/audit_v5_1_fresh_capture.sh
scripts/stageb/run_v5_2_dev_smoke.sh
scripts/stageb/run_v5_2_frame_group.sh
scripts/stageb/audit_v5_2_frame_group.sh
```

They require explicit commit/branch/SHA/GPU UUID bindings and reject dirty
worktrees or old output roots. `run_v5_2_frame_group.sh` intentionally exits
before real GPU work until the real V5.2 runner is authorized after V5.1 PASS.

## Verification

CPU tests:

```text
python -m py_compile \
  scripts/stageb/run_m3_arm_v5_clean_capture.py \
  scripts/stageb/audit_m3_arm_v5_clean_capture.py \
  scripts/stageb/run_m3_arm_v5_frame_group.py \
  scripts/stageb/audit_m3_arm_v5_frame_group.py

PYTHONPATH=. pytest \
  tests/stageb/test_m3_arm_v5_clean_capture_runner.py \
  tests/stageb/test_m3_arm_v5_event_panel.py \
  tests/stageb/test_m3_arm_v5_attack_harness.py -q
```

Expected result:

```text
all CPU tests pass
```

## Next Authorized Action

Wait for Layer2 jobs to finish, then perform the planned physical cold boot and
GPU qualification. Only after qualification should a fresh V5.1 capture be
authorized on one healthy GPU pair.

## Forbidden Claims

Do not claim:

- V5.1 denominator is frozen;
- V5.2 has run;
- TRUE_PGD beats RAND;
- official-token attack effect is established;
- closed-loop Layer3 is solved.
