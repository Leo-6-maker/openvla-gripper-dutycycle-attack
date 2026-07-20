# Official V3 R10.4D Single-Episode Passive Smoke Preparation

## Decision basis

R10.4-R4C classified the replay mismatch as:

```text
CONTACT_DYNAMICS_REPLAY_DIVERGENCE
```

The original CLEAN `step_records.features_25d` and S1
`student_input_records.features_25d` are exactly identical for all 520 steps.
The replay action sequence was unchanged, and the first divergence occurred in
direct 13D physical observations after 140 exact steps.  This does not justify
Teacher, Student, threshold, or FSM retraining.

## This branch prepares, but does not execute

- one real passive OpenVLA–LIBERO episode;
- parent `libero_10/task_00/state_20` only;
- frozen R10.3 dual-head GRU checkpoint loaded with `strict=True`;
- canonical `SC5StreamingFeatureAdapterV2` through
  `OfficialStreamingFeatureAdapter`;
- official `OfficialOpenVLAActionAdapter` and official image helper;
- exactly one measured generation per step;
- one-shot frozen vertical-lift FSM;
- clean postprocessed action copied exactly to the environment;
- privileged information written only to an isolated post-hoc Teacher sidecar.

This branch does **not** include a live authorization receipt and does not run
OpenVLA, LIBERO, the detector, command-OPEN, VIS, RAND, or any attack.

## Added files

- `src/gripper_attack/r10_4d_passive.py`
- `scripts/r10_4/run_r10_4d_passive_smoke.py`
- `scripts/r10_4/build_r10_4d_authorization_receipt.py`
- `scripts/r10_4/audit_r10_4d_passive_prep.py`
- `tests/test_r10_4d_passive.py`
- `configs/R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_V1.json`
- `.github/workflows/cpu-r10-4d-prep.yml`

## Hard runtime gates

Before any 7B model load, the runner requires:

1. a clean checkout at the exact receipt-bound Git HEAD;
2. one frozen parent manifest selecting only
   `libero_10/task_00/state_20`;
3. a checksum-closed R10.3 deployment bundle;
4. the exact detector checkpoint SHA256;
5. the exact OpenVLA checkpoint-tree SHA256;
6. an R4C classification record proving original CLEAN–S1 parity and excluding
   training-source and feature-adapter failure;
7. a machine-built authorization receipt bound to the GitHub authorization
   comment;
8. a new non-existing output root.

The receipt generator performs hashing and contract validation only.  It does
not import OpenVLA, torch, LIBERO, or the detector.

## Runtime pass interpretation

Both outcomes below are valid integration passes:

```text
PASS_RUNTIME_NO_EMIT
PASS_RUNTIME_EMIT_OBSERVED
```

No emit is acceptable for a single parent.  Task success is not required for
an integration pass.  The smoke fails on any of the following:

- missing, zero, multiple, or boolean generation count;
- invalid/non-finite 25D feature stream;
- detector checkpoint strict-load failure;
- unsupported or substituted parent;
- any action difference between clean and executed 7D action;
- duplicate emit;
- privileged sidecar use as detector input;
- existing output root;
- receipt, bundle, checkpoint, model-tree, source-HEAD, or parent binding
  mismatch.

## Stop boundary after execution

After exactly one episode, stop and report:

- final source commit and clean-worktree proof;
- authorization receipt SHA256;
- model tree and detector checkpoint SHA256;
- parent/BDDL/init-state binding;
- total policy steps and measured generation count per step;
- maximum error for all seven action dimensions;
- feature validity;
- Student probability/FSM/event trace;
- emit count and one of the two valid runtime pass labels;
- task success as a descriptive field only;
- output `SHA256SUMS` and `SHA256SUMS.sha256` digests.

A successful smoke does not authorize a second episode, a task panel,
command-OPEN, VIS, RAND, threshold tuning, FSM changes, or retraining.
