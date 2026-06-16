# M3 Arm-V5 Clean Capture Partial SIGTERM Audit - 2026-06-16

```text
STAGE: M3_ARM_V5_CLEAN_CAPTURE_V5_1
RESULT_CLASS: V5_CAPTURE_PARTIAL_POST_ACTION_TERMINATION
SCIENTIFIC_STATUS: INVALID_FOR_V5_1_DENOMINATOR
V5_2_ATTACK: NOT_RUN
TRUE_PGD21: NOT_RUN
RAND21: NOT_RUN
SHUFFLED_GRAD21: NOT_RUN
LIBERO_ROLLOUT_ATTACK: NOT_RUN
```

## Context

This run used the GPU45 stable diagnostic profile discovered in the earlier
GPU45 infrastructure stage:

```text
CUDA_VISIBLE_DEVICES=5,4
GPU5 = GPU-9794d733-042f-46a2-fc86-5a3fe32a158a
GPU4 = GPU-d0a54f5d-938c-a148-fff9-c135201e3f61
```

It attempted V5.1 clean capture only. No PGD, RAND21, shuffled-gradient, or
attack rollout was launched.

## Run Binding

```text
repo_commit: 1f2e84d16313014d580a2b29a1470ba6ad36c362
branch: exp/m3-arm-v5-clean-close-event-panel-20260616
output_dir: /data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039
config_sha256: 670d7e435c990676e8bbfee5c5c01bf8fd42bc25c3482e21f609f9153ce8ca61
ledger_sha256: dbb7a7f198e8c27113898767a1bf53d06f60472cd5596c8b74da655836d431c0
state_pool_sha256: 3350d8bd9b0a7f855efaec8a1b7b99bf0ffc1059f08cc5948bb7c5ecf98a0721
```

## Failure

The process terminated externally after partial progress. It did not complete
the 20-state clean capture pool and therefore cannot freeze the V5.1
denominator.

The terminal boundary is post-action for `butter_s23`:

```text
state: butter_s23
attempt: attempt_0
markers:
  ATTEMPT_STARTED
  MODEL_READY
  ENV_READY
  FIRST_ACTION_GENERATED
  FIRST_ACTION_TAKEN
missing:
  CAPTURE_COMPLETED
clean_records: missing
step_artifacts: 146
```

Per the frozen V5 capture policy, any crash after `FIRST_ACTION_TAKEN` is a
post-action failure and cannot be automatically retried as a valid V5.1
capture attempt.

## Partial Attempt Summary

The small summary table is committed at:

```text
tables/m3_arm_v5_clean_capture_partial_sigterm_attempts_20260616.csv
```

Captured states:

```text
alphabet_soup_s14
alphabet_soup_s9
bbq_sauce_s18
bbq_sauce_s22
butter_s30
```

Interrupted state:

```text
butter_s23
```

Not started:

```text
14 frozen state-pool candidates
```

## Server-Side Evidence

Postmortem evidence was added to the server output directory without
overwriting prior artifacts:

```text
postmortem_json:
  /data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039/m3_arm_v5_partial_capture_sigterm_postmortem.json
postmortem_json_sha256:
  7f338bfb1ae4384a77dfa6df83806ce631cdbc3b726fc42da43ae7cd5ea590db
recursive_manifest:
  /data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039/m3_arm_v5_partial_capture_sigterm_recursive_sha256_manifest.csv
recursive_manifest_sha256:
  95f5da275b74b1b7c310e85794e6a453dfa13fb5e0d694bf2d634f7c0ede8e9e
```

GPU4/5 were idle during postmortem inspection. Other jobs were present on
physical GPUs 1, 2, 3, and 6 and were not touched.

## Code Repair

The capture runner has been repaired after this failure so future terminations
write stronger fail-closed evidence:

```text
repair_commit: eed2951c020b45e1ad6dad619aaeabb2705a512c
```

The repair:

- installs SIGTERM/SIGINT handlers that raise a catchable capture termination;
- writes the attempt ledger at attempt start, capture success, and failure;
- classifies signal termination after `FIRST_ACTION_TAKEN` as
  `CAPTURE_FAILED_POST_ACTION`;
- adds regression coverage for SIGTERM-style post-action termination.

CPU verification:

```text
python -m py_compile scripts/stageb/run_m3_arm_v5_clean_capture.py scripts/stageb/audit_m3_arm_v5_clean_capture.py
PYTHONPATH=. pytest tests/stageb/test_m3_arm_v5_clean_capture_runner.py tests/stageb/test_m3_arm_v5_event_panel.py -q

44 passed
```

## Allowed Claim

The run produced partial clean-capture artifacts and exposed a fail-closed
capture interruption after `FIRST_ACTION_TAKEN`. The runner now records terminal
ledger evidence more robustly for future captures.

## Forbidden Claim

This run does not establish a V5.1 denominator, does not establish exact frozen
V5 inputs, does not authorize seed428198, and provides no evidence that TRUE_PGD
or VIS outperforms random controls.

## Next Gate

External audit is required before deciding whether to authorize a fresh V5.1
clean capture attempt from the repaired commit. No V5.2 attack or panel
comparison should run from this partial output.
