# VIS Threshold Sweep Context Reuse Status

Date: 2026-05-31

## Status

Implemented a no-rollout diagnostic plumbing improvement:

```text
scripts/diagnostics/vis_one_frame_loader.py
scripts/diagnostics/vis_token_flip_threshold.py
```

The one-frame loader now exposes:

```text
prepare_one_frame_context(...)
run_one_frame_attack(...)
```

`vis_token_flip_threshold.py` now prepares the OpenVLA model, frame, instruction,
and clean decode once, then reuses that context across objective / epsilon / step
combinations.

## Why

The previous threshold diagnostic path called `run_one_frame(...)` for every
parameter combination. That would reload the OpenVLA model and re-decode the
same clean frame repeatedly, making the intended one-frame threshold sweep
unnecessarily slow and more likely to hit OOM.

This change keeps the diagnostic scope unchanged:

- no rollout
- no env step
- no training
- no production runner default change
- no use of `action_adv`
- adversarial action still re-decodes from `debug["adv_inputs"]`

## Test Coverage

Added:

```text
tests/v4/test_vis_token_flip_threshold_sweep.py
```

The test mocks the real model path and verifies:

- one-frame context preparation is called once
- attack/re-decode is called once per parameter combination
- the output CSV contains the expected sweep rows

Server validation:

```text
pytest tests/v4/test_vis_token_flip_threshold_sweep.py tests/v4/test_token_prefix_pgd_interface.py
```

Result:

```text
8 passed
```

## Gate Status

This is harness-only progress. It does not change the current VIS gate decision.

Current status remains:

```text
VIS-Loader: PASS
VIS-1: FAIL from the valid-budget one-row smoke
VIS rollout: blocked
```

The next allowed VIS diagnostic is a one-frame threshold sweep using this
context-reuse path. It is still diagnostic-only and must not be interpreted as
VIS rollout evidence.
