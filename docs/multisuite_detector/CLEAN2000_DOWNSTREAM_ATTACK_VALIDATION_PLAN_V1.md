# CLEAN2000 Downstream Attack Validation Plan V1

## Status

DRY_RUN_ONLY — no attack launch, no formal output roots, no GPU usage at this stage.

## Purpose

Define the interface contract for future cross-suite VIS attack validation comparing Student-triggered TRUE_T10 vs Random-Time TRUE_T10, using detectors trained under this pipeline.

## Attack Validation Design

For each detector variant that passes evaluation gates:
1. Select frozen clean parent pool from CLEAN2000 primary eligible set
2. Generate paired {Student-triggered, Random-Time} manifest entries
3. Each pair shares: same parent, same state, same perturbation seeds
4. Attack parameters unchanged from TRUE_T10 spec

## Fixed Attack Parameters

```
epsilon: 2/255
PGD steps: 20
K: 10
arm_lock: false
target_token: 31744 (gripper OPEN)
objective: autoregressive_prefix_gripper_target_token_logratio_arm_v3
preprocessing_backend: upstream_tf_jpeg
termination_policy: episode_end
no_emission_policy: ITT_RETAIN
```

## Parent Selection Rules

- Selected BEFORE seeing any attack outcome
- NOT filtered by detector test performance
- NOT filtered by expected attack success
- All mechanism-eligible, teacher-positive, clean-success episodes
- Same state selection rules as TRUE_T10 (first N eligible per fold)
- No-emission episodes retained in ITT denominator

## Manifest Builder Interface

```
Input:
  --detector_bundle: path to frozen detector bundle
  --clean_parent_pool: path to frozen clean parent manifest
  --condition_spec: path to condition spec
  --output: path to dry-run manifest (PREVIEW ONLY)

Output:
  DRY_RUN_MANIFEST.jsonl (marked DRY_RUN_ONLY, no output roots created)
```

## Forbidden at This Stage

- Creating formal output roots
- Launching any attack jobs
- Computing attack ASR
- Comparing detectors by attack outcome
- Selecting "best" detector for downstream use
