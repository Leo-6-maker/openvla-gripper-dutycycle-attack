# Codex C3-G-DEV Stage 3-R1 handoff

## Decision

`C3-G-DEV Stage 3-R1 = PASS`.

Both the main comparison and an independent reviewer passed. The previous
Stage 3 roots remain historical `DEVELOPMENT_NON_CONSUMABLE` evidence and were
not modified or relabeled.

```text
C3-G-DEV-STAGE3-R1-MAIN        = PASS
C3-G-DEV-STAGE3-R1-INDEPENDENT = PASS
C3-G-DEV                       = PASS
C3-T                           = NOT STARTED
CLEAN2000 / D0                 = NOT READ / HOLD
TEACHER / STUDENT              = NOT STARTED
ROLLOUT                        = NOT STARTED
ATTACK                         = NOT STARTED
```

## Code and contract binding

```text
execution_commit = 7b33f6da1e4f3a9ae6d6c9a5e480441938595806
execution_tree   = 8fff73da1ae2697e346aa625920dd045f114a6e7
dataset_commit   = fc18cd965d237c4d40ae60dc8f25be2d8dc98a29
contract         = configs/C3_G_PREDICATE_CONTRACT_V1_1.json
contract_sha256  = 352839f1701a0178e267446c74a84d1312ad7d96ddd869493672fb06a3b94344
evaluator_sha256 = 0b14ada598b5caf8c3161e9da3784032eb62f393449b41ac5bf0b78a1ce3aed2
stage3_sha256    = 4f50847f5d4dd6c1f458e3ed16b93d4d4645e11d3bfc9ebb173b98d1c7f67cf4
```

V1.1 explicitly separates numerical epsilon, containment margin, support
vertical tolerance, and horizontal overlap threshold. Its tolerance
provenance is synthetic-fixture-only; it is not a LIBERO physical-contact
tolerance and does not authorize Clean2000 consumption.

## Fresh sealed evidence

```text
root = /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_g_dev_stage3_r1/evidence_7b33f6d_20260727_0055
run_A/SHA256SUMS.sha256 = d3c3bee999266bd20e77c058db301ee8d1fb64a0fef23b97512a86f3b6383f3b
run_B/SHA256SUMS.sha256 = d28191aebabbdb6e114173c5e600c82c04362f08f5b0ccd5bb0e4e2676530059
comparison/SHA256SUMS.sha256 = 5ab147ba59b4c78d5e81417d963b6ee5b8b268f44ce50d786334d79e1dcc2add
canonical_digest = c93aca0f91f26849368ccbd373efc748baf8ad81caa83c2a9b5c6facc30379e3
```

Each run contains independently consumable `case_inputs.jsonl` and
`predicate_records.jsonl`, plus `input_binding.json`, `canonical_payload.json`,
`summary.json`, `MANIFEST.json`, `SHA256SUMS`, and `SHA256SUMS.sha256`.

## Exact closure

```text
case inputs                 = 264
predicate records           = 264
relations                   = 44
cases per relation         = 6
TRUE cases                  = 44
FALSE cases                 = 44
BOUNDARY cases              = 44
IDENTITY_MISMATCH cases     = 44
POSE_HARD_NEGATIVE cases    = 44
UNKNOWN cases               = 44
observed TRUE               = 88
observed FALSE              = 88
observed UNKNOWN            = 88
missing / extra / duplicate = 0 / 0 / 0
failed case IDs             = none
```

The main A/B stream digests are identical. The independent reviewer recomputed
the canonical streams as:

```text
A/B input digest  = 706ab8073594c5a6ade33dfbb45b23be40f60ec76d426b8cdfdcf86cd67d87b9
A/B output digest = 6a6e8a4c86b4dbe5a53db5f37216d50adc9f140e40bd679788686eedc7135e3d
```

## Independent review

The reviewer used a separate quaternion/target-local implementation and did
not import the producer evaluator or producer runner.

```text
tri-state recomputation = 264/264
case diffs              = 0
recursive seal          = PASS
strict JSON              = PASS (raw NaN/Inf rejected)
case/input SHA binding   = PASS
record SHA binding       = PASS
q/-q equivalence         = PASS
protected reads         = 0
GPU/model/training       = not used
```

Mutation controls:

```text
sealed input mutation    = REJECTED by case hash and recursive seal
translation              = FALSE
target rotation          = TRUE (legal rotation-invariance control on symmetric fixture)
target quaternion -q     = TRUE / equivalent
target extents shrink    = FALSE
identity swap            = UNKNOWN
role swap                = UNKNOWN
```

The integrity mutation was performed on a temporary copy; no sealed evidence
root was changed.

## Test gate

Official environment:

```text
/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
```

Command:

```text
python -m pytest -q \
  tests/test_c3_s3a_fresh_synthetic.py \
  n5/phase3_student/tests/test_c3_s3_geometry.py \
  n5/phase3_student/tests/test_c3_g_dev_stage1.py \
  n5/phase3_student/tests/test_c3_g_predicate_evaluator.py \
  n5/phase3_student/tests/test_c3_g_stage3_synthetic.py
```

Result: `55 passed, 0 failed, 0 errors`.

## Boundary

This PASS establishes only the synthetic C3-G geometry predicate contract and
its independently consumable evidence. It does not establish placement recall
on Clean2000, Teacher quality, Student learnability, rollout readiness, or
attack efficacy. The next authorized boundary is a handoff for review; no
C3-T, Clean2000, Teacher/Student training, rollout, or attack was started.
