# DeepSeek Handoff — R10.4D Single-Episode Passive Smoke

## Status at handoff

This document prepares one real passive integration episode.  It does not by
itself authorize execution.  Execution requires a later GitHub Issue #88
comment ID and a machine-built receipt generated from the final clean HEAD.

```text
Teacher / Student / threshold / FSM = FROZEN
Parent                            = libero_10/task_00/state_20
Episodes                          = exactly 1
GPU / render GPU                  = physical GPU0 only
Action mutation                   = prohibited
Command-OPEN / VIS / RAND         = prohibited
Training / retuning               = prohibited
```

## 1. Use a clean isolated checkout

Do not touch or clean the existing dirty main worktree.

```bash
git -C /mnt/sdc/dty_user/openvla_attack fetch origin
HEAD_SHA=$(git -C /mnt/sdc/dty_user/openvla_attack rev-parse origin/codex/r10-4d-passive-smoke-prep-20260720)
WT=/mnt/sdc/dty_user/worktrees/r10_4d_passive_${HEAD_SHA:0:8}
git -C /mnt/sdc/dty_user/openvla_attack worktree add --detach "$WT" "$HEAD_SHA"
test -z "$(git -C "$WT" status --porcelain)"
```

Do not proceed until the Draft PR CPU workflows pass at this exact HEAD.

## 2. Seal the R4C classification record

Create a new non-overwrite JSON outside the repository from the sealed R4C
evidence.  Do not infer or edit the underlying evidence.

Required fields:

```json
{
  "schema": "R10_4C_DIVERGENCE_CLASSIFICATION_V1",
  "classification": "CONTACT_DYNAMICS_REPLAY_DIVERGENCE",
  "clean_s1_exact_parity": true,
  "clean_s1_max_abs_error": 0.0,
  "action_mutated": false,
  "first_divergence_layer": "DIRECT_13D",
  "feature_adapter_bug": false,
  "training_source_binding_failure": false,
  "first_divergence_step": 140,
  "source_evidence_root": "<ABSOLUTE SEALED ROOT>",
  "source_sha256s_sha256": "<64 HEX>"
}
```

The source root and digest must be verified before writing the classification
record.  If the sealed R4C source root or digest is unavailable, stop with:

```text
R10_4D = HOLD_R4C_SOURCE_BINDING
```

## 3. Verify GPU0 without changing unrelated processes

Read only:

```bash
nvidia-smi -i 0
```

Proceed only when GPU0 has sufficient free memory and no unrelated job would
be disturbed.  Do not switch to another GPU.  If GPU0 is unavailable, stop:

```text
R10_4D = HOLD_GPU0_UNAVAILABLE
```

## 4. Build the one-use authorization receipt

The receipt must be written outside the Git worktree so the checkout remains
clean.  Substitute the actual Issue #88 authorization comment ID.

```bash
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python

$PY "$WT/scripts/r10_4/build_r10_4d_authorization_receipt.py" \
  --model-path <PINNED_OPENVLA_LIBERO10_CHECKPOINT_DIR> \
  --detector-bundle /mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720 \
  --parent-manifest <SEALED_R10_4_PARENT_MANIFEST_JSON> \
  --r4c-classification <SEALED_R4C_CLASSIFICATION_JSON> \
  --protocol "$WT/configs/R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_V1.json" \
  --authorization-comment-id <ISSUE_88_COMMENT_ID> \
  --output /mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_authorization_<HEAD8>.json
```

Verify that both receipt files exist:

```text
r10_4d_authorization_<HEAD8>.json
r10_4d_authorization_<HEAD8>.json.sha256
```

The receipt builder must not load OpenVLA, LIBERO, torch, or the detector.

## 5. Run exactly one episode

Use a new non-existing output root.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
$PY "$WT/scripts/r10_4/run_r10_4d_passive_smoke.py" \
  --model-path <PINNED_OPENVLA_LIBERO10_CHECKPOINT_DIR> \
  --detector-bundle /mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720 \
  --parent-manifest <SEALED_R10_4_PARENT_MANIFEST_JSON> \
  --protocol "$WT/configs/R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_V1.json" \
  --authorization-receipt /mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_authorization_<HEAD8>.json \
  --output-root /mnt/sdc/dty_user/openvla_attack_evidence/R10_4D_PASSIVE_SMOKE_<HEAD8>_<TIMESTAMP> \
  --upstream-root <PINNED_OPENVLA_UPSTREAM_CHECKOUT> \
  --gpu 0 \
  --render-gpu 0
```

No second attempt is authorized.  A runtime exception, strict-load failure,
binding mismatch, or output failure must be preserved and reported; do not
change parent, GPU, model, threshold, FSM, or code and retry.

## 6. Valid outcomes

Both are valid integration passes:

```text
PASS_RUNTIME_NO_EMIT
PASS_RUNTIME_EMIT_OBSERVED
```

No emit is acceptable.  Task failure is descriptive and does not fail the
integration smoke when all runtime invariants pass.

Hard failures include:

- generation count missing, boolean, zero, or not equal to one;
- invalid/non-finite 25D stream;
- detector strict-load or parameter-count failure;
- action parity above exactly zero in any of seven dimensions;
- unsupported/substituted parent;
- duplicate emit;
- privileged sidecar used as detector input;
- receipt, receipt sidecar, protocol, parent manifest, bundle, checkpoint,
  model tree, source HEAD, BDDL, or init-state mismatch;
- existing output root;
- any attack or action-override path.

## 7. Stop and report

After the first episode, stop.  Report:

```text
R10.4D source HEAD:
clean worktree:
Issue #88 authorization comment ID:
authorization receipt SHA256:
model tree SHA256:
detector checkpoint SHA256:
detector bundle SHA256SUMS SHA256:
parent / BDDL SHA256 / init-state SHA256:
policy steps:
generation passes per step:
max action error across all 7 dimensions:
feature valid steps:
Student emit count and step:
FSM violations:
task success:
result label:
output root:
SHA256SUMS SHA256:
SHA256SUMS.sha256 SHA256:
```

Do not start a second passive episode, a 10-task panel, command-OPEN, VIS,
RAND, training, or tuning.  Those remain separately gated.
