# R7 K10 Opportunity Labeler V1.2.1 Protocol

Status: FIT-only, clean-only development protocol. This protocol does **not** authorize protected-split reads, formal model selection, simulator branching, VIS attacks, or CS200 execution.

## Scientific claim boundary

The labeler identifies a **clean gripper-critical opportunity start**. It does not label a VIS-exploitable window and does not estimate counterfactual task-failure probability.

A window becomes VIS-exploitable only after a separately frozen intervention study shows:

1. command-level OPEN at the selected start causes physical/contact degradation;
2. targeted VIS at the same start changes the gripper command and physical state;
3. targeted VIS is stronger than a same-window matched random perturbation;
4. the same targeted VIS is stronger at the Detector window than at a matched random window.

## Frozen source

- Source: Physics Teacher V2.1
- Source `SHA256SUMS` SHA-256:
  `18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da`
- Scope: FIT states 0–19 only, 4 suites × 10 tasks × 20 states = 800 identities
- Attack outcomes, manual failure labels, FIT-DEV, CAL, CHECK, and CS200 are forbidden inputs.

## Step-level opportunity predicate

For each clean step `t`, define `critical_t = true` only when all of the following hold:

```text
known_mask
AND student_valid
AND candidate_close
AND task_grasp_necessity >= 0.5
AND component_valid_mask.relative_pose_stability
AND stable_grasp_score >= 0.5
AND (
      valid lift_score >= 0.3
      OR valid support_removed >= 0.3
      OR valid+known target_progress > 0.05
    )
AND component_valid_mask.release_risk
AND release_risk <= 0.5
AND component_valid_mask.regrasp_or_instability_risk
AND regrasp_or_instability_risk <= 0.5
```

`task_grasp_necessity` is a task-role applicability proxy derived from the BDDL task decoder. It is not a step-level active-object classifier. Reports must describe this limitation explicitly, especially for multi-object LIBERO-10 tasks.

## K=10 feasible start

For fixed intervention length `K=10`, start `t` is feasible iff:

```text
critical_t ... critical_(t+9) are all true
AND all ten steps share the same non-empty candidate window_id
AND the burst does not exceed the episode horizon
```

The dense feasible-start mask is the primary clean-only Teacher target. `first_feasible_start` is diagnostic only and must not replace the dense mask during training.

## Component attribution

Each critical step stores an independent bitmask:

```text
1 = lift evidence
2 = support-removed evidence
4 = target-progress evidence
```

Per-start source categories use the union of bitmasks over all ten steps. Reports must call this an **any-step K10 union**, not simultaneous same-step evidence. Persistent support may additionally be reported using the ten-step intersection.

## Required generator outputs

The sealed label root must contain at least:

- `PROTOCOL.json`
- `SOURCE_BINDING.json`
- `MANIFEST.json`
- `AUDIT.json`
- `EPISODE_SUMMARY.csv`
- `TASK_GEOMETRY.csv`
- 800 `labels/<suite>/task_XX/state_YY/k10_labels_v121.jsonl` files
- `SHA256SUMS`
- `SHA256SUMS.sha256`

`SOURCE_BINDING.json` must contain the Physics Teacher digest, `K`, schema, runtime Git commit, and SHA-256 of the executed labeler file.

## Independent acceptance audit

The generator's own `AUDIT.json` is not sufficient. Acceptance requires the independent read-only auditor:

```text
scripts/detector_v4/audit_k10_v121_artifact.py
```

The independent auditor must verify:

- exact checksum/file-set closure for source and output roots;
- all 800 source identities and every source row;
- row schema, canonical identity, step continuity, types, validity masks, and finite numeric values;
- clean repository worktree and source-binding agreement;
- exact recomputation of every output feasible start;
- all ten steps of each positive burst are known, Student-valid, candidate-close, critical, release/regrasp-valid, nonzero-component, and in one candidate segment;
- `AUDIT.json`, `EPISODE_SUMMARY.csv`, and `TASK_GEOMETRY.csv` agree with recomputed labels.

## Hard gates

```text
Teacher sealed-root closure                  = PASS
Teacher identity closure                     = 800/800
Teacher malformed/non-finite required rows   = 0
Output sealed-root closure                   = PASS
Output identity closure                      = 800/800
Segment-crossing feasible starts             = 0
Out-of-bound feasible starts                 = 0
Unknown steps in feasible bursts             = 0
Student-invalid steps in feasible bursts     = 0
Release-risk-invalid steps in feasible bursts= 0
Regrasp-risk-invalid steps in feasible bursts= 0
Zero-component steps in feasible bursts      = 0
Summary/recomputed aggregate mismatch        = 0
Protected split reads                        = 0
Attack/manual outcome reads                  = 0
Source artifact mutation                     = 0
```

Any failure keeps R7.2 on HOLD.

## Promotion boundary

Passing R7.1 only authorizes R7.2: read-only offline replay of already sealed V5 predictions/checkpoints against the new K10 labels. It does not authorize new training, threshold tuning on protected splits, exact-prefix implementation, or attack execution.
