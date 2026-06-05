# VisualTransferHead Design V0

## Motivation

Detector v1 did not beat the prevalence baseline. The current evidence says proprio-only features may be useful as a high-recall phase/candidate gate, but not yet as a complete `vulnerability_ready` selector.

The missing signal may depend on object geometry, gripper-object relation, and scene context. The visual module is therefore scoped as a Stage 2 transfer/risk selector, not a replacement for ProprioNoStep.

Current mainline:

1. Stage 1: Proprio / phase candidate gate.
2. Stage 2: `vulnerability_ready` selector.
3. VIS only after the selector and matched controls.

VisualTransferHead augments Stage 2 on an already proposed candidate window. It asks:

- Is VIS likely to produce a physical qpos response?
- Is that response likely to become task failure?
- Should the selector abstain?
- If compression later succeeds, which budget bucket is plausible: L8 / L10 / L12 / L18?

## Architecture

```text
clean/probed rollout
-> phase/candidate gate
-> trigger-centered feature extraction
-> VisualTransferHead
-> outputs:
   qpos_response_score
   failure_susceptibility_score
   control_false_positive_risk
   abstain_score
   optional_budget_bucket
```

ProprioNoStep remains the timing/contact gate. VisualTransferHead estimates whether the candidate is transferable and risky enough to attack.

## Allowed Online Inputs

- Current RGB frame.
- Past RGB frames only.
- Task instruction.
- Frozen visual embedding.
- Proprio/action summary before trigger.
- Phase gate score / candidate role.
- Gripper qpos / width / command history before trigger.
- EEF pose / velocity / action history before trigger.

## Forbidden Inputs

- Future frames.
- `done` / success.
- `claim_usable`.
- `VIS_OPEN`.
- `qpos_delta_after_attack`.
- `denominator_clean` as model input.
- Random / oracle / manual audit outcome.
- `object_pose` / `target_pose` for deployable student.
- Any attack outcome field.

Forbidden fields may appear only as labels or audit metadata where explicitly marked.

## Label Targets

- `physical_response_label`
- `vulnerability_ready_label`
- `control_false_positive_label`
- `budget_bucket`, only if compression later succeeds and is validated.

## Evaluation Plan

Baselines:

- `always_positive`
- `task_key_only`
- `phase_only`
- `proprio_summary_only`
- `vision_only`
- `vision + proprio`
- `vision + task`
- `vision + proprio + task`

Splits and metrics:

- Leave-task-out when enough tasks exist.
- False positives on controls.
- Balanced accuracy.
- Macro F1.
- Negative recall.
- MCC.
- TP / FP / FN / TN.

No visual result is meaningful unless real frozen visual features beat task-key and prevalence baselines while reducing control false positives.

## Claim Boundary

This is v0 design/scaffold only.

Do not claim:

- Deployable visual detector.
- Cross-suite generalization.
- Real robot readiness.
- Detector-triggered online VIS.
- VisualTransferHead effectiveness.

Allowed claim:

- Existing data can be audited for trigger-centered visual-path availability.
- Dummy features can validate pipeline shape only.
