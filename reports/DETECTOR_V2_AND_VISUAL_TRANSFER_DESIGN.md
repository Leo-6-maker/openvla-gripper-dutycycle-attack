# Detector v2 and VisualTransfer Design

## Current Route

Stage 1: phase / candidate gate.

Stage 2: `vulnerability_ready` selector.

Attack: VIS is attempted only after selector approval and matched denominator controls.

This is not detector-triggered online VIS unless ProprioNoStep actually supplies the phase windows online under the frozen protocol.

## Detector v1 Conclusion

Current v1 label state:

- 19 train labels.
- 11 positive / 8 negative.
- Always-positive baseline matches LR task-key F1.
- No model beats prevalence in a meaningful way.
- v1 is underpowered and diagnostic-only.

Allowed claim: v1 revealed the prevalence/task-key confound.

Forbidden claim: v1 learned `vulnerability_ready`.

## Why Batch3b / Batch3c Must Be Added

Batch3b/c are needed because v1 lacks enough controls and negative recall pressure:

- `stable_post_lock` controls test whether late gripper opening is task-relevant or denominator-confounded.
- `far_too_early` controls test whether early closed windows are false-positive prone.
- `pre_lock` negatives/controls test whether the candidate gate overselects pre-lock windows.
- Negative recall must improve; otherwise the selector is just an always-positive/prevalence shortcut.
- Control false positives must decrease before any deployment-style interpretation.

Polluted, random-failed, denominator-failed, infra-failed, Xid/OOM, missing trace, provenance-failed, schema-incomplete, ambiguous, or manual-review rows must not enter training.

## Proprio-Only v2 Pass Conditions

Proprio-only v2 is useful only if it satisfies all of the following diagnostic criteria:

- Balanced accuracy beats prevalence baseline.
- Macro F1 beats prevalence baseline.
- Negative recall is meaningfully above zero.
- Control false positives decrease, especially for stable_post_lock and far_too_early rows.
- Performance beats task-key-only and phase-bin-only baselines.
- LOTO fold warnings are acknowledged rather than treated as clean generalization.

Passing a minimum row-count gate permits diagnostic training only. It does not establish deployment readiness.

## When Visual Module Becomes Necessary

Add a visual module if, after N >= 40 valid labels:

- Proprio-only still cannot beat prevalence or task-only baselines.
- Negative recall remains low.
- Controls remain false-positive heavy.
- The same proprio phase is vulnerable for one object/context but safe for another.

This would indicate that timing alone is insufficient and that object/context appearance mediates whether VIS transfers into physical/task failure.

## VisualTransferHead Design

Roles:

- ProprioNoStep = timing / contact phase gate.
- VisualTransferHead = whether/how long/risk selector.
- VIS = attack payload, only after selector approval.

Initial design:

- Use frozen visual embeddings first.
- Do not start end-to-end visual training first.
- Avoid future frames for online model inputs.
- Use only causally available frames/features at or before the candidate window.
- Treat `qpos_delta_after_window`, VIS OPEN counts, task outcomes, manual audit outcomes, random outcomes, oracle outcomes, and attack-result fields as label-only or audit-only.
- Never feed `qpos_delta_after_window` or attack outcome fields as detector inputs.

Suggested baselines:

- task_only
- visual_only
- proprio_only
- visual+proprio

Minimum reporting:

- Balanced accuracy.
- Macro F1.
- Negative recall.
- MCC.
- TP/FP/FN/TN.
- Control false-positive rate.
- Per-task and per-phase split warnings.

## Real-Robot Deployment Boundary

Proprio detector can be a high-recall monitor for contact/phase timing, but not a complete vulnerability selector.

`vulnerability_ready` may require visual/context inputs before active use.

Real robot use must start with passive logging:

1. Log clean proprio/visual features.
2. Verify phase gate alignment.
3. Verify selector false positives on controls.
4. Compare offline predicted vulnerability to physical gripper/contact evidence.
5. Only then consider active attack tests under explicit approval.

No real-robot active attack is authorized by this design note.
