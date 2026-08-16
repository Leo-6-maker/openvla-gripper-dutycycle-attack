# Stage VI Case-B development handoff

Status: `STAGE_VI_DEVELOPMENT_NO_IMPROVEMENT`

Stage V remains immutable. Its final conclusion is still
`STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED` with AUROC `0.5153`.

## Bound evidence

- PR: [#115](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/115)
- Diagnostic source: `52c48e8f2afbf5046863980d3374aa6e71023d3d`
- Development source: `04aa50f7a2a6a6b48b5718e0754633ac293afd20`
- Frozen Case-B criteria SHA256: `307d6bc545a0eb821a733251903de3faffeb24c6f68a655cff9e42fb72771a73`
- Clean-only R3 aggregate: `STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_AGGREGATE_20260816T181000Z`, seal `33e00bb74c59fed332d9a6a1fbb385f374d8c145435f9b81ac92c65d4bb3729c`
- Root-cause diagnostic: `STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_M4_R3_RECONSTRUCTED_20260816T181500Z`, seal `7fdd566a5c27ef6636255379e37d367f5d63d0ac0659774f0468f94399ec3dfb`
- S6-C development root: `STAGE_VI_CASE_B_DEVELOPMENT_S6C_20260816T183000Z`, seal `b0b5140d68a0f43174e4e2703a04e736a83faef3a6efb924352924e5b7bbbe70`

The aggregate contains 40 unique parents and 7,297 complete R3 clean rows.
It has zero intervention execution, zero formal labels, `Eval160=UNREAD`, and
all protected counters zero. The reconstructed R3 labels are diagnostic joins
only and are not consumable M4 labels or Student-training labels.

## Root-cause result

The read-only Stage V development decomposition classifies exactly Case B:

- Original clean-only Teacher → `V_phys@T5`: AUROC `0.5211`.
- 25D instantaneous grouped probe: AUROC `0.9287`.
- Privileged clean-state grouped probe: AUROC `0.8750`.
- 16-step causal window grouped probe: AUROC `0.9045`.
- Frozen Student → `V_phys@T5`: AUROC `0.5153`.
- Student ↔ reconstructed Teacher coverage: `858/858` rows joined.

This supports a Teacher-target mismatch diagnosis, not a claim that the
original Teacher is useful for physical vulnerability.

## Frozen S6-C gate result

The only preregistered candidate was `T_v` (privileged logistic Teacher) plus a
16-step causal 25D Student. The gate was frozen before training:

- Development-check AUROC gain: `+0.2468` — pass.
- Development-check AUPRC gain: `+0.0019` — fail; required `+0.10`.
- Per-suite minimum AUROC: `0.5341` on `libero_10` — fail; required `0.60`.
- Emission coverage at fixed threshold `0.5`: `0.7235` — pass.

Therefore S6-C is not promoted. No `STAGE_VI_PRE_HOLDOUT_LOCK` was created,
no fresh 16-parent M4 or 64-branch timing matrix was launched, and no
outcome-informed retuning is authorized. The next legal state is the sealed
negative development conclusion, not fresh held-out execution.

`Eval160` and protected evaluation remain unread.
