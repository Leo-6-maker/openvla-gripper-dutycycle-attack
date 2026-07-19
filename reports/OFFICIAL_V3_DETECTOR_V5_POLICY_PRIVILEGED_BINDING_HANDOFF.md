# Official V3 V5 policy/privileged source binding handoff

Snapshot: 2026-07-18  
Scope: Official V3 FIT states 0--19 only.  
Execution: read-only; no model inference, replay, training, protected-split read, or attack.

## Result

The raw asset audit was only a metadata/availability result. This follow-up
performed the missing field-level binding against the sealed registry, S1 root,
and every Official V3 FIT artifact.

```text
POLICY_INTENT_BINDING                 = PASS 800/800
POLICY_INTENT_STEPS                   = PASS 176336/176336
POLICY_ACTION_TOKEN_ALIGNMENT         = PASS 176336/176336
POLICY_9D_FINITE                      = PASS 176336/176336
GENERATION_AND_SCORE_PARITY           = PASS 176336/176336
PRIVILEGED_REQUIRED_FIELDS            = PASS 176336/176336
PRIVILEGED_PHYSICS_SCHEMA             = PASS_TASK_CONDITIONAL_SCHEMA
FORMAL_TRAINING_AUTHORIZED            = false
FORMAL_ATTACK_AUTHORIZED              = false
```

## Sealed server roots

Use the `_02` roots below. The `_01` roots are retained as superseded audit
evidence; `_02` records the corrected task-conditional object-state schema.

```text
POLICY_ROOT
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_02
SHA256SUMS SHA = d0a534da50df1f0e341c06d649cd8f52b89707d50b88ff56e02bb2b234451123

PRIVILEGED_AUDIT_ROOT
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1_20260718_02
SHA256SUMS SHA = abd25de6dcf18d5c6ca198f49d337e8598b46317107eb5940e1fd7322709bf08
```

Both roots passed `sha256sum -c SHA256SUMS` and
`sha256sum -c SHA256SUMS.sha256`.

## Input closure

```text
registry CSV SHA                  = 09f71b3a9b8250c80735382ba5deab6dbcadfa21b645e4a981eefb114b236af5
registry root SHA256SUMS SHA      = b42cc794bcf9e837106ecb54f99d70d85e2f47f8d44b1ce08862aebf9ef892f7
S1 root SHA256SUMS SHA             = db5ea2c8a4a24bd50e032e44f4cb54089d131b7497daf4aa731d625b536cb93f
raw asset audit root SHA256SUMS SHA= 325827de58fb637ba8da18b96e2fb563ad074f56c55db833c63ac48cd69c5da3
FIT artifact index SHA             = 19f7d5de804b2ce3abacd87b19c5bbb712a599bff922e041e4d5e05df717dc86
binding protocol SHA               = e8b98924adb5136240abc39d780bbbf9fa6347b4dc896e3c06eb42637cf31b4d
```

The auditor revalidated each artifact's recursive checksum closure and exact
file set before reading the three aligned step streams.

## Policy Student root

The sealed derived root contains only policy telemetry, identity/step binding,
and audit metadata. It does not contain `object_state`, contact pairs, worker
fields, collector provenance as model input, model paths, or attack outcomes.

Every FIT step has:

- a finite 9D policy-intent vector;
- finite open/close probability and token telemetry;
- token IDs equal to the corresponding `step_records.jsonl` action token IDs;
- score-head top tokens equal to the action token IDs;
- `generation_passes_per_step = 1`;
- `single_generation_parity_pass = true`;
- `score_adapter_parity_pass = true`.

This makes Official policy-intent a usable V5-B input candidate, subject to
the normal V5 model/protocol freeze. It does not authorize training.

## Privileged Physics source

All required physics fields are present and finite at every FIT step:

```text
object_state
mujoco_contact_pairs
contact_count
contact_capture_valid
robot0_eef_pos / robot0_eef_quat
robot0_gripper_qpos
eef_feature_pos / eef_alias_valid
```

`object_state` is not one global vector: observed dimensions are 28, 56, 70,
98, and 112. They are constant within each of the 40 tasks, so the audit passes
as `PASS_TASK_CONDITIONAL_SCHEMA`, not as a globally uniform schema. A Physics
Teacher must use a task-bound decoder/field map before deriving utility labels.
The privileged root is Teacher-only and cannot be consumed by Student.

## Current gates

```text
RAW_ASSET_AUDIT                    = PASS METADATA ONLY
POLICY_INTENT_BINDING              = PASS 800/800
PRIVILEGED_FIELD_AUDIT             = PASS TASK-CONDITIONAL SCHEMA
PHYSICS_TEACHER_V2                 = NOT BUILT
V5-B SMOKE                         = NOT STARTED
V5-C / V5-D                        = HOLD pending C2F trajectory binding
V5-A FULL TRAINING                 = HOLD
FIT-DEV / CAL / CHECK              = NOT READ
ATTACK                             = NOT STARTED
SOURCE ARTIFACT MUTATION           = 0
```

The C2F RGB root remains an unbound parallel source. No RGB was copied or
joined in this audit. The next safe code step is the task-conditional Physics
Teacher decoder and matched V5-B development smoke; the C2F trajectory-binding
audit can proceed separately.
