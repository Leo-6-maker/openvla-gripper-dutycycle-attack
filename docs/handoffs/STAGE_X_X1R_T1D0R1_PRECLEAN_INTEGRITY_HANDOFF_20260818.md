# STAGE X X1R T1-D0R1 Pre-clean Integrity Handoff

Date: 2026-08-18
Status: `STATIC_ONLY_HOLD_PENDING_OFFICIAL_AUDIT`

## Scope

This handoff records the owner-authorized T1-D0R1 repair after the live PR
#126 review. It is limited to CPU/static metadata, Git-object reconciliation,
the prospective timing contract, and an outcome-blind pre-clean authority
freeze. It does not authorize model loading, inference, clean rollout,
simulator execution, `env.step`, PGD, physical intervention, outcome reads,
Eval160, or protected evaluation.

The historical PR #125 and PR #126 artifacts remain immutable records. This
successor restores the original D0 selection salt for source-driven
recomputation; it does not accept D0R as an execution authority and does not
use any new outcome information.

## Immutable input bindings

- Reviewed PR #126 HEAD: `aabd419942b7a5de27b2f11da90160d4a7affebc`
- Reviewed PR #126 tree: `f31aa395cbd082f921b19d8b3c6ed50dcefd26f7`
- PR #126 governing review comment: `5323186497`
- Supersession comment: `5323188393`
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- T1-D0 selection salt restored exactly:
  `STAGE_X_X1R_T1D0_PARENT_AUTHORITY_V1_20260818`
- Nominal design: 40 cells; executable fresh population: 39 parents;
  missing cell: `libero_goal/task_01`; replacement is forbidden.

The code-only pre-evidence source for this handoff is the branch commit
`165efe5018224ac1da97699d45bd2bda9130e9fb` with tree
`fa1721e881c80283a9532d87f4e8a5f2fee1ef10`. The evidence-producing commit is
intentionally not embedded in this handoff or its root seal, because doing so
would be self-referential. The live GitHub HEAD/tree must be reported after
publication as a separate binding.

## Required static closure

The audit must independently reconstruct the 1200-row G10 universe, all
exclusion sources, the exclusion union, fresh candidates, 40 design cells,
and the 39 selected parents. D0's selected list is comparison-only and is
checked only after derivation. D0R's salt/list is historical comparison-only.

The two prior physical-intervention directory names are explicitly mapped to
`libero_10/task_08/state_28`. Their physical-intervention semantics are
`NOT_IDENTIFIABLE`; no historical outcome is read. Alias-set invariance must
hold for union, fresh candidates, and selected parents.

The local Stage IX contract and native-token source must be reconciled from
Git objects across the c6a4c5a, 2881722b, and aabd419 refs. Stale D0 digest
claims must remain historical and non-authoritative; current bytes must not be
rewritten to fit a historical claim.

## Prospective timing and seed freeze

The timing origin is `NEW_PROSPECTIVE_PI_FREEZE_20260818`:

- `emit_step = t_emit`; `attack_start_step = t_emit`;
- attack window: `[t_emit, ..., t_emit+4]`, five inclusive steps;
- physical follow-up: `[t_emit+5, ..., t_emit+14]`, ten inclusive steps;
- legal horizon: `t_emit + 5 + 10 <= episode_length`;
- one-shot and `NO_EMIT` semantics are frozen;
- `prev_delta` is zeroed at parent/condition/episode/window boundaries and
  carried only within the same five-step window.

The pure contract lives at
`src/gripper_attack/stage_x_x1r_v2_schedule_contract.py`. The future clean
plan is hard-disabled and cannot be called by this gate.

Clean seeds, if a later gate is separately authorized, are derived only as
`uint32(first_8_hex(sha256(namespace + "|" + canonical_parent_key)))` with
namespace `STAGE_X_X1R_T1D0R1_CLEAN_SEED_V1`. No outcome field can influence
selection or seeding.

## Known fail-closed runtime boundary

The authority deliberately requires exact task-success and episode-horizon
path bindings before any future clean materialization. It also verifies the
declared frozen Student model-source digest against the current prospective
file and preserves any discrepancy as a HOLD. No current file hash is used to
retroactively identify historical provenance.

Expected terminal behavior for this static-only gate is therefore a concrete
HOLD if either runtime authority remains unresolved. A HOLD is evidence of an
unclosed authority boundary, not permission to infer, roll out, or attack.

## Authorization and protected boundary

At publication, all of the following remain false/zero:

```text
model_inference_authorized = false
clean_parent_materialization_authorized = false
pgd_authorized = false
env_step_authorized = false
physical_intervention_authorized = false
attack_outcome_authorized = false
protected_authorized = false
Eval160 = UNREAD
protected evaluation = UNREAD
```

The only legal next step after the new Draft PR is independent GPT/owner
review of the sealed static evidence. The next gate, if explicitly approved,
is `CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED`; T1-D1 is not authorized by
this handoff.

