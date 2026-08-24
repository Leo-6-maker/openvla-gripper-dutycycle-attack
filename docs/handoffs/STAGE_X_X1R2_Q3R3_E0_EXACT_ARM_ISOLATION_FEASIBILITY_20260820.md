# STAGE X X1R2 Q3R3 E0 exact-arm isolation feasibility audit — 2026-08-20

## Decision

Status: `HOLD_Q3R3_E0_FEASIBILITY_EVIDENCE_INSUFFICIENT`

This is Case C under the superseding PR #135 instruction. The four D2 TRUE
receipts establish a repeated frozen-selector failure
`STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE`, but they do not seal the
candidate-level token, arm, gripper, margin, or pixel-budget values needed to
decide whether the frozen method is feasible.

The result is therefore neither a physical negative nor a feasibility-boundary
claim. No strict attack-efficacy escalation is authorized from this audit.

## Takeover readback

The machine-readable takeover record is:

`reports/STAGE_X_X1R2_Q3R3_TAKEOVER_READBACK_V1.json`

Live PR #135 was rechecked as OPEN / DRAFT / MERGEABLE with CLEAN merge state,
HEAD `9ae3ba4fe1af4f8064cc76d29a54ae1e6b26451a`, and tree
`a0682095f2df27cbe628d4f3ff987b5741fca2a`. The three CPU checks
`source-registry`, `detector-v5-cpu`, and `stageb-cpu` are SUCCESS.

The strongest positive result remains X0's sealed physical duty-cycle
mechanism evidence. VI-B2, VII, VIII, and IX remain their sealed negative or
non-promotion conclusions. X1/T0/V1/RCA1 remain diagnostic or blocked and do
not provide a valid scientific PGD cohort. Q3R3-A/B/C established the branch
estimand and branch-replay engineering contract. D2 remains engineering-only.

## Frozen evidence checked

- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- D2 source: `b86006d95b20b82b1dbdf91d159e8269c112b6fa`
- D2 tree: `1e10664e02541fac5287c36e6514f3d5df2c71eb`
- D2 protocol SHA-256: `e2d53e32d4091cf5b8abc233fdc38a78b877fed52452869b5d3f4799cde8db94`
- C2 root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_c2_20260820`
- C2 root-seal SHA-256: `fe23991c94852bc65269d740a6c67f782350dc61e600f766fdfbf85461abbdcc`
- D2 root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_d2_20260820`
- D2 aggregate SHA-256: `daa74f4279ac4672adc7b233b2c7c72ea444aa9e13f24c0d4c03b8c2fb6617ca`

Each of the four `TRUE_PGD_T5_ENGINEERING` directories contains only
`arm_receipt.json`. The four receipt hashes are recorded in the E0 JSON audit.
The receipts report zero materialized rows and zero attacked environment
steps. The D2 logs contain the selector traceback but no candidate values.

## Static candidate-flow result

The frozen source constructs the expected six-slot sequence:

`delta0 -> pgd_iteration_1 -> pgd_iteration_2 -> pgd_iteration_3 -> pgd_iteration_4 -> pgd_iteration_5`

The selector evaluates direct generated seven-token prefixes and requires all
of the following:

1. exact equality of arm token IDs `[0:6]`;
2. clean gripper is not native open;
3. candidate gripper token is in the frozen native-open set; and
4. candidate gripper token changes from clean.

The selector stores its candidate audit in `last_attack_diagnostics` and on
exception places it in `execution_trace`. The D2 runner's failure serializer
does not persist either `execution_trace` or `attack_contract_diagnostics`.
This is an evidence-materialization gap; it is not evidence that the selector
wrongly rejected a concrete candidate.

The complete matrix is:

- JSON: `reports/STAGE_X_X1R2_Q3R3_E0_EXACT_ARM_ISOLATION_FEASIBILITY_AUDIT_V1.json`
- CSV: `reports/STAGE_X_X1R2_Q3R3_E0_CANDIDATE_MATRIX_V1.csv`

All 24 candidate slots are classified
`NOT_IDENTIFIABLE_FROM_SEALED_EVIDENCE`. No slot can be assigned one of the
four physical/structural categories without rerunning forbidden inference or
recovering a sealed candidate artifact that is not present in the D2 roots.

## Protected and resource boundary

`Eval160=UNREAD`, protected reads are zero, V_phys reads are zero, physical
interventions are zero, and attack-outcome reads are zero. E0 used only local
static parsing, hashing, source inspection, and durable-root inventory.

The worker resource contract is recorded as strictly more than 20,480 MiB
free per GPU, at most one project worker per physical GPU, and at most eight
GPUs. A live read-only snapshot showed all eight GPUs above that free-memory
threshold, but several have active foreign processes; no worker was mounted,
and this E0 gate does not authorize a worker.

## Required next decision

Owner review is required for the evidence gap. Until that review changes the
authority boundary, do not:

- launch GPU/model/simulator work;
- rerun or replace the four consumed D2 fixtures;
- change epsilon, steps, arm-preservation weight, target, or gate;
- allow arm-token drift or enter RAND/SHUFFLED/random-time branches; or
- enter R0/R1/R2, physical/V_phys/outcome, Eval160, or protected evaluation.

No source or validator patch was applied in E0. The strict efficacy line
remains held for insufficient candidate-level evidence, not closed as a
method-level feasibility negative.
