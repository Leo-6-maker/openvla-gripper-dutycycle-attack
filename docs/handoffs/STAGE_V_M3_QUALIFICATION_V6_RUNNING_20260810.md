# Stage V M3 fresh qualification V6 handoff

Status: `SUPERSEDED — V6 COMPLETE_VALID / QUALIFICATION FAIL`

> This file preserves the historical in-run snapshot below. The current
> closeout is [STAGE_V_M3_5_V6_FORENSIC_CLOSEOUT_20260810.md](STAGE_V_M3_5_V6_FORENSIC_CLOSEOUT_20260810.md).
> V6 finished with `480/480 DONE_VALID`, frozen qualification `FAIL`, and
> independent audit `FAIL`; no V6 rerun-to-pass was performed.

This handoff is a live progress snapshot for draft review. It does not promote
the qualification run, the historical M4 root, or any downstream detector
stage to PASS.

## Binding

- Branch: `codex/stage-v-resource-contract-20260810`
- Source commit: `857972a70448c645f1c3d1d8929ddd69aaea2187`
- Source tree: `45e95a60c4d9735f863873b4404937551b68ac50`
- Runtime root: `/mnt/sdc/dty_user/stage-v-resource-contract-c5563cf8`
- Protocol: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_PROTOCOL_857972a7_20260810T013514Z/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_PROTOCOL_V6.json`
- Candidate manifest: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_CANDIDATE_V6_20260810T013514Z.json`
- Candidate scale: `240` identities, `60/suite`; target `40` qualified parents, `10/suite`.

## Live V6 queue snapshot

Read-only snapshot: `2026-08-10 04:42 UTC`.

| Item | Value |
|---|---:|
| Registered qualification tasks | `480` (`240` candidates × `A/B`) |
| `DONE_VALID` | `189` |
| `LEASED` | `8` |
| `PENDING` | `283` |
| Controller | PID `1963943`, alive |
| Latest queue event | `TASK_COMMITTED`, `2026-08-10T04:41:49.815618+00:00` |
| Final report / rows / independent audit | not yet present |
| Formal/science parent manifests | not yet present |

The observed completion rate is approximately `64.3 tasks/hour` over the full
window and `81.3 tasks/hour` over the latest 20 commits. With `291` tasks not
yet terminal, the current evidence-based ETA for V6 completion is roughly
`3.6–4.5 hours`, followed by a short CPU-side audit and split-freeze step.
This is an estimate, not a scientific result or a deadline.

## Atomic scheduling evidence

- V6 uses atomic queue claims and one GPU lease per project worker.
- The current V6 unit is one atomic `A` or `B` arm task; both arms of one
  parent remain pinned by the frozen parent/GPU affinity rule.
- Eight workers are active across the available GPUs, with at most one project
  worker per GPU. GPU3 is admitted under `MODE_B_THROUGHPUT_SCIENCE` when it
  satisfies the free-memory rule; its foreign workload remains untouched.
- M4 will use the stricter parent-level atomic unit: one parent bundle owns all
  `72` registered branch rows on one GPU.

## Freshness and exposure

- G10 held-out source: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G10_TEST_MANIFEST.json` (`1200` identities).
- Recomputed exposure manifest: `STAGE_V_EXPOSURE_UNION_V3_20260810T013514Z.json`, `53` excluded identities.
- Clean-attempt exclusion: `STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1_20260810T013514Z.json`, `117` prior clean/control identities.
- Candidate intersection with either exclusion set: `0`.
- Selection: frozen SHA rank; no clean or vulnerability outcome was read for selection.

## Runtime contract

- Python: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`
- Attention binding: `OPENVLA_ATTN_IMPLEMENTATION=eager`
- Resource mode: `MODE_B_THROUGHPUT_SCIENCE`
- Admission: `>=20480 MiB` free, partial fleet allowed, one project worker/GPU, foreign workloads allowed.
- Launch GPUs: `0,1,2,3,4,5,6,7`.
- GPU3 PID `964381` (`huanzze`, Isaac GR00T) was recorded and left untouched.

## Boundaries

- `Eval160 reads = 0`
- `protected_eval_reads = 0`
- `attack_rollouts = 0`
- `VIS/PGD rollouts = 0`
- Historical roots and valid V5 results were not modified or rerun.

## Complete experimental ledger

| Stage | Current status | Evidence / interpretation |
|---|---|---|
| `M0_V2_1_1_STATIC` | `PASS` | Static correctness closure and focused regression passed before runtime. |
| `M1_RUNTIME_DETERMINISM` | `PASS` | V2.2 primary clean cohort: `7` GPUs `[0,1,2,4,5,6,7]`; GPU3 contention was recorded, not hidden. |
| `M2_EXECUTION_EQUIVALENCE` | `PASS` | Prospective RB1-V2 audit: `28` causal-equivalence pairs, protected reads `0`. |
| `M3_FRESH_QUALIFICATION` | `RUNNING` | V6 queue snapshot above; no final PASS claim yet. |
| `M4_COUNTERFACTUAL_V_MAP` | `HOLD / NOT STARTED` | Historical R2A root remains immutable HOLD (`6/40` parents, `432/2880` rows); new M4 waits for V6 PASS plus independent audit. |
| `M5_V_TEACHER` | `NOT_STARTED` | No V-Teacher training or TEST read. |
| `M6_V_STUDENT` | `NOT_STARTED` | No causal Student training or TEST read. |
| `M7_EVENT_SCHEDULER` | `NOT_STARTED` | No scheduler tuning on Stage O. |
| `M8_STUDENT_TIME_CAUSAL` | `NOT_STARTED` | No Student-Time or Random-Time direct-OPEN evaluation. |
| `M9_VIS_SELECTIVITY` | `NOT_STARTED` | Correctly conditional on a positive timing gate; no VIS/PGD rollout. |
| `M10_FINAL_FREEZE` | `NOT_STARTED` | No final model/protocol freeze exists. |
| `M11_READY_FOR_PROTECTED_EVAL` | `NOT_STARTED` | Protected Eval160 remains untouched. |

## Review gates and next action

1. Let the already-started V6 controller finish without changing its frozen
   protocol, source binding, candidate pool, or retry policy.
2. Read and independently audit the final report, rows, formal manifest, and
   science manifest. A valid FAIL is sealed; it is never rerun to pass.
3. Only after report and audit PASS, freeze the parent `TRAIN/VAL/TEST` split
   and create a fresh M4 source/root with the `40 × 72 = 2880` parent-level
   atomic design.

The draft PR is intentionally not a scientific qualification approval. GPT
review should specifically check the live-count interpretation, the V6
arm-level versus M4 parent-level atomicity, the exposure/split boundaries, and
the requirement that all downstream stages remain blocked until the stated
PASS gates exist.

## Milestones

| Milestone | Status |
|---|---|
| `M0_V2_1_1_STATIC` | `PASS` |
| `M1_RUNTIME_DETERMINISM` | `PASS` — primary clean cohort 7 GPUs |
| `M2_EXECUTION_EQUIVALENCE` | `PASS` — RB1-V2 |
| `M3_FRESH_QUALIFICATION` | `RUNNING` |
| `M4_COUNTERFACTUAL_V_MAP` | `NOT_STARTED` |
| `M5_V_TEACHER` | `NOT_STARTED` |
| `M6_V_STUDENT` | `NOT_STARTED` |
| `M7_EVENT_SCHEDULER` | `NOT_STARTED` |
| `M8_STUDENT_TIME_CAUSAL` | `NOT_STARTED` |
| `M9_VIS_SELECTIVITY` | `NOT_STARTED` |
| `M10_FINAL_FREEZE` | `NOT_STARTED` |
| `M11_READY_FOR_PROTECTED_EVAL` | `NOT_STARTED` |

The next autonomous action after V6 controller exit is an independent audit of
the completed V6 report. Only a PASS report and PASS audit may create the
formal 40-parent manifest and enter M4; a valid FAIL remains sealed without
retry-to-pass.
