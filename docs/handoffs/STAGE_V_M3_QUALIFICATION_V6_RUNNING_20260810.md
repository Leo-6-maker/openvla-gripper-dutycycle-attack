# Stage V M3 fresh qualification V6 handoff

Status: `RUNNING`

## Binding

- Branch: `codex/stage-v-resource-contract-20260810`
- Source commit: `857972a70448c645f1c3d1d8929ddd69aaea2187`
- Source tree: `45e95a60c4d9735f863873b4404937551b68ac50`
- Runtime root: `/mnt/sdc/dty_user/stage-v-resource-contract-c5563cf8`
- Protocol: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_PROTOCOL_857972a7_20260810T013514Z/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_PROTOCOL_V6.json`
- Candidate manifest: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_CANDIDATE_V6_20260810T013514Z.json`
- Candidate scale: `240` identities, `60/suite`; target `40` qualified parents, `10/suite`.

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

The next autonomous action is an independent audit of the completed V6 report. Only a PASS report and PASS audit may create the formal 40-parent manifest and enter M4; a valid FAIL remains sealed without retry-to-pass.
