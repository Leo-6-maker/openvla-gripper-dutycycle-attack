# Stage V M4 counterfactual-map handoff

Status: `HOLD`

## Binding

- Branch: `codex/stage-v-resource-contract-20260810`
- HEAD: `5d88c3ca1c370cff386be1f5ceb68eb7427db6ad`
- Tree: `1883552d7b2b1f8e1f2873a38a1c720013fedbb7`
- M4 runtime source: `f903f942e04e92f67308d35bb0259c477d9b23e4`
- M4 runtime tree: `3cdb1af4893d9873e4954e95d8c33650911344eb`
- M4 root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_R2A_COUNTERFACTUAL_MAP_f903f942_20260809T204000Z`
- M4 protocol scale: `40 parents × 72 registered rows = 2880`

## Upstream gates

- M1 V2.2: `PASS`, primary clean cohort `7` GPUs: `[0,1,2,4,5,6,7]`.
- RB1-V2: `PASS`, causal execution equivalence; protected reads `0`.
- Fresh qualification: clean union qualified counts `11/11/15/12` by suite; no attack/VIS/protected reads.

## M4 evidence

- Independent audit: `FAIL`.
- Completed/audited parents: `6/40`.
- Registered branch rows present: `432/2880`.
- Missing identities: `34`.
- Failed parent: `libero_10/task_03/state_48` on GPU7.
- Failure class: `clean_success=false` with a complete valid runtime artifact; frozen interpretation is `CLEAN_REPEATABILITY_FAIL_no_retry`.
- Dispatcher marker: `RuntimeError:WORKER_EXIT:1543639:1`.
- Eval160/protected reads: `0`.
- Exposure binding audit: `FAIL`; the 40-parent formal manifest intersects the prior exclusion manifest in `17/40` identities, by suite `libero_10=7`, `libero_goal=4`, `libero_object=4`, `libero_spatial=2`.
- Exposure input: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/runner_binding_forensics_c80eb290_20260808T013003Z/COUNTERFACTUAL_EXPOSURE_EXCLUSION_V1.json`, SHA256 `d980340275e721cdee2d0ebc3c40a3c4e3805c9ccb54e22cf0998ed370e09689`.
- Overlap identities: `libero_10/task_00/state_47`, `libero_10/task_00/state_49`, `libero_10/task_01/state_47`, `libero_10/task_01/state_49`, `libero_10/task_03/state_47`, `libero_10/task_03/state_48`, `libero_10/task_05/state_49`, `libero_goal/task_01/state_48`, `libero_goal/task_02/state_47`, `libero_goal/task_04/state_47`, `libero_goal/task_04/state_48`, `libero_object/task_00/state_48`, `libero_object/task_01/state_47`, `libero_object/task_04/state_47`, `libero_object/task_04/state_49`, `libero_spatial/task_00/state_48`, `libero_spatial/task_02/state_47`.

This is both a scientific clean-repeatability failure and an identity-exposure gate failure, not an OOM or GPU3 resource failure. The parent and partial map artifacts are retained; no parent was deleted, overwritten, or rerun-to-pass. M5 Teacher, M6 Student, M7 scheduler, M8 timing, and M9 VIS were not started.

## Resource evidence

GPU3 foreign PID `964381` (`huanzze`, Isaac GR00T) remained alive and unmodified. The M4 throughput run admitted GPU3 with approximately `74508 MiB` free. After dispatcher failure, only the exact project child process groups were terminated; GPU leases for the dead project workers were recovered as stale. No foreign process was signaled.

## Code change in this branch

`run_stage_v_dynamic_dispatcher.py` now reaps registered child process groups before recovering same-run stale leases. The prospective resource contract now requires a SHA-bound exposure manifest for MODE_B, accepts partial eligible fleets, makes the independent auditor enforce the same binding, and filters future formal selection against that ledger. Linux self-check passed; `py_compile` and `git diff --check` passed. Local `pytest` was unavailable (`No module named pytest`). These changes were not used to modify or repair the M4 root.

## Claims

Allowed: M1/RB1 execution stability under the seven-GPU primary clean cohort; M4 is a reproducible HOLD caused by clean repeatability failure plus exposure overlap.

Forbidden: a valid 40-parent V_t map; V-Teacher/Student performance; Student-Time causal utility; VIS selectivity; any protected-Eval160 claim.

Before any future science stage, publish a new SHA-bound exposure manifest formed from the old exclusion manifest plus all 14 M4 attempted identities, then select a disjoint parent set under the new prospective resource contract. The launcher now rejects missing or overlapping exposure bindings before workers start. Continuing requires a new prospective protocol/selection decision; the frozen M4 result must not be made to pass by replacement or rerun.
