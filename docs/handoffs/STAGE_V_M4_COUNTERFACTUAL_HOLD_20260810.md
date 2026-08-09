# Stage V M4 counterfactual-map handoff

Status: `HOLD`

## Binding

- Branch: `codex/stage-v-resource-contract-20260810`
- HEAD: `bd3020721cd6a3d97a562597c47c30d8ff6b454a`
- Tree: `1e46adffdfa161db2143f8801565bf0bfac73582`
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

This is a scientific clean-repeatability failure, not an OOM or GPU3 resource failure. The parent and partial map artifacts are retained; no parent was deleted, overwritten, or rerun-to-pass. M5 Teacher, M6 Student, M7 scheduler, M8 timing, and M9 VIS were not started.

## Resource evidence

GPU3 foreign PID `964381` (`huanzze`, Isaac GR00T) remained alive and unmodified. The M4 throughput run admitted GPU3 with approximately `74508 MiB` free. After dispatcher failure, only the exact project child process groups were terminated; GPU leases for the dead project workers were recovered as stale. No foreign process was signaled.

## Code change in this branch

`run_stage_v_dynamic_dispatcher.py` now reaps registered child process groups before recovering same-run stale leases. Linux self-check passed; `py_compile` and `git diff --check` passed. Local `pytest` was unavailable (`No module named pytest`). This change was not used to modify or repair the M4 root.

## Claims

Allowed: M1/RB1 execution stability under the seven-GPU primary clean cohort; M4 is a reproducible HOLD caused by clean repeatability failure.

Forbidden: a valid 40-parent V_t map; V-Teacher/Student performance; Student-Time causal utility; VIS selectivity; any protected-Eval160 claim.

Before any future science stage, recompute exposure from the old exclusion manifest plus all M4 attempted identities. Continuing would require a new prospective protocol/selection decision; the frozen M4 result must not be made to pass by replacement or rerun.
