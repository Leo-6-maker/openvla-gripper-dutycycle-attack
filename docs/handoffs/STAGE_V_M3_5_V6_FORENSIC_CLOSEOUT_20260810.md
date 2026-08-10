# Stage V M3.5 / V6 forensic closeout — 2026-08-10

## Current decision

V6 is sealed as a valid completed negative qualification outcome:

```text
M3_V6_RUNTIME_EXECUTION      = COMPLETE_VALID
M3_V6_QUALIFICATION_REPORT   = FAIL
M3_V6_INDEPENDENT_AUDIT      = FAIL
M3_FRESH_QUALIFICATION       = HOLD
M3.5_LABEL_VALIDATION        = REQUIRED / IN PROGRESS
M4_FRESH_V_MAP               = BLOCKED / NOT_STARTED
```

The V6 root and its report, rows, audit, queue database, source binding, and
candidate pool are immutable. No V6 rerun-to-pass was performed.

## Server completion check

- Host: `pm-364c0001` via SSH alias `dty`.
- Python binding: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`.
- V6 run: `STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_20260810_G10_V2`.
- Queue: `480/480 DONE_VALID`; run state `COMPLETE`; controller PID `1963943` absent.
- GPUs 0–7 had no project compute process after completion.
- GPU3 foreign workload was not killed, paused, reniced, migrated, signaled, or
  modified.
- Protected counters remained zero: Eval160 reads, protected reads, attack
  rollouts, and VIS/PGD rollouts.

## Read-only forensic result

Source artifact:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_V6_FORENSIC_CLOSEOUT_V1_20260810T093233Z/V6_FORENSIC_MATRIX.json`

SHA256: `9c926db007331fa0b4bea25aaddf3dda61a18db7e30bf437dd71bbf0062c0ad7`

| Suite | A/B engineering-valid | A/B both clean-success | Hash equal | Hash unequal |
|---|---:|---:|---:|---:|
| `libero_10` | 60/60 | 27 | 23 | 4 |
| `libero_goal` | 60/60 | 46 | 0 | 46 |
| `libero_object` | 60/60 | 42 | 42 | 0 |
| `libero_spatial` | 60/60 | 48 | 36 | 12 |
| **total** | **240/240** | **163** | **101** | **62** |

The key result is `libero_goal = 46/60` A/B both-clean-success, while all
`46/46` have unequal frozen `terminal_state_sha256`. Therefore the producer
`0/10` Goal selection is not evidence that Goal has zero clean-success yield;
it is dominated by the V6 exact terminal-hash gate. This does not change the
V6 FAIL decision.

`TASK_FAILURE` is engineering-complete when `exit_code == 0` and the artifact
checks pass. V6 currently labels it as `NOT_COMPLETE` as well as clean failure;
the prospective contract separates those layers.

## M3.5 prospective contract

The V1 files below are historical drafts and are not runtime authority. GPT's
static review identified P0 gaps; the prospective V1.1 patch is now the only
candidate contract, and it remains runtime-unauthorized until a later explicit
audit/authorization boundary:

- `configs/STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1.json`
- `configs/STAGE_V_FRESH_SCIENCE_PARENT_QUALIFICATION_CONTRACT_V1.json`
- `configs/STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_1.json`
- `configs/STAGE_V_FRESH_SCIENCE_PARENT_QUALIFICATION_CONTRACT_V1_1.json`

The V1.1 patch binds the latest 53-parent exposure union and the mechanically
computed 357-parent cumulative clean-attempt union. Its exact freeze receipt is
`docs/handoffs/STAGE_V_M3_5_V1_1_PATCH_20260810.md`.

The contract requires treatment compliance receipts for canonical raw
`1.0 -> env -1.0 OPEN`, sufficient causal-state binding, matched-control arm
isolation, `CONTROL x3 / T3,T5,T10 x3` diagnostics, a physical truth table with
explicit abstains, and per-probe rather than global terminal/horizon gating.

Preliminary read-only historical gap audit:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_5_HISTORICAL_EXPOSED_GAP_AUDIT_V1_20260810T094153Z/M3_5_HISTORICAL_EXPOSED_GAP_AUDIT.json`

SHA256: `d8b2c50ab66baa1f5623d280416174efbcd448997c415c779e419b76e757c444`

It inspected 6 historical exposed parents / 432 branch rows. Treatment
compliance, full simulator-state binding, policy-input binding, surgical arm
isolation, and repeated outcome-stability receipts were all `0/432`. Existing
prefix replay and qpos/object/contact comparisons are useful diagnostics but do
not satisfy the M3.5 gates.

The historical exposure inventory is read-only input only:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/runner_binding_forensics_c80eb290_20260808T013003Z/COUNTERFACTUAL_EXPOSURE_EXCLUSION_V1.json`

It contains 50 exposed parent identities, records `branch_results_read=false`,
and has zero protected reads. V6's 240 clean-attempt identities are also
excluded from future V7 selection; neither set may be used to tune outcomes or
select positive parents.

## Next legal gate

The V1.1 static contract audit is complete, but no M3.5 diagnostic runtime is
authorized by this handoff. A later authorized M3.5 run must produce its
validation receipt without changing the frozen contract SHA. Then create a new
V7 source/root/salt/candidate pool; only a fresh V7 PASS may create the formal
40-parent split and unlock M4.
