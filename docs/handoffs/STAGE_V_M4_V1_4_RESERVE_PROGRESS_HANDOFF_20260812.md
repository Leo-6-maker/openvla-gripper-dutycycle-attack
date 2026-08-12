# Stage V M4 V1.4 reserve progress handoff

Status: `RESERVE_A_B_PASS_FORMAL_M4_BLOCKED`

Updated: `2026-08-12T12:36:26Z`

This is the takeover point for a new Codex window. It records the live server
state, the exact source bindings, and the boundary between clean preflight
evidence and still-unstarted causal-label production.

## Takeover first

- Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- PR: [#111](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/111)
- PR state: `OPEN / DRAFT / UNMERGED`
- Branch: `codex/stage-v-resource-contract-20260810`
- HEAD: `92ff4c2d43fb327b661552d483db6e1298c19833`
- Tree: `72707afd011bb67fc5f759de3ab3cad2c517faa1`
- A800 SSH alias: `dty-server` (do not use the unrelated `vla` alias)
- Remote worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-v7-terminal-gate-fix-20260812`
- Official Python: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`
- Do not resume or modify the sealed V1.3.4 root. Do not launch formal M4,
  label generation, Teacher, Student, scheduler, timing, or VIS until the
  stable formal corridor is replenished and re-authorized.

## Current gate truth

| Gate | State | Meaning |
|---|---|---|
| M0 static | PASS | Baseline static contract remains valid. |
| M1 runtime determinism | PASS_CLASSIFIED | Rendered RGB can diverge; action/physical equivalence held. |
| M2 execution equivalence | PASS 28/28 | Physical execution contract held. |
| M3 V6 runtime | COMPLETE_VALID | Runtime completion evidence exists. |
| M3 V6 qualification | FAIL_SEALED | Sealed; do not resume/rerun/relax. |
| V1.3.4 M3.5 measurement | HOLD | Exact re-rendered policy-input binding failed; zero valid V_t rows. |
| V1.4 implementation/static/auth | PASS | Observation-snapshot and reserve-manifest plumbing is frozen and authorized. |
| V7 fresh qualification | PASS | Formal V7 receipt is PASS; 40 parents, split 24/8/8. |
| Formal M4 corridor | HOLD | A/B reconciliation has only 29 stable identities out of the required 40. |
| Reserve M4 preflight A/B | PASS | Both 7-parent reserve runs completed; 3 stable corridor PASS, 4 stable clean failures. |
| M4 labels | NOT STARTED | No counterfactual outcome/label rows exist. |
| M5 Teacher / M6 Student / M7 scheduler / M8 timing / M9 VIS | BLOCKED | Correctly downstream of valid M4 labels. |

The reserve runs are clean-only, outcome-blind qualification diagnostics. Their
`M4_CORRIDOR_PREFLIGHT.json` files are not V_t labels and must not be promoted
to labels.

## Source and frozen input bindings

The V1.4 runtime source binding is:

```text
commit = 3bcc42850f69e35ac006dc023771384a3ab7e19c
tree   = 2c8e1fc8f5d1d3af37b257f6c6061b8fa9cfa0e0
```

The dispatcher/doc branch is HEAD `92ff4c2d` / tree `72707afd`. The distinction
is intentional: the child M4 runner validates against the V1.4 protocol source
binding above, while the queue/dispatcher provenance is the branch HEAD.

Frozen files:

- Protocol: `configs/STAGE_V_M4_CORRIDOR_RESERVE_QUALIFICATION_PROTOCOL_V1.json`
  - SHA256 `cf747c2c8850b19ff64f462d5b6f0934a9bfa7be43e7e61614f448680e350024`
- Reserve parent manifest:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_RESERVE_MANIFEST_V1_20260812T121054Z/STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1.json`
  - schema `STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1`
  - status `FROZEN`
  - SHA256 `d883d120d4daf4b7481db3a9c23b87269c6a72119661a7593ee6da82eb869849`
- Reserve runtime authorization:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_RESERVE_RUNTIME_AUTHORIZATION_V1_20260812T122000Z/STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION.json`
  - status `PASS`
  - SHA256 `0ef31a7c406f23829778a91313de4fbfb62bee6bfbb89ab1bde9e54bb67c095f`
- Reserve static audit: `reports/STAGE_V_M4_CORRIDOR_RESERVE_STATIC_AUDIT_V1.json`
  - status `PASS_STATIC_DESIGN_ONLY`
  - SHA256 `e04e1d565d7b574c52544a69bc03d84d187b64c8dabb2debecae7e23c9d31c9c`
- V7 formal receipt SHA256 `36af16ea9c79a7a6a6537d5be2de3b18a69fed2cce523b05c368009c99709118`
- V7 control qualification rows SHA256 `75beaeee27f91c5410d105c6fbec41655123a669cf00f91b81b42ca758fec72b`
- Exposure union:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V5_POST_M35_20260811T162000Z/STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V5.json`
  - SHA256 `2ef448f172dc2c8fa2304a6772a7024d76391ed2f6cbca1fc8e79b8a6425dd5a`
  - reserve A/B binding status `PASS`, overlap `0`, excluded identities `53`

Owner authorization is the explicit 2026-08-12 instruction permitting GPUs
0-7, foreign workloads when free memory is above 20480 MiB, and no project
interference. All runs record protected/Eval160/attack counters as zero.

## Reserve parent set

The frozen reserve manifest has seven V7-qualified, formally unselected
identities:

```text
libero_goal/task_07/state_22
libero_goal/task_07/state_32
libero_goal/task_07/state_35
libero_goal/task_08/state_40
libero_object/task_03/state_28
libero_object/task_06/state_27
libero_object/task_09/state_22
```

There are seven tasks, not eight. Eight GPUs are worker capacity; they do not
create additional parent identities. The dispatcher starts up to eight workers
and each worker repeatedly calls `claim_task()`. One worker is therefore
expected to become `IDLE`/`QUEUE_DRAINED` immediately, and other workers become
idle as their claimed parent finishes. `dynamic_gpu_claim_count` is a lifetime
lease/process-claim count, not a simultaneous GPU count.

## Live reserve evidence

### A2 (invalid launch, retained for provenance)

Root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_RESERVE_PREFLIGHT_A2_20260812T122500Z`

All seven children failed before simulator initialization with
`M4_CORRIDOR_SOURCE_BINDING_MISMATCH`. The launch command passed a 39-character
tree value; the protocol requires the full 40-character tree shown above.
No `M4_CORRIDOR_PREFLIGHT.json`, label, outcome, protected read, or attack
rollout was produced. This root is not consumable and must not be rerun.

### A3 (independent reserve replicate A)

Root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_RESERVE_PREFLIGHT_A3_20260812T123000Z`

- Dispatcher: `PASS`, `planned_parents=7`, `completed_parents=7`
- Approved GPUs: `[0,1,2,3,4,5,6,7]`
- Resource mode: `MODE_B_THROUGHPUT_SCIENCE`, min free `20480 MiB`
- Exposure binding: `PASS`, overlap `0`
- Protected/Eval160/VIS attack counters: `0/0/0`
- Lifetime dynamic GPU claims: `115`
- Classification: `PASS=3`, `CLEAN_FAILURE=4`
- The three PASS parents had corridor counts `56, 73, 57` and 24 planned probes.

### B3 (independent reserve replicate B)

Root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_RESERVE_PREFLIGHT_B3_20260812T123500Z`

- Dispatcher: `PASS`, completed at `2026-08-12T12:36:26Z`, `7/7`
- Approved GPUs: `[0,1,2,3,4,5,6,7]`
- Exposure binding: `PASS`, overlap `0`
- Protected/Eval160/VIS attack counters: `0/0/0`
- Lifetime dynamic GPU claims: `99`
- Classification: `PASS=3`, `CLEAN_FAILURE=4`
- A/B key set, status, reason, corridor count, and probe count are identical;
  reconciliation differences are empty.

The four reserve `CLEAN_FAILURE` rows are deterministic fixture-taxonomy
abstentions (goal fixtures without an `in`/`on` source object), not renderer
or GPU failures. The three stable PASS keys are the three libero_object
reserve identities listed above.

## Formal M4 status

Formal preflight roots remain:

- A: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_PREFLIGHT_A_20260812T123000Z`
- B: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_CORRIDOR_PREFLIGHT_B_20260812T125000Z`
- Reconciliation: `reports/STAGE_V_M4_CORRIDOR_AB_RECONCILIATION_V1.json`

Reconciliation status is `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`: only 29 of
the required 40 formal identities are stable across A/B
(`libero_10=9`, `libero_goal=4`, `libero_object=10`, `libero_spatial=6`).
The reserve A/B result adds three stable PASS identities, but it does not
retroactively enlarge or replace the frozen formal 40-parent manifest. A new
disjoint candidate manifest, protocol, static audit, and runtime authorization
are required before formal M4 can be launched. No M4 labels exist.

## Required next actions

1. Keep V1.3.4, the formal V7 split, formal M4 roots, and A2 immutable.
2. Reconcile the three stable reserve PASS identities with the formal corridor;
   do not silently substitute them into the formal split.
3. Replenish fresh, exposure-disjoint V7-qualified identities until the chosen
   formal corridor satisfies the frozen count/lineage requirement.
4. Freeze a new manifest/protocol, run static audit, issue runtime
   authorization, then run independent clean A/B preflight again.
5. Only after that PASS may matched-action formal M4 produce V_phys/V_t labels.
   Teacher, Student, scheduler, timing, and VIS remain downstream and
   unstarted.

## Review checklist for the next Codex window

```bash
git show --stat --oneline 92ff4c2d43fb327b661552d483db6e1298c19833
git status --short
/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python --version
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

Verify the A3/B3 `DISPATCHER_COMPLETE.json`, all 14 reserve preflight
artifacts, the A/B equality, and the zero protected counters before making any
new scientific claim. The correct claim boundary is: exact causal snapshot
plumbing and clean reserve repeatability are demonstrated; a formal V_t map,
prevalence, learnability, timing utility, and attack utility are still
untested.

