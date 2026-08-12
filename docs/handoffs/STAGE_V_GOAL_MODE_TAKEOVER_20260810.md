# Stage V Goal Mode takeover handoff — 2026-08-10

Generated from a read-only reconciliation of the local worktree, live A800
server, GitHub PR, immutable experiment roots, and the Goal Mode master plan.
This handoff does not authorize protected Eval160 and does not promote any
unfinished stage.

## 1. Executive decision

```text
CURRENT_SCIENTIFIC_GATE       = HOLD_BEFORE_VALID_M3_5_RUNTIME
NEXT_LEGAL_WORK               = PROSPECTIVE_M3_5_PROTOCOL_REVISION_AND_REAUDIT
M1_RUNTIME_DETERMINISM        = PASS_CLASSIFIED_ON_7_GPU_CLEAN_COHORT
M2_EXECUTION_EQUIVALENCE      = PASS_28_OF_28
M3_V6_RUNTIME                 = COMPLETE_VALID_AND_SEALED
M3_V6_QUALIFICATION           = FAIL_AND_SEALED
M3_5_LABEL_VALIDATION         = NOT_ESTABLISHED
V7_FRESH_QUALIFICATION        = NOT_STARTED
M4_FRESH_COUNTERFACTUAL_MAP   = BLOCKED_NOT_STARTED
PROJECT_GPU_WORKERS_OBSERVED  = 0
PROTECTED_EVAL160             = NO_READ_HARD_STOP
```

The project is not blocked by GPU3, memory, SSH, or a running rollout. It is
blocked by a prospective scientific-contract issue: the frozen M3.5 selector
requires six eligible states in each of four strict phases, while the completed
clean-only census found only three qualifying exposed parents, all from
`libero_10`. No counterfactual branch from that census was started.

Do not rerun the old eight-parent M3.5 selection. Preserve every historical
root. The next Codex must freeze a new prospective protocol, source/tree,
selection, static audit, authorization receipt, and runtime root before any new
counterfactual branch.

## 2. Prompt for the new Codex window

Paste this into a new Codex conversation and enter Goal Mode:

```text
Take over the OpenVLA Stage V Teacher-Student project from:

D:\vla_attack\repo_work\openvla-gripper-dutycycle-attack-resource-contract-20260810\docs\handoffs\STAGE_V_GOAL_MODE_TAKEOVER_20260810.md

Read that handoff completely, then read the controlling Goal Mode master plan:

C:\Users\刘宇\.codex\attachments\3729b217-e532-42b2-a5ae-901e405aa019\pasted-text-1.txt

Create a persistent goal whose objective is to continue prospectively from the
current M3.5 HOLD through fresh V7 qualification, the fresh M4 V_phys map,
privileged Teacher, causal Student, TRAIN/VAL-only scheduler, and the fresh
Student-Time versus phase/contact-matched Random-Time actual intervention
experiment, stopping at READY_FOR_PROTECTED_EVAL unless the owner separately
authorizes Eval160. Preserve all sealed roots, fail closed, use the exact A800
environment, and do not stop merely because GPU3 has a foreign process.

First re-verify the live state and report the exact current gate. Do not launch
a GPU experiment until the prospective M3.5 protocol/accounting/selection gaps
in the handoff are closed and independently audited. Thereafter proceed
autonomously through valid PASS gates; seal and report any valid FAIL rather
than rerunning to pass.
```

The persistent-goal query in this handoff session returned `goal: null`; the
new window must create the new goal rather than assuming an old goal remains
active.

## 3. North-Star scientific objective

The final objective is a complete causal chain, not script completion and not
high offline AUROC:

```text
fresh clean-successful manipulation parent
  -> exact clean probe state and causal input binding
  -> direct gripper OPEN intervention with measured compliance
  -> measured physical vulnerability V_phys
  -> privileged V-Teacher trained on measured counterfactual labels
  -> deployable causal Student using only online clean history
  -> frozen online event scheduler
  -> fresh actual Student-Time vs phase/contact-matched Random-Time OPEN_T5
  -> parent-paired physical vulnerability risk difference
```

The only positive final detector conclusion is:

```text
STUDENT_TIMING_CAUSAL_UTILITY_ESTABLISHED
```

If the fresh timing experiment does not support a positive preregistered
paired effect, the correct conclusion is:

```text
STUDENT_TIMING_CAUSAL_UTILITY_NOT_ESTABLISHED
```

Teacher agreement, Student AUROC, historical attack evidence, and VIS results
cannot substitute for that causal gate. VIS is conditional and may start only
after the positive timing conclusion.

## 4. Authority and conflict resolution

Use this order of authority:

1. The owner instructions and the complete Goal Mode master plan at the path
   above.
2. Immutable machine-readable producer/auditor receipts and their SHA256s.
3. A freshly verified clean source commit/tree and runtime binding.
4. Historical handoffs as interpretation aids only.

Two stale statements must not be propagated:

- The master plan names historical HEAD `813a2ddb`; live local, server, and
  GitHub head are now `a33b0f40...` with tree `1e7825b...`.
- Old handoffs describe an M4 design as `40 x 72 = 2880` branch rows. The
  controlling formal design is `3840` physical branches and `2880` matched
  treatment-label rows: 40 parents x 24 probes x four physical conditions.

Historical protocol files and roots remain authoritative for what happened
under them. They must never be edited to match the new plan.

## 5. Live source, server, and GitHub snapshot

Snapshot time for live GPU/process evidence: `2026-08-10T15:34:15Z`.

| Item | Verified value |
|---|---|
| Local worktree | `D:\vla_attack\repo_work\openvla-gripper-dutycycle-attack-resource-contract-20260810` |
| Branch | `codex/stage-v-resource-contract-20260810` |
| Local/server/GitHub HEAD | `a33b0f402f4857b1c4a9f1a56c621f3246e4dc2a` |
| Tree | `1e7825b16484e67aa2313a356cbe73038bbc337a` |
| Server alias / host | `ssh dty` / `pm-364c0001` |
| Isolated server worktree | `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-resource-contract-20260810-c8f81820` |
| Required Python | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python` |
| Python version | `3.10.16` |
| GitHub PR | `#111`, OPEN, DRAFT |
| PR URL | `https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/111` |
| PR base | `codex/m1-v2-eight-gpu-diagnostic-20260808` |
| PR merge state | `UNSTABLE` |

Before this handoff file was added, both local and isolated server worktrees
were clean and matched the HEAD/tree above. The server worktree is detached by
design. The server checkout `/mnt/sdc/dty_user/openvla_attack` is unrelated and
dirty; do not clean, reset, or use it as the experiment source.

Use `dty`, not the stale `vla` alias. Do not mutate global SSH configuration.
Direct server GitHub access has been intermittent. The verified fallback is a
local `git bundle`, `scp`, then server-side `git fetch` from the bundle. Use the
fallback only when direct fetch fails; do not weaken TLS or global routing.

### GitHub CI issue that must be separated from science

PR #111 currently has:

| Check | Status |
|---|---|
| `source-registry` | PASS |
| `stageb-cpu` | PASS |
| `detector-v5-cpu` | FAIL |

The failing job is run `31396725309`, job `93481322819`. Its compile shell
block lacks continuation backslashes after
`materialize_factorized_v2_offline_evaluation_bundle.py` and the following
R3 paths. GitHub therefore tries to execute
`scripts/detector_v5/audit_r3_contact_input.py` directly and exits `126` with
`Permission denied`. The workflow has no diff relative to the PR base, so this
is inherited rather than introduced by the Stage V changes. It still needs a
minimal workflow fix and a green rerun before merge. It is not evidence about
M3.5 or any scientific gate.

## 6. Live GPU/resource snapshot

| GPU | UUID | Free MiB | Current condition |
|---:|---|---:|---|
| 0 | `GPU-bf4309d3-8cba-437e-8d87-cee9f1e6d232` | 81213 | baseline only |
| 1 | `GPU-f6910e5c-f41e-109e-43d0-f01f0d77dbf2` | 81213 | baseline only |
| 2 | `GPU-7b06162a-27e4-2552-e891-d201e3fae6b9` | 81213 | baseline only |
| 3 | `GPU-41cd4b75-e3d4-92b8-ec37-ddca13e3761a` | 74458 | foreign GR00T process |
| 4 | `GPU-e85ed586-ba64-a9e3-8fa9-07f16f84dcda` | 81213 | baseline only |
| 5 | `GPU-185b30c0-074c-6f07-aa8b-a67d00e8e4a9` | 81213 | baseline only |
| 6 | `GPU-92963392-f77a-85ce-4ba7-7a8288429ca5` | 81213 | baseline only |
| 7 | `GPU-bd2cfcc1-64ab-c2d0-9ae3-245fc8d21a76` | 81213 | baseline only |

No project M3.5/dynamic/M1/RB1 worker was observed. GPU3's current foreign
process is PID `1125866`, owner `huanzze`, executable
`/home/huanzze/isaac-gr00t-n1.7/.venv/bin/python3`, approximately 6748 MiB,
running `gr00t/eval/run_gr00t_server_timed.py`. Historical PID `964381` is not
the current PID. Always re-enumerate rather than hard-coding a PID.

Never kill, pause, renice, migrate, signal, or modify a foreign process. GPU3
is not a global veto. For M3.5/V7/M4/Stage O/VIS and training, an otherwise
eligible GPU may be used when free memory is at least `20480 MiB`, with at most
one project worker per physical GPU. Use a partial fleet and parent-level
atomic jobs. Admission order is:

```text
check free memory
  -> atomically acquire project GPU lease
  -> recheck free memory
  -> launch one parent bundle
  -> flush and audit artifacts
  -> release only after all child processes exit
```

## 7. Evidence-backed milestone ledger

| Milestone | Status | Evidence boundary |
|---|---|---|
| M0 static closure | PASS | Prior frozen gate; rerun regression after any new source change. |
| M1 runtime determinism | PASS_CLASSIFIED | Directly re-read immutable M1 audit and COMPLETE receipt. Seven-GPU primary clean cohort, not an eight-GPU clean PASS. |
| M2 execution equivalence | PASS | Directly re-read RB1-V2 audit: 28/28 pairs. |
| M3 V6 runtime | COMPLETE_VALID | 480/480 tasks `DONE_VALID`; sealed. |
| M3 V6 qualification | FAIL | Producer and independent audit FAIL; sealed, no rerun-to-pass. |
| M3.5 label validation | HOLD / NOT ESTABLISHED | Static/runtime machinery exists, but no valid counterfactual branch result or validation receipt exists. |
| V7 fresh qualification | NOT_STARTED | Blocked until M3.5 PASS. |
| M4 fresh V map | BLOCKED / NOT_STARTED | Historical R2A root is immutable HOLD and cannot be reused as the fresh map. |
| M5 V-Teacher | NOT_STARTED | No valid fresh M4 labels. |
| M6 causal Student | NOT_STARTED | No Teacher/fresh labels. |
| M7 scheduler | NOT_STARTED | No frozen Student. |
| M8 timing causal test | NOT_STARTED | No fresh Stage-O run. |
| M9 VIS selectivity | BLOCKED | Requires positive M8 timing gate. |
| M10 final freeze | NOT_STARTED | No final bundle. |
| M11 protected Eval160 | NOT_STARTED / HARD STOP | Owner authorization required; do not read automatically. |

### M1 direct evidence

Root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/M1_V2_2_DYNAMIC_COHORT_f0deb3c4_20260809T174134Z`

Key receipts:

- `M1_V2_INDEPENDENT_AUDIT.json`, SHA256
  `d521665f1a6f62b5b5347dd128ca6a2adf2eb56e8e8b342e66f2344a223fa059`.
- `M1_V2_COMPLETE.json`, SHA256
  `c7913e5d0986f23715e7a6d1fc40c67bcd6972fa419deabca4d3423d152cecb3`.

Verified facts:

- Status `PASS_CLASSIFIED`; COMPLETE owner `INDEPENDENT_AUDITOR`.
- Classification `HETEROGENEOUS_MULTI_GPU_DIVERGENCE`.
- Primary clean GPU set `[0,1,2,4,5,6,7]`.
- R1 runs `28`; local pairs `28`; cross-GPU pairs `84`.
- Visual/input divergence exists, including same-mode and cross-GPU effects.
- `action_stable=true`; no action-divergent pair.
- Eval160, protected, attack, intervention, and VIS/PGD counters are all zero.

This is evidence of action-stable classified divergence, not evidence that
rendered inputs are bitwise deterministic across eight clean GPUs.

### M2 direct evidence

Root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/RB1_V2_CAUSAL_EQUIVALENCE_f0deb3c4_83dfe538_20260809T185200Z`

`RB1_V2_CAUSAL_AUDIT.json` SHA256:
`103aca1c86587af5d00b1c52cefb55dc771a3aa579928628ce4dee2b43092b48`.

Verified facts:

- Status `PASS_CLASSIFIED`; verdict `PASS`.
- `pair_count=28`, `pair_pass_count=28`.
- `causal_execution_equivalence=PASS`.
- Allowed observed differences are diagnostic/model-input/observation hashes;
  causal action and physical execution remain equivalent.
- M1 artifacts were not modified; protected counters are zero.

## 8. V6: valid execution, failed qualification

Immutable root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_857972a7_20260810T013514Z`

Key artifacts:

- `CONTROL_QUALIFICATION_REPORT.json`, SHA256
  `171a0a9e16a41d07aa250645dae5c5c51266b4f57bae90e1a28bf90332f93a3e`.
- `CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json`, SHA256
  `55ad72195e24731baf517014714b4b691e17c61b63fdc2ff78365b682afb98d6`.

Permanent facts:

- 480/480 A/B tasks completed engineering-valid.
- Runtime is `COMPLETE_VALID`.
- Frozen producer report is `FAIL`; independent audit is `FAIL`.
- Frozen qualified counts are `10/0/10/10` for
  `libero_10/goal/object/spatial` because terminal hash equality was an old
  hard gate.
- No formal40 and no fresh M4 were created.
- Protected counters are zero.

Read-only forensic artifact:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_V6_FORENSIC_CLOSEOUT_V1_20260810T093233Z/V6_FORENSIC_MATRIX.json`

SHA256:
`9c926db007331fa0b4bea25aaddf3dda61a18db7e30bf437dd71bbf0062c0ad7`.

| Suite | Both A/B clean-success | Terminal hash equal | Terminal hash unequal |
|---|---:|---:|---:|
| `libero_10` | 27/60 | 23 | 4 |
| `libero_goal` | 46/60 | 0 | 46 |
| `libero_object` | 42/60 | 42 | 0 |
| `libero_spatial` | 48/60 | 36 | 12 |
| Total | 163/240 | 101 | 62 |

The forensic result proves the V6 Goal yield was suppressed by the old exact
terminal-hash gate; it does not convert V6 to PASS and does not authorize
reusing V6 parents in V7.

## 9. Freshness ledgers already closed before M3.5

Counterfactual exposure union V4:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4_20260810T104027Z/STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4.json`

- File SHA256: `b234c896b4eeee862914a717e56a79cefa3ee8ba43fdd4b8c7aafb027ec0a612`.
- Canonical union SHA256: `62012ba0246d123fb6caa59a8dbc80bc3bd4fb27ea32a6a6c4f87bb3c111dc4a`.
- Count `53`; branch outcomes and protected data not read.

Cumulative clean-attempt exclusion V2:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2_20260810T104027Z/STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2.json`

- File SHA256: `a8be0582c5cbe3ae2224cf6786ac18a0558f2105962aef73ea69fffd462e81e6`.
- Canonical union SHA256: `bbe427d645efcaa683d7cfb305014333317ed9cd1e369a1825406b08c2e6302a`.
- Count `357`: 117 prior attempts plus 240 disjoint V6 identities.

These exact artifacts are historical inputs. Before V7, rebuild both unions
mechanically from all then-current sources. Do not assume the count remains
53 or 357 and do not hard-code either value.

## 10. What the previous Goal Mode work implemented

The old plan named HEAD `813a2ddb`. Between it and current `a33b0f40`, 39
files changed with approximately 4958 inserted lines. The work added:

- V4 exposure and V2 clean-attempt union builders and audits.
- Outcome-blind diagnostic selection from already exposed identities.
- Clean-only phase classifier, deterministic probe-plan builder, physical
  taxonomy, direct-OPEN intervention runner, dynamic parent launcher, worker,
  dispatcher, lease validation, static auditor, and authorization issuer.
- Treatment compliance and exact runtime/source/model/input bindings.
- `CONTROL x3`, `T3 x3`, `T5 x3`, `T10 x3` repeatability execution support.
- Coverage-only mode that writes clean trajectory, phase coverage, and parent
  result without launching a counterfactual branch.
- Fixes for structured launcher arguments, M3.5 diagnostic exposure mode,
  selection aliases, atomic-attempt output ownership, EGL physical-GPU
  binding, wrapper visibility, and runtime UUID lookup through `nvidia-smi`.
- Variable-size clean coverage census support in the static auditor while
  retaining exact 8-parent/2-per-suite checks for the old normal M3.5 role.

Relevant current source files:

- `configs/STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_2_1.json`
- `configs/STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_2.json`
- `scripts/detector_v5/build_stage_v_m3_5_probe_plan.py`
- `scripts/detector_v5/run_stage_v_m3_5_intervention_parent.py`
- `scripts/detector_v5/run_stage_v_m3_5_dynamic_parent.py`
- `scripts/detector_v5/run_stage_v_dynamic_dispatcher.py`
- `scripts/detector_v5/run_stage_v_dynamic_worker.py`
- `scripts/detector_v5/stage_v_dynamic_common.py`
- `src/gripper_attack/stage_v_m3_5_phase_classifier.py`
- `src/gripper_attack/stage_v_m3_5_physical_taxonomy.py`
- `scripts/detector_v5/audit_stage_v_m3_5_static_contract_v1_2.py`

At current source, the exact A800 environment previously reported `267 passed,
6 skipped`, and each coverage-census static receipt reports 36 checks, zero
failures, PASS. The 267/6 result was observed in session output rather than a
dedicated immutable test receipt; rerun the focused and full regression after
the next patch and preserve a durable receipt.

## 11. M3.5 runtime attempts and clean-only coverage census

The early M3.5 V1-V7 attempts failed before valid science because of launcher,
selection, atomic-output, EGL, or UUID engineering issues. They produced no
valid scientific counterfactual result and must not be mined as labels.

V8 root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_5_RUNTIME_V1_2_1_M35MODE_V8_20260810T134608Z`

All eight GPUs passed resource admission, including GPU3. Clean trajectories
were produced, but the old strict four-phase coverage gate failed before
counterfactual branches. This is not M3.5 label PASS or label FAIL.

The subsequent clean-only census used all 53 exposed identities across:

1. `STAGE_V_M3_5_CLEAN_PHASE_COVERAGE_CENSUS_V1_20260810T141343Z`
2. `STAGE_V_M3_5_CLEAN_PHASE_COVERAGE_CENSUS_CONTINUATION_V1_20260810T142645Z`
3. `STAGE_V_M3_5_CLEAN_PHASE_COVERAGE_CENSUS_CONTINUATION2_V1_20260810T142959Z`

The first two stopped on deterministic task-local taxonomy failures. The
second continuation completed all 36 remaining entries. Each root has its own
prospective protocol, selection, 36/36 PASS static audit, and runtime
authorization. No counterfactual branch was started.

Combined independent coverage audit:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_5_SELECTION_PHASE_COVERAGE_AUDIT_V2_20260810T144319Z/STAGE_V_M3_5_SELECTION_PHASE_COVERAGE_AUDIT_V2.json`

SHA256:
`fba19831bc7c9f0470f7bdcadd203ceffc1a1ef86824b5363bc498e1236e0573`.

Verified result:

```text
status                         = PASS_COVERAGE_UNION_CLOSED_BUT_FORMAL_COHORT_NOT_READY
coverage union                 = 53/53, missing 0
valid clean coverage artifacts = 51
explicit taxonomy failures     = 2
coverage-qualified parents     = 3
qualified by suite             = libero_10 3, goal 0, object 0, spatial 0
required by old protocol       = 2 per suite
formal M3.5 cohort ready        = false
counterfactual branches started= 0
protected counters             = 0
```

The three strict-phase-qualified parents are:

| Parent | PRE | CONTACT | LIFT | CARRY |
|---|---:|---:|---:|---:|
| `libero_10/task_00/state_49` | 111 | 72 | 6 | 115 |
| `libero_10/task_06/state_49` | 71 | 60 | 56 | 15 |
| `libero_10/task_07/state_47` | 24 | 21 | 73 | 305 |

Taxonomy eligibility audit:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_5_SELECTION_TAXONOMY_ELIGIBILITY_AUDIT_V1_20260810T145500Z/STAGE_V_M3_5_SELECTION_TAXONOMY_ELIGIBILITY_AUDIT_V1.json`

SHA256:
`e445883a6903da5db373bb06dfd5fc0a0ad0d612af5430f02da5d871a0b3a4d7`.

It records 51 eligible and two explicitly ineligible parents:

- `libero_goal/task_00/state_48`
- `libero_goal/task_00/state_49`

Both fail `PhysicalTaxonomyError:GOAL_OBJECT_BINDING_EMPTY`. Their BDDL task is
`open_the_middle_drawer_of_the_cabinet.bddl`, whose goal is fixture-region
opening rather than carrying a movable `In/On` object. Do not misbind the
cabinet fixture as the manipulated transport object merely to make the parser
pass. Either prospectively exclude unsupported fixture-only tasks with an
explicit taxonomy-eligibility rule or prospectively implement a separate,
scientifically justified fixture-opening physical taxonomy.

## 12. Critical data-quality and protocol findings

### P0 — old formal M3.5 cohort is not ready

The census closed all 53 selected identities and found only three parents
meeting the old `>=6` candidates in each of PRE/CONTACT/LIFT/CARRY. Therefore
the previous eight-parent selection cannot legally launch. GPU availability
does not change this result.

### P0 — current selector conflicts with the controlling A4 design

Current `build_stage_v_m3_5_probe_plan.py` selects six hash-ranked states from
each of four strict phase buckets. The master plan requires one
intervention-eligible clean corridor, sorted by timestep, followed by 24
deterministic quantile positions and deterministic deduplication. Eligibility
must require a valid clean state, physically meaningful object/contact state,
not intentional post-release, and enough horizon for T10 plus the physical
observation window.

Changing from strict phase buckets to the corridor is a material prospective
scientific design change, not a bug fix. It requires a new protocol version,
new source/tree, tests, static independent audit, selection receipt, and fresh
runtime root. It must be frozen before reading any new branch outcome.

### P0 — diagnostic repetition accounting is conflated with labels

Current V1.2.1 calls the M3.5 design:

```text
288 physical executions / parent
216 treatment_label_rows / parent
```

Those 216 are repeated treatment executions: 24 probes x three doses x three
repetitions. The controlling formal M4 accounting is:

```text
96 physical executions / parent
72 matched treatment-label rows / parent
```

The next M3.5 protocol must name three distinct quantities:

1. physical diagnostic executions (`288` if the 3x design is retained),
2. treatment repetition observations (`216`), and
3. repeatability-collapsed probe-dose labels (`72`).

Current branch rows embed a control/treatment pair but do not expose the
required explicit `shared_control_branch_id` and
`shared_control_result_sha256` lineage. Freeze deterministic replicate pairing
and aggregate-label semantics prospectively. Every label must be traceable to
its matched control; no orphan or implicit `F_control` is allowed.

### P0 — no runtime evidence yet validates the full label contract

Code and static tests exist, but no valid counterfactual M3.5 branches exist.
Therefore none of the following has runtime PASS evidence yet:

- direct `raw 1.0 -> env -1.0` treatment compliance across four suites,
- full causal-state restore and policy-input equivalence,
- surgical non-gripper action equality during treatment,
- 3x class repeatability,
- physical taxonomy correctness against blinded/manual review,
- dose sanity and per-dose horizon semantics.

Do not infer these gates from code compilation or clean-only census artifacts.

### P1 — GitHub CI is red for an inherited shell-continuation bug

Fix the workflow minimally and rerun it. Do not mix this engineering repair
with a scientific PASS claim.

### P1 — current head and historical docs are easy to confuse

Any future runtime must record the new actual commit/tree. Never copy
`813a2ddb`, `208461a2`, or `a33b0f40` into a new protocol without verifying the
new clean source. Existing roots retain their original bindings.

## 13. Immediate autonomous continuation plan

### Step 0 — re-establish exact state

Perform the read-only commands in section 17. Confirm no unexpected local or
server edits, recheck PR checks, enumerate all GPU processes, and verify that
no project rollout appeared after this snapshot.

### Step 1 — restore repository CI hygiene

Apply the minimum continuation fix in
`.github/workflows/cpu-detector-v5.yml`. Run the exact compile list and relevant
CPU tests, commit intentionally, push PR #111, and verify all checks. Do not
change experiment roots or scientific receipts.

### Step 2 — freeze the next prospective M3.5 protocol

Create a new version; do not edit V1/V1.1/V1.2/V1.2.1 historical files.
Resolve before runtime:

- intervention-eligible corridor and deterministic 24-quantile selector,
- deterministic insufficient-plan behavior,
- per-dose horizons T3/T5/T10 plus fixed `H_phys` and secondary `H_task`,
- treatment, mediator, physical failure, and task consequence separation,
- explicit supported-task physical taxonomy and fixture-only exclusions,
- exact control lineage for every treatment repetition and collapsed label,
- 288 diagnostic execution / 216 treatment repetition / 72 aggregate label
  vocabulary, kept separate from formal M4's 96/72 accounting,
- exact state, policy input, source/model/runtime, and GPU UUID bindings,
- atomic parent execution and retry policy,
- all M3.5 PASS outputs and protected zero counters.

Use existing clean-only census artifacts only to test the prospective corridor
and select exposed diagnostic identities. Do not read historical branch
outcomes. Select deterministically from the exposure inventory, cover all four
suites, target two eligible exposed parents per suite, and record any
prospectively unavoidable shortfall before branches.

Add focused tests for the selector, accounting, matched-control lineage,
taxonomy eligibility, and auditor rejection paths. Run focused plus full
regression in the exact A800 environment. Preserve a machine-readable test
receipt.

### Step 3 — static independent audit and freeze

Only after source is clean:

```text
record source commit/tree
record every script/config/dependency SHA256
freeze protocol and deterministic selection
run independent static auditor
issue auditor-bound runtime authorization
create a fresh isolated server worktree/root
```

The producer must not own final COMPLETE. Do not launch if producer/auditor
disagree or if any source/tree/SHA binding differs.

### Step 4 — run M3.5 once under the frozen design

Use already exposed diagnostic identities only. Use the dynamic eligible-GPU
queue, up to one parent worker per GPU, one complete parent bundle per physical
GPU, and the `>=20480 MiB` rule. GPU3 may participate if admitted; its foreign
process remains untouched. No batch-wide wait for a perfectly empty fleet.

After launch, do not pull, hot-patch, change protocol, substitute parents, or
delete valid results. A preregistered retry is allowed only for a pure
infrastructure failure before any valid scientific result for that parent.

### Step 5 — independently close M3.5

Require all of the following:

```text
treatment semantics PASS
treatment compliance PASS
causal-state sufficiency PASS
surgical arm isolation PASS
repeatability PASS
physical taxonomy PASS
probe/dose/horizon PASS
producer/auditor reconciliation PASS
protected counters zero
```

Required closeout includes protocol/SHA, execution-semantics receipt,
treatment-compliance audit, state audit, surgical audit, taxonomy audit,
repeatability audit, probe/dose receipt, independent audit, and final validation
receipt. Only `M3.5_LABEL_VALIDATION = PASS` unlocks V7. A valid FAIL is sealed
and reported; it is not rerun to pass.

## 14. Downstream plan after a valid M3.5 PASS

### V7 fresh qualification

- Recompute latest exposure and clean-attempt unions.
- Create new source/tree, root, protocol, salt, and G10-derived candidate pool.
- Enforce zero intersection with both unions and M3.5 intervention identities.
- Never read V6, M3.5, or vulnerability outcomes for selection.
- Treat valid `TASK_FAILURE` as engineering-complete when exit/artifact/runtime/
  provenance checks pass; do not confuse task outcome with infrastructure.
- Qualify only A/B both-clean-success with exact initial/source/model/runtime/
  GPU affinity; terminal hash is descriptive, not a hard gate.
- Freeze replicate A as canonical before probe sufficiency; never choose the
  prettier or more probe-rich replicate.
- Consume each frozen SHA-ranked suite prefix until ten qualify; never skip an
  inconvenient valid candidate.
- PASS requires producer and auditor PASS, exactly 10/suite and 40 total,
  engineering-invalid zero, overlap zero, protected zero, and valid 24-probe
  plan for every formal parent.

Qualification scheduling must keep each parent A+B bundle atomic on one GPU.

### Formal40 split

Before any formal vulnerability outcome, freeze:

```text
TRAIN 24 = 6/suite
VAL    8 = 2/suite
TEST   8 = 2/suite
```

Use fixed-salt parent hashing and preserve parent grouping. Record separate
qualification-source and split-builder commit/tree/script SHA provenance.

### M4 fresh counterfactual map

- 40 parents x 24 frozen probes x `[CONTROL,T3,T5,T10]`.
- 96 physical branches and 72 matched treatment labels per parent.
- 3840 physical branches and 2880 treatment labels total.
- Primary estimand `V_phys@T5`; task outcome is secondary.
- One complete 96-branch parent bundle on one physical GPU.
- Report prevalence, per-suite rates, dose response, contamination, abstains,
  taxonomy, latency, and within-coarse-phase positive/negative heterogeneity.
- Independent audit must reconcile every branch, control link, snapshot,
  label, parent cluster, and protected counter before M5.

### M5 privileged Teacher

- Target measured `V_phys@T5`, never old critical phase.
- Inputs are clean privileged state/history at or before t only.
- Train on TRAIN parents; select at most three preregistered configs on VAL;
  evaluate frozen once on TEST.
- Keep all 72 rows from a parent in one split; exclude ABSTAIN from primary
  binary training rather than forcing negative.
- Gate: VAL AUROC at least 0.75 and VAL AUPRC at least 1.5 times prevalence,
  with calibration, per-suite, phase-conditioned, and parent-bootstrap metrics.

### M6 causal Student

- Use only deployable online clean observations/history: proprio, online robot
  state, clean action/history, causal window.
- Forbid simulator geometry, unavailable contact truth, future state, Teacher
  features, and counterfactual outcomes; run a feature-leakage audit.
- Hard anchors are measured V_phys labels. Teacher soft probabilities on TRAIN
  may be auxiliary distillation targets, never ground truth.
- Separate Student-vs-Teacher fidelity, Student-vs-measured-V_phys validity,
  and actual timing utility. TEST is a single frozen evaluation with no retune.

### M7 scheduler

- Freeze at most three preregistered scheduler variants using TRAIN/VAL only.
- Secondary offline gates: vulnerable-anchor recall at least 0.60, false onset
  at most 0.10, median delay at most +2 steps.
- Do not treat dense Teacher predictions as true event onset and do not expand
  search because GPUs are idle.

### M8 fresh Student-Time causal experiment

- Use a new fresh Stage-O cohort, suggested 40 parents / 10 per suite, disjoint
  from all prior exposure, clean attempts, M3.5, and M4 parents.
- Freeze Student checkpoint/config, feature contract, scheduler, threshold,
  cooldown/hysteresis, and source/tree before Stage O.
- For each parent run one atomic four-branch bundle on one GPU:
  `CONTROL_STUDENT_TIME`, `OPEN_T5_STUDENT_TIME`,
  `CONTROL_RANDOM_TIME`, `OPEN_T5_RANDOM_TIME`.
- Random time must come from the same canonical clean trajectory and same
  intervention-eligible phase/contact regime under a frozen salt.
- Primary statistic is the parent-paired physical vulnerability risk
  difference, Student-Time minus matched Random-Time, with preregistered 95% CI.
- A negative result is final scientific evidence, not a tuning signal.

### M9 VIS and protected stop

Only after `STUDENT_TIMING_CAUSAL_UTILITY_ESTABLISHED`, freeze timing and use a
new fresh M9 cohort for CLEAN, TRUE VIS, and matched random VIS under identical
budgets/support. Require both a physical selectivity effect and the complete
visual-to-action-to-physical mechanism trace. VIS cannot rescue failed timing.

The default autonomous stopping point is:

```text
READY_FOR_PROTECTED_EVAL
```

Do not enter Eval160/M11 without a new explicit owner authorization.

## 15. Hard stops and immutable boundaries

Stop and report, without improvising around the gate, for:

- any Eval160 or protected-evaluation access request before owner authorization,
- post-outcome protocol, matching, threshold, taxonomy, or cohort modification,
- parent/split/exposure leakage,
- source/tree/model/runtime/GPU UUID mismatch,
- scientific artifact corruption or missing control lineage,
- producer/auditor irreconcilable disagreement,
- a valid scientific result that would need deletion, substitution, or rerun
  to pass,
- label validity failure or Student timing causal failure.

Do not stop merely because one GPU is contended, utilization is nonzero, the
fleet is partial, or a foreign process exists while admission still passes.

## 16. Claims that are currently forbidden

Do not claim any of the following from current evidence:

- "eight-GPU clean determinism PASS";
- visual/input bitwise determinism;
- V6 fresh qualification PASS;
- valid M3.5 physical labels or repeatability;
- a valid fresh formal40 or M4 V_t map;
- Teacher or Student performance;
- Student-Time causal utility;
- VIS selectivity or exploitability;
- readiness for protected Eval160.

Use exact statuses such as PASS, FAIL, HOLD, NOT_STARTED, COMPLETE_VALID,
INVALID_ENGINEERING, ABSTAIN, and READY_FOR_PROTECTED_EVAL. Do not write
"near-pass", "mostly pass", or "probably fine".

## 17. First read-only commands for the new window

Local PowerShell:

```powershell
$repo = 'D:\vla_attack\repo_work\openvla-gripper-dutycycle-attack-resource-contract-20260810'
git -C $repo status --short --branch
git -C $repo rev-parse HEAD
git -C $repo rev-parse 'HEAD^{tree}'
git -C $repo log -12 --oneline
gh pr view 111 --repo Leo-6-maker/openvla-gripper-dutycycle-attack --json state,isDraft,headRefOid,baseRefName,mergeStateStatus,statusCheckRollup,url
gh pr checks 111 --repo Leo-6-maker/openvla-gripper-dutycycle-attack
```

Server:

```bash
ssh dty 'hostname; date -u; /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python --version'
ssh dty "cd /mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-resource-contract-20260810-c8f81820 && git status --short --branch && git rev-parse HEAD && git rev-parse 'HEAD^{tree}'"
ssh dty "nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"
ssh dty "pgrep -af 'run_stage_v_m3_5|run_stage_v_dynamic_(dispatcher|worker)|run_stage_v_m1|RB1_V2' || true"
```

The `pgrep` command can match its own inspection shell; reconcile hits with
`ps` and `nvidia-smi` before claiming a project worker exists.

## 18. Existing repo handoffs worth reading

- `docs/handoffs/STAGE_V_M3_5_V6_FORENSIC_CLOSEOUT_20260810.md`
- `docs/handoffs/STAGE_V_M3_5_V1_1_PATCH_20260810.md`
- `docs/handoffs/STAGE_V_M4_COUNTERFACTUAL_HOLD_20260810.md`
- `docs/handoffs/STAGE_V_M3_QUALIFICATION_V6_RUNNING_20260810.md` only as a
  historical in-run snapshot; its header marks it superseded.
- `docs/handoffs/STAGE_V_M1_HANDOFF_20260808.md` only for early history; the
  later M1 V2.2 and RB1-V2 receipts in section 7 are the current evidence.

## 19. Takeover acceptance checklist

The new Codex has correctly taken over when it can state all of the following:

```text
I verified current HEAD/tree locally, on dty, and on GitHub.
I understand M1 is a 7-GPU classified PASS and RB1 is 28/28 PASS.
I will preserve V6 FAIL and every historical M3.5/M4 root.
I understand there are no valid M3.5 counterfactual labels yet.
I will not rerun the old eight-parent selection.
I will prospectively resolve corridor selection and 288/216/72 accounting.
I will make every label's matched-control lineage explicit.
I will fix the inherited CI continuation issue without calling it science.
I will use the exact openvla-official-a800 Python and atomic GPU leases.
I will use GPU3 if admitted but will never modify its foreign process.
I will proceed through PASS gates and seal valid FAIL results.
I will not read Eval160 and will stop at READY_FOR_PROTECTED_EVAL.
```

The highest-priority scientifically legal task is the prospective M3.5
protocol/accounting/selection closure, with the small CI repair performed in
parallel before the next published runtime commit.
