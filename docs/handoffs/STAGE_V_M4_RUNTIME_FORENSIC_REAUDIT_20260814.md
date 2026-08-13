# Stage V M4 runtime forensic re-audit — 2026-08-14

Status: `HOLD`; this handoff stops at `READY_FOR_ZERO_TREATMENT_Q00_REVIEW`.
It records read-only audit and authority preparation only. No Q00 canary, formal
M4 branch, attack/VIS evaluation, Eval160 read, or protected evaluation was run.

## Verdict and claim boundary

The formal M4 HOLD is correctly scoped. The two closure claims are also valid
only at their stated structural boundaries:

- Parent00 closure: `PASS_PREINTERVENTION_STRUCTURAL_INVALIDATION`, 96/96
  branches failed at `PRE_PRIMARY_WINDOW_RUNTIME_EXACTNESS`, with zero rows,
  actions, treatment receipts, consumable binary labels, and valid V_phys labels.
  Report SHA256:
  `adfec5855839c80a6a38042573567dc4ee4c041f6294a00e8dc0cb6c9bd29242`.
- Parent01 closure: `PASS_PREBRANCH_ABORT_CLOSURE`, zero branch records, zero
  primary-window steps, zero forced-open steps, zero receipts, and
  `outcomes_read_uncertain=true`. Report SHA256:
  `48e51bf359e146036f83aa12e16880e76af673eca7babbcf39f0a0831c21185c`.

The historical global hold is immutable and its SHA256 remains
`f5e3b37db9c438e0c9219d56b8bf92e7d02d6e8c40c69f03bc6986c9bc203db5`.
It explicitly records `m4_outcomes_materialized=true`,
`v_phys_artifacts_materialized=true`, and `outcomes_read=true`, while the
protected counters remain zero and the global status is
`HOLD_STOP_GLOBAL_SCHEDULING`. Therefore the safe claim is:

> Historical outcome/label artifact files exist, but no valid consumable formal
> M4 result or downstream V_phys claim exists under the HOLD.

Do not restate this as “no outcome artifacts existed,” and do not edit the
historical root, Parent00/01 roots, frozen exact plan, or old authorizations.

## Live bindings reviewed

GitHub PR: [#112](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/112)

```text
head branch: codex/m4-corridor-replenishment-post-32-of-40-hold-20260813
head commit: 1b4c7a53b96a9d9f9292ce6c9072488d5da49480
head tree:   5d7d0ad9031df655e38e2cc24e789d26088cbab7
base commit: fcaa59cacf1895cc9f1d372944366b7b2952911c
state:       OPEN, DRAFT, mergeable
checks:      detector-v5-cpu PASS; source-registry PASS; stageb-cpu PASS
```

The local worktree is clean and the branch tip matches the remote. A full
`git fetch --all --prune` remains non-blockingly broken by the remote's stale
`experiment/moka-twopot-window-theory` ref; the target branch fetch, push, and
GitHub checks work.

Server access is direct through SSH alias `dty` (no jump host):

```text
host: pm-364c0001
user: dty_user
official Python: /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
Python: 3.10.16
resolved Python: /home/sz/miniconda3/envs/hallo/bin/python3.10
formal historical worktree: /mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-primary-3370d8cf
formal historical HEAD/tree: 8951731c34595bcc73295f3c9c7390b13a219154 / 8ba22918c8c3fb1db4a6586525f3ab1b08e0d259
formal/M4 processes observed: none
```

The current GPU snapshot is not a launch authorization:

```text
GPU 0-5,7: approximately 1.2–1.7 GiB free; foreign purge-paper workers present
GPU 6:      64098 MiB free; foreign /home/zkx/miniconda3/envs/CAMEF/bin/python, PID 3139120
```

The successor resource rule is strict `memory_free_mib > 20480`; foreign work
is telemetry and is not killed or interfered with. The project lease table
allows at most one active project worker per physical GPU. No project worker
was mounted in this turn.

## Minimal repairs pushed in PR #112

- Shared GPU admission and recheck now reject equality (`<=20480`) and reject a
  caller-supplied threshold below the successor contract. The M3.5 direct guard
  and M4 outer launch binding use the same strict rule; the outer binding and
  protocol must both equal `20480`.
- Parent00 closure binds the historical Gate-B runner SHA
  `aa98cd25b9f1a37ae747dcc95ddcf2a7b270d135997a741590958d309e65b972`, reports
  missing/duplicate identities, and forces HOLD for any row/action/receipt,
  treatment-compliant branch, consumable label, or valid V_phys label.
- Parent01 forbidden-science-artifact scanning is recursive.
- `stage_v_runtime_diff.py` emits deterministic structured mismatches with
  canonical path, type, shape, SHA256, parent/probe/branch, snapshot source,
  current runtime, exact-plan, provenance, and closure bindings.
- `run_stage_v_m4_zero_treatment_auditor.py` is dependency-injected and only
  replays frozen clean actions, restores/captures state, compares primary input,
  closes the environment, and emits a fail-closed receipt. Its AST boundary
  test rejects intervention/label-producer imports and forbidden symbols.
- `capture_stage_v_runtime_provenance.py` records the external Python,
  worktree, snapshot/upstream, module-origin hashes, artifact/file bindings, and
  an explicit null diagnostic GPU identity when no canary is authorized.

## Evidence produced

```text
local py_compile and direct self-checks: PASS
server official-environment targeted suite: 22 passed
GitHub detector-v5-cpu/source-registry/stageb-cpu: PASS
server provenance root:
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_EXTERNAL_RUNTIME_PROVENANCE_READONLY_1B4C7A53_20260814
receipt SHA256:
  6867f8f89a87b40293ab5e0b42545d6bf67ce6bb795610a7180c3edd66a89eea
student checkpoint SHA256: e24d00ca30c8fe0d5ef066e90872f010556bfabec13f78d4275962c6b35ca227
official snapshot commit/tree: 4d7a9daeb2bf9cf6b5d911f0ddcd93c254d0362b / cd5fe5263afd66d9131159d97bba1d1d660be5df
upstream commit/tree: c8f03f48af692657d3060c19588038c7220e9af9 / c326be57bb61629a3efd2b968ef141fa576a623f
```

## Legal next action and stop conditions

The next action is GPT/owner review of a prospective zero-treatment Q00
canary, not execution. Any future canary must use `dty`, the official Python
above, a fresh GPU/process snapshot before and after lease, strict free memory
greater than 20480 MiB, one project worker per GPU, no treatment branch, no
labels/V_phys, and zero protected counters. A foreign owner is never killed.

Stop immediately on source/tree/hash mismatch, missing official environment,
GPU threshold failure, foreign-process interference risk, any treatment
receipt/label/V_phys artifact, nonzero protected counter, probe reselection,
rerun-to-pass behavior, or any request to reinterpret historical artifacts as
current formal outcomes.

### Next-window prompt

```text
Continue Stage V from PR #112 at head 1b4c7a53b96a9d9f9292ce6c9072488d5da49480.
Use D:\vla_attack\repo_work\openvla-gripper-dutycycle-attack-post-32-of-40-hold-20260813
locally and SSH alias dty directly. Re-read GitHub head/tree/checks, the clean
server worktree, GPU ownership, and the immutable Parent00/01/global-hold roots.
The only current scope is CPU/read-only audit and authority preparation. Do not
launch Q00, formal M4, CONTROL/T3/T5/T10, attack/VIS, Eval160, protected eval,
Teacher/Student changes, probe reselection, or rerun-to-pass. Treat historical
materialized artifacts as non-consumable under HOLD. Pause for GPT/owner review
whenever an authority or scientific-interpretation choice appears.
```

Request: review this re-audit and return the successor Goal Mode prompt and
authority decision. Keep PR #112 draft; do not merge.
