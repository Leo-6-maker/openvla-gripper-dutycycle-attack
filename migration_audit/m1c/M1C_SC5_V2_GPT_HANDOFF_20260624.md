# M1C SC5-V2 GPT Handoff — 2026-06-24

> **Audience:** the next GPT conversation taking over this project.  
> **Repository:** `Leo-6-maker/openvla-gripper-dutycycle-attack`  
> **Working branch:** `feature/sc5-abstention-v2-20260622`  
> **Last GitHub head verified during handoff:** `d01d5ce15e9d80b779b9d50707ebcadeca320a1d`  
> **Server:** A800 `pm-364c0001`  
> **Project:** OpenVLA Gripper Duty-Cycle Attack / M1C Clean-Only Abstention Repair

---

## 0. Takeover instruction

You are taking over an audit-sensitive robotics experiment. Treat user-reported runtime status, GitHub state, and frozen protocols as separate evidence sources. Before authorizing any new collection, training, validation, or attack, verify the current branch head, active server processes, actual script hashes, output manifests, and frozen split constraints.

The project has repeatedly uncovered implementation bugs after apparently successful runs. Do not accept a result merely because all cells completed or a metric looks plausible. Always distinguish:

- engineering completion;
- data integrity;
- evaluation validity;
- independent validation;
- causal evidence;
- publication-level claims.

---

## 1. Non-negotiable invariants

These constraints remain in force unless Leo explicitly changes them in writing:

1. **Clean-only data path.** `condition=CLEAN`, `attack_frames=0`.
2. **All attacks remain HOLD.** Do not run VIS/PGD/action/gripper attacks.
3. **Blind-v2 remains HOLD.** Do not touch states `38–49` for training, dev, reserve, tuning, or diagnostics.
4. **Legacy validation states `28–37` are read-only.** They were already used for confirmatory R1/R2 evaluation and must not enter SC5-v2 train/dev.
5. **Compromised old blind data remains diagnostic-only.** The quarantined 12 cells must never be merged into formal train/dev/blind.
6. **No real robot.**
7. **C2 causal ablation remains HOLD / lower priority** because of storage and because data readiness is currently the blocker.
8. **Do not modify completed corpus files in place.** Use sidecars, new output roots, and append-only closure manifests.
9. **Do not touch an active collection checkout/process/output from an audit worktree.** Use a separate worktree for code changes.
10. **No formal SC5-v2 training until the data-readiness gates pass.** Protocol drafting and code preparation are allowed.

---

## 2. Historical status that remains valid

### 2.1 V4 object clean corpus

- Train: `250/250`
- Validation: `100/100`
- Formal blind: `0`
- Compromised blind: `12`, quarantined
- Total formal corpus: `350/350`
- RC errors: `0`
- Attack violations: `0`

### 2.2 P3 corpus integrity

- `P3A_EVALUATION_READINESS = PASS`
- `P3B_CRYPTO_PROVENANCE = PARTIAL`

Reason for P3B partial:

- original bridge did not record native `initial_state_sha256`;
- trajectory hashes were backfilled by sidecar and were unique;
- task/state membership overlap was checked;
- byte-level initial-state split isolation was **not** cryptographically verified.

Use the wording:

> task/state membership overlap = 0; trajectory duplicate = 0; initial-state byte-level leakage = not verified.

Do not write simply “0 split leaks.”

### 2.3 P4 v2 Teacher correction

P4 v1 was invalidated because it default-filled target positions with zeros and read `anchor_candidate` instead of `anchor`.

P4 v2 corrected:

- target position resolved from LIBERO/MuJoCo;
- unresolved target should fail closed;
- Teacher anchor read from `anchor["anchor"]`;
- `teacher_valid=True` implies `anchor>=0`;
- empty `f_gripper_opening_proxy` fallback was added in `a6bbc4c`.

### 2.4 P5 runtime-only evidence

Corrected P5 results reported by Leo:

- validation labels: `86 TV / 14 NC`;
- R1: coverage `83/86 = 0.965`, K10 `81/83 = 0.976`, false-early `0`, post-release `2/83 = 0.024`, median anchor error `2`, abstain `7/14 = 0.500`;
- R2 72-grid: highest observed abstain `9/14 = 0.643`, with substantial coverage loss; no survivors.

Scientific interpretation:

- strong evidence of a model-level selectivity deficit;
- runtime FSM expansion is unlikely to solve the problem;
- move toward SC5-v2 model repair and more clean negatives.

But formal caveats remain:

- validation NC denominator `14 < preregistered minimum 20`;
- P5 v3 feature-valid gate/CI still had residual implementation issues in the verified GitHub head;
- formal result artifacts were not yet independently frozen in GitHub.

Therefore use:

```text
RUNTIME_ONLY_REPAIR_INSUFFICIENT = STRONGLY_SUPPORTED
RUNTIME_ONLY_FORMALLY_ESTABLISHED = NOT YET
SC5_V2_PROTOCOL_AND_CODE_PREP = GO
SC5_V2_FORMAL_TRAINING = HOLD
```

---

## 3. SC5-v2 data-expansion frozen protocol

Frozen file:

`migration_audit/m1c/sc5_v2_data/SC5_V2_DATA_EXPANSION_PROTOCOL_FROZEN.json`

The split constraints are explicit:

```text
SC5-v2 train parent states: 3–22
SC5-v2 dev parent states:   23–27
legacy validation:          28–37 (read-only, confirmatory only)
future Blind-v2:            38–49 (HOLD)
```

Training-readiness minimums:

```text
train TV >= 120
train NC >= 80
dev TV >= 30
dev NC >= 20
parent-state overlap = 0
perturbation-seed overlap = 0
target resolution = 100%
initial-state SHA = 100%
trajectory SHA unique = 100%
attack_frames = 0
asset drift = 0
```

The original frozen scale was:

- primary train `160`;
- primary dev `120`;
- train reserve `80`;
- dev reserve `80`.

Leo later collected a different primary layout (`200 train + 50 dev = 250`). This requires an explicit amendment before training.

---

## 4. Latest user-reported runtime status

At approximately 2026-06-24 14:00 CST, Leo reported:

### Primary collection

- `250/250` cells have valid telemetry after repair;
- total steps: `53,215`;
- initial 7-GPU run produced `245 .done` but only `152` telemetry files because of cascading OOM;
- `98/98` missing telemetry cells were later recollected successfully;
- Teacher labels:
  - train: `159 TV / 41 NC / 200 total`;
  - dev: `36 TV / 14 NC / 50 total`.

Current gaps:

```text
train NC gap = 39
dev NC gap   = 6
```

### Protocol-violating reserve

Leo reported a reserve run using states `28–37`, with `100` planned and about `35/100` completed at the time of the report.

This reserve is **not eligible for train/dev** because states `28–37` are frozen legacy validation. The next GPT must not authorize continuation or merging.

Required disposition:

```text
RESERVE_28_37 = STOP_NEW_LAUNCHES
COMPLETED_28_37 = QUARANTINE_DIAGNOSTIC_ONLY
MERGE_INTO_TRAIN_DEV = FORBIDDEN
```

States `38–47` must also remain forbidden because they overlap future Blind-v2.

---

## 5. Critical collector audit findings

The verified GitHub head contains `scripts/stageb/run_v5_lean_collector.py`, but the server’s actually executed script may differ. Freeze the live script hash and command line before trusting any output.

### 5.1 Perturbation ordering bug

Verified GitHub code computes hashes and applies perturbation **before** calling `set_init_state(state_id)`:

```text
compute original hash
apply perturbation
compute perturbed hash
set_init_state(selected state)
```

The subsequent reset likely overwrites the perturbation. Therefore current “perturbed” hashes may not describe the actual rollout state.

Correct order:

```text
build/reset env
set selected init state
compute original state hash
apply perturbation to the selected state
sim.forward
compute perturbed state hash
dummy wait
rollout
```

### 5.2 Target resolver not fail-closed

Verified collector initializes target to `[0,0,0]` and continues when resolution fails. This violates the 100% target-resolution gate.

Required behavior:

```python
if target is None:
    raise RuntimeError("TARGET_UNRESOLVABLE")
```

### 5.3 Undefined variables in verified head

The collector writes telemetry fields from `post_qpos` and `post_eef`, but these variables were not defined in the verified file. The server must therefore be running a different local version or an uncommitted patch.

Freeze:

- actual collector path;
- actual launcher path;
- `sha256sum` of both;
- `/proc/<PID>/cwd`;
- `/proc/<PID>/cmdline`;
- `git status --short`;
- `git diff`.

### 5.4 Incomplete or misleading provenance

Verified collector does not natively record every SHA required by the frozen protocol. Its `target_resolver_sha256` is derived from the collector file itself rather than the resolver implementation.

Do not call current sidecar status “full native provenance.” Use:

```text
trajectory/file completeness = PASS
native asset provenance = PARTIAL
initial-state semantics = UNVERIFIED
```

### 5.5 `.done` is not a trustworthy completion marker

The first run produced `.done` without telemetry for many cells. Future collectors must write atomically:

```text
write cell.tmp/
write telemetry and summary
fsync
validate hashes and row counts
atomic rename to final cell directory
write .done last
```

Closure must assert:

- telemetry exists;
- summary exists;
- `.done.telemetry_sha` matches actual file hash;
- `summary.n_steps` equals telemetry row count;
- `exit_code=0`;
- one valid run UUID per cell;
- no stale partial files.

### 5.6 Manifest duplicates

The frozen primary manifests contain repeated execution-equivalent rows. `perturbation_type` is metadata only; the collector does not use it to change behavior. Several rows therefore share the same:

```text
(task, parent_state, template, seed)
```

Audit with the effective execution key:

```text
(task, parent_state, effective dx, effective dy, effective dyaw, seed)
```

Trajectory uniqueness does not prove initial-condition uniqueness.

### 5.7 Seed semantics

For perturbations P1–P6, changing `base_seed` does not change the perturbation. Only P7 is seed-dependent in the verified generator.

A new reserve must vary actual magnitude/angle, not merely seed. Suggested clean families:

```text
XY offsets: 2.5 / 5.0 / 7.5 / 10.0 mm
Yaw:        2.5 / 5.0 / 7.5 deg
Random XY:  explicit seed-dependent offsets
```

Write actual `dx`, `dy`, and `dyaw` into manifest and summary.

---

## 6. Immediate takeover checklist

Before giving any new authorization, perform these checks in order.

### A. GitHub state

1. Fetch current branch head.
2. Compare it with `d01d5ce15e9d80b779b9d50707ebcadeca320a1d`.
3. Inspect diffs affecting:
   - `run_v5_lean_collector.py`;
   - perturbation generator;
   - collection manifests;
   - Teacher labeler;
   - P5 evaluator.
4. Check whether the 250-cell amendment and any reserve amendment exist.

### B. Live server state

Ask Leo to provide or run:

```bash
ps -ef | grep -E 'run_v5_lean_collector|reserve|launch' | grep -v grep
readlink -f /proc/<PID>/cwd
tr '\0' ' ' < /proc/<PID>/cmdline
sha256sum <actual_collector> <actual_launcher>
git status --short
git diff
df -h /mnt/sdc
```

### C. Reserve control

- stop launching new `28–37` cells;
- let already-running cells finish only if stopping them risks corruption;
- quarantine completed outputs under a diagnostic-only root;
- freeze before/after hashes;
- never merge them into train/dev.

### D. Primary-250 closure

Verify all 250 cells with the stronger completion criteria above. Create:

- `SC5_V2_PRIMARY_250_AMENDMENT.json`;
- exact member manifest;
- content hashes;
- duplicate execution-key audit;
- collector/launcher provenance;
- OOM repair appendix.

---

## 7. Correct next collection design

### 7.1 Train reserve v2

Use only parent states `3–22`.

Train-only Teacher labels may be used for hard-negative mining. Freeze all cells before launch and collect the entire frozen batch; do not stop after the 39th NC.

Recommended 80-cell composition:

- 60 cells from known train-only NC parent states, balanced across tasks with a per-task cap;
- 20 cells for task0–2 and other sparse tasks to avoid task collapse.

### 7.2 Dev reserve v2

Use only parent states `23–27`.

Do **not** mine using current dev Teacher outcomes. Use a uniform, predeclared design. Suggested scale:

- `10 tasks × 4 new perturbations = 40 cells`.

Collect all 40 even if the dev NC gap is filled early.

### 7.3 Collector smoke before reserve v2

Run a 10-task smoke after patching:

- P0 original hash matches reconstructed selected state;
- P1–P7 produce changed hashes where expected;
- same state/template/seed reproduces the same perturbed hash;
- unresolved target aborts;
- no undefined telemetry fields;
- `.done` written last;
- all native SHA fields present;
- attack frames remain zero.

Do not start reserve v2 until smoke passes.

---

## 8. Training-readiness decision

Formal SC5-v2 training remains HOLD until all conditions are true:

```text
primary amendment frozen
collector semantics verified
train TV >= 120
train NC >= 80
dev TV >= 30
dev NC >= 20
train/dev parent overlap = 0
legacy validation overlap = 0
future Blind-v2 overlap = 0
effective perturbation duplicates audited
target resolution = 100%
initial-state hashes semantically correct
trajectory SHA complete and unique
asset drift = 0
attack_frames = 0
```

Allowed now:

```text
SC5_V2_PROTOCOL_DRAFT = GO
SC5_V2_IMPLEMENTATION = GO
COLLECTOR_FIX_AND_SMOKE = GO
TRAIN_RESERVE_V2 = GO_AFTER_SMOKE
DEV_RESERVE_V2 = GO_AFTER_SMOKE
```

Not allowed now:

```text
SC5_V2_FORMAL_TRAINING = HOLD
LEGACY_VALIDATION_REUSE = FORBIDDEN
BLIND_V2 = HOLD
ATTACKS = HOLD
C2 = HOLD
```

---

## 9. Correct reporting language

Use the following distinction:

```text
Primary telemetry/file completeness = PASS
Primary protocol compliance = amendment required
Primary native provenance = partial
Primary perturbation semantics = unverified until collector fix
Train NC gate = FAIL (41 < 80)
Dev NC gate = FAIL (14 < 20)
28–37 reserve = protocol violation, diagnostic-only
SC5-v2 formal training = HOLD
```

Do not claim:

- “full SHA provenance” merely because a sidecar exists;
- “unique initial conditions” from unique trajectory SHA;
- “formal independent validation preserved” after training on states 28–37;
- “future blind remains independent” after using states 38–47.

---

## 10. First recommended response to Leo

The next GPT’s first response should be concise and decisive:

1. acknowledge the 250/250 repair and P4 labels;
2. state that train NC and dev NC both fail;
3. order an immediate stop/quarantine of the 28–37 reserve;
4. forbid 38–47;
5. request the actual live collector/launcher SHA and process command;
6. patch and smoke the collector before any new eligible reserve;
7. keep formal training and all attacks on HOLD.

---

## 11. Key commits and files

Known milestones:

- `9ab9f26` — M1B closed baseline
- `827bf00` — M1C-RM route freeze
- `07bd43b` — `SC5DetectorRuntimeV1R`
- `d0b006a` — replay infrastructure
- `1df5877` — R1 freeze v2
- `b0a5bf5` — object clean protocol
- `853336d` — V4 closure
- `96f23ad` — P3A PASS / P3B PARTIAL
- `75dcfe4` — P4/P5 major corrections
- `a6bbc4c` — empty opening-proxy fix
- `8000a247` — P5 v3 follow-up
- `d01d5ce` — latest verified branch head during handoff

Critical files:

- `migration_audit/m1c/sc5_v2_data/SC5_V2_DATA_EXPANSION_PROTOCOL_FROZEN.json`
- `migration_audit/m1c/sc5_v2_data/T2_TRAIN_PRIMARY_MANIFEST.csv`
- `migration_audit/m1c/sc5_v2_data/T2_TRAIN_RESERVE_MANIFEST.csv`
- `migration_audit/m1c/sc5_v2_data/V2_DEV_PRIMARY_MANIFEST.csv`
- `migration_audit/m1c/sc5_v2_data/V2_DEV_RESERVE_MANIFEST.csv`
- `scripts/stageb/run_v5_lean_collector.py`
- `src/gripper_attack/v5_perturbation.py`
- `scripts/migration/label_m1c_object_teacher.py`
- `scripts/migration/evaluate_m1c_fsm_validation.py`

---

## 12. Bottom line

The project has enough evidence to justify model-level SC5-v2 work, but not enough clean, protocol-compliant negative data to begin formal training. The highest-priority task is **not** expanding into new state ranges. It is:

```text
stop and quarantine the invalid 28–37 reserve
freeze the actual server code provenance
fix collector semantics
smoke-test perturbations and hashes
collect eligible train reserve inside 3–22
collect eligible dev reserve inside 23–27
pass all data-readiness gates
then train SC5-v2
```
