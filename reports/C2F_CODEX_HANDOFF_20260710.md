# C2f Codex Handoff — Audit First, Execute Second

**Date:** 2026-07-10  
**Repository:** `Leo-6-maker/openvla-gripper-dutycycle-attack`  
**Recommended starting ref:** `750afdb`  
**Primary rule:** **Do not start new GPU experiments until the audit phase below is completed and written to a report.**

---

## 0. Mission for Codex

Take over the C2f OpenVLA-SigLIP detector and online canary line, verify the current evidence from raw artifacts, resolve protocol/provenance issues, then continue with the smallest experiments needed to determine whether C2f deserves a formal secondary online table.

The immediate objective is **not** to replace the frozen D7 Table 1. The immediate objective is to answer:

1. Is the reported Object-suite TRUE/RAND gap real under paired analysis?
2. Is the current online experiment a detector-triggered **command-space intervention** or a D7-comparable **image-space PGD attack**?
3. Are all 144 episodes valid and generated under a homogeneous final protocol?
4. Does Goal recover when rerun with the authentic Goal checkpoint?
5. Is Spatial’s 100% emit behavior genuine phase coverage or detector/label over-emission?

Codex must first produce an audit report. Only after the audit gates pass may it launch new jobs.

---

## 1. Current state

### 1.1 Frozen main attack evidence

```text
D7_TABLE1 = FROZEN_MAIN_ATTACK_RESULT
```

Do not modify, overwrite, reinterpret, or replace D7 Table 1 during this handoff. C2f is a detector/materialization upgrade and a candidate secondary online validation line.

### 1.2 Full offline C2f ablation

Reference report:

```text
reports/C2F_SIGLIP_FULL_ABLATION_20260709.md
commit: fbaa36f
```

Full Clean2000 / 363,513-window A/B/C/D result:

```text
D Full:       Recall 97.7%, FP 4.2%, F1 0.954
C Visual:     Recall 96.9%, FP 3.9%, F1 0.952
A 25D base:   Recall 93.3%, FP 4.6%, F1 0.927
B Language:   Recall 92.1%, FP 5.0%, F1 0.918
```

Interpretation boundary:

- Full D vs A gain is +4.4 pp recall.
- Visual-only C vs A is the clean visual contribution: +3.6 pp recall.
- Language-on-top-of-visual D vs C is +0.8 pp overall recall, with the largest apparent gain on Object.
- Language-only B is weaker than A.
- This offline ablation supports the detector architecture; it does not prove online attack success.

### 1.3 Online canary v1

Reference report:

```text
reports/C2F_CANARY_V1_RESULTS_20260710.md
commit: 750afdb
run root:
/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106
```

Reported matrix:

```text
48 parents × 3 conditions = 144 episodes
12 parents per suite
conditions: CLEAN / TRUE_T10 / RAND_T10
D Full detector default gate:
  tau_emit    = 0.33
  tau_suppress= 0.67
  tau_abstain = 0.5
  tau_primary = 0.5
```

Reported outcomes:

```text
Object:
  CLEAN 10/12 = 83%
  TRUE   6/12 = 50%
  RAND  10/12 = 83%
  nominal TRUE-RAND gap = -33 pp

Spatial:
  CLEAN 11/12 = 92%
  TRUE  11/12 = 92%
  RAND  10/12 = 83%
  100% emit, no negative TRUE/RAND gap

L10:
  CLEAN 5/12 = 42%
  TRUE  4/12 = 33%
  RAND  4/12 = 33%
  no gap

Goal:
  CLEAN/TRUE/RAND = 0/0/0
  invalid because a libero-10 substitute policy was used
```

Current interpretation:

```text
C2F_CANARY_V1_144EP = COMPLETE_DIAGNOSTIC
C2F_OBJECT_SIGNAL = STRONG_PRELIMINARY
C2F_GOAL = INVALID_SUBSTITUTE_MODEL_PENDING_RERUN
C2F_SPATIAL = OVER_EMIT_OR_LABEL_BROAD / NO_GAP
C2F_L10 = INCONCLUSIVE_LOW_BASE_SR / NO_GAP
```

Do not call the Object gap statistically significant until the exact paired table and test are recomputed from raw artifacts.

---

## 2. Important commits and bug history

Review ancestry and actual diffs; do not trust commit messages alone.

```text
fbaa36f  full SigLIP A/B/C/D ablation report
8b5f19c  initial C2f runtime + worker + launcher
12dbc12  checkpoint/gate/T10/concurrency hardening
c116b01  RGB float [0,1] -> uint8 conversion fix
7c02187  attack API workaround using direct action manipulation
172b78d  write metadata before env.close()
f3c9fc0  catch episode-loop exceptions to survive EGL failures
1c181f8  Goal unnorm-key workaround
1616f52  queue auto-retry / final overnight launcher state
750afdb  canary-v1 result report and fill scripts
```

Known P0 bug history:

1. RGB float image was cast directly to uint8, producing near-black images and near-zero detector outputs.
2. The attempted visual attack API did not accept the supplied `epsilon` argument.
3. EGL teardown could destroy evidence before metadata was written.
4. Episode-loop errors could leave missing artifacts.
5. Goal used an invalid substitute model and a temporary unnormalization-key workaround.

Old runs before the RGB fix must not be used.

---

## 3. Critical protocol caveat

The current worker at `750afdb` does not implement the original D7 image-space PGD protocol.

Current implementation:

```python
TRUE_T10:
    attack_action[-1] = 1.0

RAND_T10:
    noise = np.random.randn(*attack_action.shape)
    attack_action = clip(attack_action + normalized_noise * (6/255))
```

Therefore the scientifically accurate names are closer to:

```text
TRUE_CMDOPEN_T10_C2f
RAND_ACTION_NOISE_T10_C2f
```

The current canary is evidence for a detector-triggered command/action-space gripper-opening intervention. It is **not automatically comparable** to D7 TRUE_T10 image-space PGD.

Codex must make an explicit protocol decision after the audit:

### Track A — Command-space secondary table

Use the current intervention family, rename conditions accurately, freeze a deterministic implementation, rerun invalid/required suites, and replicate Object.

### Track B — D7 protocol-parity online attack

Reuse the exact D7 image-space attack helper and preprocessing path, smoke-test it with C2f triggering, then run a separate paired matrix. Do not mix Track A and Track B results.

Do not silently call Track A “D7-compatible PGD.”

---

## 4. Hard boundaries

Codex must preserve all of the following:

1. D7 Table 1 remains frozen.
2. OpenVLA-SigLIP must not be called CLIP.
3. Language input is Llama input-token embedding mean-pool, not a full LLM forward.
4. Student inputs must not include privileged object pose, target pose, object-target distance, outcome labels, or post-attack hidden states.
5. Goal substitute-model episodes must not enter final pooled results.
6. Runtime-error episodes must not be counted as attack failures.
7. Thresholds must not be tuned on the same online test parents and then reported as preregistered results.
8. Default-gate and best-F1 metrics must not be mixed.
9. New runs must use new roots; never overwrite the original canary artifacts.
10. Any code fix after a partial run requires a provenance audit and rerun of affected jobs under one frozen final commit.

---

## 5. Audit phase — mandatory before any new jobs

Create:

```text
reports/C2F_CANARY_V1_AUDIT_20260710.md
```

The report must end with explicit GO/HOLD decisions for Goal rerun, Object replication, Spatial audit, and any D7-parity run.

### 5.1 Repository and artifact provenance

Run and record:

```bash
cd /mnt/sdc/dty_user/openvla_attack
git rev-parse HEAD
git status --short
git log --oneline --decorate -20
git diff 1616f52..750afdb -- scripts/stageb/run_c2f_canary_worker.py
```

Verify and record SHA256 for:

```text
C2f checkpoint
parent manifest
source D7 parent CSV
runtime file
worker file
launcher/fill scripts
postrun_audit.json
summary tables
```

Important: inspect the per-episode `git_commit` field. Determine whether the 144 episodes were generated under one final code path or a mixture of commits/fixes.

Required output:

```text
n episodes per git_commit
n episodes per suite / condition / git_commit
n episodes with missing git_commit
```

If protocol-affecting code changed between groups, do not treat the 144 episodes as one homogeneous experiment. Identify and rerun affected jobs under one frozen commit.

### 5.2 Completeness and duplication audit

Verify from raw artifacts:

```text
metadata files = 144
step_records files = 144
unique parents = 48
per condition = 48 CLEAN / 48 TRUE / 48 RAND
each parent has all 3 conditions
duplicate parent-condition keys = 0
missing keys = 0
```

Do not rely only on the existing postrun audit; independently recompute.

### 5.3 Runtime-validity audit

The worker currently catches broad exceptions and writes an error record. Therefore “metadata exists” does not imply “valid episode.”

Scan all step records for:

```text
"error"
step = -1
traceback-derived messages
zero/abnormally short trajectories
EGL errors
CUDA OOM
blank RGB errors
```

Create fields:

```text
runtime_valid
error_type
error_message
last_valid_step
```

Required result:

```text
n valid episodes
n runtime-error episodes by suite/condition
```

Any runtime-error episode must be excluded from outcome denominators or rerun. Never count it as `success=false` attack evidence.

### 5.4 Model-path and unnormalization audit

For every suite, record:

```text
resolved model path
checkpoint/model file hashes
processor path
norm_stats keys
chosen unnorm_key
```

Goal is invalid in canary v1 because a libero-10 substitute was used. Confirm the authentic Goal model is complete before rerunning.

Do not keep the current hard-coded Goal `unnorm_key="libero_10"` without inspecting the authentic Goal checkpoint’s normalization-stat keys.

### 5.5 Detector input parity audit

Compare online runtime against full materialization for:

```text
RGB extraction and dtype/range
SigLIP image preprocessing
vision-backbone pooling
L2 normalization
language tokenization / max length / mean-pool
108D context construction
25D feature ordering
EEF velocity calculation
window length and endpoint semantics
four-condition gate
```

Write a compact parity table with PASS/FAIL for each component.

### 5.6 TRUE/RAND pre-trigger parity

TRUE and RAND rollouts for the same parent should be identical before the first detector trigger if the environment/model path is deterministic.

Audit where possible:

```text
first emit step
emit/suppress/abstain/primary traces before first emit
clean policy actions before first emit
trajectory length before first emit
```

Required questions:

1. Do TRUE and RAND emit on the same Object parents?
2. Do they emit at the same step?
3. Are the pre-trigger score traces equal within numerical tolerance?
4. If not, is the discrepancy due to nondeterminism, RNG leakage, or mixed code?

If the required raw fields were not recorded, state that limitation and add instrumentation before replication.

### 5.7 RAND reproducibility audit

The current worker uses `np.random.randn()` without a recorded deterministic per-job seed.

Before any replication:

- derive a stable seed from `sha256(parent_key | condition | base_seed)`;
- use a local NumPy generator rather than global RNG;
- record the seed in metadata;
- never reuse TRUE and RAND output directories across reruns.

The original canary may remain diagnostic, but new results must be reproducible.

### 5.8 Delivery audit

For TRUE and RAND, compute:

```text
first attack step
delivery_count
delivery step list
full 10-step delivery rate
truncated-delivery rate
reason for truncation: success, done, crash, max-step, other
```

Object reported mean delivery count is below 10. Determine whether the lower count is caused by valid early episode termination, runtime errors, or bookkeeping.

### 5.9 Recompute all outcomes from raw artifacts

Do not copy the report table. Recompute:

```text
CLEAN SR
TRUE SR
RAND SR
emit rate
delivery rate
median/mean first-emit step
full-delivery rate
```

Report per suite and pooled only over valid comparable episodes.

For CLEAN emit rate, use `step_records.jsonl`; CLEAN metadata does not encode attack-window start.

### 5.10 Object paired analysis

Build the exact 12-row parent table:

```text
parent_key
CLEAN success
TRUE success
RAND success
TRUE emit
RAND emit
TRUE first emit
RAND first emit
TRUE delivery count
RAND delivery count
runtime-valid flags
```

Compute:

```text
TRUE fail / RAND success = b
RAND fail / TRUE success = c
exact two-sided McNemar p
exact one-sided McNemar p, clearly labeled exploratory
paired risk difference
bootstrap confidence interval at parent level
```

Also compute a delivered-pair subset only when TRUE and RAND both delivered. Do not replace the all-parent analysis with the delivered subset; show both.

Do not describe the gap as statistically significant unless the exact test supports it.

### 5.11 Spatial audit

Spatial has 100% emit with no harmful TRUE/RAND separation.

Audit:

```text
first emit distribution
number and duration of emitted segments
primary_p / abstain_p around first emit
manual phase category for a small sample
comparison against clean teacher labels if available offline
```

Determine whether this is:

- correct broad primary-phase coverage;
- label-driven over-emission;
- context shortcut;
- or a payload-insensitive suite.

Do not tune the gate on these same 12 parents.

### 5.12 L10 audit

Verify the claim that tasks `00/01/06/07` have zero primary labels in the exact training dataset and inspect which L10 tasks were sampled in the online manifest.

Report task-level:

```text
parent count
clean SR
emit rate
TRUE/RAND outcomes
training primary-window rate
```

Treat L10 as inconclusive unless a paired signal appears on valid, attackable tasks.

---

## 6. Audit GO/HOLD gates

### Goal rerun = GO only if

```text
authentic Goal model is complete
all model shards hash/read successfully
processor loads locally
norm_stats key is resolved from the actual model
1-parent Goal CLEAN smoke completes
CLEAN action outputs are finite and plausible
no runtime error
```

### Object replication = GO only if

```text
original Object raw paired table is verified
runtime-error episodes are removed/rerun
TRUE/RAND pre-trigger pairing is acceptable
protocol is explicitly named and frozen
RAND RNG is deterministic
final worker commit is frozen
```

### Spatial expansion = HOLD by default

Do not spend large GPU budget on more Spatial episodes until the 100% emit/no-gap behavior is explained.

### Full C2f Table1 claim = HOLD

Remain HOLD until:

```text
Goal authentic-model rerun completes
Object gap is replicated or statistically supported
protocol naming is corrected
runtime-validity and provenance audits pass
```

---

## 7. Execution phase after audit

### 7.1 Freeze a final worker before reruns

Recommended additions to metadata:

```text
protocol_id
runtime_valid
error_type
error_message
model_path
model_sha256 or manifest hash
processor path
unnorm_key
rng_seed
first_emit_step
emitted_step_count
delivery_steps
termination_reason
RGB dtype/min/max/mean for first frame
worker commit
checkpoint sha256
```

Recommended condition names:

```text
CLEAN
TRUE_CMDOPEN_T10_C2f
RAND_ACTION_NOISE_T10_C2f
```

If implementing D7 image-space parity, use a completely separate protocol id and output root.

### 7.2 Authentic Goal rerun

Use a new root, for example:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2f/goal_realmodel_rerun_<timestamp>
```

Sequence:

1. Goal model integrity audit.
2. One CLEAN smoke parent.
3. One paired TRUE/RAND smoke parent.
4. Three-parent mini matrix: 3 parents × 3 conditions.
5. If all pass, run the frozen 12 Goal parents × 3 conditions = 36 episodes.
6. Produce a separate Goal report before merging any summary.

Never overwrite substitute-model Goal artifacts.

### 7.3 Object replication

Prefer 24 new Object parents if available; otherwise use the largest disjoint set supported by the D7 preregistered pool.

Recommended matrix:

```text
new Object parents × CLEAN / TRUE_CMDOPEN_T10 / RAND_ACTION_NOISE_T10
same detector checkpoint
same default gate
same final worker commit
same attack horizon
stable per-job RNG
```

Primary endpoint:

```text
paired TRUE vs RAND success difference
```

Secondary endpoints:

```text
emit coverage
paired first-emit parity
full-delivery rate
manual/contact failure on discordant pairs
```

### 7.4 Optional D7-parity track

Only after Track A is stabilized:

1. Identify the exact D7 image-preprocessing and PGD injection helper.
2. Write a parity test against frozen D7 parameters.
3. Run one-parent CLEAN/TRUE/RAND smoke.
4. Run Object 12-parent paired canary.
5. Keep the output and report separate from command-space results.

### 7.5 Spatial diagnostic, not large expansion

Use existing 12 parents plus a small manually reviewed sample. Do not launch a full new Spatial matrix merely to increase N until the detector’s 100% emission behavior is understood.

---

## 8. Expected deliverables

Codex should produce, in order:

1. `reports/C2F_CANARY_V1_AUDIT_20260710.md`
2. A deterministic, provenance-complete worker patch, if required.
3. `reports/C2F_GOAL_REALMODEL_RERUN_<date>.md`
4. `reports/C2F_OBJECT_PAIRED_REPLICATION_<date>.md`
5. `reports/C2F_SPATIAL_EMIT_AUDIT_<date>.md`
6. A final decision report:
   - command-space secondary table ready;
   - D7-parity run required;
   - or C2f online claim held.

Every report must include:

```text
run root
exact commit
checkpoint hash
model path/hash
parent manifest hash
protocol id
thresholds
valid/invalid episode counts
raw paired contingency
caveats and boundaries
```

---

## 9. Recommended commit discipline

Use small commits with explicit scope, for example:

```text
audit(C2f): verify canary-v1 provenance and paired Object table
fix(C2f): record runtime validity and deterministic RAND seed
fix(C2f): resolve Goal model norm key from authentic checkpoint
exp(C2f): add frozen Goal real-model rerun manifest
report(C2f): Goal real-model paired results
report(C2f): Object replication and exact paired analysis
```

Do not combine code fixes, experiment outputs, and interpretive claims into one opaque commit.

---

## 10. Required first response from Codex

Before running anything, Codex should reply with an audit summary in this format:

```text
AUDIT STATUS
- repository/provenance:
- episode completeness:
- runtime-valid episodes:
- mixed-commit/protocol risk:
- model-path/unnorm-key status:
- TRUE/RAND pre-trigger parity:
- Object paired contingency and p-value:
- Spatial interpretation:
- Goal rerun readiness:

GO/HOLD
- Goal real-model rerun:
- Object replication:
- Spatial expansion:
- D7-parity experiment:

FILES/COMMITS TO CHANGE
- ...

EXPERIMENTS TO LAUNCH AFTER APPROVAL
- ...
```

If any required raw artifact is missing, Codex must state exactly what is missing and stop before launching new experiments.

---

## 11. Final scientific wording boundary

Safe current statement:

> The full offline C2f detector ablation supports a strong visual contribution. In the first online command-space canary, Object showed a large preliminary TRUE/RAND gap, while Spatial and L10 did not. Goal was invalid because the authentic suite checkpoint was unavailable. The Object result requires raw paired audit and replication before inclusion in a formal secondary table.

Unsafe statements:

```text
C2f replaces D7 Table1.
C2f visual PGD is proven online.
The Object -33pp result is statistically significant.
All four suites are healthy.
Goal is a negative result.
Spatial 100% emit proves perfect detector transfer.
```

---

## 12. Current recommended priority order

```text
P0  Raw canary-v1 provenance/runtime/paired audit
P0  Authentic Goal model integrity + one-parent smoke
P0  Fix deterministic RNG and runtime-validity metadata
P1  Goal 36-episode real-model rerun
P1  Object paired exact test and disjoint-parent replication
P1  Spatial emit/label diagnostic
P2  Decide command-space secondary table vs D7 PGD-parity track
P2  Final C2f secondary-table decision report
```

**Do not launch more suites or tune thresholds before completing P0.**
