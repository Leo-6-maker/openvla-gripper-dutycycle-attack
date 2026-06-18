# Codex Full Handoff — OpenVLA Gripper Duty-Cycle Attack

**Date:** 2026-06-18  
**Repository:** `Leo-6-maker/openvla-gripper-dutycycle-attack`  
**Working branch:** `exp/l2-sc5-data-census-tcn-v2-20260618`  
**Expected branch head:** `a53df7efc93be8b960ce42ecc92a2c9f43cd2ceb`  
**Primary tracking issue:** `#28`

---

## 0. Your role

You are taking full ownership of the project from this point onward.

Operate autonomously, but do not silently change frozen experimental definitions. Work gate-by-gate, preserve provenance, and stop when a gate fails. Prefer mature, already-debugged code over new parallel implementations.

For every proposed new file, state:

1. which mature file is reused;
2. the smallest semantic delta added;
3. why a wrapper/new file is necessary;
4. which frozen files remain untouched.

Do not mix data changes, label changes, model changes, trigger changes, and attack changes in one commit.

---

# 1. Scientific objective and current claim boundary

The project studies inference-time gripper duty-cycle vulnerability in OpenVLA/LIBERO.

The narrow current result is:

> A clean-phase-aligned forced-OPEN intervention applied at a stable-carry-derived window is substantially more consequential than the old D5/grasp-close trigger.

The current strongest command-level rule is frozen as:

```text
SC5 = earliest stable_carry_start + 5
K = 10 policy steps
```

The project does **not** yet establish:

- that the deployment-safe Layer-2 detector can reliably select a full valid SC5 K10 start corridor across broad tasks;
- that the frozen visual perturbation attack reproduces the command-level SC5 effect;
- arm-selective gripper attribution for the VIS attack;
- broad cross-task or cross-suite generalization;
- a universal exact physical failure boundary.

Always separate:

- **observed**;
- **proven by an audited gate**;
- **inferred/hypothesized**;
- **not established**.

---

# 2. Frozen gripper semantics

These semantics are critical and were previously corrected after an inversion bug:

```text
raw gripper <= 0.5  = CLOSE
raw gripper >  0.5  = OPEN

env gripper > 0 = CLOSE
env gripper < 0 = OPEN
```

For command-hold OPEN:

```python
executed_action = clean_env_action.copy()
executed_action[-1] = -1.0
```

Only the final env gripper dimension may change.

Any raw/env semantic inconsistency must fail closed.

---

# 3. Historical experimental progress

## 3.1 Phase 0 — D5 Layer-3 temporal semantic result

Auditor-v3 commit:

```text
0bf2f4d77e2859356b995e7ba06a8cd5d9b70117
```

Final classification:

```text
L3_D5_VIS_TEMPORAL_SEMANTIC_COMMAND_DUTY_PASS
PHYSICAL_RESPONSE_OBSERVED
ARM_SELECTIVITY_NOT_ESTABLISHED
TASK_EFFECT_NOT_PROVEN
```

Key gates:

- G0 provenance: PASS;
- G1 semantic OPEN token duty: PASS;
- G2 arm selectivity: FAIL;
- G3 env OPEN duty: PASS;
- G4 paired physical response: PASS;
- G5 task effect: not proven.

The old D5 timing could shape OPEN duty, but non-gripper arm tokens collapsed and task failure was not established.

Do not overstate this result.

---

## 3.2 Phase 1 — data inventory and Teacher correction

Inventory commit:

```text
8c9031ae2273c95944b72337509581900cfe3f17
```

Prior inventory found approximately:

- 3,002 artifact-rich trajectories;
- 1,278 known-task trajectories;
- only 30 Butter runs with states 0–9;
- exact Butter s11 absent before the one-off canary;
- 8 clean-success Butter states in the initial set.

Teacher/canary commits:

```text
c5c1a67c6e08beb1e2a53c354e564341eb5a1c4e

d77fc4baee4bafe3cd2994e542df7ddfb550dd8f

b417f0eb47c011a080108580439d4bd201043025
```

`b417f0e` is the important semantic correction.

Correct Butter s11 clean timeline:

```text
step  0–59   approach / OPEN
step 60      grasp_close / D5 first CLOSE
step 62      stable_grasp
step 83      stable_carry / first lift
step 88      stable_carry + 5 = SC5
step 121–159 pre-place oscillation
step 160     release_safe
```

Interpretation:

- D5 is contact onset / first CLOSE;
- D5 is not the failure-critical carry phase;
- stable-carry timing is the correct direction;
- this was initially clean-only phase evidence, not attack evidence.

---

# 4. Phase 3 — audited command-hold timing result

## 4.1 Pilot and final commits

Pilot commit:

```text
a95f919b20b348c80bb5a54d652268b5a59ce8f3
```

Final Phase-3 commit:

```text
e7c33552a3ff0efc4b2013e752390f6cc771e167
```

Frozen rule:

```text
SC5 = earliest stable_carry_start + 5
K = 10
```

Final audited state matrix:

| State | Role | CLEAN | D5 | SG5 | SC5 |
|---|---|---:|---:|---:|---:|
| s0 | dev | 140 | 347 success | 400 fail | 400 fail |
| s2 | dev | 158 | 158 success | 248 success | 400 fail |
| s7 | dev | 149 | 178 success | 181 success | 400 fail |
| s8 | held-out | 160 | 400 fail | 400 fail | 400 fail |
| s9 | held-out | 157 | 400 fail | 213 success | 400 fail |
| s3 | supplementary | 152 | 179 success | 180 success | beyond natural episode |
| s5 | excluded | 400 clean failure | 400 | 400 | 400 |

Valid-state comparison:

```text
SC5: 5/5 task failures
D5:  2/5 task failures
SG5: 2/5 task failures
```

SC5 was never weaker than D5 or SG5 in a valid state and produced additional failures in three states.

Allowed claim:

```text
SC5_COMMAND_HOLD_TIMING_RULE_PASS
CROSS_STATE_FAILURE_CONSISTENCY_OBSERVED
D5_STATE_SENSITIVITY_OBSERVED
```

Not allowed:

- universal LIBERO failure boundary;
- SC5 visual-attack success;
- deployment-detector success;
- arm-selective VIS attribution.

### Important documentation correction

The `e7c3355` commit message says `D5: 0/3 dev recover`, which is backwards. The matrix shows D5 recovered/succeeded on all three dev states s0/s2/s7.

Correct wording:

```text
D5: 3/3 dev recover
```

or:

```text
D5: 0/3 dev fail
```

Do not propagate the incorrect sentence.

---

# 5. GPU / EGL operational facts

Only use verified ordered GPU/render mappings.

Known-good pairs:

```text
CUDA_VISIBLE_DEVICES=1,5 ; render_gpu_device_id=5
CUDA_VISIBLE_DEVICES=2,6 ; render_gpu_device_id=6
```

The root cause of prior EGL failures was mixing CUDA-local ordinals with physical EGL GPU indices.

Rule:

```text
render_gpu_device_id is the physical GPU index,
not the CUDA_VISIBLE_DEVICES remapped local ordinal.
```

Avoid unreliable/damaged GPUs:

```text
GPU0 hardware damaged/unreliable
GPU3 EGL unreliable
```

Reuse `libero_v4_env_factory.py`; do not create another environment factory.

---

# 6. Layer 1 / Layer 2 redesign history

## 6.1 Audit commit

```text
018b15ffef5748efa489f887692dd1acad034146
```

Findings:

- frozen D5 is a CLOSE-onset candidate detector;
- D5 cannot continuously monitor stable carry;
- the legacy Proprio student was an MLP, not a TCN;
- the legacy model used `normalized_step`;
- replay was row-level and lacked first-trigger lock / K10 validation;
- old Teacher anchor preferred pre-place;
- SC5 required a new explicit anchor function.

Keep D5 frozen as a baseline.

---

## 6.2 Layer 1 implementation commit

```text
c52b106a96e7cefa7d510d29394f8f22fec40c96
```

Added:

- `find_sc5_anchor_v2()`;
- continuous per-step SC5 streaming features;
- SC5 labels and first dataset builder.

Initial dataset:

- 81 clean-success episodes;
- 12,695 rows;
- one LIBERO-object subset;
- all 81 had valid SC5 corridors;
- no genuine no-corridor negatives.

---

## 6.3 First Layer 2 MLP commit

```text
5d1f18da913658f02c4d19f1682a384600d0a1a3
```

Initial results used flawed containment semantics:

```text
coverage = 0.963
false early = 0
median anchor error = 1
window containment = 0.321
```

The 0.321 metric was effectively exact-anchor matching, not true valid-start containment.

Do not use the old Gate result.

---

## 6.4 P0 correction commit

```text
bdccf366b10659ef02563bb3f554c72a7a1a3f48
```

Fixes:

- `action_dy` no longer duplicated from `action_dz`;
- fail-closed field checks;
- held-out exclusion from calibration/training;
- content-based dedup;
- actual valid K10 start corridor;
- separate labels for exact SC5, attack window, corridor, and full-K10 validity;
- best validation checkpoint saving;
- proper replay containment semantics.

Corrected 81-episode MLP result:

```text
coverage                 1.000
false early              0.000
median absolute error    1.0 step
valid K10 containment    0.654
exact anchor match       0.198
held-out phase accuracy  73.2%
```

Interpretation:

- the MLP has strong SC5 timing signal;
- it often misses the narrow valid-start corridor by 1–2 steps;
- this does not yet satisfy the 0.85 containment gate;
- before blaming model capacity, the dataset needed expansion.

---

# 7. Exhaustive data census

## 7.1 Census commits

```text
4475d7afa98c45b6456a06c8916778dc43005871

a53df7efc93be8b960ce42ecc92a2c9f43cd2ceb
```

Observed server-wide counts:

```text
8,879 directories scanned
3,008 step_records.jsonl
2,997 manifests
798 clearly identified clean-success
475 clean-fail
1,735 initially unknown task names
```

Deep manifest analysis estimated candidate pools:

| Category | Candidate count | Initial SC5 interpretation |
|---|---:|---|
| LIBERO-object pick-and-place | 836 | primary candidate |
| LIBERO-10 multi-object | ~200 | conditional; must segment events |
| LIBERO-90 place tasks | ~90 | conditional; validate object/target identity |
| drawer/stove/push/articulation | ~200 | incompatible; OOD abstain only |
| clean-fail | ~475 | audit only |

Total discovered:

```text
3,008
```

Candidate SC5-compatible before canonical validation:

```text
~1,100
```

## 7.2 Critical caveat

The candidate counts are not yet the final usable dataset.

They still require:

- explicit clean provenance;
- schema normalization;
- unique transported-object validation;
- multi-stage event segmentation;
- full sequence hashing and dedup;
- train-only Teacher calibration;
- strict held-out enforcement;
- phase-order validation;
- valid SC5 corridor generation.

At the time of this handoff, a comparison of `bdccf36..a53df7e` showed only `reports/V2_SC5_PRE_TCN_CODE_STATUS.md` tracked in Git. The census scripts/tables/manifests were not yet reproducibly frozen in the repository.

Therefore the first Codex action is to verify the actual branch contents and freeze the census implementation/artifacts before canonical corpus construction.

---

# 8. Current branch / commit state

Expected branch history:

```text
018b15f audit(layer12): code audit
c52b106 feat(layer1): SC5 labels + streaming features + dataset
5d1f18d feat(layer2): MLP training + causal replay (old flawed metric)
bdccf36 fix(layer12): action_dy, corridor, held-out, fail-closed
4475d7a audit(data): exhaustive SC5 source census
a53df7e audit(data): deep manifest analysis
```

Before changing code, run:

```bash
git status --short
git branch --show-current
git log --oneline --decorate -12
git diff bdccf36..a53df7e --stat
```

Confirm:

- the active branch is the expected data-census branch;
- no uncommitted changes exist;
- the latest commits are present;
- frozen files have not changed;
- census scripts/tables/manifests actually exist or are missing as previously observed.

---

# 9. Frozen files and definitions

Do not modify:

```text
src/gripper_attack/d5_frozen_*.py
src/gripper_attack/attack_adapter.py
scripts/stageb/run_l3_d5_vis_temporal.py
scripts/stageb/audit_l3_d5_vis_temporal_v3.py
Phase 3 command-hold artifacts/configs/reports
```

Do not change:

```text
SC5 guard = 5
K = 10
raw/env gripper semantics
action postprocessing
OpenVLA model loading
environment construction
GPU physical render mapping
Phase-3 state/result matrix
held-out Butter states s8/s9/s11
```

Do not use Phase-3 attack outcomes as Layer-2 labels.

---

# 10. Strict held-out policy

Butter states:

```text
s8
s9
s11
```

must not enter:

- Teacher calibration;
- feature normalization;
- training;
- validation;
- early stopping;
- threshold tuning;
- architecture selection;
- feature selection.

`s5` is excluded because CLEAN failed.

`s3` is supplementary/no-valid-corridor evidence and must not be converted into a normal positive episode.

Duplicate groups, initial-state groups, and parent episodes of segmented events may not cross splits.

---

# 11. Immediate next task — freeze census and build canonical corpus

## 11.1 Commit A — census freeze

Create one small commit before canonicalization:

```text
fix(data): freeze census implementation and reconcile candidate counts
```

Required tracked artifacts:

```text
scripts/stageb/inventory_all_sc5_sources_v2.py
configs/v2_sc5_data_roots.yaml
configs/v2_sc5_schema_aliases.yaml
reports/V2_SC5_SOURCE_CENSUS.md
tables/v2_sc5_source_roots.csv
tables/v2_sc5_episode_inventory.csv
tables/v2_sc5_exclusion_reasons.csv
artifacts/v2_sc5_source_inventory.json
```

Do not commit raw trajectories, images, videos, or model checkpoints.

The report must reconcile:

```text
3,008 discovered
798 known clean-success
475 clean-fail
1,735 initially unknown
~1,100 candidate compatible
```

Every count must be traceable to episode rows and exclusion reasons.

---

## 11.2 Commit B — canonical sequence corpus

Commit message:

```text
feat(data): canonicalize and deduplicate expanded SC5 sequence corpus
```

### 11.2.1 Task tiers

**Tier A — primary positive candidates**

- single-object;
- one grasp/lift/carry/release chain;
- clean success;
- unique transported object;
- pick-and-place / single transfer.

**Tier B — conditional place candidates**

- bowl-on-stove;
- wine-on-rack;
- mug/bowl placement;
- other single-object place tasks.

Only include if object identity, target/support identity, and stable-carry semantics are validated.

**Tier C — multi-object / multi-stage**

Examples:

- many LIBERO-10 tasks;
- moka-like tasks;
- repeated grasp/release episodes;
- multiple stable-carry phases.

Do not assign one global first-SC5 label to the full episode.

Either:

1. segment each auditable grasp→lift→carry→release event;
2. bind the event to one unique transported object;
3. assign an event-local SC5 anchor;
4. keep K10 inside the event boundary;

or classify the episode as:

```text
OOD_MULTI_STAGE_ABSTAIN
```

**Tier D — incompatible mechanisms**

- drawer;
- stove turning;
- push;
- button;
- articulated-only manipulation;
- tasks without unsupported carry.

Use only for OOD/abstain evaluation.

---

## 11.3 Canonical schema

Every training step must produce the deployment-safe inputs:

```text
gripper_command
gripper_qpos
gripper_opening_proxy
eef_x/eef_y/eef_z
eef_vx/eef_vy/eef_vz
action_dx/action_dy/action_dz
action_gripper
```

plus the current 25D causal feature vector.

Each field must record its source:

```text
direct
vector_extracted
causally_derived
missing
ambiguous
```

Allowed derivations:

- backward-difference EEF velocity from current/past EEF positions;
- opening proxy from q7/q8;
- action dimensions from a clean action vector.

Forbidden:

- missing→0;
- using attacked/executed action as clean action;
- using privileged object state as Student input;
- using future rows;
- using task/state/run identity as Student features;
- using normalized or absolute episode step.

---

## 11.4 Strong dedup

Compute:

```text
trajectory_content_sha256
proprio_sequence_sha256
privileged_sequence_sha256
initial_state_sha256
source_file_sha256
```

Hash the full canonical sequence, not only directory names or the first few frames.

When duplicate copies exist, prefer:

1. complete provenance;
2. complete privileged state;
3. newest compatible schema;
4. complete hash/manifest.

All duplicate/init-state group members must stay in the same split.

---

## 11.5 Teacher labels

Split groups first; calibrate Teacher on train groups only.

Frozen labels per step/event:

```text
teacher_phase
teacher_stable_carry_start
teacher_sc5_anchor
teacher_sc5_ready
teacher_sc5_attack_window_active
teacher_sc5_corridor_active
teacher_full_k10_valid_at_t
teacher_release_safe
teacher_recovery
teacher_abstain
```

Frozen phase order for positive episodes:

```text
grasp_close
→ stable_grasp
→ first_lift
→ stable_carry
→ release_safe
```

Reject or abstain on incompatible phase order.

Frozen SC5 definition:

```text
anchor = earliest stable_carry_start + 5
K = 10
```

Do not move guard/K after observing model performance.

---

## 11.6 Corpus classes

Build four explicit groups:

```text
PRIMARY_SC5_POSITIVE
NO_CORRIDOR_NEGATIVE
OOD_ABSTAIN
EXCLUDED_AUDIT_ONLY
```

`NO_CORRIDOR_NEGATIVE` and `OOD_ABSTAIN` are important. The original 81-episode set contained only valid SC5 positives and was insufficient for abstain learning.

---

## 11.7 Canonical corpus outputs

Required code/config:

```text
src/gripper_attack/sc5_schema_adapter_v2.py
src/gripper_attack/sc5_event_segmenter_v2.py
scripts/stageb/build_sc5_canonical_corpus_v2.py
configs/v2_sc5_schema_aliases.yaml
configs/v2_sc5_split_policy.yaml
```

Required compact outputs:

```text
tables/v2_sc5_canonical_episode_manifest.csv
tables/v2_sc5_canonical_event_manifest.csv
tables/v2_sc5_duplicate_groups.csv
tables/v2_sc5_exclusion_reasons.csv
tables/v2_sc5_split_manifest.csv
tables/v2_sc5_sequence_index.csv
artifacts/v2_sc5_teacher_config.json
artifacts/v2_sc5_teacher_config.sha256
artifacts/v2_sc5_sequence_manifest.json
artifacts/v2_sc5_data_gate.json
reports/V2_SC5_CANONICAL_CORPUS_AUDIT.md
```

Large tensors remain on the server and are referenced by path/hash only.

---

# 12. Canonical corpus gates

Pass only if all are true.

## G-C0 Inventory reconciliation

Candidate, included, excluded, duplicate, unknown, and failed counts reconcile to the source census.

## G-C1 Clean provenance

Every positive episode/event is explicitly clean and clean-success.

Attack contamination:

```text
0
```

## G-C2 Schema

Every required deployment feature is measured or causally derived with a recorded source.

```text
missing-to-zero = 0
gripper semantic conflicts = 0
```

## G-C3 Teacher

- train-only calibration;
- Teacher config SHA frozen;
- SC5 guard=5;
- K=10;
- valid phase order.

## G-C4 Object/event identity

Every positive has one unique transported object.

Multi-stage episodes are segmented or abstained.

## G-C5 Dedup/split

No exact/content/init-state group crosses splits.

## G-C6 Strict held-out

Butter s8/s9/s11 contamination in train/val/calibration:

```text
0
```

## G-C7 Corpus diversity

Report, do not fabricate:

- usable episodes/events;
- positive;
- no-corridor;
- OOD;
- suites;
- tasks;
- segmented multi-stage events;
- abstained multi-stage episodes;
- exclusions by reason.

Allowed statuses:

```text
SC5_CANONICAL_CORPUS_PASS
SC5_CANONICAL_CORPUS_DATA_LIMITED
SC5_CANONICAL_CORPUS_BLOCKED_PROVENANCE
SC5_CANONICAL_CORPUS_BLOCKED_SCHEMA
SC5_CANONICAL_CORPUS_BLOCKED_EVENT_AMBIGUITY
SC5_CANONICAL_CORPUS_BLOCKED_LEAKAGE
```

Do not train while this gate is unresolved.

---

# 13. Expanded-data MLP control

After canonical corpus PASS, retrain the exact same MLP before implementing TCN.

Commit:

```text
fix(eval): retrain MLP on expanded corpus with frozen splits
```

Use:

- same 25D features;
- 64 hidden units;
- same heads;
- same explicit SC5 state machine;
- same split manifest;
- three seeds;
- train-only normalization;
- best validation checkpoint.

Required metrics:

```text
coverage >= 0.80
false early <= 0.10
post-release <= 0.05
median abs anchor error <= 8
valid K10 start containment >= 0.85
no-corridor abstain >= 0.90
strict held-out contamination = 0
```

If MLP passes:

```text
SC5_EXPANDED_MLP_PASS
TCN_NOT_REQUIRED_FOR_PRIMARY_MODEL
```

TCN may then be run only as an architecture ablation.

If MLP still fails containment:

```text
SC5_EXPANDED_MLP_FAILS_CONTAINMENT_GO_FOR_TCN
```

---

# 14. Conditional TCN stage

Only after the expanded MLP control fails.

Commit:

```text
feat(layer2): train causal TCN on expanded SC5 sequence corpus
```

Recommended architecture:

```text
history = 32
input_dim = 25
hidden = 64
kernel = 3
dilations = [1,2,4,8]
dropout = 0.10
causal left padding
residual blocks
```

Outputs at the current timestep:

```text
phase_logits
corridor_logit
release_logit
confidence_logit
```

Forbidden:

- bidirectional recurrence;
- non-causal attention;
- future padding leakage;
- whole-episode normalization;
- normalized_step/absolute step;
- test-driven threshold tuning;
- changing SC5 guard/K;
- changing the split manifest.

Use three seeds.

Compare on exactly the same corpus/splits:

```text
D5 frozen baseline
rule proxy
time-only
expanded MLP
TCN history32
TCN history64 ablation
label shuffle
feature shuffle
```

TCN must pass all Layer-2 gates and improve K10 containment by at least 5 percentage points over the expanded MLP to support a structural-benefit claim.

Allowed statuses:

```text
SC5_TCN_REPLAY_PASS
SC5_TCN_REPLAY_FAIL
```

---

# 15. Required tests

Add/maintain:

```text
tests/stageb/test_sc5_source_inventory.py
tests/stageb/test_sc5_schema_adapter.py
tests/stageb/test_sc5_full_trajectory_dedup.py
tests/stageb/test_sc5_group_split.py
tests/stageb/test_sc5_heldout_exclusion.py
tests/stageb/test_sc5_sequence_causality.py
tests/stageb/test_sc5_padding_mask.py
tests/stageb/test_sc5_tcn_no_future_leak.py
tests/stageb/test_sc5_tcn_shapes.py
tests/stageb/test_sc5_replay_same_split.py
```

Must prove:

- future rows do not alter current outputs;
- held-out episodes do not enter train statistics;
- duplicate groups do not cross splits;
- padding does not affect valid history outputs;
- `action_dy` is distinct from `action_dz`;
- missing fields fail closed;
- raw/env semantics are enforced;
- Teacher labels can be independently recomputed;
- MLP and TCN use the identical split.

---

# 16. After Layer-2 offline PASS

Do **not** jump directly to Student-triggered VIS.

Next stage is:

```text
Student-triggered command-hold
```

Compare:

```text
Teacher-SC5 command hold
Student-triggered command hold
D5-triggered command hold
```

Gate:

- Student emit falls inside the Teacher valid-start corridor;
- full K10 is legal;
- Student-triggered consequence is close to Teacher-SC5;
- no early grasp-close trigger;
- no release-safe/recovery trigger;
- all action-isolation contracts pass.

Only after that:

1. privileged-SC5 VIS timing bridge;
2. Student-triggered VIS;
3. separate arm-selectivity repair as an independent Layer-3 line.

Never modify timing and attack optimizer in the same experiment.

---

# 17. Privileged-SC5 VIS bridge — future plan

This is not the immediate task, but preserve the plan.

Use mature v1 temporal VIS runner and change only trigger source.

Primary states:

```text
s0
s2
```

Conditions:

```text
CLEAN
D5_TRUE_T10
SC5_TRUE_T10
SC5_RAND_T10
SC5_SHUFFLED_T10
```

Frozen Layer-3 settings:

```text
K = 10
epsilon = 6/255
PGD steps = 20
same force-OPEN objective
same temporal initialization
same controls
```

Success requires SC5 timing separation from D5 and controls while arm selectivity is reported separately.

If timing passes but arm selectivity fails:

```text
SC5_VIS_TIMING_EFFECT_PASS
ARM_SELECTIVITY_NOT_ESTABLISHED
```

---

# 18. Hard stops

Stop immediately and report evidence if any occurs:

- census totals cannot be reconciled;
- attack trajectories contaminate clean positives;
- held-out contamination;
- duplicate/init-state groups cross splits;
- missing values are silently filled with zero;
- object identity is ambiguous in a positive event;
- multi-stage trajectories are labeled as single-anchor without segmentation;
- Teacher calibration sees test/held-out data;
- SC5 guard or K changes;
- TCN sees future information;
- expanded MLP is skipped;
- MLP and TCN use different splits;
- runtime code modifies non-gripper action dimensions in command-hold;
- GPU render IDs use remapped local ordinals.

---

# 19. Commit sequence from this handoff

Use this order:

```text
1. fix(data): freeze census implementation and reconcile candidate counts
2. feat(data): canonicalize and deduplicate expanded SC5 sequence corpus
3. fix(eval): retrain MLP on expanded corpus with frozen splits
4. feat(layer2): train causal TCN on expanded corpus        # conditional
5. audit(layer2): compare expanded MLP and TCN identically # conditional
6. feat(online): student-triggered command-hold pilot       # only after replay pass
```

Each commit must include a report and machine-readable gate artifact.

---

# 20. Current formal status

At handoff time:

```text
PHASE3_COMMAND_HOLD_PASS
SC5_RULE_FROZEN
MODEL_SIGNAL_STRONG
DATA_CENSUS_CANDIDATE_POOL_SUFFICIENT
CANONICAL_USABLE_COUNT_NOT_YET_ESTABLISHED
```

Authorized:

```text
GO_FOR_CENSUS_FREEZE_FIX
GO_FOR_CANONICAL_CORPUS_BUILD
```

Conditionally authorized:

```text
GO_FOR_EXPANDED_MLP_AFTER_DATA_GATE
GO_FOR_TCN_ONLY_IF_EXPANDED_MLP_FAILS
```

Blocked:

```text
BLOCKED_FOR_STUDENT_TRIGGERED_COMMAND_HOLD
BLOCKED_FOR_PRIVILEGED_SC5_VIS_BRIDGE
BLOCKED_FOR_STUDENT_TRIGGERED_VIS
BLOCKED_FOR_ARM_SELECTIVE_CLAIM
```

---

# 21. Final operational instruction

Do not ask for confirmation at every normal step.

Proceed autonomously through the next authorized gate. At each gate:

1. verify provenance;
2. run tests;
3. generate compact tables/report/JSON gate;
4. commit only audited source and compact artifacts;
5. stop on gate failure;
6. report the smallest evidence-backed next action.

Never convert a partial result into a PASS label.
