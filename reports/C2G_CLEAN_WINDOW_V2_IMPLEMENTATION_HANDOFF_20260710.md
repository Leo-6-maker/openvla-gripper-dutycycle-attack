# C2g Clean-Window Detector v2 — Implementation and Codex Server Handoff

Date: 2026-07-10

## Repository identity

```text
repository = Leo-6-maker/openvla-gripper-dutycycle-attack
branch = assistant/c2g-clean-window-v2-20260710
base_branch = assistant/c2g-p0-static-patch-20260710
base_sha = e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85
head_at_report_creation = c4508f538baa8b8195c1b2f9f88fe9642c8d03da
```

Always bind execution to the exact result of `git rev-parse HEAD`; do not infer the head from this report if the branch moves during review.

## Corrected objective

The primary detector is now a **clean-rollout gripper-critical window detector**. It is not trained to predict attacked-rollout failure, command-open causal harm, or counterfactual replay outcome.

```text
clean observation / clean causal history
  -> clean OpenVLA decode and clean policy-intent signals
  -> detector chooses a gripper-critical start time
  -> fixed preregistered B-frame gripper-targeted VIS-PGD burst
  -> adversarial re-decode and execution
```

Primary clean Teacher-v2 target:

```text
y_gripper_critical_window =
    target_relevant
    AND gripper_dependency
    AND clean_close_intent
    AND lift_transport_or_constraint
    AND NOT release_safe
```

The teacher may use clean privileged simulator state to define labels. The student may only use causal, deployment-visible clean inputs. Attacked observations, attacked actions, post-intervention state, attack outcomes, manual failure labels, task identity shortcuts, and future clean steps are forbidden primary inputs.

## Implemented files

### Clean Teacher schema and leakage guards

`src/gripper_attack/c2g_clean_window_schema.py`

- freezes the clean-only label vocabulary;
- requires explicit known/unknown semantics;
- requires null labels for unknown rows rather than implicit negatives;
- enforces the clean conjunction defining the critical window;
- enforces release-safe veto and fixed-burst feasibility semantics;
- rejects attacked-rollout and post-intervention fields;
- rejects task index/hash, suite ID, normalized step, teacher labels, and outcome fields as primary student inputs.

### Clean OpenVLA policy-intent signals

`src/gripper_attack/c2g_clean_policy_signals.py`

- consumes only clean gripper-row logits before adversarial re-decode;
- requires externally audited OPEN and CLOSE token sets;
- never guesses token polarity;
- exports open/close probability mass, log-mass margin, normalized entropy, top-1 semantics, and normalized OPEN/CLOSE ranks.

### Clean causal detector and scheduler

`src/gripper_attack/c2g_gripper_critical_window_detector.py`

- reuses the mature causal GRU, masked weighted BCE, contiguous persistence loss, and language-query patch pooling;
- supports the planned ablation ladder:
  - temporal/proprio only;
  - + clean policy intent;
  - + global SigLIP visual features;
  - + language-conditioned patch attention;
- exposes clean heads for critical window, contact/grasp, close intent, transport/constraint, release safety, grounding confidence, window start, and window activity;
- rejects post-attack/counterfactual targets from training;
- provides a stateful 2-of-3 scheduler that uses release/grounding vetoes before trigger and then emits exactly B attacked frames independent of later scores.

### Clean Teacher-v2 label builder

`tools/multisuite_detector/c2g_clean_window_label_builder.py`

- reuses the existing structured BDDL target resolver;
- reuses the existing MuJoCo contact-name canonicalizer and role-aware contact identity;
- supports pick-place, multi-object, articulated, and constrained manipulation routes;
- treats unsupported mechanisms as abstain/unknown;
- converts distractor contact into a known non-target negative when identity is resolved;
- fails closed on unresolved target, contact, gripper command polarity, progress, or release semantics;
- explicitly rejects absolute EEF-z as sufficient progress/lift evidence;
- derives fixed-B burst-feasible steps and one clean attack-start target.

### Dataset adapter and split gates

`tools/multisuite_detector/c2g_clean_dataset_adapter.py`

- maps the clean teacher fields to model targets and masks;
- preserves unknown rows;
- derives a fully-known-negative episode flag only when every row is known and no critical positive occurs;
- reuses the mature episode-level split, persistence, and fold-viability audit with the corrected clean-window target;
- enforces exact, leak-free student feature payloads.

### Matched-load 2x2 experiment contract

`src/gripper_attack/c2g_matched_load_manifest.py`

Freezes the five-condition primary matrix:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

For every clean parent it requires:

- exact condition closure and no duplicate jobs;
- identical clean parent, initial-state, detector checkpoint, and detector-config hashes;
- identical burst length, epsilon, step size, PGD steps, projection, cast, preprocessing, input size, initialization policy, and temporal-init policy;
- identical loss-forward, backward, and adversarial-decode counts;
- identical detector timing within the two detector conditions;
- identical random timing within the two random-time conditions;
- one frozen compute-matched random/non-gripper objective family;
- a random-time start distinct from the detector-selected start;
- deterministic per-parent/per-condition objective seeds.

Uniform noise is rejected as the primary same-load random control because it is epsilon-matched but not compute-matched.

### Static tests and CI

`tests/test_c2g_clean_window_v2.py`

Covers:

- schema known/unknown/release-safe/leakage rules;
- audited OPEN/CLOSE token feature extraction;
- causal-prefix invariance;
- proprio/policy/global-visual/patch-attention model paths;
- clean-only loss and outcome-target rejection;
- contiguous 2-of-3 trigger and exact fixed burst;
- target, distractor, release-safe, unresolved polarity, absolute-z rejection, and articulated manipulation labels;
- unknown-safe episode-negative derivation;
- corrected split coverage;
- exact matched-load manifest closure and mismatch rejection.

`.github/workflows/cpu-c2g-static.yml` now compiles the new modules and runs the new tests in addition to the complete pre-existing C2g/Track-A suite.

## Mature assets deliberately reused

```text
structured BDDL target resolver
MuJoCo contact canonicalization / bilateral grasp identity
LanguageQueryPatchPool
25D causal temporal encoder pattern
masked weighted BCE with active-weight normalization
unknown-safe episode losses
contiguous 2-of-3 persistence contract
within-task / LOTO / LOSO split scaffold
wrong-language-cross-task diagnostic
static BDDL/MuJoCo asset inventory
Track A commit/protocol binding and audit infrastructure
existing visual TokenPrefixPGDAttacker and processor-space projection/cast path
```

The existing visual attacker must remain the attack payload. The detector must not be connected to command-space force-open as the primary method.

## Demoted but preserved components

```text
C2gCausalVulnerabilityDetector
counterfactual Teacher-v2 causal labels
counterfactual manifest / replay contract
```

They remain useful for optional post-hoc oracle analysis, restore audits, and diagnostics. They must not define primary clean detector labels, threshold calibration, split selection, or model selection.

Teacher-v1 labels remain audit/baseline evidence only and remain HOLD for Detector-v2 training.

## Local validation completed before server handoff

A dependency-isolated synthetic CPU run of the new test module completed:

```text
new clean-window tests = 30 PASS
GPU episodes = 0
LIBERO rollouts = 0
OpenVLA inference = 0
counterfactual replays = 0
datasets materialized = 0
detectors trained = 0
```

This local result is not a substitute for repository CI or the server-mounted real BDDL/MuJoCo/data inventory.

## Codex server validation prompt

Execute audit-first and CPU-only. Do not start OpenVLA, LIBERO, a GPU process, training, replay, or large materialization.

```text
You are validating the clean-only C2g Detector-v2 implementation in:

repository: Leo-6-maker/openvla-gripper-dutycycle-attack
branch: assistant/c2g-clean-window-v2-20260710
expected base: e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85

Hard boundaries:
- CPU and read-only inventory first.
- Do not run OpenVLA inference.
- Do not run LIBERO rollouts.
- Do not launch any GPU process.
- Do not materialize the full CLEAN2000 dataset.
- Do not train a detector.
- Do not run counterfactual replay.
- Do not modify D7 artifacts or claims.

Step 1 — identity and cleanliness
1. Clone/fetch into a new clean checkout.
2. Checkout assistant/c2g-clean-window-v2-20260710.
3. Record:
   git rev-parse HEAD
   git rev-parse HEAD^
   git status --short
   git diff --check
4. Verify the base is an ancestor:
   git merge-base --is-ancestor e3bec3b82ac104c633d8dacc5fd27f9cf30a7e85 HEAD
5. Record Python, platform, CPU, RAM, and free-space facts.

Step 2 — compile and static tests
Run:

python -m py_compile \
  src/gripper_attack/c2g_clean_policy_signals.py \
  src/gripper_attack/c2g_clean_window_schema.py \
  src/gripper_attack/c2g_gripper_critical_window_detector.py \
  src/gripper_attack/c2g_matched_load_manifest.py \
  tools/multisuite_detector/c2g_clean_window_label_builder.py \
  tools/multisuite_detector/c2g_clean_dataset_adapter.py \
  tests/test_c2g_clean_window_v2.py

python -m unittest -v tests.test_c2g_clean_window_v2

python -m unittest -v \
  tests.test_c2f_track_a_static \
  tests.test_c2g_static \
  tests.test_c2g_p0_patch \
  tests.test_c2g_teacher_v2_target_resolution \
  tests.test_c2g_teacher_v2_contact_identity \
  tests.test_c2g_teacher_v2_schema \
  tests.test_c2g_counterfactual_manifest \
  tests.test_c2g_static_asset_inventory \
  tests.test_c2g_clean_window_v2

bash -n scripts/stageb/run_c2f_track_a_smoke5.sh
bash -n scripts/stageb/run_c2f_table1_candidate_gpu17.sh

Step 3 — read-only live asset inventory
Use the existing audit_c2g_static_assets.py on the server-mounted official LIBERO BDDL and MuJoCo XML roots.
Write outputs only to a new external audit directory. Record all input paths, sizes, SHA256 values, aggregate digest, BDDL operator census, unresolved operators, MuJoCo finger aliases, target entity names, and contact name coverage.
Fail closed on missing files, parser errors, unsupported operators, or unresolved left/right gripper identity.

Step 4 — tiny clean-label dry build
Select exactly two existing clean episodes per suite, with no new rollout:
- one mechanism-eligible episode where possible;
- one unsupported/ambiguous/boundary episode where possible.

Use only already captured clean step records and clean metadata. Produce a temporary external audit output. Do not publish it as a dataset.

For every dry-build row audit:
- uses_attack_outcome == false;
- uses_future_student_input == false;
- known rows have complete explicit labels;
- unknown rows have all label values null;
- target identity and contacted entity consistency;
- distractor contact is not a positive;
- release-safe is never critical-positive;
- absolute EEF-z alone never produces progress/critical positive;
- unsupported mechanisms abstain;
- fixed-B triggerability and attack-start uniqueness;
- no task index/hash, normalized step, teacher field, object/target pose, contact identity, attack result, or future field enters the proposed student feature list.

Step 5 — server report
Commit no generated server artifact into the repository. Return:
- branch/base/head and worktree cleanliness;
- exact commands;
- compile/test counts and failures;
- BDDL/MuJoCo inventory summary and SHA256 manifest;
- per-suite dry-build episode/row/known/unknown/positive/negative/start counts;
- all reason-code counts;
- any fail-closed problem with exact file/episode/step;
- GPU process count attributable to this task = 0;
- LIBERO rollout count = 0;
- OpenVLA inference count = 0;
- training count = 0;
- replay count = 0;
- proposed next GO/HOLD gate.

Do not authorize training or GPU work. Stop after the independent server report.
```

## Acceptance gates for this handoff

```text
GATE_S0_REPOSITORY_STATIC
  all new tests PASS
  complete pre-existing C2g static suite PASS
  no protected output mutation

GATE_S1_LIVE_ASSET_INVENTORY
  all official BDDL operators supported or explicitly held
  left/right finger identities resolved
  target/contact declarations covered
  input manifest and aggregate SHA frozen

GATE_S2_TINY_CLEAN_LABEL_DRY_BUILD
  zero attacked/outcome field use
  zero unknown-to-negative conversion
  zero absolute-z-only positive
  zero release-safe positive
  target/distractor checks pass
  fixed-B triggerability reported

Only after independent review of S0/S1/S2 may a separately authorized CPU-only small CLEAN2000 label materialization be considered.
```

## Frozen boundaries

```text
D7_TABLE1 = STILL_FROZEN
BLACK_BOWL_FIXED_WINDOW_EVIDENCE = HISTORICAL_MECHANISM_EVIDENCE
TEACHER_V1_FOR_TRAINING = HOLD
COUNTERFACTUAL_REPLAY = OPTIONAL_POSTHOC_ONLY
C2G_CLEAN_WINDOW_V2_TRAINING = NOT_AUTHORIZED
C2G_ONLINE_ROLLOUT = NOT_AUTHORIZED
GPU_EPISODES_LAUNCHED_BY_THIS_BRANCH = 0
LIBERO_ROLLOUTS_LAUNCHED_BY_THIS_BRANCH = 0
OPENVLA_INFERENCE_RUNS_BY_THIS_BRANCH = 0
```
