# C2g Clean-Window Detector v2 — End-to-End Completion Addendum

Date: 2026-07-10

This addendum supersedes the earlier statement that PR #58 contains only a static
skeleton. The branch now contains a repository-side executable path from frozen
clean parent selection through training, online VIS-PGD execution, audit, and paired
analysis. Live server compatibility and scientific performance remain unverified
until Codex runs the staged validation plan.

## Correct scientific contract

```text
clean observation and causal clean history
  -> clean OpenVLA decode
  -> physical criticality + clean policy susceptibility detector
  -> fixed preregistered B-frame visual PGD burst
  -> adversarial OpenVLA re-decode
  -> execution
```

The detector is never trained or calibrated using attacked outcomes,
post-intervention state, counterfactual replay outcomes, or manual attacked-failure
labels.

## Full entry point

```bash
bash scripts/stageb/run_c2g_clean_window_full.sh <phase>
```

Phases:

```text
manifests
collect
audit
materialize
dataset_audit
train
calibrate
folds
clean_timing
bind_parents
build_jobs
run_jobs
audit_jobs
analyze
all
```

`all` is an executable convenience path, not an authorization to launch expensive
jobs. Server work must remain phase-gated.

## Completed repository components

### 1. Deterministic clean parent preregistration

```text
scripts/stageb/build_c2g_clean_manifests.py
```

- selects official LIBERO init-state IDs deterministically;
- creates disjoint clean-training and online-evaluation cohorts;
- freezes parent keys, seeds, counts, and manifest hashes;
- launches no rollout and reads no attacked outcome.

### 2. Clean privileged rollout collector

```text
scripts/stageb/collect_c2g_clean_window_rollouts.py
src/gripper_attack/c2g_bddl_metadata.py
```

The collector records deployment-visible student streams and clean privileged
Teacher-v2 evidence:

```text
student-visible:
  RGB
  25D clean causal proprio/action history
  task language
  9D clean OpenVLA gripper policy-intent

teacher-only:
  structured BDDL target declarations
  MuJoCo contact pairs
  target/destination pose evidence
  object-relative lift
  target-relative progress
  articulated fixture motion
  target support and release-safe evidence
```

It runs only the clean policy and never launches a visual attack.

### 3. Four-suite dataset materialization

```text
tools/multisuite_detector/materialize_c2g_clean_window_dataset.py
tools/multisuite_detector/materialize_c2g_multisuite_dataset.py
```

The multisuite wrapper binds each suite to the exact OpenVLA visual/language
checkpoint used for that suite, creates per-suite hash-bound artifacts, and merges
only shape-compatible arrays. Suite and task identities remain audit metadata and
are not model features.

### 4. Trainability and leakage audit

```text
tools/multisuite_detector/validate_c2g_clean_window_dataset.py
```

Before training it fails closed on:

```text
empty train/val/test support
episode split leakage
zero known positives or negatives
zero triggerable 2-of-3 positive episodes
cardinality or finite-value failures
```

### 5. Detector training and clean-only calibration

```text
tools/multisuite_detector/train_c2g_clean_window_detector.py
tools/multisuite_detector/calibrate_c2g_clean_susceptibility.py
```

The trainer exports a strict runtime checkpoint with validation-selected detector
thresholds. The susceptibility calibration uses only validation-split clean
policy-intent and known clean critical labels to freeze:

```text
require_clean_close
minimum OPEN-minus-CLOSE log mass
minimum clean gripper-token entropy
```

The calibration schema explicitly records `uses_attack_outcomes=false` and the
runtime rejects any checkpoint that violates this condition.

### 6. LOTO and LOSO folds

```text
tools/multisuite_detector/run_c2g_clean_window_folds.py
```

Fold construction modifies only the episode split field. It does not regenerate
labels or features.

### 7. Online Detector-v2 + VIS-PGD runtime

```text
src/gripper_attack/c2g_clean_window_runtime.py
scripts/stageb/run_c2g_clean_window_vis_pgd.py
```

At each online step:

```text
clean RGB
  -> clean OpenVLA action and gripper token logits
  -> 25D + policy-intent + visual/language detector input
  -> release/grounding/susceptibility/persistence gate
  -> fixed B-frame TokenPrefixPGDAttacker
  -> adversarial re-decode
  -> env.step(adversarial action)
```

The runtime derives OPEN/CLOSE token groups from actual model decoding semantics and
does not use a universal hard-coded token ID.

### 8. Clean detector timing and parent binding

```text
scripts/stageb/run_c2g_clean_timing_jobs.py
scripts/stageb/extract_c2g_detector_timing.py
scripts/stageb/prepare_c2g_eval_parents.py
```

The CLEAN pass delivers no attack. Parent preparation binds:

```text
clean metadata + step record hash
official LIBERO init-state dtype/shape/content hash
detector start or explicit no-emit
observed clean trajectory horizon
```

No-emit and burst-infeasible parents remain in the denominator report rather than
being silently removed.

### 9. Matched-load five-condition execution

```text
src/gripper_attack/c2g_matched_load_manifest.py
scripts/stageb/build_c2g_matched_load_jobs.py
scripts/stageb/run_c2g_matched_load_jobs.py
```

Conditions:

```text
CLEAN
DET_GRIPPER_VIS_PGD
DET_RANDOM_VIS_ATTACK
RANDTIME_GRIPPER_VIS_PGD
RANDTIME_RANDOM_VIS_ATTACK
```

The executable primary control is `SHUFFLED_GRIPPER_GRADIENT`. The launcher fails
closed on control objective families not yet implemented by the runtime.

Matched fields include:

```text
parent and official init state
detector checkpoint and config
burst length
processor-space epsilon and step size
PGD iterations
projection / cast / preprocessing
random-start and temporal initialization
route-reported loss-forward/backward/decode counts
paired objective seeds across detector and random timing
```

Before execution, the launcher independently recomputes the clean-parent, init-state,
checkpoint, and config hashes.

### 10. Closed-world runtime audit and statistics

```text
scripts/stageb/audit_c2g_matched_load_run.py
scripts/stageb/analyze_c2g_matched_load_results.py
```

The audit verifies exact job closure, delivery, timing, compute counts, Linf budget,
objective/checkpoint/seed binding, and pre-trigger clean parity. Analysis is permitted
only after the audit returns PASS and reports:

```text
condition success rates
paired success-flip tables
exact two-sided McNemar/binomial p-values
timing effect under gripper targeting
objective specificity under detector timing
2x2 difference-in-differences interaction
per-suite results
detector emit and burst-feasible denominator
```

## Independent server validation sequence

Codex should validate one phase at a time:

```text
V0  exact branch/head/worktree and all CPU CI
V1  official BDDL and MuJoCo asset census
V2  manifest builder dry run; inspect train/eval disjointness
V3  one clean collector episode per suite
V4  tiny Teacher-v2 dry audit
V5  tiny four-suite materialization and dataset trainability audit
V6  one-epoch CPU/small-GPU training and strict checkpoint reload
V7  clean susceptibility calibration and runtime checkpoint binding
V8  one detector-only clean timing parent per suite
V9  one-parent five-condition command dry run
V10 explicitly authorized one-parent online GPU smoke
V11 runtime audit before any expansion
```

Codex must patch genuine live incompatibilities it finds and rerun the affected gate.
It must not infer that green synthetic CI proves official LIBERO/OpenVLA compatibility.

## Current claim boundary

```text
REPOSITORY_SIDE_E2E_IMPLEMENTATION = COMPLETE
STATIC_AND_SYNTHETIC_CONTRACT_TESTS = REQUIRED_GREEN
LIVE_SERVER_COMPATIBILITY = NOT_YET_VERIFIED
FULL_DATA_COLLECTION = NOT_YET_RUN
DETECTOR_TRAINING = NOT_YET_RUN
ONLINE_VIS_PGD_MATRIX = NOT_YET_RUN
SCIENTIFIC_EFFECTIVENESS = NOT_YET_ESTABLISHED
D7_TABLE1 = STILL_FROZEN
```
