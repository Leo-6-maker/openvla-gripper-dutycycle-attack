# C2g Detector-v2 R5 C0-C3 Review and R6 Dataset-Audit Codex Handoff

Date: 2026-07-11

Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`

R5 materialization head:

```text
91dc39b70e4f92d91621f0d157710ecf5799043c
```

R6 code branch:

```text
assistant/c2g-r6-dataset-audit-20260711
```

## 1. Accepted R5 evidence

The reported C0-C3 execution is accepted as a successful bounded
materialization smoke:

```text
C0 current-head static gate: PASS
R4 mandatory dual-head lineage: PASS
R5 dry-run preview: PASS
R5 bounded materialization: PASS
collection mutation: none
model-byte verification: PASS
model loads during preview: 0
LIBERO environments: 0
new clean rollouts: 0
attacked rollouts: 0
training epochs: 0
D7 changes: none
```

Exact R5 artifacts:

```text
combined dataset:
  /mnt/sdc/dty_user/openvla_attack_evidence/c2g/
  c2g_r5_materialized_91dc39b_codex_20260711/dataset/
  c2g_clean_window_w16_openvla_siglip_within_task.npz

combined dataset SHA256:
  284be7eaf73e7735c297c2069a3c1e8be1143b948ba1747f497f94610ea78159

base report:
  /mnt/sdc/dty_user/openvla_attack_evidence/c2g/
  c2g_r5_materialized_91dc39b_codex_20260711/dataset/
  c2g_multisuite_materialization_report.json

base report SHA256:
  23718ac167cafa8132291fb203d4951175207290d8b55191c64a9a92c7e3ef2c

bound report:
  /mnt/sdc/dty_user/openvla_attack_evidence/c2g/
  c2g_r5_materialized_91dc39b_codex_20260711/dataset/
  c2g_r5_bound_materialization_report.json

bound report SHA256:
  2c382b06d73d89227272ea4a680c9a46219db74a024b052f03b6c786fa67e4d5
```

The materialized output contains 726 overlapping W16 windows from exactly
four retained clean episodes:

```text
libero_10      285 windows
libero_goal    233 windows
libero_object  151 windows
libero_spatial  57 windows
```

The split counts are:

```text
train 518
val    57
test  151
```

The arithmetic strongly indicates the following episode-level assignment,
which R6 must verify from the NPZ rather than assume:

```text
train = libero_10 + libero_goal = 285 + 233
val   = libero_spatial          = 57
test  = libero_object           = 151
```

## 2. Scientific interpretation

R5 proves:

- the four suite-specific OpenVLA/SigLIP embedding paths execute;
- the combined NPZ schema is writable and readable;
- the R4 provenance and model-byte bindings remain intact;
- the materialized feature payload contains only proprio, clean policy,
  clean visual, and language tensors;
- the output tree is closed and the error ledgers are empty.

R5 does **not** prove trainability.

The independent statistical unit is the clean episode, not each overlapping
window. Therefore:

```text
reported samples = 726
effective episode count = 4
```

Treating 726 highly overlapping windows as 726 independent observations would
substantially overstate support.

The current dataset also has three immediate scientific limitations:

1. each suite contributes only one episode;
2. each suite is confined to one global split, so suite and split are
   confounded;
3. the retained Goal episode has zero positive critical labels.

The Goal zero-positive slice is not automatically a label bug. It may be a
valid fully-known negative episode. It is nevertheless insufficient positive
support for Goal training or evaluation.

## 3. Why the old trainability audit is insufficient

`validate_c2g_clean_window_dataset.py` checks aggregate train/val/test support
and episode leakage. With one episode per suite, aggregate window counts may
look large enough to pass even when:

- a split contains only one episode;
- validation and test each contain only one suite;
- no suite appears in all train/val/test splits;
- one suite has zero positive episodes;
- overlapping labels are counted repeatedly.

R6 adds a bound read-only audit that reconstructs unique episode timelines
from overlapping W16 windows before measuring support.

## 4. R6 code contract

New audit tool:

```text
tools/multisuite_detector/audit_c2g_r5_bound_dataset.py
```

New launcher:

```text
scripts/stageb/run_c2g_r6_dataset_audit.sh
```

New regression tests:

```text
tests/test_c2g_r6_bound_dataset_audit.py
```

The audit verifies:

- exact expected dataset/base-report/bound-report SHA256 values;
- exact R5 materialization head;
- R4 binding bytes, status, and audit head;
- clean-only and no-outcome boundary attestations;
- exact NPZ field closure;
- frozen 25D + 9D clean-policy feature contract;
- no attacked/post-intervention/outcome fields;
- all four per-suite dataset/report hashes;
- empty per-suite error ledgers;
- exact 19-file R5 output closure;
- overlapping-window target/mask consistency for every head;
- episode metadata and split consistency;
- unique episode-level positive, negative, unknown, and 2-of-3 support;
- per-suite and per-split episode/task/suite coverage.

The result separates:

```text
integrity_status
engineering_smoke_status
scientific_trainability_status
training_authorization
```

A scientific trainability HOLD is an expected successful audit outcome and
does not make the audit command fail. Provenance or integrity violations do.

Default scientific governance minima are:

```text
total episodes >= 12
total tasks >= 8
episodes per suite >= 3
tasks per suite >= 2
each suite represented in train/val/test
train episodes >= 4 and train suites >= 4
val episodes >= 2 and val suites >= 2
test episodes >= 2 and test suites >= 2
each suite and split has positive, negative, and triggerable support
```

These values are conservative lower-bound authorization gates, not a claim of
statistical sufficiency.

Even if every scientific minimum passes, the audit records:

```text
HOLD_PENDING_EXPLICIT_TRAINING_AUTHORIZATION
```

It never automatically launches training.

## 5. Codex authorization

Codex is authorized only for:

```text
R6A current-head static validation
R6B command preview
R6C one read-only dataset audit
STOP
```

Codex is not authorized to:

- modify repository code;
- rewrite or regenerate R5 artifacts;
- collect new clean episodes;
- load OpenVLA models;
- create LIBERO environments;
- train or calibrate Detector-v2;
- run clean timing;
- run VIS-PGD or any attacked rollout;
- merge a PR;
- modify D7;
- delete or clean storage.

If any repository compatibility problem is found, stop and report it. Do not
patch server-side code.

## 6. R6A current-head static gate

```bash
git fetch origin --prune
git checkout assistant/c2g-r6-dataset-audit-20260711
git reset --hard origin/assistant/c2g-r6-dataset-audit-20260711

export REMOTE_HEAD="$(git rev-parse origin/assistant/c2g-r6-dataset-audit-20260711)"
export EXECUTED_HEAD="$(git rev-parse HEAD)"
export PYTHONPATH="$(git rev-parse --show-toplevel)/src:$(git rev-parse --show-toplevel)${PYTHONPATH:+:$PYTHONPATH}"

test "$REMOTE_HEAD" = "$EXECUTED_HEAD"
test -z "$(git status --short)"
git diff --check

python -m unittest discover -s tests -p 'test_c2g*.py' -v

python -m py_compile \
  tools/multisuite_detector/audit_c2g_r5_bound_dataset.py \
  tests/test_c2g_r6_bound_dataset_audit.py

bash -n scripts/stageb/run_c2g_r6_dataset_audit.sh
```

Required:

```text
REMOTE_HEAD = EXECUTED_HEAD
tests failed = 0
tests skipped = 0
py_compile = PASS
bash syntax = PASS
worktree clean
```

## 7. R6B exact preview

```bash
export R5_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r5_materialized_91dc39b_codex_20260711/dataset

export R5_DATASET="$R5_ROOT/c2g_clean_window_w16_openvla_siglip_within_task.npz"
export R5_BASE_REPORT="$R5_ROOT/c2g_multisuite_materialization_report.json"
export R5_BOUND_REPORT="$R5_ROOT/c2g_r5_bound_materialization_report.json"

export EXPECTED_DATASET_SHA256=284be7eaf73e7735c297c2069a3c1e8be1143b948ba1747f497f94610ea78159
export EXPECTED_BASE_REPORT_SHA256=23718ac167cafa8132291fb203d4951175207290d8b55191c64a9a92c7e3ef2c
export EXPECTED_BOUND_REPORT_SHA256=2c382b06d73d89227272ea4a680c9a46219db74a024b052f03b6c786fa67e4d5
export EXPECTED_MATERIALIZATION_HEAD=91dc39b70e4f92d91621f0d157710ecf5799043c

export AUDIT_HEAD="$EXECUTED_HEAD"
export R6_AUDIT_ROOT=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r6_dataset_audit_${EXECUTED_HEAD:0:8}_20260711
export R6_AUDIT_REPORT="$R6_AUDIT_ROOT/c2g_r6_bound_dataset_audit.json"

test ! -e "$R6_AUDIT_REPORT"

bash scripts/stageb/run_c2g_r6_dataset_audit.sh preview \
  | tee /tmp/c2g_r6_dataset_audit_preview.json
```

Preview requirements:

```text
status = PASS_C2G_R6_DATASET_AUDIT_PREVIEW
command contains all three expected SHA256 values
command contains materialization head and current audit head
command writes outside the immutable R5 output tree
R5 output tree unchanged
R6 report absent
```

## 8. R6C one read-only audit

```bash
before_tree="$(find "$R5_ROOT" -type f -printf '%P %s\n' | sort | sha256sum | awk '{print $1}')"

bash scripts/stageb/run_c2g_r6_dataset_audit.sh run \
  | tee /tmp/c2g_r6_dataset_audit_stdout.json

after_tree="$(find "$R5_ROOT" -type f -printf '%P %s\n' | sort | sha256sum | awk '{print $1}')"

test "$before_tree" = "$after_tree"
test -f "$R6_AUDIT_REPORT"
sha256sum "$R6_AUDIT_REPORT"

python - "$R6_AUDIT_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
for key in (
    "status",
    "integrity_status",
    "engineering_smoke_status",
    "scientific_trainability_status",
    "training_authorization",
    "sample_count",
    "episode_count",
    "split_counts",
    "output_file_count",
    "scientific_trainability_violation_count",
    "next_stage",
    "boundaries",
):
    print(f"{key}={report.get(key)!r}")
print("per_suite=")
for suite, value in report["episode_support"]["per_suite"].items():
    print(suite, value)
print("per_split=")
for split, value in report["episode_support"]["per_split"].items():
    print(split, value)
PY
```

## 9. Expected interpretation

The exact values must come from R6. Based on the accepted R5 ledger, the likely
outcome is:

```text
status = PASS_C2G_R6_BOUND_DATASET_AUDIT
integrity_status = PASS_C2G_R6_DATASET_INTEGRITY
engineering_smoke_status = PASS or HOLD, determined by actual labels
scientific_trainability_status = HOLD_C2G_R6_SCIENTIFIC_TRAINABILITY
training_authorization = HOLD_INSUFFICIENT_SCIENTIFIC_SUPPORT
sample_count = 726
episode_count = 4
output_file_count = 19
next_stage = STOP_FOR_TRAINABILITY_REVIEW
```

Do not force these values. Any discrepancy must be reported.

The expected scientific HOLD reasons include:

- total episode count below 12;
- total task count below 8;
- one episode and likely one task per suite;
- one split per suite rather than three;
- validation and test each containing only one suite;
- Goal zero positive and zero triggerable-positive support.

## 10. Required Codex result format

```text
R6_HEAD
REMOTE_HEAD
EXECUTED_HEAD
WORKTREE_CLEAN

TESTS_PASSED
TESTS_FAILED
TESTS_SKIPPED
PY_COMPILE
BASH_SYNTAX

R6_PREVIEW_STATUS
R6_AUDIT_STATUS
R6_AUDIT_REPORT
R6_AUDIT_REPORT_SHA256

INTEGRITY_STATUS
ENGINEERING_SMOKE_STATUS
SCIENTIFIC_TRAINABILITY_STATUS
TRAINING_AUTHORIZATION

SAMPLE_COUNT
EPISODE_COUNT
SPLIT_COUNTS
OUTPUT_FILE_COUNT

PER_SUITE_SUPPORT
PER_SPLIT_SUPPORT
SCIENTIFIC_VIOLATION_COUNT
SCIENTIFIC_VIOLATIONS

R5_OUTPUT_TREE_UNCHANGED
MODEL_LOADS
LIBERO_ENVIRONMENTS
CLEAN_ROLLOUTS
ATTACKED_ROLLOUTS
TRAINING_EPOCHS
CALIBRATION_RUNS

P0_FINDINGS
P1_FINDINGS
P2_WARNINGS
GO_HOLD_NEXT_STAGE
```

Expected stage boundary:

```text
GO_HOLD_NEXT_STAGE = HOLD_FOR_R6_AUDIT_REVIEW
```

No subsequent stage is authorized by this handoff.
