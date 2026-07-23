# Codex Handoff — Phase B2 Teacher Prerequisite Audit and Materialization

Date: 2026-07-23

## Authoritative code refs

- Scheduler/runtime base: PR #99, `codex/factorized-v2-scheduler-inputs-20260722`
- Scientific consumer/validator: PR #98, `deepseek/factorized-v2-l3-analysis-20260722`
- PR #98 handoff HEAD: `9e2c141c96116a19c719da1a9d9e4ef657c59ff7`
- Production producer/integration: PR #100, `codex/factorized-v2-production-inputs-20260722`
- PR #100 audited HEAD: `e9addbe3dcbbe68ee642b8216a8cca200508a194`

Do not merge these Draft PRs merely to execute the server phase. Use exact refs and record all checked-out SHAs.

## Current frozen evidence

### Phase A W32 integrity

- Status: PASS
- W32 top seal: `253d9f3a0b62b2e4dec45830b953b1d22c6c8143619b18f254653ca538814869`
- Checkpoint/prediction sub-seals: 12/12 PASS
- Important boundary: existing W32 predictions are inner-CV validation outputs, not authoritative heldout-L3 predictions.

### Phase B identity allocation

- Identity verdict: `PASS_DETERMINISTIC_ALLOCATION`
- Pairwise T/C/P/H/A intersections: zero for all 12 splits
- Cohort membership: PASS
- Statistical coverage: `HOLD_INSUFFICIENT_STATISTICAL_COVERAGE`
- Phase C authorization: HOLD
- Reason: Teacher labels currently cover only the old FIT pool; sealed C/P/H identities do not yet have authoritative Teacher rows.

Phase B evidence root:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B_IDENTITY_MANIFESTS_E9ADD_20260723
```

Top seal:

```text
81150bab3d1799f631a035deda327f0f74917f0cbc1962f709b2b8479313d28a
```

## Required outcome of this handoff

Produce an immutable, sealed Phase B2 root that answers two questions before any checkpoint inference:

1. Can the exact sealed `C ∪ P ∪ H` identities be labeled from their original clean privileged artifacts using the frozen Teacher implementation and contract?
2. After materialization, do calibration, policy-selection, and heldout Teacher coverage/closure pass the authoritative validator?

Attack-eval identities `A` are not part of Phase B2 Teacher materialization. Do not label or inspect attack outcomes.

## Non-negotiable scientific constraints

1. Use the exact identities from the sealed C, P, and H manifests. Do not select by state-id range and do not substitute identities.
2. Do not use Student predictions, attack outcomes, manual failure labels, simulator SR, or future attack artifacts to generate Teacher rows.
3. Do not infer missing Teacher labels from 25D/9D Student inputs. Missing privileged source data remains missing.
4. Use the same frozen Teacher code, Teacher contract, K10 schema, known/unknown semantics, and official horizon used by the accepted corpus.
5. Every output row must bind to exactly one canonical source artifact SHA and declare the exact source episode step count.
6. C/P/H bundles must be written to new immutable roots. Never modify W32 or the Phase B identity root.
7. No checkpoint inference on H and no heldout-L3 replay is authorized in this phase.

## Step 1 — Checkout and CPU gate

Record:

```text
repo commit
PR98 commit
PR100 commit
Python environment
CUDA visibility
```

Run at minimum:

```bash
python -m py_compile \
  analysis/student_trigger_calibration/validate_factorized_identity_disjointness.py \
  analysis/student_trigger_calibration/run_factorized_l3_analysis.py \
  analysis/student_trigger_calibration/fit_factorized_calibrators.py

pytest -q \
  tests/analysis/student_trigger_calibration/test_factorized_l3_metrics.py \
  tests/analysis/student_trigger_calibration/test_teacher_source_binding_regressions.py
```

Hard stop on any failure.

## Step 2 — Full state/cohort closure audit

Before label generation, build a machine-readable ledger for all canonical identities in the source universe, including the currently ambiguous state ranges.

Required columns:

```text
canonical_parent_key
suite
task
state_id
seed
cohort
split
role
source_artifact_path
source_artifact_recursive_sha256
source_episode_step_count
allocation_status
reserved_reason
```

Explicitly account for every state/identity previously summarized as:

```text
0–19   old FIT/Teacher-covered pool
20–23  currently unexplained
24–26  C candidate range
27–29  P candidate range
30–34  H candidate range
35–44  A candidate range
45–49  currently unexplained/reserved
```

These ranges are diagnostic only. The authoritative assignment comes from canonical identity manifests, not from the ranges.

Required verdicts:

```text
C ∪ P == DETECTOR_VAL          or explicit pre-registered reserved identities
H == DETECTOR_TEST             or explicit pre-registered reserved identities
A == ATTACK_EVAL               or explicit pre-registered reserved identities
unassigned identities accounted for
no identity replacement
no post-hoc reallocation based on labels or predictions
```

Any unresolved identity is a HOLD, not an automatic retrain.

## Step 3 — Teacher prerequisite audit for exact C/P/H identities

For every target identity, verify the original clean artifact contains all fields required by the frozen Teacher contract. Classify each identity with exactly one status:

```text
MATERIALIZABLE
MISSING_RAW_ROLLOUT
MISSING_PRIVILEGED_STATE
MISSING_TASK_GEOMETRY
MALFORMED_STEPS
SOURCE_SEAL_FAILURE
TEACHER_CONTRACT_INCOMPATIBLE
```

The audit must verify at least:

```text
canonical identity and split binding
complete step sequence starting at 0
original clean rollout seal
object/target or fixture binding required by the Teacher
contact/support/manipulation evidence required by the Teacher
complete official trajectory horizon
frozen Teacher contract SHA
frozen K10 schema
```

If privileged prerequisites are absent, do not synthesize labels. Emit a HOLD receipt listing exact identities and missing fields.

## Step 4 — Materialize C/P/H Teacher bundles

Only for `MATERIALIZABLE` identities, run the frozen Teacher labeler and create three independent roots:

```text
CALIBRATION_TEACHER_BUNDLE_ROOT
POLICY_TEACHER_BUNDLE_ROOT
HELDOUT_TEACHER_BUNDLE_ROOT
```

Required per-split file:

```text
<root>/<split>/factorized_teacher_v1.jsonl
```

Every row must include at minimum:

```text
canonical_parent_key
step
strict_k10_feasible
strict_k10_known_mask
strict_k10_binding_schema
teacher_contract_sha256
source_artifact_recursive_sha256
source_episode_step_count
```

Calibration rows must additionally contain strict boolean target and known-mask fields for:

```text
grasp_established
manipulation_active
release_or_instability
```

Seal each root with complete `SHA256SUMS` and `SHA256SUMS.sha256`. No unlisted files, symlinks, path escapes, duplicate JSON keys, duplicate `(identity, step)` rows, step gaps, or mixed source SHAs are allowed.

## Step 5 — Authoritative Phase B rerun

Run `validate_factorized_identity_disjointness.py` in authoritative mode using:

- sealed Phase B source-discovery and T/C/P/H/A manifests;
- deterministic allocation receipt and actual parent-cohort manifest;
- frozen Teacher contract file;
- exact frozen K10 schema;
- the three new sealed Teacher roots;
- all 12 frozen splits.

The validator must recompute rather than trust manifest summaries.

Required PASS conditions:

```text
identity_disjointness = PASS
calibration_contract_integrity_pass = true
policy_contract_integrity_pass = true
heldout_teacher_closure_pass = true
k10_contract_parity = PASS
calibration_coverage_pass = true
policy_coverage_pass = true
phase_b_data_integrity = PASS
phase_b_scientific_coverage = PASS
phase_b_overall = PASS
cp_inference_authorized = true
heldout_l3_inference_authorized = false
heldout_l3_blocker = PENDING_EXTERNAL_FREEZE
```

A coverage HOLD after correct label materialization is not contamination and does not imply nested retraining.

## Step 6 — Decision matrix

### Outcome A — PASS

Authorize Phase C checkpoint inference on C and P. H may be materialized as data-ready, but heldout-L3 inference/replay remains blocked until calibrators and joint scheduler thresholds are externally frozen.

### Outcome B — identity clean, coverage insufficient

Return:

```text
HOLD_INSUFFICIENT_STATISTICAL_COVERAGE
```

Report per split/head:

```text
known positives
known negatives
negative episodes
K10-positive episodes
eligible episodes
known denominator episodes
heldout identities with valid K10 denominator
```

Do not choose replacement identities using Student predictions, Teacher positivity, or attack results. Any reallocation must be pre-registered and rerun through the full identity gate.

### Outcome C — missing privileged prerequisites

Return a sealed prerequisite-audit HOLD. The allowed remedies are recollecting clean privileged rollouts for the pre-registered identities or defining a new independent cohort before observing predictions/outcomes. Do not infer labels from deployment-safe features.

### Outcome D — contamination

Only proven cross-role identity overlap or duplicate role assignment yields:

```text
NESTED_RETRAIN_REQUIRED
```

## Phase C boundary after Phase B2

Even after Phase B2 PASS:

```text
C inference                 allowed
P inference                 allowed
calibrator fit on C         allowed after sealed predictions
joint threshold search on P allowed after sealed predictions
H prediction materialize    only under the separately frozen Phase C plan
heldout-L3 replay           forbidden until calibration + threshold contract freeze
Full-FIT                    HOLD
passive shadow              HOLD
active canary               HOLD
attack                      HOLD
```

## Required final response from Codex

Return:

```text
checked-out SHAs
CPU test results
Phase B2 output root
SHA256SUMS.sha256
state/cohort closure verdict
prerequisite status counts and identity list for every non-materializable item
C/P/H row and identity counts by split
Teacher contract SHA
K10 schema
Phase B authoritative receipt path and SHA
all gate statuses
exact next authorized action
```

Do not summarize Phase B as complete unless both data integrity and scientific coverage are PASS.
