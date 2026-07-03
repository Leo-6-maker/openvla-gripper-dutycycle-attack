# Codex Repository Audit V1

Status: READY_FOR_REVIEW

Tasks: C0_01, C0_02, C0_03

## Executive Verdict

The repository has a credible Label V2 builder/validator and partial detector
tooling, but it does not yet contain a formal end-to-end implementation path
from frozen Label V2 plus frozen clean features to detector freeze, exact-prefix
branching, attack canary, CQ audit, statistics, and paper tables.

Current audit classification:

| Area | Classification |
|---|---|
| Label V2 formal ledger builder and independent output validator | EXISTS_AND_REVIEWED |
| Label V2 downstream five-file ingestion | EXISTS_AND_REVIEWED_AFTER_C1 |
| 25D SC5 feature schema | EXISTS_AND_REVIEWED |
| Frozen feature artifact ingestion and Label V2 feature join | MISSING_IMPLEMENTATION |
| Parent split tooling | EXISTS_NEEDS_HARDENING |
| State-hash leakage validation | MISSING_IMPLEMENTATION |
| SC5MLPV1 train/eval/FSM tooling | EXISTS_NEEDS_HARDENING |
| Detector event/timing metrics | EXISTS_NEEDS_HARDENING |
| Exact-prefix snapshot/restore | MISSING_IMPLEMENTATION |
| Matched branch queue and same-parent worker assignment | MISSING_IMPLEMENTATION |
| Attack adapter primitives | LEGACY_NOT_FORMAL |
| Runtime telemetry schema and budget helpers | EXISTS_NEEDS_HARDENING |
| Contact-quality automatic evaluator | LEGACY_NOT_FORMAL |
| Blind CQ audit manifest builder | MISSING_IMPLEMENTATION |
| Paired statistics and paper table/figure builders | MISSING_IMPLEMENTATION |
| Authorization boundaries | EXISTS_AND_REVIEWED |

No scientific settings were changed. No server command, real artifact read,
training, inference, rollout, attack, A800 query, or GPU work was run.

## Repository And Branch Identity

```text
repository = D:/vla_attack/repo_work/pr47_closeout_codex
branch = plan/codex-gated-experiment-v1
audited_source_head = 4d1a646100738ca4b5bc86076a080cfd1b895465
audit_commit = 59ba119901a1019e37c69cde7ae68a9fa2f530ad
base producer = af8217c934e5894c87d3db73b031a93f2536624d
```

PR #50 is a planning/audit branch. The server producer checkout and PR #49
authorization-only checkout were not used.

## Frozen Protocol Cross-Reference

The controlling documents reviewed were:

- `CODEX_EXPERIMENT_PLAN_V1.md`
- `CODEX_TASK_MATRIX_V1.csv`
- `CODEX_IMPLEMENTATION_AUTHORIZATION_V1.md`
- `CLEAN2000_LABEL_V2_SPEC.md`
- `POPULATION_DEFINITION_V1.md`
- `SPLIT_AND_LEAKAGE_SPEC.md`
- `DETECTOR_PROTOCOL_V1.md`
- `EXACT_PREFIX_BRANCHING_SPEC_V1.md`
- `ATTACK_PROTOCOL_V1.md`
- `BASELINE_PROTOCOL_V1.md`
- `CONTACT_QUALITY_PROTOCOL_V1.md`
- `METRIC_DEFINITIONS_V1.md`
- `STATISTICAL_ANALYSIS_PLAN_V1.md`
- `EXPERIMENT_MATRIX_V2.csv`

## Component-By-Component Audit

| Component | Classification | Current path / entry | Evidence | Audit result |
|---|---|---|---|---|
| Label V2 ledger build and validation | EXISTS_AND_REVIEWED | `tools/multisuite_detector/build_clean2000_label_v2.py` blob `2d7059e` CLI modes `synthetic-dry-run`, `formal-ledger-build`, `validate-formal-output`, `self-test-closeout` | `tests/test_build_clean2000_label_v2.py` blob `257dafb`; `py_compile` PASS | Covers ledger-only source semantics, exact 2000-row formal closure, five-file output, atomic publish, manual sample, SHA256SUMS, and independent validation. Execution remains NOT_AUTHORIZED. |
| Label V2 downstream ingestion | EXISTS_AND_REVIEWED_AFTER_C1 | `tools/multisuite_detector/load_label_v2_artifact.py` | `tests/test_load_label_v2_artifact.py` | C1 adds a read-only five-file consumer with file-set/SHA, row invariant, formal-count, manual-sample, manifest, summary, and CLI validation. It does not read source ledgers or feature artifacts. |
| 25D SC5 feature order | EXISTS_AND_REVIEWED | `src/gripper_attack/sc5mlp_v1.py` blob `72db0d5`, `SC5_FEATURES`, `SC5MLPV1` | `tests/multisuite_detector/test_feature_contract.py` blob present; direct run blocked because local Python lacks `torch` | Canonical 25D order and 64-64 three-head MLP exist. |
| Streaming feature extraction | EXISTS_NEEDS_HARDENING | `src/gripper_attack/sc5_streaming_features_v2.py` blob `bcb243d`, `SC5StreamingFeatureAdapterV2` | no formal artifact-ingestion test found | Causal feature adapter exists, but no formal frozen feature artifact reader/manifest validator joins it to Label V2. |
| Exact-set episode joins | EXISTS_NEEDS_HARDENING | `tools/multisuite_detector/strict_loader.py`, `strict_join()` | used by `train_detector.py` and `evaluate_detector.py` | Joins split episodes to features/labels, but accepts legacy JSONL/CSV schemas rather than the formal Label V2 five-file artifact plus frozen feature artifact. |
| Parent/state splits | EXISTS_NEEDS_HARDENING | `tools/multisuite_detector/build_detector_splits.py` blob `e8bdde5`, `validate_no_parent_leakage()`; `validate_detector_splits.py` blob `c739782` | `tests/multisuite_detector/test_split_leakage.py` direct run PASS | Parent leakage is checked. Initial-state hash leakage, three named split manifests, and formal Label V2 population contracts are not complete. |
| Detector train CLI | EXISTS_NEEDS_HARDENING | `tools/multisuite_detector/train_detector.py` blob `8503c55`, CLI `--config --feature_csv --label_csv --episode_index --split_file` | `py_compile` PASS; synthetic e2e blocked because local Python lacks `numpy` | SC5MLPV1 training, train-only normalization, clean worktree check, and checkpoint metadata exist. Cohorts are legacy names and inputs are not formal Label V2 manifests. |
| Detector eval CLI and FSM | EXISTS_NEEDS_HARDENING | `tools/multisuite_detector/evaluate_detector.py` blob `b4b5f31`; `score_fsm_legacy_v1.py` blob `fa712a4` | `score_fsm_legacy_v1.py` smoke PASS | Strict checkpoint load and event/timing metrics exist, but formal Gate A2 metrics and threshold-selection contract are incomplete. |
| Detector bundle | EXISTS_NEEDS_HARDENING | `export_detector_bundle.py` blob `79c7957`; `verify_detector_bundle.py` blob `ff0f6e9` | `py_compile` PASS | Bundle file-set/SHA verification exists; needs binding to Gate A2 selected checkpoint, split, normalization, threshold/FSM, and environment manifest. |
| Exact-prefix snapshot/restore | MISSING_IMPLEMENTATION | none found for required fields | `rg` found no formal `simulator_state_hash`, `prefix_action_hash`, `prefix_observation_hash` implementation | Frozen exact-prefix identity, restore, off-by-one validation, and parity checks are not implemented. |
| Matched branch queue | MISSING_IMPLEMENTATION | legacy `scripts/stageb/*queue*`; no formal C7/C8 queue builder | no formal tests found | No formal same-parent branch family queue, worker assignment by `sha256(parent_key)`, or output-root isolation validator. |
| Attack primitives | LEGACY_NOT_FORMAL | `src/gripper_attack/attack_adapter.py` blob `6603e14`; `route_contract.py` blob `98aaf52`; `m3_controls.py` blob `e82baa7`; `execution_target.py` blob `b8be7d2` | `py_compile` PASS | Token-prefix PGD, strict-route checks, target-token validation, shuffled gradient, and random processor deltas exist. They are not wrapped in the formal exact-prefix matched condition protocol. |
| Formal primary attack conditions | MISSING_IMPLEMENTATION | none found | no tests found | `OURS`, `RAND_DIRECTION`, `RANDOM_TIME`, and `Adapted TMA-OPEN` are not implemented as atomic formal branch specs sharing exact prefix, K, epsilon, preprocessing, denominator, and telemetry validators. |
| Mechanism controls | LEGACY_NOT_FORMAL | partial legacy support in attack adapter and Stage-B bridge scripts | no formal tests found | `SHUFFLED_GRADIENT` and untargeted objectives exist as primitives. `EARLY_SHIFT`, `ARM_TARGETED`, and `COMMAND_OPEN_ORACLE` are not formal branch implementations. |
| Runtime telemetry and budget | EXISTS_NEEDS_HARDENING | `src/gripper_attack/logging_schema.py` blob `632bc6a`; `budget.py` blob `3d0e604`; `metrics.py` blob `7a592d1` | `py_compile` PASS; pytest unavailable; direct tests require pytest | Step/run schemas and budget helpers exist, but the formal attack telemetry validator for actual attacked frame count, per-frame Linf, nonzero deltas, prefix hashes, and final status is missing. |
| Contact-quality evaluator | LEGACY_NOT_FORMAL | `scripts/extract_contact_quality_metrics.py` blob `89d0677`, `summarize_run()` | `tests/v4/test_contact_quality_v2.py` exists but pytest unavailable | Existing CQ extractor is older Black Bowl-style logic and does not implement all frozen CQ formulas, task object binding, missing telemetry disposition, or blind audit manifest. |
| Manual CQ audit | MISSING_IMPLEMENTATION | none found | no formal tests found | No blind condition-ID manifest, second-reviewer overlap, kappa, or SR/CQ disagreement sampler exists. |
| Statistical analysis | MISSING_IMPLEMENTATION | `src/gripper_attack/metrics.py` has simple bootstrap helpers; `scripts/v4_aggregate_metrics.py` blob `a6b9b3f` writes placeholder `bootstrap_ci.json` | no formal tests found | No parent-level paired risk difference, exact McNemar, Holm correction, task/parent cluster bootstrap, ITT/emitted-only table builder, or figure data builder. |
| Paper table schemas | EXISTS_NEEDS_HARDENING | `scripts/stageb/build_paper_table_schemas.py` blob `8334368` | `py_compile` PASS | Empty schema writer exists. It is not a result builder and does not bind denominators, artifact manifests, CIs, or figures. |
| Authorization boundaries | EXISTS_AND_REVIEWED | `CODEX_IMPLEMENTATION_AUTHORIZATION_V1.md`, `LABEL_V2_BUILD_EXECUTION_AUTHORIZATION_V1.md`, task matrix | reviewed by repository inspection | Current docs keep server/GPU/formal execution NOT_AUTHORIZED and separate C0 audit from later implementation. |

## Source Path And Test Evidence

Commands run:

```text
git fetch origin +refs/pull/50/head:refs/remotes/origin/pr/50
git switch -C plan/codex-gated-experiment-v1 refs/remotes/origin/pr/50
rg ... docs/tools/scripts/src/tests
python -m py_compile <key Label V2, detector, attack, telemetry files>
python tests/multisuite_detector/test_feature_contract.py
python tests/multisuite_detector/test_split_leakage.py
python tools/multisuite_detector/score_fsm_legacy_v1.py
python tests/multisuite_detector/test_synthetic_e2e.py
python -m pytest tests/test_build_clean2000_label_v2.py tests/multisuite_detector tests/v4/... -q
```

Results:

```text
py_compile = PASS
test_split_leakage.py = PASS
score_fsm_legacy_v1.py = PASS
test_feature_contract.py = BLOCKED_LOCAL_ENV_NO_TORCH
test_synthetic_e2e.py = BLOCKED_LOCAL_ENV_NO_NUMPY
pytest targeted suite = BLOCKED_LOCAL_ENV_NO_PYTEST
```

The local environment is not a project test environment. I did not install
dependencies because this batch is an audit, not environment setup.

## Scientific-Semantic Conflicts

1. `DETECTOR_PROTOCOL_V1.md` still says checkpoint selection is "validation loss
   or validation suite-macro event F1"; implementation supports both. A later
   Gate A2 implementation batch must freeze one primary rule before real runs.
2. `strict_loader.py` uses legacy cohorts `primary_eligible`,
   `safety_abstention`, and `all`, not the frozen `DETECTOR_ELIGIBLE`,
   `DETECTOR_SAFETY`, and `DETECTOR_MULTI_EVENT` populations.
3. `evaluate_detector.py` treats `teacher_window_end` as inclusive in its
   window check, while Label V2 specifies exclusive `window_end`. A formal
   ingestion adapter must normalize this explicitly.
4. Existing CQ code infers condition from run names and uses older proxy
   thresholds. Frozen CQ requires task ontology binding, explicit formulas, and
   `CQ_TELEMETRY_MISSING`.
5. Legacy Stage-B attack bridges use conditions such as `TRUE_T10`, `RAND_T10`,
   and `SHUFFLED_T10`, not the atomic PR #50 condition IDs.

## Authorization-Boundary Audit

PASS. Repository documents and task matrix currently keep:

```text
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
DETECTOR_TRAINING_EXECUTION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

This audit did not invoke server SSH, formal build, formal validation, detector
training, OpenVLA inference, LIBERO, attack rollout, or GPU commands.

## Dependency Graph From Inputs To Paper Tables

```text
Frozen source availability/census/crosstab
  -> build_clean2000_label_v2.py
  -> Label V2 five-file artifact
  -> MISSING: downstream Label V2 five-file ingestion validator
  -> MISSING: frozen clean feature artifact ingestion
  -> MISSING: exact-set Label V2 x feature dataset manifest
  -> build_detector_splits.py / validate_detector_splits.py
  -> NEEDS HARDENING: parent + state-hash split closure
  -> train_detector.py / evaluate_detector.py / score_fsm_legacy_v1.py
  -> NEEDS HARDENING: Gate A2 checkpoint + threshold/FSM freeze
  -> MISSING: exact-prefix snapshot and restore
  -> MISSING: matched branch queue and same-parent worker assignment
  -> LEGACY primitives: attack_adapter.py / route_contract.py / m3_controls.py
  -> MISSING: formal condition implementations and telemetry validator
  -> LEGACY: extract_contact_quality_metrics.py
  -> MISSING: blind CQ audit and kappa
  -> MISSING: paired statistics, ITT/emitted-only table builders, figure data
  -> Paper tables and figures
```

## P0 / P1 / P2 Implementation Gaps

P0:

- Implement Label V2 five-file ingestion and downstream schema validator.
- Implement Label V2 plus frozen clean feature exact-set join.
- Implement state-hash leakage validation and named split manifests.
- Freeze detector checkpoint-selection rule and threshold-selection CLI.
- Implement exact-prefix snapshot/restore schema and parity validator.
- Implement matched branch queue builder before any attack canary.

P1:

- Harden detector metrics for Label V2 exclusive windows, no-emit ITT, and
  DETECTOR_* populations.
- Implement formal attack telemetry validator.
- Implement frozen CQ evaluator and blind audit manifest builder.
- Implement paired statistics with McNemar, Holm, and clustered bootstrap.

P2:

- Replace legacy table schema placeholders with result builders.
- Add deployment overhead and figure data builders.
- Package final detector bundle with full Gate A2 environment lock.

## Recommended Implementation Batches

1. C1/C2 CPU-only ingestion batch: Label V2 reader, frozen feature reader,
   exact-set join, dataset manifest validator, state-hash split checks.
2. C3 detector hardening batch: formal cohort names, threshold-selection freeze,
   Label V2 exclusive-window metrics, synthetic end-to-end detector smoke.
3. C7 engineering batch: exact-prefix snapshot schema, restore validator, matched
   queue builder, telemetry validator.
4. C9/C10 analysis batch: CQ formulas, blind audit manifest, paired statistics,
   table/figure builders.

Do not start C4/C5/C7 server or GPU work before the earlier gates are reviewed.

## Exact Commands/Searches/Tests Run

```text
rg -n "C0_01|C0_02|C0_03|CODEX_INITIAL_AUDIT|CODEX_TASK_MATRIX|paper_driven_multisuite" C:\Users\刘宇\.codex\memories\MEMORY.md
git status --short
git rev-parse --show-toplevel
git rev-parse HEAD
gh pr view 50 --repo Leo-6-maker/openvla-gripper-dutycycle-attack --json ...  # blocked: gh auth missing
git fetch origin +refs/pull/50/head:refs/remotes/origin/pr/50
Invoke-RestMethod https://api.github.com/repos/Leo-6-maker/openvla-gripper-dutycycle-attack/pulls/50
git switch -C plan/codex-gated-experiment-v1 refs/remotes/origin/pr/50
Get-Content docs/paper_driven_multisuite_confirmatory_v1/CODEX_INITIAL_AUDIT_HANDOFF_V1.md
Get-Content docs/paper_driven_multisuite_confirmatory_v1/CODEX_TASK_MATRIX_V1.csv
rg --files docs/paper_driven_multisuite_confirmatory_v1 tools scripts src tests .github
Get-Content protocol docs listed in the handoff
rg -n "SC5_FEATURES|class SC5MLPV1|threshold|checkpoint|FSM|split|parent_key|state_hash|episode_key" tools src tests
rg -n "exact.prefix|snapshot|restore|branch|queue|RANDOM_TIME|RAND_DIRECTION|TMA|COMMAND_OPEN|EARLY_SHIFT|SHUFFLED|UNTARGETED|telemetry|delta_linf|CQ|McNemar|bootstrap|Holm|table|figure" scripts src tools tests docs
git ls-tree -r HEAD -- selected implementation paths
python -m py_compile tools/multisuite_detector/build_clean2000_label_v2.py ... src/gripper_attack/metrics.py
python -m pytest tests/test_build_clean2000_label_v2.py tests/multisuite_detector tests/v4/test_contact_quality_v2.py tests/v4/test_budget.py tests/v4/test_directional.py tests/v4/test_metrics.py tests/v4/test_logging_schema.py -q
python tests/multisuite_detector/test_feature_contract.py
python tests/multisuite_detector/test_split_leakage.py
python tools/multisuite_detector/score_fsm_legacy_v1.py
python tests/multisuite_detector/test_synthetic_e2e.py
```

## Final Gate State

```text
CODEX_INITIAL_REPOSITORY_AUDIT = READY_FOR_REVIEW
CODEX_IMPLEMENTATION_AFTER_AUDIT = NOT_AUTHORIZED
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
A800_GPU_EXPERIMENT_EXECUTION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```
