# Test and CI matrix V1

Status: `CODE_R4_TEST_CI_MATRIX_PASS`

Audit basis: all tracked `.github/workflows/*.yml` at CODE-R4. This document
describes software guardrails; CI PASS is not scientific evidence or execution
authorization.

## Workflow map

| Workflow / job | Trigger and lifecycle | Purpose and source families | Environment | GPU, model, simulator |
| --- | --- | --- | --- | --- |
| `repository-hygiene.yml` / `repository-hygiene-cpu` | Every pull request and push to `main`; `CURRENT_REQUIRED_ALL_PR` | Immutable authority registry; repository maps/ledgers/navigation; Paper V1 read-only claims; deterministic Paper V2 exports; Stage Z static contracts | Ubuntu, Python 3.11, pytest; full Git history; target under 10 minutes | None. No Torch/CUDA/model/simulator dependency and no `env.step`. |
| `cpu-b3-official-v3.yml` / `source-registry` | Every pull request and push to `main`; `CURRENT_REQUIRED_ALL_PR` | Official V3 source registry and B3 formal/training/prediction/attack-manifest contracts | Ubuntu, Python 3.10, editable package, pytest/numpy/pillow, CPU Torch | CPU Torch tests only; no model checkpoint, GPU, simulator, or environment execution. |
| `cpu-detector-v5.yml` / `detector-v5-cpu` | Every pull request and push to `main`; `CURRENT_REQUIRED_ALL_PR` | V5/R3/factorized detector contracts, Stage V M3.5/M4 governance, and Stage X audit-only contracts | Ubuntu, Python 3.10, editable package, pytest/numpy/pillow, CPU Torch | CPU tensors/fixtures/mocks only; no model checkpoint, GPU, simulator, attack run, or real `env.step`. |
| `cpu-stageb.yml` / `stageb-cpu` | Every pull request and push to `main`; `CURRENT_REQUIRED_ALL_PR` | Stage-B clean collection/audits, B3 retention/materialization, source inventory, and generated-evidence exclusion | Ubuntu, Python 3.10, editable package, pytest/numpy/pillow | No GPU/model/simulator execution. Runner files are compiled, not invoked. |
| `cpu-factorized-phase-c.yml` / `factorized-phase-c-cpu` | Path-filtered pull requests and `main` pushes; `CURRENT_REQUIRED_WHEN_TOUCHED` | Factorized calibration/freeze/held-out authorization and scheduler contracts | Ubuntu, Python 3.10, editable package, pytest/jsonschema | No GPU/model/simulator. Static/synthetic analysis only. |
| `cpu-factorized-l3.yml` / `factorized-l3-cpu` | Path-filtered pull requests and `main` pushes; `CURRENT_REQUIRED_WHEN_TOUCHED` | Factorized L3 handoff, identity, calibration, threshold, and analysis contracts | Ubuntu, Python 3.10, editable package, pytest/jsonschema | No GPU/model/simulator. Static/synthetic analysis only. |
| `cpu-pilot-analysis.yml` / `pilot-analysis-cpu` | Path-filtered pull requests and `main` pushes; `HISTORICAL_NONPROMOTIONAL_COMPATIBILITY_WHEN_TOUCHED` | Pilot manifest/execution-record integrity and blind-review analysis | Ubuntu, Python 3.10, pytest | No GPU/model/simulator. Tests operate on synthetic/static records; pilot status is not promoted. |
| `d8-h1-contract-tests.yml` / `contract-tests` | Push only to `deepseek/detector-v3-d8-continuation-20260730` plus manual dispatch; `HISTORICAL_BRANCH_SCOPED_COMPATIBILITY` | D8 source contract, cache/train parity, and formal-module assert prohibition | Ubuntu, Python 3.11, pytest/numpy, CPU Torch | CPU Torch only. The GPU-smoke source is compiled but not executed; no model/simulator. |

No workflow is removed by CODE-R4. The D8 branch-scoped workflow is retained
because it protects historical exact-contract behavior even though it does not
run on PR #136.

## Exact test commands

### Repository hygiene

```text
python scripts/repository/audit_immutable_authority_paths.py
python -m json.tool docs/repository/REPOSITORY_LIFECYCLE_LEDGER_V1.json
python -m json.tool docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json
python -m py_compile scripts/repository/audit_immutable_authority_paths.py scripts/paper/check_paper_v1_claims.py scripts/paper_v2/export_paper_v2_evidence.py
python scripts/paper/check_paper_v1_claims.py
python scripts/paper_v2/export_paper_v2_evidence.py --check
pytest -q tests/repository tests/paper_v2 tests/stage_z/test_stage_z_preparation.py
git show --check --format= HEAD
git diff --check
```

### Official V3 / B3

```text
pytest -q tests/test_official_v3_source_registry.py tests/test_official_v3_sprint0.py tests/test_b3_official_v3_s1.py tests/test_b3_formal.py tests/test_b3_v3_dataset.py tests/test_b3_v3_trainer_and_viability.py tests/test_b3_v3_training_contracts.py tests/test_b3_v3_preparation.py tests/test_b3_v3_prediction_pipeline.py tests/test_b3_v3_viability_decision.py
git diff --check
```

### Detector V5 / Stage V / Stage X

```text
pytest -q tests/test_v5_contracts.py tests/test_v5_dataset.py tests/test_v5_window_geometry.py tests/test_v5_loss_v2.py tests/test_v5_online_evaluator.py tests/test_v5_baselines.py tests/test_v5_physics_decoder.py tests/test_v5_physics_teacher.py tests/test_clean2000_raw_asset_audit.py tests/test_official_v3_fit_sources.py tests/test_factorized_v3_contract.py tests/test_factorized_scheduler.py tests/test_factorized_runtime.py tests/test_factorized_scheduler_bridge.py tests/test_factorized_handoff.py tests/test_factorized_v3_1_contract.py tests/test_r3_dev_protocol.py tests/test_v5_r3_teacher_contract.py tests/test_v5_r3_student_controls.py tests/test_v5_r3_smoke_entrypoint.py tests/test_v5_r3_input_audit.py
pytest -q tests/detector_v5/test_stage_v_dynamic8.py tests/detector_v5/test_stage_v_gpu_resource_contract.py tests/detector_v5/test_stage_v_m3_5_phase_classifier.py tests/detector_v5/test_stage_v_m3_5_physical_taxonomy.py tests/detector_v5/test_stage_v_m3_5_intervention_parent.py tests/detector_v5/test_stage_v_m3_5_selection_v2.py tests/detector_v5/test_stage_v_m3_5_runtime_audit_v1_3.py
pytest -q tests/detector_v5/test_stage_v_m4_matched_parent.py tests/detector_v5/test_stage_v_m4_formal_resource_gate.py
pytest -q tests/stage_x tests/test_stage_x_primary_matrix_runner.py
git diff --check
```

### Stage B

```text
pytest -q tests/test_b3_retention_multievent.py tests/test_b3_retention_materializer.py tests/test_b3_retention_real_artifact_audit.py tests/test_b3_fit_census.py tests/test_b3_s1_distribution.py tests/test_b3_teacher_training_adapter.py tests/stageb/test_cross_suite_postrun_audit_tools.py tests/stageb/test_cross_suite_clean_queue.py tests/stageb/test_cross_suite_clean_collector.py tests/stageb/test_cross_suite_task_inventory.py
```

The workflow also rejects committed files under `generated/`,
`audit_outputs/`, `paper/generated/`, and `artifacts/`.

### Factorized Phase C and L3

```text
pytest -q tests/analysis/student_trigger_calibration
git diff --check
```

Both path-filtered workflows run this suite after compiling their own explicit
Phase C or L3 script list.

### Pilot analysis

```text
pytest -q tests/analysis/pilot_attack
git diff --check
```

### D8 H1 compatibility

```text
pytest -q tests/detector_v5/test_d8_source_contract.py tests/detector_v5/test_d8_train_core_parity.py
```

The workflow separately compiles the H1 module list and rejects Python
`assert` statements in the formal modules.

## Test taxonomy

| Taxonomy | Current tests/workflows | Coverage statement |
| --- | --- | --- |
| Core source contract tests | Official V3/B3, detector V5, Stage-B, factorized Phase C/L3 | Broad PR guardrails plus path-specific deep suites; CPU only. |
| Stage X engineering/audit tests | `tests/stage_x` and `tests/test_stage_x_primary_matrix_runner.py` in `detector-v5-cpu` | Current audit-only contracts; no scientific runner execution. |
| Stage Z static/adapter tests | `tests/stage_z/test_stage_z_preparation.py` in `repository-hygiene-cpu` | Synthetic action/queue/replan/panel/disable guards; Z0R2 remains HOLD. |
| Paper-analysis/export tests | Paper V1 read-only checker plus Paper V2 CSV/JSON/TeX byte-rebuild, manifest-binding, claim-ID, denominator, censoring, and population-separation tests in `repository-hygiene-cpu` | V1 wording and deterministic V2 exports are covered without scientific execution. |
| Legacy compatibility tests | D8 H1, pilot analysis, retained B3/SC5/factorized suites | Preserved because current or historical contracts depend on them; PASS does not promote old evidence. |

## Known boundaries

- Workflows compile some historical runner/producer files to catch syntax
  regressions. Compilation does not authorize or execute those paths.
- CPU Torch is a test dependency in three workflows; it does not imply GPU or
  model inference.
- Path-filtered workflows do not protect unrelated changes. The four all-PR
  jobs provide the current broad guardrail.
- CI cannot replace source root seals, cohort freezes, protected-read counters,
  or PI review.
