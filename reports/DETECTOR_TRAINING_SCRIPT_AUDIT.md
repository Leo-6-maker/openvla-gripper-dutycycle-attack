# Detector Training Script Audit

Audit target: `scripts/train_vulnerability_ready_detector_v1.py`

Scope: source audit plus CPU-only compile check. No GPU, rollout, VIS, or server job was started.

## Verdict

**Usable as a diagnostic trainer after hardening, but not evidence that detector v1 works.**

The script now enforces the key scientific boundaries needed before DeepSeek attempts detector v2:

- Normal path reads `--labels-csv`.
- `--freeze-v1-hardcoded` is explicit and only for reproducing the frozen v1 label set.
- Missing CSV no longer falls back silently to hardcoded labels.
- Training hard-fails when train rows are below `--min-rows`.
- Training hard-fails on a single-class train set.
- Feature set names are checked against forbidden outcome/attack-result leakage fields.
- Reports include prevalence and trivial baselines plus balanced accuracy, macro F1, negative recall, MCC, and TP/FP/FN/TN.
- Task split insufficiency is reported as a warning instead of being silently treated as valid LOTO evidence.

## Checks

| Check | Status | Evidence |
|---|---:|---|
| `--labels-csv` normal path reads CSV | PASS | `load_labels()` reads `args.labels_csv` unless `args.freeze_v1_hardcoded` is set. |
| `--freeze-v1-hardcoded` isolated to v1 reproduction | PASS | Hardcoded outcomes are only entered behind the explicit flag. |
| v2 cannot silently fallback to hardcoded 19/20 labels | PASS | Missing labels CSV exits with error. |
| Forbidden/outcome feature leakage guarded | PASS | `FORBIDDEN_FEATURE_PATTERNS` plus `_assert_feature_names_safe()` guard feature names. |
| Metrics include required baselines | PASS | `always_positive`, `always_negative`, `prevalence_random`, `A_task_key_only`, `B_phase_bin_only`. |
| Metrics include required diagnostics | PASS | `balanced_accuracy`, `macro_F1`, `negative_recall`, `MCC`, `tp/fp/fn/tn`. |
| Rows below `--min-rows` hard fail | PASS | Exits code 2. |
| Single class hard fail | PASS | Exits code 2. |
| Task split insufficiency warning | PASS | Warnings for invalid LOTO fold class balance and tasks with fewer than 2 train rows. |

## Verification

Commands run:

```bash
python -m py_compile scripts/diagnostics/audit_label_schema.py scripts/diagnostics/generate_window_compression_candidates.py scripts/train_vulnerability_ready_detector_v1.py scripts/diagnostics/finalize_phase_response_labels.py
```

Result: PASS.

Training smoke attempted with local CPU-only outputs:

```bash
python scripts/train_vulnerability_ready_detector_v1.py \
  --labels-csv tables/object_phase_response_labels_v1.csv \
  --min-rows 15 \
  --output-metrics reports_temp/codex_train_audit_smoke/metrics.csv \
  --output-predictions reports_temp/codex_train_audit_smoke/predictions.csv \
  --output-report reports_temp/codex_train_audit_smoke/report.md
```

Default Windows Python failed before script logic because `numpy` is unavailable.

Bundled Codex Python reached model import and failed because `sklearn` is unavailable. This is a local dependency limitation, not evidence of server runtime failure. DeepSeek should run with the official server env after labels v2 pass schema audit.

## Remaining Risks

1. The script name still says v1. For v2 diagnostics this is acceptable only if reports clearly state diagnostic-only status.
2. Feature leakage guard checks selected feature names, not arbitrary downstream ad hoc additions. Keep schema audit mandatory before training.
3. LOTO remains weak if tasks or negative controls are sparse; warnings must be interpreted as scientific limitations.

## Overclaim Boundary

Do not claim:

- Detector v1 is effective.
- vulnerability_ready is learned.
- Object-wide generalization.
- Detector-triggered VIS is validated.

Allowed wording:

- "Detector v1 exposed a prevalence confound."
- "v2 training is diagnostic-only until controls and negative recall improve."
- "CSV-reading path is hardened for labels v2, pending label schema pass."
