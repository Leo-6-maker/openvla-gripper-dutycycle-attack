# Existing Evidence Freeze Report 2026-07-02

## Executive Diagnosis

- Server access: PASS.
- Object formal condition rows: `{'CLEAN': 162, 'COMMAND_OPEN_ORACLE_T10': 141, 'EARLY_SHIFT_T10': 141, 'RANDOM_TIME_V3': 162, 'RAND_T10': 162, 'TRUE_T10': 162}`.
- TRUE selected/excluded: 141 selected, 21 excluded.
- EARLY/ORACLE selected-set reconciliation: all four set differences are empty.
- RAND_T10: 162/162 success; `attack_applied` field is false for all rows, while 141 emitted rows have `attack_frames=10`; protocol semantics remain unsealed.
- Object preprocessing: `upstream_tf_jpeg` with JPEG roundtrip recovered from summaries.
- Object full protocol provenance: PARTIAL; key attack hyperparameters remain UNVERIFIED.
- CLEAN2000 census: 2000 rows, suite counts `{'libero_object': 500, 'libero_10': 500, 'libero_spatial': 500, 'libero_goal': 500}`.
- CLEAN2000 timing labels: CONSTANT_DEFAULT_LIKE_TIMING_FIELDS; timing detector training not authorized.
- Backup: preflight only; real backup not executed.

## Gate

`HOLD_AUDIT_INCOMPLETE`
