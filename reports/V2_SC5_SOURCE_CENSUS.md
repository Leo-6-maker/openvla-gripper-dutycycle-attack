# V2 SC5 Source Census

Status: `SC5_SOURCE_CENSUS_FROZEN_WITH_CURRENT_SCAN_DRIFT`

This is a source census only. It does not build a canonical corpus, train an MLP/TCN, run GPU jobs, or launch LIBERO.

## Counts

- clean_fail: 571
- directories_scanned: 11484
- known_clean_success: 1679
- manifests: 3002
- step_records: 3013
- unknown_task_names: 0

## Tier Counts

- CONDITIONAL_MULTI_STAGE_CANDIDATE: 110
- CONDITIONAL_PLACE_CANDIDATE: 129
- EXCLUDED_AUDIT_ONLY: 2275
- OOD_ABSTAIN: 50
- PRIMARY_SC5_POSITIVE_CANDIDATE: 449

Interpretation:

- `PRIMARY_SC5_POSITIVE_CANDIDATE` is restricted to clean-success `libero_object` rows with the required deployment fields already present under the current alias table.
- `CONDITIONAL_PLACE_CANDIDATE` and `CONDITIONAL_MULTI_STAGE_CANDIDATE` are not training positives until object/event identity and event-local SC5 boundaries are validated.
- `SCHEMA_MISSING_FIELDS` appears in 876 excluded rows and may contain recoverable sources after the canonical schema adapter is expanded; those rows are not counted as usable here.
- The previous handoff count of approximately 1,100 compatible candidates remains a pre-canonical estimate, not the current usable corpus count.

## Repo Provenance

- repo_head: `cc356f30a9444b4c8bcbd8296282c0280207c756`
- repo_branch: `exp/l2-sc5-census-freeze-fix-codex-20260618`
- repo_dirty: `CLEAN`
- repo_provenance: `PASS`

## Current Minus Historical

- clean_fail: +96
- directories_scanned: +2605
- initially_unknown_task_names: -1735
- known_clean_success: +881
- manifests: +5
- step_records: +5

## Claim Boundary

Allowed: source availability, provenance, schema and exclusion census.
Forbidden: canonical usable count, expanded MLP pass/fail, TCN need, Student trigger success, VIS bridge success.
