# Fast VIS Calibration Candidate Audit

Date: 2026-06-05

Input: `tables/fast_vis_calibration_candidates_v0.csv`

Scope: CSV schema and label-role sanity check only. No GPU, rollout, VIS, watcher, or detector training was started.

## Verdict

Status: PASS

The candidate table is usable as the initial 8-row Fast cascade calibration set.

## Required Columns

All required fields are present:

- `task_key`
- `state_id`
- `parent_window_start`
- `parent_window_end`
- `phase_bin_proxy`
- `full_vis_label`
- `source_batch`
- `reason_selected`

Additional useful field present:

- `label_status`

## Row Counts

- Total rows: 8
- Positive rows: 4
- Negative/control rows: 4
- Duplicate task/state/window rows: 0

## Candidate Rows

| task_key | state_id | parent_window | phase_bin_proxy | full_vis_label | label_status | source_batch |
|---|---:|---|---|---:|---|---|
| cream_cheese | 4 | [28,45] | near_closed | 1 | positive | batch3 |
| milk | 4 | [19,36] | near_closed | 1 | positive | batch3 |
| ketchup | 1 | [21,38] | near_closed | 1 | positive | batch3 |
| butter | 5 | [25,42] | far_closed | 1 | positive | batch3 |
| salad_dressing | 0 | [7,24] | far_closed | 0 | negative | batch3 |
| bbq_sauce | 5 | [27,44] | near_closed | 0 | negative | batch3 |
| ketchup | 5 | [9,26] | far_closed | 0 | negative | batch3 |
| milk | 5 | [25,42] | far_closed | 0 | negative | batch3 |

## Exclusion Check

No rows contain disallowed statuses or reasons matching:

- `polluted`
- `manual_review`
- `infra_failed`

## Claim Boundary

- This table is a calibration candidate set, not a result table.
- The `full_vis_label` column is a reference label for calibration/audit only.
- Fast cascade outputs must still record provenance, denominator status, runtime, GPU pair, label source, and label confidence before any comparison can be trusted.
