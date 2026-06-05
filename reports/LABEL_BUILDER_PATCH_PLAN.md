# Label Builder Patch Plan

Audit target: `scripts/diagnostics/finalize_phase_response_labels.py`

## Verdict

The current label builder is **not v2-ready**.

It supports:

- `--batch1-merged`
- `--batch2b-vis`
- `--batch3-vis`
- `--output-labels`

It does **not** support:

- `--batch3b-vis`
- `--batch3c-vis`

It also still contains strict 9-label assertions from the earlier Batch2b freeze path, which will fail on any real v2 multi-source merge.

## Current Gaps

| Requirement | Status | Notes |
|---|---:|---|
| Multi-source CSV reading | PARTIAL | Batch1, Batch2b, Batch3 only. |
| `source_batch` preservation | PARTIAL | Internal source exists, but output currently omits explicit `source_batch`. |
| `candidate_role` preservation | PARTIAL | Output has field, but Batch3 CSV ingestion does not preserve it. |
| `denominator_type` preservation | PARTIAL | Output field exists, but not consistently populated. |
| `action_bridge_confounded` preservation | PARTIAL | Output field exists, but role-aware path is limited. |
| Role-specific taxonomy support | PARTIAL | Control roles use `role_specific_gates.py`, but only if role metadata reaches classifier. |
| Duplicate conflict hard fail | MISSING | Current dedupe keeps first row and drops later duplicates silently. |
| Train/ignore/manual_review separation | PARTIAL | Positive/negative/ignore exists; manual_review is not robustly separated for controls. |
| v2 assertions | MISSING | Current assertions hard-code 9 total / 4 pos / 4 neg / 1 ignore. |

## Minimal Patch Plan

1. Add CLI inputs:

```bash
--batch3b-vis tables/object_phase_response_batch3b_vis_summary.csv
--batch3c-vis tables/object_phase_response_batch3c_vis_summary.csv
```

2. Replace the fixed source loop with a source registry:

```python
sources = [
    ("batch1", args.batch1_merged),
    ("batch2b", args.batch2b_vis),
    ("batch3", args.batch3_vis),
    ("batch3b", args.batch3b_vis),
    ("batch3c", args.batch3c_vis),
]
```

3. Normalize and preserve these fields for every row:

- `source_batch`
- `task_key`
- `state_id`
- `window_start`
- `window_end`
- `candidate_role`
- `phase_bin_proxy`
- `denominator_type`
- `provenance_status`
- `action_bridge_confounded`
- `taxonomy_label`

4. Replace silent dedupe with conflict-aware duplicate handling:

- Same task/state/window/source with identical normalized label metadata: allow one row.
- Same task/state/window with conflicting label_status or label_vulnerability_ready: hard fail.
- Same task/state/window with different candidate_role: preserve as separate only if an explicit role key is included; otherwise hard fail.

5. Implement role-specific taxonomy gates before train eligibility:

- `stable_post_lock`: late-open control denominator; skip clean-open unsuitable rows; `done=False` requires manual review unless mechanism is clean.
- `far_too_early`: negative/control taxonomy; never automatic positive.
- `pre_lock`: negative/control taxonomy unless VIS/random/qpos evidence satisfies the frozen positive gate.

6. Replace v1 hardcoded assertions with schema and minimum-count assertions:

- Required columns present.
- Allowed label_status values only.
- Train rows only positive/negative.
- Blocked provenance/taxonomy rows cannot enter train.
- Optional minimum counts are warnings or CLI-configured hard gates, not fixed 9-row assumptions.

7. Run the new schema audit after label build:

```bash
python scripts/diagnostics/audit_label_schema.py \
  --labels-csv tables/object_phase_response_labels_v2.csv
```

## Recommended Implementation Boundary

Do not patch the full builder blindly until Batch3b/Batch3c CSV schemas are available. The minimum safe next change is to add source registry plumbing and conflict-hard-fail logic, then test on v1/Batch3 local CSVs before DeepSeek runs the server-side v2 merge.
