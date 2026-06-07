# End-to-End Detector Join Audit
**Generated**: 2026-06-06 15:19:14

## Summary

| Metric | Value |
|--------|-------|
| Total unique rows | 77 |
| Fully joined (phase+vuln+mech) | 7 |
| Vuln+mech (no phase) | 15 |
| Vuln only | 0 |
| Label only | 9 |
| Orphan | 46 |

## Coverage by Source

| Source | Rows | Keys Matched |
|--------|------|-------------|
| Labels v2 | 31 | 65 |
| Mechanism Taxonomy | 60 | 58 |
| Detector v27 Predictions | 2984 → 22 unique | 22 |
| Phase (covered subset) | 24 | 24 |
| Clean Control Bank | 22 | 22 |
| Adaptive 1R | 39 | 39 |

## Join Status Distribution

| Status | Count |
|--------|-------|
| fully_joined | 7 |
| vuln_mech_only_no_phase | 15 |
| vuln_only | 0 |
| label_only | 9 |
| orphan | 46 |

## Mechanism Type Distribution (in joined table)

| Mechanism Type | Count |
|----------------|-------|
| clean_control_negative | 22 |
| mechanism_unclear | 14 |
| negative_unclassified | 13 |
| physical_bridge_positive | 9 |

## Label Status Distribution

| Status | Count |
|--------|-------|
| negative | 35 |
| ignore | 9 |
| positive | 9 |

## Coverage by Task

| Task | Total | Phase | Vuln | Mech | Label |
|------|-------|-------|------|------|-------|
| alphabet_soup | 10 | 0 | 2 | 8 | 10 |
| bbq_sauce | 7 | 0 | 3 | 5 | 7 |
| butter | 3 | 0 | 2 | 2 | 3 |
| cream_cheese | 9 | 9 | 1 | 9 | 9 |
| ketchup | 24 | 13 | 4 | 13 | 14 |
| milk | 16 | 0 | 5 | 14 | 14 |
| orange_juice | 1 | 0 | 1 | 1 | 1 |
| salad_dressing | 3 | 2 | 2 | 2 | 3 |
| tomato_sauce | 4 | 0 | 2 | 4 | 4 |

## Phase Coverage Detail

ProprioNoStep model covers: ['cream_cheese', 'ketchup', 'salad_dressing']

| Task | Has Phase Coverage |
|------|-------------------|
| alphabet_soup | NO (0 rows with phase) |
| bbq_sauce | NO (0 rows with phase) |
| butter | NO (0 rows with phase) |
| cream_cheese | YES (9 rows with phase) |
| ketchup | YES (13 rows with phase) |
| milk | NO (0 rows with phase) |
| orange_juice | NO (0 rows with phase) |
| salad_dressing | YES (2 rows with phase) |
| tomato_sauce | NO (0 rows with phase) |

## Fully Joined Rows (7)

| task | state | window | mechanism | vuln_pred | phase_available | phase_source |
|------|-------|--------|-----------|-----------|-----------------|-------------|
| cream_cheese | 4 | [28,45] | physical_bridge_positive | 1 | 1 | covered_subset |
| ketchup | 0 | [16,33] | physical_bridge_positive | 1 | 1 | covered_subset |
| ketchup | 1 | [21,38] | physical_bridge_positive | 1 | 1 | covered_subset |
| ketchup | 4 | [28,45] | negative_unclassified | 1 | 1 | covered_subset |
| ketchup | 5 | [9,26] | negative_unclassified | 1 | 1 | covered_subset |
| salad_dressing | 0 | [7,24] | negative_unclassified | 0 | 1 | covered_subset |
| salad_dressing | 5 | [28,45] | negative_unclassified | 0 | 1 | covered_subset |

## Missing Links

### Rows missing phase (ProprioNoStep model covers cream_cheese/ketchup/salad_dressing only)

**Tasks with ProprioNoStep model but missing phase data:**
- ketchup s2 [39,56] — source=gold_v2
- salad_dressing s0 [31,48] — source=gold_v2

### Rows missing vuln prediction:
- alphabet_soup s0 [3,20] — source=gold_v2
- alphabet_soup s2 [11,28] — source=eligible_1r_silver_positive
- alphabet_soup s3 [21,38] — source=gold_v2
- alphabet_soup s4 [13,30] — source=eligible_1r_silver_positive
- alphabet_soup s4 [18,35] — source=eligible_1r_silver_positive
- alphabet_soup s4 [3,20] — source=eligible_1r_silver_positive
- alphabet_soup s6 [19,36] — source=eligible_1r_silver_positive
- alphabet_soup s8 [29,46] — source=gold_v2
- bbq_sauce s0 [15,32] — source=gold_v2
- bbq_sauce s0 [30,47] — source=gold_v2
- bbq_sauce s0 [55,72] — source=clean_control_negative
- bbq_sauce s4 [55,72] — source=clean_control_negative
- butter s3 [29,46] — source=gold_v2
- cream_cheese s1 [49,66] — source=clean_control_negative
- cream_cheese s1 [59,76] — source=clean_control_negative
- cream_cheese s1 [69,86] — source=clean_control_negative
- cream_cheese s1 [74,91] — source=clean_control_negative
- cream_cheese s3 [64,81] — source=clean_control_negative
- cream_cheese s3 [69,86] — source=clean_control_negative
- cream_cheese s8 [44,61] — source=clean_control_negative

### Rows missing mechanism type:
- alphabet_soup s0 [3,20] — source=gold_v2
- alphabet_soup s8 [29,46] — source=gold_v2
- bbq_sauce s0 [15,32] — source=gold_v2
- bbq_sauce s0 [30,47] — source=gold_v2
- butter s3 [29,46] — source=gold_v2
- ketchup s1 [18,35] — source=
- ketchup s1 [23,40] — source=
- ketchup s2 [24,41] — source=
- ketchup s2 [39,56] — source=gold_v2
- ketchup s3 [28,45] — source=
- ketchup s3 [33,50] — source=
- ketchup s3 [8,25] — source=
- ketchup s4 [18,35] — source=
- ketchup s4 [8,25] — source=
- ketchup s5 [15,32] — source=
- ketchup s5 [5,22] — source=
- milk s1 [28,45] — source=
- milk s9 [20,37] — source=
- salad_dressing s0 [31,48] — source=gold_v2

## Recommendations

1. **Phase detector coverage is limited to 3 tasks** (cream_cheese, ketchup, salad_dressing).
   - Only 36 rows belong to these tasks.
   - 24 of those have phase scores available.
   - Phase detector CANNOT serve as a universal pipeline stage.

2. **Vulnerability detector coverage**: 22 unique keys have predictions.
   - Canonical predictions (V3_weighted LR): covers 0 rows.

3. **Mechanism coverage**: 58 rows have mechanism types.
   - 9 physical_bridge_positive
   - 22 clean_control_negative
   - 14 mechanism_unclear

4. **Full pipeline feasibility**:
   - Only 7 rows can run through the complete pipeline (phase+vuln+mech).
   - Phase detector is the bottleneck — only 3 tasks covered.
