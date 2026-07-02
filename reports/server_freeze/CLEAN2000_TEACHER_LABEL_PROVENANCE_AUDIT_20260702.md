# CLEAN2000 Teacher Label Provenance Audit 2026-07-02

Source: authoritative `CLEAN2000_CANONICAL_V1` files on dty-server.

## File-Level Counts

- CLEAN2000 index rows: 2000
- teacher label rows: 2000
- primary rows: 1043
- safety rows: 957
- suite counts: `{'libero_object': 500, 'libero_10': 500, 'libero_spatial': 500, 'libero_goal': 500}`

## Timing Fields

- anchor_zero_count: 1350
- confidence_0_5_count: 1350
- window_0_10_count: 1350
- event_id_nonempty_count: 0
- placeholder_default_suspected_count: 1350

## Finding

`CONSTANT_DEFAULT_LIKE_TIMING_FIELDS`

The teacher label file has per-episode records, but the positive timing fields are constant/default-like for the 1350 positive labels: anchor 0, confidence 0.5, window [0,10], and no non-empty teacher event id recovered by this parser.

`TIMING_DETECTOR_TRAINING_READY = NO`
