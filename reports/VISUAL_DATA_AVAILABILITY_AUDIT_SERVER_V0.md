# Visual Data Availability Audit V0

CPU-only path audit. No images were read and no embeddings were extracted.

**Total train rows**: 19
**Rows with trigger RGB**: 0
**Rows with past RGB**: 0
**Missing path count**: 19
**Visual readiness verdict**: `NOT_READY_VISUAL_PATHS_MISSING`

## Trace Roots

- `/data/liuyu/outputs/nightly_object_batch3_20260604`
- `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604`
- `/data/liuyu/outputs/nightly_object_batch3b_20260604`

## By Source

| Source | Count |
|---|---:|
| batch1 | 2 |
| batch2b | 6 |
| batch3 | 11 |

## By Task

| Task | Count |
|---|---:|
| alphabet_soup | 2 |
| bbq_sauce | 4 |
| butter | 4 |
| cream_cheese | 1 |
| ketchup | 4 |
| milk | 3 |
| salad_dressing | 1 |

## Boundary

- Missing visual paths do not invalidate labels; they only block visual-feature extraction.
- This audit does not support any visual detector claim.
