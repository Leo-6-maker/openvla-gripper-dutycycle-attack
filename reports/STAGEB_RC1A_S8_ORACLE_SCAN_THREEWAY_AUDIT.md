# S8 ORACLE Open Physical Scan — Three-Way Shard Audit

**Git HEAD**: 631e4d5

## Shard Assignment

| Shard | GPU | Pairs | Jobs | Tasks | L values |
|-------|-----|-------|------|-------|----------|
| shard10 | 1,0 | 6 | 12 | butter,cream,milk,tomato | [10, 20] |
| shard45 | 4,5 | 5 | 10 | butter,cream,milk,tomato | [20, 30] |
| shard26 | 2,6 | 5 | 10 | butter,cream,milk,tomato | [30, 40] |

## Audit Gates

- [PASS] Total jobs = 32
- [PASS] Physical pairs = 16
- [PASS] Each pair has clean + oracle_open
- [PASS] No cross-shard pair duplication
- [PASS] No missing manifest rows
- [PASS] No extra pairs
- [PASS] shard10 = 12 jobs
- [PASS] shard45 = 10 jobs
- [PASS] shard26 = 10 jobs
- [PASS] No GPU 3,7
- [PASS] Separate output dirs

**Audit: ALL GATES PASS**
