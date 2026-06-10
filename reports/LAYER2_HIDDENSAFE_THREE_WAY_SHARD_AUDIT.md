# Layer-2 HiddenSafe Three-Way Shard Audit

**Git HEAD**: de31cea

## Shard Assignment

| Shard | GPU | Pairs | Jobs | H Pairs | B Pairs |
|-------|-----|-------|------|---------|--------|
| shard10 | 1,0 | 6 | 12 | 3 | 3 |
| shard45 | 4,5 | 5 | 10 | 3 | 2 |
| shard26 | 2,6 | 5 | 10 | 2 | 3 |

## Gates

All audit gates: PASS

All 9 gates passed.

## Launch

```
tmux new -s s7_l2_confirm_shard10
tmux new -s s7_l2_confirm_shard45
tmux new -s s7_l2_confirm_shard26
```
