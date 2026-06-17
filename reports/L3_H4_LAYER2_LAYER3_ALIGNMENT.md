# H4: L2→L3 Timing Alignment

## Attack-Effective Windows

- butter_s11: point-only at step60 (anchor=D5=60)
- tomato_sauce_s23: point-only at step141 (D5=69, early by 72)
- salad_dressing_s11: point-only at step59 (D5=128, late by 69)

## Direct D5 Emit

- butter: hit (exact timing co-located)
- tomato: early by 72 steps
- salad: late by 69 steps

## Global Delay Sweep

- No single nonnegative delay can align all 3 parents
- Tomato needs +72, salad needs -69 (impossible with nonnegative constraint)
- Best compromise: single-parent D5 direct on butter

## Conclusion

- D5 direct emitter works for exact-timing parents (butter)
- Early/late parents need Teacher-P oracle for alignment
- H5 oracle closed-loop uses butter_s11 step60 (D5=anchor=60)
