# H4: L2→L3 Timing Alignment — Corrected Report

## Direct D5 Emit

- butter_s11: hit (D5=60 = effective point 60)
- tomato_sauce_s23: early by 72 steps (D5=69, effective point 141)
- salad_dressing_s11: late by 69 steps (D5=128, effective point 59)
- Overall: 1/3 direct hits

## Global Delay Sweep (0-20)

- Best delay: 0 (hits=1/3)
- No single nonnegative delay can fix both early and late parents
- Butter (exact timing) is the only parent with D5↔attack alignment

## First-CLOSE (Teacher-P anchor)

- 3/3 hits by construction (anchor = effective point)

## Teacher-P Oracle

- 3/3 hits by construction (oracle upper bound)

## Conclusion

- D5 direct emitter works for exact-timing parent (butter)
- H6 detector-triggered POC should use butter_s11 only
- Early/late parents require Teacher-P oracle for alignment
