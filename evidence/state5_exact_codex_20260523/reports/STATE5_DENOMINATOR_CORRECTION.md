# State5 Denominator Correction

**Generated**: 2026-05-23T17:30Z

## Correction Summary

The original combined interpretation (`state5_exact_codex_combined_interpretation.csv`)
reported `eligible_repeat=true` for all 6 repeats (r0,r1,r2,r13,r18,r5) and
`repeat_pass` = True,True,True,false,true,false → 4/6 pass.

This denominator was INCORRECT. r5 should NOT be counted as eligible.

## Correction: r5 → QUARANTINED

**Reason**: r5 has no independent `clean_detect` baseline.

| Evidence | r5 | r18 (valid) |
|----------|----|-------------|
| clean_detect run exists | NO | YES |
| autowindow source | inherited from r18 | independent clean_trajectory |
| window [93,102] provenance | copied from r18 autowindow | fresh autowindow from clean_detect |
| random control SR | 0.0 (FAIL) | 1.0 (PASS) |
| VIS_current control SR | 1.0 | 1.0 |
| targeted attack SR | 1.0 | 0.0 |

Without a clean_detect baseline, the autowindow for r5 is not independently
validated. The random control failure (SR=0) is a secondary signal of contamination:
random noise should NOT cause task failure if the control protocol is clean.

**Disposition**: Excluded from valid eligible denominator. Preserved in quarantine
for transparency.

## Corrected Accounting

```
Before correction:
  eligible = 6  (r0,r1,r2,r13,r18,r5)
  pass     = 4  (r0,r1,r2,r18)
  pass_rate = 4/6 = 66.7%

After correction:
  eligible = 5  (r0,r1,r2,r13,r18)
  pass     = 4  (r0,r1,r2,r18)
  pass_rate = 4/5 = 80.0%
  quarantined = 1 (r5)
```

## r13: Valid Eligible, Not Pass

r13 IS a valid eligible repeat (independent clean_detect, own autowindow [65,74],
correct protocol parameters applied). The targeted attack did NOT cause failure
(SR=1.0) — the task is phase-robust at window [65,74].

This is a legitimate null result, not a protocol error. See STATE5_WINDOW_PHASE_SENSITIVITY.md.
