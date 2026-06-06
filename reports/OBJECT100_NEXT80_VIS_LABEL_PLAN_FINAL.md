# Next80 VIS Label Plan — Final

**Jobs**: 60 (VIS PGD20 + matched random Linf pairs)

## Strata

- **teacher_high**: 20 windows
- **adjacent_hard_control**: 20 windows
- **teacher_medium**: 20 windows

## Worker Distribution

- **Worker 0 (worker_26 (GPU 2,6))**: 60 jobs
- **Worker 1 (worker_45 (GPU 4,5))**: 30 jobs
- **Worker 2 (worker_10 (GPU 1,0))**: 30 jobs

## Hard Controls Rationale

Adjacent hard controls: same window length as teacher window, positioned
just before (pre-teacher) or after (post-teacher). Same time band,
same length, non-overlapping. These test whether the localizer
distinguishes opportunity from nearby non-opportunity.
