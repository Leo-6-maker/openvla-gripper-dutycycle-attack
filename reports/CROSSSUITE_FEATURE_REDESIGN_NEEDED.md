# CrossSuite Feature Redesign Needed

Date: 2026-05-31

## Status

CrossSuite feature work should continue offline, but CrossSuite-v2 training is blocked from the current dataset.

## Why

Relative `eef_z` is promising and substantially reduces Object-to-Spatial/Goal distribution shift. However, the available 2B student dataset does not expose complete EEF xyz/velocity features:

- `eef_x`: missing
- `eef_y`: missing
- `eef_vx`: missing
- `eef_vy`: missing
- `mechanism_eligible`: missing

## Decision

Do not train CrossSuite-ProprioNoStep-v2 yet.

Next step is to build or locate a richer artifact index with complete EEF xyz/velocity fields and mechanism eligibility labels, or explicitly design a separate EEF-z-only smoke baseline with clear claim limits.
