# VIS Auto-Window / ProprioNoStep Bridge Smoke

## Status

Pending trace aggregation. 3-window parallel VIS running on GPU 2-7.

## Purpose

This is a pipeline smoke:
clean rollout -> frozen ProprioNoStep trigger T -> shifted windows -> VIS prefix_margin / random / clean comparison.

It is not a final ProprioNoStep-guided VIS claim.

## Detector

- Detector: frozen ProprioNoStep (milestone 2c CausalTCN)
- Trigger step T: 93 (known ketchup auto-window default)
- Feature schema: 13-D proprioceptive
- Uses attacked outcomes: No

## Windows

| Window | Range | Clean natural OPEN | Natural-release confounded | Prefix OPEN | Random OPEN | Prefix qposΔ | Random qposΔ | Prefix done | Random done | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| W-20 | 73-90 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| W-10 | 83-100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| W0 | 93-110 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Interpretation rules

- If clean_natural_open_ratio > 0.5: mark natural_release_confounded
- If W0 succeeds and is not confounded: candidate detector timing anchor
- If W0 is confounded but W-10/W-20 succeeds: shifted detector-anchor strategy
- If all fail: frozen ProprioNoStep is not aligned with this VIS vulnerability

## Forbidden claims

- No ProprioNoStep-guided VIS established
- No online detector-triggered VIS solved
- No broad LIBERO-wide claim
