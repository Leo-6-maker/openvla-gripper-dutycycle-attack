# Contact-Quality Metrics — Attack Pilot Readiness

**Date**: 2026-05-29
**Status**: PARTIAL — has basic failure phase, missing CQFR/CQSR scoring

## Existing CQ Metrics

### `src/gripper_attack/grasp.py::infer_failure_phase()`
Classifies failure into 5 phases:
- `success_libero` — task succeeded
- `no_grasp` — EEF never close enough or gripper never closed
- `lift_fail` — object lifted < 0.03 or < 0.08
- `transport_fail` — object ended far from target (dxy > 0.05)
- `placement_unstable` — object dropped near target

### Available per-step proxy fields (from `grasp.py`):
- `grasp_bowl_z_delta` — object height change (lift detection)
- `grasp_eef_bowl_dist` — EEF-to-object distance (grasp quality)
- `grasp_close_intent` — gripper closing intention
- `grasp_bowl_plate_dxy` — object-to-target horizontal distance

## Missing for Attack Pilot

### Required
1. **CQFR (Contact-Quality Failure Rate)**
   - Fraction of episodes with contact-quality failure
   - Definition: `not success_libero AND not no_grasp`
   - Implementation needed: standalone scoring function

2. **CQSR (Contact-Quality Success Rate)**
   - Fraction of episodes passing all contact-quality checks
   - Definition: `success_libero OR (not failed by contact criteria)`
   - Implementation needed: standalone scoring function

3. **SR-CQ Mismatch**
   - Official SR minus CQSR
   - Identifies episodes where LIBERO says success but contact quality says failure
   - Implementation needed: comparison across episodes

### Desired
4. **Premature release detection** — gripper opens before object reaches target
5. **Drop after lift** — object height drops sharply mid-transport
6. **Unstable transport** — large object oscillation during carry
7. **Uncontrolled final drop** — object released from height

## Implementation Plan

### Minimal viable (for attack pilot):
```python
def compute_cq_metrics(step_records: list[dict], success: bool) -> dict:
    phase = infer_failure_phase(step_records, success)
    return {
        "cq_failure": phase not in ("success_libero", "no_grasp"),
        "cq_success": phase == "success_libero",
        "failure_phase": phase,
        "max_lift": max(r.get("grasp_bowl_z_delta", 0) for r in step_records),
        "min_eef_dist": min(r.get("grasp_eef_bowl_dist", 1e9) for r in step_records),
        "ever_close": any(r.get("grasp_close_intent") for r in step_records),
    }
```

### Per-episode aggregation:
```python
def aggregate_cq(episode_results: list[dict]) -> dict:
    n = len(episode_results)
    n_cq_fail = sum(1 for r in episode_results if r["cq_failure"])
    n_cq_success = sum(1 for r in episode_results if r["cq_success"])
    n_official_success = sum(1 for r in episode_results if r["success"])
    return {
        "CQFR": n_cq_fail / n,
        "CQSR": n_cq_success / n,
        "official_SR": n_official_success / n,
        "sr_cq_mismatch": (n_official_success - n_cq_success) / n,
    }
```

## Calibration Rule
- CQ thresholds must be calibrated on CLEAN data only (or clean + oracle + random-control, NOT VIS)
- Do not tune CQ thresholds to maximize attack effect size
- Pre-register thresholds before viewing VIS outcomes

## Readiness Verdict
- **Basic failure phase**: READY (grasp.py has `infer_failure_phase()`)
- **CQFR/CQSR aggregation**: NEEDS IMPLEMENTATION (simple, ~20 lines)
- **Premature release**: NEEDS IMPLEMENTATION (moderate effort)
- **Drop detection**: PARTIAL (z_delta available, threshold needed)
- **Manual audit protocol**: NOT DEFINED

**Overall**: PARTIALLY READY. Core infrastructure exists in `grasp.py`. CQFR/CQSR aggregation is a trivial wrapper. The missing pieces (premature release, drop detection) are nice-to-have but not blocking for initial attack pilot given the existing `infer_failure_phase()` covers the main failure modes.
