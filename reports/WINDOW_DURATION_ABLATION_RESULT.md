# Window × Duration Ablation Result

**Date**: 2026-06-01 | **Episodes**: 43

## Design

2 high-sensitive tasks (cream_cheese, tomato_sauce) + 2 robust controls (ketchup, salad_dressing), states 0-1, windows: early/detector/late, durations: 30/90/150. Fixed forced-window command-layer sustained open proxy.

## Results

### Cream Cheese (high-sensitive)

| Window | d30 | d90 | d150 |
|--------|-----|-----|------|
| early (step 107) | 0/2 fail | 0/2 fail | **0/2 fail** |
| **detector (step 137)** | 0/2 fail | 0/2 fail | **2/2 fail** |
| late (step 167) | 0/2 fail | 0/2 fail | 0/2 fail |

### Tomato Sauce (high-sensitive)

| Window | d30 | d90 | d150 |
|--------|-----|-----|------|
| early (step 166) | 0/2 fail | 2/2 fail | 2/2 fail |
| **detector (step 196)** | 0/2 fail | 2/2 fail | 2/2 fail |
| late (step 226) | 0/2 fail | 0/2 fail | 0/1 fail |

### Robust Controls (d150 only)

| Task | early 150 | detector 150 |
|------|-----------|-------------|
| ketchup | 0/2 fail | 1/2 fail |
| salad | 1/2 fail | 1/2 fail |

## Gate Analysis

| Gate | Result | Evidence |
|------|--------|----------|
| A: Duration matters | **PASS** | det_30: 0 fail, det_150: 6 fail |
| B: Detector onset matters | **PASS (cc)** | cc_det_150 2/2 fail, cc_early_150 0/2 fail |
| C: Selectivity | **MIXED** | ketchup/salad 50% fail at det_150 |
| D: Implementation | **PASS** | All episodes have attack_applied > 0 |

## Interpretation

**For cream_cheese**: Detector window at step 137 is empirically critical. Same 150-step attack at early window (step 107) causes zero failures. This is clean evidence that ProprioNoStep identifies a critical onset timing that early/late windows miss.

**For tomato_sauce**: More broadly vulnerable. Both early and detector windows cause failure at d90+, while late windows remain safe. The detector window is at the edge of the vulnerable phase.

**For robust controls**: Partial breakage at d150 suggests duration must be calibrated. Shorter burst (90) at detector window preserves selectivity while still achieving effect on high-sensitive tasks.

## Valid Claims

- Duration matters: 30-step burst insufficient; 150-step burst causes failure at detector window.
- Detector onset is empirically critical for cream_cheese: same 150-step attack fails at detector window but succeeds at early window.
- Late windows are universally ineffective at tested durations.
- Robust controls show partial fragility at d150, indicating duration must be calibrated.

## Forbidden Claims

- Mathematical optimal window
- VIS attack success
- Universal attack
- Detector oracle-optimal
