# S20F Complete Experimental Results — 2026-06-11

**Commit:** `7f8a0e4` | **Branch:** exp/vis-prefix-margin-repair-20260603

## 1. Track A: Known-Parent Lift to Official/V4 Layer3

**Status: CONFIRMED (2/3 seeds)**

### tomato_sauce_s0_w70-80 (confirmed vulnerable parent)

| Seed | Clean | RAND | VIS | VIS-RAND Δ | Classification |
|------|-------|------|-----|------------|----------------|
| 80 | 209✓ | 185✓ (open=3,str=2) | **280✗ timeout** (open=10,str=10) | +7 | TASK_EFFECT_POSITIVE |
| 81 | 209✓ | 229✓ (open=5,str=5) | **280✗ timeout** (open=9,str=9) | +4 | TASK_EFFECT_POSITIVE |
| 82 | 209✓ | 230✓ (open=8,str=7) | 175✓ (open=10,str=10) | +2 | CONTACT_WEAK |

Attribution: GRIPPER_CONTACT_ATTRIBUTED (EEF converges, gripper oscillates).

### ketchup_s0_w150-160

| Seed | Clean | RAND | VIS | Classification |
|------|-------|------|-----|----------------|
| 80 | 159✓ | 154✓ (open=4,str=4) | 170✓ (open=2,str=1) | NO_OFFICIAL_TRANSFER |

## 2. v0.3.1 Labeling Data Accumulation

### Scale
| Metric | Count |
|--------|-------|
| Total labeling jobs run | 102 |
| RAND labels | 96 |
| VIS labels | 6 |
| Paired RAND+VIS labels | **6** |
| Total JSON outputs | 194 |

### RAND Results
| Category | Count |
|----------|-------|
| STRICT pass (open≤3, streak≤3) | 23 |
| USABLE pass (open≤5) | 11 |
| RANDOM_CONFOUNDED (timeout/failure) | 31 |
| RAND open distribution | {0:18, 1:5, 2:8, 3:5, 4:7, 5:11, 6:9, 7:9, 8:10, 9:9, 10:5} |

### VIS Results
Only 6 VIS runs (from GPU10 extra queue). All on STRICT RAND-pass windows (open=0-1). No VIS effects observed (all windows were late-transport/preplace phases with strong CLOSE preference).

## 3. GPU Utilization

| GPU | Jobs Completed | Status |
|-----|---------------|--------|
| 1,0 | 10 (6 VIS + 4 RAND) | Complete |
| 2,6 | 46 RAND | Complete |
| 4,5 | 46 RAND | Complete |

## 4. Claim Matrix

| Claim | Status |
|-------|--------|
| Known vulnerable parent (tomato_sauce_s0_w70-80) transfers to official/V4 Layer3 | **CONFIRMED (2/3)** |
| VIS-specific task failure under matched RAND control | **CONFIRMED (2/3)** |
| Gripper-contact attribution (not model collapse) | **SUPPORTED** |
| v0.3.1 detector-selected window | **NOT CLAIMED** |
| Object-wide / task-wide attack success | **NOT CLAIMED** |
| v0.3.1 training complete | **NOT READY (6 paired labels, need 30+)** |

## 5. Next Steps

1. **VIS-fill overnight**: Generate large VIS queue from 34 RAND-pass windows, run on all 3 GPUs to reach 30+ paired labels
2. **Train v0.3.1**: After reaching 30 paired labels, train phase-aware detector
3. **Detector-selected pipeline**: Run full v0.3.1 → RAND-veto → VIS → L3 on S20d/V4 distribution
