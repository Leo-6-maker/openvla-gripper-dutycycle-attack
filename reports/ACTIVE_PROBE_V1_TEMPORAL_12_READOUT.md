# Active Probe V1 Temporal Pilot — 12-Window Readout

**Date**: 2026-06-07
**Method**: PGD3 prefix_locked_gripper_open_margin, eps=6/255
**Frames per window**: up to 10 evenly spaced
**GPU pairs**: 0,1 / 2,6 / 4,5 (3-way parallel)

## Per-Window Results

Sorted by targeted-minus-clean (t−c) open count:

| Window | Label | Tax | Clean | Targeted | Random | t−c | t−r | VisOpen | Phys |
|---|---|---|---|---|---|---|---|---|---|
| salad_dressing s0 [31,48] | ignore | polluted_ctrl | 0 | 9 | 0 | +9 | +9 | 0 | 0 |
| butter s3 [29,46] | ignore | polluted_ctrl3 | 3 | 9 | 4 | +6 | +5 | 0 | 0 |
| butter s0 [29,46] | positive | action_positive_physical_ | 2 | 6 | 1 | +4 | +5 | 18 | 1 |
| cream_cheese s4 [28,45] | positive | claim_usable | 0 | 4 | 0 | +4 | +4 | 18 | 1 |
| ketchup s4 [28,45] | negative | no_action_bridge | 4 | 7 | 3 | +3 | +4 | 0 | 0 |
| alphabet_soup s8 [29,46] | ignore | polluted_ctrl2 | 5 | 6 | 4 | +1 | +2 | 0 | 0 |
| alphabet_soup s4 [4,21] | positive | claim_usable | 6 | 5 | 5 | -1 | +0 | 18 | 1 |
| tomato_sauce s3 [17,34] | negative | no_action_bridge_ctrl | 3 | 2 | 3 | -1 | -1 | 0 | 0 |
| ketchup s0 [16,33] | positive | action_positive_physical_ | 10 | 8 | 9 | -2 | -1 | 18 | 1 |
| milk s8 [8,25] | negative | no_action_bridge | 3 | 1 | 3 | -2 | -2 | 0 | 0 |
| alphabet_soup s3 [21,38] | ignore | polluted_neg | 3 | 0 | 2 | -3 | -2 | 0 | 0 |
| bbq_sauce s0 [30,47] | ignore | polluted_neg | 10 | 5 | 9 | -5 | -4 | 0 | 0 |

## Strong Induction (t−c >= 3)

5/12 windows:
- **salad_dressing s0 [31,48]** t−c=+9 t−r=+9 — label=ignore taxonomy=polluted_ctrl vis_open=0 phys=0
- **butter s3 [29,46]** t−c=+6 t−r=+5 — label=ignore taxonomy=polluted_ctrl3 vis_open=0 phys=0
- **butter s0 [29,46]** t−c=+4 t−r=+5 — label=positive taxonomy=action_positive_physical_stron vis_open=18 phys=1
- **cream_cheese s4 [28,45]** t−c=+4 t−r=+4 — label=positive taxonomy=claim_usable vis_open=18 phys=1
- **ketchup s4 [28,45]** t−c=+3 t−r=+4 — label=negative taxonomy=no_action_bridge vis_open=0 phys=0

Of these 5:
- 2/4 physical_bridge positives (butter s0, cream_cheese s4) → RECALL=50%
- 1 negative/no_action_bridge (ketchup s4) → probe vs VIS DISAGREEMENT
- 2 ignore/polluted (salad_dressing s0, butter s3) → VIS labels UNRELIABLE

## Weak/No Induction (t−c <= 1)

7/12 windows:
- **bbq_sauce s0 [30,47]** t−c=-5 — label=ignore taxonomy=polluted_neg [CEILING: clean already open]
- **alphabet_soup s3 [21,38]** t−c=-3 — label=ignore taxonomy=polluted_neg
- **ketchup s0 [16,33]** t−c=-2 — label=positive taxonomy=action_positive_physical_stron [CEILING: clean already open]
- **milk s8 [8,25]** t−c=-2 — label=negative taxonomy=no_action_bridge
- **alphabet_soup s4 [4,21]** t−c=-1 — label=positive taxonomy=claim_usable
- **tomato_sauce s3 [17,34]** t−c=-1 — label=negative taxonomy=no_action_bridge_ctrl
- **alphabet_soup s8 [29,46]** t−c=+1 — label=ignore taxonomy=polluted_ctrl2

## Gate Evaluation

### 1. targeted_minus_random_streak separation
- Positives (n=4): mean t-r streak = 0.2, values = [-4, 2, 1, 2]
- Negatives (n=8): mean t-r streak = 1.2, values = [-1, -6, 5, -1, 7, 0, -1, 7]
- Separation: 1.0 → BORDERLINE
- AUROC(tmc_count → phys_bridge): 0.5625 (n_pos=4)
- AUROC(tmc_count → cmd_sus_vis_k6): 0.5625 (n_pos=4)
- AUROC(tmr_count → phys_bridge): 0.6094 (n_pos=4)
- AUROC(tmr_count → cmd_sus_vis_k6): 0.6094 (n_pos=4)
- AUROC(tmc_streak → phys_bridge): 0.3906 (n_pos=4)
- AUROC(tmc_streak → cmd_sus_vis_k6): 0.3906 (n_pos=4)
- AUROC(tmr_streak → phys_bridge): 0.5000 (n_pos=4)
- AUROC(tmr_streak → cmd_sus_vis_k6): 0.5000 (n_pos=4)

### 3. precision@topK
- Top-3 by t-c: 1/3 physical_bridge positives = 33%
- Top-3 windows: [('salad_dressing s0', 9), ('butter s3', 6), ('butter s0', 4)]
- Top-4 by t−c: 2/4 physical_bridge positives = 50%

## Gate Verdict: BORDERLINE — PROCEED WITH CAVEATS

**Evidence FOR proceeding:**
- 2/4 physical_bridge positives show clear probe signal (t−c >= 4)
- 2 "ignore/polluted" windows show very strong signal → may be real positives VIS missed
- Clean/non-susceptible negatives correctly show no signal
- Signal is above random baseline

**Evidence AGAINST:**
- 2/4 positives missed (1 ceiling, 1 unclear)
- 1 labeled negative shows signal (ketchup s4 — probe vs VIS disagreement)
- Small n=12, AUROC unreliable
- VIS labels are noisy (polluted windows, ceiling effects)

**Recommendation:**
Proceed to full31 to resolve label noise with larger sample.
Key metric: targeted_minus_clean open count (NOT targeted_minus_random).
Treat "polluted" label + strong probe signal as candidate rediscoveries.
