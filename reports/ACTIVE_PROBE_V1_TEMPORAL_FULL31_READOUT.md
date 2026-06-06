# Active Probe V1 Temporal — Full31 Readout

**Date**: 2026-06-07
**Windows**: 31 total
**Ceiling excluded**: 25
**Method**: PGD3 prefix_locked_gripper_open_margin, eps=6/255, 10 frames/window

## Label Distribution

- command_susceptible_positive: 14/31 positive
- physical_bridge_positive: 13/31 positive
- negative_clean: 9/31 positive
- ceiling_flag: 6/31

## Metrics — Ceiling-Excluded Subset (n=25)

| Score | Label | AUROC | AUPRC | P@3 | P@5 | P@Top20% | Top20%Hit |
|---|---|---|---|---|---|---|---|
| targeted_minus_clean_open_count | command_susceptible_positive | 0.4263 | 0.4611 | 0.3333 | 0.4 | 0.4 | 0.1667 |
| targeted_minus_clean_open_count | physical_bridge_positive | 0.4545 | 0.45 | 0.3333 | 0.4 | 0.4 | 0.1818 |
| targeted_minus_clean_open_count | negative_clean | 0.4563 | 0.4055 | 0.3333 | 0.4 | 0.4 | 0.2857 |
| targeted_minus_clean_streak | command_susceptible_positive | 0.3814 | 0.4643 | 0.3333 | 0.4 | 0.4 | 0.1667 |
| targeted_minus_clean_streak | physical_bridge_positive | 0.3994 | 0.4362 | 0.3333 | 0.4 | 0.4 | 0.1818 |
| targeted_minus_clean_streak | negative_clean | 0.4683 | 0.2846 | 0.3333 | 0.2 | 0.2 | 0.1429 |
| targeted_open_rate | command_susceptible_positive | 0.3974 | 0.4381 | 0.3333 | 0.2 | 0.2 | 0.0833 |
| targeted_open_rate | physical_bridge_positive | 0.4091 | 0.4128 | 0.3333 | 0.2 | 0.2 | 0.0909 |
| targeted_open_rate | negative_clean | 0.3214 | 0.2376 | 0.0 | 0.2 | 0.2 | 0.1429 |
| targeted_longest_open_streak | command_susceptible_positive | 0.4295 | 0.4403 | 0.3333 | 0.2 | 0.2 | 0.0833 |
| targeted_longest_open_streak | physical_bridge_positive | 0.4318 | 0.4048 | 0.3333 | 0.2 | 0.2 | 0.0909 |
| targeted_longest_open_streak | negative_clean | 0.3294 | 0.2439 | 0.0 | 0.0 | 0.0 | 0.0 |
| targeted_minus_random_open_count | command_susceptible_positive | 0.3974 | 0.4521 | 0.6667 | 0.4 | 0.4 | 0.1667 |
| targeted_minus_random_open_count | physical_bridge_positive | 0.4351 | 0.4399 | 0.6667 | 0.4 | 0.4 | 0.1818 |
| targeted_minus_random_open_count | negative_clean | 0.4246 | 0.4045 | 0.3333 | 0.4 | 0.4 | 0.2857 |
| targeted_minus_random_streak | command_susceptible_positive | 0.4231 | 0.4931 | 0.6667 | 0.6 | 0.6 | 0.25 |
| targeted_minus_random_streak | physical_bridge_positive | 0.4383 | 0.4746 | 0.6667 | 0.6 | 0.6 | 0.2727 |
| targeted_minus_random_streak | negative_clean | 0.4286 | 0.2567 | 0.0 | 0.0 | 0.0 | 0.0 |

## Best Score per Label (no_ceiling)

- **command_susceptible_positive**: best AUROC=0.4295 (score=targeted_longest_open_streak), P@3=0.3333, Top20%Hit=0.0833
- **physical_bridge_positive**: best AUROC=0.4545 (score=targeted_minus_clean_open_count), P@3=0.3333, Top20%Hit=0.1818
- **negative_clean**: best AUROC=0.4683 (score=targeted_minus_clean_streak), P@3=0.3333, Top20%Hit=0.1429

## Gate Verdict

- command_susceptible AUROC: 0.4295 (score=targeted_longest_open_streak)
- **BELOW 0.65**
- physical_bridge AUROC: 0.4545 (score=targeted_minus_clean_open_count)
- P@3: 0.3333 (random baseline: ~0.48)

## Disagreement Queue (11 entries)

- [HIGH_PROBE_NEGATIVE_LABEL] **salad_dressing s5 [28,45]** — t-c=7, clean_rate=0.0, targeted_rate=0.7, label=negative, tax=no_action_bridge
- [HIGH_PROBE_NEGATIVE_LABEL] **butter s3 [29,46]** — t-c=6, clean_rate=0.3, targeted_rate=0.9, label=ignore, tax=polluted
- [HIGH_PROBE_NEGATIVE_LABEL] **alphabet_soup s6 [40,57]** — t-c=6, clean_rate=0.1, targeted_rate=0.7, label=negative, tax=no_action_bridge
- [HIGH_PROBE_NEGATIVE_LABEL] **salad_dressing s0 [31,48]** — t-c=4, clean_rate=0.3, targeted_rate=0.7, label=ignore, tax=polluted
- [POSITIVE_LABEL_LOW_PROBE] **cream_cheese s4 [28,45]** — t-c=0, clean_rate=0.3, targeted_rate=0.3, label=positive, tax=claim_usable
- [POSITIVE_LABEL_LOW_PROBE] **milk s5 [25,42]** — t-c=0, clean_rate=0.2, targeted_rate=0.2, label=negative, tax=action_positive_physical_
- [POSITIVE_LABEL_LOW_PROBE] **alphabet_soup s4 [4,21]** — t-c=-1, clean_rate=0.6, targeted_rate=0.5, label=positive, tax=claim_usable
- [POSITIVE_LABEL_LOW_PROBE] **ketchup s1 [21,38]** — t-c=-1, clean_rate=0.5, targeted_rate=0.4, label=positive, tax=claim_usable
- [CEILING_POSITIVE] **ketchup s0 [16,33]** — t-c=-2, clean_rate=1.0, targeted_rate=0.8, label=positive, tax=action_positive_physical_
- [POSITIVE_LABEL_LOW_PROBE] **milk s4 [19,36]** — t-c=-2, clean_rate=0.4, targeted_rate=0.2, label=positive, tax=claim_usable
- [CEILING_POSITIVE] **ketchup s5 [9,26]** — t-c=-3, clean_rate=0.9, targeted_rate=0.6, label=negative, tax=action_positive_physical_
