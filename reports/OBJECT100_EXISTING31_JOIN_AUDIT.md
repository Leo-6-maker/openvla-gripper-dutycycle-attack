# Object100 × Existing31 Join Audit

**Date**: 2026-06-07
**Object100 episodes**: 100
**Object100 teacher windows**: 100
**Existing31 diagnostic windows**: 31

## Join Summary

| Class | Count |
|---|---|
| exact_match_clean_and_label | 31 |

## Eligible for Training

- **2/31** windows eligible for detector training

## Per-Window Detail

| Window | Class | Clean | Shadow | VIS | Teacher | Eligible | Action |
|---|---|---|---|---|---|---|---|
| alphabet_soup_s0_w3_20 | exact_match_clean_and_lab | Y | Y | Y | Y | Y | use directly; extract online features + attach lab |
| alphabet_soup_s3_w21_38 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| alphabet_soup_s4_w4_21 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| alphabet_soup_s6_w40_57 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| alphabet_soup_s8_w29_46 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| bbq_sauce_s0_w30_47 | exact_match_clean_and_lab | Y | Y | N | Y | N | provenance not clean; audit before use |
| bbq_sauce_s0_w15_32 | exact_match_clean_and_lab | Y | Y | N | Y | N | provenance not clean; audit before use |
| bbq_sauce_s0_w5_22 | exact_match_clean_and_lab | Y | Y | Y | Y | N | provenance not clean; audit before use |
| bbq_sauce_s5_w27_44 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| bbq_sauce_s9_w22_39 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| butter_s0_w29_46 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| butter_s3_w29_46 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| butter_s5_w25_42 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| cream_cheese_s4_w28_45 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| ketchup_s0_w16_33 | exact_match_clean_and_lab | Y | Y | Y | Y | Y | use directly; extract online features + attach lab |
| ketchup_s1_w21_38 | exact_match_clean_and_lab | Y | Y | Y | Y | N | provenance not clean; audit before use |
| ketchup_s2_w39_56 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| ketchup_s4_w28_45 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| ketchup_s5_w9_26 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| milk_s1_w8_25 | exact_match_clean_and_lab | Y | Y | Y | Y | N | provenance not clean; audit before use |
| milk_s1_w18_35 | exact_match_clean_and_lab | Y | Y | N | Y | N | provenance not clean; audit before use |
| milk_s4_w19_36 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| milk_s5_w25_42 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| milk_s8_w8_25 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| milk_s9_w5_22 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| orange_juice_s2_w17_34 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| salad_dressing_s0_w31_48 | exact_match_clean_and_lab | Y | Y | N | Y | N | provenance not clean; audit before use |
| salad_dressing_s0_w7_24 | exact_match_clean_and_lab | Y | Y | Y | Y | N | provenance not clean; audit before use |
| salad_dressing_s5_w28_45 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |
| tomato_sauce_s1_w23_40 | exact_match_clean_and_lab | Y | N | Y | Y | N | provenance not clean; audit before use |
| tomato_sauce_s3_w17_34 | exact_match_clean_and_lab | Y | N | N | Y | N | provenance not clean; audit before use |

## Actionable Summary

### exact_match_clean_and_label (n=31)
- **alphabet_soup** s0 [3,20] ignore | teacher: pick_place_transfer [134,143]
- **alphabet_soup** s3 [21,38] ignore | teacher: pick_place_transfer [128,137]
- **alphabet_soup** s4 [4,21] positive | teacher: pick_place_transfer [132,141]
- **alphabet_soup** s6 [40,57] negative | teacher: pick_place_transfer [136,145]
- **alphabet_soup** s8 [29,46] ignore | teacher: pick_place_transfer [131,140]
- **bbq_sauce** s0 [30,47] ignore | teacher: pick_place_transfer [157,166]
- **bbq_sauce** s0 [15,32] ignore | teacher: pick_place_transfer [157,166]
- **bbq_sauce** s0 [5,22] negative | teacher: pick_place_transfer [157,166]
- **bbq_sauce** s5 [27,44] negative | teacher: pick_place_transfer [180,189]
- **bbq_sauce** s9 [22,39] positive | teacher: pick_place_transfer [128,137]
- **butter** s0 [29,46] positive | teacher: pick_place_transfer [130,139]
- **butter** s3 [29,46] ignore | teacher: pick_place_transfer [223,232]
- **butter** s5 [25,42] positive | teacher: pick_place_transfer [150,159]
- **cream_cheese** s4 [28,45] positive | teacher: pick_place_transfer [166,175]
- **ketchup** s0 [16,33] positive | teacher: pick_place_transfer [145,154]
- **ketchup** s1 [21,38] positive | teacher: pick_place_transfer [160,169]
- **ketchup** s2 [39,56] ignore | teacher: pick_place_transfer [123,132]
- **ketchup** s4 [28,45] negative | teacher: pick_place_transfer [116,125]
- **ketchup** s5 [9,26] negative | teacher: pick_place_transfer [102,111]
- **milk** s1 [8,25] positive | teacher: pick_place_transfer [111,120]
- **milk** s1 [18,35] negative | teacher: pick_place_transfer [111,120]
- **milk** s4 [19,36] positive | teacher: pick_place_transfer [108,117]
- **milk** s5 [25,42] negative | teacher: pick_place_transfer [199,208]
- **milk** s8 [8,25] negative | teacher: pick_place_transfer [141,150]
- **milk** s9 [5,22] ignore | teacher: pick_place_transfer [125,134]
- **orange_juice** s2 [17,34] negative | teacher: pick_place_transfer [123,132]
- **salad_dressing** s0 [31,48] ignore | teacher: pick_place_transfer [111,120]
- **salad_dressing** s0 [7,24] negative | teacher: pick_place_transfer [111,120]
- **salad_dressing** s5 [28,45] negative | teacher: pick_place_transfer [149,158]
- **tomato_sauce** s1 [23,40] negative | teacher: pick_place_transfer [157,166]
- **tomato_sauce** s3 [17,34] negative | teacher: pick_place_transfer [121,130]

## Next Steps

1. **exact_match_clean_and_label** (31): Extract online features, train detector v0
2. **clean_exists_label_missing** (0): Schedule VIS PGD20+random attack labeling
3. **label_exists_clean_missing** (0): Recover clean rollout from Object100 step_records
4. **key_mismatch / clean_failed**: Audit manually, do NOT run rollouts
