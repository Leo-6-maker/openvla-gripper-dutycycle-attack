# Object Window Compression Candidate Plan

**Input**: `tables/object_phase_response_batch3_vis_summary.csv`
**Output**: `tables/object_window_compression_candidates.csv`
**Candidates**: 15

## Selection

- Positives: cream_cheese s4 [28,45], milk s4 [19,36], ketchup s1 [21,38]
- Negatives: salad_dressing s0 [7,24], bbq_sauce s5 [27,44]
- Compression lengths: L12, L10, L8 centered inside each parent window.
- CPU-only generation; no rollout, VIS, GPU, or server output mutation.

## Candidate Table

| target_id | role | parent | compressed |
|---|---|---|---|
| cream_cheese_s4_p28_45_L12_w31_42 | positive | [28,45] | [31,42] |
| cream_cheese_s4_p28_45_L10_w32_41 | positive | [28,45] | [32,41] |
| cream_cheese_s4_p28_45_L8_w33_40 | positive | [28,45] | [33,40] |
| milk_s4_p19_36_L12_w22_33 | positive | [19,36] | [22,33] |
| milk_s4_p19_36_L10_w23_32 | positive | [19,36] | [23,32] |
| milk_s4_p19_36_L8_w24_31 | positive | [19,36] | [24,31] |
| ketchup_s1_p21_38_L12_w24_35 | positive | [21,38] | [24,35] |
| ketchup_s1_p21_38_L10_w25_34 | positive | [21,38] | [25,34] |
| ketchup_s1_p21_38_L8_w26_33 | positive | [21,38] | [26,33] |
| salad_dressing_s0_p7_24_L12_w10_21 | negative | [7,24] | [10,21] |
| salad_dressing_s0_p7_24_L10_w11_20 | negative | [7,24] | [11,20] |
| salad_dressing_s0_p7_24_L8_w12_19 | negative | [7,24] | [12,19] |
| bbq_sauce_s5_p27_44_L12_w30_41 | negative | [27,44] | [30,41] |
| bbq_sauce_s5_p27_44_L10_w31_40 | negative | [27,44] | [31,40] |
| bbq_sauce_s5_p27_44_L8_w32_39 | negative | [27,44] | [32,39] |

## Missing Source Rows

- None.

## Use Boundary

These rows are candidate windows only. They do not establish compressed-window effectiveness until DeepSeek runs matched VIS/random under clean denominator controls.
