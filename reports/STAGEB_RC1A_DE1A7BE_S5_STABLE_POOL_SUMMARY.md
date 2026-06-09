# Stage-B RC1a de1a7be — S5 Stable Parent Pool Summary

**Date**: 2026-06-09
**Commit**: de1a7be
**Pool**: 24 K-repeat stable parents (K5: 8 + K5b: 16)

## Provenance

- All labels: fixed-env K-repeat (env_seed fixed, attack_seed 0..4)
- K=5 per parent (5 VIS + 5 RAND each)
- 240 total GPU jobs (80 K5 + 160 K5b), 240/240 validator PASS, 0 failures
- Runner: 0e3428f (--env_seed/--attack_seed separation, seeded RAND generator)

## Label Distribution

| Label | Count |
|-------|-------|
| stable_cmd_specific | 11 |
| stable_rand_sensitive | 6 |
| stable_negative | 5 |
| unstable_or_unknown | 2 |
| stable_vis_phys | 5 |

## Task Coverage

8 tasks: milk, tomato_sauce, bbq_sauce, salad_dressing, alphabet_soup, cream_cheese, orange_juice, butter

## Key Windows

| Window | pV_cmd | pR_cmd | VIS | Label |
|--------|--------|--------|-----|-------|
| milk [230,240] | 1.0 | 0.0 | [11,11,11,11,11] | stable cmd |
| milk [240,250] | 1.0 | 0.0 | [11,11,11,11,11] | stable cmd |
| tomato [55,65] | 1.0 | 0.0 | [7,7,7,8,7] | stable cmd+phys |
| tomato [90,100] | 1.0 | 0.0 | [7,7,6,6,7] | stable cmd+phys |
| salad [70,80] | 0.0 | **1.0** | VIS=[4,4,4,4,4] RAND=[10,8,10,10,10] | **rand confound** |

## Claim Status

- 72-pair pool: Bronze exploratory (not training labels)
- K5/K5b stable pool: detector label source
- Old 8/8 unstable: seed-coupling protocol bug (resolved)
- Detector route: reopened on stable labels
