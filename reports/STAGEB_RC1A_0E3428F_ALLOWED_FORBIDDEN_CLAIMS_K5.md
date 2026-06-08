# Stage-B RC1a 0e3428f — Allowed & Forbidden Claims After K5

**Date**: 2026-06-09
**Commit**: 0e3428f

## Allowed Claims

1. Old 8/8 instability was primarily caused by seed-coupled replay protocol bug, not intrinsic attack stochasticity
2. Fixed env_seed + varied attack_seed is necessary for valid repeat-stability evaluation
3. K5 fixed-env protocol shows 8/8 parents with stable label structure
4. Corrected VIS produces repeatable command OPEN (milk VIS=[11,11,11,11,11])
5. tomato [90,100] is current strongest stable_cmd + strict vis_phys positive
6. tomato [115,125] is current genuine stable_rand_sensitive confound (pV=0.8, pR=0.8)
7. Online detector route is reopened, but ONLY on K-repeat stable labels
8. 72-pair pool remains valid as Bronze candidate discovery pool
9. Detector training is feasible on fixed-env K-repeat stable labels

## Forbidden Claims

1. Detector solved or near-solved
2. Object-wide detector established
3. 72-pair AUROC is final detector evidence
4. Global visual route is permanently dead (tested on protocol-bugged labels)
5. Old 8/8 unstable proves intrinsic attack randomness
6. Single-shot labels are ground truth
7. Shared phys equals strict physical bridge
8. Detector can be trained on 72-pair single-shot labels
