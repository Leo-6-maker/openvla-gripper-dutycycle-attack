# CPU Phase Detector Smoke

**Status**: offline smoke only. Do NOT claim online detector works.

## Classifier Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| LogisticRegression | 0.8757 | 0.7452 |
| RandomForest | 0.9974 | 0.9965 |
| GradientBoosting | 1.0 | 1.0 |

Best: GradientBoosting (macroF1=1.0)

## Top Features

| Feature | Mean Importance |
|---------|----------------|
| clean_open_count | 0.132183 |
| qpos_end | 0.124855 |
| clean_open_ratio | 0.085329 |
| raw_gripper_mean | 0.058428 |
| eef_z_delta | 0.055376 |
| qpos_max | 0.048941 |
| qpos_min | 0.046283 |
| qpos_start | 0.023022 |
| eef_speed_mean | 0.021901 |
| qpos_velocity_mean | 0.019611 |

## Phase Bin Distribution

| Bin | Count |
|-----|-------|
| approach_far_closed_proxy | 40 |
| approach_near_closed_proxy | 95 |
| grasp_formation_pre_lock_proxy | 11 |
| natural_open_or_release_proxy | 224 |
| pre_lock_closed_proxy | 8 |

## Vulnerability Descriptor Audit (Batch1)

Top features separating positive (ketchup/butter) vs negative (alphabet_soup):
| eef_speed_max | pos=0.317095 | neg=0.45281 | diff=-0.135715 |
| eef_displacement | pos=0.161107 | neg=0.091411 | diff=0.069696 |
| eef_speed_mean | pos=0.237649 | neg=0.191531 | diff=0.046118 |
| eef_z_delta | pos=0.000636 | neg=0.002977 | diff=-0.002341 |
| qpos_start | pos=0.039393 | neg=0.03896 | diff=0.000433 |

## Verdict

Phase bins are learnable from runtime features.
The top models achieve macroF1=1.0.
This suggests a Causal TCN detector can identify pre-grasp vs grasp-formation windows
from clean runtime features, even without privileged state.

Next: replace coarse phase_bin_proxy with Batch2b VIS-informed vulnerability_ready label.
