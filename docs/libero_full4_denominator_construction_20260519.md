# LIBERO Full4 Clean Denominator Construction

Status as of 2026-05-19.

This document records the clean-only denominator construction run used to evaluate whether the project has enough non-Black-Bowl LIBERO samples for a later microbreadth VIS/random pilot.

No VIS, random-window, oracle, benchmark, or attack run is part of this phase.

## Code And Run Scope

- Branch base: `refactor/generic-autowindow-cqv2-20260517`
- Driver: `scripts/libero_full4_state0_balanced_driver_20260519_gpu014526.sh`
- Prepare/summarize helper: `scripts/libero_full4_state0_balanced_prepare_20260518.py`
- Watcher: `scripts/libero_full4_state0_balanced_watcher_20260519_gpu014526.sh`
- Full4 task config generated on server: `configs/v4_tasks_libero_full4_20260518.yaml`
- GPU queue used on the 2080Ti server: `0,1`, `4,5`, `2,6`
- GPU3 and GPU7 were avoided for formal denominator rollouts.

## Server Output Root

```text
/data/liuyu/outputs/libero_full4_state0_balanced_clean_20260519_gpu014526
```

Important generated tables:

```text
tables/full4_dataset_manifest_frozen.csv
tables/phase2_smoke_manifest.csv
tables/phase2_5_seed_state_sanity_comparison.csv
tables/phase3_clean_jobs.csv
tables/clean_success_manifest.csv
tables/mechanism_eligibility_manifest.csv
tables/clean_cqv2_metrics.csv
tables/microbreadth_pilot_candidate_queue.csv
```

## Final Decision

```text
detector_not_general_enough
```

The run completed normally, but did not produce a non-Black-Bowl eligible denominator.

Summary counts:

```text
clean rows: 107
progress status: 71 done, 36 failed
non-Black-Bowl eligible rows: 0
microbreadth pilot candidate rows: 0
```

Suite-level clean outcomes:

```text
libero_spatial: 34 rows, 18 clean-success, all Black Bowl sanity
libero_object: 31 rows, 0 clean-success
libero_goal: 31 rows, 16 clean-success
libero_10: 11 rows, 0 clean-success
```

Mechanism/window outcome:

```text
pick_place classified rows: 26
articulated_or_non_transfer rows: 8
clean_unstable rows: 73
window_detected: 0
mechanism_eligible: 0
```

Main blockers:

```text
detector_abstain_or_low_confidence
clean_failure
artifact_failure due missing LIBERO assets in Object/LIBERO-10 paths
non_transfer_mechanism for articulated/open/turn/push tasks
```

## Interpretation

This result supports continuing denominator repair before any attack expansion. The current non-Black-Bowl breadth denominator is not ready for a VIS/random microbreadth pilot because clean-success trajectories did not yield medium/high generic-v4 windows.

The Black Bowl generic-v4 + CQ-v2 result remains the strongest mechanism evidence. This Full4 run should be treated as a denominator diagnostic, not an attack result.
