# Claim and Evidence

## Main Claim

The frozen claim is duty-cycle shaping via gripper-targeted temporal PGD: during contact-critical black-bowl manipulation windows, inference-time visual perturbations can bias the model toward repeated gripper-open commands, producing visible grasp/lift/carry failures.

## Template B Interpretation

The result is not framed as `prev_delta` uniqueness or margin-objective uniqueness. The supported interpretation is that gripper-targeted temporal perturbations can shape the command duty cycle in a vulnerable task phase.

## Frozen State7 Evidence

| Condition | Manual outcome | Interpretation |
|---|---:|---|
| VIS margin prev-delta | 4/5 manual gripper-opening failures | Primary positive evidence |
| Same-gate zero-margin | 3/5 comparable failures | Effect is not unique to `prev_delta`/margin |
| Random direction | 0/5 failures | Not arbitrary perturbation |
| Oracle continuous | 3/3 physical upper-bound failures | Task phase is physically sensitive to gripper opening |

## Claim Boundaries

- Current main evidence is LIBERO black-bowl state7 plus within-task state5 generalization.
- We do not claim broad cross-dataset generalization.
- We do not claim `prev_delta` is necessary.
- Simulator SR alone is insufficient; manual video review overrides SR for contact-rich slip/drop cases.
- Moka pots is Future Work / Dataset Diagnostics, not main attack evidence.

## Task Identity (added 2026-05-23)

### Primary experiment task

| Field | Value |
|-------|-------|
| **runner_task_id** | `libero_spatial_black_bowl` |
| **semantic_task_name** | `goal_put_the_bowl_on_the_plate` |
| **suite** | `libero_spatial` (LIBERO-Spatial) |
| **claim_label** | LIBERO Spatial black-bowl-to-plate |
| **is_black_bowl_related** | true |

The config `configs/v4_tasks_libero.yaml` maps `libero_spatial_black_bowl` to the LIBERO Spatial task
"pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate".

In run_ids and reports, the semantic label `goal_put_the_bowl_on_the_plate` is used for readability and
Table1 manifest cross-reference. The actual `--task_id` passed to the runner is always `libero_spatial_black_bowl`.

### Non-Black-Bowl (non-BB) diagnostic experiments

Non-BB bowl-on-plate diagnostic experiments (e.g., `overnight_multilane_followup_20260522_fixed`,
`state5_repair_only_followup_20260522`) use the **same** `libero_spatial_black_bowl` task. They are
mechanism diagnostics on the same black-bowl task, NOT true non-black-bowl task validations.

A true non-Black-Bowl claim requires a separate task whose runner_task_id and object semantics are
non-black-bowl (e.g., a LIBERO Goal/10/Object task with no black bowl in the scene).

Legacy directory names containing "nonbb" are historical; the claim boundary is corrected here.
