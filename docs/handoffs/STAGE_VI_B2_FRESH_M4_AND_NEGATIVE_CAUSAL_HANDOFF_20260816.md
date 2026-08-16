# Stage VI-B2 fresh M4 and causal handoff — 2026-08-16

## Current decision

The fresh Stage VI-B2 formal population completed structurally, but the frozen B2-C detector did not establish held-out causal localization. This is a valid negative scientific conclusion, not a runtime hold.

`STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`

The conditional 16-parent / 64-branch timing matrix was not authorized because the causal gate failed. Eval160 and protected evaluation remain unread and unauthorized.

## Frozen bindings

- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- Formal runtime worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-vi-b2-formal-source-14606ba`
- Formal runtime commit/tree: `800b19341e39e71ce75991dc9a13f796bdf5ffdf` / `71e5c28a99752960b092ab2a21d732069ed438ab`
- Selected detector: `B2-C_SOFT_TV_DISTILL_DIRECT_VPHYS`
- Input: frozen `16x25D_causal_window`
- Threshold: `0.69`
- Model SHA-256: `fc047fd8b1b2d118c3275d51b56cec7306725e4c41aff020121312e6a4daca8b`
- Feature order SHA-256: `3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366`
- Teacher, Student weights, features, and thresholds were not changed after the lock.

## Formal M4 evidence

Fresh V3 population: 16 parents, 384 probes, 1,536 planned branches. The formal aggregate passed with 1,152 treatment labels, 1,004 consumable binary labels, and 148 preserved abstains/censored labels. T5 primary closure was 384 branches: 333 consumable and 51 abstain/censored.

- Zero-treatment plan: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FRESH_ZERO_TREATMENT_PLAN_V3_DUAL_SOURCE_20260816T052000Z`
- Formal M4 scheduler: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FORMAL_M4_V3_DUAL_SOURCE_20260816T053000Z`
- Formal M4 aggregate: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FORMAL_M4_AGGREGATE_V3_DUAL_SOURCE_20260816T061432Z`
- Aggregate `SHA256SUMS.sha256`: `04ebc2f8179b7954ac188d791b4a06640cd88c62d88e7709ce6bbf8200887c11`

All historical Stage V and rejected Stage VI-B2 identities remain nonconsumable for this fresh population.

## Clean feature reconstruction

The fresh M4 clean files did not contain the exact telemetry fields required by the frozen R3 25D materializer, so a diagnostic-only clean replay reconstructed them. The replay passed 16/16 parents and 16/16 full-coverage validations. It used no policy inference, reward/done/info, intervention, outcome, label, or protected read.

- R3 reconstruction root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FRESH_M4_R3_CLEAN_RECONSTRUCTION_V3_20260816T063900Z`
- Registry binding: `c1_v2_r7/run_B/per_task`
- R3 source binding: `800b19341e39e71ce75991dc9a13f796bdf5ffdf` / `71e5c28a99752960b092ab2a21d732069ed438ab`

Earlier registry/launcher failures were pre-action engineering holds and contain no intervention or label evidence; they remain preserved separately.

## Held-out B2-C result

Predictions were generated once from the frozen checkpoint at threshold `0.69`. Abstains were excluded from binary consumption and retained in a separate censoring map.

- Validation root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FRESH_HOLDOUT_B2C_VALIDATION_V1_20260816T064700Z`
- T5 consumable rows: `333` (`223` V_PHYS, `110` NO_PHYSICAL_VULNERABILITY)
- T5 abstains: `51`
- Overall AUROC: `0.6246432939`
- Overall AUPRC: `0.7976720489`
- Overall AUPRC lift: `1.1911425664`
- Overall top-decile lift: `1.4493537325`
- Overall ECE-10: `0.4606357016`
- Fixed-threshold emission: `0.4324324324`

Suite-level failure modes were material: `libero_10` emitted at rate `0`, `libero_spatial` at `0.9583333333`, `libero_object` had no negative consumable examples and therefore no identifiable AUROC, and overall ECE exceeded the frozen `0.25` limit.

## Final seal

- Decision root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VI_B2_FINAL_CAUSAL_DECISION_V1_20260816T065000Z`
- Decision status: `PASS_VALID_NEGATIVE_CAUSAL_CONCLUSION`
- Timing matrix: `NOT_AUTHORIZED_CAUSAL_GATE_FAIL`
- Protected counters: `protected_reads=0`, `eval160_reads=0`, `attack_rollouts=0`, `vis_pgd_attack_rollouts=0`

This handoff records sealed server evidence only; it does not promote B2-C or alter any frozen Stage V artifact.
