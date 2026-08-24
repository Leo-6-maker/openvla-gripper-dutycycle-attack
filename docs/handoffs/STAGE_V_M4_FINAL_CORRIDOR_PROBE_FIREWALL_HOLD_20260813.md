# Stage-V M4 / primary-data hold — 2026-08-13

## Current verdict

The final candidate identity and split are valid: 40 unique parents, split TRAIN/VAL/TEST = 24/8/8, with 6/2/2 per LIBERO suite. The V1.4 clean-only final-candidate preflight completed all 22 A/B branches:

- 16 `PASS`
- 4 `CLEAN_FAILURE`
- 2 `INELIGIBLE`

This is a clean preflight only. It generated no M4 labels, read no outcomes, and executed no intervention.

## Exact-probe gate

The server reconciliation supports 37/40 final parents and 888 exact probes. The three unsupported identities are:

- `libero_10/task_08/state_28`
- `libero_goal/task_07/state_41`
- `libero_spatial/task_03/state_40`

The historical 29-parent A/B support has 12/29 exact-step mismatches. The 8 newly supported parents also have mixed A/B plans, with only 4 exact matches. Because the historical support uses commit/tree `7d5cc9762f2c0e66df6a701ceb76bda902318543` / `d152141b91d8d02e268a390d99c3c109775c6611` and the current preflight uses `73bc7287a03457b105c035025a91dcb03883876f` / `763291b23174affe63296941efc4dc65ee7ec829`, the exact-probe manifest remains `HOLD_MIXED_HISTORICAL_7d5_AND_CURRENT_73bc_PROVENANCE`. No averaging or outcome-informed replicate selection is allowed.

Evidence pointers are recorded in [STAGE_V_M4_FINAL_CORRIDOR_PROBE_SUPPORT_RECONCILIATION_V1.json](D:/vla_attack/repo_work/openvla-gripper-dutycycle-attack-resource-contract-20260810/reports/STAGE_V_M4_FINAL_CORRIDOR_PROBE_SUPPORT_RECONCILIATION_V1.json).

## Primary-data firewall

The read-only firewall confirms zero overlap between the final 40 and all 8 primary FIT/G1 manifests (FIT670 allowlist/shards, G1 episode train/val/test, and G1 task train/val/test). The historical G10 registry overlaps 40/40, so it remains quarantined with outcomes unread and is not primary evidence.

The current primary Teacher/Student line is `NOT_STARTED_OR_NOT_CONSUMABLE`: no current CAL/CHECK/model-selection manifest is bound, and formal training/inference are unauthorized. Historical R3/G2/G6/G7 artifacts remain non-consumable; they do not satisfy the new clean-only primary freeze.

Evidence pointers are recorded in [STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V2_1.json](D:/vla_attack/repo_work/openvla-gripper-dutycycle-attack-resource-contract-20260810/reports/STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V2_1.json).

## Downstream boundary

Protected reads, Eval160 reads, attack rollouts, and VIS-PGD attack rollouts are all zero. Do not launch matched M4, Teacher/Student runtime, VIS, or the 32-cell attack canary until the exact-probe support and current primary firewall are closed under one current source binding. The attack protocol binding remains `HOLD`; no attack hyperparameters were selected from historical text.
