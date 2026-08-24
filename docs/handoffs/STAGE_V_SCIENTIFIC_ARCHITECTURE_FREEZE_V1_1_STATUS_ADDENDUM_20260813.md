# Stage V Scientific Architecture Freeze V1.1 status addendum — 2026-08-13

This is an append-only status correction. It does not edit or weaken `STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1` and changes no architecture semantics.

The scientific separation remains `C_t != V_t(d) != E_t`. The mandatory order remains clean privileged Teacher, clean-Teacher-supervised causal Student, complete model/feature/threshold/scheduler freeze, and only then held-out M4 intervention outcomes. No `M4 → Teacher/Student` primary path is authorized.

## Status supersession

The old `29/40` fields in the architecture freeze are historical snapshots. The authoritative current corridor state is the immutable `32/40` terminal HOLD:

- `PASS/PASS = 32/40`
- `PASS/CLEAN_FAILURE = 3`
- `CLEAN_FAILURE/CLEAN_FAILURE = 4`
- `INELIGIBLE/INELIGIBLE = 1`
- status: `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`

V2 is not repaired, upgraded, or reopened. Its terminal report is:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_ARCHITECTURE_FREEZE_REBUILD_V1_20260813T000000Z/STAGE_V_M4_CORRIDOR_CURRENT_SOURCE_AB_RECONCILIATION_HOLD_V1.json`

SHA-256: `866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9`.

No formal M4 authorization exists, no M4 intervention outcome has been read, and no `V_phys` map exists. Historical Teacher/G0 work remains `ENGINEERING_HISTORY / NONCONSUMABLE`; there is no consumable primary Teacher or Student manifest, checkpoint, inference, or model-selection line.

Protected state remains exactly:

- `protected_reads = 0`
- `eval160_reads = 0`
- `attack_rollouts = 0`
- `vis_pgd_attack_rollouts = 0`

## Exact architecture binding

The current whole-file SHA-256 of `configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json` is `377912a916bd155f2217084ba7385d42b5eae1803c48e0497cd8982a51e1571b`.

Its relevant Git history is append-only:

- initial freeze: commit `73bc7287a03457b105c035025a91dcb03883876f`, tree `763291b23174affe63296941efc4dc65ee7ec829`
- latest architecture update: commit `15fc24e53a7627e8fa64e13981f1245d46faad48`, tree `15f9120fdb3919fe77732bef2cd2af6a14f21b1e`

The immutable corridor execution plane remains:

- server worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m4-governed-20260812`
- commit: `3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2`
- tree: `2492a075e782a112d1e857248956b2647e751039`
- corridor runner SHA-256: `26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279`

At takeover, PR #111 was OPEN/DRAFT at `fcaa59cacf1895cc9f1d372944366b7b2952911c` (tree `61b1bfd76b18dd584144a9b105ad0abf3f18dac0`) with green engineering checks. The stacked PR #112 was OPEN/DRAFT at initial head `280dba8dec66adbdd50dc213983adfae392ae707` (tree `a0f0c21ba2f59884a6270f4075fd318b505cad8d`). These GitHub checks are engineering evidence only.

## Locked next order

The legal sequence is:

`post-HOLD corridor replenishment → independent composite reconciliation → final40/split freeze → exact 40×24 plan-and-snapshot-only gate → primary data firewall → clean Teacher freeze → causal Student freeze → pre-M4 outcome lock → formal M4 authorization → held-out M4 outcome read`.

Eval160 remains unread and unauthorized.
