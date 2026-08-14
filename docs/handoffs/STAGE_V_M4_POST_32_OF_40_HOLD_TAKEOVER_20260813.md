# Stage V post-32/40 HOLD takeover — 2026-08-13

## Current scientific state

The V2 current-source corridor remains permanently sealed as:

`HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`

The authoritative report is mirrored at [`reports/server_evidence/STAGE_V_M4_CORRIDOR_CURRENT_SOURCE_AB_RECONCILIATION_HOLD_V1.json`](../../reports/server_evidence/STAGE_V_M4_CORRIDOR_CURRENT_SOURCE_AB_RECONCILIATION_HOLD_V1.json) with SHA-256 `866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9`.

The sealed 40-parent result is:

- PASS/PASS: `32/40`
- PASS/CLEAN_FAILURE: `3`
- CLEAN_FAILURE/CLEAN_FAILURE: `4`
- INELIGIBLE/INELIGIBLE: `1`
- protected reads, attack, VIS, and Eval160: `0`
- formal M4 authorized: `false`
- M4 outcomes and `V_phys`: not read/generated

The predecessor source is immutable `3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2` / tree `2492a075e782a112d1e857248956b2647e751039`. The unchanged corridor science runner is SHA-256 `26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279`.

## Frozen correction

[`configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM.json`](../../configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM.json) is append-only. It does not modify the original freeze. The locked order is:

`corridor replenishment → composite reconciliation → final40/split → exact 40×24 plan/snapshot → primary firewall → Teacher freeze → Student freeze → formal M4 outcome read`

M4 outcome read additionally requires the frozen Teacher/Student hashes, feature schema, threshold, and primary firewall hash. No Teacher/Student or M4 runtime is consumable from this branch yet.

## Post-HOLD population

[`configs/STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1.json`](../../configs/STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1.json) is frozen before any new receipt. It contains 25 candidates:

- L10: `9`, target stable A/B pairs: `1`
- Goal: `8`, target stable A/B pairs: `3`
- Object: `0`, target stable A/B pairs: `0`
- Spatial: `8`, target stable A/B pairs: `4`

The 25 count is not a hand blacklist: every V7-qualified unattempted candidate was checked with the existing uniform `taxonomy_eligibility_from_bddl()` rule. Goal fixture-only identities are excluded by that same structural rule. The manifest also binds the mirrored terminal report and rejects any intersection with the already-attempted current 40 identities.

[`configs/STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1.json`](../../configs/STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1.json) fixes the per-suite rank and stopping rule. A candidate that is attempted is quarantined from primary Teacher/Student FIT, CAL, CHECK, threshold selection, and model selection, regardless of its result.

## Execution adapter

[`scripts/detector_v5/run_stage_v_m4_post_hold_sequential.py`](../../scripts/detector_v5/run_stage_v_m4_post_hold_sequential.py) is an outer adapter only. It:

1. validates the post-HOLD protocol, candidate hash, sealed terminal report, source binding, and no-overlap firewall;
2. materializes a compatibility protocol/manifest for the unchanged science runner;
3. runs A/B clean-only per candidate in frozen suite/rank order;
4. stops each suite at its pre-registered target or pool exhaustion;
5. fails closed on runner/receipt/boundary errors and never reads intervention outcomes.

The local plan-only check passed with no GPU work. The command is:

```powershell
python scripts/detector_v5/run_stage_v_m4_post_hold_sequential.py `
  --post-hold-protocol configs/STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1.json `
  --candidate-manifest configs/STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1.json `
  --science-runner scripts/detector_v5/run_stage_v_m4_corridor_preflight.py `
  --python <official-python> --official-snapshot-root <snapshot> `
  --upstream-root <upstream> --model-root <models> --output-root <new-output-root> `
  --source-commit 3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2 `
  --source-tree 2492a075e782a112d1e857248956b2647e751039 `
  --gpus 1,2,3,4,5,6,7 `
  --owner-basis "Goal Mode continuation: clean-only post-HOLD corridor replenishment; no M4/attack/Teacher/Student runtime." `
  --plan-only
```

## Server handoff

Use environment:

`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

Immutable science worktree:

`/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m4-governed-20260812`

Current report root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_ARCHITECTURE_FREEZE_REBUILD_V1_20260813T000000Z`

Before any GPU launch, query `nvidia-smi --query-compute-apps` and exclude foreign owners dynamically. Do not terminate the historical `zkx`/`zcs` processes on GPU0 or the old `dty_user` controllers. GPUs 1–7 were idle at takeover, but that is not a permanent authorization.

The next legal action after this branch is published is: checkout the exact branch tip on the server, run the same plan-only gate, perform a fresh GPU/process preflight, then run only the clean-only post-HOLD adapter. Do not freeze final40/split, start Teacher/Student, read M4 outcomes, launch attack/VIS, or read Eval160 from this handoff.

## Verification boundary

Passed locally: JSON parse, `git diff --check`, Python compileall, and the post-HOLD plan-only gate. The Windows checkout has no `pytest` module, so pytest was not claimed as passed. No new GPU receipt has been generated by this branch.
