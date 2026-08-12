# Stage V M4 Spatial Replenishment Qualification V3

Date: 2026-08-13 (Asia/Shanghai)

## Current disposition

PASS_CLEAN_QUALIFICATION_ONLY. Four libero_spatial identities passed fresh
clean A/B qualification and independent audit. This is qualification evidence,
not M4 intervention evidence and not a formal M4 label source.

The final formal corridor remains HOLD until the new four identities are
merged into a new 40-parent manifest and the complete firewall, supersession,
and independent reconciliation gates pass. Teacher, Student, M4 outcomes,
VIS, attack, and protected Eval160 remain untouched.

## Frozen bindings

- Protocol:
  configs/STAGE_V_M4_SPATIAL_REPLENISHMENT_QUALIFICATION_PROTOCOL_V3.json
  SHA256 46c306d644d2f850792fcdf4643df68f4c30a3070b15f4af729cb0a3bcc9af6f
- Scientific architecture freeze SHA256:
  377912a916bd155f2217084ba7385d42b5eae1803c48e0497cd8982a51e1571b
- Runtime code snapshot: commit
  73bc7287a03457b105c035025a91dcb03883876f, tree
  763291b23174affe63296941efc4dc65ee7ec829
- Official Python:
  /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
- Governed worktree:
  /mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m4-governed-20260812
- Qualification output:
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_SPATIAL_REPLENISHMENT_QUALIFICATION_V3_20260812T162520Z

## Result receipts

- CONTROL_QUALIFICATION_REPORT.json
  SHA256 c6c7d7dfe10479aacf221d7a1561e50681adea0ce6900ec455fd0616a221d524
- CONTROL_QUALIFICATION_ROWS.jsonl
  SHA256 00c3ec026767a04d3ca2821004fe35c4cf4ebb43c4abf21b5c935ab9a25893fe
- CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json
  SHA256 abdb581467dfc0c035b43d9e8d97cf395d224d49acde67d79fb7412c865722db
- STAGE_V_FORMAL_PARENT_MANIFEST_V1.json
  SHA256 5d028dbe824ad8130448b9af18b092d740d275081be74e3c48919b51c2f9b9ec
- STAGE_V_FORMAL_PARENT_MANIFEST_V2.json
  SHA256 364c51681791c7403937ba720477ddc2ba00b002077cfcff4ffeba99151310cd
- STAGE_V_R2_PARENT_MANIFEST_A.json
  SHA256 364c51681791c7403937ba720477ddc2ba00b002077cfcff4ffeba99151310cd
- PRE_QUALIFICATION_RESOURCE_RECEIPT.json
  SHA256 91fb5deaf096d0c2192eb1858a49ca9db045053f9c02ab417a229c88a9b29cec

The V1/V2 manifest checksum sidecars both self-verify. The terminal queue was
done=8, locked=0, pending=0, retry_ready=0, running=0.

## Qualified identities

| Suite | Parent |
| --- | --- |
| libero_spatial | libero_spatial/task_03/state_40 |
| libero_spatial | libero_spatial/task_06/state_29 |
| libero_spatial | libero_spatial/task_06/state_45 |
| libero_spatial | libero_spatial/task_07/state_39 |

All four rows have qualified=true, empty producer errors, and independent
recomputed qualification. Both A and B for each row passed clean success,
task identity, exact snapshot restore, runtime validity, finite metrics,
artifact validation, and complete horizon checks. Every A/B result bound the
runtime source to the frozen commit/tree and reported empty worker Git status.

## Boundary audit

- eval160_reads=0
- protected_eval_reads=0
- vis_pgd_attack_rollouts=0
- attack_rollouts=0
- old_artifacts_reused=false
- source_artifacts_modified=false
- no M4 intervention labels or outcomes generated
- no Teacher/Student or attack runtime started

An initial SSH wrapper attempt failed at argparse because the runner-command
quote was split; it produced no output directory and no worker. The retained
qualification output above is the single corrected V3 run, with a new output
root and complete receipts. This wrapper incident is engineering provenance,
not a scientific branch result.

## Next legal gate

Reconcile this PASS against the existing 29 stable plus 7 valid replenishment
identities, create a new final 40-parent manifest, rerun the full firewall and
formal authorization audit, and only then freeze the clean Teacher/Student
primary evidence package. The small-attack binding remains
HOLD_AUTHORITATIVE_OWNER_APPROVAL_UNRESOLVED; no attack protocol may be
selected or executed from this qualification result.
