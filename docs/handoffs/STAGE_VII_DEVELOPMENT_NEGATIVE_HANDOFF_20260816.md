# Stage VII development negative handoff — final A/B/C decision

Status: `STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR`

This final handoff supersedes the initial negative handoff preserved at
`STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_INITIAL_20260816_SUPERSEDED.md`.
The earlier split and forensic handoffs remain historical snapshots of their
pre-training states; this document is the current candidate status.

## Immutable bindings

- PR: #116.
- Decision source commit/tree: `8ad9859a61a0083948c4e7b73eed72d7bf1d2aad` / `d4f6da984041924db3f35eb2be812cd9e8c444fb`.
- Decision root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_DEVELOPMENT_DECISION_20260816T150100Z`.
- Forensic root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_DOMAIN_SHIFT_FORENSIC_V2_20260816T093631Z`.
- S7-A root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_S7A_CORRECTED_CANDIDATE_DEVELOPMENT_20260816T081039Z`.
- S7-B root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_S7B_CANDIDATE_DEVELOPMENT_20260816T182500Z`.
- S7-C root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_S7C_CANDIDATE_DEVELOPMENT_20260816T184000Z`.

The decision script independently verified the candidate `SHA256SUMS`,
sidecars, and root seals. No candidate root was regenerated or overwritten.

## Frozen candidate outcomes

All candidates used the frozen threshold `0.69`, parent-grouped evaluation,
and the predeclared promotion gates. All three reached sealed fail-closed
development outcomes; none was promoted.

| candidate | DEVTEST AUROC | AUPRC lift | top-decile lift | ECE-10 | decisive suite failure |
|---|---:|---:|---:|---:|---|
| S7-A | 0.7235759 | 1.2199992 | 0.9938350 | 0.2196616 | `libero_object` AUROC 0.4994438; `libero_spatial` AUROC 0.0531250 |
| S7-B | 0.8379506 | 1.3469357 | 1.2276786 | 0.1426743 | `libero_object` AUROC 0.5939933; `libero_spatial` AUROC 0.0718750 and emission 0.9420290 |
| S7-C | 0.8259068 | 1.3334355 | 1.2861395 | 0.1330486 | top-decile gate; `libero_spatial` AUROC 0.0546875 and emission 0.9855072 |

The negative result is not a claim that every runtime feature is
uninformative. It records that none of the three frozen candidates met the
predeclared generalization and selectivity gates across the identifiable
suites.

## Authorization boundary

- Stage V and Stage VI-B2 remain immutable; Stage VI-B2 retains
  `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`.
- No fresh holdout, new formal M4, timing/selectivity matrix, or PGD rollout
  is authorized from this Stage VII result.
- `Eval160=UNREAD`; protected evaluation remains `UNREAD`.
- Protected counters are all zero: `attack_rollouts=0`, `eval160_reads=0`,
  `protected_reads=0`, and `vis_pgd_attack_rollouts=0`.
- No retraining, threshold retuning, outcome-driven parent replacement, or
  rerun-to-pass occurred.
