# C2F Teacher v1 Label Audit

Date: 2026-07-10

Status: `PASS_C2F_TEACHER_V1_LABEL_AUDIT_WITH_PROVENANCE_LIMITS`

## Inputs and boundary

The CPU-only audit read the existing Clean2000 merged root and replaced its stale Object labels with the explicit Object v1.1 source:

- base: `/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_merged_199af7b`
- Object override: `/mnt/sdc/dty_user/openvla_attack_evidence/c2f/object500_v1.1_fd3e2db`
- Teacher source SHA256: `e6a074dcc5ccfce01bb2e2f43d6185d681c396723497a6b000751fd9338f38de`

The scan covered 2,000 episodes and 393,513 clean step rows with zero read errors. It did not load OpenVLA, create a LIBERO environment, replay a counterfactual, or train a detector.

## Suite findings

| Suite | Episodes | Step rows | Stable carry | Primary rows | Primary density | Grounding proxy coverage | Episodes with primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| LIBERO-10 | 500 | 139,085 | 68,513 | 43,983 | 31.62% | 65.10% | 245 |
| Goal | 500 | 78,959 | 78,055 | 36,178 | 45.82% | 52.29% | 427 |
| Object v1.1 | 500 | 100,325 | 17,896 | 12,673 | 12.63% | 76.28% | 226 |
| Spatial | 500 | 75,144 | 74,644 | 50,618 | 67.36% | 69.15% | 498 |

Grounding is a Teacher-v1 proxy: `primary_attackable` and `distractor_or_setup` imply that the labeler selected an object; `unsupported_or_abstain` implies no confident object. Teacher-v1 did not store the contacted object or the target-match decision itself.

## Pathologies confirmed

- Spatial has 56,704 absolute-z fallback candidates, 75.97% of its stable-carry rows. Exact fallback branch usage was not recorded, so this is a conservative 25D-derived candidate count.
- Object task 1 has no primary labels. Object task 5 has only 0.78% primary density. Object stable-carry misses comprise 4,245 no-grounded-object rows and 978 target-mismatch/distractor rows.
- LIBERO-10 tasks 0, 1, 4, 6, and 7 have no primary labels. L10 stable-carry misses comprise 23,912 no-grounded-object rows and 618 target-mismatch/distractor rows, consistent with unresolved multi-object grounding.
- Goal task 0 has no primary labels. Goal contains 37,242 no-grounded-object and 4,635 target-mismatch/distractor stable-carry rows.
- Release-safe is nearly absent: 7 rows in L10, 21 in Goal, and zero in Object/Spatial. Source inspection confirms it is triggered by a gripper closed-to-open transition rather than target-relative placement evidence.
- Task-level first-primary medians range from 41-89 steps in L10, 25.5-94.5 in Goal, 86-108 in Object, and 24-44 in Spatial. Full per-task timing and duration fields are in the committed CSV.

## Provenance gaps

Teacher-v1 rows do not record contacted object identity, target-match provenance, stable reason codes, or exact absolute-z fallback usage. These missing fields are unknown, not negative evidence. The audit tool therefore reports explicit-field coverage separately and derives only named proxies from phase, role, and 25D values.

Artifacts:

- `reports/c2f_teacher_v1_audit/teacher_v1_audit_report.json`
- `reports/c2f_teacher_v1_audit/teacher_v1_by_suite_task.csv`
- `reports/c2f_teacher_v1_audit/teacher_v1_reason_codes.csv`

Gate:

```text
TARGET_GROUNDING_AUDIT = PASS_WITH_V1_PROVENANCE_LIMITS
SPATIAL_LABEL_DENSITY_EXPLAINED = PASS
OBJECT_MISSED_PRIMARY_CAUSES_EXPLAINED = PASS_PROXY
TEACHER_V1_FOR_TRAINING = HOLD_PENDING_TEACHER_V2
```
