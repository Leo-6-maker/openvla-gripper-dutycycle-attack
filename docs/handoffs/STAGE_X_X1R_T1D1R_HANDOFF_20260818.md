# Stage X X1R T1-D1R handoff — 2026-08-18

## Decision state

`PASS_D1R_CONTINUATION_CENSUS_PRE_MANUAL_REVIEW`

This is a `SCREENING_CLEAN` census only. It is not X1R efficacy evidence and
does not authorize PGD, random controls, shuffled-gradient controls, physical
intervention, V_phys reads, Eval160, or protected evaluation.

## Frozen population closure

- 40 nominal design cells.
- `libero_goal/task_01` remains the one structural missing cell.
- 39 frozen executable identities.
- Ordinals 1, 11, 20, and 30 remain immutable D1 runtime-invalid canaries
  after the first policy decision. They were not rerun, replaced, or removed.
- D1R continuation contains exactly 35 identities: ordinals 2–10, 12–19,
  21–29, and 31–39. No replacement and no rerank occurred.

## D1R census

The aggregate audit re-read every durable parent receipt, episode manifest,
step telemetry file, worker receipt, and referenced video hash.

| quantity | count |
|---|---:|
| continuation planned | 35 |
| runtime-valid receipts | 35 |
| runtime-invalid continuation receipts | 0 |
| clean success | 23 |
| clean failure | 12 |
| first Student emit | 25 |
| first emit with legal horizon | 25 |
| first emit with illegal horizon | 0 |
| NO_EMIT retained | 10 |
| pre-manual-review attack-eligible | 14 |

The 14 eligible rows are only `attack_eligible_pre_manual_review`; clean
contact validity has not been manually reviewed. `done=True` is therefore not
being promoted to physical-contact validity.

## Provenance boundaries

- PR #129 immutable live head: `4b0ceb65f8f7babdd29163e032c56fed3ba57526`,
  tree `d7b688e82bf0b9c5e91c08b3ad15c3a6d94b89ad`.
- Historical D1 runtime source: `f079dcfa2bdafe65291cfdd1fbd2913b2a42668f`,
  tree `6efe25a144efc43473897e5ff8e08e2124940b70`.
- Repaired source pre-evidence: `109cb9a4698fa462de55e5453e3c24175a81e946`,
  tree `a1675c0a604a4f2d43e652f17339acddb78bce01`.
- D1R episode evidence runtime source: `d74b8b7aff311c4ebbd51bf83ff026efe48d0236`,
  tree `2ee7425fc9177d70abb61f12b644833ec20d0a06`.
- Census audit source is recorded separately in the root seal; it must not be
  confused with the source that generated the episodes.

## Protected boundary

All forbidden counters are zero. `Eval160=UNREAD` and
`protected_evaluation=UNREAD`. Teacher/Student checkpoint, features,
normalization, thresholds, and scheduler semantics were not changed.

## Durable evidence

- Durable root:
  `/llm_jzm/dty_user/openvla_attack_d1_screening_clean_20260818/D1R_CONTINUATION`
- Census:
  `reports/STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1.json`
- Hash list:
  `reports/STAGE_X_X1R_T1D1R_SHA256SUMS.txt`
- Root seal:
  `reports/STAGE_X_X1R_T1D1R_ROOT_SEAL.json`
- Census SHA256:
  `7617a911f01f4a7919f9275eed0f57c3693a701e06a0e9ce267bc70551183a91`
- SHA256SUMS SHA256:
  `f078c88033fcd31a2e4b19bb06256843800eb43460a9322031af757a0f16bc88`
- Root-seal SHA256:
  `0410c8042aa1cd0f6b3011c48c548130cf196670d3bb359294ea22e0baadf088`

## Stop condition

Stop here for owner/GPT review of the census and the 35 clean videos. The next
legal decision is `OWNER_REVIEW_D1R_CENSUS_AND_MANUAL_CONTACT_VALIDITY`.
