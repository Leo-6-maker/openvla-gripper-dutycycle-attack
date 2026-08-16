# Stage VIII R0 relative timing identifiability handoff

Historical R0 handoff. Superseded for current Stage VIII status by
`STAGE_VIII_R1_RELATIVE_SELECTOR_NEGATIVE_HANDOFF_20260817.md`.

Status: `STAGE_VIII_RELATIVE_TIMING_IDENTIFIABILITY_ESTABLISHED`

R0 was a read-only audit over consumed Stage V/VI development evidence. It
did not train a model, retune a threshold, execute an intervention, launch
M4/PGD, or read Eval160/protected data.

## Immutable bindings

- PR: #117.
- Protocol/source commit: `40894eac67e0aae303da3eb461a7fbfcf0992540`.
- Source tree: `5ffe0944fc97d347fb8308c7d6897d53124b243a`.
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
- R0 root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VIII_RELATIVE_TIMING_SELECTOR/STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY_20260816T154705Z`.
- R0 root `SHA256SUMS`, `SHA256SUMS.sha256`, and `ROOT_SEAL.sha256` independently pass.
- The frozen protocol was copied into the R0 root and hashed before metrics were read.

## Population and source status

The primary population was T5 only, using binary-consumable labels and
masking all censoring/abstain classes. Comparisons were within the same
`canonical_parent_key`; no row-random split or manual Spatial score reversal
was used. There were 1,191 exact consumable T5 rows across Stage V and
Stage VI-B2, with 28 pair-eligible parents containing both V_PHYS and
NO_PHYSICAL_VULNERABILITY probes.

- S7-A: exact coverage, 1,191/1,191 consumable rows.
- S7-B: exact coverage, 1,191/1,191 consumable rows.
- S7-C: exact coverage, 1,191/1,191 consumable rows.
- B2-C: marked `NOT_IDENTIFIABLE_SCORE_SOURCE` under the fail-closed rule
  because its historical root lacks `ROOT_SEAL.sha256`; the prediction file
  was not altered or approximated.

## Relative results

| source | parent-macro AUC | pooled pair AUC | top-1 lift | top-3 lift | zero-regret rate | 95% parent-bootstrap CI |
|---|---:|---:|---:|---:|---:|---|
| S7-A | 0.8112 | 0.8049 | 1.5089 | 1.3996 | 0.8214 | [0.6927, 0.9169] |
| S7-B | 0.7931 | 0.7883 | 1.5089 | 1.3996 | 0.8214 | [0.6648, 0.9069] |
| S7-C | 0.8567 | 0.8183 | 1.5089 | 1.4652 | 0.8214 | [0.7475, 0.9472] |

The frozen R0 gate required parent-macro AUC `>=0.65`, every identifiable
suite AUC `>=0.58`, top-1 lift `>=1.30`, top-3 lift `>=1.20`, and zero-regret
margin over the parent random baseline `>=0.15`. S7-A, S7-B, and S7-C each
passed all five checks.

Per-suite parent-macro AUC:

| source | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---:|---:|---:|---:|
| S7-A | 0.8891 | 0.7156 | 0.7812 | 0.6667 |
| S7-B | 0.8672 | 0.6813 | 0.7701 | 0.6792 |
| S7-C | 0.9162 | 0.7413 | 0.9093 | 0.6667 |

The result answers the R0 question positively: the existing S7 score sources
contain useful within-trajectory ranking signal despite the Stage VII global
development generalization failure. This is development evidence, not fresh
causal validation and not a universal probability claim.

## Decision and next gate

R1 relative timing selector development is authorized. It must use at most
the two preregistered candidates, same-parent T5 pairwise ranking, frozen
parent-grouped splits, and the separate R1 promotion gate. No R1 training has
started from this handoff; no fresh physical intervention is authorized yet.

Protected boundary remains unchanged: `Eval160=UNREAD`, protected evaluation
`UNREAD`, and all protected counters are zero.
