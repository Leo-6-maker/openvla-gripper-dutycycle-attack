# Stage VII domain-shift forensic handoff

> Historical pre-training snapshot. The current A/B/C candidate status is
> sealed in `STAGE_VII_DEVELOPMENT_NEGATIVE_HANDOFF_20260816.md`; the
> S7-A-only next-gate text below records the state before S7-B/S7-C were
> materialized and evaluated.

Status: `PASS_STAGE_VII_DOMAIN_SHIFT_FORENSIC`

The forensic is development-only. It did not train S7-A/B/C, modify Stage V or Stage VI-B2, launch M4, read Eval160, or read protected evaluation.

## Immutable bindings

- PR #115 remains the immutable Stage VI-B2 negative handoff.
- Stage VII PR: #116.
- Forensic source commit/tree: `e8c8c59b4259d4a584444e7d19736bf19e13210d` / `44d79f80cce49fd58a919656ee5c918f7146e914`.
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
- Sealed output: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR/STAGE_VII_DOMAIN_SHIFT_FORENSIC_20260816T075057Z`.
- Root seal: `PASS_STAGE_VII_DOMAIN_SHIFT_FORENSIC`; `SHA256SUMS` independently verified with no mismatches.

Stage V contributed 40 parents / 2,880 labels and Stage VI-B2 contributed 16 parents / 1,152 labels. Parent identity overlap is empty. T5 scoring covered 1,344 rows: 1,191 consumable and 153 abstain; abstains were masked and never treated as negatives.

## B2-C drift

The frozen B2-C checkpoint and threshold `0.69` were used unchanged.

| development source | T5 rows | consumable | abstain | AUROC | AUPRC lift | ECE | emission |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage V development | 960 | 858 | 102 | 0.9671 | 1.4465 | 0.0500 | 0.6189 |
| Stage VI-B2 fresh M4 | 384 | 333 | 51 | 0.6246 | 1.1911 | 0.4606 | 0.4324 |

The Stage VI row exactly reproduces the frozen B2-C fresh validation metrics (`AUROC=0.624643`, `AUPRC=0.797672`, `AUPRC lift=1.191143`). This is a domain-shift/calibration diagnostic, not a new causal conclusion.

## 25D distribution shift

The clean 25D comparison used 7,297 Stage V reference rows and 3,326 Stage VI rows, without labels or outcomes. Overall diagnostics were mean absolute SMD `0.1950`, maximum SMD `1.6258`, regularized Mahalanobis mean shift `155.7411`, mean PSI `1.9246`, and maximum PSI `17.1953`. These are strong distribution-shift evidence and do not authorize outcome-driven representation edits.

## Context probes

- P0, 16-step 25D parent-grouped diagnostic: OOF AUROC `0.7770`, AUPRC lift `1.2615`, top-decile lift `1.3283`, ECE `0.1527`, emission `0.5013`. It clears several preliminary gates but misses the frozen top-decile gate `1.50`; it is not a promoted detector.
- P2, 25D plus clean policy intent: only 467/1,191 consumable T5 rows (`39.21%`) had an exact usable join. The clean policy source has 3,402 conflicting duplicate identity-step joins, so it is diagnostic-only and not candidate-ready.
- P1/P3: `UNAVAILABLE_NO_FROZEN_LANGUAGE_EMBEDDING`.
- P4/P5: `UNAVAILABLE_NO_FROZEN_VISUAL_EMBEDDING`. The sealed snapshots contain raw clean RGB, `pixel_values`, and `input_ids` bytes with zero missing array files; raw bytes are not treated as embeddings.
- Oracle suite/task identity is recorded only as a diagnostic upper bound and is forbidden from final inputs.

## Decision and next gate

The forensic gate passes, but no context-conditioned candidate is authorized from the available P1/P2/P4 inputs. Freeze deterministic parent-grouped, suite-stratified TRAIN/VAL/DEVTEST and LOSO splits, then evaluate only S7-A (multidose 25D control) under the frozen Stage VII promotion criteria. Do not train S7-B/C until their required frozen embeddings and complete clean joins are independently available. Do not launch a new M4 or timing matrix from this handoff.

Protected boundary remains unchanged: `Eval160=UNREAD`, protected evaluation `UNREAD`, and all protected counters are zero.
