# Stage X X1R pre-experiment audit — 2026-08-17

## Decision

`STAGE_X_X1R_HOLD_VICTIM_PROVENANCE_MISMATCH`

This is a read-only forensic audit. No X1R protocol, runner, test suite, clean
rollout, GPU worker, physical intervention, X2 authorization, or protected read
was started. The historical X1 root remains immutable and nonconsumable.

The hold is required because the frozen Stage IX PGD implementation loads one
victim checkpoint for every row, while the Stage V and Stage VI-B2 clean/snapshot
manifests bind different policy checkpoints for three suites. The historical X1
sequence population contains only `libero_goal` and `libero_spatial` starts, so
none of its 121 starts has a path-level match to the canonical PGD victim.

## Immutable bindings reviewed

| Item | Binding |
|---|---|
| Historical PR #119 HEAD | `05f09962942a0d4e88addd7c3efa6c6b4b658768` |
| Historical PR #119 tree | `d3d637a4956f0e90f6efb5d6c9b263c8f566d9e4` |
| Historical X1 runtime commit/tree | `d5919552990bc98eb85ee2fbc45715e314d4ef81` / `0bcb4f87e1fbb9699039b3708df95988d790c53d` |
| Historical X1 result root | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_DUTY_CYCLE_MECHANISM/STAGE_X1_SEQUENTIAL_PGD_20260817T101200Z_RESEALED_V1` |
| Historical X1 result | `STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK` |
| Historical X1 root seal | SHA256SUMS verification passed; summary SHA `0677a2488d409beb5e5d7f590f1765413e8a8536bed7b97cd668eb453fa91808` |
| Official environment | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` |
| Stage IX canonical PGD contract | `08b21774141022bb199c48ef7b9b85e3b4338c2ab1d9123e1fd57f5882e42d83` |

PR #119 is still OPEN / DRAFT / MERGEABLE, with `source-registry`,
`detector-v5-cpu`, and `stageb-cpu` successful. It was not modified by this
audit branch.

## 1A. Historical decision mapping

The frozen protocol maps:

* A → `STAGE_X_SEQUENTIAL_PGD_REALIZABILITY_SIGNAL_ESTABLISHED`
* B → `STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK`
* C → `STAGE_X_NO_SEQUENTIAL_PGD_REALIZABILITY_SIGNAL`

The sealed result root and `STAGE_X_X1_RESULT.json` both state
`STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK`, therefore the historical scientific
interpretation is B. The historical handoff line 3 says “protocol outcome
`X1=C`”; that literal letter is inconsistent with the frozen mapping. This
audit records the inconsistency and does not edit or retroactively promote the
historical result.

## 1B. Historical sequence enumeration defect

The clean input audit contains 1,344 exact current frames and 56 parents. Its
exact nonempty-start census is:

| Available clean length | Starts | Parents |
|---:|---:|---:|
| 1 | 1,123 | 56 |
| 2 | 100 | 15 |
| 3–4 | 69 | 10 |
| 5–9 | 43 | 6 |
| ≥10 | 9 | 2 |
| Total | 1,344 | 56 |

The historical input audit exposes only `eligible_starts["3"]`,
`eligible_starts["5"]`, and `eligible_starts["10"]`, with counts 121, 52,
and 9. The historical runner iterates only `eligible_starts["3"]`, producing
121 sequence rows and omitting 1,223 exact nonempty starts, including all L1
and L2 Q2 opportunities. This is a structural enumeration defect, not a
scientific negative result.

## 1C. LOSO naming defect

The frozen protocol declares `evaluation.loso=true`. The runner's
`metric_bundle()` computes AUC independently within each suite and stores the
arithmetic mean in `mean_loso_auc`; it does not fit three training suites and
evaluate the held-out fourth suite. The historical field is therefore
`mean_identifiable_suite_auc`, not LOSO. X1R must use explicit names for
identifiable suite AUC, mean, and worst values and must not claim LOSO without a
fitted leave-one-suite-out procedure.

## 1D. Opportunity-gate funnel defect

The runner calls Stage IX `gate_for_row()` and only enters PGD when the gate is
true. It does not materialize an explicit all-start → nonempty → gate-eligible
→ PGD-evaluated → V_phys-consumable → split → metric-identifiable funnel.

The historical X1 root has 121 sequence rows, 118 gate-eligible rows, 3 gate-
false rows, 577 PGD frame results, zero physical interventions, and zero
protected reads. Gate-false rows were retained as empty structural rows but the
disposition is not represented as the required prospective ledger.

## 1E. Victim/policy provenance mismatch — blocking finding

The frozen Stage IX canonical contract and the actual X1 runner load this one
PGD victim for every row:

```text
model_id:   openvla-7b-finetuned-libero-10
model_path: /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10
```

The immutable Stage V and Stage VI-B2 exact snapshot manifests declare the
following clean/snapshot policy paths:

| Stage | Suite | Parents | Manifest-declared clean/snapshot policy | Path match to PGD victim |
|---|---|---:|---|---|
| Stage V | `libero_10` | 10 | `/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10` | true |
| Stage V | `libero_goal` | 10 | `/mnt/sdc/dty_user/openvla_attack/models/libero-goal` | false |
| Stage V | `libero_object` | 10 | `/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object` | false |
| Stage V | `libero_spatial` | 10 | `/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620` | false |
| Stage VI-B2 | `libero_10` | 7 | `/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10` | true |
| Stage VI-B2 | `libero_goal` | 3 | `/mnt/sdc/dty_user/openvla_attack/models/libero-goal` | false |
| Stage VI-B2 | `libero_object` | 2 | `/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object` | false |
| Stage VI-B2 | `libero_spatial` | 4 | `/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620` | false |

The per-parent manifest path is the available source-policy binding. The
trajectory and causal snapshot payloads do not independently repeat a full
checkpoint path, so exact per-snapshot weight SHA is additionally
`NOT_IDENTIFIABLE`; the common `openvla_model_sha256` in the stateless receipts
does not establish equality of the suite-specific checkpoint directories.

The historical X1 sequence rows are:

| Stage | Suite | Starts | Gate-true starts | Gate-false starts |
|---|---|---:|---:|---:|
| Stage V | `libero_goal` | 46 | 44 | 2 |
| Stage V | `libero_spatial` | 36 | 36 | 0 |
| Stage VI-B2 | `libero_spatial` | 39 | 38 | 1 |
| Total | | 121 | 118 | 3 |

Thus the historical PGD evidence contains 121/121 sequence starts from suites
whose declared clean/snapshot policy path differs from the single canonical
PGD victim path. This can change tokenization, action statistics, logits, and
the relationship between PGD realizability and the corresponding V_phys label.
It is sufficient to hold X1R; it cannot be repaired after observing X1
outcomes by changing the victim checkpoint.

## 1F. Chronology around the pre-PGD amendment

The relevant commits are:

| Commit | Time (+08:00) | Change |
|---|---|---|
| `304e509f6263e7988272ff3eb0ed0419f7e3f683` | 18:00:59 | Freeze X1 protocol |
| `893117a580e6078fec17d152c8d08ece22e99653` | 18:02:28 | Audit clean sequence inputs |
| `8d7e9f6b3aa0b2689cdf644218cad52187e3d51e` | 18:06:23 | Accept sealed prospective snapshots in the input audit |
| `d5919552990bc98eb85ee2fbc45715e314d4ef81` | 18:16:09 | Add metric-level short-sequence rule, Stage IX baseline map/split binding, and X1 runner; then execute PGD |

The broad clean-sequence rule existed in the initial protocol. The additional
Q2 short-sequence metric rule and Stage IX baseline/split fields were added in
`d591955` after the clean-input audit artifact and before any X1 PGD result.
This is recorded as an `OUTCOME_BLIND_PRE_PGD_PROTOCOL_AMENDMENT`, not as
outcome-driven tuning. It still requires prospective protocol freezing and
direct tests in X1R.

## 1G. Direct X1 test gap

The repository contains no `tests/stage_x/test_stage_x1_protocol.py` or other
direct X1 test. Existing Stage X tests cover X0 mediator availability only.
The requested X1R CPU tests for A/B/C mapping, L1/L2 Q2 inclusion, Q1/Q3
short-sequence abstention, gate funnel accounting, metric naming, common-parent
baseline improvement, abstain masking, protected counters, and the
no-env-step/no-physical path were therefore not run or passed. Because the
victim provenance hold occurs first, this audit does not add speculative X1R
code or tests.

## Protected boundary and resource state

The historical X1 result root, Stage IX F0 root, and Stage X0 root all report
zero protected reads, zero Eval160 reads, zero physical interventions, and zero
perturbed-action environment steps. `Eval160` and protected evaluation remain
`UNREAD`. No current X1/X1R process was found. Existing GPU compute processes
belong to other users/workloads and were not touched. No new GPU was claimed;
the official environment was used only for read-only JSON/hash inspection.

## Required disposition

1. Keep historical X1 and PR #119 immutable and nonconsumable.
2. Do not launch X1R, do not create a fresh X1R population, and do not run X2.
3. Resolve the source-policy/checkpoint identity before any new PGD authority;
   the resolution must be frozen before PGD and must not alter the historical
   result.
4. After a provenance-consistent prospective source is independently reviewed,
   create a new X1R protocol/runner/tests and rerun only the authorized
   zero-treatment/clean gates before any PGD GPU work.
5. Even a future X1R A result would stop at
   `STAGE_X_X1R_REVIEW_REQUIRED_FOR_X2`; X2 remains unauthorized.

This branch intentionally contains only this audit and its machine-readable
report.
