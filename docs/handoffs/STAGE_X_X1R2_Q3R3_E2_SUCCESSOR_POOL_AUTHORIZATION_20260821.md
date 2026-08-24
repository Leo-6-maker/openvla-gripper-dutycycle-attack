# STAGE X X1R2 Q3R3 E2 successor engineering-pool authorization — 2026-08-21

## Decision

`PASS_E2_SUCCESSOR_ENGINEERING_POOL_AUTHORIZED_PRE_GPU`

The user explicitly authorized autonomous engineering continuation after the
previous E2 hold. This authorizes a fresh, outcome-blind, permanently excluded
engineering pool only. It does not authorize a method change, scientific
population change, protected access, or an efficacy claim.

## Binding

- Source commit/tree: `5bf29dd4492c578d22a551ba99705c90930472ad` /
  `237d941d4f646668f0c3d2dab9c99a9e3064d784`.
- Pool: `reports/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_ENGINEERING_POOL_V1.json`.
- Pool raw SHA-256: `b7774b7f6bcfcb6e46056b739c86f764c5bc7dfb4304ed0eb2b31898c690acda`.
- Protocol: `configs/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_PROTOCOL_V1.json`.
- Protocol raw SHA-256: `edb10f8010ca7579ca55bdab56117120989d16789e1bbfa88594eb669266341c`.
- Predecessor hold: `HOLD_Q3R3_E2_FEASIBILITY_EVIDENCE_INSUFFICIENT_FROZEN_POOL_EXHAUSTED`.

## Authorized pool

| suite | fixtures | count |
| --- | --- | ---: |
| `libero_10` | `Q3R2-LIBERO_10-08..10` | 3 |
| `libero_goal` | `Q3R2-LIBERO_GOAL-09..11` | 3 |
| `libero_object` | `E2S-LIBERO_OBJECT-01..03` | 3 |
| `libero_spatial` | `Q3R2-LIBERO_SPATIAL-07..09` | 3 |
| **total** |  | **12** |

All 12 identities are permanently excluded from scientific populations and
cannot become X1R2 efficacy evidence. Selection uses only the frozen identity
ledger and frozen pool order/hash rank; clean, emit, attack, physical, and
protected outcomes are forbidden inputs.

## Frozen execution boundary

- A800 environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
- Free memory must be strictly greater than 20 GiB immediately before launch.
- At most one project worker per physical GPU and at most eight workers total;
  foreign GPU owners remain untouched.
- E2 is prospective feasibility only: reference clean, branch replay, then TRUE
  selector audit; no attacked `env.step` and no physical outcome read.
- Method remains epsilon `0.03`, step `0.006`, PGD-5, no random start,
  `STRICT_CANDIDATE_AUDIT_V1`, exact arm `[0:6]`, native `NATIVE_OPEN`, and no
  fallback/actuator overwrite/decode-reencode.

## Stop conditions

Any incomplete six-row candidate audit, diagnostics mismatch, source mismatch,
foreign GPU ownership, protected access, or method/estimand change is a HOLD.
Case C or any required scientific interpretation returns to Owner/PI review.

Next legal gate: `STAGE_X_X1R2_Q3R3_E2_PREATTACK_BRANCH_QUALIFICATION`.
