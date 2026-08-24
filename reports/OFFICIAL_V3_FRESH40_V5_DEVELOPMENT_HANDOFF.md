# Official V3 Fresh40 V5 Development Handoff

## Decision

`FRESH40_V5_DEVELOPMENT_CANARY = HOLD`.

This is a FIT-only development result. It is not a formal Student result,
not a vulnerability result, and not an attack result. Fresh670 extension is
not authorized by this handoff.

## Source and boundaries

- code branch: `codex/fresh40-v5-canary-20260728`
- code commit: `7d0be9299089392ab787572423413565b7a57453`
- source: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5f/run_B`
- source `MANIFEST.json` SHA256:
  `d6df050d30ca4c771f9864f65a16bcb26fc0a961e1979f4079b57df7a637da71`
- source `SHA256SUMS` SHA256:
  `8b88f5c207c5f4c50160f64c870ac4c39262603a1974d3816596f9581314a6cd`
- source closure: 40 identities, 4 suites, 10 tasks/suite, 11,710 steps
- source flags: `teacher_labels_generated=false`, `attack_enabled=false`
- exact `mujoco_contact_pairs`: unavailable at all 11,710 steps
- newly generated OpenVLA inference: no
- protected/CAL/CHECK/G10/T2R-D reads: no

The Teacher therefore uses direct recorded entity poses and a clearly named
global-contact/distance proxy. The labels are `DEVELOPMENT_PROXY_NONCONSUMABLE`.

## Sealed roots

| stage | root | SHA256SUMS.sha256 |
|---|---|---|
| E0 audit | `fresh40_v5_e0_audit_d57afd2_20260728_20260728T061818Z` | `5dc5de39eff44f3225878d3579c63f4dd1f7bdff74a796698af76a86026ceca2` |
| E1 proxy Teacher | `fresh40_v5_teacher_proxy_3494582_20260728T062620Z` | `63369fbb732b720f1f0f8edd3e04e622cbdb90d3914ec4bb59d7c0830ef54274` |
| E2 causal dataset | `fresh40_v5_dataset_3494582_20260728T062642Z` | `b4694aed2e848cc4c55152c63dd01c20c78faf770b5f2cf767c1ae0d5b594b9a` |
| E3 full-five checkpoint | `fresh40_v5_student_full5_7d0be92_20260728T063510Z` | `dfe41ba9419e509c945fe21fb2bd574342c852c2f10106a23d309b19bd922b86` |
| E4 full-five shadow | `fresh40_v5_shadow_full5_7d0be92_20260728T063729Z` | `48a930e68b5ab32510ebb944962ef425f330d9fd6a728486238c8b188a98b8c2` |
| E3 critical-only checkpoint | `fresh40_v5_student_critical_7d0be92_20260728T063921Z` | `3fcd01fe9dfaf5307b1acb100f7576464ec36e9f75504fe2b043269de8d93827` |
| E4 critical-only shadow | `fresh40_v5_shadow_critical_7d0be92_20260728T064053Z` | `2c2eeba1bd6fc5b054712b713df7259068f50a49f39be65bf639cea35595e975` |
| E3 three-head checkpoint | `fresh40_v5_student_three_7d0be92_20260728T063921Z` | `f9062a7d35437cc0c08e1c49837f96370bb600df56b17d34ba4998af0f0ec8af` |
| E4 three-head shadow | `fresh40_v5_shadow_three_7d0be92_20260728T064053Z` | `a4fe7d4612b78f873c70296a0a86b8e2d6aa15375a70b1fac73c0a038ce0bc9d` |

The two early failed staging attempts were retained on the server and were
not reused: one exposed EEF list/vector handling; the other exposed nested
identity directory creation.

## Teacher and dataset evidence

E1 head step counts:

| head | TRUE | FALSE | UNKNOWN |
|---|---:|---:|---:|
| physical criticality | 641 | 354 | 10,715 |
| K10 feasible | 10,440 | 666 | 604 |
| safe release | 356 | 10,750 | 604 |
| instability | 16 | 11,056 | 638 |
| gripper closing state | 1,081 | 10,330 | 299 |

E2 uses exact `SC5StreamingFeatureAdapterV2` 25D causal features, 32 train
and 8 dev identities, deterministic two-dev-per-suite selection, and a
train-only normalizer. Student records contain no Teacher geometry or outcome
fields.

## Student/shadow results

All variants trained for 12 epochs with seed `20260717`, FP32, hidden 64,
RF32/RF128, AdamW, and no validation early stopping.

| variant | shadow emits | critical recall | UNKNOWN emits | action mutation |
|---|---:|---:|---:|---|
| critical-only | 0 | 0/106 | 0 | false |
| three-head | 18 | 1/106 | 17 | false |
| full-five | 19 | 1/106 | 18 | false |

The one-shot event latch ran without action mutation. The scientific canary
does not pass: recall is effectively zero and most positive-looking emits
land on UNKNOWN proxy labels. This is consistent with an observability/label
quality limitation, not evidence that attack timing is effective.

## Final boundary

```text
FRESH40_SOURCE_AUDIT       = PASS_WITH_CONTACT_LIMITATION
FRESH40_TEACHER_ENGINEERING = PASS_PROXY_ONLY
FRESH40_DATASET             = PASS_DEVELOPMENT
FRESH40_STUDENT_TRAINING    = PASS_DEVELOPMENT_ONLY
FRESH40_SHADOW_RUNTIME      = PASS_ACTION_IMMUTATION_ZERO
FRESH40_DEVELOPMENT_CANARY  = HOLD
FRESH670                    = NOT RUN
PROTECTED_READS             = 0
OPENVLA_NEW_INFERENCE       = NOT RUN
ROLLOUT                      = NOT RUN
ATTACK                       = NOT STARTED
```
