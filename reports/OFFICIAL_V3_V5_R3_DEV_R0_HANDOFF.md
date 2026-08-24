# Official V3 V5 Student Learnability R3-DEV R0 Handoff

## Scope

R3-0 freezes the Teacher, Student, split, threshold, metric, and authorization
contracts before any new FIT670 V2 episode or label is consumed. This is a
source/contract gate only. The historical Fresh40 proxy is not an input
substitute.

## Source closure

| Item | Binding |
|---|---|
| branch | `codex/v5-student-learnability-r3-20260728` |
| base commit | `6504e94567d9f6bc6394185daf26a60eccf0bb19` |
| base tree | `8efbb1910f14bcf391e3c3069ea7ab100700ecb1` |
| valid R2 execution commit | `f6f619b4fc6b1706aff1cf1967c73e8cc10b8c28` |
| valid R2 execution tree | `1d3bc47e4ce83bc9a445e72688a8f9ad65ace736` |
| protocol freeze commit | `35b48e171f183830058754c229c814d1b2805d4c` |
| protocol freeze tree | `9c57933a91681b1e379718ff453542902fd37004` |
| protocol file | `configs/R3_DEV_PROTOCOL.json` |
| protocol SHA256 | `05699e65c778101bb8eb97dd43ed2562e899ce1c88d0c1745cd5058396accf3d` |
| protocol commit time | `2026-07-28T22:14:21+08:00` |
| base report commit time | `2026-07-28T17:43:40+08:00` |
| R2 execution commit time | `2026-07-28T17:23:23+08:00` |

The protocol was committed before any new input consumption. The later report
commit is metadata only and does not alter the frozen contract.

## Frozen contract summary

- Contact-complete FIT670 V2 canary is the only consumable source.
- Required canary status is `PASS_ENGINEERING_CONSUMABLE_INPUT_GATE`.
- Tranches are fixed as `8 → 40 → 80 → 160 → 670`; identity order is frozen
  before reading labels.
- Five heads use `TRUE/FALSE/UNKNOWN`; UNKNOWN is masked, never FALSE.
- Teacher inputs are causal current/past telemetry only. Outcome, terminal,
  reward, attack, and future fields are forbidden.
- Contact records require complete pairs, `contact_ncon_total` equality,
  `contact_truncated=false`, contact position/normal, named normal constraint
  force, object-gripper binding, and `forward_before_capture=true`.
- Student input is `SC5StreamingFeatureAdapterV2_25D`, FP32, train-only
  normalization, no Teacher fields and no candidate gate in the model input.
- Engineering threshold is fixed at `0.5`; the single development threshold
  selection occurs only on FIT_DEV under the recorded maximum-feasible-threshold
  rule. CAL/CHECK/G10/T2R-D are forbidden.
- Metrics keep head-level masks, tri-valued event labels, independent Teacher
  event/candidate ceilings, UNKNOWN coverage, and causal shadow parity separate.

## Server verification

Official environment:
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

Server: `pm-364c0001`, user `dty_user`.
The original detached R2 worktree remained clean at `f6f619b4...`. A new
detached R3 worktree was created at:
`/mnt/sdc/dty_user/worktrees/codex-v5-student-r3-20260728`

```text
HEAD    = 35b48e171f183830058754c229c814d1b2805d4c
status  = clean
py_compile = PASS
pytest tests/test_r3_dev_protocol.py n5/phase4_fresh40/tests/test_fresh40_v5_contracts.py
          = 19 passed, 0 failed, 0 critical skip
GPU tasks started = 0
```

## Gate state

```text
R3_0_SOURCE_AND_PROTOCOL_CLOSURE = PASS
R3_1_CONTACT_COMPLETE_INPUT      = HOLD_INPUT
R3_2_V23_TEACHER_CANARY           = NOT RUN
R3_3_COVERAGE_LADDER              = NOT RUN
R3_4_STUDENT_LEARNABILITY         = NOT RUN
R3_5_HELDOUT_DEVELOPMENT          = NOT RUN
R3_6_SHADOW                       = NOT RUN
```

R3-1 is blocked until the independent FIT670 V2 canary has the exact required
consumable status and passes recursive source, contact, geometry, identity,
and nonfinite checks. No Fresh40 proxy telemetry is used as fallback.

## Boundary declaration

```text
new Teacher labels       = 0
protected reads          = 0
CAL/CHECK/G10/T2R-D      = not read
Student training         = not run
OpenVLA inference        = not run
rollout                  = not run
attack                   = not run
```
