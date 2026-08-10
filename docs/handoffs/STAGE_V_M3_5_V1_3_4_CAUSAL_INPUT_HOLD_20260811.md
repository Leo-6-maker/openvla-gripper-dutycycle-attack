# Stage V M3.5 V1.3.4 causal-input HOLD — 2026-08-11

## Decision

```text
M3_5_LABEL_VALIDATION          = HOLD_MEASUREMENT_CONTRACT
V1.3.4 FORMAL DIAGNOSTIC      = SEALED_STRUCTURAL_FAILURE_CAUSAL_INPUT_REPLAY
V7_FRESH_QUALIFICATION        = BLOCKED / NOT_STARTED
M4_FRESH_COUNTERFACTUAL_MAP   = BLOCKED / NOT_STARTED
TEACHER / STUDENT / TIMING    = BLOCKED / NOT_STARTED
PROTECTED EVAL160             = NO_READ HARD STOP
```

This is a valid scientific gate failure, not a retryable queue failure. The
frozen causal-state contract requires both exact simulator-state restoration
and exact policy-input binding. At probe step 53, a CONTROL branch completed,
then a fresh T3 branch restored the exact simulator-state hash but failed the
policy-RGB hash check before its treatment action passed the causal-input gate.
Deleting or weakening that check would violate the Goal Mode master plan.

The formal root is immutable and non-consumable:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M3_5_DIAGNOSTIC_V1_3_4_20260810T201623Z`

- Structural failure receipt SHA256:
  `13d6c5de2d138917ed6b5016b7d30970433193d52d9d17e3b6ba330e96fc556a`.
- Root `SHA256SUMS` SHA256:
  `b5cb2b08531eb49ec44ad231b8b9388b3f486838de892606317b1116eee02c8e`.
- Root and all child directories are mode `0555`; files are mode `0444`.
- All project workers were reaped. GPU3 retained only the foreign `huanzze`
  PID `1125866`, which was not touched.

## Frozen source and pre-runtime gates

Runtime source:

- Commit: `d104713027a82eeb858ba9036200d7ab010959cc`.
- Tree: `3f22ea412975f294b59bc569ef9fb896eff8d410`.
- Worktree:
  `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m35-v1_3_4-protocol-d1047130`.
- Protocol:
  `configs/STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_3_4.json`.
- Protocol SHA256:
  `2e3364e5e86894ff7b5b9300dbc03875d1e9b5acb10f923ec798b4d06babba41`.

The implementation fix that added deterministic PIL byte descriptors is
commit `623bf99b6ff1eff37e5ca38b65ec5f6c2b03696a`, tree
`fcb40bd6d5026bcdb7915b1ccec1869b00d2adf4`. It fixed the earlier
`UNSUPPORTED_TRACE_VALUE:Image` structural failure without adding a PIL
dependency to the shared core.

All prospective V1.3.4 gates passed before the formal launch:

| Gate | Result | Receipt SHA256 | Root `SHA256SUMS` SHA256 |
|---|---|---|---|
| Exact A800 regression | `209/209 PASS` | `f35213028f43e69b10f4d5a10adc6089d0597265fa1c6df35c6623d03a3054d8` | `76adca5edb2db607f2c1e147c94527c92f6b79f8206f5a536b1c740e5ebdb050` |
| Static independent audit | `98/98 PASS` | `39f8ebf093463cf42c80e89f6796a3534c41d30cec802cea9092f278c3a5f97c` | `f64403d43266272f8a493225f1bb5b0e14b61475092f8b162c83c363ac44508f` |
| Runtime authorization | `PASS` | `2b5aafde5af4571add27b4e62e81b04f14e619240ea85b43acb32ace770d7055` | `04908a4acc757e3840f13105141356cfee889e42b78d95f4634f1dba62ccb6e7` |
| EGL/PIL bootstrap smoke | `PASS`, reset/step/model all zero | `a3488ec9b5799661eb092848d8fe499e67861d010d12ee1eca86ede9751fcba4` | `f8707ef3dff2360c131d63318a9bdb33c95c78c0f335ad6e9b8042b22b6121e1` |

The smoke's separate runtime-binding receipt SHA256 is
`7c4e91f1e3e514ca8b4e5c3df4250f8d7a661b61f698cb8f58d9043b4dea3210`.

Two non-consumable smoke-preparation roots were also preserved rather than
silently reused:

- `.../STAGE_V_M3_5_EGL_BOOTSTRAP_SMOKE_V1_3_4_20260810T201059Z` contains
  only the preparation script and a sealed checksum; no smoke ran.
- `.../STAGE_V_M3_5_EGL_BOOTSTRAP_SMOKE_V1_3_4_20260810T201202Z` failed on
  smoke-only `repo/src` import ordering with reset/step/model all zero; receipt
  SHA256 `a0542d7b37e44ab4dcabdcc767b023666fe9a44e1e68d0424b71015cefe56000`,
  root checksum SHA256 `eb3c5c0be19ad98cca2cffdc5c6a8753bbe21666dcd8f42d76d005527a2ced8d`.

## Formal terminal inventory

The dispatcher used GPUs `[0,1,2,4,5,6,7]`, excluded GPU3, and enforced one
worker per GPU with `max_attempts=1`. At the sealed terminal snapshot:

| Artifact/state | Count |
|---|---:|
| Runtime-binding receipts | 7 |
| Clean trajectories | 4 |
| Corridor/probe plans | 4 |
| Registered completed branches from progress | 2 |
| Parent results | 0 |
| Persisted physical-branch files | 0 |
| Persisted treatment-observation files | 0 |
| Consumable label rows | 0 |

The two registered branches were CONTROL branches. One T3 branch failed its
causal-input check before its treatment action passed the gate. Another T3
branch was in progress when the dispatcher performed the global reap, so the
receipt conservatively records that unpersisted treatment steps may have
occurred. It does not claim zero intervention steps. No partial artifact is
eligible for labels or downstream training.

## Why downstream stops

The authoritative M1 result was `HETEROGENEOUS_MULTI_GPU_DIVERGENCE`: rendered
policy inputs diverged in both same-mode and cross-GPU repeats, while actions
remained stable. M2 therefore established action/physical execution
equivalence, not bitwise policy-input identity. M3.5 has a stronger causal
requirement: each branch must bind the same exact policy input as well as the
same simulator state. V1.3.4 demonstrates that the current fresh-environment
replay design does not meet that requirement.

The legal next action is design-only review of a new prospective replay
contract that can preserve exact policy-input identity without relaxing the
gate. Any future runtime requires a new source commit/tree, protocol SHA,
exact regression, static audit, authorization, smoke, and fresh root.

Forbidden actions are: resume or reuse V1.3.4, increase attempts, weaken the
RGB/input equality gate, consume partial branches as labels, start V7/M4 or
any Teacher/Student/timing/VIS stage, read Eval160, or touch the GPU3 foreign
process.

## GitHub state

PR #111 remains open, draft, unmerged, and currently has merge state `CLEAN`.
`source-registry`, `detector-v5-cpu`, and `stageb-cpu` are all green. This
handoff is a docs-only descendant of the frozen runtime commit; it must not be
treated as a new runtime source freeze.
