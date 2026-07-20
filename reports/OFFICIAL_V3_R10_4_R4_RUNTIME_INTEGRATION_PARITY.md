# Official V3 R10.4-R4 Runtime Integration and LIBERO Replay Parity

## Scope and authorization

This report records the authorized R10.4-R4 work from GitHub Issue #88, comment `5018133445`:

- runtime integration closure without loading OpenVLA 7B;
- LIBERO replay-only numerical parity;
- no detector execution;
- no action mutation;
- no command-OPEN, VIS, RAND, canary, or attack execution.

The replay was performed once on the fixed selection `[0, 99, 199]` from the 200 sorted eligible multi-object FIT identities. Selection was frozen before replay outcomes were available.

## Source and execution binding

| Field | Value |
|---|---|
| Repository | `Leo-6-maker/openvla-gripper-dutycycle-attack` |
| Branch | `codex/r10-4-r4-runtime-parity-20260720` |
| Base | `archive/official-v3-b3-25d-execution-5e27d7c` |
| HEAD | `28532699b948390963759756b6dcbbc48fba1b43` |
| Server | `pm-364c0001` |
| Server user | `dty_user` |
| Python | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python` |
| Python version | `3.10.20` |
| `CUDA_VISIBLE_DEVICES` | `1` |
| `MUJOCO_GL` | `egl` |
| Render device | GPU 1, NVIDIA A800-SXM4-80GB |
| OpenVLA model loaded | `false` |
| Detector executed | `false` |
| Source artifact mutation | `0` |

The server repository checkout was not modified. The replay used an isolated temporary source overlay containing the committed R4 files; the overlay was verified to import `r10_4_runtime.py` and `sc5_streaming_features_v2.py` from that overlay.

## Runtime integration closure

The committed runtime path now:

- delegates 25D streaming features to `SC5StreamingFeatureAdapterV2`;
- uses the frozen official feature order and SHA `3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366`;
- keeps official adapter image preprocessing behind `predict_action(image_np, task_label, capture=True)`;
- performs one-time environment initialization and dummy wait;
- validates sealed inputs before any model loader can be called;
- preserves sealed 7D clean actions for replay;
- has no detector or action-mutating path in the replay script.

The CPU static/runtime-contract audit passed, and the local targeted test suite passed `6/6`. OpenVLA 7B was intentionally not loaded, so this is not evidence that a real OpenVLA passive runtime episode is executable.

## Numerical parity result

The replay compared the online 25D stream against the sealed S1 25D rows at every replay step with absolute tolerance `1e-6`. `action_mutated=false` was recorded for all three episodes. The sealed source did not contain event/reset state fields, so event/reset parity is explicitly `NOT_PRESENT_IN_S1_SOURCE` rather than treated as a pass.

| Identity | Steps | Feature-pass steps | Valid-match steps | Max abs 25D error | Mean abs 25D error | Action sequence SHA256 | Result |
|---|---:|---:|---:|---:|---:|---|---|
| `libero_10/task_00/state_00` | 299 | 299/299 | 299/299 | 0.0000000000 | 0.0000000000 | `467eea9fcc6a75768d74ec9e0df9da43573c408f4d22f14361b7848aa3354951` | PASS |
| `libero_10/task_04/state_19` | 520 | 520/520 | 520/520 | 0.0000000000 | 0.0000000000 | `6fc9de1f166502115edd17a56f9502f33a3abff64052ed205ff47a2a2682fd46` | PASS |
| `libero_10/task_09/state_19` | 520 | 140/520 | 520/520 | 0.0419542789 | 0.0167083104 | `c3d1a67f8de4f94807a3b7eaea739effb52e80cf40039924f6e7afe2b9c3d7bc` | FAIL |

For the failing identity, the first mismatch is at step `140`; the read-only parity records show the sealed and online validity masks still match and the mismatch grows thereafter. This was not rerun, tuned, or replaced.

## Sealed evidence

| Evidence | Value |
|---|---|
| Output root | `/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_R10_4_R4_LIBERO_REPLAY_PARITY_20260720_2853269` |
| Root status | `FAIL` |
| `SHA256SUMS` SHA256 | `0c963e532d0a035ebddeb5a7f9f864ce9ae7aa642b293b404f023e17d3e798b0` |
| `SHA256SUMS.sha256` SHA256 | `c661e67dcf657c5d20a56be210a968bf182c954b958ac005bd6fa2ebeac8d503` |
| Selection status | `FROZEN_BEFORE_PARITY_OUTCOME` |
| Protected split reads | `0` |
| Attack data read | `0` |

The root was created through staging and atomic promotion. It contains the selection manifest, source bindings, protocol, runtime environment, parity rows, per-step parity records, runtime audit, and checksum closure. The root is preserved despite the failed aggregate result.

## Gate decision

```text
R10_4_R4_STATIC_RUNTIME_CONTRACT       = PASS
R10_4_R4_CPU_TARGETED_TESTS             = PASS (6/6)
R10_4_R4_LIBERO_REPLAY_EXECUTION        = COMPLETE (3 selected episodes)
R10_4_R4_NUMERICAL_PARITY               = HOLD (2/3 PASS, 1/3 FAIL)
R10_4_R4_ROOT_SEAL                      = PASS (sealed FAIL report)
OPENVLA_7B_LOAD                        = NOT STARTED
DETECTOR_EXECUTION                     = NOT STARTED
ACTION_MUTATION                        = 0
REAL_PASSIVE_SMOKE                     = NOT STARTED
COMMAND_OPEN / VIS / RAND / ATTACK     = NOT STARTED
```

The failed parity row is a source/runtime closure blocker. No conclusion is drawn about OpenVLA action generation or detector behavior. Any follow-up must first explain the task-09 replay divergence under a separately authorized audit; this R4 execution is stopped here.

