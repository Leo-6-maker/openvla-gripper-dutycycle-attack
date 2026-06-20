# A800 Runtime Profiles — OpenVLA Gripper Duty-Cycle Attack

**Version:** v1.0-draft
**Date:** 2026-06-20
**Status:** Definitions only (no runtime validation yet)

---

## Profile A1: legacy_black_bowl_attack_runtime

**Purpose:** Reconstruct the runtime used for historical Black Bowl attack experiments on 2080Ti.

**Status:** DEFINED — PENDING EVIDENCE RECONSTRUCTION

| Field | Value |
|---|---|
| Source commit | UNKNOWN_PENDING_EVIDENCE |
| Runner path | UNKNOWN_PENDING_EVIDENCE |
| Launch command | UNKNOWN_PENDING_EVIDENCE |
| Model dtype | UNKNOWN_PENDING_EVIDENCE |
| Device map | UNKNOWN_PENDING_EVIDENCE |
| Per-GPU cap | UNKNOWN_PENDING_EVIDENCE |
| Preprocess backend | UNKNOWN_PENDING_EVIDENCE |
| EOS behavior | UNKNOWN_PENDING_EVIDENCE |
| Checkpoint | UNKNOWN_PENDING_EVIDENCE |
| Unnorm key | UNKNOWN_PENDING_EVIDENCE |
| Max steps | UNKNOWN_PENDING_EVIDENCE |
| Gripper semantics | UNKNOWN_PENDING_EVIDENCE |

## Profile A2: legacy_cross_suite_clean_runtime

**Purpose:** Reconstruct the runtime used for cross-suite CLEAN300 evaluation on 2080Ti (Codex branch `exp/cross-suite-clean-train300-s10-19-20260620`).

**Status:** DEFINED — PENDING EVIDENCE RECONSTRUCTION

Known partial evidence:
- Active Codex eval running on 2080Ti GPU 4,5 (2026-06-20)
- Source commit: `63793972743f667c6a6bcc12e9700f322f261147` (observed from process command)
- Repo: `train300_collector_freeze_141657f`
- Runner: `scripts/stageb/run_sc5_cross_suite_clean.py`
- Env: `/data/aviary/envs/openvla_official_libero_20260525`
- Model: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal`
- Unnorm key: `libero_goal`
- Render GPU: 5
- `--save_video` flag present

| Field | Value |
|---|---|
| Source commit | `63793972743f667c6a6bcc12e9700f322f261147` |
| Runner path | `scripts/stageb/run_sc5_cross_suite_clean.py` |
| Model dtype | UNKNOWN_PENDING_EVIDENCE |
| Device map | UNKNOWN_PENDING_EVIDENCE |
| Per-GPU cap | UNKNOWN_PENDING_EVIDENCE |
| Preprocess backend | UNKNOWN_PENDING_EVIDENCE |
| EOS behavior | UNKNOWN_PENDING_EVIDENCE |
| Max steps | UNKNOWN_PENDING_EVIDENCE |
| Gripper semantics | UNKNOWN_PENDING_EVIDENCE |

## Profile B: openvla_official_a800

**Purpose:** Clean OpenVLA reference following upstream README exactly.

**Status:** DEFINED — not yet validated (no model loaded, no rollout run)

### B.1 Prompt Contract

| Field | Value |
|---|---|
| **Value** | `"In: What action should the robot take to {task.lower()}?\nOut:"` |
| **Source file** | `prismatic/vla/constants.py` (ACTION_PROPMPT_TEMPLATE) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED — requires model load to verify tokenization |
| **Note** | The `In:` prefix and `\nOut:` suffix are required by official OpenVLA. Not equivalent to `"What action should the robot take to ..."` without prefix/suffix. |

### B.2 EOS Contract

| Field | Value |
|---|---|
| **Value** | Token ID 29871 (`</s>`) appended after prompt |
| **Source file** | `prismatic/models/vlas/openvla.py` (generate_response) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED — requires full tokenization trace |
| **Note** | EOS is appended during tokenization, NOT as a separate generation step. Tokenizer pads to max_length after EOS. |

### B.3 Image Rotation Contract

| Field | Value |
|---|---|
| **Value** | 180° rotation applied to agentview image |
| **Source file** | `prismatic/vla/datasets/rlds/obs_transforms.py` |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED — requires raw image comparison |
| **Note** | Official LIBERO benchmark applies 180° rotation to agentview. Project repo: `99a51fb` ("Align v4 OpenVLA clean preprocessing with corrected official LIBERO eval"). |

### B.4 Resize Contract

| Field | Value |
|---|---|
| **Value** | PIL Image.LANCZOS resize to 224×224 |
| **Source file** | `prismatic/models/vlas/openvla.py` (image_transform) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED |
| **Note** | Official code uses `transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS)`. No bicubic, no bilinear. |

### B.5 Center Crop Contract

| Field | Value |
|---|---|
| **Value** | Center crop applied before resize; final output 224×224 |
| **Source file** | `prismatic/models/vlas/openvla.py` (image_transform) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED |
| **Note** | Official flow: raw → 180° rotate → center crop → LANCZOS 224×224. No JPEG encode/decode round-trip. |

### B.6 Dtype Contract

| Field | Value |
|---|---|
| **Value** | `torch.bfloat16` |
| **Source file** | `prismatic/models/vlas/openvla.py` (load) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED — import-only verified |
| **Note** | Official loads in BF16. Eager attention used for correctness baseline. flash_attention_2 deferred to single-factor experiment. |

### B.7 Attention Contract

| Field | Value |
|---|---|
| **Value (baseline)** | `eager` |
| **Value (optimized)** | `flash_attention_2` (separate experiment, M1B) |
| **Source file** | `prismatic/models/vlas/openvla.py` (load) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED (baseline eager) / NOT INSTALLED (flash_attn 2.5.5) |

### B.8 Action Decode Contract

| Field | Value |
|---|---|
| **Value** | Generated token IDs → vocab mapping → bin centers → q01/q99 mask → unnormalize |
| **Source file** | `prismatic/vla/action_tokenizer.py` (ActionTokenizer.decode) |
| **Source commit** | `c8f03f48af692657d3060c19588038c7220e9af9` |
| **Validation status** | UNVALIDATED — requires model load |
| **Note** | Action dim = 7 (6 EEF + 1 gripper). Bin centers derived from action stats. q01/q99 mask clips to training data range. |

### B.9 Unnorm Key Contract

| Field | Value |
|---|---|
| **Value** | Suite-specific from model metadata (`dataset_statistics`) |
| **Source file** | Model `preprocessor_config.json` / `dataset_statistics.json` |
| **Options** | `libero_object`, `libero_spatial`, `libero_goal`, `libero_10` |
| **Validation status** | UNVALIDATED — requires model load and metadata inspection |
| **Note** | Must be confirmed programmatically from loaded model, not hardcoded. |

### B.10 Gripper Normalize/Invert Contract

| Field | Value |
|---|---|
| **Normalize** | Raw `[0, 1]` → `[-1, 1]`: `gripper_action = (raw_gripper * 2) - 1` |
| **Binarize** | `gripper_action = np.sign(gripper_action)` |
| **Invert** | `gripper_action = -gripper_action` (project convention; OPEN/CLOSE sign) |
| **Source file** | Project: `src/gripper_attack/env_factory.py` (unnormalize_action) |
| **Source commit** | `141657f` (main HEAD) |
| **Validation status** | UNVALIDATED |
| **Note** | The sign inversion is a PROJECT convention to match LIBERO environment OPEN/CLOSE. Official OpenVLA may not include this step. Must verify against official action output. |

### B.11 Wait Steps Contract

| Field | Value |
|---|---|
| **Value** | 6 dummy wait steps before policy execution |
| **Source file** | LIBERO benchmark (`libero/libero/envs/bddl_utils.py`) |
| **Source commit** | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| **Validation status** | UNVALIDATED |

### B.12 Max Policy Steps Contract

| Field | Value |
|---|---|
| **Value** | Suite-dependent (typically 400–500) |
| **Source file** | LIBERO task definitions |
| **Validation status** | UNVALIDATED — must be confirmed per-suite |
| **Note** | Must match old server's exact per-task max_step values. Not to be assumed. |

### B.13 Success Predicate Contract

| Field | Value |
|---|---|
| **Value** | LIBERO `env.reward` check; `success = reward >= threshold` |
| **Source file** | `libero/libero/envs/env.py` |
| **Source commit** | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| **Validation status** | UNVALIDATED |

### Upstream References
- OpenVLA: `c8f03f48af692657d3060c19588038c7220e9af9`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`

### Suites
| Suite | Checkpoint | Unnorm Key |
|---|---|---|
| LIBERO-Object | `openvla/openvla-7b-finetuned-libero-object` | `libero_object` |
| LIBERO-Spatial | `openvla/openvla-7b-finetuned-libero-spatial` | `libero_spatial` |
| LIBERO-Goal | `openvla/openvla-7b-finetuned-libero-goal` | `libero_goal` |
| LIBERO-10 | `openvla/openvla-7b-finetuned-libero-10` | `libero_10` |

All unnorm keys must be confirmed from model metadata at load time.

## Profile C: project_a800

**Purpose:** Profile B + project telemetry + detector hook. Must preserve CLEAN action path.

**Status:** NOT YET DEFINED — blocked by M2/M3

| Contract | Detail |
|---|---|
| CLEAN action path | Must be byte-identical to Profile B |
| Allowed additions | Telemetry, SC5 streaming, artifact provenance, detector runtime, attack adapter interface |
| Forbidden changes | Prompt, preprocessing, EOS, generation, action decode, unnorm stats, gripper semantics, environment |

---

## Parity Sequence (Phase B)

Single-factor changes on A800, applied sequentially:

| Step | Change | Profile |
|---|---|---|
| P0 | Legacy 2080Ti replay (multi-GPU, eager, 10G cap, legacy preproc) | A |
| P1 | P0 + official PIL preprocessing | — |
| P2 | P1 + explicit EOS 29871 | — |
| P3 | P2 + single-GPU BF16 (no memory cap) | — |
| P4 | P3 + official step semantics | — |
| P5 | P4 + flash_attention_2 | B (optimized) |

Each step reuses the same 20 frozen frames (5 per suite). Output a diff matrix per step.

---

## GPU Lease Protocol

1. Before ANY GPU use: `nvidia-smi` + `nvidia-smi pmon -c 1`
2. Target: GPU 4 or 6 (historically idle)
3. Create lease: `/mnt/sdc/dty_user/openvla_attack/gpu_locks/gpu{N}.lock`
4. Lease fields: hostname, GPU index, UUID, owner, PID, start time, expected end time, git commit, runtime profile, output root, command SHA
5. After completion: remove lease file

**Never assume GPU is free based on prior snapshots.**
