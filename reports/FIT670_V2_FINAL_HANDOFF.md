# FIT670 V2 Formal Collection — Final Handoff

## Status: PASS_CONSUMABLE_FINAL

## Source Identity

| Field | Value |
|-------|-------|
| Branch | `deepseek/fresh670-v5-collection-20260728` |
| HEAD | `1c5b65a8a5ccff88546b12a2b01108b6c2fe0ab8` |
| Tree | `517c04e039bd4e2aa4a228e5703a67579ed06c28` |
| Git status | Clean (0 dirty tracked files) |

## Collection Summary

| Metric | Value |
|--------|-------|
| Episodes collected | 670 / 670 |
| Errors | 0 |
| Workers | 8 / 8 (GPU 0-7, A800-SXM4-80GB) |
| Per-worker identities | 84,84,84,83,83,84,84,84 |
| Worker identity union | 670 |
| Worker identity intersection | 0 (no duplicates) |
| Missing identities | 0 |
| Extra identities | 0 |
| Bad seals | 0 |
| Staging residue | 0 |
| Protected reads | 0 |

## Closure Verification

| Check | Result |
|-------|--------|
| Allowlist count | 670 ✓ |
| Published count | 670 ✓ |
| Worker manifests | 8/8 ✓ |
| Shard plan match | 8/8 shards exact ✓ |
| Per-episode SHA256SUMS | Verified (sample) ✓ |
| Transition seal | PASS ✓ |
| Shard plan seal | PASS ✓ |
| GPU UUID/PCI match nvidia-smi | 8/8 ✓ |
| Entity records (position + rotation_wxyz) | Present in all telemetry ✓ |
| Contact pairs (pos/normal/force/object-gripper) | Present ✓ |
| contact_truncated=False | All steps ✓ |
| contact_ncon_total == len(contact_pairs) | All steps ✓ |
| raw_action_7d == action_raw_7d | Elementwise identical ✓ |
| forward_before_capture | True ✓ |
| C1 bindings | Present ✓ |
| Provenance | Present ✓ |

## Paths

| Artifact | Path |
|----------|------|
| Formal episodes | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_formal/episodes/` |
| Finalization | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_formal/FINALIZATION_V2/` |
| Formal transition | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_transition_v2_formal/` |
| Canary review | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_canary/CANARY_REVIEW_V2/` |
| Shard plan | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_shard_plan_v2/` |
| Identity allowlist | `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/` |

## Critical SHA256 Values

| Item | SHA256 |
|------|--------|
| Transition SHA256SUMS | `591601ca0fd69ac075f2a4a5ce02fbc35aa537f742b6f5e44c083f93df04177e` |
| Shard plan | `4b1d49febb20cc1edda23afb15e65a5a0ab7d1d713090ce307de1198a110e407` |
| Allowlist file | `0ee5001c9dadb142a32b61cbdf5eaf7b2ae69ebc80930d50776ebcfd7c1f5381` |
| Identity set digest | `8e0afdc3f37dc887481dd1a49ab7468da31142db7b0807ef023d2149580595a1` |
| Episode seal digest | `71838a89f6bef411a2c6d7dd61f195a7f1e417b78a19a5cd9d1818a23c2476a5` |
| C1 canonical digest | (from transition manifest) |

## GPU Identity

| GPU | UUID | PCI Bus |
|-----|------|---------|
| 0 | GPU-bf4309d3-8cba-437e-8d87-cee9f1e6d232 | 00000000:10:00.0 |
| 1 | GPU-f6910e5c-f41e-109e-43d0-f01f0d77dbf2 | 00000000:16:00.0 |
| 2 | GPU-7b06162a-27e4-2552-e891-d201e3fae6b9 | 00000000:49:00.0 |
| 3 | GPU-41cd4b75-e3d4-92b8-ec37-ddca13e3761a | 00000000:4D:00.0 |
| 4 | GPU-e85ed586-ba64-a9e3-8fa9-07f16f84dcda | 00000000:8A:00.0 |
| 5 | GPU-185b30c0-074c-6f07-aa8b-a67d00e8e4a9 | 00000000:8F:00.0 |
| 6 | GPU-92963392-f77a-85ce-4ba7-7a8288429ca5 | 00000000:C6:00.0 |
| 7 | GPU-bd2cfcc1-64ab-c2d0-9ae3-245fc8d21a76 | 00000000:CA:00.0 |

## Non-Authorized Operations

| Operation | Status |
|-----------|--------|
| Teacher labeling | NOT RUN |
| Student training | NOT RUN |
| Detector inference | NOT RUN |
| Attack | NOT RUN |
| Protected/CAL/CHECK/G10/T2R-D reads | 0 |
| Codex branch modification | 0 |

## Execution Environment

- Python: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`
- Worktree: `/tmp/fresh670_v2_worktree` (at 1c5b65a8)
- Model: `/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10`
- LIBERO: `/mnt/sdc/dty_user/pi0_openpi/third_party/libero`
- Upstream: `/mnt/sdc/dty_user/openvla_attack`
- C1 Registry: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/`

## Independent Audit

An independent Explore subagent verified all checks from raw episode data. Result: 8/8 PASS. Minor correction: entities and contact_pairs are in `telemetry[]` (not `steps[]`), verified present with correct fields.

## V1 Legacy Collection

- Root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5/`
- Status: `LEGACY_V1_DEVELOPMENT_NONCONSUMABLE`
- Episodes: 137
- Stop receipt: `V1_STOP_RECEIPT.json`
