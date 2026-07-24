# Server Code and Artifact Divergence — 2026-07-02

## GitHub vs Server Divergence Summary

Multiple code versions exist across environments with NO single source of truth:

| Environment | Branch | Commit | Dirty |
|---|---|---|---|
| Local (Windows) | `feature/multisuite-clean2000-detector-prep-v1` | `9f7d711` | YES |
| dty-server | `feature/sc5-abstention-v2-20260622` | `ace18762` | YES |
| vla (RC1A) | `exp/vis-prefix-margin-repair-20260603` | `a8e14ba` | YES |
| vla (REVIEWED) | `exp/l12-production-streaming-adapter-20260615` | `52bdc33` | YES |
| GitHub (origin) | `feature/multisuite-clean2000-detector-prep-v1` | `9f7d711` | Clean |

**Four different branches, three different commits, all dirty except GitHub origin.**

---

## dty-server Modified Files (attack-critical)

These 4 files are modified on the server but the diffs are NOT committed anywhere:

### 1. `src/gripper_attack/attack_adapter.py`
- **Risk**: HIGH — core attack logic
- **Impact**: epsilon, PGD steps, preprocessing, target token selection could differ from claimed protocol
- **Action needed**: Capture diff, compare against frozen protocol

### 2. `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py`
- **Risk**: HIGH — bridge/rollout execution
- **Impact**: Model loading, attack application, artifact writing path
- **Action needed**: Capture diff, verify it matches the UMA/TMA/SHUFFLED execution

### 3. `scripts/v4_run_eval_openvla.py`
- **Risk**: MEDIUM — evaluation script
- **Impact**: Success metric computation, frame counting
- **Action needed**: Capture diff, verify success criteria consistent

### 4. `scripts/stageb/run_sc5_cross_suite_clean.py`
- **Risk**: MEDIUM — cross-suite data collection
- **Impact**: CLEAN2000 data collection protocol
- **Action needed**: Capture diff, verify data schema consistency

---

## dty-server Untracked Files (selected, attack-relevant)

| File | Category | Relevance |
|---|---|---|
| `configs/cross_suite_clean1500_protocol_v1.json` | Config | Cross-suite protocol |
| `configs/cross_suite_object_target_registry_v1.json` | Config | Object target registry |
| `configs/detector/shard/*.json` | Config | Detector shard configs |
| `configs/phase8_primary_object_sites.json` | Config | Attack site selection |
| `docs/gpu/LOTO_*.json` | Docs | LOTO detector protocol freeze |
| `patch_tma.py` | Script | TMA-specific patch |

---

## vla Server (RC1A) Modified Files

| File | Branch |
|---|---|
| `scripts/stageb/audit_vis_runner_attack_method.py` | exp/vis-prefix-margin-repair-20260603 |
| `scripts/stageb/run_s20d_v5_token_pgd_fixed_window_l3_runner.py` | same |
| `scripts/stageb/run_s20d_v5_token_pgd_one_step_smoke.py` | same |
| `scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py` | same |
| `src/gripper_attack/attack_adapter.py` | same — ALSO modified here! |
| `src/gripper_attack/gripper_semantics.py` | same |
| `tables/layer3_registry.csv` | same |

**`attack_adapter.py` is modified on BOTH dty-server AND vla-server, potentially with different changes.**

---

## vla Server (REVIEWED) Modified Files

| File | Branch |
|---|---|
| `scripts/stageb/run_d4_clean_shadow.py` | exp/l12-production-streaming-adapter-20260615 |

---

## Server-Only Scripts (no GitHub equivalent)

These exist on dty-server but have no corresponding committed file in the GitHub repo:

| Path | Category |
|---|---|
| `/mnt/sdc/dty_user/table1_sota_execution_v1/commands/run_sota_worker.py` | Worker |
| `/mnt/sdc/dty_user/table1_sota_execution_v1/commands/validate_sota_condition.py` | Validator |
| `/mnt/sdc/dty_user/table1_sota_execution_v1/commands/build_formal_manifest.py` | Manifest builder |
| `/mnt/sdc/dty_user/table1_sota_execution_v1/commands/chain_stage5_aggregate.sh` | Aggregation |
| `scripts/tmp/auto_launch_uma_shuffled.sh` | Auto-launch (local, untracked) |
| `scripts/tmp/wait_tma_pair.sh` | Wait script (local, untracked) |
| `scripts/tmp/sota_supervisor_v2.sh` | Supervisor (local, untracked) |
| `scripts/tmp/sota_supervisor_v3.sh` | Supervisor v3 (local, untracked) |

---

## Protocol Drift Assessment

### What was CLAIMED:
- epsilon = 6/255
- PGD = 20 steps
- K = 10
- target_token = 31744
- preprocessing = PIL Lanczos
- strict route
- no fallback

### What was ACTUALLY RUN:

**Cannot confirm without auditing the actual running code.** The `attack_adapter.py` is modified on both servers. The bridge script `run_v2_vis_sc5_mlp_bridge.py` is modified on dty-server. These modifications have NOT been reviewed against the frozen protocol.

The UMA/SHUFFLED logs show jobs running with `_CLEAN` suffix — suggesting they ran clean-input baselines rather than actual attack perturbations. This may be intentional (clean baselines for comparison) or a configuration error.

### Specific Concerns:

1. **PR #43 conflict**: 2/255 + upstream_tf_jpeg draft vs 6/255 + PIL Lanczos frozen protocol — which was actually used?
2. **attack_adapter.py divergence**: If the server version differs from frozen protocol, the evidence was generated with different parameters than claimed
3. **Job naming**: `UMA::vis_formal_f09_s1_d3_p1_CLEAN` — the `_CLEAN` suffix and `UMA::` prefix together are ambiguous

---

## Artifact-Command Consistency

For the Object frozen evidence, the exact command lines used to generate each condition are NOT captured in the evidence directories. There is no `command.sh` or `protocol.json` in the artifact leaves.

To reconstruct what was actually run:
1. The `table1_sota_execution_v1/condition_specs/` directory may contain condition definitions
2. The `table1_sota_execution_v1/commands/` directory contains the worker/validator scripts
3. Log files show job names but not full command lines

---

## Recommended Resolution

1. **Capture all server diffs**: `git diff > server_dirty_diff.patch` on dty-server
2. **Hash the running code**: SHA256 of `attack_adapter.py`, bridge script, eval script as actually used
3. **Compare against frozen protocol**: Line-by-line audit of attack parameters
4. **Document any deviations**: If server code differs from protocol, record the actual parameters used
5. **Commit or discard**: Decide whether server changes should be committed or reverted
6. **Single source of truth**: Choose ONE branch/commit as the authoritative code version

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
