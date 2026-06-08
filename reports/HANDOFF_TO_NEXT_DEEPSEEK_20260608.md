# HANDOFF — Next DeepSeek Session

**Date**: 2026-06-08
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Remote SHA**: `da2b421`

## What we're doing

Inference-time VIS PGD attack on OpenVLA-7B in LIBERO Object. Goal: predict vulnerable windows from clean rollout features, apply low-budget perturbation, induce gripper OPEN, create physical response.

## The one thing that almost killed us

Gripper open/close semantics were INVERTED for months. The old code said `env > 0 = OPEN` but physically `env = -1` opens the gripper. All pre-RC1a labels are quarantined. The frozen standard is:

```
raw_gripper > 0.5  → env = -1 → OPEN
raw_gripper < 0.5  → env = +1 → CLOSE
raw_gripper == 0.5 → boundary (excluded)
```

## Trusted code (everything imports from spec)

| File | Role |
|------|------|
| `src/gripper_attack/openvla_libero_exec_spec.py` | Single source of truth |
| `src/gripper_attack/attack_adapter.py` | VIS objective (open_token_ids now correct) |
| `scripts/run_stageb_vis_labeling.py` | v1.1 runner, 53-column trace |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | Validator, hard-fail gates |
| `scripts/stageb/build_pair_labels_v1_1.py` | Pair label builder (key includes seed) |
| `scripts/diagnostics/run_detector_v0_fixed.py` | Detector readout, 3 tiers |

All traces must have: `trace_version=corrected_stageb_v1_1`, `source_snapshot_id=f9840cb1`.

## Data we have (all RC1a provenance, all validator PASS)

| Phase | Pairs | cmd_sus | phys | rand | Key finding |
|-------|-------|---------|------|------|-------------|
| Bronze 48 | 45 | 11 | 15 | 8 | Corrected VIS baseline |
| Silver P1A | 37 (from 23) | stable=9 | stable=4 | stable=6 | Stability confirmed |
| P1b | 18 | 2 | 3 | 4 | Negatives filled |
| Rescue | 18 rows, 12 parents | 11 agg | 3 agg | 3 agg | Multi-repeat aggregated |

Server roots:
- `/data/liuyu/outputs/stageb_v1_1_bronze_batch_rc1a_20260607/`
- `/data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608/`
- `/data/liuyu/outputs/stageb_v1_1_silver_p1b_rc1a_20260608/`
- `/data/liuyu/outputs/stageb_v1_1_random_confounded_rescue_rc1a_20260608/`
- `/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/`
- `/data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/`

Execution copy (not dirty reviewed worktree):
- `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/`

## Detector v0: what it can and cannot do

NOT a final detector. Exploratory multi-head selector.

**Targets**: `cmd_specific` (cmd AND NOT random-sensitive), `vis_specific_physical`, `random_sensitive` (abstain).

| Target | Best Model | P@5 | AUROC | Verdict |
|--------|-----------|-----|-------|---------|
| cmd_specific | TaskOnly | 0.60 | 0.68 | **Task-biased** (butter dominates) |
| vis_specific_phys | CleanNoTaskNoTiming | 0.40 | 0.46 | Promising but underpowered |
| random_sensitive | CleanNoTaskNoTiming | 0.60 | **0.77** | Real signal, needs abstain head |

Clean features alone do NOT beat TaskOnly for cmd. Physical bridge has signal but needs more positives. Random-sensitive is the strongest and most consistent signal.

## Quarantined (NEVER use)

1. Old 44-row patched rerun (VIS objective was inverted)
2. Old overnight labels (wrong open convention)
3. Pre-v1.1 traces
4. Active Probe v0b/v1
5. ProprioNoStep as detector

## What to do next (do NOT start without reviewing)

1. **Targeted expansion, not blind scaling**: need non-butter cmd positives, more physical positives, hard negatives, same-task contrasts. Target: ≥25 cmd_specific, ≥15 phys, ≥25 rand, ≥40 hard_neg across tasks.

2. **Visual sidecar pilot**: extract frozen CLIP/DINOv2 embeddings from clean frames. Compare ProprioOnly vs VisualOnly vs Proprio+Visual on same grouped splits. See if visual features reduce task bias.

3. **Multi-head formulation**: `attack_score = p(phys_bridge) - λ·p(random_sensitive)`. Attack only when score high.

## Server access

```
ssh vla   (jump host: scene@10.60.133.3 → liuyu@10.60.133.4)
```

GPU pairs: `1,0` / `2,6` / `4,5`. GPU 3,7 blacklisted.

Conda env: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`

## Quick start commands

```bash
# Verify spec
cd /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
PYTHONPATH=src python -c "from gripper_attack.openvla_libero_exec_spec import *; print(env_gripper_is_open(-1.0))"  # True

# Run tests
PYTHONPATH=src pytest tests/stageb/ -q  # 47 passed

# Run detector readout
python scripts/diagnostics/run_detector_v0_fixed.py --label-tier bronze
python scripts/diagnostics/run_detector_v0_fixed.py --label-tier silver_override
```
