# Stage-B RC1a Session Handoff — 2026-06-08

**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Latest commit**: `1284c8f` (remote tip)
**Status**: Freeze — do not continue mainline without new session review

## 0. Project One-Liner

Inference-time VIS PGD attack on OpenVLA-7B in LIBERO Object tasks. Target: identify an online-safe vulnerability window from clean rollout features, apply low-budget visual perturbation (PGD20, eps=6/255), induce gripper OPEN commands, and ideally create physical gripper/qpos response. Research highlight: window selection + low-budget targeted VIS, not brute-force attack.

## 1. Critical Semantic Correction

OpenVLA-LIBERO gripper execution semantics are frozen as RC1a:

```
raw_gripper > 0.5  → env_action_6 = -1.0 → physical OPEN
raw_gripper < 0.5  → env_action_6 = +1.0 → physical CLOSE
raw_gripper == 0.5 → boundary/neutral, excluded from OPEN/CLOSE
```

Official chain: `raw → normalize_gripper_action(binarize=True) → invert_gripper_action → env.step`

VIS token region: OPEN tokens decode to `raw > 0.5` and `env < -0.5`. Trusted files: `src/gripper_attack/openvla_libero_exec_spec.py`, `src/gripper_attack/attack_adapter.py`.

**All old results using `env > 0` or `raw < 0.5` as OPEN are INVALID and QUARANTINED.**

## 2. Quarantined / Deprecated

1. **Old 44-row patched rerun** — VIS objective inverted (targeting physical CLOSE)
2. **Old overnight Stage-B labels** — wrong open convention, qpos issues
3. **Pre-v1.1 traces** — missing metadata columns
4. **Active Probe v0b/v1** — no-env surrogate unreliable
5. **ProprioNoStep as detector** — detects contact/opportunity, not pre-grasp VIS vulnerability
6. **Pre-fix detector readouts** — `DIAGNOSTIC_ONLY_PRE_FIX`

## 3. Trusted RC1a Code Chain

| File | Role |
|------|------|
| `src/gripper_attack/openvla_libero_exec_spec.py` | RC1a gripper semantics |
| `src/gripper_attack/attack_adapter.py` | Corrected VIS token region |
| `scripts/run_stageb_vis_labeling.py` | v1.1 runner, 53-col trace |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | Validator with hard-fail gates |
| `scripts/stageb/postprocess_traces_v1_1.py` | Trace-level qpos recount |
| `scripts/stageb/build_pair_labels_v1_1.py` | Pair label builder (key includes seed) |
| `scripts/diagnostics/run_detector_v0_fixed.py` | Unified detector (bronze/silver/rescue tiers) |

Trace required metadata: `trace_version=corrected_stageb_v1_1`, `source_snapshot_id=f9840cb1`, `prompt_style=official_in_out`, `image_preprocess_style=official_rot180_only`, `qpos_source=obs_robot0_gripper_qpos`.

## 4. Data Stages and Experimental Results

### A. RC1a Clean Reachability Scan
- 27/27 clean rollouts, 0 infra fail
- 1198 reachable window candidates (sliding windows, actual trace length)
- All provenance: `corrected_stageb_v1_1`, `f9840cb1`

### B. Smoke3-B (3 windows)
- 6/6 validator PASS
- **cream_cheese s2 [45,55]**: VIS open=8 vs random=0 — command-level effect confirmed
- Physical qpos bridge weak in smoke

### C. Pilot12 (12 windows)
- 24/24 validator PASS
- cmd_sus=4, phys=2, rand_conf=0
- **butter s0 [70,80], [75,85]**: command + physical response (first phys bridge)

### D. Bronze Batch (48 windows, 96 jobs)
- 96/96 validator PASS, 45 valid pairs
- cmd_sus=11 (24%), rand_conf=8 (18%), phys=15 (33%), vis_spec=7 (16%)
- Corrected VIS produces stable nontrivial signal

### E. Silver P1A (84 jobs, enriched repeats)
- 84/84 validator PASS, 37 pairs from 23 parents
- stable_cmd=9, stable_phys=4, stable_rand=6, unstable=6
- pos_stability=0.64, rand_stability=0.75
- Random-sensitive windows are real and stable

### F. P1b (36 jobs)
- 36/36 validator PASS, 18 pairs
- cmd_sus=2, phys=3, rand=4 — adds negatives + underrepresented tasks

### G. Random-Confounded Rescue (42 jobs)
- 42/42 validator PASS, 18 rows, 12 parents, 6 multi-repeat
- Aggregated with stability logic (same as Silver), NOT last-row overwrite
- Rescue override: cmd=11, phys=3, rand=3

## 5. Detector v0 Status

**The detector is NOT a final online vulnerable-window detector.** It is an exploratory multi-head selector.

Targets: `cmd_specific` (cmd_any AND NOT random_sensitive), `vis_specific_physical`, `random_sensitive` (abstain head).

| Target | Best Tier | Best Model | P@5 | Enrich | AUROC |
|--------|-----------|-----------|-----|--------|-------|
| cmd_specific | Rescue | TaskOnly | 0.60 | 2.5x | 0.68 |
| cmd_specific | Bronze | CleanNoTaskNoTiming | 0.40 | 1.6x | 0.46 |
| vis_specific_phys | Silver | CleanNoTaskNoTiming | 0.40 | 1.8x | 0.46 |
| random_sensitive | Silver | CleanNoTaskNoTiming | 0.60 | 1.5x | 0.77 |

**Interpretation**: cmd_specific dominated by task bias (butter). Clean features do not robustly add beyond task identity. Physical bridge signal promising but underpowered. Random-sensitive is a real confounder (AUROC=0.77) requiring abstain head.

## 6. Allowed vs Forbidden Claims

**Allowed**: RC1a corrected VIS produces repeatable command-level effects. Some windows show VIS-specific physical response. Random-sensitive windows are real confounds. Detector v0 is exploratory.

**Forbidden**: Final detector solved. Broad LIBERO generalization. Real robot ready. Official SR as primary metric. Old 44-row as baseline. Bronze labels as gold. Merging random_sensitive into negative.

## 7. Next Steps (do NOT execute until new session review)

1. **Freeze state** — verify `a3aecd7` pushed, detector uses aggregated rescue labels
2. **Targeted expansion** — more non-butter cmd positives, physical positives, hard negatives, same-task contrasts. Target: ≥25 cmd_specific, ≥15 phys, ≥25 rand, ≥40 hard_neg across tasks
3. **Visual sidecar pilot** — offline CLIP/DINOv2 from clean frames, compare ProprioOnly/VisualOnly/Proprio+Visual on same grouped splits
4. **Multi-head detector** — `attack_score = p(phys_bridge) - λ·p(random_sensitive)`, attack only when score high

**Do NOT**: blind 48-window expansion, butter-only sampling, treat random_sensitive as negative, merge old labels.

## 8. Artifact Index

### Server Output Roots
- `/data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/` — 27 clean rollouts
- `/data/liuyu/outputs/stageb_v1_1_bronze_batch_rc1a_20260607/` — 96 traces, 45 pairs
- `/data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608/` — 84 traces, 37 pairs
- `/data/liuyu/outputs/stageb_v1_1_silver_p1b_rc1a_20260608/` — 36 traces, 18 pairs
- `/data/liuyu/outputs/stageb_v1_1_random_confounded_rescue_rc1a_20260608/` — 42 traces, 12 parents
- `/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/` — detector readouts
- `/data/liuyu/outputs/stageb_v1_1_corrected_smoke3b_rc1a_20260607/` — smoke3-B

### Server Execution Copy
- `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/` — isolated RC1a execution tree (dirty reviewed worktree at `...-reviewed-20260605` is deprecated)

### Key Local Tables
- `tables/stageb_v1_1_reachable_window_candidates.csv` — 1198 candidates
- `tables/stageb_v1_1_clean_rollout_summary.csv` — 27 rollouts
- `tables/stageb_v1_1_corrected_smoke3b_pair_labels.csv`
- `tables/stageb_v1_1_corrected_pilot12_queue_v2.csv`
- `tables/stageb_v1_1_bronze_batch_queue_rc1a.csv`

### Key Reports
- `reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md`
- `reports/STAGEB_V1_1_RC1_FREEZE_REPORT.md`
- `reports/STAGEB_V1_1_REACHABLE_WINDOW_CANDIDATE_GENERATION.md`
- `reports/STAGEB_V1_1_CORRECTED_SMOKE3B.md`
- `reports/STAGEB_V1_1_QPOS_DIRECTION_AUDIT.md`
- `reports/STAGEB_V1_1_CANDIDATE_DISTRIBUTION_AUDIT.md`
- `reports/STAGEB_V1_1_SILVER_P1A_FINAL_READOUT.md` (MISSING — generate if needed)
- `reports/STAGEB_V1_1_DETECTOR_V0_REPAIR_AUDIT.md` (MISSING)

## One-Paragraph Takeaway

The corrected RC1a VIS pipeline is now trustworthy and produces repeatable command/physical attack effects, but the detector is not yet solved. Command-specific prediction is currently task-biased (butter dominates); the more promising route is a multi-head selector that predicts physical bridge susceptibility while abstaining from random-sensitive windows. Next work should focus on targeted balanced data expansion (more non-butter positives, physical bridge, hard negatives) and a visual sidecar pilot (CLIP/DINOv2 from clean frames to reduce task bias), not blind scaling or old-result reuse.
