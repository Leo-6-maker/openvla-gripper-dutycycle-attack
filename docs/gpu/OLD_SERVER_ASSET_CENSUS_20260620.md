# Old Server Asset Census — 2080Ti (klfy-SYS-4028GR-TR2)

**Date:** 2026-06-20
**Audit type:** REMOTE READ-ONLY METADATA CENSUS
**Auditor:** DeepSeek (migration lead)
**Server access:** via `ssh vla` (no direct IP disclosure)

---

## 1. Server Summary

| Field | Detail |
|---|---|
| Hostname | klfy-SYS-4028GR-TR2 |
| Root FS | 916G total, 310G used (559G free, 36%) |
| Data FS | /data: 1.8T total, 636G used (1.1T free, 37%) |
| GPUs | 8× NVIDIA GeForce RTX 2080 Ti (11 GB each) |
| GPUs in use | GPU 4 (41%), GPU 5 (39%) — Codex cross-suite eval |
| GPUs idle | 0, 1, 2, 3, 6, 7 |
| Active user | `liuyu` running cross-suite clean train300 eval |
| Total /data/liuyu | **52 GiB** |

---

## 2. Asset Root Map

```
/data/liuyu/repos/          — 58+ code repository checkouts (~747 MiB largest: LIBERO)
/data/liuyu/outputs/        — 501 output directories (45 GiB total)
/data/aviary/models/openvla/ — OpenVLA fine-tuned checkpoints (4 suites)
/data/aviary/envs/          — 3 conda environments
```

---

## 3. Models (Tier T2)

| Model | Path | Size Estimate |
|---|---|---|
| openvla-7b (base) | `/data/aviary/models/openvla/openvla-7b/` | ~14 GB |
| openvla-7b-finetuned-libero-object | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object/` | ~14 GB |
| openvla-7b-finetuned-libero-spatial | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial/` | ~14 GB |
| openvla-7b-finetuned-libero-goal | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal/` | ~14 GB |
| openvla-7b-finetuned-libero-10 | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-10/` | ~14 GB |

**Note:** These are fine-tuned weights from the OpenVLA project. Hash and verify against upstream before declaring as reference checkpoints.

---

## 4. Conda Environments (Tier T2)

| Env | Path | Purpose |
|---|---|---|
| openvla_official_libero_20260525 | `/data/aviary/envs/openvla_official_libero_20260525/` | **Current official reference** — used by active cross-suite eval |
| openvla_compat | `/data/aviary/envs/openvla_compat/` | Legacy compatibility env |
| openvla_sparse | `/data/aviary/envs/openvla_sparse/` | Sparse variant |

The `openvla_official_libero_20260525` env is the canonical runtime for the old server. Its lock file must be captured for parity comparison.

---

## 5. Key Code Repositories

### Active / Production

| Repo | Commit/SHA | Status |
|---|---|---|
| `openvla-gripper-dutycycle-attack-reviewed-20260605` | branch: `exp/l12-production-streaming-adapter-20260615` | **Current production mainline** (git remote `vla`) |
| `train300_collector_freeze_141657f` | `141657f` | **Active Codex eval** — cross-suite clean train300 (2026-06-20) |
| `sc5_census_freeze_cc356f3_20260618` | `cc356f3` | Frozen census branch (SC5 canonical corpus) |
| `sc5_census_freeze_e69d9a1_20260618` | `e69d9a1` | Frozen census variant |
| `sc5_census_freeze_7ab15f1_20260618` | `7ab15f1` | Frozen census variant |

### DeepSeek Historical (L3)

| Repo | Purpose |
|---|---|
| `l3_deepseek_main` | L3 autonomous experiment base |
| `m3_gpu15_autonomous_d02db0b` | GPU15 autonomous run |
| `m3_arm_v4_b2c1dc0` | ARM v4 experiment |
| `m3_v5_seal_2a57ffe` | M3 v5 sealed results |

### Codex Workspace (DO NOT TOUCH)

| Repo | Branch |
|---|---|
| `codex_stageb_openvla_alignment_rc1a_20260607` | Codex active workspace |

### Layer 1 H2 Variants (multiple bundles)

| Repo | Note |
|---|---|
| `layer1_h2_0d7f91d/` | Working copy |
| `layer1_h2_592adc0/` | Working copy |
| `layer1_h2_68f02dd/` | Working copy |
| `layer1_h2_a161c1f/` | Working copy |
| `layer1_h2_0c466a9_bundle/` | Bundle (not expanded) |
| `layer1_h2_6eb8863_bundle/` | Bundle (not expanded) |

### LIBERO Official

| Repo | Branch | Note |
|---|---|---|
| `LIBERO-official-20260525/` | `master` | Cloned 2026-05-25; need to verify upstream SHA |

### Other Historical

- ~40 additional repos spanning milestones 2c through m3_l3telemetry

---

## 6. Output Directory Census (Top-Level by Size)

Total outputs: **45 GiB** across 501 directories.

| Tier | Directory | Size | Category |
|---|---|---|---|
| 🔴 | `cross_suite_clean_train300_s10_19_20260620/` | 5.2G | **Active Codex eval** — still running |
| 🔴 | `milestone_2e4_cross_suite300_privileged_artifact_rich_20260527/` | 4.0G | Frozen privileged artifacts |
| 🔴 | `cross_suite_clean_300_20260619_r1_6379397/` | 3.1G | Cross-suite clean 300 run 1 |
| 🔴 | `milestone_2e5_l10100_parser_v2_privileged_rerun_20260527/` | 2.2G | Parser v2 privileged |
| 🔴 | `cross_suite_clean_300_20260619_waveb_worker15_6379397/` | 2.1G | Cross-suite wave B |
| 🟡 | `stageb_s20f_queues_20260611/` | 1.6G | Queue runner outputs |
| 🟡 | `milestone_2f_object_oracle_sensitivity_full10x5_20260529/` | 1.5G | Object oracle sensitivity |
| 🟡 | `stageb_s20m4_rand_stability_20260613/` | 1.4G | Random stability |
| 🟡 | `milestone_2e2_object100_privileged_artifact_rich_20260527/` | 1.4G | Object100 privileged |
| 🟡 | `m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039/` | 1.1G | ARM v5 clean capture |
| 🟢 | `libero_full4_clean_official_aligned_eager_10states_20260525/` | 615M | Official-aligned clean eval |
| ... | *(490 more directories)* | ... | Various stages |

**Key frozen outputs identified:**
- Cross-suite clean 300 (tag: `freeze/cross-suite-clean300-20260619`)
- Layer 1/2/3 POC (tag: `freeze/layer123-poc-20260618`)
- Table1 clean patched v4 (tag: `table1-clean-patched-v4-20260527`)

---

## 7. Freeze Tags (from GitHub)

| Tag | Date | Description |
|---|---|---|
| `freeze/cross-suite-clean300-20260619` | 2026-06-19 | Cross-suite CLEAN300 final freeze |
| `freeze/layer123-poc-20260618` | 2026-06-18 | Layer 1/2/3 end-to-end POC |
| `l12-d5-v1-production-20260617` | 2026-06-17 | L12 production handoff |
| `milestone-2c-proprio-causal-student-20260526` | 2026-05-26 | Proprio causal student |
| `milestone-2c1-student-replay-ablation-20260526` | 2026-05-26 | Student replay ablation |
| `milestone-r3-pil-object-80-20260527` | 2026-05-27 | PIL Object 80 |
| `table1-clean-patched-v4-20260527` | 2026-05-27 | Table 1 clean patched |

---

## 8. Codex Branch Inventory (DO NOT TOUCH)

| Branch | Remote | Status |
|---|---|---|
| `codex/layer3-feasible-route-20260613` | origin | Active |
| `exp/l2-sc5-census-freeze-fix-codex-20260618` | origin | Active |
| `exp/l3-independent-audit-prep-20260617` | origin | Audit prep |
| `exp/l3-vis-handoff-contract-repair-20260617` | origin | Contract repair |
| `feature/sc5-cross-suite-layer1-resolver-20260619` | origin | **Active — Layer 1 resolver** |

Codex is actively running cross-suite eval from `exp/cross-suite-clean-train300-s10-19-20260620` on GPU 4 and 5 of the old server.

---

## 9. High-Level Space Summary

| Asset Group | Size | Server |
|---|---|---|
| Outputs (all) | 45 GiB | 2080Ti |
| Code repos (all) | ~2 GiB | 2080Ti |
| Models (OpenVLA ×5) | ~70 GiB | 2080Ti |
| Conda envs | ~15 GiB | 2080Ti |
| **Total eligible for transfer** | **~132 GiB** | 2080Ti |

Well within the 650–700 GiB initial transfer budget, assuming `/mnt/sdc` maintains 716 GiB free.

---

## 10. Active Writers Detected

| Directory | Process | Status |
|---|---|---|
| `/data/liuyu/outputs/cross_suite_clean_train300_s10_19_20260620/` | Codex cross-suite eval (goal suite) | **ACTIVE WRITER** — do NOT transfer |

This is the Codex-owned eval process. Must be excluded from any transfer manifest.

---

## 11. Preliminary Tier Classification

### T0 — Frozen Scientific Evidence (priority transfer)

| Asset | Location | Tag |
|---|---|---|
| CLEAN300 accepted dataset | GitHub evidence/manifests/ | `freeze/cross-suite-clean300-20260619` |
| Layer 1/2/3 POC outputs | `/data/liuyu/outputs/layer123_final*/` | `freeze/layer123-poc-20260618` |
| Detector checkpoint | Needs exact path confirmation | Pending |

### T1 — Complete but not yet frozen

| Asset | Location |
|---|---|
| Train300 outputs | Multiple milestone directories |
| H2 blind validation | layer1_h2_* repos |
| Cross-suite canary results | `cross_suite_clean_*_canary_*` |

### T2 — Models and Environments

| Asset | Location |
|---|---|
| OpenVLA checkpoints | `/data/aviary/models/openvla/` |
| Official conda env | `/data/aviary/envs/openvla_official_libero_20260525/` |
| LIBERO upstream | `/data/liuyu/repos/LIBERO-official-20260525/` |

### T3 — Regenerable

| Asset | Location |
|---|---|
| Overlay videos | Inside output directories |
| CSVs, plots | Output metadata |
| Cache files | Various |

### T4 — Legacy Reference

| Asset | Location |
|---|---|
| Historical experiment outputs (pre-May) | Multiple directories |
| Old runner formats | Various |

### TX — Rejected for Transfer

| Asset | Location | Reason |
|---|---|---|
| Active cross-suite eval | `cross_suite_clean_train300_s10_19_20260620/` | **ACTIVE WRITER** |
| Temporary/partial files | Various | Incomplete |
| Direct test dirs | `direct_test/`, `gpu*_smoke_*` | Debug artifacts |

---

*Census is metadata-only. No files were read, transferred, modified, or deleted.*
*T0/T1 classification is preliminary — requires user confirmation before any transfer.*
