# /mnt/sdc Capacity Delta Audit — 2026-06-20

**Audit time:** 2026-06-20T20:30 CST
**Auditor:** DeepSeek (migration lead)

---

## 1. Delta Summary

| Metric | First Audit (08:07 UTC) | After M1A (17:40) | After Cleanup (20:20) | Total Delta |
|---|---|---|---|---|
| /mnt/sdc free | ~701 GiB | ~389 GiB | ~147 GiB | **-554 GiB** |

The 701→147 GiB delta represents ~554 GiB consumed over ~12 hours.

## 2. dty_user Consumption on /mnt/sdc

| Directory | Size | Category | Our Asset |
|---|---|---|---|
| `pi0/` | 759 GB | π0 model data & training | ❌ Pre-existing |
| `Emu3.5/` | 132 GB | Emu3.5 model | ❌ Pre-existing |
| `cache/` | 88 GB | HF/modelscope/wandb cache | ❌ Pre-existing |
| `RoboTwin_official/` | 85 GB | RoboTwin benchmark | ❌ Pre-existing |
| `pi0_openpi/` | 53 GB | OpenPI variant | ❌ Pre-existing |
| `openvla_attack/` | **38 GB** | **Our migration env** | ✅ New |
| `dp_env/` | 5.2 GB | Python venv for dp_attack | ❌ Pre-existing |
| Others | ~27 GB | Various | ❌ Pre-existing |
| **Total dty_user** | **~1.19 TB** | | |

## 3. Other Users on /mnt/sdc

| Directory | Owner | Size |
|---|---|---|
| `yangyenan/` | yyn | 126 GB |
| `yyn_bavit_new/` | yyn | 73 GB |
| `zkx/` | zkx | 51 GB |
| `taozhen/` | tz | <1 GB |
| **Total others** | | **~250 GB** |

## 4. Active Writers (last 30 min)

| Writer | Path | Type |
|---|---|---|
| yangyenan (.codex) | `/mnt/sdc/yangyenan/.codex/logs_2.sqlite` | Codex logs |
| dp_grid_results | `/mnt/sdc/dty_user/dp_grid_results/*.log` | dp_attack eval logs |
| lerobot-train | `/mnt/sdc/lerobot_piper/data/` | ACT training (PID 4025523) |

## 5. Our Contribution to Delta

| Item | Size |
|---|---|
| openvla_attack env (conda + pip) | 38 GB |
| conda_pkgs (package cache) | Included |
| **Total our contribution** | **~38 GB** |

The remaining ~516 GB delta is from non-migration sources (pre-existing pi0 data at 759 GB, other users' caches, ongoing dp_grid_results and lerobot training).

## 6. Model Sync Space Budget

| Parameter | Value |
|---|---|
| /mnt/sdc currently free | 147 GB |
| Per-checkpoint estimate | ~14 GB |
| 4 suites total estimate | ~56 GB |
| After 1 suite (spatial) | ~133 GB |
| After all 4 suites | ~91 GB |
| Gate threshold | **200 GB** |

**Model sync cannot proceed: both ROOT (< 20 GiB) and /mnt/sdc (< 200 GiB) gates fail.**

## 7. Recommendations

1. **dty_user can clean:** `pi0/` directory (759 GB) — if π0 model data is not needed for migration
2. **Admin action needed:** Root cleanup of other users' homes
3. **Await natural cleanup:** dp_grid_results and lerobot training will eventually finish
