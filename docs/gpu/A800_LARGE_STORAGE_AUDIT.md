# A800 Large Storage Audit — 2026-06-20

**Audit time:** 2026-06-20T20:40–21:00 CST
**Auditor:** DeepSeek (migration lead)

---

## 1. Physical Storage Inventory

| Device | Size | Mount | FS | Free | Status |
|---|---|---|---|---|---|
| sda (446G) | via LVM | `/` | xfs | 76M | 🔴 Root — no |
| nvme0n1 | 2.9T | `/llm_jzm` | ext4 | 57G | 🔴 98% full, shared, policy blocked |
| nvme1n1 | 2.9T | `/mnt/sdc` | ext4 | 143G | 🟡 **ONLY CANDIDATE** |

**No additional physical devices exist.** There are no `/data`, `/scratch`, or other large mounts.

## 2. Candidate Evaluation: `/mnt/sdc`

| Criterion | Required | Actual | Status |
|---|---|---|---|
| available ≥ 500 GiB | 500 GiB | 143 GiB | **FAIL (can be fixed)** |
| post-migration free ≥ 300 GiB | 300 GiB | — | Conditional |
| inode free ≥ 20% | 20% | TBD | To verify |
| Stable non-ephemeral | Yes | ext4 on NVMe | PASS |
| dty_user r/w/rename/fsync | All | Confirmed | PASS |
| No high-growth active writer | — | dp_grid_results active | ⚠️ Growth monitored |
| Not root filesystem | Yes | /mnt/sdc | PASS |
| Not /llm_jzm | Yes | /mnt/sdc | PASS |

### Path verification

```
write:    touch /mnt/sdc/dty_user/.writetest → OK
fsync:    python -c "open('/mnt/sdc/dty_user/.writetest','a').flush(); os.fsync(...)" → OK
rename:   mv old new → OK
readback: sha256sum match → OK
delete:   rm → OK
```

## 3. Space Reclaim Potential

### dty_user major consumers on /mnt/sdc

| Directory | Size | Active Writer? | Safe to Move/Archive? |
|---|---|---|---|
| `pi0/data/lerobot_datasets/` | 377 GB | No | ✅ Static datasets |
| `pi0/checkpoints/` | 281 GB | No | ✅ Model checkpoints |
| `pi0/data/robotwin_raw/` | 45 GB | No | ✅ Raw data |
| `pi0/data/robotwin_lerobot/` | 45 GB | No | ✅ Processed data |
| `pi0/data/pi0_base/` | 14 GB | No | ✅ Base data |
| `Emu3.5/model/` | 131 GB | No | ✅ Model weights |
| `pi0_openpi/checkpoints/` | 44 GB | No | ✅ Checkpoints |
| **Total reclaimable** | **~937 GB** | | |

### Active pi0 processes

- 8 pi0 eval_policy processes using `/llm_jzm/mt/conda_envs/pi0/` (shared env, NOT /mnt/sdc data)
- 2 pi0_openpi spawn processes using `/mnt/sdc/dty_user/pi0_openpi/.venv/`
- These read checkpoints but do NOT write to the static data directories

### Projected post-cleanup

| Scenario | /mnt/sdc Free |
|---|---|
| Current | 143 GB |
| After removing pi0/data/lerobot_datasets (377G) | ~520 GB |
| After removing pi0/ entirely (759G) | ~902 GB |
| After removing pi0 + Emu3.5 + pi0_openpi (944G) | ~1087 GB |

## 4. Growth Observation

| Timestamp | /mnt/sdc Free | Delta |
|---|---|---|
| 08:07 UTC (first audit) | 701 GB | — |
| ~16:00 UTC (M1A start) | ~650 GB | -51 GB |
| 17:40 UTC (M1A end) | ~417 GB | -233 GB |
| 20:20 UTC (post-cleanup) | 147 GB | -270 GB |
| 20:50 UTC (now) | 143 GB | -4 GB |

Growth rate: ~2-4 GB/hour from dp_grid_results + lerobot training + mmunlearner.

## 5. Decision

**Selected storage root: `/mnt/sdc`** — the only physically available large-storage location.

To meet the 500 GB gate, dty_user must clean legacy pi0/Emu3.5/pi0_openpi data. This is dty_user's own data, not shared.

### Recommendation

1. Archive or remove `pi0/data/lerobot_datasets/` (377 GB) — largest single consumer
2. Keep pi0/checkpoints/ (281 GB) and Emu3.5/ (132 GB) pending user confirmation
3. Target: ≥ 500 GB free post-cleanup

**Deletion requires per-path user approval (APPROVE_GATE_MIG_DELETE_EXACT_PATHS).**
