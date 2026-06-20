# A800 Root Cleanup Report — 2026-06-20

**Date:** 2026-06-20T20:19–20:21 CST
**Auditor:** DeepSeek (migration lead)
**Gate:** MIG0_OWN_CACHE_CLEANUP
**Approval:** APPROVE_GATE_MIG0_ROOT_CLEANUP_OWN_CACHE_ONLY

---

## 1. Pre-Cleanup Audit

### Process Reference Check

All 5 candidate paths had zero active process references (`fuser` confirmed).

### Key Discovery: `.cache` is symlinked

`/home/dty_user/.cache` → `/mnt/sdc/b1/dty_user_home_redirects/.cache`

This means the 65 GB of cache data (huggingface, uv, vllm, openpi, pip, torch, etc.) was **already on /mnt/sdc**, never on root. The earlier `du -sh ~/.cache/*` measurements followed the symlink and reported /mnt/sdc data as if on root.

This significantly limits how much root space dty_user can reclaim — only 16 GB of dty_user data is actually on root, mostly IDE server installations and small data directories.

## 2. Cleanup Ledger

| Path | Size Before | Refs | Decision | Root Reclaimed | Note |
|---|---|---|---|---|---|
| `~/.cache/uv` | 14.9 GB | 0 | DELETED | 0 | Was on /mnt/sdc (via symlink) |
| `~/.cache/vllm` | 5.5 GB | 0 | DELETED | 0 | Was on /mnt/sdc (via symlink) |
| `~/.cache/electron` | 108 MB | 0 | DELETED | 0 | Was on /mnt/sdc (via symlink) |
| `~/.cache/electron-builder` | 50 MB | 0 | DELETED | 0 | Was on /mnt/sdc (via symlink) |
| `~/model_cache` | 581 MB | 0 | DELETED | **581 MB** | Was on root directly |

### /mnt/sdc reclaim (side effect)

The uv/vllm/electron caches deleted through the symlink freed ~12 GB on /mnt/sdc.

## 3. Space Summary

| Metric | Before | After | Delta |
|---|---|---|---|
| Root free | 23 MiB | **604 MiB** | +581 MiB |
| /mnt/sdc free | 135 GiB | **147 GiB** | +12 GiB |
| Root usage % | 100% | 100% | — |

## 4. Held Items (not deleted)

| Path | Size | Reason |
|---|---|---|
| `~/.cache/huggingface` | 33 GB | Already on /mnt/sdc; protocol blocked |
| `~/.cache/openpi` | 12 GB | Already on /mnt/sdc; pi0 processes running |
| `~/.cache/torch` | 239 MB | Already on /mnt/sdc; negligible |
| `~/.vscode-server` | 7.0 GB | IDE server; requires VS Code disconnect |
| `~/.trae-cn-server` | 3.4 GB | IDE server; may be in use |
| `~/.cursor-server` | 2.3 GB | IDE server; may be in use |
| `~/.trae-server` | 1.5 GB | IDE server; may be in use |
| `~/.nvm` | 1.0 GB | Node.js; may be needed |
| `~/.conda` | 579 MB | Has envs symlink; keep |
| `~/.claude` | 240 MB | Session history; keep |
| `~/.ssh` | 84 KB | Critical; never delete |

## 5. M0 Gate Status After Cleanup

| Check | Current | Required | Status |
|---|---|---|---|
| Root free | 604 MiB | ≥ 20 GiB | **FAIL** |
| /mnt/sdc free | 147 GiB | ≥ 200 GiB (model sync gate) | **FAIL** |

**M0 remains BLOCKED_BY_ROOT_FULL.**

dty_user has limited root data to clean — the bulk is already on /mnt/sdc via symlinks. Root space must come from admin action on other users' homes (ysc2: 177 GB, sz: 44 GB, huanzze: 27 GB).

## 6. Process Safety

| Check | Status |
|---|---|
| Active references found | 0 |
| Other users touched | 0 |
| sudo used | No |
| Processes killed | 0 |
| Paths deleted outside approved list | 0 |

---

*Cleanup ledger: see `migration_audit/host/root_cleanup_ledger_20260620.csv` in repo.*
