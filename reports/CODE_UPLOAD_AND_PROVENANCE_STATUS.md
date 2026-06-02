# Code Upload and Provenance Status

**Date**: 2026-05-30 | **Branch**: `exp/sustained-proxy-burst-control-20260530`

## Commit Lineage

| Location | Commit | Notes |
|----------|--------|-------|
| Server | `703c172` | Hardening (local SHA from `git am` of format-patch) |
| Local | `07e13a0` | Hardening (original commit) |
| Remote (GitHub) | `07e13a0` | Already pushed — same hardening logic |
| Server base | `f07c98a` | Sustained proxy (equivalent to local `e7e5bd1`) |

**Remote-visible lineage**: `e7e5bd1` (sustained proxy) → `07e13a0` (hardening)

## Verification

| Check | Result |
|-------|--------|
| py_compile `run_official_eval_artifact_rich.py` | PASS |
| py_compile `test_sustained_proxy_burst.py` | PASS |
| Test discovery (13 cases) | PASS |
| Large files (>5M) | NONE |
| Modified tracked files | NONE |
| Untracked (configs/docs only) | OK |
| `attack_burst_steps` guarded to sustained proxy | CONFIRMED |
| Remote has hardening logic | CONFIRMED |

## Push Status

**Not pushed** — no action needed:
- Remote already contains `07e13a0` (hardening)
- Server `703c172` is the format-patch equivalent
- No new code commits since last push
- Local and remote are in sync

## Branch Status

`exp/sustained-proxy-burst-control-20260530`: **production-ready**, no further changes needed.
