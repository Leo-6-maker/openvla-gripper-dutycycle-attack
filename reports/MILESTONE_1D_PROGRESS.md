# Milestone 1D: MuJoCo 2.3.7 Compat — Progress Report

**Timestamp**: 2026-05-26
**Status**: `blocked_by_target_ssh_instability`

## Accomplished

### Phase 0: Preflight ✅
- Server: klfy-SYS-4028GR-TR2 (10.60.133.4), Python 3.10.18 main env
- Current env: MuJoCo 3.8.0, robosuite 1.4.1, numpy 2.2.6, torch 2.6.0+cu124
- All GPUs idle, persistence ON, driver 530.41.03
- Object checkpoint, v4 runner, prior audits, clean baseline — all exist
- Object task IDs confirmed:
  - `object_pick_up_the_bbq_sauce_and_place_it_in_the_basket`
  - `object_pick_up_the_cream_cheese_and_place_it_in_the_basket`
  - `object_pick_up_the_ketchup_and_place_it_in_the_basket`
  - `object_pick_up_the_milk_and_place_it_in_the_basket`

### Phase 1: Compat Env Setup ✅ (90%)
- Env: `/data/aviary/envs/openvla_official_libero_20260525`
- Python 3.10.16
- **MuJoCo 2.3.7** installed ✅
- **numpy 1.26.4** installed ✅
- Overlay architecture:
  1. compat env site-packages (mujoco 2.3.7, numpy 1.26.4) — TAKES PRECEDENCE
  2. openvla_sparse overlay (robosuite 1.4.1, libero 0.1.1, transformers) 
  3. conda py310 system (torch 2.6.0+cu124)
- All imports verified: mujoco, numpy, torch, transformers, robosuite, gym, libero
- `libero.libero.benchmark` loads successfully

### Phase 2: Smoke Test 🟡
- Script deployed to /tmp/smoke_test.py
- MuJoCo 2.3.7 confirmed (import + version check)
- robosuite 1.4.1 confirmed
- LIBERO benchmark dict loads
- **Bug**: `get_libero_path()` requires a `query_key` arg — script needs fix
- **Blocked**: Cannot fix script because target SSH is down

## Blocker

**Target server SSH daemon on 10.60.133.4 keeps crashing.** 
The machine is UP (ping 0.3ms, port 80 HTTP responds), but SSH on port 22 becomes unresponsive after a few connections. This has happened 3+ times during this session.

## Next Steps (when SSH recovers)

1. Fix smoke_test.py: replace `get_libero_path()` with `get_libero_path("datasets")` or appropriate key
2. Re-run smoke test: env creation, reset, step with open/close gripper
3. If smoke passes → Phase 3: Run 40-episode clean matrix
4. Generate all output tables and final diagnosis

## Scripts Ready on Server

- /tmp/smoke_test.py (needs fix)
- /tmp/setup_compat.sh (completed)
- /tmp/fix_compat.sh (completed)
- /tmp/preflight.sh (completed)
