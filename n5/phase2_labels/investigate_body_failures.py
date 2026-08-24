"""[DeepSeek] R5-B: A/B/C Triple-Read Investigation.

Tests: A = pre-forward body pose, B = after 1st forward, C = after 2nd forward.
Hypothesis: failures are stale reads (env.step() may not update all body buffers).
"""
import json, os, sys, math, copy, numpy as np
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
import random as _random

DUMMY = [0,0,0,0,0,0,-1]
SEED = 20260717
TASK = ("libero_10", 0, 15)

_r = _random; _r.seed(SEED)
bm = get_benchmark(TASK[0])(0); t = bm.get_task(TASK[1])
bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
sd = get_benchmark_dict(); so = sd[TASK[0]]()
env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256,
                         render_gpu_device_id=-1, has_renderer=False,
                         has_offscreen_renderer=False, horizon=520)
env.seed(SEED); env.reset(); env.set_init_state(copy.deepcopy(so.get_task_init_states(TASK[1])[TASK[2]]))
for _ in range(10): env.step(DUMMY)

model = env.sim.model
# Get object body IDs (skip robot/gripper/world/floor/mount)
body_ids = []
for bid in range(model.nbody):
    name = model.body_id2name(bid)
    if name and all(k not in name for k in ("robot","gripper","world","floor","mount")):
        body_ids.append((bid, name))

print(f"Task: {TASK[0]}/task_{TASK[1]:02d}/state_{TASK[2]}")
print(f"Object bodies: {len(body_ids)}")
print(f"{'='*70}")

n_total = 0; n_ab_fail = 0; n_bc_fail = 0; n_stale = 0
for step in range(20):
    saved_qpos = env.sim.data.qpos.copy()
    saved_time = float(env.sim.data.time)

    # A: read WITHOUT explicit forward (simulates collector's direct read after env.step)
    A_pos = {}
    for bid, name in body_ids:
        A_pos[bid] = env.sim.data.body_xpos[bid].copy()

    # B: after 1st forward
    env.sim.forward()
    B_pos = {}
    for bid, _ in body_ids:
        B_pos[bid] = env.sim.data.body_xpos[bid].copy()

    # C: after 2nd forward
    env.sim.forward()
    C_pos = {}
    for bid, _ in body_ids:
        C_pos[bid] = env.sim.data.body_xpos[bid].copy()

    for bid, name in body_ids:
        n_total += 1
        ab_err = float(np.max(np.abs(A_pos[bid] - B_pos[bid])))
        bc_err = float(np.max(np.abs(B_pos[bid] - C_pos[bid])))
        if ab_err > 1e-12:
            n_ab_fail += 1
            if bc_err <= 1e-15:
                n_stale += 1
                if ab_err > 1e-8:
                    print(f"  STALE step={step} {name}: A→B={ab_err:.2e} B→C={bc_err:.2e}")
            else:
                n_bc_fail += 1
                print(f"  NON_DETERM step={step} {name}: A→B={ab_err:.2e} B→C={bc_err:.2e}")

    env.step([0.0]*7)

print(f"\n{'='*70}")
print(f"Total body-step cases: {n_total}")
print(f"A→B failures (>1e-12): {n_ab_fail} ({n_ab_fail/n_total*100:.1f}%)")
print(f"  of which stale (B==C): {n_stale}")
print(f"  of which non-deterministic (B!=C): {n_bc_fail}")
print(f"B→C failures (>1e-15): {n_bc_fail}")

if n_bc_fail > 0:
    print("\nVERDICT: NON_DETERMINISTIC — sim.forward() produces different results. STOP.")
    sys.exit(5)
elif n_stale > 0:
    print(f"\nVERDICT: STALE_READ — {n_stale} cases. Fix: call sim.forward() before recording.")
    print("Root cause: env.step() may not update body_xpos buffers for all bodies.")
    print("Required: collector must call sim.forward() atomically before reading poses.")
    sys.exit(5)
else:
    print(f"\nVERDICT: ALL_CONSISTENT — A==B==C for all cases.")
    print("The 18 previous failures were likely from implementation artifact.")
    sys.exit(0)

env.close()
