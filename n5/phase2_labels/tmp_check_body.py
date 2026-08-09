"""Verify object-level body pose parity before/after sim.forward()."""
import json, os, copy, numpy as np, random as _random
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

DUMMY = [0,0,0,0,0,0,-1]
seed = 20260717
_r = _random; _r.seed(seed)
bm = get_benchmark("libero_10")(0); t = bm.get_task(0)
bp = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
sd = get_benchmark_dict(); so = sd["libero_10"]()
env = OffScreenRenderEnv(bddl_file_name=bp, camera_heights=256, camera_widths=256, render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=520)
env.seed(seed); env.reset(); env.set_init_state(copy.deepcopy(so.get_task_init_states(0)[15]))
for _ in range(10): env.step(DUMMY)

model = env.sim.model
object_bodies = []
for bid in range(model.nbody):
    name = model.body_id2name(bid)
    if name and "robot" not in name and "gripper" not in name and "world" not in name and "floor" not in name and "mount" not in name:
        object_bodies.append((bid, name))

print(f"Object bodies: {len(object_bodies)}")
all_ok = True
for bid, name in object_bodies:
    a = env.sim.data.body_xpos[bid].copy()
    env.sim.forward()
    b = env.sim.data.body_xpos[bid].copy()
    diff = float(np.max(np.abs(a - b)))
    ok = diff == 0
    if not ok: all_ok = False
    print(f"  {name}: diff={diff:.2e} {'OK' if ok else 'FAIL'}")

print(f"\nObject bodies parity: {'PASS' if all_ok else 'FAIL'}")

# Also check sites
sites_ok = True
for sid in range(model.nsite):
    name = model.site_id2name(sid)
    if name and "_region" in name:
        a = env.sim.data.site_xpos[sid].copy()
        env.sim.forward()
        b = env.sim.data.site_xpos[sid].copy()
        diff = float(np.max(np.abs(a - b)))
        ok = diff == 0
        if not ok: sites_ok = False
        print(f"  site {name}: diff={diff:.2e} {'OK' if ok else 'FAIL'}")

print(f"Site parity: {'PASS' if sites_ok else 'FAIL'}")
env.close()
print(f"\nVERDICT: {'SAME_LIVE_PARITY_CONFIRMED' if all_ok and sites_ok else 'FAIL'}")
