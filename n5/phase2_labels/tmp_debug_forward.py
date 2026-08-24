"""Quick debug: compare recorded vs state-forward body pose."""
import json, os, glob, numpy as np
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

ep_dirs = sorted(glob.glob("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/grec_fallback_batch_l10_b5c7853_20260727/episodes/*/"))
ep = json.load(open(os.path.join(ep_dirs[0], "episode.json")))
t0 = ep["telemetry"][0]
ss = t0["sim_state"]
entities = t0["entities"]
print("Episode:", ep["episode_id"])
print("Recorded qpos[:10]:", np.array(ss["qpos"][:10]))
print("Recorded qvel[:10]:", np.array(ss["qvel"][:10]))
print("N entities:", len(entities))

suite = t0["suite"]; task_idx = t0["task_idx"]
benchmark = get_benchmark(suite)(0)
task = benchmark.get_task(task_idx)
bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=224, camera_widths=224, render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False, horizon=500)
env.reset()
print("After reset qpos[:10]:", env.sim.data.qpos[:10])
print("model.nq:", env.sim.model.nq, "nmocap:", env.sim.model.nmocap)
print("model.nv:", env.sim.model.nv)
print("recorded qpos len:", len(ss["qpos"]), "qvel len:", len(ss["qvel"]))

env.sim.data.qpos[:] = np.array(ss["qpos"], dtype=np.float64)
env.sim.data.qvel[:] = np.array(ss["qvel"], dtype=np.float64)
env.sim.data.time = float(ss.get("time", 0))
env.sim.forward()

# Compare first 5 entities
for e in entities[:5]:
    etype = e["entity_type"]; eid = e["entity_id"]
    wp = e["world_pose"]
    rec = np.array(wp["position"])
    quat = wp.get("quaternion", [0,0,0,1])
    if etype == "body":
        fwd = env.sim.data.body_xpos[eid].copy()
    elif etype == "site":
        fwd = env.sim.data.site_xpos[eid].copy()
    elif etype == "geom":
        fwd = env.sim.data.geom_xpos[eid].copy()
    else:
        continue
    diff = rec - fwd
    print(f"  {e['entity_name']}({etype}#{eid}): rec={rec} fwd={fwd} diff_Linf={np.max(np.abs(diff)):.6f}")

env.close()
