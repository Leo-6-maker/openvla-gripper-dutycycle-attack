#!/usr/bin/env python3
"""A3: Model-free binding preflight for all 10 LIBERO-Spatial tasks."""
import json, os, sys, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
suite = bd["libero_spatial"]()
results = []
all_pass = True

for ti in range(10):
    task = suite.get_task(ti)
    init_states = suite.get_task_init_states(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    task_name = task.language if hasattr(task, "language") else task.name

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             has_renderer=True, has_offscreen_renderer=True,
                             render_gpu_device_id=0, use_camera_obs=True)
    env.seed(42)
    obs = env.reset()
    obs = env.set_init_state(init_states[0])

    obj_key, tgt_key = None, None
    for pk in ["akita_black_bowl_1_pos", "akita_black_bowl_pos"]:
        if pk in obs:
            obj_key = pk; break

    for pk in ["plate_1_pos", "plate_pos"]:
        if pk in obs:
            tgt_key = pk; break

    obj_finite = np.all(np.isfinite(obs.get(obj_key, [np.nan]*3))) if obj_key else False
    tgt_finite = np.all(np.isfinite(obs.get(tgt_key, [np.nan]*3))) if tgt_key else False

    classification = "BINDING_INVALID"
    if obj_key and tgt_key and obj_finite and tgt_finite:
        # Check if target is a site-based "plate_region" or body-based "plate_1"
        if "plate_region" in str(tgt_key) or "region" in str(tgt_key):
            classification = "TARGET_REGION_EXACT"
        else:
            classification = "PLATE_BODY_CENTER_PROXY"

    row = {
        "task_idx": ti, "task_name": task_name[:80],
        "configured_object": "akita_black_bowl_1",
        "configured_target": "plate_1 / plate_region",
        "resolved_obj_key": obj_key, "resolved_tgt_key": tgt_key,
        "object_pose_finite": bool(obj_finite),
        "target_pose_finite": bool(tgt_finite),
        "classification": classification,
    }
    results.append(row)

    if classification == "BINDING_INVALID":
        all_pass = False
        print(f"  task{ti}: {classification}")
    else:
        print(f"  task{ti}: {classification} obj={obj_key} tgt={tgt_key}")

    env.close()

out = {"tasks": results, "all_pass": all_pass, "classification_counts": {}}
for r in results:
    out["classification_counts"][r["classification"]] = out["classification_counts"].get(r["classification"], 0) + 1

os.makedirs("migration_audit/detector", exist_ok=True)
json.dump(out, open("migration_audit/detector/all_task_binding_preflight.json", "w"), indent=2)

for cls, cnt in sorted(out["classification_counts"].items()):
    print(f"{cls}: {cnt}/10")

if all_pass:
    print("ALL_TASK_BINDING_PREFLIGHT: PASS")
    sys.exit(0)
else:
    print("ALL_TASK_BINDING_PREFLIGHT: FAIL")
    sys.exit(1)
