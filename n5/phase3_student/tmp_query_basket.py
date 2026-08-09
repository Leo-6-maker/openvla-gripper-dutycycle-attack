"""Query basket_1_contain_region site local pose from MuJoCo."""
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
import os

benchmark = get_benchmark("libero_object")(0)
task = benchmark.get_task(0)
bddl_path = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(
    bddl_file_name=bddl_path, camera_heights=224, camera_widths=224,
    render_gpu_device_id=-1, has_renderer=False, has_offscreen_renderer=False,
    horizon=500)
env.reset()
model = env.sim.model

site_name = "basket_1_contain_region"
sid = model.site_name2id(site_name)
print("Site:", site_name)
print("  site_pos (local):", [float(x) for x in model.site_pos[sid]])
print("  site_quat (local):", [float(x) for x in model.site_quat[sid]])
print("  parent_body_id:", int(model.site_bodyid[sid]))
print("  parent_body_name:", model.body(model.site_bodyid[sid]).name)
print("  site_size:", [float(x) for x in model.site_size[sid]])

# Also query all region sites for reference
for sn in model.site_names:
    if "_region" in sn:
        s = model.site_name2id(sn)
        print(f"\n{sn}:")
        print(f"  pos={[float(x) for x in model.site_pos[s]]}")
        print(f"  quat={[float(x) for x in model.site_quat[s]]}")
        print(f"  size={[float(x) for x in model.site_size[s]]}")

env.close()
