#!/usr/bin/env python3
"""R10.4E panel — run remaining tasks 04-09 (skip 00-03 already handled)."""
import sys, json, os
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/src")
sys.path.insert(0, "/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/scripts/r10_4")
import torch; torch.set_grad_enabled(False)
from run_r10_4d_passive_smoke import load_openvla
from gripper_attack.libero_v4_env_factory import build_v4_exact_env
from experiments.robot.libero.libero_utils import get_libero_image
from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
from libero.libero import benchmark, get_libero_path
from pathlib import Path
import gripper_attack.r10_4d_passive as _m
dev = torch.device("cuda:0")
rec = json.loads(open("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_auth_receipt_20260720").read())
BUNDLE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")
MODEL = Path("/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10")
det, _ = _m.load_detector_bundle(BUNDLE, device=dev, expected_checkpoint_sha256=rec["detector_checkpoint_sha256"], expected_bundle_sha256s=rec["bundle_sha256s_sha256"])
model, proc, md, uk = load_openvla(MODEL, "libero_10")
adapter = OfficialOpenVLAActionAdapter(model=model, processor=proc, device=md, unnorm_key=uk)
img = lambda obs: get_libero_image(obs, 224)
cons = benchmark.get_benchmark_dict(); si = cons["libero_10"]()
OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_passive_panel_20260720")
print("Ready. Running tasks 04-09.")
for ti in [4, 5, 6, 7, 8, 9]:
    parent = "libero_10/task_%02d/state_20" % ti
    print("[%d/10] %s RUNNING..." % (ti+1, parent), flush=True)
    task = si.get_task(ti); states = si.get_task_init_states(ti); istate = states[20]
    bddl = Path(str(get_libero_path("bddl_files"))) / str(task.problem_folder) / str(task.bddl_file)
    env, obs = build_v4_exact_env(str(bddl), render_gpu_device_id=0, max_steps=520)
    obs = env.set_init_state(istate)
    for _ in range(10): obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
    try:
        r = _m.run_passive_episode(env=env, initial_state=istate, task_language=str(task.language),
            identity=parent, openvla_adapter=adapter, detector=det, image_getter=img, max_steps=520)
        vi = r.get("violations", [])
        print("  %s steps=%d emits=%d vi=%d" % (r.get("status","?"), r.get("n_steps",0), r.get("emit_count",0), len(vi)))
        if vi: print("  VIOLATIONS:", vi)
    except Exception as e:
        print("  FAILED:", e)
    env.close()
print("Done.")
