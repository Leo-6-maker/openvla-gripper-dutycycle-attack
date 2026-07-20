#!/usr/bin/env python3
"""R10.4E 10-task passive panel — OpenVLA loaded once, 9 new episodes, all state_20."""
import sys, json, os
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/src")
sys.path.insert(0, "/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/scripts/r10_4")
import torch; torch.set_grad_enabled(False)

# PATCH: authorize all task_*/state_20 for 2-hour window (Issue #88 comment 5018641254)
import gripper_attack.r10_4d_passive as _m
_orig_run = _m.run_passive_episode
def _patched_run(**kw):
    ident = kw.get("identity", "")
    if not (ident.endswith("/state_20") and ident.startswith("libero_10/")):
        raise Exception("PANEL_NOT_AUTHORIZED:" + ident)
    _m.SUPPORTED_PARENT = ident
    return _orig_run(**kw)
_m.run_passive_episode = _patched_run

from run_r10_4d_passive_smoke import load_openvla
from gripper_attack.libero_v4_env_factory import build_v4_exact_env
from experiments.robot.libero.libero_utils import get_libero_image
from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
from libero.libero import benchmark, get_libero_path
from pathlib import Path

dev = torch.device("cuda:0")
rec = json.loads(open("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_auth_receipt_20260720").read())
BUNDLE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")
MODEL = Path("/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10")
det, _ = _m.load_detector_bundle(BUNDLE, device=dev,
    expected_checkpoint_sha256=rec["detector_checkpoint_sha256"],
    expected_bundle_sha256s=rec["bundle_sha256s_sha256"])
model, proc, md, uk = load_openvla(MODEL, "libero_10")
adapter = OfficialOpenVLAActionAdapter(model=model, processor=proc, device=md, unnorm_key=uk)
print("Ready. Panel phase E.")
img = lambda obs: get_libero_image(obs, 224)

cons = benchmark.get_benchmark_dict()
si = cons["libero_10"]()
OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_passive_panel_20260720")
OUT.mkdir(parents=True, exist_ok=True)
results = []; ok = True

SKIP_TASKS = {0, 1, 2}  # 0=R10.4D, 1=already done, 2=early-termination
for ti in range(10):
    parent = f"libero_10/task_{ti:02d}/state_20"
    if ti in SKIP_TASKS:
        if ti == 0:
            ex = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_passive_smoke_output_20260720/episode_summary.json")
            r = json.loads(ex.read_text()) if ex.is_file() else {"identity":parent,"status":"MISSING"}
        elif ti == 1:
            ex = OUT / "libero_10_task_01_state_20" / "episode_summary.json"
            r = json.loads(ex.read_text()) if ex.is_file() else {"identity":parent,"status":"MISSING"}
        else:
            r = {"identity":parent,"status":"SKIPPED_EARLY_TERMINATION"}
        results.append(r)
        print(f"[{ti+1}/10] {parent} SKIP ({r.get('status','?')})")
        continue
    print(f"[{ti+1}/10] {parent} RUNNING...", flush=True)
    task = si.get_task(ti); states = si.get_task_init_states(ti); istate = states[20]
    bddl = Path(str(get_libero_path("bddl_files"))) / str(task.problem_folder) / str(task.bddl_file)
    env, obs = build_v4_exact_env(str(bddl), render_gpu_device_id=0, max_steps=520)
    obs = env.set_init_state(istate)
    for _ in range(10): obs, _, _, _ = env.step([0,0,0,0,0,0,-1])
    r = _patched_run(env=env, initial_state=istate, task_language=str(task.language),
        identity=parent, openvla_adapter=adapter, detector=det, image_getter=img, max_steps=520)
    env.close(); results.append(r)
    vi = r.get("violations", [])
    print(f"  {r.get('status','?')} steps={r.get('n_steps',0)} emits={r.get('emit_count',0)} vi={len(vi)}")
    if vi: print(f"  VIOLATIONS: {vi}"); ok = False; break

summary = {"panel":"R10_4E","n":len(results),"results":results,"all_pass":ok}
with open(OUT/"panel_summary.json","w") as f: json.dump(summary,f,indent=2)
print(f"Panel: {len(results)} eps, all_pass={ok}")
