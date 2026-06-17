#!/usr/bin/env python3
"""Phase 3 pilot: single-process — loads model once, runs all conditions, frees between.

Reuses v1 model loading and env setup exactly.
"""
import csv, gc, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

MANIFEST_PATH = "/data/liuyu/audit_v3/v2_phase3_anchor_manifest.json"
OUT = "/data/liuyu/outputs/v2_phase3_pilot"
K = 10; TASK_IDX = 6; NUM_WAIT = 10; MAX_STEPS = 400

state_id = int(sys.argv[1])
render_gpu = int(sys.argv[2])

m = json.load(open(MANIFEST_PATH))
sk = "state_%d" % state_id
sd = m["states"][sk]
windows = sd["windows"]

# Build condition list
conditions = [("CLEAN", 0)]
for wname, cond in [("W0_D5", "HOLD_D5_K10"), ("W1_SG5", "HOLD_SG5_K10"),
                     ("W2_SC5", "HOLD_SC5_K10"), ("W3_MID", "HOLD_MID_K10")]:
    wi = windows[wname]
    if wi["valid"]:
        conditions.append((cond, wi["anchor"]))
    else:
        print("SKIP %s: %s" % (cond, wi.get("reason", "invalid")))

# --- Load model once (identical to v1 runner) ---
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()

def load_model():
    model = AutoModelCls.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto', max_memory={idx: '10000MiB' for idx in range(visible)} | {'cpu': '128GiB'},
        attn_implementation='eager')
    model_dtype = next(model.parameters()).dtype
    device = 'cuda:0'
    for v in model.hf_device_map.values():
        if isinstance(v, int): device = f'cuda:{v}'; break
    return model, model_dtype, device

# --- Imports (same as v1) ---
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(TASK_IDX); init_states = suite.get_task_init_states(TASK_IDX)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

results = {}

for cond, anchor in conditions:
    print("=== %s s%d anchor=%d ===" % (cond, state_id, anchor))
    d = os.path.join(OUT, "s%d" % state_id, cond)
    os.makedirs(d, exist_ok=True)

    # Load model fresh for each condition (freed after)
    t0 = time.time()
    try:
        model, model_dtype, device = load_model()
    except torch.cuda.OutOfMemoryError:
        print("  OOM during model load — skipping remaining")
        break

    action_dim = int(model.get_action_dim('libero_object'))

    # --- Setup env (same as v1) ---
    env, obs = build_v4_exact_env(bddl, render_gpu, MAX_STEPS, NUM_WAIT)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, NUM_WAIT)

    obj_sid = env.sim.model.site_name2id('butter_1_default_site')
    obj_z0 = float(env.sim.data.site_xpos[obj_sid][2])
    target_sid = env.sim.model.site_name2id('basket_1_default_site')

    telemetry = []
    is_hold = cond != "CLEAN"

    for step in range(MAX_STEPS):
        if 'agentview_image' not in obs: break
        raw = np.asarray(obs['agentview_image']).copy()

        from v4_run_eval_openvla import physical_gripper_state
        gs = physical_gripper_state(env, obs)
        q7 = float(gs['qpos'][0]) if gs and len(gs.get('qpos',[])) > 0 else float('nan')
        q8 = float(gs['qpos'][1]) if gs and len(gs.get('qpos',[])) > 1 else float('nan')
        qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float('nan')

        eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
        eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

        obj_xyz = env.sim.data.site_xpos[obj_sid]
        obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
        target_xyz = env.sim.data.site_xpos[target_sid]
        eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))
        obj_target_dist = float(np.sqrt((obj_x-float(target_xyz[0]))**2 + (obj_y-float(target_xyz[1]))**2 + (obj_z-float(target_xyz[2]))**2))

        # Decode clean action (same as v1)
        t_vla0 = time.perf_counter()
        action, _, _, _ = decode_with_scores(
            model, processor, device, raw, instruction, 'libero_object', 8,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
        clean_raw_grip = float(action[-1])
        clean_env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
        t_vla = time.perf_counter() - t_vla0

        # Command hold
        in_window = is_hold and anchor <= step <= anchor + K - 1
        executed_action = clean_env_action.copy()
        if in_window:
            executed_action[-1] = -1.0

        arm_delta = np.asarray(executed_action[:6]) - np.asarray(clean_env_action[:6])

        telemetry.append({
            'step': step, 'state_id': state_id, 'condition': cond,
            'anchor': anchor, 'in_window': in_window,
            'arm_delta_max': float(np.max(np.abs(arm_delta))),
            'arm_delta_ok': np.max(np.abs(arm_delta)) == 0.0,
            'qpos_sum': qpos_sum,
            'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
            'obj_x': obj_x, 'obj_y': obj_y, 'obj_z': obj_z,
            'obj_z0': obj_z0, 'eef_obj_dist': eef_obj_dist,
            'obj_target_dist': obj_target_dist,
            'model_ms': round(t_vla * 1000, 2),
        })

        obs, _, done, _ = env.step(executed_action)
        if done: break

    success = bool(env.check_success()) if hasattr(env, 'check_success') else False
    env.close()

    # Write outputs
    with open(os.path.join(d, 'step_telemetry.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
        w.writeheader(); w.writerows(telemetry)

    summary = {'condition': cond, 'state_id': state_id, 'anchor': anchor,
               'K': K, 'n_steps': len(telemetry),
               'hold_steps': sum(1 for r in telemetry if r['in_window']),
               'arm_delta_all_zero': all(r['arm_delta_ok'] for r in telemetry),
               'task_success': success}
    with open(os.path.join(d, 'episode_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print("  steps=%d hold=%d arm_ok=%s success=%s elapsed=%.1fs" % (
        len(telemetry), summary['hold_steps'], summary['arm_delta_all_zero'], success, elapsed))

    results[cond] = {'steps': len(telemetry), 'success': success, 'elapsed': elapsed, 'has_csv': True}
    del model; gc.collect(); torch.cuda.empty_cache()

print("\n=== STATE %d COMPLETE ===" % state_id)
with open(os.path.join(OUT, "summary_s%d.json" % state_id), "w") as f:
    json.dump(results, f, indent=2, default=str)
