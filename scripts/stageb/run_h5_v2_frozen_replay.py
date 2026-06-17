#!/usr/bin/env python3
"""H5-V2-C: Frozen replay closed-loop runner. GPU(1,5)."""
import argparse, csv, hashlib, io, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
EPOCH = '/data/liuyu/outputs/l3_h5_v2_candidate_epoch_20260617_r1'
CLEAN_PV_SHA = '7eaaac7d3be8cc5de0171c225998e98c56dc4ff0bf4fdc1964e0d04092293d7d'
CLEAN_ARM = [31971, 31904, 31926, 31882, 31827, 31921]

CID_MAP = {81: {'TRUE': 20, 'RAND': 12, 'SHUFFLED': 6},
           82: {'TRUE': 8, 'RAND': 3, 'SHUFFLED': 13}}

ap = argparse.ArgumentParser()
ap.add_argument('--condition', required=True, choices=['CLEAN','TRUE','RAND','SHUFFLED'])
ap.add_argument('--seed_id', type=int, required=True, choices=[81,82])
ap.add_argument('--output_dir', required=True)
ap.add_argument('--render_gpu', type=int, default=5)
ap.add_argument('--attack_step', type=int, default=60)
ap.add_argument('--max_steps', type=int, default=400)
args = ap.parse_args()


def tsha(t):
    b = io.BytesIO(); torch.save(t.detach().cpu(), b)
    return hashlib.sha256(b.getvalue()).hexdigest()


# Load model
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    device_map='auto', max_memory={idx: '10000MiB' for idx in range(visible)} | {'cpu': '128GiB'},
    attn_implementation='eager')
model_dtype = next(model.parameters()).dtype
device = 'cuda:0'
for v in model.hf_device_map.values():
    if isinstance(v, int): device = f'cuda:{v}'; break
action_dim = int(model.get_action_dim('libero_object'))

# Load frozen tensor if attack condition
adv_pv = None
if args.condition != 'CLEAN':
    cond_lower = args.condition.lower()
    cid = CID_MAP[args.seed_id][args.condition]
    sd = Path(EPOCH) / f'seed{args.seed_id}'
    adv_pv = torch.load(sd / f'{cond_lower}_cand{cid}_adv_pv.pt', map_location='cpu', weights_only=True)
    clean_pv_ref = torch.load(sd / 'clean_pixel_values.pt', map_location='cpu', weights_only=True)
    print(f'Loaded frozen {args.condition}: seed={args.seed_id} id={cid}')

# Replay butter_s11
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(6); init_states = suite.get_task_init_states(6)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, args.max_steps, 10)
obs = env.set_init_state(init_states[11])
env, obs = apply_dummy_wait(env, obs, 10)

telemetry = []
attack_applied = False
decision_record = {}

for step in range(args.max_steps):
    if 'agentview_image' not in obs: break
    raw = np.asarray(obs['agentview_image']).copy()

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    qpos = float(np.sum(gs['qpos'])) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[]))>0 else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

    try:
        width = float(env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.actuator_trnid[7,0]]])
    except Exception:
        width = float('nan')

    try:
        obj_id = env.sim.model.body_name2id('butter_1')
        obj_xyz = env.sim.data.body_xpos[obj_id]
        obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    except Exception:
        obj_x = obj_y = obj_z = float('nan')

    t0 = time.perf_counter()

    if step == args.attack_step and args.condition != 'CLEAN' and adv_pv is not None:
        proc_image = prepare_openvla_image_for_attack(
            raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224)
        inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
        inputs.pop('attention_mask', None)
        input_ids = inputs['input_ids'].to(device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
        clean_pv_current = inputs['pixel_values'].to(device=device, dtype=model_dtype)

        current_sha = tsha(clean_pv_current)
        if current_sha != CLEAN_PV_SHA:
            print(f'FATAL: Clean PV SHA mismatch! current={current_sha[:16]} expected={CLEAN_PV_SHA[:16]}')
            sys.exit(1)

        adv_pv_dev = adv_pv.to(device=device, dtype=model_dtype)
        with torch.inference_mode():
            gen_out = model.generate(input_ids=input_ids, pixel_values=adv_pv_dev,
                max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
        grip = int(tokens[-1])
        # Token-to-action decode
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        discretized = np.clip(vocab_size - np.array([int(t) for t in tokens]) - 1, 0, model.bin_centers.shape[0] - 1)
        norm_actions = model.bin_centers[discretized]
        s = model.get_action_stats('libero_object')
        lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
        mask = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
        action = np.where(mask, 0.5*(norm_actions+1)*(hi-lo)+lo, norm_actions).astype(np.float32)
        env_action = postprocess_openvla_action_for_libero(action, enabled=True)
        attack_applied = True
        arm_match = sum(1 for a,b in zip(list(tokens[:6]), CLEAN_ARM) if a==b)
        decision_record = {
            'step': step, 'condition': args.condition, 'seed': args.seed_id,
            'clean_pv_sha': current_sha, 'clean_pv_sha_ok': current_sha == CLEAN_PV_SHA,
            'candidate_id': CID_MAP[args.seed_id][args.condition],
            'official_tokens': [int(t) for t in tokens], 'gripper_token': grip,
            'arm_match': arm_match, 'attack_applied': True,
        }
        print(f'ATTACK: token={grip} arm={arm_match}/6')
    else:
        action, _scores, _dt, _gen = decode_with_scores(
            model, processor, device, raw, instruction, 'libero_object', 8,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
        env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    t_vla = time.perf_counter() - t0

    raw_grip = float(env_action[-1]) if hasattr(env_action, '__getitem__') else float('nan')
    env_grip = -1.0 if raw_grip > 0.5 else 1.0

    telemetry.append({
        'step': step, 'condition': args.condition, 'seed': args.seed_id,
        'raw_gripper': raw_grip, 'env_gripper': env_grip,
        'gripper_qpos': qpos, 'gripper_width': width,
        'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
        'obj_x': obj_x, 'obj_y': obj_y, 'obj_z': obj_z,
        'attack_applied': attack_applied, 'model_ms': round(t_vla*1000, 2),
    })

    obs, reward, done, info = env.step(env_action)
    if done: break

success = bool(env.check_success()) if hasattr(env, 'check_success') else False
env.close()

out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / 'step_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
with open(out / 'decision_record.json', 'w') as f:
    json.dump(decision_record, f, indent=2)
summary = {'condition': args.condition, 'seed': args.seed_id, 'n_steps': len(telemetry),
           'attack_applied': attack_applied, 'task_success': success}
with open(out / 'episode_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'DONE: steps={len(telemetry)} attack={attack_applied} success={success}')
