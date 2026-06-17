#!/usr/bin/env python3
"""H6: D5 online detector-triggered closed-loop POC on GPU(1,5)."""
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
D5_CKPT = '/data/liuyu/outputs/d5_training/d5_candidate_best.pt'
D5_CFG = '/data/liuyu/outputs/d5_training/d5_frozen_config.json'

CID_MAP = {81: {'TRUE': 20, 'RAND': 12, 'SHUFFLED': 6}}
EXPECTED = {
    ('TRUE', 20): {'token': 31744, 'min_arm': 5},
    ('RAND', 12): {'token': 31872, 'min_arm': 6},
    ('SHUFFLED', 6): {'token': 31872, 'min_arm': 6},
}

ap = argparse.ArgumentParser()
ap.add_argument('--condition', required=True, choices=['CLEAN','TRUE','RAND','SHUFFLED'])
ap.add_argument('--output_dir', required=True)
ap.add_argument('--render_gpu', type=int, default=5)
args = ap.parse_args()

SEED = 81; MAX_STEPS = 400


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

# Load D5 online detector
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
detector = D5FrozenOnlineDetectorV1(D5_CKPT, D5_CFG)
detector.reset()

# Load frozen tensor
adv_pv = None; cond_meta = None
if args.condition != 'CLEAN':
    cid = CID_MAP[SEED][args.condition]
    cl = args.condition.lower()
    sd = Path(EPOCH) / f'seed{SEED}'
    adv_pv = torch.load(sd / f'{cl}_cand{cid}_adv_pv.pt', map_location='cpu', weights_only=True)
    clean_ref = torch.load(sd / 'clean_pixel_values.pt', map_location='cpu', weights_only=True)
    delta_ref = torch.load(sd / f'{cl}_cand{cid}_delta.pt', map_location='cpu', weights_only=True)
    cond_meta = {'adv_pv': adv_pv, 'adv_sha': tsha(adv_pv), 'delta_sha': tsha(delta_ref),
                 'clean_sha': tsha(clean_ref), 'cid': cid, 'linf': delta_ref.float().abs().max().item()}

# Replay
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(6); init_states = suite.get_task_init_states(6)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, MAX_STEPS, 10)
obs = env.set_init_state(init_states[11])
env, obs = apply_dummy_wait(env, obs, 10)

obj_sid = env.sim.model.site_name2id('butter_1_default_site')
obj_z0 = float(env.sim.data.site_xpos[obj_sid][2])

telemetry = []; decision = None
d5_triggered = False; d5_emit_step = -1; attack_ever = False

for step in range(MAX_STEPS):
    if 'agentview_image' not in obs: break
    raw = np.asarray(obs['agentview_image']).copy()

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs['qpos'][0]) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[]))>0 else float('nan')
    q8 = float(gs['qpos'][1]) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[]))>1 else float('nan')
    qpos_sum = q7+q8 if not (np.isnan(q7) or np.isnan(q8)) else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2+(eef_y-obj_y)**2+(eef_z-obj_z)**2))

    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(
        model, processor, device, raw, instruction, 'libero_object', 8,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)

    raw_grip = float(action[-1])
    env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_clean = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # D5 online detector
    detector.update(step, raw_grip, env_grip, qpos_sum if not np.isnan(qpos_sum) else float('nan'),
                    eef_x, eef_y, eef_z, 1 if raw_grip>0.5 else 0,
                    raw_valid=True, env_valid=True, qpos_valid=not np.isnan(qpos_sum), eef_valid=True)

    attack_this_step = False
    if not d5_triggered and detector.emit_step >= 0:
        d5_triggered = True; d5_emit_step = detector.emit_step
        if args.condition != 'CLEAN' and cond_meta is not None:
            proc_image = prepare_openvla_image_for_attack(
                raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
                center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
            inputs.pop('attention_mask', None)
            input_ids = inputs['input_ids'].to(device)
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
            clean_pv_current = inputs['pixel_values'].to(device=device, dtype=model_dtype)
            if tsha(clean_pv_current) != CLEAN_PV_SHA:
                print(f'FATAL: clean PV SHA mismatch at D5 emit')
                sys.exit(1)
            adv_pv_dev = cond_meta['adv_pv'].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                gen_out = model.generate(input_ids=input_ids, pixel_values=adv_pv_dev,
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
            grip = int(tokens[-1]); arm_match = sum(1 for a,b in zip(list(tokens[:6]), CLEAN_ARM) if a==b)
            vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
            disc = np.clip(vocab_size-np.array([int(t) for t in tokens])-1, 0, model.bin_centers.shape[0]-1)
            na = model.bin_centers[disc]
            s = model.get_action_stats('libero_object')
            lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
            mk = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
            attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
            env_action_clean = postprocess_openvla_action_for_libero(attack_action, enabled=True)
            raw_grip = float(attack_action[-1]); env_grip = float(env_action_clean[-1])
            attack_this_step = True; attack_ever = True
            exp = EXPECTED[(args.condition, cond_meta['cid'])]
            decision = {
                'trigger': 'D5_ONLINE', 'd5_emit': d5_emit_step, 'd5_version': 'd5_frozen_online_v1',
                'condition': args.condition, 'cid': cond_meta['cid'],
                'clean_pv_ok': True, 'token': grip, 'arm': f'{arm_match}/6',
                'raw_grip': raw_grip, 'env_grip': env_grip,
                'cmd': 'OPEN' if raw_grip>0.5 else 'CLOSE',
            }
            print(f'D5 EMIT={d5_emit_step}: token={grip} arm={arm_match}/6')

    t_vla = time.perf_counter()-t0
    open_frac = (0.0056-qpos_sum)/0.0112 if not np.isnan(qpos_sum) else float('nan')

    telemetry.append({
        'step': step, 'condition': args.condition, 'seed': SEED,
        'raw_gripper': raw_grip, 'env_gripper': env_grip,
        'qpos_7': q7, 'qpos_8': q8, 'qpos_sum': qpos_sum, 'open_frac': open_frac,
        'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
        'obj_x': obj_x, 'obj_y': obj_y, 'obj_z': obj_z, 'eef_obj_dist': eef_obj_dist,
        'd5_triggered': d5_triggered, 'attack_this': attack_this_step,
        'attack_ever': attack_ever, 'model_ms': round(t_vla*1000, 2),
    })

    obs, _, done, _ = env.step(env_action_clean)
    if done: break

success = bool(env.check_success()) if hasattr(env, 'check_success') else False
env.close()

out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / 'step_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
if decision:
    with open(out / 'decision_record.json', 'w') as f: json.dump(decision, f, indent=2)
summary = {'condition': args.condition, 'seed': SEED, 'n_steps': len(telemetry),
           'd5_emit': d5_emit_step, 'd5_triggered': d5_triggered,
           'attack': attack_ever, 'success': success}
with open(out / 'episode_summary.json', 'w') as f: json.dump(summary, f, indent=2)
print(f'{args.condition}: steps={len(telemetry)} d5_emit={d5_emit_step} attack={attack_ever} success={success}')
