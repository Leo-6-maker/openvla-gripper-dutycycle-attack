#!/usr/bin/env python3
"""D5-triggered temporal VIS: K=10 steps, persistent attacker with prev_delta.
Fixes: D5 emit for all conditions, TRUE_SINGLE=1 frame, prev_delta warm start,
       q7/q8 separate, full per-frame attack telemetry.
"""
import argparse, csv, hashlib, io, json, os, sys, time, numpy as np, torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
D5_CKPT = '/data/liuyu/outputs/d5_training/d5_candidate_best.pt'
D5_CFG = '/data/liuyu/outputs/d5_training/d5_frozen_config.json'

EPSILON = 0.023529411764705882  # 6/255
TARGET_TOKEN = 31744; ARM_GATE = 5; PGD_STEPS = 20

ap = argparse.ArgumentParser()
ap.add_argument('--condition', required=True,
                choices=['CLEAN_D5','TRUE_SINGLE','TRUE_TEMPORAL_K10',
                         'RAND_TEMPORAL_K10','SHUFFLED_TEMPORAL_K10'])
ap.add_argument('--seed_id', type=int, required=True, choices=[81,82])
ap.add_argument('--output_dir', required=True)
ap.add_argument('--K', type=int, default=10, help='attack window length')
ap.add_argument('--render_gpu', type=int, default=5)
args = ap.parse_args()

K = args.K; IS_TEMPORAL = 'TEMPORAL' in args.condition
IS_SINGLE = args.condition == 'TRUE_SINGLE'
IS_RAND = 'RAND' in args.condition
IS_SHUFFLED = 'SHUFFLED' in args.condition
IS_CLEAN = args.condition == 'CLEAN_D5'
IS_ATTACK = not IS_CLEAN
ATTACK_FRAMES = 1 if IS_SINGLE else (K if IS_TEMPORAL else 0)

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
print(f'Model on {device} K={K} condition={args.condition}')

# D5 detector
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
detector = D5FrozenOnlineDetectorV1(D5_CKPT, D5_CFG)
detector.reset()

# Persistent attacker for temporal window
attacker = None
if IS_ATTACK and not IS_RAND:
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker
    opt = {
        'method': 'token_prefix_pgd', 'objective': 'autoregressive_prefix_gripper_target_token_logratio_arm_v3',
        'target_token_id': TARGET_TOKEN, 'epsilon': EPSILON, 'num_steps': PGD_STEPS,
        'step_size': EPSILON * 0.075, 'random_start': True, 'prefix_refresh_interval': 1,
        'surrogate_score_path': 'cached_autoregressive_generate_v1',
        'gripper_margin': 5.0, 'arm_preserve_weight': 0.5, 'arm_gate_min_match_count': ARM_GATE,
        'strict_route': True, 'allow_fallback': False, 'temporal_init': 'prev_delta',
        'target_execution_class': 'CLIP_MEDIATED_OPEN',
    }
    if IS_SHUFFLED:
        opt['gradient_transform'] = 'permute'; opt['gradient_transform_seed'] = args.seed_id + 100000
    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor, config={'attack_optimizer': opt},
        seed=args.seed_id, preprocess_kwargs={'libero_official_preprocess': False,
            'libero_preprocess_backend': 'official_pil_lanczos', 'center_crop': True, 'resize_size': 224},
        device=device)
    from gripper_attack.route_contract import route_config_from_attack_config
    route = route_config_from_attack_config({'attack_optimizer': opt})
    print(f'Attacker ready: temporal_init=prev_delta')

# Replay Butter
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.v3_generation_parity import extract_exact_new_tokens
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(6); init_states = suite.get_task_init_states(6)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[11])
env, obs = apply_dummy_wait(env, obs, 10)

obj_sid = env.sim.model.site_name2id('butter_1_default_site')

telemetry = []; d5_emit_step = -1; d5_triggered = False
attack_count = 0; prev_delta_used_flags = []

for step in range(400):
    if 'agentview_image' not in obs: break
    raw = np.asarray(obs['agentview_image']).copy()

    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs['qpos'][0]) if gs and len(gs.get('qpos',[]))>0 else float('nan')
    q8 = float(gs['qpos'][1]) if gs and len(gs.get('qpos',[]))>1 else float('nan')
    qpos_sum = q7+q8 if not (np.isnan(q7) or np.isnan(q8)) else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

    # Clean decode
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(
        model, processor, device, raw, instruction, 'libero_object', 8,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # D5 update (always)
    detector.update(step, raw_grip, env_grip, qpos_sum if not np.isnan(qpos_sum) else float('nan'),
                    eef_x, eef_y, eef_z, 1 if raw_grip>0.5 else 0,
                    raw_valid=True, env_valid=True, qpos_valid=not np.isnan(qpos_sum), eef_valid=True)
    d5_score = detector.audit_records[-1].get('score', 0) if detector.audit_records else 0

    # Record D5 emit (always)
    if d5_emit_step < 0 and detector.emit_step >= 0:
        d5_emit_step = detector.emit_step

    # Attack trigger
    attack_this = False; adv_token = None; adv_arm = 0; prev_flag = False
    if IS_ATTACK and d5_emit_step >= 0 and attack_count < ATTACK_FRAMES and step >= d5_emit_step:
        if step == d5_emit_step and attack_count == 0:
            d5_triggered = True

        if IS_RAND:
            # Single random direction per frame
            from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
            from PIL import Image
            proc_image = prepare_openvla_image_for_attack(
                raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
                center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
            inputs.pop('attention_mask', None)
            iids = inputs['input_ids'].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            x = inputs['pixel_values'].to(device=device, dtype=model_dtype)
            delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=args.seed_id+100000+attack_count,
                                           dtype=torch.float32, device=x.device)
            proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
            adv_pv = proj.detach().to(dtype=model_dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=adv_pv,
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
        else:
            # TRUE/SHUFFLED: PGD with persistent attacker
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
            from gripper_attack.route_contract import validate_true_pgd_attack_result
            clean_action_np = np.asarray(action, dtype=np.float32)

            # Build clean_gen for this frame
            from PIL import Image
            proc_image = prepare_openvla_image_for_attack(
                raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
                center_crop=True, resize_size=224)
            inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
            inputs.pop('attention_mask', None)
            iids = inputs['input_ids'].to(device)
            if not torch.all(iids[:, -1] == 29871):
                iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
            pv = inputs['pixel_values'].to(device=device, dtype=model_dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=pv,
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            clean_gen = type('CleanGen', (), {})()
            clean_gen.sequences = torch.tensor([iids[0].detach().cpu().tolist() + [int(t) for t in clean_tokens]],
                                               dtype=torch.long, device=device)
            clean_gen.scores = []

            attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen, unnorm_key='libero_object')
            validate_true_pgd_attack_result(attack_result, route)
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)
            adv_pv = adv_inputs['pixel_values']
            # Get tokens
            with torch.inference_mode():
                go_adv = model.generate(input_ids=iids, pixel_values=adv_pv.to(device=device, dtype=model_dtype),
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            adv_tokens = extract_exact_new_tokens(go_adv.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
            # Check prev_delta flag
            prev_flag = attack_result.debug.get('temporal_prev_delta_used', False) if hasattr(attack_result, 'debug') else False

        grip = int(adv_tokens[-1])
        # Action decode
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        disc = np.clip(vocab_size - np.array([int(t) for t in adv_tokens]) - 1, 0, model.bin_centers.shape[0]-1)
        na = model.bin_centers[disc]
        s = model.get_action_stats('libero_object')
        lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
        mk = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
        attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
        env_action_final = postprocess_openvla_action_for_libero(attack_action, enabled=True)
        raw_grip = float(attack_action[-1]); env_grip = float(env_action_final[-1])
        adv_arm = sum(1 for a,b in zip(list(adv_tokens[:6]), list(clean_tokens[:6])) if a==b) if 'clean_tokens' in dir() else 0
        attack_this = True; attack_count += 1
        prev_delta_used_flags.append(prev_flag)
        adv_token = grip

    t_vla = time.perf_counter()-t0

    telemetry.append({
        'step': step, 'condition': args.condition, 'seed': args.seed_id,
        'raw_gripper': raw_grip, 'env_gripper': env_grip,
        'q7': q7, 'q8': q8, 'qpos_sum': qpos_sum,
        'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
        'd5_emit_step': d5_emit_step, 'd5_score': d5_score,
        'attack_count': attack_count, 'attack_this': attack_this,
        'adv_token': adv_token if adv_token else '', 'adv_arm': adv_arm if attack_this else '',
        'prev_delta_used': prev_flag,
        'model_ms': round(t_vla*1000, 2),
    })

    obs, _, done, _ = env.step(env_action_final)
    if done: break

success = bool(env.check_success()) if hasattr(env, 'check_success') else False
env.close()

# ── Duty cycle metrics ──
atk_rows = [r for r in telemetry if r['attack_this'] == True or r['attack_this'] == 'True']
n_atk = len(atk_rows)
n_open_token = sum(1 for r in atk_rows if str(r.get('adv_token','')) != '' and int(r['adv_token']) == TARGET_TOKEN)
n_arm_ok = sum(1 for r in atk_rows if str(r.get('adv_arm','')) != '' and int(r['adv_arm']) >= ARM_GATE)
n_env_open = sum(1 for r in atk_rows if float(r['env_gripper']) < 0)

# Physical metrics
pre_steps = [r for r in telemetry if d5_emit_step-5 <= int(r['step']) < d5_emit_step] if d5_emit_step >= 0 else []
post_steps = [r for r in telemetry if d5_emit_step <= int(r['step']) < d5_emit_step+30] if d5_emit_step >= 0 else []
baseline_qpos = np.mean([float(r['qpos_sum']) for r in pre_steps if str(r['qpos_sum']) not in ('nan','')]) if pre_steps else float('nan')
post_qpos = [(int(r['step']), float(r['qpos_sum'])) for r in post_steps if str(r['qpos_sum']) not in ('nan','')]
peak = min(post_qpos, key=lambda x: x[1]) if post_qpos else (0, float('nan'))
mean_post = np.mean([q for _,q in post_qpos]) if post_qpos else float('nan')
peak_delta = baseline_qpos - peak[1] if not np.isnan(baseline_qpos) and not np.isnan(peak[1]) else float('nan')

summary = {
    'parent': 'butter_s11', 'condition': args.condition, 'seed': args.seed_id, 'K': K,
    'n_steps': len(telemetry), 'd5_emit_step': d5_emit_step, 'd5_triggered': d5_triggered,
    'attack_frames': n_atk, 'open_tokens': n_open_token, 'arm_ok_frames': n_arm_ok,
    'env_open_frames': n_env_open,
    'token_open_duty': round(n_open_token/n_atk, 3) if n_atk>0 else 0,
    'arm_duty': round(n_arm_ok/n_atk, 3) if n_atk>0 else 0,
    'env_open_duty': round(n_env_open/n_atk, 3) if n_atk>0 else 0,
    'baseline_qpos': baseline_qpos, 'peak_qpos': peak[1], 'mean_post_qpos': mean_post,
    'peak_delta': peak_delta,
    'prev_delta_flags': prev_delta_used_flags,
    'task_success': success,
}
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / 'step_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
with open(out / 'episode_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'{args.condition}: steps={len(telemetry)} d5_emit={d5_emit_step} '
      f'atk={n_atk} token_duty={summary["token_open_duty"]:.2f} '
      f'env_duty={summary["env_open_duty"]:.2f} peak_delta={peak_delta:.5f} success={success}')
