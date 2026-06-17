#!/usr/bin/env python3
"""D5-triggered temporal VIS: K=10 state-conditioned PGD with prev_delta warm start."""
import argparse, csv, hashlib, io, json, os, sys, time, numpy as np, torch, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts')); sys.path.insert(0, str(REPO / 'scripts' / 'stageb'))

os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
D5_CKPT = '/data/liuyu/outputs/d5_training/d5_candidate_best.pt'
D5_CFG = '/data/liuyu/outputs/d5_training/d5_frozen_config.json'

K_WINDOW = 10; EPSILON = 0.023529411764705882  # 6/255
TARGET_TOKEN = 31744; ARM_GATE = 5; NUM_STEPS = 20

TASK_IDX = {'butter': 6, 'ketchup': 4}
PARAMS = {
    'butter': {'state': 11, 'obj_site': 'butter_1_default_site'},
    'ketchup': {'state': 18, 'obj_site': 'ketchup_1_default_site'},
}

ap = argparse.ArgumentParser()
ap.add_argument('--parent', required=True, choices=['butter','ketchup'])
ap.add_argument('--condition', required=True,
                choices=['CLEAN_D5','TRUE_SINGLE','TRUE_TEMPORAL_K10',
                         'RAND_TEMPORAL_K10','SHUFFLED_TEMPORAL_K10','COMMAND_HOLD_K10'])
ap.add_argument('--seed_id', type=int, default=81)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--render_gpu', type=int, default=5)
args = ap.parse_args()

p = PARAMS[args.parent]; tidx = TASK_IDX[args.parent]


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
print(f'Model on {device}')

# D5 detector
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
detector = D5FrozenOnlineDetectorV1(D5_CKPT, D5_CFG)
detector.reset()

# Attack config (frozen V4 contract)
attack_cfg = {
    'attack_optimizer': {
        'method': 'token_prefix_pgd', 'objective': 'autoregressive_prefix_gripper_target_token_logratio_arm_v3',
        'target_token_id': TARGET_TOKEN, 'epsilon': EPSILON, 'num_steps': NUM_STEPS,
        'step_size': EPSILON * 0.075, 'random_start': True, 'prefix_refresh_interval': 1,
        'surrogate_score_path': 'cached_autoregressive_generate_v1',
        'gripper_margin': 5.0, 'arm_preserve_weight': 0.5, 'arm_gate_min_match_count': ARM_GATE,
        'strict_route': True, 'allow_fallback': False,
    },
    'preprocess': {'libero_official_preprocess': False, 'libero_preprocess_backend': 'official_pil_lanczos',
                   'center_crop': True, 'resize_size': 224},
}

# Replay
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(tidx); init_states = suite.get_task_init_states(tidx)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[p['state']])
env, obs = apply_dummy_wait(env, obs, 10)

obj_sid = env.sim.model.site_name2id(p['obj_site'])
obj_z0 = float(env.sim.data.site_xpos[obj_sid][2])

telemetry = []; d5_emit_step = -1; d5_triggered = False
window_active = False; window_step = 0; prev_delta = None
is_temporal = 'TEMPORAL' in args.condition
is_rand = 'RAND' in args.condition
is_shuffled = 'SHUFFLED' in args.condition
is_single = args.condition == 'TRUE_SINGLE'
is_attack = is_temporal or is_single or is_rand or is_shuffled
is_cmd_hold = args.condition == 'COMMAND_HOLD_K10'

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
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])

    # Clean decode first
    t0 = time.perf_counter()
    action, _, _, _ = decode_with_scores(
        model, processor, device, raw, instruction, 'libero_object', 8,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)

    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0
    env_action_clean = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

    # D5 update
    detector.update(step, raw_grip, env_grip, qpos_sum if not np.isnan(qpos_sum) else float('nan'),
                    eef_x, eef_y, eef_z, 1 if raw_grip>0.5 else 0,
                    raw_valid=True, env_valid=True, qpos_valid=not np.isnan(qpos_sum), eef_valid=True)

    attack_this = False; adv_tokens = None; adv_arm = 0
    d5_score = detector.audit_records[-1].get('score', 0) if detector.audit_records else 0

    # D5 trigger
    if not d5_triggered and detector.emit_step >= 0 and is_attack:
        d5_triggered = True; d5_emit_step = detector.emit_step
        if is_temporal:
            window_active = True; window_step = 0; prev_delta = None
        print(f'D5 EMIT={d5_emit_step} window_active={window_active}')

    # Command hold proxy
    if is_cmd_hold and ((d5_triggered and window_step < K_WINDOW) or (not d5_triggered and step >= PARAMS[args.parent].get('anchor_approx', 60))):
        if not d5_triggered: d5_triggered = True; d5_emit_step = step; window_active = True; window_step = 0
        if window_step < K_WINDOW:
            env_action_clean[-1] = -1.0
            attack_this = True
            window_step += 1
        else:
            window_active = False

    # Temporal/single attack
    elif window_active and is_attack and window_step < (1 if is_single else K_WINDOW):
        from gripper_attack.attack_adapter import OpenVLAVisualAttacker
        from gripper_attack.route_contract import route_config_from_attack_config, validate_true_pgd_attack_result

        opt = dict(attack_cfg['attack_optimizer'])
        if not is_single and window_step == 0:
            opt['random_start'] = True
        elif not is_single and prev_delta is not None:
            opt['random_start'] = False  # use prev_delta warm start

        if is_shuffled:
            opt['gradient_transform'] = 'permute'; opt['gradient_transform_seed'] = args.seed_id + 100000 + window_step

        attacker = OpenVLAVisualAttacker(
            model=model, processor=processor, config={'attack_optimizer': opt},
            seed=args.seed_id + window_step, preprocess_kwargs=dict(attack_cfg.get('preprocess', {})), device=device)

        clean_gen = type('CleanGen', (), {})()
        proc_image = prepare_openvla_image_for_attack(
            raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224)
        inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
        inputs.pop('attention_mask', None)
        input_ids = inputs['input_ids'].to(device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)

        # Get clean tokens for this frame
        with torch.inference_mode():
            go = model.generate(input_ids=input_ids, pixel_values=inputs['pixel_values'].to(device=device, dtype=model_dtype),
                max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
        clean_gen.sequences = torch.tensor([input_ids[0].detach().cpu().tolist() + [int(t) for t in clean_tokens]],
                                           dtype=torch.long, device=device)
        clean_gen.scores = []

        if is_rand:
            # RAND: sample random direction
            from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
            x = inputs['pixel_values'].to(device=device, dtype=model_dtype)
            delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=args.seed_id+100000+window_step,
                                           dtype=torch.float32, device=x.device)
            proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
            adv_pv = proj.detach().to(dtype=model_dtype)
            with torch.inference_mode():
                go2 = model.generate(input_ids=input_ids, pixel_values=adv_pv,
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
            adv_tokens = extract_exact_new_tokens(go2.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
            prev_delta = (adv_pv.float() - x.float()).detach()
        else:
            # TRUE/SHUFFLED: PGD
            attack_result = attacker.attack(
                raw, instruction, np.asarray(action, dtype=np.float32), np.asarray(action, dtype=np.float32),
                clean_gen, unnorm_key='libero_object')
            validate_true_pgd_attack_result(attack_result, route_config_from_attack_config({'attack_optimizer': opt}))
            debug = attack_result.debug
            adv_tokens_list = debug.get('adv_logit_audit', {}).get('official_tokens')
            if adv_tokens_list:
                adv_tokens = [int(t) for t in adv_tokens_list] if isinstance(adv_tokens_list[0], (int, np.integer)) else None
            # Get adv inputs
            from gripper_attack.attack_adapter import get_adv_inputs_from_attack_result
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)
            adv_pv = adv_inputs['pixel_values']
            prev_delta = (adv_pv.float() - inputs['pixel_values'].to(device=device, dtype=torch.float32)).detach()

        # Decode adversarial action
        if adv_tokens is None:
            with torch.inference_mode():
                go_adv = model.generate(input_ids=input_ids, pixel_values=adv_pv.to(device=device, dtype=model_dtype),
                    max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                adv_tokens = extract_exact_new_tokens(go_adv.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)

        grip = int(adv_tokens[-1])
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        disc = np.clip(vocab_size - np.array([int(t) for t in adv_tokens]) - 1, 0, model.bin_centers.shape[0]-1)
        na = model.bin_centers[disc]
        s = model.get_action_stats('libero_object')
        lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
        mk = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
        attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
        env_action_clean = postprocess_openvla_action_for_libero(attack_action, enabled=True)
        raw_grip = float(attack_action[-1]); env_grip = float(env_action_clean[-1])
        adv_arm = sum(1 for a,b in zip(list(adv_tokens[:6]), list(clean_tokens[:6])) if a==b)
        attack_this = True; window_step += 1
        if is_single or window_step >= (1 if is_single else K_WINDOW):
            window_active = False
    else:
        window_active = False

    t_vla = time.perf_counter()-t0

    telemetry.append({
        'step': step, 'condition': args.condition, 'seed': args.seed_id,
        'raw_gripper': raw_grip, 'env_gripper': env_grip,
        'qpos_sum': qpos_sum, 'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
        'obj_x': obj_x, 'obj_y': obj_y, 'obj_z': obj_z,
        'd5_emit': d5_emit_step, 'd5_triggered': d5_triggered,
        'window_active': window_active, 'window_step': window_step,
        'attack_this': attack_this, 'adv_token': grip if attack_this else '',
        'adv_arm': adv_arm if attack_this else '', 'd5_score': d5_score,
        'model_ms': round(t_vla*1000, 2),
    })

    obs, _, done, _ = env.step(env_action_clean)
    if done: break

success = bool(env.check_success()) if hasattr(env, 'check_success') else False
env.close()

out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / 'step_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)

# Duty cycle metrics
atk_steps = [r for r in telemetry if r['attack_this'] == 'True' or r['attack_this'] == True]
n_atk = len(atk_steps)
n_open = sum(1 for r in atk_steps if float(r['env_gripper']) < 0)
n_arm_ok = sum(1 for r in atk_steps if str(r.get('adv_arm','')) != '' and int(r['adv_arm']) >= ARM_GATE)

duty = {'n_attack_steps': n_atk, 'n_open': n_open, 'n_arm_ok': n_arm_ok,
        'open_duty': round(n_open/n_atk, 3) if n_atk>0 else 0,
        'arm_duty': round(n_arm_ok/n_atk, 3) if n_atk>0 else 0}

summary = {'parent': args.parent, 'condition': args.condition, 'seed': args.seed_id,
           'n_steps': len(telemetry), 'd5_emit': d5_emit_step, 'success': success, **duty}
with open(out / 'episode_summary.json', 'w') as f: json.dump(summary, f, indent=2)

print(f'{args.condition}: steps={len(telemetry)} d5_emit={d5_emit_step} '
      f'atk_steps={n_atk} open={n_open} arm_ok={n_arm_ok} duty={duty["open_duty"]:.2f} success={success}')
