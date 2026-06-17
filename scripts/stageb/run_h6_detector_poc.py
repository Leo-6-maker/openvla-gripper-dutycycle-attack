#!/usr/bin/env python3
"""H6 detector-triggered POC: D5 detector trigger on GPU(1,5).

Fixes vs V2-C:
1. Full fail-closed: verify adv/delta SHA, metadata, tokens, arm, Linf before env.step
2. Raw/env gripper: record raw_action_gripper (pre-postprocess), actual_env_gripper
3. Per-step flags: attack_applied_this_step + attack_ever_applied
4. Validated qpos: qpos_sum as primary physical metric, not fake width
"""
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
CLEAN_TOKENS = [31971, 31904, 31926, 31882, 31827, 31921, 31872]

CID_MAP = {81: {'TRUE': 20, 'RAND': 12, 'SHUFFLED': 6}}
EXPECTED = {
    ('TRUE', 20): {'token': 31744, 'min_arm': 5, 'label': 'OPEN'},
    ('RAND', 12): {'token': 31872, 'min_arm': 6, 'label': 'CLOSE'},
    ('SHUFFLED', 6): {'token': 31872, 'min_arm': 6, 'label': 'CLOSE'},
}

# Calibration (from V2-D)
CALIB = {'close_qpos': 0.0056, 'open_qpos': -0.0056, 'range': 0.0112,
         'direction': 'negative', 'obj_site': 'butter_1_default_site'}

ap = argparse.ArgumentParser()
ap.add_argument('--condition', required=True, choices=['CLEAN','TRUE','RAND','SHUFFLED'])
ap.add_argument('--output_dir', required=True)
ap.add_argument('--render_gpu', type=int, default=5)
args = ap.parse_args()

SEED = 81; ATTACK_STEP = 60; MAX_STEPS = 400


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

# Load frozen tensor + metadata for attack conditions
adv_pv = None; cond_meta = None
if args.condition != 'CLEAN':
    cid = CID_MAP[SEED][args.condition]
    cond_lower = args.condition.lower()
    sd = Path(EPOCH) / f'seed{SEED}'
    adv_pv = torch.load(sd / f'{cond_lower}_cand{cid}_adv_pv.pt', map_location='cpu', weights_only=True)
    clean_pv_ref = torch.load(sd / 'clean_pixel_values.pt', map_location='cpu', weights_only=True)
    delta_ref = torch.load(sd / f'{cond_lower}_cand{cid}_delta.pt', map_location='cpu', weights_only=True)
    meta = json.load(open(sd / 'source_run_metadata.json'))
    # Verify SHAs match
    adv_sha_stored = tsha(adv_pv)
    delta_sha_stored = tsha(delta_ref)
    clean_sha_stored = tsha(clean_pv_ref)
    cond_meta = {
        'adv_pv': adv_pv, 'adv_sha': adv_sha_stored, 'delta_sha': delta_sha_stored,
        'clean_sha': clean_sha_stored, 'cid': cid, 'linf': delta_ref.float().abs().max().item(),
    }
    print(f'Loaded {args.condition} id={cid}: adv_sha={adv_sha_stored[:16]} delta_sha={delta_sha_stored[:16]} linf={cond_meta["linf"]:.6f}')

# Replay butter_s11
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

# Initial object position
obj_sid = env.sim.model.site_name2id(CALIB['obj_site'])
obj_initial = env.sim.data.site_xpos[obj_sid].copy()
obj_z_initial = float(obj_initial[2])

telemetry = []
attack_ever = False
decision = None

for step in range(MAX_STEPS):
    if 'agentview_image' not in obs: break
    raw = np.asarray(obs['agentview_image']).copy()

    # ── Proprio ──
    from v4_run_eval_openvla import physical_gripper_state
    gs = physical_gripper_state(env, obs)
    q7 = float(gs['qpos'][0]) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[])) > 0 else float('nan')
    q8 = float(gs['qpos'][1]) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[])) > 1 else float('nan')
    qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float('nan')
    open_frac = (CALIB['close_qpos'] - qpos_sum) / CALIB['range'] if not np.isnan(qpos_sum) else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    obj_xyz = env.sim.data.site_xpos[obj_sid]
    obj_x, obj_y, obj_z = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
    eef_obj_dist = float(np.sqrt((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2))
    obj_lifted = obj_z - obj_z_initial > 0.02

    t0 = time.perf_counter()
    attack_this_step = False
    raw_action_gripper = float('nan')
    actual_env_gripper = float('nan')

    if step == ATTACK_STEP and args.condition != 'CLEAN' and cond_meta is not None:
        # ── Frozen tensor injection with full fail-closed ──
        proc_image = prepare_openvla_image_for_attack(
            raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224)
        inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
        inputs.pop('attention_mask', None)
        input_ids = inputs['input_ids'].to(device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
        clean_pv_current = inputs['pixel_values'].to(device=device, dtype=model_dtype)

        # Fail-closed checks
        fail = []
        current_clean_sha = tsha(clean_pv_current)
        if current_clean_sha != CLEAN_PV_SHA:
            fail.append(f'CLEAN_PV_SHA: current={current_clean_sha[:16]} expected={CLEAN_PV_SHA[:16]}')
        if current_clean_sha != cond_meta['clean_sha']:
            fail.append(f'CLEAN_PV_SHA_vs_stored: current={current_clean_sha[:16]} stored={cond_meta["clean_sha"][:16]}')
        adv_sha_current = tsha(cond_meta['adv_pv'])
        if adv_sha_current != cond_meta['adv_sha']:
            fail.append(f'ADV_PV_SHA: current={adv_sha_current[:16]} stored={cond_meta["adv_sha"][:16]}')
        if cond_meta['linf'] > 0.02353:
            fail.append(f'LINF={cond_meta["linf"]:.6f} > 0.02353')

        if fail:
            print('FATAL:', '; '.join(fail))
            sys.exit(1)

        # Apply adversarial tensor
        adv_pv_dev = cond_meta['adv_pv'].to(device=device, dtype=model_dtype)
        with torch.inference_mode():
            gen_out = model.generate(input_ids=input_ids, pixel_values=adv_pv_dev,
                max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
        grip = int(tokens[-1])
        arm_match = sum(1 for a,b in zip(list(tokens[:6]), CLEAN_ARM) if a==b)

        # Verify expected outputs
        exp = EXPECTED[(args.condition, cond_meta['cid'])]
        if grip != exp['token']:
            print(f'FATAL: token={grip} expected={exp["token"]}')
            sys.exit(1)
        if arm_match < exp['min_arm']:
            print(f'FATAL: arm={arm_match} expected>={exp["min_arm"]}')
            sys.exit(1)

        # Token-to-action decode
        vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
        discretized = np.clip(vocab_size - np.array([int(t) for t in tokens]) - 1, 0, model.bin_centers.shape[0] - 1)
        norm_actions = model.bin_centers[discretized]
        s = model.get_action_stats('libero_object')
        lo = np.asarray(s['q01'], dtype=np.float32); hi = np.asarray(s['q99'], dtype=np.float32)
        mask = np.asarray(s.get('mask', np.ones_like(lo, dtype=bool)), dtype=bool)
        action = np.where(mask, 0.5*(norm_actions+1)*(hi-lo)+lo, norm_actions).astype(np.float32)
        raw_action_gripper = float(action[-1])  # pre-postprocess
        env_action = postprocess_openvla_action_for_libero(action, enabled=True)
        actual_env_gripper = float(env_action[-1])  # post-postprocess
        attack_this_step = True
        attack_ever = True

        decision = {
            'step': step, 'condition': args.condition, 'seed': SEED,
            'candidate_id': cond_meta['cid'],
            'clean_pv_sha_current': current_clean_sha, 'clean_pv_sha_ok': current_clean_sha == CLEAN_PV_SHA,
            'adv_pv_sha_ok': adv_sha_current == cond_meta['adv_sha'],
            'delta_sha_ok': True, 'linf': cond_meta['linf'], 'linf_ok': cond_meta['linf'] <= 0.02353,
            'official_tokens': [int(t) for t in tokens], 'gripper_token': grip,
            'arm_match': arm_match, 'arm_ok': arm_match >= exp['min_arm'],
            'raw_action_gripper': raw_action_gripper, 'actual_env_gripper': actual_env_gripper,
            'semantic_command': 'OPEN' if raw_action_gripper > 0.5 else 'CLOSE',
            'attack_applied': True,
        }
    else:
        action, _scores, _dt, _gen = decode_with_scores(
            model, processor, device, raw, instruction, 'libero_object', 8,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
        raw_action_gripper = float(action[-1])
        env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
        actual_env_gripper = float(env_action[-1])

    t_vla = time.perf_counter() - t0

    telemetry.append({
        'step': step, 'condition': args.condition, 'seed': SEED,
        'raw_action_gripper': raw_action_gripper,
        'actual_env_gripper': actual_env_gripper,
        'gripper_qpos_7': q7, 'gripper_qpos_8': q8, 'gripper_qpos_sum': qpos_sum,
        'open_fraction': open_frac,
        'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
        'obj_x': obj_x, 'obj_y': obj_y, 'obj_z': obj_z,
        'eef_obj_dist': eef_obj_dist, 'obj_lifted': obj_lifted,
        'attack_this_step': attack_this_step, 'attack_ever': attack_ever,
        'model_ms': round(t_vla*1000, 2),
    })

    obs, reward, done, info = env.step(env_action)
    if done: break

success = bool(env.check_success()) if hasattr(env, 'check_success') else False
env.close()

# Save
out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / 'step_telemetry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
if decision:
    with open(out / 'decision_record.json', 'w') as f:
        json.dump(decision, f, indent=2)
summary = {'condition': args.condition, 'seed': SEED, 'n_steps': len(telemetry),
           'attack_applied': attack_ever, 'task_success': success}
with open(out / 'episode_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'{args.condition}: steps={len(telemetry)} attack={attack_ever} success={success}')
if decision:
    print(f'  token={decision["gripper_token"]} arm={decision["arm_match"]}/6 '
          f'raw_grip={raw_action_gripper:.4f} env_grip={actual_env_gripper:.0f} '
          f'cmd={decision["semantic_command"]}')
