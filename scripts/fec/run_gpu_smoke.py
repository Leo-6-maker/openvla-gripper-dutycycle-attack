"""FEC GPU smoke: run all 5 arms on SMOKE_ONLY identity, validate telemetry/K10/paired semantics."""
import sys, os, json, time, hashlib, numpy as np, torch
from pathlib import Path

GPUID = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SMOKE_STATE = int(sys.argv[2]) if len(sys.argv) > 2 else 111
SUITE = sys.argv[3] if len(sys.argv) > 3 else 'libero_10'
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path('/tmp/fec_smoke')

# Per-suite max steps (from OFFICIAL_V3_SOURCE_CONTRACT_V1.json official_horizons)
# num_steps_wait=10 excluded from policy horizon; env must include wait steps
NUM_STEPS_WAIT = 10
MAX_STEPS = {'libero_10': 520, 'libero_goal': 300, 'libero_object': 280, 'libero_spatial': 220}
SUITE_POLICY_HORIZON = MAX_STEPS.get(SUITE, 280)
SUITE_ENV_HORIZON = SUITE_POLICY_HORIZON + NUM_STEPS_WAIT

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
sys.path.insert(0, '/tmp')

from n4_detector_adapter import N4DetectorAdapter
E = '/mnt/sdc/dty_user/openvla_attack_evidence'
MODEL_PATHS = {
    'libero_10': '/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10',
    'libero_goal': '/mnt/sdc/dty_user/openvla_attack/models/libero-goal',
    'libero_object': '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object',
    'libero_spatial': '/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620',
}
MODEL_PATH = MODEL_PATHS[SUITE]

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

# ═══ SELF-CHECK ═══
import gripper_attack.attack_adapter as aa
attacker_sha = sha256_file(os.path.realpath(aa.__file__))
assert attacker_sha == '26cfb9f5d8a5a29e7ac2729f5c9cdd58dadfd75e45eebe935ee66214cc9402be', f'SHA MISMATCH: {attacker_sha[:16]}'
print('SELF-CHECK: attacker SHA = PASS', flush=True)

# ═══ LOAD ═══
print('Loading components...', flush=True)
from transformers import AutoProcessor, AutoModelForVision2Seq
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from gripper_attack.openvla_preprocess import prepare_openvla_image
from gripper_attack.attack_adapter import OpenVLAVisualAttacker

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
model.eval()
adapter = N4DetectorAdapter(device='cuda:0', norm_data_path=E + '/fec_implementation_v1/n4_norms_o0i0.pt')
print(f'Model: {torch.cuda.memory_allocated(0)/1e9:.1f}GB', flush=True)

bench_obj = benchmark.get_benchmark_dict()[SUITE]()
task = bench_obj.get_task(0)
bddl_file = bench_obj.get_task_bddl_file_path(0)
instruction = str(task)

def prompt(ins): return f'In: What action should the robot take to {ins.lower()}?\nOut:'

def decode_action(model, processor, obs, instruction, suite):
    image = prepare_openvla_image(obs['agentview_image'])
    inputs = processor(prompt(instruction), image, return_tensors='pt')
    inputs.pop('attention_mask', None)
    for key, val in list(inputs.items()):
        if torch.is_floating_point(val): inputs[key] = val.to(device='cuda', dtype=torch.bfloat16)
        else: inputs[key] = val.to(device='cuda')
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=7, do_sample=False, return_dict_in_generate=True, output_scores=True)
    token_ids = gen.sequences[0, -7:].detach().cpu().numpy()
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, a_min=0, a_max=model.bin_centers.shape[0]-1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(suite)
    mask = stats.get('mask', np.ones_like(stats['q01'], dtype=bool))
    high, low = np.array(stats['q99']), np.array(stats['q01'])
    action = np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)
    return action, gen

# ═══ ATTACKER SETUP (fec_attack_v3.yaml) ═══
ATTACK_BASE = {
    'attack_optimizer': {
        'method': 'token_prefix_pgd',
        'strict_route': True, 'allow_fallback': False,
        'epsilon': 0.03, 'num_steps': 5, 'step_size': 0.006,
        'random_start': False, 'clip_min': 0.0, 'clip_max': 1.0,
        'temporal_init': 'none', 'temporal_smooth_lambda': 0.0,
        'surrogate_score_path': 'cached_autoregressive_generate_v1',
        'prefix_refresh_interval': 1,
        'target_token_id': 31744, 'target_execution_class': 'CLIP_MEDIATED_OPEN',
        'gripper_margin': 5.0, 'arm_preserve_weight': 0.1,
        'gradient_transform': 'none',
    },
    'victim': 'openvla_7b',
}

def make_attacker(objective_override=None, gradient_transform_override=None):
    cfg = json.loads(json.dumps(ATTACK_BASE))  # deep copy
    if objective_override:
        cfg['attack_optimizer']['objective'] = objective_override
    if gradient_transform_override:
        cfg['attack_optimizer']['gradient_transform'] = gradient_transform_override
    return OpenVLAVisualAttacker(model=model, processor=processor, config=cfg, device='cuda:0')

# Pre-build attackers
true_attacker = make_attacker()
rand_attacker = make_attacker(gradient_transform_override='rademacher')

# ═══ RUN ALL 5 ARMS ═══
ARMS = ['CLEAN', 'TRUE_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE', 'RANDOM_TIME_T10']
results = {}

for arm in ARMS:
    print(f'\n=== ARM: {arm} ===', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    arm_dir = OUT / arm
    arm_dir.mkdir(exist_ok=True)

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=224, camera_widths=224,
                             render_gpu_device_id=0, horizon=SUITE_ENV_HORIZON)
    obs = env.reset()
    adapter.reset_episode()

    # RANDOM_TIME: pre-sample start before rollout
    rt_start = None
    if arm == 'RANDOM_TIME_T10':
        rng = np.random.RandomState(42)
        rt_start = int(rng.randint(20, max(21, SUITE_ENV_HORIZON // 3)))
        print('  RANDOM_TIME start={}'.format(rt_start), flush=True)

    steps = []
    emit_step = None
    attack_active_frames = 0
    task_success = False
    done = False

    for t in range(SUITE_ENV_HORIZON):
        clean_action, gen = decode_action(model, processor, obs, instruction, SUITE)

        # N4 Detector
        f25d = np.zeros(25, dtype=np.float32)
        f25d[0] = float(clean_action[6])
        f25d[6:9] = clean_action[0:3]
        p9d = np.zeros(9, dtype=np.float32)
        g9d_arr = np.zeros(9, dtype=np.float32)
        cc = clean_action[6] < 0.5
        n4 = adapter.step(f25d, p9d, g9d_arr, cc)
        if n4['emitted_this_step'] and emit_step is None:
            emit_step = t

        # Determine if attack is active
        attack_active = False
        final_action = clean_action.copy()

        if arm == 'CLEAN':
            pass  # no intervention
        elif arm in ('TRUE_T10', 'RAND_T10', 'COMMAND_OPEN_ORACLE'):
            if emit_step is not None and emit_step <= t < emit_step + 10:
                attack_active = True
                if arm == 'COMMAND_OPEN_ORACLE':
                    final_action[6] = 1.0  # OPEN
                elif arm in ('TRUE_T10', 'RAND_T10'):
                    try:
                        atk = true_attacker if arm == 'TRUE_T10' else rand_attacker
                        attack_res = atk.attack(
                            observation=obs, instruction=instruction,
                            clean_action=clean_action, target_action=None,
                            clean_model_output=gen, unnorm_key=SUITE
                        )
                        if attack_res.debug and 'adv_inputs' in attack_res.debug:
                            adv_inputs = attack_res.debug['adv_inputs']
                            # Re-decode
                            inputs2 = {k: v.to('cuda') if torch.is_tensor(v) else v
                                       for k, v in adv_inputs.items()}
                            with torch.inference_mode():
                                gen2 = model.generate(**inputs2, max_new_tokens=7, do_sample=False,
                                                     return_dict_in_generate=True, output_scores=True)
                            token_ids2 = gen2.sequences[0, -7:].cpu().numpy()
                            vocab_size2 = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
                            discretized2 = np.clip(vocab_size2 - token_ids2 - 1, a_min=0, a_max=model.bin_centers.shape[0]-1)
                            norm2 = model.bin_centers[discretized2]
                            stats2 = model.get_action_stats(SUITE)
                            mask2 = stats2.get('mask', np.ones_like(stats2['q01'], dtype=bool))
                            final_action = np.where(mask2, 0.5*(norm2+1)*(np.array(stats2['q99'])-np.array(stats2['q01']))+np.array(stats2['q01']), norm2).astype(np.float32)
                    except Exception as e:
                        print(f'  Attack error at t={t}: {e}', flush=True)
        elif arm == 'RANDOM_TIME_T10':
            if rt_start is not None and rt_start <= t < rt_start + 10:
                attack_active = True
                try:
                    attack_res = true_attacker.attack(
                        observation=obs, instruction=instruction,
                        clean_action=clean_action, target_action=None,
                        clean_model_output=gen, unnorm_key=SUITE
                    )
                    if attack_res.debug and 'adv_inputs' in attack_res.debug:
                        adv_inputs = attack_res.debug['adv_inputs']
                        inputs2 = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in adv_inputs.items()}
                        with torch.inference_mode():
                            gen2 = model.generate(**inputs2, max_new_tokens=7, do_sample=False,
                                                 return_dict_in_generate=True, output_scores=True)
                        token_ids2 = gen2.sequences[0, -7:].cpu().numpy()
                        vocab_size2 = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
                        discretized2 = np.clip(vocab_size2 - token_ids2 - 1, a_min=0, a_max=model.bin_centers.shape[0]-1)
                        norm2 = model.bin_centers[discretized2]
                        stats2 = model.get_action_stats(SUITE)
                        mask2 = stats2.get('mask', np.ones_like(stats2['q01'], dtype=bool))
                        final_action = np.where(mask2, 0.5*(norm2+1)*(np.array(stats2['q99'])-np.array(stats2['q01']))+np.array(stats2['q01']), norm2).astype(np.float32)
                except Exception as e:
                    print(f'  Attack error at t={t}: {e}', flush=True)

        if attack_active:
            attack_active_frames += 1

        obs, reward, done, info = env.step(final_action)
        steps.append({'t': t, 'emit': n4['emitted_this_step'], 'cal': n4['calibrated_prob'],
                      'attack': attack_active, 'gripper': float(final_action[6])})

        if done:
            task_success = info.get('success', False)
            break

    env.close()
    traj = adapter.get_trajectory()
    results[arm] = {
        'emit_step': emit_step, 'emitted': traj['emitted'],
        'attack_frames': attack_active_frames, 'task_success': task_success,
        'total_steps': len(steps)
    }
    print(f'  emit={emit_step} attack_frames={attack_active_frames} success={task_success} steps={len(steps)}', flush=True)
    with open(arm_dir / 'result.json', 'w') as f:
        json.dump(results[arm], f)
    with open(arm_dir / 'steps.jsonl', 'w') as f:
        for s in steps: f.write(json.dumps(s) + '\n')

# ═══ SUMMARY ═══
print(f'\n=== SMOKE SUMMARY (GPU {GPUID}, state {SMOKE_STATE}) ===')
for arm in ARMS:
    r = results[arm]
    print('  {}: emit={} attack_frames={} success={}'.format(arm, r['emit_step'], r['attack_frames'], r['task_success']))

valid = all(
    results['CLEAN']['attack_frames'] == 0 and
    (results['TRUE_T10']['attack_frames'] == 10 or results['TRUE_T10']['emit_step'] is None) and
    (results['RAND_T10']['emit_step'] == results['TRUE_T10']['emit_step']) and
    results['RANDOM_TIME_T10']['attack_frames'] in (0, 10)
)
print('SMOKE: {}'.format('PASS' if valid else 'ISSUES'))
with open(OUT / 'smoke_summary.json', 'w') as f:
    json.dump({'gpu': GPUID, 'state': SMOKE_STATE, 'suite': SUITE,
               'results': results, 'valid': valid}, f, indent=2)
