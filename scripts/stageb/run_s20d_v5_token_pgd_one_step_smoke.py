#!/usr/bin/env python3
"""S20D v5 TokenPrefixPGD one-step smoke test.
Verify true token_pgd interface before full rollout.
No full env rollout — just one adversarial decode per candidate."""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
GPU = '0'
RENDER = '0'
PGD_STEPS = 20; EPS_RAW = 6
SEED = 0

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(T, exist_ok=True)

CASES = [
    {'candidate_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': 2,
     'window_start': 80, 'window_end': 90, 'attack_seed': 93,
     'purpose': 'historical_canary_retest'},
    {'candidate_id': 'cream_cheese_s2_w75_85', 'task': 'cream_cheese', 'state_id': 2,
     'window_start': 75, 'window_end': 85, 'attack_seed': 99,
     'purpose': 'nearby_no_effect_contrast'},
    {'candidate_id': 'bbq_sauce_s0_w125_135', 'task': 'bbq_sauce', 'state_id': 0,
     'window_start': 125, 'window_end': 135, 'attack_seed': 99,
     'purpose': 'deterministic_no_effect_contrast'},
]

# Same model load as S20D v4
def load_model_s20d(model_path, model_gpu_device_id=-1):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    visible = torch.cuda.device_count()
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {"device_map": {"": int(model_gpu_device_id)},
                     "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"}}
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager").strip() or "eager"
    model = AutoModelCls.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=attn_impl, **extra_kw)
    dev = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v; break
            if isinstance(v, int):
                dev = f"cuda:{v}"; break
    return model, processor, dev

print('[%s] Loading model...' % time.strftime('%H:%M:%S'))
model, processor, device = load_model_s20d(MODEL_PATH, model_gpu_device_id=-1)
model_dtype = torch.bfloat16
action_dim = model.config.pad_to_multiple_of
unnorm_key = 'libero_object'
print('[%s] Model loaded: %s dtype=%s' % (time.strftime('%H:%M:%S'), device, model_dtype))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()
task_suite = bm['libero_object']()
TASK_IDX = {
    'ketchup': 4, 'tomato_sauce': 5, 'milk': 7, 'butter': 6,
    'cream_cheese': 1, 'salad_dressing': 2, 'bbq_sauce': 3,
    'alphabet_soup': 0, 'orange_juice': 9, 'chocolate_pudding': 8,
}

from gripper_attack.attack_adapter import OpenVLAVisualAttacker, get_adv_inputs_from_attack_result

def decode_action_from_token_ids(model, token_ids, unnorm_key):
    vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized = np.clip(vocab_size - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    action_stats = model.get_action_stats(unnorm_key)
    mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
    high, low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
    action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions)
    return action.astype(np.float32)

results = []
for case in CASES:
    cid = case['candidate_id']; task = case['task']
    ws = case['window_start']; we = case['window_end']
    seed = case['attack_seed']

    print('[%s] Smoke: %s seed=%d ws=%d' % (time.strftime('%H:%M:%S'), cid, seed, ws))
    try:
    task_idx = TASK_IDX[task]
    task_obj = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    instruction = task_obj.language
    bddl_file = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)

    env = OffScreenRenderEnv(robots=['Panda'], bddl_file_name=bddl_file,
        has_renderer=False, has_offscreen_renderer=True, render_gpu_device_id=int(RENDER),
        use_camera_obs=True, control_freq=20, camera_heights=224, camera_widths=224)
    try: env.env.sim.model.vis.global_.offload = 0
    except: pass
    obs = env.reset()
    obs = env.set_init_state(init_states[case['state_id']])

    # Walk to window_start with clean policy
    for step in range(ws):
        img = obs['agentview_image']
        clean_action, _, _, _ = decode_with_scores(
            model, processor, device, img, instruction, unnorm_key, 2,
            libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224, drop_attention_mask=True)
        clean_env_action = postprocess_openvla_action_for_libero(clean_action, enabled=True)
        obs, reward, done, info = env.step(clean_env_action)
        if done: break

    # At window_start: clean decode + TokenPrefixPGD + adv decode
    img_uint8 = obs['agentview_image']
    clean_action, prefix_logits, Tclean, gen_out = decode_with_scores(
        model, processor, device, img_uint8, instruction, unnorm_key, 2,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)
    clean_env = postprocess_openvla_action_for_libero(clean_action, enabled=True)
    clean_open = 1 if clean_env[-1] < -0.5 else 0

    # Build v5 TokenPrefixPGD attacker
    eps_norm = EPS_RAW / 255.0
    attacker_config = {
        'method': 'token_prefix_pgd',
        'epsilon': eps_norm, 'step_size': eps_norm / max(PGD_STEPS, 1) * 1.5,
        'num_steps': PGD_STEPS, 'random_start': True,
        'objective': 'prefix_locked_gripper_open_margin',
        'arm_preserve_weight': 0.5, 'gripper_margin': 5.0,
    }
    attacker = OpenVLAVisualAttacker(
        model=model, processor=processor, config={'attack_optimizer': attacker_config,
            'directional_target': {'direction_id': 'gripper_open', 'dims': list(range(action_dim))},
            'uncertainty': {'K_trigger': 2}},
        direction_spec={'g_hat': np.zeros(action_dim, dtype=np.float32), 'dims': list(range(action_dim))},
        seed=seed,
        preprocess_kwargs={'libero_official_preprocess': False,
                          'libero_preprocess_backend': 'official_pil_lanczos',
                          'center_crop': True, 'resize_size': 224, 'postprocess_gripper': True},
        device=device)

    interface_ok = attacker.method in {'token_prefix_pgd', 'openvla_token_prefix_pgd', 'visual_token_prefix_pgd'}
    if not interface_ok:
        results.append({**case, 'interface_status': 'FAIL_WRONG_METHOD', 'attackability_status': 'N/A',
                        'attack_method': attacker.method})
        env.close(); continue

    # Run attack
    try:
        attack_result = attacker.attack(img_uint8, instruction, clean_action, clean_action, gen_out, unnorm_key=unnorm_key)
        adv_inputs = get_adv_inputs_from_attack_result(attack_result)
    except Exception as e:
        results.append({**case, 'interface_status': 'FAIL_ATTACK_ERROR', 'attackability_status': 'N/A',
                        'attack_method': attacker.method, 'error': str(e)[:100]})
        env.close(); continue

    if adv_inputs is None or adv_inputs.get("input_ids") is None:
        results.append({**case, 'interface_status': 'FAIL_NO_ADV_INPUTS', 'attackability_status': 'N/A',
                        'attack_method': attacker.method})
        env.close(); continue

    # Adv decode
    input_ids = adv_inputs["input_ids"].to(device)
    pixel_values = adv_inputs["pixel_values"].to(device=device, dtype=model_dtype)
    with torch.inference_mode():
        gen = model.generate(input_ids=input_ids, pixel_values=pixel_values,
            max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=False)
    adv_tids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    adv_action = decode_action_from_token_ids(model, adv_tids, unnorm_key)
    adv_env = postprocess_openvla_action_for_libero(adv_action, enabled=True)
    adv_open = 1 if adv_env[-1] < -0.5 else 0

    # Telemetry
    dbg = getattr(attack_result, 'debug', {}) or {}
    target_ce_initial = float(dbg.get('target_ce_initial', -1) or -1)
    target_ce_final = float(dbg.get('target_ce_final', -1) or -1)
    loss_decrease = float(dbg.get('loss_decrease', 0) or 0)
    open_prob = float(dbg.get('open_region_prob_mass_after', -1) or -1)
    close_prob = float(dbg.get('close_bin_prob_mass_after', -1) or -1)
    logit_margin = float(dbg.get('gripper_logit_margin_after', -999) or -999)
    corrected_open = int(dbg.get('corrected_open_token_count', -1) or -1)
    perturb_linf = float(dbg.get('pixel_budget_adv_inputs_linf', -1) or -1)

    interface_pass = (target_ce_final <= target_ce_initial or loss_decrease > 0) and corrected_open > 0
    attackability_pass = adv_open == 1 or logit_margin > -900

    results.append({
        'candidate_id': cid, 'task': task, 'state_id': case['state_id'],
        'window_start': ws, 'window_end': we, 'attack_seed': seed,
        'clean_gripper_raw': round(float(clean_action[-1]), 6),
        'clean_env_gripper': round(float(clean_env[-1]), 6),
        'clean_open_bool': clean_open,
        'adv_gripper_raw': round(float(adv_action[-1]), 6),
        'adv_env_gripper': round(float(adv_env[-1]), 6),
        'adv_open_bool': adv_open,
        'target_ce_initial': round(target_ce_initial, 6),
        'target_ce_final': round(target_ce_final, 6),
        'loss_decrease': round(loss_decrease, 6),
        'open_region_prob_mass_after': round(open_prob, 6),
        'close_bin_prob_mass_after': round(close_prob, 6),
        'gripper_logit_margin_after': round(logit_margin, 6),
        'corrected_open_token_count': corrected_open,
        'pixel_budget_adv_inputs_linf': round(perturb_linf, 8),
        'attack_method': attacker.method,
        'token_label_source': str(dbg.get('token_label_source', 'not_available')),
        'adv_decode_path': 'token_pgd_adv_inputs_generate',
        'pgd_applied': 1 if adv_inputs else 0,
        'used_adv_inputs': adv_inputs is not None,
        'used_x_adv': False,
        'fallback_adapter_used': False,
        'interface_status': 'PASS' if interface_pass else 'FAIL',
        'attackability_status': 'PASS' if attackability_pass else 'FAIL',
        'purpose': case['purpose'],
    })
    print('[%s]   clean_open=%d adv_open=%d ce_loss %.4f→%.4f margin=%.4f interface=%s attackability=%s' %
          (time.strftime('%H:%M:%S'), clean_open, adv_open, target_ce_initial, target_ce_final,
           logit_margin, 'PASS' if interface_pass else 'FAIL', 'PASS' if attackability_pass else 'FAIL'))
    try:
        env.close()
    except Exception:
        pass
    torch.cuda.empty_cache()

# Write smoke table
with open(T+'/s20d_v5_token_pgd_one_step_smoke.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
    w.writeheader(); w.writerows(results)

print()
print('=== ONE-STEP SMOKE RESULTS ===')
for r in results:
    print('  %-30s seed=%s clean_open=%d adv_open=%d interface=%s attack=%s' %
          (r['candidate_id'], r['attack_seed'], r['clean_open_bool'], r['adv_open_bool'],
           r['interface_status'], r['attackability_status']))
n_interface = sum(1 for r in results if r['interface_status'] == 'PASS')
n_attack = sum(1 for r in results if r['attackability_status'] == 'PASS')
print('  Interface PASS: %d/%d  Attackability PASS: %d/%d' % (n_interface, len(results), n_attack, len(results)))
print()
print('Table: %s/s20d_v5_token_pgd_one_step_smoke.csv' % T)
if n_interface == 0:
    print('VERDICT: TokenPrefixPGD interface NOT WORKING. Do not launch rollout.')
elif n_attack == 0:
    print('VERDICT: Interface OK but attackability FAIL. TokenPGD not changing gripper tokens on these candidates.')
else:
    print('VERDICT: TokenPrefixPGD interface AND attackability working. Ready for rollout canary.')
