#!/usr/bin/env python3
"""G0: Exact cohort parent eligibility — clean replay, anchor verify, D5 emit, object binding."""
import sys, os, json, torch, io, hashlib, numpy as np, time, csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
D5_CKPT = '/data/liuyu/outputs/d5_training/d5_candidate_best.pt'
D5_CFG = '/data/liuyu/outputs/d5_training/d5_frozen_config.json'

TASK_IDX = {'butter': 6, 'ketchup': 4, 'milk': 7}
PARENTS = {
    'ketchup_s18': {'task': 'ketchup', 'state_id': 18, 'anchor': 84},
    'milk_s7': {'task': 'milk', 'state_id': 7, 'anchor': 41},
}

parent = sys.argv[1]; cfg = PARENTS[parent]
task = cfg['task']; state_id = cfg['state_id']; anchor = cfg['anchor']
render_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 5
output_dir = sys.argv[3] if len(sys.argv) > 3 else f'/data/liuyu/outputs/cohort_g0_{parent}'

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
print(f'Model loaded on {device}')

# Load D5 detector
from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
detector = D5FrozenOnlineDetectorV1(D5_CKPT, D5_CFG)
detector.reset()

# Replay
from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
from libero.libero import benchmark, get_libero_path

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_idx = TASK_IDX[task]
task_obj = suite.get_task(task_idx); init_states = suite.get_task_init_states(task_idx)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

env, obs = build_v4_exact_env(bddl, render_gpu, 400, 10)
obs = env.set_init_state(init_states[state_id])
env, obs = apply_dummy_wait(env, obs, 10)
print(f'Task: {task} state={state_id} anchor={anchor}')

# Object binding
obj_sites = [n for n in env.sim.model.site_names if task in n.lower() and 'default' in n]
obj_site = obj_sites[0] if obj_sites else env.sim.model.site_names[0]
obj_sid = env.sim.model.site_name2id(obj_site)
print(f'Object site: {obj_site}')

# Gripper calibration
from v4_run_eval_openvla import physical_gripper_state
gs_init = physical_gripper_state(env, obs)
qpos_init_7 = float(gs_init['qpos'][0]) if gs_init and len(gs_init.get('qpos',[]))>0 else float('nan')
qpos_init_8 = float(gs_init['qpos'][1]) if gs_init and len(gs_init.get('qpos',[]))>1 else float('nan')

raw_at_anchor = None; anchor_clean_grip = None; clean_pv_sha = None
success = False; d5_emit = -1; anchor_reached = False

for step in range(400):
    if 'agentview_image' not in obs: break
    raw = np.asarray(obs['agentview_image']).copy()
    action, _, _, _ = decode_with_scores(
        model, processor, device, raw, instruction, 'libero_object', 8,
        libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
        center_crop=True, resize_size=224, drop_attention_mask=True)

    gs = physical_gripper_state(env, obs)
    qpos_sum = float(np.sum(gs['qpos'])) if gs and gs.get('qpos') is not None and len(gs.get('qpos',[]))>0 else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0

    # D5 update
    detector.update(step, raw_grip, env_grip, qpos_sum if not np.isnan(qpos_sum) else float('nan'),
                    float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
                    1 if raw_grip>0.5 else 0,
                    raw_valid=True, env_valid=True, qpos_valid=not np.isnan(qpos_sum), eef_valid=True)

    if step == anchor:
        raw_at_anchor = raw.copy()
        anchor_reached = True
        # Canonical preprocessing
        proc_image = prepare_openvla_image_for_attack(
            raw, libero_official_preprocess=False, libero_preprocess_backend='official_pil_lanczos',
            center_crop=True, resize_size=224)
        inputs = processor(prompt(instruction), proc_image, return_tensors='pt')
        inputs.pop('attention_mask', None)
        input_ids = inputs['input_ids'].to(device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
        clean_pv = inputs['pixel_values'].to(device=device, dtype=model_dtype)
        clean_pv_sha = tsha(clean_pv)
        # Official decode at anchor
        with torch.inference_mode():
            gen_out = model.generate(input_ids=input_ids, pixel_values=clean_pv,
                max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        tokens = extract_exact_new_tokens(gen_out.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=action_dim)
        anchor_clean_grip = int(tokens[-1])
        anchor_arm = [int(t) for t in tokens[:6]]
        anchor_tokens = [int(t) for t in tokens]

        # Save anchor package
        out_pkg = Path(output_dir) / f'{parent}_anchor'
        out_pkg.mkdir(parents=True, exist_ok=True)
        np.save(out_pkg / 'raw_frame.npy', raw)
        torch.save({'input_ids': input_ids.detach().cpu(), 'pixel_values': clean_pv.detach().cpu()},
                   out_pkg / 'processor_inputs_attack.pt')
        gen_data = {
            'parent': parent, 'task': task, 'state_id': state_id, 'anchor_step': anchor,
            'instruction': instruction, 'prompt': prompt(instruction),
            'clean_tokens': anchor_tokens, 'clean_arm': anchor_arm,
            'clean_gripper_token': anchor_clean_grip,
            'clean_pv_sha256': clean_pv_sha,
            'clean_eligible': anchor_clean_grip == 31872,
        }
        with open(out_pkg / 'clean_generation.json', 'w') as f:
            json.dump(gen_data, f, indent=2)
        print(f'Anchor step{anchor}: grip={anchor_clean_grip} eligible={anchor_clean_grip==31872} sha={clean_pv_sha[:16]}')

    env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
    obs, _, done, _ = env.step(env_action)
    if done:
        success = bool(env.check_success()) if hasattr(env, 'check_success') else False
        break

env.close()
d5_emit = detector.emit_step

result = {
    'parent': parent, 'task': task, 'state_id': state_id, 'anchor': anchor,
    'anchor_reached': anchor_reached, 'clean_gripper': anchor_clean_grip,
    'clean_eligible': anchor_clean_grip == 31872, 'clean_pv_sha': clean_pv_sha,
    'task_success': success, 'd5_emit': d5_emit,
    'd5_anchor_offset': (d5_emit - anchor) if d5_emit >= 0 else None,
    'object_site': obj_site, 'total_steps': step + 1,
}
with open(Path(output_dir) / 'eligibility.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'G0 {parent}: success={success} grip={anchor_clean_grip} eligible={anchor_clean_grip==31872} '
      f'd5_emit={d5_emit} offset={result["d5_anchor_offset"]} steps={step+1}')
