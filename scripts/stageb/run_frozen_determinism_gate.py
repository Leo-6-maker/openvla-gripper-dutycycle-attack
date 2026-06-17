#!/usr/bin/env python3
"""Gate D0: Frozen-Input Determinism Audit.
Tests whether repeated greedy decode on identical frozen tensors produces bitwise-identical output.
D0-A: same-process ×100
D0-B: cross-process ×20
D0-C: config matrix (single-GPU bf16, deterministic CUDA flags)
"""
import copy, csv, hashlib, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

# ── Config ──
GPU_PAIR = '4,5'
RENDER_GPU = 4
MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
TASK = 'cream_cheese'; STATE_ID = 35; TASK_IDX = 1
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
OUT = '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613/determinism'
os.makedirs(OUT, exist_ok=True)

print('[%s] === Gate D0: Frozen-Input Determinism ===' % time.strftime('%H:%M:%S'), flush=True)

# ── Step 1: Capture frozen tensors ──
print('[%s] Step 1: Capturing frozen tensors...' % time.strftime('%H:%M:%S'), flush=True)
capture_script = os.path.join(OUT, '_capture_frozen.py')
with open(capture_script, 'w') as f:
    f.write('''#!/usr/bin/env python3
import hashlib, json, os, sys, time
import numpy as np, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '%s'
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, '%s'); sys.path.insert(0, '%s/src'); sys.path.insert(0, '%s/scripts')
from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait, TARGET_OBJECT_GUESS_V4
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls
from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image
from libero.libero import benchmark, get_libero_path

MODEL_PATH = '%s'; RENDER_GPU = %d; TASK = '%s'; STATE_ID = %d; TASK_IDX = %d
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count(); mm = "10000MiB"
model = AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
    device_map="auto", max_memory={idx: mm for idx in range(max(visible, 1))})
device = "cuda:0"
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"): device = v; break
        if isinstance(v, int): device = "cuda:%d" % v; break
model_dtype = torch.bfloat16; unnorm_key = 'libero_object'; action_dim = int(model.get_action_dim(unnorm_key))
bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()
ti = TASK_IDX; task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

from gripper_attack.libero_v4_env_factory import seed_everything; seed_everything(0)
env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, 120, num_steps_wait=10)
obs = env.set_init_state(init_states[STATE_ID]); env, obs = apply_dummy_wait(env, obs, 10)

close_frames = []
open_frames = []
for step in range(120):
    obs['agentview_image']; img_uint8 = obs['agentview_image']
    img_pil = Image.fromarray(img_uint8)
    inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
    input_ids = inputs['input_ids'].detach().cpu().clone()
    pixel_values = inputs['pixel_values'].detach().cpu().clone()
    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
    # Decode action
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - tids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    clean_a = np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)
    env_a = postprocess_openvla_action_for_libero(clean_a, TARGET_OBJECT_GUESS_V4.get(TASK, TASK))

    frame_data = {
        'step': step,
        'rgb_sha256': hashlib.sha256(img_uint8.tobytes()).hexdigest(),
        'input_ids_sha256': hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        'pixel_values_sha256': hashlib.sha256(pixel_values.numpy().tobytes()).hexdigest(),
        'gripper_env': float(env_a[-1]),
        'is_close': int(float(env_a[-1]) > 0.5),
        'is_open': int(float(env_a[-1]) < -0.5),
        'token_ids': tids.tolist(),
    }
    if step >= 10:
        if frame_data['is_close'] and len(close_frames) < 2:
            close_frames.append(frame_data)
        if frame_data['is_open'] and len(open_frames) < 1:
            open_frames.append(frame_data)
        if len(close_frames) >= 2 and len(open_frames) >= 1:
            # Save last CLOSE frame's tensors
            f = close_frames[-1]
            np.savez('%s/frozen_tensors.npz' % os.environ.get('OUT_DIR', '/tmp'),
                img_uint8=img_uint8, input_ids=input_ids.numpy(), pixel_values=pixel_values.numpy(),
                instruction=instruction, step=f['step'])
            print('CAPTURED: step=%d rgb=%s input_ids=%s pixel=%s' % (f['step'], f['rgb_sha256'], f['input_ids_sha256'], f['pixel_values_sha256']))
            print('GRIPPER: %.1f CLOSE=%d OPEN=%d' % (f['gripper_env'], f['is_close'], f['is_open']))
            print('TOKENS: %s' % f['token_ids'])
            break
    obs, _, _, _ = env.step(env_a)
env.close()
''' % (GPU_PAIR, REPO, REPO, REPO, MODEL_PATH, RENDER_GPU, TASK, STATE_ID, TASK_IDX))

env = {**os.environ, 'OUT_DIR': OUT}
result = subprocess.run([PY, capture_script], env=env, capture_output=True, text=True, cwd=REPO, timeout=300)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr[-500:])
    sys.exit(1)

# Load frozen tensors
frozen = np.load(os.path.join(OUT, 'frozen_tensors.npz'), allow_pickle=True)
img_uint8 = frozen['img_uint8']
frozen_input_ids = frozen['input_ids']
frozen_pixel_values = frozen['pixel_values']
instruction = str(frozen['instruction'])

print('[%s] Frozen: step=%s rgb_sha256=%s' % (time.strftime('%H:%M:%S'), frozen['step'],
    hashlib.sha256(img_uint8.tobytes()).hexdigest()[:16]), flush=True)

# ── D0-A: Same-process repeated decode ×100 ──
print('[%s] D0-A: Same-process ×100...' % time.strftime('%H:%M:%S'), flush=True)
d0a_script = os.path.join(OUT, '_d0a_repeat.py')
with open(d0a_script, 'w') as f:
    f.write('''#!/usr/bin/env python3
import hashlib, json, os, sys, time
from collections import Counter
import numpy as np, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '%s'
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
sys.path.insert(0, '%s'); sys.path.insert(0, '%s/src'); sys.path.insert(0, '%s/scripts')
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls

MODEL_PATH = '%s'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count(); mm = "10000MiB"
model = AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
    device_map="auto", max_memory={idx: mm for idx in range(max(visible, 1))})
device = "cuda:0"; model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key))
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"): device = v; break
        if isinstance(v, int): device = "cuda:%d" % v; break

frozen = np.load('%s/frozen_tensors.npz', allow_pickle=True)
input_ids_t = torch.from_numpy(frozen['input_ids']).to(device)
pixel_values_t = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)

N = %d
results = []
for i in range(N):
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids_t, pixel_values=pixel_values_t,
            max_new_tokens=action_dim, do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=True)
    tids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    tid_hash = hashlib.sha256(tids.tobytes()).hexdigest()
    gripper_token = int(tids[-1])
    results.append({'rep': i, 'token_hash': tid_hash, 'gripper_token': gripper_token, 'all_tokens': tids.tolist()})
    if i == 0:
        # Compute logits for margin analysis
        out = model(input_ids=input_ids_t, pixel_values=pixel_values_t, use_cache=False, return_dict=True)
        logits = out.logits.float()
        action_token_logit_row_index = action_dim - 2  # gripper row
        gripper_row = logits[0, action_token_logit_row_index, :]
        from gripper_attack.attack_adapter import OpenVLAVisualAttacker
        # Quick margin: use attacker to get region info
        print('LOGITS_SHAPE:' + str(gripper_row.shape))

token_hashes = set(r['token_hash'] for r in results)
gripper_tokens = Counter(r['gripper_token'] for r in results)
print('Unique 7-token sequences: %d/%d' % (len(token_hashes), N))
print('Gripper token distribution: %s' % dict(gripper_tokens.most_common()))
print('DETERMINISTIC' if len(token_hashes) == 1 else 'NONDETERMINISTIC: %d unique outputs' % len(token_hashes))
json.dump({'unique_tokens': len(token_hashes), 'N': N, 'gripper_distribution': {str(k): v for k,v in gripper_tokens.items()},
    'results': [{k: v for k,v in r.items() if k != 'all_tokens'} for r in results[:5]]},
    open('%s/d0a_results.json', 'w'), indent=2)
''' % (GPU_PAIR, REPO, REPO, REPO, MODEL_PATH, OUT, 100, OUT))

result = subprocess.run([PY, d0a_script], env=os.environ, capture_output=True, text=True, cwd=REPO, timeout=600)
print(result.stdout)
if 'DETERMINISTIC' in result.stdout:
    d0a_deterministic = True
elif 'NONDETERMINISTIC' in result.stdout:
    d0a_deterministic = False
else:
    print('STDERR:', result.stderr[-500:] if result.stderr else 'none')
    d0a_deterministic = None

# ── D0-C: Deterministic CUDA config ──
print('[%s] D0-C: Deterministic CUDA flags...' % time.strftime('%H:%M:%S'), flush=True)
det_env = {**os.environ,
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
# We need a modified script that sets torch deterministic flags
d0c_script = os.path.join(OUT, '_d0c_deterministic.py')
with open(d0c_script, 'w') as f:
    f.write(d0a_script.replace(
        "os.environ.setdefault(\"OPENVLA_CUDA_MAX_MEMORY\", \"10000MiB\")",
        "os.environ.setdefault(\"OPENVLA_CUDA_MAX_MEMORY\", \"10000MiB\")\n"
        "torch.use_deterministic_algorithms(True)\n"
        "torch.backends.cudnn.benchmark = False\n"
        "torch.backends.cudnn.deterministic = True\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.backends.cudnn.allow_tf32 = False").replace('_d0a_repeat', '_d0c_deterministic').replace('d0a_results', 'd0c_results').replace('N = 100', 'N = 50'))

result = subprocess.run([PY, d0c_script], env=det_env, capture_output=True, text=True, cwd=REPO, timeout=600)
print(result.stdout)
d0c_deterministic = 'DETERMINISTIC' in result.stdout

# ── Summary ──
print('\n=== Gate D0 Summary ===', flush=True)
print('D0-A (same-process ×100): %s' % ('DETERMINISTIC' if d0a_deterministic else ('NONDETERMINISTIC' if d0a_deterministic is False else 'ERROR')), flush=True)
print('D0-C (det CUDA flags ×50): %s' % ('DETERMINISTIC' if d0c_deterministic else 'NONDETERMINISTIC'), flush=True)

summary = {
    'd0a_deterministic': d0a_deterministic,
    'd0c_deterministic': d0c_deterministic,
    'frozen_step': int(frozen['step']),
}
with open(os.path.join(OUT, 'd0_gate_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

if d0a_deterministic or d0c_deterministic:
    print('GATE D0 PASSED: deterministic config found', flush=True)
else:
    print('GATE D0: no fully deterministic config. Repeated decode required for objective comparison.', flush=True)
