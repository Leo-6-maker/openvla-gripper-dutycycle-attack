#!/usr/bin/env python3
"""Phase 0 D0-C: Determinism matrix on exact frozen tensors.
C1: generate() x100 (baseline)
C2: manual greedy use_cache=False x100
C3: manual greedy use_cache=True x100
C4: deterministic CUDA flags + generate() x100
C5: fixed hf_device_map x100
"""
import hashlib, json, os, sys, time
from collections import Counter
import numpy as np
import torch

GPU_PAIR = os.environ.get('GPU_PAIR', '4,5')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_PAIR
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO + '/scripts')

OUT = os.environ.get('OUT_DIR', '/data/liuyu/outputs/stageb_v5_frozen_objective_day3_20260613/d0')
os.makedirs(OUT, exist_ok=True)
FROZEN = os.environ.get('FROZEN_NPZ',
    '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613/determinism/frozen_close_0.npz')

# ── Load model and frozen tensors ──
print('[%s] Loading model...' % time.strftime('%H:%M:%S'), flush=True)
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
visible = torch.cuda.device_count(); mm = "10000MiB"

def load_model(device_map="auto"):
    return AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
        device_map=device_map, max_memory={idx: mm for idx in range(max(visible, 1))})

model = load_model()
device = "cuda:0"
if hasattr(model, "hf_device_map"):
    for v in model.hf_device_map.values():
        if isinstance(v, str) and v.startswith("cuda"): device = v; break
        if isinstance(v, int): device = "cuda:%d" % v; break
model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7
device_map_snap = {str(k): str(v) for k,v in model.hf_device_map.items()} if hasattr(model, "hf_device_map") else {}
print('[%s] Model loaded. device_map has %d entries' % (time.strftime('%H:%M:%S'), len(device_map_snap)), flush=True)

frozen = np.load(FROZEN, allow_pickle=True)
input_ids_t = torch.from_numpy(frozen['input_ids']).to(device)
pixel_values_t = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)
print('[%s] Frozen: step=%s gripper=%.1f' % (time.strftime('%H:%M:%S'), frozen['step'], frozen['gripper']), flush=True)

# Compute region info
stats_d = model.get_action_stats(unnorm_key)
low = np.asarray(stats_d['q01'], dtype=np.float32); high = np.asarray(stats_d['q99'], dtype=np.float32)
centers = np.asarray(model.bin_centers, dtype=np.float32)
vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
gripper_dim = action_dim - 1
open_tokens = []; close_tokens = []
for disc in range(len(centers)):
    norm = centers[disc]
    da = float(0.5 * (norm + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim])
    env_val = 2.0 * da - 1.0; tid = int(vocab_size - disc - 1)
    if env_val < -0.5: open_tokens.append(tid)
    elif env_val > 0.5: close_tokens.append(tid)
open_token_ids = torch.tensor(sorted(set(open_tokens)), dtype=torch.long, device=device)
close_token_ids = torch.tensor(sorted(set(close_tokens)), dtype=torch.long, device=device)

def decode_action(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

def run_test(name, N, generate_fn, extra_info=None):
    print('[%s] %s: %d decodes...' % (time.strftime('%H:%M:%S'), name, N), flush=True)
    results = []
    for i in range(N):
        tids = generate_fn()
        tid_hash = hashlib.sha256(tids.tobytes()).hexdigest()
        arm_hash = hashlib.sha256(tids[:6].tobytes()).hexdigest()
        action = decode_action(tids)
        env_g = -1.0 if action[-1] > 0.5 else (1.0 if action[-1] < -0.5 else 0.0)
        is_open = int(env_g < -0.5)
        results.append({'tid_hash': tid_hash, 'arm_hash': arm_hash, 'gripper': int(tids[-1]), 'is_open': is_open})

    token_hashes = set(r['tid_hash'] for r in results)
    arm_hashes = set(r['arm_hash'] for r in results)
    gripper_dist = Counter(r['gripper'] for r in results)
    open_count = sum(1 for r in results if r['is_open'])
    close_count = N - open_count

    det = len(token_hashes) == 1
    cls = 'STABLE_OPEN' if open_count >= 0.9*N else ('STABLE_CLOSE' if close_count >= 0.9*N else 'MIXED')
    print('  %s: unique=%d arms=%d open=%d/%d (%.0f%%) cls=%s' % (
        'DET' if det else 'NON', len(token_hashes), len(arm_hashes), open_count, N, 100*open_count/N, cls), flush=True)
    return {'name': name, 'N': N, 'deterministic': det, 'unique_tokens': len(token_hashes),
        'unique_arm_prefixes': len(arm_hashes), 'gripper_dist': {str(k):v for k,v in gripper_dist.most_common()},
        'open_count': open_count, 'close_count': close_count, 'frame_class': cls,
        'extra': extra_info or {}}

all_results = []

# C1: generate() baseline
def c1_generate():
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids_t, pixel_values=pixel_values_t,
            max_new_tokens=action_dim, do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=False)
    return gen.sequences[0, -action_dim:].detach().cpu().numpy()

all_results.append(run_test('C1_generate', 100, c1_generate))

# C2: manual greedy use_cache=False
def c2_manual_nocache():
    with torch.no_grad():
        past = None; cur_input = input_ids_t; generated = []
        for _ in range(action_dim):
            out = model(input_ids=cur_input, pixel_values=pixel_values_t if past is None else None,
                       past_key_values=past, use_cache=False, return_dict=True)
            past = out.past_key_values if hasattr(out, 'past_key_values') else None
            logits = out.logits[0, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated.append(int(next_token))
            cur_input = next_token
    return np.array(generated)

all_results.append(run_test('C2_manual_nocache', 100, c2_manual_nocache))

# C3: manual greedy use_cache=True
def c3_manual_cache():
    with torch.no_grad():
        past = None; cur_input = input_ids_t; generated = []
        for _ in range(action_dim):
            out = model(input_ids=cur_input, pixel_values=pixel_values_t if past is None else None,
                       past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            generated.append(int(next_token))
            cur_input = next_token
    return np.array(generated)

all_results.append(run_test('C3_manual_cache', 100, c3_manual_cache))

# C4: deterministic CUDA flags + generate()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
except:
    pass
print('[%s] Deterministic flags enabled' % time.strftime('%H:%M:%S'), flush=True)

all_results.append(run_test('C4_deterministic_generate', 100, c1_generate,
    {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'use_deterministic_algorithms': True}))

# C5: fixed hf_device_map (reload model with explicit device_map)
print('[%s] C5: reloading with fixed device_map...' % time.strftime('%H:%M:%S'), flush=True)
del model; torch.cuda.empty_cache()
fixed_map = {"": 0}  # single GPU
try:
    model = load_model(device_map=fixed_map)
    device = "cuda:0"
    # Re-prepare tensors on correct device
    input_ids_t = torch.from_numpy(frozen['input_ids']).to(device)
    pixel_values_t = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)
    open_token_ids = open_token_ids.to(device); close_token_ids = close_token_ids.to(device)
    all_results.append(run_test('C5_single_gpu', 30, c1_generate, {'device_map': str(fixed_map)}))
except Exception as e:
    print('C5 failed: %s' % str(e)[:100], flush=True)
    all_results.append({'name': 'C5_single_gpu', 'N': 0, 'deterministic': False, 'unique_tokens': -1,
        'frame_class': 'ERROR', 'extra': {'error': str(e)[:200]}})

# ── Summary ──
print('\n=== D0-C Matrix Summary ===', flush=True)
for r in all_results:
    print('  %-25s N=%-3d det=%-5s unique=%-3d arms=%-3d open=%-3d cls=%s' % (
        r['name'], r.get('N',0), r.get('deterministic','?'), r.get('unique_tokens','?'),
        r.get('unique_arm_prefixes','?'), r.get('open_count','?'), r.get('frame_class','?')), flush=True)

with open(os.path.join(OUT, 'd0c_matrix.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print('Output: %s/d0c_matrix.json' % OUT, flush=True)
