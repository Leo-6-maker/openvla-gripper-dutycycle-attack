#!/usr/bin/env python3
"""D0-A/B/C: Repeated greedy decode on frozen tensors.
Usage: python d0_repeat_decode.py --frozen FROZEN.npz --N 100 [--deterministic]
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--frozen', required=True)
ap.add_argument('--N', type=int, default=100)
ap.add_argument('--deterministic', action='store_true')
ap.add_argument('--output', default='')
args = ap.parse_args()

GPU_PAIR = os.environ.get('GPU_PAIR', '4,5')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_PAIR
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")

if args.deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('[%s] Deterministic CUDA flags enabled' % time.strftime('%H:%M:%S'), flush=True)

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO + '/scripts')

from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

print('[%s] Loading model...' % time.strftime('%H:%M:%S'), flush=True)
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
model_dtype = torch.bfloat16; unnorm_key = 'libero_object'
action_dim = int(model.get_action_dim(unnorm_key)); assert action_dim == 7

# Load frozen tensors
frozen = np.load(args.frozen, allow_pickle=True)
input_ids_t = torch.from_numpy(frozen['input_ids']).to(device)
pixel_values_t = torch.from_numpy(frozen['pixel_values']).to(device=device, dtype=model_dtype)
gripper_original = float(frozen['gripper'])
instruction = str(frozen['instruction'])
step = int(frozen['step'])

print('[%s] Frozen: step=%d gripper=%.1f N=%d' % (time.strftime('%H:%M:%S'), step, gripper_original, args.N), flush=True)
print('  input_ids hash: %s' % hashlib.sha256(frozen['input_ids'].tobytes()).hexdigest()[:16], flush=True)
print('  pixel_values hash: %s' % hashlib.sha256(frozen['pixel_values'].tobytes()).hexdigest()[:16], flush=True)

# Get region info for margin analysis
from gripper_attack.attack_adapter import OpenVLAVisualAttacker
dummy_attacker = OpenVLAVisualAttacker(model, processor, {
    'method': 'token_prefix_pgd', 'epsilon': 6/255, 'num_iter': 1,
    'token_label_source': 'prefix_locked_gripper_open_margin',
    'K_trigger': 8, 'use_restart': False, 'random_start': False,
}, device=device)
region_info = dummy_attacker.get_gripper_region_by_decoded_action(unnorm_key)
open_token_ids = region_info['open_token_ids']
close_token_ids = region_info['close_token_ids']
print('  OPEN tokens: %d, CLOSE tokens: %d' % (len(open_token_ids), len(close_token_ids)), flush=True)

# Run repeated decode
results = []
for i in range(args.N):
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids_t, pixel_values=pixel_values_t,
            max_new_tokens=action_dim, do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=True)
    tids = gen.sequences[0, -action_dim:].detach().cpu().numpy()
    tid_hash = hashlib.sha256(tids.tobytes()).hexdigest()
    gripper_token = int(tids[-1])
    arm_tokens = tids[:6].tolist()

    # Decode action
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - tids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    action = np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)
    env_gripper = -1.0 if action[-1] > 0.5 else (1.0 if action[-1] < -0.5 else 0.0)
    is_open = int(env_gripper < -0.5)

    if i == 0:
        # Get logits for margin on first decode
        out = model(input_ids=input_ids_t, pixel_values=pixel_values_t, use_cache=False, return_dict=True)
        logits = out.logits.float()
        gripper_row_idx = action_dim - 2  # predicts last action token
        gripper_row = logits[0, gripper_row_idx, :]
        max_open = float(gripper_row[open_token_ids].max())
        max_close = float(gripper_row[close_token_ids].max())
        margin = max_open - max_close
        print('  max_open=%.4f max_close=%.4f margin=%.4f' % (max_open, max_close, margin), flush=True)

    results.append({
        'rep': i, 'token_hash': tid_hash, 'gripper_token': gripper_token,
        'arm_tokens': arm_tokens, 'is_open': is_open, 'env_gripper': env_gripper,
    })

# Analysis
token_hashes = set(r['token_hash'] for r in results)
gripper_dist = Counter(r['gripper_token'] for r in results)
arm_hashes = set(hashlib.sha256(bytes(r['arm_tokens'])).hexdigest() for r in results)
open_count = sum(1 for r in results if r['is_open'])
close_count = args.N - open_count

print('\n=== D0 Results (N=%d) ===' % args.N, flush=True)
print('Unique 7-token sequences: %d/%d' % (len(token_hashes), args.N), flush=True)
print('Unique arm prefixes: %d' % len(arm_hashes), flush=True)
print('Gripper token distribution: %s' % dict(gripper_dist.most_common(10)), flush=True)
print('OPEN rate: %d/%d (%.1f%%)' % (open_count, args.N, 100*open_count/args.N), flush=True)
print('CLOSE rate: %d/%d (%.1f%%)' % (close_count, args.N, 100*close_count/args.N), flush=True)
print('Margin (first decode): max_open=%.4f max_close=%.4f margin=%.4f' % (max_open, max_close, margin), flush=True)

is_deterministic = len(token_hashes) == 1
print('VERDICT: %s' % ('DETERMINISTIC' if is_deterministic else 'NONDETERMINISTIC (%d unique)' % len(token_hashes)), flush=True)

# Frame classification
if open_count >= args.N * 0.9:
    frame_class = 'STABLE_OPEN'
elif close_count >= args.N * 0.9:
    frame_class = 'STABLE_CLOSE'
elif open_count >= args.N * 0.1 and close_count >= args.N * 0.1:
    frame_class = 'NUMERICALLY_UNSTABLE_BOUNDARY'
else:
    frame_class = 'MIXED'

print('FRAME CLASS: %s' % frame_class, flush=True)

# Save
out_data = {
    'N': args.N, 'deterministic': is_deterministic,
    'unique_tokens': len(token_hashes), 'unique_arm_prefixes': len(arm_hashes),
    'gripper_distribution': {str(k): v for k,v in gripper_dist.most_common()},
    'open_count': open_count, 'close_count': close_count,
    'max_open': max_open, 'max_close': max_close, 'margin': margin,
    'frame_class': frame_class,
    'frozen_step': step, 'frozen_gripper': gripper_original,
}
out_path = args.output if args.output else args.frozen.replace('.npz', '_d0.json')
with open(out_path, 'w') as f:
    json.dump(out_data, f, indent=2)
print('Output: %s' % out_path, flush=True)
