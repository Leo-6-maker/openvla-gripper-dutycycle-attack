#!/usr/bin/env python3
"""D0 Step 1: Walk to capture frozen CLOSE + OPEN tensors."""
import hashlib, os, sys, time
import numpy as np
import torch

GPU_PAIR = os.environ.get('GPU_PAIR', '4,5')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_PAIR
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("OPENVLA_CUDA_MAX_MEMORY", "10000MiB")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO + '/scripts')

from gripper_attack.libero_v4_env_factory import (
    build_v4_exact_env, apply_dummy_wait, seed_everything, TARGET_OBJECT_GUESS_V4)
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls
from v4_run_eval_openvla import postprocess_openvla_action_for_libero
from PIL import Image
from libero.libero import benchmark, get_libero_path

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
RENDER_GPU = 4
TASK = 'cream_cheese'; STATE_ID = 35; TASK_IDX = 1

OUT = os.environ.get('OUT_DIR', '/data/liuyu/outputs/stageb_v5_day2_c2o_mechanism_20260613/determinism')
os.makedirs(OUT, exist_ok=True)

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
device_map_snap = {str(k): str(v) for k,v in model.hf_device_map.items()} if hasattr(model, "hf_device_map") else {}
print('[%s] Model ready. Device map: %s' % (time.strftime('%H:%M:%S'), str(device_map_snap)[:200]), flush=True)

bm = benchmark.get_benchmark_dict(); task_suite = bm['libero_object']()
ti = TASK_IDX; task_obj = task_suite.get_task(ti); init_states = task_suite.get_task_init_states(ti)
bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language

def decode_action(token_ids):
    v = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    d = np.clip(v - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    na = model.bin_centers[d]
    stats = model.get_action_stats(unnorm_key)
    mk = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
    return np.where(mk, 0.5 * (na + 1) * (hi - lo) + lo, na).astype(np.float32)

seed_everything(0)
env, obs = build_v4_exact_env(bddl_file, RENDER_GPU, 120, num_steps_wait=10)
obs = env.set_init_state(init_states[STATE_ID]); env, obs = apply_dummy_wait(env, obs, 10)

close_frames = []; open_frames = []
for step in range(120):
    obs['agentview_image']; img_uint8 = obs['agentview_image'].copy()
    img_pil = Image.fromarray(img_uint8)
    inputs = processor(text=instruction, images=img_pil, return_tensors='pt')
    inputs = {k: v.to(device=device, dtype=model_dtype if v.dtype in (torch.float32, torch.bfloat16) else v.dtype) for k,v in inputs.items()}
    input_ids_cpu = inputs['input_ids'].detach().cpu().numpy().copy()
    pixel_values_cpu = inputs['pixel_values'].detach().cpu().numpy().copy()

    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, num_beams=1, return_dict_in_generate=True, output_scores=True)
    tids = gen_out.sequences[0, -action_dim:].detach().cpu().numpy()
    clean_a = decode_action(tids)
    env_a = postprocess_openvla_action_for_libero(clean_a, TARGET_OBJECT_GUESS_V4.get(TASK, TASK))
    gripper = float(env_a[-1])
    is_close = int(gripper > 0.5)

    if step >= 10:
        if is_close and len(close_frames) < 2:
            close_frames.append((step, img_uint8, input_ids_cpu, pixel_values_cpu, tids, gripper))
            print('[%s] CLOSE frame at step %d, gripper=%.1f' % (time.strftime('%H:%M:%S'), step, gripper), flush=True)
        if not is_close and len(open_frames) < 1:
            open_frames.append((step, img_uint8, input_ids_cpu, pixel_values_cpu, tids, gripper))
            print('[%s] OPEN frame at step %d, gripper=%.1f' % (time.strftime('%H:%M:%S'), step, gripper), flush=True)
        if len(close_frames) >= 2 and len(open_frames) >= 1:
            break
    obs, _, _, _ = env.step(env_a)
env.close()

# Save
for label, frames in [('close', close_frames), ('open', open_frames)]:
    for i, (step, img, ids, pv, tids, grip) in enumerate(frames):
        prefix = '%s_%d' % (label, i)
        rgb_hash = hashlib.sha256(img.tobytes()).hexdigest()
        ids_hash = hashlib.sha256(ids.tobytes()).hexdigest()
        pv_hash = hashlib.sha256(pv.tobytes()).hexdigest()
        np.savez(os.path.join(OUT, 'frozen_%s.npz' % prefix),
            img_uint8=img, input_ids=ids, pixel_values=pv,
            instruction=instruction, step=step, token_ids=tids, gripper=grip)
        print('SAVED %s: step=%d rgb=%s ids=%s pv=%s grip=%.1f tokens=%s' % (
            prefix, step, rgb_hash[:16], ids_hash[:16], pv_hash[:16], grip, tids.tolist()), flush=True)

print('CAPTURE DONE', flush=True)
