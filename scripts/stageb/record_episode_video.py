#!/usr/bin/env python3
"""Minimal video recorder: run MLP bridge, save agentview frames, compile mp4."""
import argparse, os, subprocess, sys, numpy as np
from pathlib import Path
os.environ.setdefault('OPENVLA_ATTN_IMPLEMENTATION', 'eager')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))

ap = argparse.ArgumentParser()
ap.add_argument('--task_idx', type=int, required=True)
ap.add_argument('--state_id', type=int, required=True)
ap.add_argument('--condition', required=True, choices=['CLEAN','VIS','RAND'])
ap.add_argument('--anchor', type=int, required=True)
ap.add_argument('--render_gpu', type=int, required=True)
ap.add_argument('--output_dir', required=True)
ap.add_argument('--fps', type=int, default=10)
args = ap.parse_args()

import torch
from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as AutoModelCls
except: from transformers import AutoModelForVision2Seq as AutoModelCls

MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
visible = torch.cuda.device_count()
model = AutoModelCls.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='auto',
    max_memory={idx: '10000MiB' for idx in range(visible)} | {'cpu': '128GiB'}, attn_implementation='eager')

from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime
MLP_PATH = '/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618/outputs/sc5_canonical_eng/sc5_mlp_s2.pt'
detector2 = SC5DetectorRuntime(MLP_PATH)

from gripper_attack.attack_adapter import OpenVLAVisualAttacker
attacker = None
if args.condition != 'CLEAN':
    is_rand = args.condition == 'RAND'
    opt = {'method':'token_prefix_pgd','objective':'autoregressive_prefix_gripper_target_token_logratio_arm_v3',
           'target_token_id':31744,'epsilon':0.02353,'num_steps':20,'step_size':0.02353*0.075,
           'random_start':True,'prefix_refresh_interval':1,'gripper_margin':5.0,'arm_preserve_weight':0.5,
           'arm_gate_min_match_count':5,'strict_route':True,'allow_fallback':False,
           'temporal_init':'prev_delta','target_execution_class':'CLIP_MEDIATED_OPEN'}
    if is_rand: opt['gradient_transform']='permute'; opt['gradient_transform_seed']=99+100000
    attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={'attack_optimizer':opt},
        seed=99, preprocess_kwargs={'libero_official_preprocess':False,
        'libero_preprocess_backend':'official_pil_lanczos','center_crop':True,'resize_size':224})

from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero
from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
from libero.libero import benchmark, get_libero_path
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
from scripts.v4_run_eval_openvla import physical_gripper_state

bm = benchmark.get_benchmark_dict(); suite = bm['libero_object']()
task_obj = suite.get_task(args.task_idx); init_states = suite.get_task_init_states(args.task_idx)
bddl = os.path.join(get_libero_path('bddl_files'), task_obj.problem_folder, task_obj.bddl_file)
instruction = task_obj.language
env, obs = build_v4_exact_env(bddl, args.render_gpu, 400, 10)
obs = env.set_init_state(init_states[args.state_id])
env, obs = apply_dummy_wait(env, obs, 10)

_streamer = SC5StreamingFeatureAdapterV2()
_mlp_emit = -1
_eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
_prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
attack_count = 0
frames_dir = Path(args.output_dir) / 'frames'
frames_dir.mkdir(parents=True, exist_ok=True)
frame_idx = 0

for step in range(400):
    img = obs['agentview_image']
    from PIL import Image; Image.fromarray(img.astype(np.uint8)).save(frames_dir / f'frame_{frame_idx:04d}.png')
    frame_idx += 1
    raw = np.asarray(img).copy()
    gs = physical_gripper_state(env, obs)
    q7 = float(gs['qpos'][0]) if gs and len(gs.get('qpos',[]))>0 else float('nan')
    q8 = float(gs['qpos'][1]) if gs and len(gs.get('qpos',[]))>1 else float('nan')
    eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('gripper0_grip_site')]
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
    action, _, _, _ = decode_with_scores(model, processor, 'cuda:0', raw, instruction,
        'libero_object', 8, libero_official_preprocess=False,
        libero_preprocess_backend='official_pil_lanczos', center_crop=True,
        resize_size=224, drop_attention_mask=True)
    env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
    raw_grip = float(action[-1]); env_grip = -1.0 if raw_grip > 0.5 else 1.0

    if not detector2.emitted:
        gripper_w = abs(q7)+abs(q8) if not (np.isnan(q7) or np.isnan(q8)) else float('nan')
        eef_valid = np.all(np.isfinite([eef_x,eef_y,eef_z]))
        _vx = eef_x-_prev_eef[0] if eef_valid else float('nan')
        _vy = eef_y-_prev_eef[1] if eef_valid else float('nan')
        _vz = eef_z-_prev_eef[2] if eef_valid else float('nan')
        if eef_valid: _prev_eef = (eef_x, eef_y, eef_z)
        try:
            _res = _streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=float(q7+q8) if not (np.isnan(q7) or np.isnan(q8)) else float('nan'),
                gripper_opening_proxy=gripper_w, eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=_vx, eef_vy=_vy, eef_vz=_vz, action_dx=float(action[0]),
                action_dy=float(action[1]), action_dz=float(action[2]), action_gripper=raw_grip)
        except: _res = {'valid':False}
        if _res.get('valid'):
            decision = detector2.update(_res['features'], step)
            if decision['emitted']: _mlp_emit = decision['emit_step']

    if args.condition != 'CLEAN' and _mlp_emit >= 0 and step >= _mlp_emit and attack_count < 10:
        attack_count += 1
        if args.condition == 'RAND':
            from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
            proc = prepare_openvla_image_for_attack(raw, libero_official_preprocess=False,
                libero_preprocess_backend='official_pil_lanczos', center_crop=True, resize_size=224)
            inp = processor(instruction, proc, return_tensors='pt')
            iids = inp['input_ids'].to('cuda:0')
            x = inp['pixel_values'].to(device='cuda:0', dtype=model.dtype)
            d = sample_processor_delta(x.shape, epsilon=6.0/255.0, seed=99+100000+attack_count,
                dtype=torch.float32, device=x.device)
            proj,_ = project_and_cast_processor_values(x, d, epsilon=6.0/255.0, candidate_is_delta=True)
            adv_pv = proj.detach().to(dtype=model.dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=adv_pv, max_new_tokens=8,
                    do_sample=False, return_dict_in_generate=True, output_scores=True)
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=8)
            na = model.bin_centers[np.clip(int(model.config.text_config.vocab_size-model.config.pad_to_multiple_of)-np.array([int(t) for t in adv_tokens])-1, 0, model.bin_centers.shape[0]-1)]
            s = model.get_action_stats('libero_object')
            lo=np.asarray(s['q01'],dtype=np.float32); hi=np.asarray(s['q99'],dtype=np.float32)
            mk=np.asarray(s.get('mask',np.ones_like(lo,dtype=bool)),dtype=bool)
            attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
            env_action_final = postprocess_openvla_action_for_libero(attack_action, enabled=True)
        else:
            from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
            clean_action_np = np.asarray(action, dtype=np.float32)
            proc = prepare_openvla_image_for_attack(raw, libero_official_preprocess=False,
                libero_preprocess_backend='official_pil_lanczos', center_crop=True, resize_size=224)
            inp = processor(instruction, proc, return_tensors='pt')
            iids = inp['input_ids'].to('cuda:0')
            pv = inp['pixel_values'].to(device='cuda:0', dtype=model.dtype)
            with torch.inference_mode():
                go = model.generate(input_ids=iids, pixel_values=pv, max_new_tokens=8,
                    do_sample=False, return_dict_in_generate=True, output_scores=True)
            from gripper_attack.v3_generation_parity import extract_exact_new_tokens
            clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=8)
            clean_gen = type('G',(),{})()
            clean_gen.sequences=torch.tensor([iids[0].detach().cpu().tolist()+[int(t) for t in clean_tokens]],
                dtype=torch.long, device='cuda:0'); clean_gen.scores=[]
            attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen,
                unnorm_key='libero_object')
            adv_inputs = get_adv_inputs_from_attack_result(attack_result)
            with torch.inference_mode():
                go_adv = model.generate(input_ids=iids, pixel_values=adv_inputs['pixel_values'].to(
                    device='cuda:0',dtype=model.dtype), max_new_tokens=8, do_sample=False,
                    return_dict_in_generate=True, output_scores=True)
            adv_tokens = extract_exact_new_tokens(go_adv.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=8)
            na = model.bin_centers[np.clip(int(model.config.text_config.vocab_size-model.config.pad_to_multiple_of)-np.array([int(t) for t in adv_tokens])-1, 0, model.bin_centers.shape[0]-1)]
            attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
            env_action_final = postprocess_openvla_action_for_libero(attack_action, enabled=True)

    obs, _, done, _ = env.step(env_action_final)
    if done: break

env.close()
out_mp4 = Path(args.output_dir) / f'{args.condition}_t{args.task_idx}_s{args.state_id}.mp4'
subprocess.run(['ffmpeg','-y','-framerate',str(args.fps),'-i',str(frames_dir/'frame_%04d.png'),
    '-c:v','libx264','-pix_fmt','yuv420p','-preset','fast',str(out_mp4)], check=True, capture_output=True)
print(f'Video: {out_mp4} ({frame_idx} frames, {args.fps} fps)')
