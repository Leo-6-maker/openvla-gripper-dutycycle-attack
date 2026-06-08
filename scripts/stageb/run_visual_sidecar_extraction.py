#!/usr/bin/env python3
"""Visual sidecar: clean-frame feature extraction for 72-pair pool.

Aligns with RC1a official execution path:
  1. Real task language from task_suite (not task_key.replace)
  2. Correct env init: reset -> set_init_state -> obs (no double reset)
  3. normalize_gripper_action(binarize=True) + invert_gripper_action before env.step
  4. Captured frames use get_libero_image_official() (rot180)
  5. Vision features via official AutoProcessor pixel_values, not manual 6-ch
  6. Dynamic task index (no hardcoded TASK_CFG)

Usage:
  CUDA_VISIBLE_DEVICES=1,0 python -u scripts/stageb/run_visual_sidecar_extraction.py \
    --labels .../all_labels_rc1a_14cfabe_72pairs.csv \
    --output .../visual_sidecar_14cfabe_72pairs \
    --max-episodes 2   # smoke first
"""
import csv, os, sys, argparse, time
import numpy as np
from collections import defaultdict
from PIL import Image

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-episodes', type=int, default=0)
    ap.add_argument('--extract-features', action='store_true', default=True)
    args = ap.parse_args()

    device = 'cuda:0'
    print('Device: %s | GPU: %s' % (device, torch.cuda.get_device_name(0)))

    # ── Load 72-pair labels ──
    with open(args.labels, 'r') as f:
        labels = list(csv.DictReader(f))
    print('Loaded %d pairs' % len(labels))

    # ── Group by episode (task, state_id, seed) ──
    episodes = defaultdict(list)
    for r in labels:
        ep = (r['task_key'], r['state_id'], r['seed'])
        ws = int(r['window_start']); we = int(r['window_end']); wc = (ws + we) // 2
        max_s = int(r.get('actual_max_step', '299'))
        episodes[ep].append({
            'pair_id': r['pair_id'],
            'ws': ws, 'we': we, 'wc': wc,
            'frame_start': max(0, ws - 2),
            'frame_center': wc,
            'frame_end': min(max_s, we + 2),
        })

    ep_list = sorted(episodes.items())
    if args.max_episodes > 0:
        ep_list = ep_list[:args.max_episodes]
    print('Episodes: %d' % len(ep_list))

    # ── Setup ──
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from gripper_attack.openvla_libero_exec_spec import (
        official_prompt, get_libero_image_official,
        normalize_gripper_raw, raw_gripper_to_env_gripper,
        OFFICIAL_UNNORM_KEY_LIBERO_OBJECT,
    )

    # P0.3: gripper postprocess helpers (matching spec)
    def normalize_gripper_action(action, binarize=True):
        """normalize_gripper_action from spec."""
        import copy
        action = copy.deepcopy(action)
        if binarize:
            action[..., -1] = np.where(action[..., -1] > 0, 1.0, 0.0)
        action[..., -1] = (action[..., -1] - 0.5) * 2.0
        return action

    def invert_gripper_action(action):
        """invert_gripper_action from spec."""
        import copy
        action = copy.deepcopy(action)
        action[..., -1] *= -1.0
        return action

    # Load OpenVLA
    print('Loading OpenVLA...')
    gpu_ids = [0, 1] if torch.cuda.device_count() >= 2 else [0]
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map='auto',
        max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'},
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vision_backbone = model.vision_backbone
    print('OpenVLA + SigLIP backbone loaded (2176-dim)')

    # P1: Dynamic task index from task_suite (no hardcoded TASK_CFG)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bm_dict = benchmark.get_benchmark_dict()
    task_suite = bm_dict['libero_object']()
    _render_gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])

    # Build task -> cfg dynamically
    task_names = [task_suite.get_task(i).name for i in range(task_suite.n_tasks)]
    task_to_cfg = {}
    for i in range(task_suite.n_tasks):
        t = task_suite.get_task(i)
        # Map bddl_file basename -> cfg index (matching runner convention)
        bddl_basename = os.path.splitext(t.bddl_file)[0]
        for known in ['ketchup', 'butter', 'cream_cheese', 'salad_dressing',
                       'bbq_sauce', 'milk', 'alphabet_soup', 'tomato_sauce', 'orange_juice']:
            if known in bddl_basename.lower():
                task_to_cfg[known] = i
                break
        else:
            task_to_cfg[bddl_basename] = i
    print('Task mapping: %d tasks' % len(task_to_cfg))
    print('LIBERO loaded')

    # ── Output dirs ──
    os.makedirs(args.output, exist_ok=True)
    frame_dir = os.path.join(args.output, 'frames')
    emb_dir = os.path.join(args.output, 'embeddings')
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    feature_rows = []
    t_start = time.time()

    for ep_idx, (ep, windows) in enumerate(ep_list):
        task, sid_str, seed_str = ep
        sid = int(sid_str); seed = int(seed_str)
        t_ep = time.time()

        # Collect needed frame steps
        frame_steps_set = set()
        for w in windows:
            for pos in ['frame_start', 'frame_center', 'frame_end']:
                frame_steps_set.add(w[pos])
        frame_steps_sorted = sorted(frame_steps_set)

        print('\n[%d/%d] %s s%d seed=%d  (%d windows, %d frame steps)' %
              (ep_idx + 1, len(ep_list), task, sid, seed, len(windows), len(frame_steps_sorted)))

        # ── P1: Dynamic task cfg ──
        cfg = task_to_cfg.get(task)
        if cfg is None:
            print('  SKIP: task %s not in task_suite mapping' % task)
            continue

        # ── P0.1 + P0.2: Correct env init ──
        try:
            task_obj = task_suite.get_task(cfg)
            initial_states = task_suite.get_task_init_states(cfg)
            if sid >= len(initial_states):
                print('  SKIP: state %d OOB (max %d)' % (sid, len(initial_states)))
                continue
            # P0.1: Real task language from task_suite
            instruction = task_obj.language if hasattr(task_obj, 'language') and task_obj.language else task
            bddl = os.path.join(get_libero_path('bddl_files'),
                                task_obj.problem_folder, task_obj.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                has_renderer=False, has_offscreen_renderer=True,
                use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                render_gpu_device_id=_render_gpu)
            env.seed(seed)
            obs = env.reset()
            env.sim.data.qvel[:] = 0
            env.sim.forward()
            # P0.2: set_init_state — NO second reset
            obs = env.set_init_state(initial_states[sid])
        except Exception as e:
            print('  SKIP: env init failed: %s' % e)
            continue

        # P0.1: Official prompt with real task language
        prompt_text = official_prompt(instruction.lower())
        print('  Task language: "%s"' % instruction)

        captured_frames = {}
        max_step = max(frame_steps_sorted) + 10
        done = False

        for step in range(max_step):
            # ── P0.4: Capture official_rot180 frame ──
            if step in frame_steps_set and step not in captured_frames:
                try:
                    official_img = get_libero_image_official(obs)
                    captured_frames[step] = official_img  # numpy array, rot180
                    frame_steps_set.discard(step)
                except Exception:
                    pass

            if done:
                break

            # ── Get action through RC1a execution chain ──
            try:
                # P0.5: Use official processor for vision features
                img_pil = Image.fromarray(
                    get_libero_image_official(obs).astype(np.uint8))
                inputs = processor(prompt_text, img_pil)
                # Convert dtypes to match model
                for k in list(inputs.keys()):
                    v = inputs[k]
                    if isinstance(v, torch.Tensor) and v.dtype != model.dtype:
                        inputs[k] = v.to(dtype=model.dtype)
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    raw_action = model.predict_action(
                        **inputs, unnorm_key=OFFICIAL_UNNORM_KEY_LIBERO_OBJECT,
                        do_sample=False)
                raw_action = raw_action.float().cpu().numpy().flatten()
            except Exception as e:
                print('  Model error at step %d: %s' % (step, str(e)[:80]))
                raw_action = np.zeros(7, dtype=np.float32)

            # P0.3: Official gripper postprocess chain
            env_action = normalize_gripper_action(raw_action.copy(), binarize=True)
            env_action = invert_gripper_action(env_action)
            obs, reward, done, info = env.step(env_action)

        env.close()

        # ── Save frames ──
        ep_tag = '%s_s%d_seed%d' % (task, sid, seed)
        for step_id, img_np in captured_frames.items():
            fname = '%s_step%04d.png' % (ep_tag, step_id)
            fpath = os.path.join(frame_dir, fname)
            # img_np is already rot180 from get_libero_image_official
            Image.fromarray(img_np.astype(np.uint8)).save(fpath)

        # ── Extract vision embeddings via official processor ──
        for w in windows:
            emb_row = {
                'pair_id': w['pair_id'], 'task_key': task,
                'state_id': str(sid), 'seed': str(seed_str),
                'window_start': str(w['ws']), 'window_end': str(w['we']),
            }
            for pos_label, step_key in [('start', 'frame_start'),
                                         ('center', 'frame_center'),
                                         ('end', 'frame_end')]:
                step_id = w[step_key]
                img_np = captured_frames.get(step_id)
                if img_np is not None:
                    # P0.5: Use official processor for vision features
                    # img_np is already rot180 from get_libero_image_official
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                    proc_inputs = processor(prompt_text, img_pil)
                    # processor returns pixel_values with proper normalization
                    pv = proc_inputs.get('pixel_values')
                    if pv is not None:
                        pv = pv.to(dtype=model.dtype).to(model.device)
                        with torch.no_grad():
                            viz_out = vision_backbone(pv)
                        # Mean pool across tokens -> 2176-dim
                        emb = viz_out.mean(dim=1).float().cpu().numpy().flatten()
                    else:
                        emb = np.zeros(2176, dtype=np.float32)
                    fname = '%s_%s.npy' % (w['pair_id'], pos_label)
                    np.save(os.path.join(emb_dir, fname), emb)
                    emb_row['emb_%s_file' % pos_label] = fname
                    emb_row['emb_%s_dim' % pos_label] = str(len(emb))

            feature_rows.append(emb_row)

        elapsed = (time.time() - t_ep) / 60
        total_elapsed = (time.time() - t_start) / 60
        eta = total_elapsed / (ep_idx + 1) * (len(ep_list) - ep_idx - 1)
        print('  Done %.1f min (total %.1f, ETA %.1f)' % (elapsed, total_elapsed, eta))

    # ── Save feature index ──
    index_path = os.path.join(args.output, 'feature_index.csv')
    if feature_rows:
        with open(index_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            w.writeheader()
            w.writerows(feature_rows)

    n_emb = sum(1 for r in feature_rows if 'emb_center_file' in r)
    print('\n=== DONE ===')
    print('Episodes: %d  Windows: %d  With embeddings: %d' %
          (len(ep_list), len(feature_rows), n_emb))
    print('Time: %.1f min' % ((time.time() - t_start) / 60))
    print('Index: %s' % index_path)


if __name__ == '__main__':
    main()
