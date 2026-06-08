#!/usr/bin/env python3
"""Visual sidecar: clean-frame DINOv2 extraction for 72-pair pool.

Groups windows by episode (task, state, seed), runs ONE clean rollout per episode,
captures frames at all needed window positions, extracts DINOv2 ViT-B/14 embeddings.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/stageb/run_visual_sidecar_extraction.py \
    --labels /path/to/all_labels_rc1a_14cfabe_72pairs.csv \
    --output /data/liuyu/outputs/visual_sidecar_14cfabe_72pairs

Only uses GPU for DINOv2 inference (~1 GiB). Model rollouts use OpenVLA.
DO NOT use GPU 3/7.
"""
import csv, os, sys, argparse, json, time
import numpy as np
from collections import defaultdict
from PIL import Image

# Lazy GPU imports
_torch = None


def _lazy_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-episodes', type=int, default=0,
                    help='Cap episodes for quick smoke (0=all)')
    args = ap.parse_args()

    torch = _lazy_torch()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: %s' % device)
    print('GPU: %s' % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))

    # ── Load 72-pair labels ──
    with open(args.labels, 'r') as f:
        labels = list(csv.DictReader(f))
    print('Loaded %d pairs' % len(labels))

    # ── Group by episode ──
    episodes = defaultdict(list)
    for r in labels:
        ep = (r['task_key'], r['state_id'], r['seed'])
        ws = int(r['window_start']); we = int(r['window_end'])
        wc = (ws + we) // 2
        episodes[ep].append({
            'pair_id': r['pair_id'],
            'window_start': ws, 'window_end': we, 'window_center': wc,
            'frame_start': max(0, ws - 2),
            'frame_center': wc,
            'frame_end': min(int(r.get('actual_max_step', '299')), we + 2),
        })

    ep_list = sorted(episodes.items())
    if args.max_episodes > 0:
        ep_list = ep_list[:args.max_episodes]
    print('Episodes: %d (%.1f hours sequential)' % (len(ep_list), len(ep_list) * 10 / 60))

    # ── Setup LIBERO + OpenVLA ──
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    MODEL_PATH = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from gripper_attack.openvla_libero_exec_spec import (
        official_prompt, get_libero_image_official,
        OFFICIAL_UNNORM_KEY_LIBERO_OBJECT,
    )

    # Load OpenVLA with device_map='auto' for multi-GPU model parallelism
    print('Loading OpenVLA...')
    from transformers import AutoModelForVision2Seq
    gpu_ids = [0, 1] if torch.cuda.device_count() >= 2 else [0]
    model_openvla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        max_memory={gpu_ids[0]: '10500MiB', gpu_ids[1]: '10500MiB', 'cpu': '64GiB'} if len(gpu_ids) >= 2 else {0: '10500MiB', 'cpu': '64GiB'},
        trust_remote_code=True,
    ).eval()
    processor_openvla = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print('OpenVLA loaded')

    # Use OpenVLA's built-in SigLIP vision backbone (already loaded, no download needed)
    vision_backbone = model_openvla.vision_backbone
    print('Using OpenVLA SigLIP vision backbone (2176-dim)')

    # Load LIBERO
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    TASK_CFG = {
        'ketchup': 0, 'butter': 1, 'cream_cheese': 2, 'salad_dressing': 3,
        'bbq_sauce': 4, 'milk': 5, 'alphabet_soup': 6, 'tomato_sauce': 7, 'orange_juice': 8,
    }
    bm_dict = benchmark.get_benchmark_dict()
    task_suite = bm_dict['libero_object']()
    _render_gpu = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    print('LIBERO loaded')

    # ── Output dirs ──
    os.makedirs(args.output, exist_ok=True)
    frame_dir = os.path.join(args.output, 'frames')
    emb_dir = os.path.join(args.output, 'embeddings')
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    # ── Process episodes ──
    feature_rows = []
    t_start = time.time()

    for ep_idx, (ep, windows) in enumerate(ep_list):
        task, sid, seed = ep
        t_ep = time.time()

        # Collect all needed frame steps for this episode
        frame_steps_set = set()
        for w in windows:
            for pos in ['frame_start', 'frame_center', 'frame_end']:
                frame_steps_set.add(w[pos])
        frame_steps_sorted = sorted(frame_steps_set)

        print('\n[%d/%d] %s s%s seed=%s  (%d windows, %d frame steps)' %
              (ep_idx + 1, len(ep_list), task, sid, seed, len(windows), len(frame_steps_sorted)))

        # ── Init env (matching runner pattern) ──
        cfg = TASK_CFG.get(task)
        if cfg is None:
            print('  SKIP: unknown task %s' % task)
            continue
        try:
            task_obj = task_suite.get_task(cfg)
            initial_states = task_suite.get_task_init_states(cfg)
            if int(sid) >= len(initial_states):
                print('  SKIP: state %s OOB (max %d)' % (sid, len(initial_states)))
                continue
            bddl = os.path.join(get_libero_path('bddl_files'),
                                task_obj.problem_folder, task_obj.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                has_renderer=False, has_offscreen_renderer=True,
                use_camera_obs=True, camera_names=['agentview'], control_freq=20,
                render_gpu_device_id=_render_gpu)
            env.seed(int(seed))
            obs = env.reset()
            env.sim.data.qvel[:] = 0
            env.sim.forward()
            env.set_init_state(initial_states[int(sid)])
        except Exception as e:
            print('  SKIP: env init failed: %s' % e)
            continue

        obs = env.reset()
        prompt = official_prompt(task.replace('_', ' '))
        captured_frames = {}  # step -> PIL Image

        max_step = max(frame_steps_sorted) + 10
        done = False

        for step in range(max_step):
            # Capture frame if this step is requested
            if step in frame_steps_set and step not in captured_frames:
                img = obs.get('agentview_image', obs.get('agentview_rgb'))
                if img is not None:
                    captured_frames[step] = img
                    frame_steps_set.discard(step)

            if done:
                break

            # Get OpenVLA action (matching runner: np.array -> PIL -> processor)
            try:
                img_np = get_libero_image_official(obs)
                img_pil = Image.fromarray(img_np.astype(np.uint8))
                inputs = processor_openvla(prompt, img_pil)
                for k in list(inputs.keys()):
                    v = inputs[k]
                    if isinstance(v, torch.Tensor) and v.dtype != model_openvla.dtype:
                        inputs[k] = v.to(dtype=model_openvla.dtype)
                inputs = {k: v.to(model_openvla.device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    action = model_openvla.predict_action(
                        **inputs, unnorm_key=OFFICIAL_UNNORM_KEY_LIBERO_OBJECT, do_sample=False
                    )
                action = action.float().cpu().numpy().flatten()
            except Exception as e:
                print('  Model inference error at step %d: %s' % (step, str(e)[:80]))
                action = np.zeros(7, dtype=np.float32)

            obs, reward, done, info = env.step(action)

        env.close()

        # ── Save frames as PNG ──
        ep_tag = '%s_s%s_seed%s' % (task, sid, seed)
        for step, img_np in captured_frames.items():
            fname = '%s_step%04d.png' % (ep_tag, step)
            fpath = os.path.join(frame_dir, fname)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            Image.fromarray(img_np).save(fpath)

        # ── Extract OpenVLA vision embeddings for each window ──
        for w in windows:
            emb_row = {
                'pair_id': w['pair_id'], 'task_key': task,
                'state_id': str(sid), 'seed': str(seed),
                'window_start': str(w['window_start']),
                'window_end': str(w['window_end']),
            }
            for pos_label, step_key in [('start', 'frame_start'), ('center', 'frame_center'), ('end', 'frame_end')]:
                step_id = w[step_key]
                img = captured_frames.get(step_id)
                if img is not None:
                    # Preprocess: resize to 224x224, normalize
                    pil_img = Image.fromarray(img if img.max() > 1 else (img * 255).astype(np.uint8))
                    pil_img = pil_img.resize((224, 224), Image.BICUBIC)
                    arr = np.array(pil_img).astype(np.float32) / 255.0
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    tensor_3ch = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(
                        model_openvla.device).to(model_openvla.dtype)
                    # Duplicate: OpenVLA expects [B, 6, H, W] (3 raw + 3 fused)
                    tensor_6ch = torch.cat([tensor_3ch, tensor_3ch], dim=1)
                    with torch.no_grad():
                        viz_out = vision_backbone(tensor_6ch)
                    # Mean pool across 256 tokens -> 2176-dim embedding
                    emb = viz_out.mean(dim=1).float().cpu().numpy().flatten()
                    fname = '%s_%s.npy' % (w['pair_id'], pos_label)
                    np.save(os.path.join(emb_dir, fname), emb)
                    emb_row['emb_%s_file' % pos_label] = fname
                    emb_row['emb_%s_dim' % pos_label] = str(len(emb))

            feature_rows.append(emb_row)

        elapsed_ep = (time.time() - t_ep) / 60
        total_elapsed = (time.time() - t_start) / 60
        eta = total_elapsed / (ep_idx + 1) * (len(ep_list) - ep_idx - 1)
        print('  Done in %.1f min (total %.1f min, ETA %.1f min)' % (elapsed_ep, total_elapsed, eta))

    # ── Save feature index ──
    index_path = os.path.join(args.output, 'feature_index.csv')
    if feature_rows:
        with open(index_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
            w.writeheader()
            w.writerows(feature_rows)

    n_with_emb = sum(1 for r in feature_rows if 'emb_center_file' in r)
    total_time = (time.time() - t_start) / 60
    print('\n=== VISUAL SIDECAR COMPLETE ===')
    print('Episodes: %d  Windows: %d  With embeddings: %d' %
          (len(ep_list), len(feature_rows), n_with_emb))
    print('Total time: %.1f min' % total_time)
    print('Index: %s' % index_path)
    print('Frames: %s/' % frame_dir)
    print('Embeddings: %s/' % emb_dir)


if __name__ == '__main__':
    main()
