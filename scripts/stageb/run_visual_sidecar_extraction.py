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

    from transformers import AutoModel, AutoImageProcessor, AutoProcessor
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

    # Load DINOv2
    print('Loading DINOv2...')
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    print('DINOv2 loaded on %s' % device)

    # Load LIBERO
    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()['libero_object']()
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
        frame_steps = set()
        for w in windows:
            for pos in ['frame_start', 'frame_center', 'frame_end']:
                frame_steps.add(w[pos])
        frame_steps = sorted(frame_steps)

        print('\n[%d/%d] %s s%s seed=%s  (%d windows, %d frame steps)' %
              (ep_idx + 1, len(ep_list), task, sid, seed, len(windows), len(frame_steps)))

        # ── Run clean rollout ──
        try:
            env = task_suite.get_task_bddl_and_init(task, int(sid))
        except Exception as e:
            print('  SKIP: env init failed: %s' % e)
            continue

        obs = env.reset()
        prompt = official_prompt(task.replace('_', ' '))
        captured_frames = {}  # step -> PIL Image

        max_step = max(frame_steps) + 10
        done = False

        for step in range(max_step):
            # Capture frame if this step is requested
            if step in frame_steps and step not in captured_frames:
                img = obs.get('agentview_rgb')
                if img is None:
                    for k in ['agentview_image', 'image', 'rgb']:
                        if k in obs:
                            img = obs[k]; break
                if img is not None:
                    captured_frames[step] = img
                    frame_steps.discard(step)

            if done:
                break

            # Get OpenVLA action
            try:
                img_processed = get_libero_image_official(obs)
                inputs = processor_openvla(prompt, img_processed, return_tensors='pt').to(device)
                with torch.no_grad():
                    action = model_openvla.predict_action(
                        **inputs, unnorm_key=OFFICIAL_UNNORM_KEY_LIBERO_OBJECT, do_sample=False
                    )
                action = action.cpu().numpy().flatten()
            except Exception as e:
                print('  Model inference error at step %d: %s' % (step, e))
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

        # ── Extract DINOv2 embeddings for each window ──
        for w in windows:
            emb_row = {
                'pair_id': w['pair_id'], 'task_key': task,
                'state_id': str(sid), 'seed': str(seed),
                'window_start': str(w['window_start']),
                'window_end': str(w['window_end']),
            }
            for pos_label, step_key in [('start', 'frame_start'), ('center', 'frame_center'), ('end', 'frame_end')]:
                step = w[step_key]
                img = captured_frames.get(step)
                if img is not None:
                    # DINOv2 inference
                    pil_img = Image.fromarray(img if img.max() > 1 else (img * 255).astype(np.uint8))
                    inputs = dino_processor(images=pil_img, return_tensors='pt').to(device)
                    with torch.no_grad():
                        outputs = dino_model(**inputs)
                        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
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
