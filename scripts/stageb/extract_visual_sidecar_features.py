#!/usr/bin/env python3
"""Visual sidecar feature extractor — DINOv2 + CLIP from clean rollout frames.

Offline sidecar: replays clean rollouts to specified frames, extracts frozen
visual embeddings WITHOUT running any attack.  Does NOT touch main runner.

Usage:
  python scripts/stageb/extract_visual_sidecar_features.py \
    --manifest tables/visual_sidecar_manifest_d4a3827.csv \
    --output features/stageb_v1_1_visual_sidecar_d4a3827/

Requires: pip install transformers torch torchvision pillow
GPU: optional (DINOv2 ViT-B/14 fits in < 1 GiB)
"""
import csv, os, sys, argparse, json
import numpy as np
from datetime import datetime

# Lazy imports — only load when actually extracting
_torch = None
_DinoVisionTransformer = None


def _lazy_import():
    global _torch, _DinoVisionTransformer
    if _torch is None:
        import torch as _t
        _torch = _t
    if _DinoVisionTransformer is None:
        from transformers import AutoImageProcessor, AutoModel
        _DinoVisionTransformer = (AutoImageProcessor, AutoModel)


def load_dino():
    _lazy_import()
    AutoImageProcessor, AutoModel = _DinoVisionTransformer
    device = 'cuda:0' if _torch.cuda.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    return processor, model, device


LBR_IMPORTED = False


def load_libero():
    global LBR_IMPORTED
    if not LBR_IMPORTED:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        LBR_IMPORTED = True
    # We use the same LIBERO loading as run_stageb_vis_labeling
    import libero
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    return task_suite


def capture_frame(env, task_i, step):
    """Advance env to step and return agentview RGB as numpy (H,W,3) uint8."""
    obs = env.reset()
    for s in range(step):
        action = np.zeros(7, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        if done:
            break
    img = obs.get('agentview_rgb')
    if img is None:
        # Try alternative keys
        for k in ['agentview_image', 'image', 'rgb']:
            if k in obs:
                img = obs[k]
                break
    return img


def extract_dino_features(processor, model, device, img_np):
    """Extract DINOv2 CLS embedding from a single RGB image (H,W,3)."""
    from PIL import Image
    if img_np is None:
        return None
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    inputs = processor(images=pil_img, return_tensors='pt').to(device)
    with _torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return cls_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', choices=['dinov2', 'clip', 'both'], default='dinov2')
    ap.add_argument('--save-frames', action='store_true', default=False,
                    help='Also save PNG frames (larger output)')
    args = ap.parse_args()

    # ── Load manifest ──
    with open(args.manifest, 'r') as f:
        manifest = list(csv.DictReader(f))
    print('Manifest: %d windows' % len(manifest))

    # ── Load DINOv2 ──
    processor, model, device = load_dino()
    print('DINOv2 loaded on %s' % device)

    # ── Load LIBERO ──
    task_suite = load_libero()
    print('LIBERO object suite loaded')

    # ── Prepare output ──
    os.makedirs(args.output, exist_ok=True)
    emb_dir = os.path.join(args.output, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    if args.save_frames:
        frame_dir = os.path.join(args.output, 'frames')
        os.makedirs(frame_dir, exist_ok=True)

    # ── Extract ──
    results = []
    task_cache = {}  # cache env per task

    for i, row in enumerate(manifest):
        task = row['task_key']
        sid = int(row['state_id'])
        ws = int(row['window_start'])
        we = int(row['window_end'])
        fs = int(row['frame_start'])
        fc = int(row['window_center'])
        fe = int(row['frame_end'])

        print('[%d/%d] %s s%d [%d,%d] frames=%d,%d,%d' %
              (i + 1, len(manifest), task, sid, ws, we, fs, fc, fe))

        # Get or create env for this task
        cache_key = (task, sid)
        if cache_key not in task_cache:
            try:
                env = task_suite.get_task_bddl_and_init(task, sid)
                task_cache[cache_key] = env
            except Exception as e:
                print('  SKIP: env init failed: %s' % e)
                continue
        else:
            env = task_cache[cache_key]

        # Capture frames at start, center, end
        frames = {}
        for label, step in [('start', fs), ('center', fc), ('end', fe)]:
            try:
                img = capture_frame(env, task, step)
                frames[label] = img
            except Exception as e:
                print('  WARN: frame %s at step %d failed: %s' % (label, step, e))

        # Extract DINOv2 embeddings
        emb_row = {
            'manifest_idx': str(i),
            'task_key': task, 'state_id': str(sid),
            'window_start': str(ws), 'window_end': str(we),
            'frame_start': str(fs), 'frame_center': str(fc), 'frame_end': str(fe),
        }
        for label in ['start', 'center', 'end']:
            img = frames.get(label)
            if img is not None:
                emb = extract_dino_features(processor, model, device, img)
                if emb is not None:
                    fname = '%s_%s_s%d_w%d_%d_%s.npy' % (task, label, sid, ws, we)
                    np.save(os.path.join(emb_dir, fname), emb)
                    emb_row['emb_%s_dim' % label] = str(len(emb))
                    emb_row['emb_%s_file' % label] = fname
                if args.save_frames:
                    from PIL import Image as PILImage
                    fname = '%s_%s_s%d_w%d_%d_%s.png' % (task, label, sid, ws, we)
                    PILImage.fromarray(img if img.max() > 1 else (img * 255).astype(np.uint8)
                                     ).save(os.path.join(frame_dir, fname))
        results.append(emb_row)

    # ── Save index ──
    index_path = os.path.join(args.output, 'feature_index.csv')
    if results:
        with open(index_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print('\nDone: %d windows, %d with embeddings' %
          (len(results), sum(1 for r in results if 'emb_center_file' in r)))
    print('Index: %s' % index_path)


if __name__ == '__main__':
    main()
