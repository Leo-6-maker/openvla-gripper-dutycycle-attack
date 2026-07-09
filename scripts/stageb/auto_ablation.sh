#!/bin/bash
# Auto A/B/C/D ablation pipeline — runs unattended on server
set -e
B=/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final
R=/mnt/sdc/dty_user/openvla_attack
V=$R/envs/openvla-official-a800/bin/python
NPZ=$B/c2f_w16_openvla_siglip_full_dataset.npz
cd $R
LOG=$B/auto_ablation.log
exec > >(tee -a $LOG) 2>&1
echo "=== Auto Ablation Start: $(date) ==="

# Step 0: Full SigLIP training (if not done)
if [ ! -f $B/train_full/c2f_training_report.json ]; then
  echo "[$(date +%H:%M:%S)] Running full SigLIP training (GPU 0)..."
  CUDA_VISIBLE_DEVICES=0 $V tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
    --dataset $NPZ --output-dir $B/train_full --batch-size 64 --epochs 10 \
    --lr 1e-3 --hidden 128 --proj 128 --device cuda --seed 42 --git-commit 68f9a3a
  echo "[$(date +%H:%M:%S)] Full training DONE"
fi

# Step 1: A baseline (zero visual+lang)
ZERO_NPZ=$B/c2f_w16_zeroed_vislang.npz
if [ ! -f $ZERO_NPZ ]; then
  echo "[$(date +%H:%M:%S)] Creating zeroed-visual-lang NPZ for A..."
  python3 -c "
import numpy as np
d = dict(np.load('$NPZ', allow_pickle=True))
d['X_visual'] = np.zeros_like(d['X_visual'], dtype=np.float16)
d['X_language'] = np.zeros_like(d['X_language'], dtype=np.float16)
np.savez_compressed('$ZERO_NPZ', **d)
print('Zeroed NPZ ready:', len(d['y_primary']), 'windows')
"
fi
echo "[$(date +%H:%M:%S)] Training A baseline (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $V tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
  --dataset $ZERO_NPZ --output-dir $B/train_A --batch-size 64 --epochs 10 \
  --lr 1e-3 --hidden 128 --proj 128 --device cuda --seed 42 --git-commit 68f9a3a
echo "[$(date +%H:%M:%S)] A baseline DONE"

# Step 2: C (visual only)
echo "[$(date +%H:%M:%S)] Creating visual-only NPZ for C..."
VIS_NPZ=$B/c2f_w16_vis_only.npz
python3 -c "
import numpy as np
d = dict(np.load('$NPZ', allow_pickle=True))
d['X_language'] = np.zeros_like(d['X_language'], dtype=np.float16)
np.savez_compressed('$VIS_NPZ', **d)
print('Vis-only NPZ ready')
"
echo "[$(date +%H:%M:%S)] Training C (visual only, GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $V tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
  --dataset $VIS_NPZ --output-dir $B/train_C --batch-size 64 --epochs 10 \
  --lr 1e-3 --hidden 128 --proj 128 --device cuda --seed 42 --git-commit 68f9a3a
echo "[$(date +%H:%M:%S)] C DONE"

# Step 3: B (language only)
echo "[$(date +%H:%M:%S)] Creating language-only NPZ for B..."
LANG_NPZ=$B/c2f_w16_lang_only.npz
python3 -c "
import numpy as np
d = dict(np.load('$NPZ', allow_pickle=True))
d['X_visual'] = np.zeros_like(d['X_visual'], dtype=np.float16)
np.savez_compressed('$LANG_NPZ', **d)
print('Lang-only NPZ ready')
"
echo "[$(date +%H:%M:%S)] Training B (language only, GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $V tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
  --dataset $LANG_NPZ --output-dir $B/train_B --batch-size 64 --epochs 10 \
  --lr 1e-3 --hidden 128 --proj 128 --device cuda --seed 42 --git-commit 68f9a3a
echo "[$(date +%H:%M:%S)] B DONE"

# Summary
echo "=== Building ablation summary ==="
python3 << 'PYEOF'
import json, os
B = "/mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final"
variants = [
    ("D_full",    B + "/train_full"),
    ("A_baseline", B + "/train_A"),
    ("C_visual",  B + "/train_C"),
    ("B_language", B + "/train_B"),
]
print("Variant       | Recall | FP    | F1     | L10_rec | Obj_rec | bestF1_FP")
print("--------------|--------|-------|--------|---------|---------|----------")
for name, d in variants:
    rp = d + "/c2f_training_report.json"
    if os.path.exists(rp):
        r = json.load(open(rp))
        tf = r.get("test_final", {})
        bf = r.get("threshold_sweep_summary", {}).get("best_f1", {})
        ps = tf.get("per_suite", {})
        obj = ps.get("libero_object", {})
        print("{:<14} | {:.1f}%  | {:.1f}%  | {:.3f}  | {:.1f}%   | {:.1f}%   | {:.1f}%".format(
            name, tf.get("recall",0)*100, tf.get("fp_rate",0)*100, tf.get("f1",0),
            bf.get("l10_recall",0)*100, obj.get("recall",0)*100, bf.get("fp_rate",0)*100))
    else:
        print("{:<14} | NOT DONE".format(name))
PYEOF

echo "=== Auto Ablation Complete: $(date) ==="
