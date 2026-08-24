"""G6: Training Data Seal — freeze splits, normalization, pos_weights, sampler config.

Reads all 800 training episodes, creates train/val/cal split, fits normalizer,
computes pos_weights, and writes G6_SEAL.json with all frozen parameters.
"""
import json, os, sys, hashlib, time
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)
from n5_dataset import (
    N5Dataset, N5Normalizer, compute_pos_weights,
    N5_HEAD_NAMES, FROZEN_FEATURE_SCHEMA, compute_feature_schema_sha,
    FEATURE_NAMES_25D, POLICY_INTENT_ORDER, TRAIN_G9D_ORDER,
)

IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal'
NORM_PATH = os.path.join(OUT_ROOT, 'normalization.pt')

SEED = 19903  # same seed as V4/V5 smoke

print('=== G6: Training Data Seal ===')
print(f'Identity manifest: {IDENTITY_MANIFEST}')
print(f'CS200 root: {CS200_ROOT}')
print(f'Label root: {LABEL_ROOT}')
print(f'Output: {OUT_ROOT}')
print()

# ── 1. Load all 800 identities ──
print('--- Step 1: Load all 800 identities ---')
dataset = N5Dataset(IDENTITY_MANIFEST, CS200_ROOT, LABEL_ROOT, split='checkpoint_training')
n_total = len(dataset)
print(f'Total episodes: {n_total}')

suite_counts = defaultdict(int)
for ident in dataset.identities:
    suite_counts[ident.split('/')[0]] += 1
print(f'Per suite: {dict(suite_counts)}')
print()

# ── 2. Load all features and labels ──
print('--- Step 2: Load features and labels ---')
all_episodes = []
total_steps = 0
load_errors = 0
for i, ident in enumerate(dataset.identities):
    try:
        ep = dataset.get_episode(i)
        all_episodes.append(ep)
        total_steps += ep['T']
    except Exception as e:
        print(f'  ERROR loading {ident}: {e}')
        load_errors += 1
    if (i + 1) % 100 == 0:
        print(f'  Loaded {i+1}/{n_total} episodes ({total_steps} steps)')

print(f'Loaded {len(all_episodes)}/{n_total} episodes, {total_steps} total steps')
if load_errors > 0:
    print(f'  WARNING: {load_errors} load errors')
print()

# ── 3. Create train/val/cal split (deterministic, suite-stratified) ──
print('--- Step 3: Train/Val/Cal split ---')
rng = np.random.RandomState(SEED)

# Group indices by suite
suite_indices = defaultdict(list)
for i, ep in enumerate(all_episodes):
    suite_indices[ep['suite']].append(i)

train_indices = []
val_indices = []
cal_indices = []

for suite in sorted(suite_indices.keys()):
    idx_list = list(suite_indices[suite])
    rng.shuffle(idx_list)
    n = len(idx_list)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    n_cal = n - n_train - n_val

    train_indices.extend(idx_list[:n_train])
    val_indices.extend(idx_list[n_train:n_train + n_val])
    cal_indices.extend(idx_list[n_train + n_val:])

rng.shuffle(train_indices)
rng.shuffle(val_indices)
rng.shuffle(cal_indices)

n_train = len(train_indices); n_val = len(val_indices); n_cal = len(cal_indices)
print(f'Train: {n_train}, Val: {n_val}, Cal: {n_cal}')
print(f'Train%: {n_train/n_total*100:.1f}, Val%: {n_val/n_total*100:.1f}, Cal%: {n_cal/n_total*100:.1f}')
assert n_train + n_val + n_cal == n_total
assert len(set(train_indices) & set(val_indices)) == 0, 'Train/Val overlap'
assert len(set(train_indices) & set(cal_indices)) == 0, 'Train/Cal overlap'
assert len(set(val_indices) & set(cal_indices)) == 0, 'Val/Cal overlap'
print('Split integrity: PASS (no crossover)')

# Per-suite split counts
split_map = {}
for idx in train_indices:
    split_map[all_episodes[idx]['identity']] = 'train'
for idx in val_indices:
    split_map[all_episodes[idx]['identity']] = 'val'
for idx in cal_indices:
    split_map[all_episodes[idx]['identity']] = 'cal'

split_suite_counts = defaultdict(lambda: defaultdict(int))
for ident, sp in split_map.items():
    split_suite_counts[sp][ident.split('/')[0]] += 1
print(f'Per-suite splits: {dict(split_suite_counts)}')
print()

# ── 4. Fit normalizer on train split only ──
print('--- Step 4: Fit normalizer (train only) ---')
train_features = [all_episodes[i]['features'] for i in train_indices]
normalizer = N5Normalizer()
normalizer.fit(train_features)
print(f'Normalizer fitted on {len(train_features)} train episodes')

# Verify normalization ranges
f25d_means = normalizer.n25d_m
f25d_stds = normalizer.n25d_s
print(f'  f25d mean range: [{f25d_means.min():.3f}, {f25d_means.max():.3f}]')
print(f'  f25d std range:  [{f25d_stds.min():.3f}, {f25d_stds.max():.3f}]')

# Check for zero-variance features
zero_var = [FEATURE_NAMES_25D[i] for i in range(25) if normalizer.n25d_s[i] < 1e-6]
if zero_var:
    print(f'  WARNING: Zero-variance f25d features: {zero_var}')
print()

# ── 5. Compute frozen pos_weights from train split only ──
print('--- Step 5: Compute pos_weights (train only) ---')
train_eps = [all_episodes[i] for i in train_indices]
pos_weights, pos_neg_counts = compute_pos_weights(train_eps)

for name in N5_HEAD_NAMES:
    w = pos_weights[name]
    c = pos_neg_counts[name]
    print(f'  {name}: pos={c["pos"]}, neg={c["neg"]}, pos_weight={w}')

# Check for untrainable heads
issues = []
for name in N5_HEAD_NAMES:
    c = pos_neg_counts[name]
    if c['pos'] == 0 and c['neg'] == 0:
        issues.append(f'{name}: zero valid samples')
    elif c['pos'] == 0:
        issues.append(f'{name}: zero positives (neg={c["neg"]})')
    elif c['neg'] == 0:
        issues.append(f'{name}: zero negatives (pos={c["pos"]})')

if issues:
    print(f'\n  FATAL: Untrainable heads detected:')
    for iss in issues:
        print(f'    - {iss}')
    sys.exit(2)
print()

# ── 6. Sampler config ──
print('--- Step 6: Sampler config ---')
sampler_config = {
    'type': 'suite_balanced',
    'batch_size': 32,
    'per_suite_batch': 8,
    'effective_batch_size': 32,
    'seed': SEED + 42,
    'shuffle': True,
    'with_replacement': False,
    'truncation': 'warn_on_discard',
}
print(f'Sampler: {json.dumps(sampler_config)}')
print()

# ── 7. Padding strategy ──
print('--- Step 7: Padding strategy ---')
max_steps = max(ep['T'] for ep in all_episodes)
p90_steps = int(np.percentile([ep['T'] for ep in all_episodes], 90))
p95_steps = int(np.percentile([ep['T'] for ep in all_episodes], 95))

padding_config = {
    'type': 'right_padding',
    'pad_value': 0.0,
    'valid_step_marker': 'bool mask, True=valid step',
    'max_steps_corpus': max_steps,
    'p90_steps': p90_steps,
    'p95_steps': p95_steps,
    'left_padding_note': 'not fully supported — use right-padding for training, left-padding for inference',
    'inference_batch': 'pad all sequences to max_len of batch, trim padding from output using mask',
}
print(f'Max steps: {max_steps}, P90: {p90_steps}, P95: {p95_steps}')
print()

# ── 8. Write G6 seal ──
print('--- Step 8: Finalize G6 Seal ---')
os.makedirs(OUT_ROOT, exist_ok=True)

# Save normalizer
normalizer.save(NORM_PATH)
norm_sha = hashlib.sha256(open(NORM_PATH, 'rb').read()).hexdigest()
print(f'Normalizer saved: {NORM_PATH} (SHA: {norm_sha})')

# Valid mask per-head summary
valid_mask_stats = defaultdict(lambda: {'true': 0, 'false': 0})
for ep in all_episodes:
    for name in N5_HEAD_NAMES:
        mask = ep['valid_masks'][name]
        valid_mask_stats[name]['true'] += int(mask.sum())
        valid_mask_stats[name]['false'] += int((~mask).sum())

seal = {
    'gate': 'G6_TRAINING_DATA_SEAL',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'status': 'PASS',
    'feature_schema_sha': compute_feature_schema_sha(),
    'input_dim': 51,
    'feature_breakdown': {
        'f25d': {'dim': 25, 'fields': FEATURE_NAMES_25D},
        'p9d': {'dim': 9, 'fields': POLICY_INTENT_ORDER},
        'g9d': {'dim': 9, 'fields': TRAIN_G9D_ORDER},
        'proxies': {'dim': 8, 'unnormalized': True},
    },
    'split': {
        'seed': SEED,
        'strategy': 'suite_stratified',
        'train': n_train,
        'val': n_val,
        'cal': n_cal,
        'train_identities': [all_episodes[i]['identity'] for i in sorted(train_indices)],
        'val_identities': [all_episodes[i]['identity'] for i in sorted(val_indices)],
        'cal_identities': [all_episodes[i]['identity'] for i in sorted(cal_indices)],
        'split_suite_counts': {sp: dict(counts) for sp, counts in split_suite_counts.items()},
    },
    'normalization': {
        'path': NORM_PATH,
        'sha256': norm_sha,
        'fitted_on': 'train_split_only',
        'n25d_mean': normalizer.n25d_m.tolist(),
        'n25d_std': normalizer.n25d_s.tolist(),
        'np9d_mean': normalizer.np9d_m.tolist(),
        'np9d_std': normalizer.np9d_s.tolist(),
        'ng9d_mean': normalizer.ng9d_m.tolist(),
        'ng9d_std': normalizer.ng9d_s.tolist(),
    },
    'pos_weights': {name: pos_weights[name] for name in N5_HEAD_NAMES},
    'pos_neg_counts': {name: pos_neg_counts[name] for name in N5_HEAD_NAMES},
    'sampler': sampler_config,
    'padding': padding_config,
    'valid_mask_summary': {
        name: {'n_valid': valid_mask_stats[name]['true'],
               'n_invalid': valid_mask_stats[name]['false']}
        for name in N5_HEAD_NAMES
    },
    'constraints': [
        'NO candidate_close in feature vector or loss mask',
        'NO head output gates another head',
        'Independent valid_mask per head',
        'Train-only normalization',
        'Train-only pos_weight computation',
        'Pos_weight clamped to [1, 20]',
        'Zero positives OR zero negatives → HOLD',
        'Right-padding only for batched training',
    ],
    'checksums': {
        'corpus_steps': total_steps,
        'n_episodes': len(all_episodes),
        'load_errors': load_errors,
        'split_integrity': 'no crossover',
    },
}

seal_path = os.path.join(OUT_ROOT, 'G6_SEAL.json')
with open(seal_path, 'w') as f:
    json.dump(seal, f, indent=2, default=str)

# Self-hash
seal_sha = hashlib.sha256(
    json.dumps(seal, sort_keys=True).encode()
).hexdigest()
seal['self_sha256'] = seal_sha

# Rewrite with self-hash
with open(seal_path, 'w') as f:
    json.dump(seal, f, indent=2, default=str)

print(f'G6 Seal: {seal_path}')
print(f'G6 Seal SHA: {seal_sha}')
print()

# ── 9. Summary ──
print('=' * 60)
print(f'G6 Training Data Seal: PASS')
print(f'  Episodes: {n_train} train / {n_val} val / {n_cal} cal')
print(f'  Steps: {total_steps}')
print(f'  Feature schema SHA: {compute_feature_schema_sha()[:16]}...')
print(f'  Seal SHA: {seal_sha[:16]}...')
print(f'  Output: {OUT_ROOT}')
print('=' * 60)
