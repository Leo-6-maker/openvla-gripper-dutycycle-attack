"""N5 Dataset: loads CS200 features + Label V2 labels, produces 51D input tensors.

51D = f25d(25) | p9d(9) | g9d(9) | proxies(8)

All features are pre-computed in CS200 step_records.jsonl — no OpenVLA inference needed.
"""
import json, os, sys, hashlib, time
import numpy as np
from typing import Dict, List, Tuple, Optional

# ── Feature Schema (frozen from n4_detector_adapter_v4.py) ──

FEATURE_NAMES_25D = [
    'gripper_command','gripper_qpos','gripper_opening_proxy',
    'eef_x','eef_y','eef_z','eef_vx','eef_vy','eef_vz',
    'action_dx','action_dy','action_dz','action_gripper',
    'recent_close_streak','recent_open_streak','recent_gripper_flip_count',
    'close_onset','time_since_close','eef_speed',
    'eef_z_delta_since_close','qpos_delta_1','qpos_delta_3',
    'opening_proxy_delta_3','opening_proxy_variance_5','eef_speed_variance_5',
]

POLICY_INTENT_ORDER = [
    'clean_open_probability_mass','clean_close_probability_mass',
    'clean_open_minus_close_log_mass','clean_action_token_entropy_normalized',
    'clean_top1_probability','clean_top1_is_open','clean_top1_is_close',
    'clean_best_open_rank_normalized','clean_best_close_rank_normalized',
]

TRAIN_G9D_ORDER = [
    'clean_close_probability_mass',
    'clean_open_probability_mass',
    'clean_top1_is_close',
    'clean_top1_is_open',
    'clean_top1_probability',
    'clean_best_close_rank_normalized',
    'clean_best_open_rank_normalized',
    'clean_action_token_entropy_normalized',
    'clean_open_minus_close_log_mass',
]

# p9d index → g9d index mapping
_p9d_by_name = {name: i for i, name in enumerate(POLICY_INTENT_ORDER)}
G9D_FROM_P9D = [_p9d_by_name[name] for name in TRAIN_G9D_ORDER]

N5_HEAD_NAMES = ['physical_criticality', 'k10_feasible', 'safe_release', 'instability', 'close_intent']

FROZEN_FEATURE_SCHEMA = {
    'schema': 'N5_51D_FEATURE_SCHEMA_V1',
    'input_dim': 51,
    'blocks': {
        'f25d': {'dim': 25, 'source': 'SC5StreamingFeatureAdapterV2', 'field_order': FEATURE_NAMES_25D},
        'p9d': {'dim': 9, 'source': '_summarize_logits', 'field_order': POLICY_INTENT_ORDER},
        'g9d': {'dim': 9, 'source': '_summarize_logits (reordered)', 'field_order': TRAIN_G9D_ORDER},
        'proxies': {'dim': 8, 'source': '_compute_proxies(f25d, p9d, g9d)',
                     'fields': ['cmd_qpos_diff','is_close','qpos_diff','close_duration',
                               'eef_speed','qpos_var_5','g9d_close_mass','g9d_entropy']},
    },
    'normalization': {
        'f25d': 'z-score (train stats)',
        'p9d': 'z-score (train stats)',
        'g9d': 'z-score (train stats)',
        'proxies': 'raw (no normalization)',
    },
}


def compute_feature_schema_sha():
    return hashlib.sha256(
        json.dumps(FROZEN_FEATURE_SCHEMA, sort_keys=True).encode()
    ).hexdigest()


def compute_proxies(f25d: np.ndarray, g9d: np.ndarray) -> np.ndarray:
    """Compute 8D causal response proxies (matches _compute_proxies in V4 adapter).

    Args:
        f25d: (T, 25) SC5 streaming features
        g9d: (T, 9) gripper token features (TRAIN_G9D_ORDER)

    Returns:
        proxies: (T, 8) proxy features
    """
    T = f25d.shape[0]
    proxies = np.zeros((T, 8), dtype=np.float32)

    cmd = f25d[:, 0]   # gripper_command
    qpos = f25d[:, 1]  # gripper_qpos

    proxies[:, 0] = cmd - qpos
    proxies[:, 1] = (cmd < 0).astype(np.float32)
    proxies[1:, 2] = np.diff(qpos)
    proxies[0, 2] = 0.0

    dur = 0
    for t in range(T):
        if cmd[t] < 0:
            dur += 1
        else:
            dur = 0
        proxies[t, 3] = float(dur)

    proxies[:, 4] = np.sqrt(f25d[:, 6]**2 + f25d[:, 7]**2 + f25d[:, 8]**2)

    for t in range(T):
        w_s = max(0, t - 4)
        w_e = min(T, t + 1)
        window = qpos[w_s:w_e]
        proxies[t, 5] = float(np.var(window)) if len(window) > 1 else 0.0

    proxies[:, 6] = g9d[:, 0]  # clean_close_probability_mass
    proxies[:, 7] = g9d[:, 7]  # clean_action_token_entropy_normalized

    return np.nan_to_num(proxies, 0).astype(np.float32)


def load_episode_features(step_records_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 51D features from CS200 step_records.jsonl.

    Returns:
        features_51d: (T, 51) full normalized feature matrix
        raw_gripper: (T,) action_raw gripper values
        candidate_close: (T,) boolean
    """
    steps = []
    with open(step_records_path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    T = len(steps)
    f25d = np.array([s['features_25d'] for s in steps], dtype=np.float32)
    p9d = np.array([s['clean_policy_intent_9d'] for s in steps], dtype=np.float32)
    g9d = p9d[:, G9D_FROM_P9D].astype(np.float32)
    proxies = compute_proxies(f25d, g9d)
    raw_gripper = np.array([s['action_raw'][6] if 'action_raw' in s
                            else s.get('clean_action_raw_7d', [0]*7)[6]
                            for s in steps], dtype=np.float32)
    candidate_close = raw_gripper <= 0.5

    features_51d = np.concatenate([f25d, p9d, g9d, proxies], axis=-1).astype(np.float32)

    return features_51d, raw_gripper, candidate_close


def load_episode_labels(label_path: str) -> Dict[str, np.ndarray]:
    """Load Label Contract V2 labels from label_contract_v2.jsonl.

    Returns dict with:
        labels[name]: (T,) float32 array (-1=negative, 0=unknown, 1=positive)
        valid_masks[name]: (T,) bool array
        reasons[name]: list of str
    """
    steps = []
    with open(label_path) as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    T = len(steps)
    labels = {}
    valid_masks = {}
    reasons = {}

    head_map = {
        'physical_criticality': 'physical_criticality',
        'k10_feasible': 'k10_feasible',
        'safe_release': 'safe_release',
        'instability': 'instability',
        'close_intent': 'gripper_closing_state',
    }

    for model_name, label_key in head_map.items():
        values = np.zeros(T, dtype=np.float32)
        masks = np.zeros(T, dtype=bool)
        head_reasons = []
        for i, s in enumerate(steps):
            head = s.get(label_key, {})
            vm = head.get('valid_mask', False)
            v = head.get('value')
            masks[i] = bool(vm)
            if not vm or v is None:
                values[i] = 0.0
            elif v is True or v == 1:
                values[i] = 1.0
            elif v is False or v == 0:
                values[i] = -1.0
            else:
                values[i] = 0.0
                masks[i] = False
            head_reasons.append(head.get('reason', 'UNKNOWN'))
        labels[model_name] = values
        valid_masks[model_name] = masks
        reasons[model_name] = head_reasons

    # attack_opportunity
    ao = np.zeros(T, dtype=np.float32)
    ao_mask = np.zeros(T, dtype=bool)
    for i, s in enumerate(steps):
        head = s.get('attack_opportunity', {})
        ao_mask[i] = bool(head.get('valid_mask', False))
        ao[i] = 1.0 if head.get('value', False) else (-1.0 if ao_mask[i] and not head.get('value', True) else 0.0)
    labels['attack_opportunity'] = ao
    valid_masks['attack_opportunity'] = ao_mask

    return labels, valid_masks, reasons


class N5Dataset:
    """Loads all training episodes with 51D features and Label V2 labels."""

    def __init__(self, identity_manifest_path: str, cs200_root: str, label_root: str,
                 split: str = 'checkpoint_training'):
        self.cs200_root = cs200_root
        self.label_root = label_root

        with open(identity_manifest_path) as f:
            splits_data = json.load(f)

        self.identities = []
        for fold_key, fold_data in splits_data['splits'].items():
            if split == 'all':
                self.identities.extend(fold_data.get('checkpoint_training', []))
            elif split == 'checkpoint_training':
                self.identities.extend(fold_data.get('checkpoint_training', []))
            elif split in fold_data:
                self.identities.extend(fold_data[split])
            else:
                raise ValueError(f'Unknown split: {split}')

        self.identities = sorted(set(self.identities))
        self.suites = [ident.split('/')[0] for ident in self.identities]

    def __len__(self):
        return len(self.identities)

    def get_episode(self, idx: int) -> dict:
        """Load a single episode's features and labels."""
        ident = self.identities[idx]
        suite, task, state = ident.split('/')

        step_path = os.path.join(self.cs200_root, suite, task, state, 'step_records.jsonl')
        label_path = os.path.join(self.label_root, suite, task, state, 'label_contract_v2.jsonl')

        features_51d, raw_gripper, candidate_close = load_episode_features(step_path)
        labels, valid_masks, reasons = load_episode_labels(label_path)

        T_features = features_51d.shape[0]
        T_labels = len(next(iter(labels.values())))
        if T_features != T_labels:
            raise RuntimeError(
                f'Step mismatch for {ident}: features={T_features}, labels={T_labels}'
            )

        return {
            'identity': ident,
            'suite': suite,
            'features': features_51d,       # (T, 51)
            'labels': labels,               # dict of (T,)
            'valid_masks': valid_masks,     # dict of (T,) bool
            'reasons': reasons,             # dict of list[str]
            'raw_gripper': raw_gripper,     # (T,)
            'candidate_close': candidate_close,  # (T,) bool
            'T': T_features,
        }


class N5Normalizer:
    """Z-score normalizer for 51D features. Fits on train data only."""

    def __init__(self):
        self.n25d_m = None; self.n25d_s = None
        self.np9d_m = None; self.np9d_s = None
        self.ng9d_m = None; self.ng9d_s = None
        self.fitted = False

    def fit(self, features_list: List[np.ndarray]):
        """Fit normalizer on list of (T, 51) arrays from train split."""
        all_f25d = np.concatenate([f[:, :25] for f in features_list], axis=0)
        all_p9d = np.concatenate([f[:, 25:34] for f in features_list], axis=0)
        all_g9d = np.concatenate([f[:, 34:43] for f in features_list], axis=0)

        self.n25d_m = all_f25d.mean(axis=0).astype(np.float32)
        self.n25d_s = all_f25d.std(axis=0).astype(np.float32) + 1e-8
        self.np9d_m = all_p9d.mean(axis=0).astype(np.float32)
        self.np9d_s = all_p9d.std(axis=0).astype(np.float32) + 1e-8
        self.ng9d_m = all_g9d.mean(axis=0).astype(np.float32)
        self.ng9d_s = all_g9d.std(axis=0).astype(np.float32) + 1e-8
        self.fitted = True

    def normalize(self, features_51d: np.ndarray) -> np.ndarray:
        """Normalize a (T, 51) feature matrix. Proxies block not normalized."""
        if not self.fitted:
            raise RuntimeError('Normalizer not fitted')
        out = features_51d.copy()
        out[:, :25] = (out[:, :25] - self.n25d_m) / self.n25d_s
        out[:, 25:34] = (out[:, 25:34] - self.np9d_m) / self.np9d_s
        out[:, 34:43] = (out[:, 34:43] - self.ng9d_m) / self.ng9d_s
        return out

    def state_dict(self) -> dict:
        return {
            'n25d_m': self.n25d_m, 'n25d_s': self.n25d_s,
            'np9d_m': self.np9d_m, 'np9d_s': self.np9d_s,
            'ng9d_m': self.ng9d_m, 'ng9d_s': self.ng9d_s,
        }

    def save(self, path: str):
        import torch
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        import torch
        state = torch.load(path, map_location='cpu', weights_only=True)
        norm = cls()
        norm.n25d_m = state['n25d_m']; norm.n25d_s = state['n25d_s']
        norm.np9d_m = state['np9d_m']; norm.np9d_s = state['np9d_s']
        norm.ng9d_m = state['ng9d_m']; norm.ng9d_s = state['ng9d_s']
        norm.fitted = True
        return norm


def compute_pos_weights(episodes: List[dict]) -> dict:
    """Compute frozen pos_weights from training labels."""
    from collections import defaultdict
    pos = defaultdict(int)
    neg = defaultdict(int)
    for ep in episodes:
        for name in N5_HEAD_NAMES:
            vals = ep['labels'][name]
            mask = ep['valid_masks'][name]
            for i in range(len(vals)):
                if mask[i]:
                    if vals[i] > 0.5:
                        pos[name] += 1
                    elif vals[i] < -0.5:
                        neg[name] += 1

    weights = {}
    counts = {}
    for name in N5_HEAD_NAMES:
        n_pos = pos[name]
        n_neg = neg[name]
        counts[name] = {'pos': n_pos, 'neg': n_neg}
        if n_pos > 0 and n_neg > 0:
            weights[name] = float(np.clip(n_neg / n_pos, 1.0, 20.0))
        else:
            weights[name] = None
    return weights, dict(counts)


# ── Self-Tests ──

def test_feature_dimensions():
    """Verify 51D feature construction."""
    f25d = np.random.randn(10, 25).astype(np.float32)
    p9d = np.random.randn(10, 9).astype(np.float32)
    g9d = p9d[:, G9D_FROM_P9D]
    proxies = compute_proxies(f25d, g9d)
    features = np.concatenate([f25d, p9d, g9d, proxies], axis=-1)
    assert features.shape == (10, 51), f'Expected (10,51), got {features.shape}'
    assert np.isfinite(features).all(), 'Features contain NaN/Inf'
    print('PASS: test_feature_dimensions')

def test_g9d_reorder():
    """Verify g9d = p9d reordered to TRAIN_G9D_ORDER."""
    sentinel = np.array([[101,102,103,104,105,106,107,108,109]], dtype=np.float32)
    g9d = sentinel[:, G9D_FROM_P9D]
    expected = np.array([[102,101,107,106,105,109,108,104,103]], dtype=np.float32)
    assert np.allclose(g9d, expected), f'g9d reorder mismatch: {g9d.tolist()} vs {expected.tolist()}'
    print('PASS: test_g9d_reorder')

def test_proxies_shape():
    """Verify proxies shape and finiteness."""
    f25d = np.random.randn(10, 25).astype(np.float32)
    g9d = np.random.randn(10, 9).astype(np.float32)
    proxies = compute_proxies(f25d, g9d)
    assert proxies.shape == (10, 8), f'Expected (10,8), got {proxies.shape}'
    assert np.isfinite(proxies).all(), 'Proxies contain NaN/Inf'
    print('PASS: test_proxies_shape')

def test_normalizer():
    """Verify normalizer fit + normalize roundtrip."""
    features = [np.random.randn(50, 51).astype(np.float32) for _ in range(3)]
    norm = N5Normalizer()
    norm.fit(features)
    assert norm.fitted
    normalized = norm.normalize(features[0])
    assert normalized.shape == (50, 51)
    # Blocks 0-42 should be normalized (mean ~0, std ~1)
    assert abs(normalized[:, :25].mean()) < 0.5
    assert abs(normalized[:, :25].std() - 1.0) < 0.5
    # Block 43-50 (proxies) should be unchanged
    assert np.allclose(normalized[:, 43:], features[0][:, 43:])
    print('PASS: test_normalizer')

def test_label_loading():
    """Verify label value convention."""
    # Simulate label entries
    pass  # Requires actual label files — tested via G6 freeze script
    print('SKIP: test_label_loading (requires label files)')

def run_all_tests():
    for test in [test_feature_dimensions, test_g9d_reorder, test_proxies_shape, test_normalizer]:
        try:
            test()
        except Exception as e:
            print(f'FAIL: {test.__name__}: {e}')
            return False
    print('\nAll N5 dataset tests PASSED.')
    return True


if __name__ == '__main__':
    ok = run_all_tests()
    print(f'\nFeature Schema SHA: {compute_feature_schema_sha()}')
    sys.exit(0 if ok else 1)
