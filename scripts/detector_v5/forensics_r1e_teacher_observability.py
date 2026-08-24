#!/usr/bin/env python3
"""Phase R1e: Teacher release causal observability probe.

Tests whether release onset is causally predictable from available features
using only t-20...t (no future leakage). Three probe levels:
  1. Single-feature AUROC
  2. Logistic regression (identity-level CV)
  3. Small MLP with temporal pooling (identity-level CV)

If even simple probes cannot separate release onset, the Teacher target
is not recoverable from runtime-observable signals.
"""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
FOLD_ROOT = OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f'
POLICY_INTENT = OPS / 'OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_01'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'
FORENSICS_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_FAILURE_FORENSICS_V1_20260721'

CAUSAL_WINDOW = 20  # steps before onset
FEATURE_25D_NAMES = [
    'gripper_qpos', 'gripper_command', 'gripper_contact',
    'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
    'eef_quat_w', 'eef_quat_x', 'eef_quat_y', 'eef_quat_z',
    'eef_vel_x', 'eef_vel_y', 'eef_vel_z',
    'eef_angvel_x', 'eef_angvel_y', 'eef_angvel_z',
    'object_eef_dist', 'object_target_dist',
    'gripper_width', 'support_frac',
    'lift_score', 'comotion_score',
    'contact_score', 'closure_score', 'opening_trend',
]


def compute_auroc_np(labels, scores):
    """AUROC via sorting."""
    if len(labels) == 0 or all(labels) or not any(labels):
        return None
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    n_pos, n_neg = len(pos), len(neg)
    neg_sorted = np.sort(neg)
    rank_sum = 0.0
    for ps in pos:
        idx = np.searchsorted(neg_sorted, ps)
        rank_sum += idx
        rank_sum += 0.5 * np.sum(neg_sorted == ps)
    return rank_sum / (n_pos * n_neg)


def extract_release_onset_windows(episodes, causal_window=20):
    """Extract (features, label) pairs for release onset steps.

    For each step where release_target=True and release_known_mask=True,
    extract the causal window of 25D features from t-window+1 to t (inclusive).
    Label=1. For negative samples (release_target=False, known=True),
    extract same window. Sub-sample negatives to balance.
    """
    X_pos, X_neg = [], []
    identities_pos, identities_neg = [], []

    for ep in episodes:
        T = len(ep.features_25d)
        for t in range(T):
            if not ep.release_known_mask[t].item():
                continue

            start = max(0, t - causal_window + 1)
            window = ep.features_25d[start:t+1].numpy()  # [W, 25]
            # Pad if window is shorter than causal_window
            if window.shape[0] < causal_window:
                pad = np.zeros((causal_window - window.shape[0], 25))
                window = np.concatenate([pad, window], axis=0)

            if ep.release_target[t].item():
                X_pos.append(window)
                identities_pos.append(ep.canonical_parent_key)
            else:
                X_neg.append(window)
                identities_neg.append(ep.canonical_parent_key)

    # Downsample negatives to ~2:1 ratio
    if len(X_neg) > 2 * len(X_pos):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_neg), 2 * len(X_pos), replace=False)
        X_neg = [X_neg[i] for i in idx]
        identities_neg = [identities_neg[i] for i in idx]

    X = np.array(X_pos + X_neg, dtype=np.float32)  # [N, W, 25]
    y = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=np.float32)
    identities = identities_pos + identities_neg

    return X, y, identities


def single_feature_auroc(X, y):
    """Compute per-feature AUROC using last-step value only (causal)."""
    results = {}
    for fi in range(X.shape[2]):
        # Use the last time step (most recent before prediction)
        scores = X[:, -1, fi]
        auroc = compute_auroc_np(y, scores)
        results[FEATURE_25D_NAMES[fi] if fi < len(FEATURE_25D_NAMES) else f'f{fi}'] = auroc
    return results


def identity_cv_logistic(X, y, identities, n_folds=4):
    """Logistic regression with identity-level cross-validation."""
    from sklearn.linear_model import LogisticRegression
    unique_ids = sorted(set(identities))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_ids)

    fold_size = len(unique_ids) // n_folds
    fold_aurocs = []

    for fi in range(n_folds):
        test_ids = set(unique_ids[fi*fold_size:(fi+1)*fold_size])
        train_idx = [i for i, ident in enumerate(identities) if ident not in test_ids]
        test_idx = [i for i, ident in enumerate(identities) if ident in test_ids]

        if len(train_idx) < 10 or len(test_idx) < 10:
            continue

        # Flatten [N, W, 25] -> [N, W*25]
        X_flat = X.reshape(X.shape[0], -1)
        X_train, y_train = X_flat[train_idx], y[train_idx]
        X_test, y_test = X_flat[test_idx], y[test_idx]

        try:
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
            clf.fit(X_train, y_train)
            scores = clf.predict_proba(X_test)[:, 1]
            auroc = compute_auroc_np(y_test, scores)
            if auroc is not None:
                fold_aurocs.append(auroc)
        except Exception:
            continue

    return mean(fold_aurocs) if fold_aurocs else None, fold_aurocs


def identity_cv_mlp(X, y, identities, n_folds=4, hidden=32):
    """Small MLP with temporal mean pooling, identity-level CV."""
    from sklearn.neural_network import MLPClassifier
    unique_ids = sorted(set(identities))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_ids)

    fold_size = len(unique_ids) // n_folds
    fold_aurocs = []

    for fi in range(n_folds):
        test_ids = set(unique_ids[fi*fold_size:(fi+1)*fold_size])
        train_idx = [i for i, ident in enumerate(identities) if ident not in test_ids]
        test_idx = [i for i, ident in enumerate(identities) if ident in test_ids]

        if len(train_idx) < 10 or len(test_idx) < 10:
            continue

        # Temporal mean + max pooling per feature
        X_train_pool = np.concatenate([X[train_idx].mean(axis=1), X[train_idx].max(axis=1)], axis=1)
        X_test_pool = np.concatenate([X[test_idx].mean(axis=1), X[test_idx].max(axis=1)], axis=1)

        try:
            clf = MLPClassifier(hidden_layer_sizes=(hidden, 16), max_iter=500,
                               early_stopping=True, random_state=42)
            clf.fit(X_train_pool, y[train_idx])
            scores = clf.predict_proba(X_test_pool)[:, 1]
            auroc = compute_auroc_np(y[test_idx], scores)
            if auroc is not None:
                fold_aurocs.append(auroc)
        except Exception:
            continue

    return mean(fold_aurocs) if fold_aurocs else None, fold_aurocs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=FORENSICS_OUT)
    ap.add_argument('--sample-folds', type=int, nargs='+', default=[0, 1])
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    print('=== R1e: Teacher Release Causal Observability Probe ===')

    # Load all train+val identities for sampled folds
    folds = load_fit_fold_bundle(FOLD_ROOT)
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}

    verify_factorized_source_roots(S1, TEACHER)
    verify_sealed_directory(POLICY_INTENT)
    from gripper_attack.v5_dataset import load_policy_intent_root
    policy_index, _ = load_policy_intent_root(POLICY_INTENT)

    all_eps = []
    for fold_id in args.sample_folds:
        fold = [f for f in folds['folds'] if f['fold_id'] == fold_id][0]
        all_ids = set(fold['train_identities']) | set(fold['validation_identities'])
        all_rows = [id_to_row[i] for i in all_ids if i in id_to_row]
        eps = load_factorized_episodes(S1, TEACHER, all_rows, policy_index=policy_index)
        all_eps.extend(eps)
        print(f'  Fold {fold_id}: {len(eps)} episodes')

    print(f'\nTotal episodes: {len(all_eps)}')

    # Filter to supported routes only
    supported_eps = [ep for ep in all_eps if ep.route_supported]
    print(f'Supported route episodes: {len(supported_eps)}')

    # Extract release onset windows
    print('\nExtracting release onset windows...')
    X, y, identities = extract_release_onset_windows(supported_eps, CAUSAL_WINDOW)
    print(f'  Positive samples: {int(y.sum())}')
    print(f'  Negative samples: {int(len(y) - y.sum())}')
    print(f'  Unique identities: {len(set(identities))}')

    # 1. Single-feature AUROC
    print('\n--- Single-feature AUROC (last causal step) ---')
    sf_auroc = single_feature_auroc(X, y)
    top_features = sorted([(v, k) for k, v in sf_auroc.items() if v is not None], reverse=True)
    for auroc, name in top_features[:10]:
        print(f'  {name}: {auroc:.4f}')
    print(f'  ...')
    for auroc, name in top_features[-5:]:
        print(f'  {name}: {auroc:.4f}')

    # 2. Logistic regression with identity CV
    print('\n--- Logistic Regression (identity-level CV) ---')
    lr_auroc, lr_folds = identity_cv_logistic(X, y, identities)
    print(f'  Mean AUROC: {lr_auroc:.4f}' if lr_auroc else '  Failed')
    if lr_folds:
        print(f'  Per-fold: {[f"{v:.4f}" for v in lr_folds]}')

    # 3. Small MLP
    print('\n--- MLP (hidden=32, identity-level CV) ---')
    mlp_auroc, mlp_folds = identity_cv_mlp(X, y, identities)
    print(f'  Mean AUROC: {mlp_auroc:.4f}' if mlp_auroc else '  Failed')
    if mlp_folds:
        print(f'  Per-fold: {[f"{v:.4f}" for v in mlp_folds]}')

    # Per-route breakdown
    print('\n--- Per-route single-feature top features ---')
    for route in SUPPORTED_ROUTES:
        route_eps = [ep for ep in supported_eps if ep.mechanism_route == route]
        if len(route_eps) < 10:
            continue
        Xr, yr, idr = extract_release_onset_windows(route_eps, CAUSAL_WINDOW)
        if len(yr) < 20 or yr.sum() < 5:
            continue
        sf_r = single_feature_auroc(Xr, yr)
        top_r = sorted([(v, k) for k, v in sf_r.items() if v is not None], reverse=True)[:5]
        print(f'  {route}:')
        for auroc, name in top_r:
            print(f'    {name}: {auroc:.4f}')

    # Per-route logistic
    print('\n--- Per-route logistic regression ---')
    for route in SUPPORTED_ROUTES:
        route_eps = [ep for ep in supported_eps if ep.mechanism_route == route]
        if len(route_eps) < 10:
            continue
        Xr, yr, idr = extract_release_onset_windows(route_eps, CAUSAL_WINDOW)
        if len(yr) < 50 or yr.sum() < 10:
            print(f'  {route}: insufficient samples')
            continue
        lr_r, _ = identity_cv_logistic(Xr, yr, idr)
        print(f'  {route}: AUROC={lr_r:.4f}' if lr_r else f'  {route}: failed')

    # ── Write results ──
    results = {
        'single_feature_auroc': {k: v for k, v in sf_auroc.items() if v is not None},
        'top_features': [(float(v), k) for v, k in top_features],
        'logistic_regression_auroc': lr_auroc,
        'logistic_per_fold': lr_folds,
        'mlp_auroc': mlp_auroc,
        'mlp_per_fold': mlp_folds,
        'n_samples': len(y),
        'n_positives': int(y.sum()),
        'n_identities': len(set(identities)),
        'causal_window': CAUSAL_WINDOW,
    }

    output_file = out_dir / 'r1e_teacher_observability.json'
    tmp = output_file.with_suffix('.tmp')
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, output_file)
    print(f'\nOutput: {output_file}')

    # Classification
    print('\n=== Observability Classification ===')
    if mlp_auroc is None:
        mlp_auroc = lr_auroc or 0

    if mlp_auroc >= 0.85:
        level = 'HIGH — Teacher target is causally recoverable; problem is GRU/loss/architecture'
    elif mlp_auroc >= 0.70:
        level = 'MODERATE — Signal exists but weak; prioritize event-level loss, onset modeling, time windows'
    else:
        level = 'LOW — Teacher release may depend on privileged geometry; consider redefining target'

    print(f'  Best probe AUROC: {mlp_auroc:.4f}')
    print(f'  Classification: {level}')


if __name__ == '__main__':
    main()
