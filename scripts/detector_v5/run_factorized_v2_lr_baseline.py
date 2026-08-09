#!/usr/bin/env python3
"""V2 LR baseline: logistic regression on same inner-CV splits.

Key contracts:
- Train: allowed negative downsampling + class_weight='balanced'
- Val: NO downsampling, NO shuffle, full metadata preserved
- Event aggregation via (canonical_parent_key, event_id) — never array-order guess
- All event fields correct: mechanism_route, event_ordinal, is_later_event, event_duration
"""
import argparse, csv, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'
SPLITS_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_LR_BASELINE_V1_20260721'

CAUSAL_WINDOW = 20
DURATION_BUCKETS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 99999)]


def _atomic_text(p, v):
    t = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.tmp')
    with t.open('x') as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {'SHA256SUMS', 'SHA256SUMS.sha256'}
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = ''.join(f'{sha256_file(p)}  {p.relative_to(root).as_posix()}\n' for p in fs)
    _atomic_text(root / 'SHA256SUMS', c)
    _atomic_text(root / 'SHA256SUMS.sha256', f'{sha256_file(root / "SHA256SUMS")}  SHA256SUMS\n')


def extract_features_with_metadata(episodes, causal_window=20):
    """Extract (X, metadata) for ALL release_known_mask steps. No downsampling."""
    X_list = []
    meta_list = []

    for ep in episodes:
        T = len(ep.features_25d)
        eids = ep.event_id
        unique_events = sorted([int(eids[t].item()) for t in range(T) if eids[t].item() >= 0])
        eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}
        event_dur = defaultdict(int)
        for t in range(T):
            eid = int(eids[t].item())
            if eid >= 0:
                event_dur[eid] += 1

        for t in range(T):
            if not ep.release_known_mask[t].item():
                continue
            eid = int(eids[t].item())

            start = max(0, t - causal_window + 1)
            window = ep.features_25d[start:t+1].numpy().astype(np.float32)
            if window.shape[0] < causal_window:
                pad = np.zeros((causal_window - window.shape[0], 25), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)

            X_list.append(window)
            meta_list.append({
                'canonical_parent_key': ep.canonical_parent_key,
                'step_index': t,
                'event_id': eid,
                'mechanism_route': ep.mechanism_route,
                'event_ordinal': eid_to_ordinal.get(eid, -1),
                'is_later_event': eid_to_ordinal.get(eid, -1) >= 1,
                'event_duration': event_dur.get(eid, 0),
                'release_target': bool(ep.release_target[t].item()),
                'release_known_mask': True,
                'grasp_target': bool(ep.grasp_target[t].item()),
                'grasp_known_mask': bool(ep.grasp_known_mask[t].item()),
                'manipulation_target': bool(ep.manipulation_target[t].item()),
                'manipulation_known_mask': bool(ep.manipulation_known_mask[t].item()),
            })

    return np.array(X_list, dtype=np.float32), meta_list


def prepare_training_data(X_train_full, meta_train, seed=42):
    """Downsample negatives to 2:1 ratio for training. Keep metadata aligned."""
    pos_idx = [i for i, m in enumerate(meta_train) if m['release_target']]
    neg_idx = [i for i, m in enumerate(meta_train) if not m['release_target']]

    if len(neg_idx) > 2 * len(pos_idx):
        rng = np.random.RandomState(seed)
        neg_idx = rng.choice(neg_idx, 2 * len(pos_idx), replace=False).tolist()

    sel_idx = sorted(pos_idx + neg_idx)
    X_bal = X_train_full[sel_idx]
    y_bal = np.array([1.0 if meta_train[i]['release_target'] else 0.0 for i in sel_idx], dtype=np.float32)
    return X_bal, y_bal


def aggregate_to_events(step_scores, step_metadata):
    """Group step scores by (identity, event_id) and compute max."""
    groups = defaultdict(list)
    for score, meta in zip(step_scores, step_metadata):
        if meta['event_id'] < 0:
            continue
        groups[(meta['canonical_parent_key'], meta['event_id'])].append((score, meta))

    events = []
    for (ident, eid), items in groups.items():
        scores = [s for s, _ in items]
        meta0 = items[0][1]
        events.append({
            'canonical_parent_key': ident,
            'event_id': eid,
            'release_event_score': float(max(scores)),
            'release_target': any(m['release_target'] for _, m in items),
            'mechanism_route': meta0['mechanism_route'],
            'event_ordinal': meta0['event_ordinal'],
            'is_later_event': meta0['is_later_event'],
            'event_duration': meta0['event_duration'],
        })

    return events


def compute_event_metrics(events):
    """Compute all metrics from event predictions."""
    m = {}
    for head in ['release']:
        labels = np.array([e[f'{head}_target'] for e in events])
        scores = np.array([e[f'{head}_event_score'] for e in events])
        pos = labels == 1

        if len(np.unique(labels)) >= 2:
            m[f'{head}_auroc'] = float(roc_auc_score(labels, scores))
        if pos.sum() > 0:
            m[f'{head}_auprc'] = float(average_precision_score(labels, scores))
        m[f'{head}_n_pos'] = int(pos.sum())
        m[f'{head}_n_total'] = len(labels)
        m[f'{head}_prevalence'] = pos.sum() / max(1, len(labels))

        # Per-route
        for route in SUPPORTED_ROUTES:
            r_idx = [i for i, e in enumerate(events) if e['mechanism_route'] == route]
            if not r_idx:
                continue
            r_labels = labels[r_idx]; r_scores = scores[r_idx]
            if len(np.unique(r_labels)) >= 2:
                m[f'{route}_{head}_auroc'] = float(roc_auc_score(r_labels, r_scores))
            if r_labels.sum() > 0:
                m[f'{route}_{head}_auprc'] = float(average_precision_score(r_labels, r_scores))

        # Duration buckets
        for lo, hi in DURATION_BUCKETS:
            b_idx = [i for i, e in enumerate(events) if lo <= e.get('event_duration', 0) < hi]
            if not b_idx:
                continue
            b_labels = labels[b_idx]; b_scores = scores[b_idx]
            if b_labels.sum() > 0:
                m[f'{head}_dur_{lo}_{hi}_auprc'] = float(average_precision_score(b_labels, b_scores))
                m[f'{head}_dur_{lo}_{hi}_n'] = len(b_idx)

        # Short event
        s_idx = [i for i, e in enumerate(events) if e.get('event_duration', 999) < 30]
        if s_idx:
            s_labels = labels[s_idx]; s_scores = scores[s_idx]
            if s_labels.sum() > 0:
                m[f'{head}_short_auprc'] = float(average_precision_score(s_labels, s_scores))

        # First/later
        for ek in ['first', 'later']:
            ek_idx = [i for i, e in enumerate(events) if (ek == 'later') == e.get('is_later_event', False)]
            if not ek_idx:
                continue
            ek_labels = labels[ek_idx]; ek_scores = scores[ek_idx]
            if ek_labels.sum() > 0:
                m[f'{head}_{ek}_auprc'] = float(average_precision_score(ek_labels, ek_scores))
                m[f'{head}_{ek}_n'] = len(ek_idx)

    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--splits-root', type=Path, default=SPLITS_ROOT)
    ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--max-iter', type=int, default=2000)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    verify_factorized_source_roots(S1, TEACHER)
    verify_sealed_directory(args.splits_root)
    split_bundle = json.loads((args.splits_root / 'inner_cv_splits.json').read_text())

    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}

    all_events = []
    all_split_metrics = {}
    all_event_lines = []

    for outer_fold in [0, 1, 2, 3]:
        for inner_fold in [0, 1, 2]:
            inner_train_ids, inner_val_ids = resolve_inner_train_val_ids(
                split_bundle, outer_fold, inner_fold)
            train_rows = [id_to_row[i] for i in inner_train_ids if i in id_to_row]
            val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]
            if len(train_rows) < 10 or len(val_rows) < 10:
                continue

            train_eps = load_factorized_episodes(S1, TEACHER, train_rows)
            val_eps = load_factorized_episodes(S1, TEACHER, val_rows)

            # Extract: train (may downsample), val (NEVER downsample)
            X_train_full, meta_train = extract_features_with_metadata(train_eps, CAUSAL_WINDOW)
            X_val, meta_val = extract_features_with_metadata(val_eps, CAUSAL_WINDOW)

            X_train, y_train = prepare_training_data(X_train_full, meta_train)

            # Fit LR
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            mean_x = X_train_flat.mean(axis=0)
            std_x = X_train_flat.std(axis=0).clip(1e-6)
            X_train_norm = (X_train_flat - mean_x) / std_x
            X_val_norm = (X_val_flat - mean_x) / std_x

            clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                                     class_weight='balanced', random_state=42)
            clf.fit(X_train_norm, y_train)
            step_scores = clf.predict_proba(X_val_norm)[:, 1]

            # Aggregate to events via metadata keys
            assert len(step_scores) == len(meta_val), \
                f'Score/metadata length mismatch: {len(step_scores)} vs {len(meta_val)}'
            events = aggregate_to_events(step_scores, meta_val)
            all_events.extend(events)

            for e in events:
                all_event_lines.append(json.dumps(e) + '\n')

            split_m = compute_event_metrics(events)
            key = f'o{outer_fold}_i{inner_fold}'
            all_split_metrics[key] = split_m
            auroc = split_m.get('release_auroc')
            print(f'  {key}: AUROC={auroc:.4f} events={len(events)}' if auroc else f'  {key}: no AUROC, events={len(events)}')

    # Pooled metrics
    pooled = compute_event_metrics(all_events)
    print(f'\nPooled: AUROC={pooled.get("release_auroc", "N/A"):.4f} '
          f'AUPRC={pooled.get("release_auprc", "N/A"):.4f} '
          f'n={pooled.get("release_n_total", 0)} '
          f'prevalence={pooled.get("release_prevalence", 0):.4f}')

    config = {'C': args.C, 'max_iter': args.max_iter, 'causal_window': CAUSAL_WINDOW,
              'class_weight': 'balanced', 'train_neg_downsample': '2:1'}
    _atomic_text(staging / 'baseline_config.json', json.dumps(config, indent=2))
    _atomic_text(staging / 'pooled_metrics.json', json.dumps(pooled, indent=2))
    _atomic_text(staging / 'per_split_metrics.json', json.dumps(all_split_metrics, indent=2))
    _atomic_text(staging / 'event_predictions.jsonl', ''.join(all_event_lines))
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'splits_root': str(args.splits_root),
        'splits_seal': sha256_file(args.splits_root / 'SHA256SUMS'),
        'n_splits': len(all_split_metrics),
        'n_total_events': len(all_events),
    }, indent=2))
    _atomic_text(staging / 'environment.json', json.dumps({
        'sklearn_version': __import__('sklearn').__version__,
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f'\nLR baseline sealed: {out}')
    print(f'Seal: {sha256_file(out / "SHA256SUMS")}')


if __name__ == '__main__':
    main()
