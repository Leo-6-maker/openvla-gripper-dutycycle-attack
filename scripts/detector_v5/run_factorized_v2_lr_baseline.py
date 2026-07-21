#!/usr/bin/env python3
"""V2 LR baseline: logistic regression on same inner-CV splits.

Uses sklearn LogisticRegression with identity-level CV.
Reports AUROC, AUPRC, per-route, per-duration, first/later metrics.
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
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_LR_BASELINE_V1_20260721'
SPLITS_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'

CAUSAL_WINDOW = 20  # same as R1e
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


def extract_causal_windows(episodes, causal_window=20):
    """Extract (X, y, identities) for release prediction from causal 25D windows."""
    X_pos, X_neg = [], []
    id_pos, id_neg = [], []

    for ep in episodes:
        T = len(ep.features_25d)
        for t in range(T):
            if not ep.release_known_mask[t].item():
                continue
            start = max(0, t - causal_window + 1)
            window = ep.features_25d[start:t+1].numpy()
            if window.shape[0] < causal_window:
                pad = np.zeros((causal_window - window.shape[0], 25), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)

            if ep.release_target[t].item():
                X_pos.append(window)
                id_pos.append(ep.canonical_parent_key)
            else:
                X_neg.append(window)
                id_neg.append(ep.canonical_parent_key)

    # Balance: downsample negatives to 2:1
    if len(X_neg) > 2 * len(X_pos):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_neg), 2 * len(X_pos), replace=False)
        X_neg = [X_neg[i] for i in idx]
        id_neg = [id_neg[i] for i in idx]

    X = np.array(X_pos + X_neg, dtype=np.float32)
    y = np.array([1]*len(X_pos) + [0]*len(X_neg), dtype=np.float32)
    ids = id_pos + id_neg
    return X, y, ids


def compute_event_metrics(event_predictions):
    """Compute event-level metrics from per-event predictions."""
    m = {}
    for head in ['release']:
        labels = [e[f'{head}_target'] for e in event_predictions]
        scores = [e[f'{head}_event_score'] for e in event_predictions]
        if len(np.unique(labels)) >= 2:
            m[f'{head}_auroc'] = float(roc_auc_score(labels, scores))
        if sum(labels) > 0:
            m[f'{head}_auprc'] = float(average_precision_score(labels, scores))
        m[f'{head}_n_pos'] = int(sum(labels))
        m[f'{head}_n_total'] = len(labels)
        m[f'{head}_prevalence'] = sum(labels) / max(1, len(labels))

        # Per-route
        for route in SUPPORTED_ROUTES:
            r_ev = [e for e in event_predictions if e.get('mechanism_route') == route]
            r_labels = [e[f'{head}_target'] for e in r_ev]
            r_scores = [e[f'{head}_event_score'] for e in r_ev]
            if len(np.unique(r_labels)) >= 2:
                m[f'{route}_{head}_auroc'] = float(roc_auc_score(r_labels, r_scores))
            if sum(r_labels) > 0:
                m[f'{route}_{head}_auprc'] = float(average_precision_score(r_labels, r_scores))

        # Duration buckets
        for lo, hi in DURATION_BUCKETS:
            b_ev = [e for e in event_predictions if lo <= e.get('event_duration', 0) < hi]
            b_labels = [e[f'{head}_target'] for e in b_ev]
            b_scores = [e[f'{head}_event_score'] for e in b_ev]
            if sum(b_labels) > 0:
                m[f'{head}_dur_{lo}_{hi}_auprc'] = float(average_precision_score(b_labels, b_scores))

        # Short event
        short_ev = [e for e in event_predictions if e.get('event_duration', 999) < 30]
        s_labels = [e[f'{head}_target'] for e in short_ev]
        s_scores = [e[f'{head}_event_score'] for e in short_ev]
        if sum(s_labels) > 0:
            m[f'{head}_short_auprc'] = float(average_precision_score(s_labels, s_scores))

        # First/later
        for ek in ['first', 'later']:
            ek_ev = [e for e in event_predictions
                     if (ek == 'later') == e.get('is_later_event', False)]
            ek_labels = [e[f'{head}_target'] for e in ek_ev]
            ek_scores = [e[f'{head}_event_score'] for e in ek_ev]
            if sum(ek_labels) > 0:
                m[f'{head}_{ek}_auprc'] = float(average_precision_score(ek_labels, ek_scores))

    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--splits-root', type=Path, default=SPLITS_ROOT)
    ap.add_argument('--C', type=float, default=1.0, help='LogisticRegression C')
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

    all_event_preds = []
    all_split_metrics = {}
    per_fold_events = []

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

            # Extract features from training identities only
            X_train, y_train, id_train = extract_causal_windows(train_eps, CAUSAL_WINDOW)
            X_val, y_val, id_val = extract_causal_windows(val_eps, CAUSAL_WINDOW)

            if len(np.unique(y_train)) < 2:
                continue

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

            # Map step scores back to event-level
            # Group validation steps by (identity, event_id)
            event_groups = defaultdict(list)
            for ep in val_eps:
                eids = ep.event_id
                for t in range(len(ep.features_25d)):
                    eid = int(eids[t].item())
                    if eid < 0 or not ep.release_known_mask[t].item():
                        continue
                    event_groups[(ep.canonical_parent_key, eid)].append({
                        'release_target': bool(ep.release_target[t].item()),
                        'release_known_mask': bool(ep.release_known_mask[t].item()),
                    })

            # Match step scores to event groups
            score_idx = 0
            for (ident, eid), step_data in event_groups.items():
                known_scores = []
                for sd in step_data:
                    if sd['release_known_mask'] and score_idx < len(step_scores):
                        known_scores.append(step_scores[score_idx])
                        score_idx += 1
                if known_scores:
                    event_pred = {
                        'canonical_parent_key': ident,
                        'event_id': eid,
                        'release_event_score': float(max(known_scores)),
                        'release_target': any(sd['release_target'] for sd in step_data),
                        'mechanism_route': 'unknown',
                        'is_later_event': False,
                        'event_duration': len(step_data),
                    }
                    all_event_preds.append(event_pred)
                    per_fold_events.append(event_pred)

            # Per-split metrics
            split_events = []
            score_idx = 0
            for ep in val_eps:
                eids = ep.event_id
                unique_events = sorted(set(int(eids[t].item()) for t in range(len(ep.features_25d)) if eids[t].item() >= 0))
                eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}
                event_dur = defaultdict(int)
                for t in range(len(ep.features_25d)):
                    eid = int(eids[t].item())
                    if eid >= 0:
                        event_dur[eid] += 1
                for eid in unique_events:
                    em_steps = [t for t in range(len(ep.features_25d))
                               if int(eids[t].item()) == eid and ep.release_known_mask[t].item()]
                    known_scores = []
                    for _ in em_steps:
                        if score_idx < len(step_scores):
                            known_scores.append(step_scores[score_idx])
                            score_idx += 1
                    if known_scores:
                        split_events.append({
                            'canonical_parent_key': ep.canonical_parent_key,
                            'event_id': eid,
                            'release_event_score': float(max(known_scores)),
                            'release_target': any(ep.release_target[t].item() and ep.release_known_mask[t].item() for t in em_steps),
                            'mechanism_route': ep.mechanism_route,
                            'is_later_event': eid_to_ordinal.get(eid, -1) >= 1,
                            'event_duration': event_dur.get(eid, 0),
                        })

            split_m = compute_event_metrics(split_events)
            key = f'o{outer_fold}_i{inner_fold}'
            all_split_metrics[key] = split_m
            print(f'  {key}: release AUROC={split_m.get("release_auroc", "N/A"):.4f}' if split_m.get('release_auroc') else f'  {key}: no release AUROC')

    # Pooled metrics
    pooled = compute_event_metrics(all_event_preds)
    print(f'\nPooled LR: AUROC={pooled.get("release_auroc", "N/A"):.4f} AUPRC={pooled.get("release_auprc", "N/A"):.4f}')

    results = {
        'config': {'C': args.C, 'max_iter': args.max_iter, 'causal_window': CAUSAL_WINDOW,
                   'class_weight': 'balanced'},
        'pooled_metrics': pooled,
        'per_split_metrics': all_split_metrics,
        'n_splits': len(all_split_metrics),
        'n_total_events': len(all_event_preds),
    }

    _atomic_text(staging / 'baseline_config.json', json.dumps(results['config'], indent=2))
    _atomic_text(staging / 'per_split_metrics.json', json.dumps(all_split_metrics, indent=2))
    _atomic_text(staging / 'pooled_metrics.json', json.dumps(pooled, indent=2))
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'splits_root': str(args.splits_root),
        'splits_seal': sha256_file(args.splits_root / 'SHA256SUMS'),
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f'\nLR baseline sealed: {out}')


if __name__ == '__main__':
    main()
