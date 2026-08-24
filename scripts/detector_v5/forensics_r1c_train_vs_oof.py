#!/usr/bin/env python3
"""Phase R1c: Train-set vs OOF release gap diagnosis.

For sampled checkpoints, runs predictions on both train and val identities,
then compares release metrics to distinguish "can't learn" from "doesn't generalize."
"""
import argparse, csv, hashlib, json, os, sys, uuid, platform
from pathlib import Path
from collections import defaultdict
from statistics import mean, median, stdev

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.v5_factorized_student import FactorizedStudent
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
FOLD_ROOT = OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f'
POLICY_INTENT = OPS / 'OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_01'
TRAINING_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_335048c_20260721'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'
FORENSICS_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_FAILURE_FORENSICS_V1_20260721'

SAMPLED_CHECKPOINTS = [
    ('25D9D', 0, 42), ('25D9D', 0, 456),
    ('25D9D', 1, 123), ('25D9D', 2, 42),
    ('25D', 0, 42), ('25D', 1, 123),
]


def compute_auroc(labels, scores):
    if not labels or all(labels) or not any(labels):
        return None
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    n_pos, n_neg = len(pos), len(neg)
    neg_sorted = sorted(neg)
    rank_sum = 0.0
    for ps in pos:
        lo, hi = 0, n_neg
        while lo < hi:
            mid = (lo + hi) // 2
            if neg_sorted[mid] < ps:
                lo = mid + 1
            else:
                hi = mid
        rank_sum += lo + 0.5 * sum(1 for ns in neg if ns == ps)
    return rank_sum / (n_pos * n_neg)


def predict_episodes(model, episodes, mean_25d, std_25d, mean_9d, std_9d, use_9d, device):
    """Run inference and return per-event predictions."""
    model.eval()
    event_preds = []
    step_preds = []

    with torch.no_grad():
        for ep in episodes:
            T = len(ep.features_25d)
            route = ep.mechanism_route

            if not ep.route_supported:
                for t in range(T):
                    step_preds.append({
                        'canonical_parent_key': ep.canonical_parent_key,
                        'event_id': int(ep.event_id[t].item()),
                        'mechanism_route': route, 'route_supported': False,
                        'step_index': t,
                        'grasp_prob': 0.0, 'manipulation_prob': 0.0, 'release_prob': 0.0,
                        'grasp_target': bool(ep.grasp_target[t].item()),
                        'grasp_known_mask': bool(ep.grasp_known_mask[t].item()),
                        'manipulation_target': bool(ep.manipulation_target[t].item()),
                        'manipulation_known_mask': bool(ep.manipulation_known_mask[t].item()),
                        'release_target': bool(ep.release_target[t].item()),
                        'release_known_mask': bool(ep.release_known_mask[t].item()),
                        'is_later_event': False, 'event_ordinal': -1,
                    })
                continue

            x25 = ((ep.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
            m25 = ep.valid_mask.unsqueeze(0).to(device)
            x9 = m9 = None
            if use_9d and ep.policy_intent_9d.numel() > 0:
                x9 = ((ep.policy_intent_9d - mean_9d) / std_9d).unsqueeze(0).to(device)
                m9 = ep.policy_intent_valid_mask.unsqueeze(0).to(device)

            probs = model.forward_sequence(x25, x9, m25, m9, route)
            g_prob = probs['grasp'][0].cpu()
            m_prob = probs['manipulation'][0].cpu()
            r_prob = probs['release'][0].cpu()

            eids = ep.event_id
            unique_events = sorted([e.item() for e in eids.unique() if e.item() >= 0])
            eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}

            for t in range(T):
                ev = int(eids[t].item())
                step_preds.append({
                    'canonical_parent_key': ep.canonical_parent_key,
                    'event_id': ev, 'mechanism_route': route, 'route_supported': True,
                    'step_index': t,
                    'event_ordinal': eid_to_ordinal.get(ev, -1),
                    'is_later_event': eid_to_ordinal.get(ev, -1) >= 1,
                    'grasp_prob': float(g_prob[t].item()),
                    'manipulation_prob': float(m_prob[t].item()),
                    'release_prob': float(r_prob[t].item()),
                    'grasp_target': bool(ep.grasp_target[t].item()),
                    'grasp_known_mask': bool(ep.grasp_known_mask[t].item()),
                    'manipulation_target': bool(ep.manipulation_target[t].item()),
                    'manipulation_known_mask': bool(ep.manipulation_known_mask[t].item()),
                    'release_target': bool(ep.release_target[t].item()),
                    'release_known_mask': bool(ep.release_known_mask[t].item()),
                })

    # Aggregate to event-level
    event_groups = defaultdict(list)
    for s in step_preds:
        if s['event_id'] >= 0 and s['route_supported']:
            event_groups[(s['canonical_parent_key'], s['event_id'])].append(s)

    for (identity, eid), steps in event_groups.items():
        r_km = [s['release_known_mask'] for s in steps]
        g_km = [s['grasp_known_mask'] for s in steps]
        m_km = [s['manipulation_known_mask'] for s in steps]
        r_probs = [s['release_prob'] for i, s in enumerate(steps) if r_km[i]]
        g_probs = [s['grasp_prob'] for i, s in enumerate(steps) if g_km[i]]
        m_probs = [s['manipulation_prob'] for i, s in enumerate(steps) if m_km[i]]

        r_max = max(r_probs) if r_probs else 0.0
        g_max = max(g_probs) if g_probs else 0.0
        m_max = max(m_probs) if m_probs else 0.0

        event_preds.append({
            'canonical_parent_key': identity, 'event_id': eid,
            'mechanism_route': steps[0]['mechanism_route'],
            'event_ordinal': steps[0]['event_ordinal'],
            'is_later_event': steps[0]['is_later_event'],
            'release_score_max': r_max, 'release_target': any(s['release_target'] and s['release_known_mask'] for s in steps),
            'grasp_score_max': g_max, 'grasp_target': any(s['grasp_target'] and s['grasp_known_mask'] for s in steps),
            'manipulation_score_max': m_max, 'manipulation_target': any(s['manipulation_target'] and s['manipulation_known_mask'] for s in steps),
            'steps_in_event': len(steps),
            'known_steps_release': sum(r_km),
        })

    return event_preds, step_preds


def compute_metrics(event_preds, threshold=0.5):
    metrics = {}
    for head in ['grasp', 'manipulation', 'release']:
        pos_events = [e for e in event_preds if e[f'{head}_target']]
        if pos_events:
            detected = sum(1 for e in pos_events if e[f'{head}_score_max'] >= threshold)
            metrics[f'{head}_recall'] = detected / len(pos_events)
            metrics[f'{head}_n_pos'] = len(pos_events)

            all_labels = [e[f'{head}_target'] for e in event_preds]
            all_scores = [e[f'{head}_score_max'] for e in event_preds]
            metrics[f'{head}_auroc'] = compute_auroc(all_labels, all_scores)
        else:
            metrics[f'{head}_recall'] = None
            metrics[f'{head}_auroc'] = None

    # Per-route
    for route in SUPPORTED_ROUTES:
        route_events = [e for e in event_preds if e['mechanism_route'] == route]
        for head in ['release']:
            pos = [e for e in route_events if e[f'{head}_target']]
            if pos:
                detected = sum(1 for e in pos if e[f'{head}_score_max'] >= threshold)
                metrics[f'{route}_{head}_recall'] = detected / len(pos)

    # Later event
    for ek, label in [('first', False), ('later', True)]:
        ek_events = [e for e in event_preds if e['is_later_event'] == label and e['release_target']]
        if ek_events:
            detected = sum(1 for e in ek_events if e['release_score_max'] >= threshold)
            metrics[f'{ek}_event_release_recall'] = detected / len(ek_events)

    # Duration buckets
    buckets = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 999)]
    for lo, hi in buckets:
        bucket_pos = [e for e in event_preds if e['release_target'] and lo <= e['steps_in_event'] < hi]
        if bucket_pos:
            detected = sum(1 for e in bucket_pos if e['release_score_max'] >= threshold)
            metrics[f'release_recall_dur_{lo}_{hi}'] = detected / len(bucket_pos)
            metrics[f'release_count_dur_{lo}_{hi}'] = len(bucket_pos)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output-dir', type=Path,
                    default=FORENSICS_OUT)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = args.output_dir.resolve()

    # Load fold bundle and registry
    folds = load_fit_fold_bundle(FOLD_ROOT)
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}

    # Verify source roots once
    verify_factorized_source_roots(S1, TEACHER)
    verify_sealed_directory(POLICY_INTENT)
    from gripper_attack.v5_dataset import load_policy_intent_root
    policy_index, _ = load_policy_intent_root(POLICY_INTENT)

    all_results = {}

    for mt, fold_id, seed in SAMPLED_CHECKPOINTS:
        use_9d = (mt == '25D9D')
        ckpt_dir = TRAINING_OUT / mt / f'fold{fold_id}_seed{seed}'
        verify_sealed_directory(ckpt_dir)

        print(f'\n=== {mt} fold{fold_id} seed{seed} ===')

        # Load checkpoint
        norm = json.loads((ckpt_dir / 'normalization.json').read_text())
        mean_25d = torch.tensor(norm['mean_25d'])
        std_25d = torch.tensor(norm['std_25d'])
        mean_9d = std_9d = None
        if use_9d:
            mean_9d = torch.tensor(norm['mean_9d'])
            std_9d = torch.tensor(norm['std_9d'])

        ckpt = torch.load(ckpt_dir / 'checkpoint.pt', map_location=device)
        model = FactorizedStudent(use_9d=use_9d).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        # Get fold identities
        fold = [f for f in folds['folds'] if f['fold_id'] == fold_id][0]
        train_ids = set(fold['train_identities'])
        val_ids = set(fold['validation_identities'])

        # Load episodes
        train_rows = [id_to_row[i] for i in train_ids if i in id_to_row]
        val_rows = [id_to_row[i] for i in val_ids if i in id_to_row]

        train_eps = load_factorized_episodes(S1, TEACHER, train_rows, policy_index=policy_index if use_9d else None)
        val_eps = load_factorized_episodes(S1, TEACHER, val_rows, policy_index=policy_index if use_9d else None)

        # Predict
        print(f'  Train: {len(train_eps)} episodes...')
        train_ev, _ = predict_episodes(model, train_eps, mean_25d, std_25d, mean_9d, std_9d, use_9d, device)
        print(f'  Val: {len(val_eps)} episodes...')
        val_ev, _ = predict_episodes(model, val_eps, mean_25d, std_25d, mean_9d, std_9d, use_9d, device)

        # Metrics
        train_m = compute_metrics(train_ev)
        val_m = compute_metrics(val_ev)
        key = f'{mt}_fold{fold_id}_seed{seed}'

        print(f'  Train release recall@0.5: {train_m.get("release_recall", "N/A")}')
        print(f'  Val   release recall@0.5: {val_m.get("release_recall", "N/A")}')
        if train_m.get('release_auroc') and val_m.get('release_auroc'):
            print(f'  Train release AUROC: {train_m["release_auroc"]:.4f}')
            print(f'  Val   release AUROC: {val_m["release_auroc"]:.4f}')

        # Duration bucket comparison
        for lo, hi in [(0, 15), (15, 30), (30, 50), (50, 100)]:
            tk = f'release_recall_dur_{lo}_{hi}'
            if tk in train_m and tk in val_m:
                print(f'  Dur [{lo},{hi}): train={train_m[tk]:.3f} val={val_m[tk]:.3f} (n_train={train_m.get(f"release_count_dur_{lo}_{hi}",0)} n_val={val_m.get(f"release_count_dur_{lo}_{hi}",0)})')

        all_results[key] = {'train': train_m, 'val': val_m}

    # ── Write output ──
    output_file = out_dir / 'r1c_train_vs_oof_metrics.json'
    tmp = output_file.with_suffix('.tmp')
    tmp.write_text(json.dumps(all_results, indent=2))
    os.replace(tmp, output_file)

    # Summary
    print('\n=== R1c Summary ===')
    for key, res in all_results.items():
        tr = res['train'].get('release_recall')
        vr = res['val'].get('release_recall')
        ta = res['train'].get('release_auroc')
        va = res['val'].get('release_auroc')
        gap_recall = (tr - vr) if (tr is not None and vr is not None) else None
        gap_auroc = (ta - va) if (ta is not None and va is not None) else None
        print(f'  {key}: train_recall={tr:.4f} val_recall={vr:.4f} gap={gap_recall:.4f} | train_auroc={ta:.4f} val_auroc={va:.4f} gap={gap_auroc:.4f}' if tr else f'  {key}: N/A')

    print(f'\nOutput: {output_file}')


if __name__ == '__main__':
    main()
