#!/usr/bin/env python3
"""Phase R1d: Hidden-state reset counterfactual experiment.

Tests three inference variants on existing checkpoints:
  A. Original continuous GRU hidden state
  B. Reset hidden state at Teacher event boundaries
  C. Reset hidden state after first release event
  N. Random-position reset (negative control)

Compares later-event release recall with duration/route matching.
Read-only forensic — does not modify checkpoints.
"""
import argparse, csv, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict
from statistics import mean

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

SAMPLED = [('25D9D', 0, 42), ('25D9D', 1, 123), ('25D', 0, 42)]


def predict_with_resets(model, episodes, mean_25d, std_25d, mean_9d, std_9d,
                        use_9d, device, reset_mode='original', random_seed=42):
    """Run inference with optional hidden-state reset strategies.

    reset_mode:
      'original' - continuous GRU (baseline)
      'event_boundary' - reset at every Teacher event_id transition
      'after_first_release' - reset after first release event ends
      'random_positions' - reset at random positions (negative control)
    """
    model.eval()
    all_event_preds = []
    rng = __import__('random').Random(random_seed)

    with torch.no_grad():
        for ep in episodes:
            T = len(ep.features_25d)
            route = ep.mechanism_route
            if not ep.route_supported:
                continue

            x25_raw = ep.features_25d
            # Normalize
            x25_norm = ((x25_raw - mean_25d) / std_25d).to(device)
            m25 = ep.valid_mask.to(device)
            x9_norm = m9 = None
            if use_9d and ep.policy_intent_9d.numel() > 0:
                x9_norm = ((ep.policy_intent_9d - mean_9d) / std_9d).to(device)
                m9 = ep.policy_intent_valid_mask.to(device)

            # Initialize hidden state
            h_25d = torch.zeros(1, model.hidden_dim, device=device)
            h_9d = torch.zeros(1, model.hidden_dim, device=device) if use_9d else None

            eids = ep.event_id
            unique_events = sorted([e.item() for e in eids.unique() if e.item() >= 0])
            eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}

            # Determine reset points
            reset_steps = set()
            if reset_mode == 'event_boundary':
                prev_eid = -1
                for t in range(T):
                    eid = int(eids[t].item())
                    if eid >= 0 and eid != prev_eid:
                        reset_steps.add(t)
                    prev_eid = eid if eid >= 0 else prev_eid
            elif reset_mode == 'after_first_release':
                # Find the step after first release event ends
                first_release_end = None
                for t in range(T):
                    eid = int(eids[t].item())
                    if eid >= 0 and eid_to_ordinal.get(eid, -1) == 0:
                        # first event
                        if ep.release_target[t].item() and ep.release_known_mask[t].item():
                            first_release_end = t
                if first_release_end is not None:
                    # Reset at the step after release ends (first non-release step)
                    for t in range(first_release_end + 1, T):
                        if not (ep.release_target[t].item() and ep.release_known_mask[t].item()):
                            reset_steps.add(t)
                            break
            elif reset_mode == 'random_positions':
                # Same number of resets as event_boundary would have
                n_resets = len(set(int(eids[t].item()) for t in range(T) if eids[t].item() >= 0))
                candidates = list(range(T // 4, 3 * T // 4))  # avoid edges
                if len(candidates) > n_resets:
                    reset_steps = set(rng.sample(candidates, n_resets))

            # Step-by-step inference
            g_probs, m_probs, r_probs = [], [], []
            for t in range(T):
                # Reset if needed
                if t in reset_steps:
                    h_25d = torch.zeros(1, model.hidden_dim, device=device)
                    if h_9d is not None:
                        h_9d = torch.zeros(1, model.hidden_dim, device=device)

                if m25[t]:
                    h_25d = model.gru_25d(x25_norm[t:t+1], h_25d)
                    if use_9d and h_9d is not None and m9 is not None and m9[t]:
                        h_9d = model.gru_9d(x9_norm[t:t+1], h_9d)

                if use_9d and h_9d is not None:
                    fused = model.fusion(torch.cat([h_25d, h_9d], dim=-1))
                else:
                    fused = h_25d

                probs = model._route_probs(fused, route)
                g_probs.append(float(probs['grasp'][0].item()))
                m_probs.append(float(probs['manipulation'][0].item()))
                r_probs.append(float(probs['release'][0].item()))

            # Aggregate to event-level
            for eid in unique_events:
                em = eids == eid
                t_indices = [t for t in range(T) if em[t].item()]
                r_km = [ep.release_known_mask[ti].item() for ti in t_indices]
                g_km = [ep.grasp_known_mask[ti].item() for ti in t_indices]
                m_km = [ep.manipulation_known_mask[ti].item() for ti in t_indices]

                r_scores = [r_probs[ti] for i, ti in enumerate(t_indices) if r_km[i]]
                g_scores = [g_probs[ti] for i, ti in enumerate(t_indices) if g_km[i]]
                m_scores = [m_probs[ti] for i, ti in enumerate(t_indices) if m_km[i]]

                r_max = max(r_scores) if r_scores else 0.0
                g_max = max(g_scores) if g_scores else 0.0
                m_max = max(m_scores) if m_scores else 0.0

                ordinal = eid_to_ordinal.get(eid, -1)
                all_event_preds.append({
                    'canonical_parent_key': ep.canonical_parent_key,
                    'event_id': eid,
                    'mechanism_route': route,
                    'event_ordinal': ordinal,
                    'is_later_event': ordinal >= 1,
                    'release_score_max': r_max,
                    'release_target': any(ep.release_target[ti].item() and ep.release_known_mask[ti].item() for ti in t_indices),
                    'grasp_score_max': g_max,
                    'grasp_target': any(ep.grasp_target[ti].item() and ep.grasp_known_mask[ti].item() for ti in t_indices),
                    'manipulation_score_max': m_max,
                    'manipulation_target': any(ep.manipulation_target[ti].item() and ep.manipulation_known_mask[ti].item() for ti in t_indices),
                    'steps_in_event': len(t_indices),
                    'known_steps_release': sum(r_km),
                })

    return all_event_preds


def compute_split_metrics(events, threshold=0.5):
    """Compute metrics separately for first/later events and by route."""
    metrics = {}

    for head in ['grasp', 'manipulation', 'release']:
        for ek, label in [('first', False), ('later', True)]:
            ek_events = [e for e in events if e['is_later_event'] == label and e[f'{head}_target']]
            if ek_events:
                detected = sum(1 for e in ek_events if e[f'{head}_score_max'] >= threshold)
                metrics[f'{head}_{ek}_recall'] = detected / len(ek_events)
                metrics[f'{head}_{ek}_count'] = len(ek_events)

    # Duration-matched comparison: split by event length
    for ek, label in [('first', False), ('later', True)]:
        for lo, hi in [(0, 20), (20, 40), (40, 80), (80, 999)]:
            ek_events = [e for e in events
                        if e['is_later_event'] == label and e['release_target']
                        and lo <= e['steps_in_event'] < hi]
            if ek_events:
                detected = sum(1 for e in ek_events if e['release_score_max'] >= threshold)
                metrics[f'release_{ek}_dur_{lo}_{hi}'] = detected / len(ek_events)

    # Per-route later-event
    for route in SUPPORTED_ROUTES:
        for ek, label in [('first', False), ('later', True)]:
            ek_events = [e for e in events
                        if e['mechanism_route'] == route
                        and e['is_later_event'] == label and e['release_target']]
            if ek_events:
                detected = sum(1 for e in ek_events if e['release_score_max'] >= threshold)
                metrics[f'release_{route}_{ek}_recall'] = detected / len(ek_events)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output-dir', type=Path, default=FORENSICS_OUT)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = args.output_dir.resolve()

    folds = load_fit_fold_bundle(FOLD_ROOT)
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}

    verify_factorized_source_roots(S1, TEACHER)
    verify_sealed_directory(POLICY_INTENT)
    from gripper_attack.v5_dataset import load_policy_intent_root
    policy_index, _ = load_policy_intent_root(POLICY_INTENT)

    all_results = {}
    modes = ['original', 'event_boundary', 'after_first_release', 'random_positions']

    for mt, fold_id, seed in SAMPLED:
        use_9d = (mt == '25D9D')
        ckpt_dir = TRAINING_OUT / mt / f'fold{fold_id}_seed{seed}'
        verify_sealed_directory(ckpt_dir)

        print(f'\n=== {mt} fold{fold_id} seed{seed} ===')

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

        fold = [f for f in folds['folds'] if f['fold_id'] == fold_id][0]
        val_ids = set(fold['validation_identities'])
        val_rows = [id_to_row[i] for i in val_ids if i in id_to_row]
        val_eps = load_factorized_episodes(S1, TEACHER, val_rows,
                                            policy_index=policy_index if use_9d else None)

        key = f'{mt}_fold{fold_id}_seed{seed}'
        all_results[key] = {}

        for mode in modes:
            print(f'  {mode}...')
            events = predict_with_resets(model, val_eps, mean_25d, std_25d,
                                         mean_9d, std_9d, use_9d, device, reset_mode=mode)
            metrics = compute_split_metrics(events)
            all_results[key][mode] = metrics

            lr = metrics.get('release_later_recall', 'N/A')
            fr = metrics.get('release_first_recall', 'N/A')
            print(f'    first_recall={fr:.4f} later_recall={lr:.4f}' if isinstance(lr, float) else f'    first_recall={fr} later_recall={lr}')

    # ── Compute delta vs original ──
    print('\n=== Reset Delta vs Original ===')
    delta_summary = {}
    for key, mode_results in all_results.items():
        baseline = mode_results.get('original', {})
        delta_summary[key] = {}
        for mode in ['event_boundary', 'after_first_release', 'random_positions']:
            if mode not in mode_results:
                continue
            mr = mode_results[mode]
            delta = {}
            for metric in ['release_later_recall', 'release_first_recall']:
                if metric in baseline and metric in mr:
                    delta[metric] = mr[metric] - baseline[metric]
            delta_summary[key][mode] = delta

            lr_d = delta.get('release_later_recall', 0)
            fr_d = delta.get('release_first_recall', 0)
            print(f'  {key} {mode}: Δlater={lr_d:+.4f} Δfirst={fr_d:+.4f}')

    # Write output
    output = {
        'per_checkpoint': all_results,
        'delta_vs_original': delta_summary,
    }
    output_file = out_dir / 'r1d_reset_counterfactual.json'
    tmp = output_file.with_suffix('.tmp')
    tmp.write_text(json.dumps(output, indent=2))
    os.replace(tmp, output_file)
    print(f'\nOutput: {output_file}')


if __name__ == '__main__':
    main()
