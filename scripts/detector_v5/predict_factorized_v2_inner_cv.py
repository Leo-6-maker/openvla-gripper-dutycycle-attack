#!/usr/bin/env python3
"""V2 inner-CV held-out prediction runner. Narrow adapter on V1 predict pattern."""
import argparse, csv, hashlib, json, os, sys, uuid, platform
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    FactorizedEpisode, load_factorized_episodes,
    verify_factorized_source_roots, SUPPORTED_ROUTES,
)
from gripper_attack.v5_factorized_student_v2 import FactorizedStudentV2
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
FOLD_ROOT = OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--inner-cv-splits-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    out = args.output_root.resolve()

    # Verify checkpoint
    verify_sealed_directory(ckpt_dir)
    run_config = json.loads((ckpt_dir / 'run_config.json').read_text())
    norm = json.loads((ckpt_dir / 'normalization.json').read_text())

    candidate = run_config['candidate']
    outer_fold = run_config['outer_fold']
    inner_fold = run_config['inner_fold']
    seed = run_config['seed']
    encoder_type = run_config['encoder_type']
    hidden_dim = run_config['hidden_dim']
    receptive_field = run_config['receptive_field']
    dropout = run_config['dropout']

    mean_25d = torch.tensor(norm['mean_25d'])
    std_25d = torch.tensor(norm['std_25d'])

    # Load inner-CV splits
    verify_sealed_directory(args.inner_cv_splits_root)
    splits = json.loads((args.inner_cv_splits_root / 'inner_cv_splits.json').read_text())
    fold_data = splits['splits'][f'fold_{outer_fold}']

    # Inner validation = specified inner fold (shared resolver)
    _, inner_val_ids = resolve_inner_train_val_ids(splits, outer_fold, inner_fold)

    # Load episodes
    verify_factorized_source_roots(S1, TEACHER)
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}
    val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]
    val_eps = load_factorized_episodes(S1, TEACHER, val_rows)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load model
    ckpt = torch.load(ckpt_dir / 'checkpoint.pt', map_location=device)
    model = FactorizedStudentV2(hidden_dim=hidden_dim, receptive_field=receptive_field,
                                 encoder_type=encoder_type, dropout=dropout,
                                 use_9d=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    step_lines = []
    event_lines = []

    with torch.no_grad():
        for ep in val_eps:
            T = len(ep.features_25d)
            route = ep.mechanism_route
            route_sup = ep.route_supported

            # Compute event durations
            eids = ep.event_id
            event_dur = defaultdict(int)
            release_pos_dur = defaultdict(int)
            for t in range(T):
                eid = int(eids[t].item())
                if eid >= 0:
                    event_dur[eid] += 1
                    if ep.release_target[t].item() and ep.release_known_mask[t].item():
                        release_pos_dur[eid] += 1

            unique_events = sorted([e for e in event_dur])
            eid_to_ordinal = {e: i for i, e in enumerate(unique_events)}

            if not route_sup:
                for t in range(T):
                    step_lines.append(json.dumps({
                        'candidate_id': candidate, 'outer_fold': outer_fold,
                        'inner_fold': inner_fold, 'seed': seed,
                        'canonical_parent_key': ep.canonical_parent_key,
                        'suite': ep.suite, 'task_idx': ep.task_idx, 'state_id': ep.state_id,
                        'mechanism_route': route, 'route_supported': False,
                        'step_index': t, 'event_id': int(eids[t].item()),
                        'event_ordinal': -1, 'is_later_event': False,
                        'event_role': ep.event_role[t],
                        'event_duration': event_dur.get(int(eids[t].item()), 0),
                        'release_positive_duration': release_pos_dur.get(int(eids[t].item()), 0),
                        'window_id': -1, 'position_in_window': -1,
                    'encoder_type': 'none', 'window_size': 0,
                        'grasp_prob': 0.0, 'manipulation_prob': 0.0, 'release_prob': 0.0,
                        'grasp_logit': -1e4, 'manipulation_logit': -1e4, 'release_logit': -1e4,
                        'grasp_target': bool(ep.grasp_target[t].item()),
                        'grasp_known_mask': bool(ep.grasp_known_mask[t].item()),
                        'manipulation_target': bool(ep.manipulation_target[t].item()),
                        'manipulation_known_mask': bool(ep.manipulation_known_mask[t].item()),
                        'release_target': bool(ep.release_target[t].item()),
                        'release_known_mask': bool(ep.release_known_mask[t].item()),
                    }) + '\n')
                continue

            x25 = ((ep.features_25d - mean_25d) / std_25d).unsqueeze(0).to(device)
            m25 = ep.valid_mask.unsqueeze(0).to(device)
            probs = model.forward_sequence(x25, None, m25, None, route)
            logits = model.forward_logits(x25, None, m25, None, route)

            g_prob = probs['grasp'][0].cpu()
            m_prob = probs['manipulation'][0].cpu()
            r_prob = probs['release'][0].cpu()
            g_logit = logits['grasp'][0].cpu()
            m_logit = logits['manipulation'][0].cpu()
            r_logit = logits['release'][0].cpu()

            # Window metadata
            if encoder_type == 'windowed_gru':
                W = receptive_field
            else:
                W = model.encoder_25d.actual_receptive_field

            for t in range(T):
                ev = int(eids[t].item())
                window_id = t // W if W > 0 else -1
                step_lines.append(json.dumps({
                    'candidate_id': candidate, 'outer_fold': outer_fold,
                    'inner_fold': inner_fold, 'seed': seed,
                    'canonical_parent_key': ep.canonical_parent_key,
                    'suite': ep.suite, 'task_idx': ep.task_idx, 'state_id': ep.state_id,
                    'mechanism_route': route, 'route_supported': True,
                    'step_index': t, 'event_id': ev,
                    'event_ordinal': eid_to_ordinal.get(ev, -1),
                    'is_later_event': eid_to_ordinal.get(ev, -1) >= 1,
                    'event_role': ep.event_role[t],
                    'event_duration': event_dur.get(ev, 0),
                    'release_positive_duration': release_pos_dur.get(ev, 0),
                    'window_id': window_id if encoder_type == 'windowed_gru' else t // W,
                    'position_in_window': t % W if W > 0 else t,
                    'encoder_type': encoder_type,
                    'window_size': receptive_field if encoder_type == 'windowed_gru' else model.encoder_25d.actual_receptive_field,
                    'grasp_prob': round(float(g_prob[t].item()), 8),
                    'manipulation_prob': round(float(m_prob[t].item()), 8),
                    'release_prob': round(float(r_prob[t].item()), 8),
                    'grasp_logit': round(float(g_logit[t].item()), 8),
                    'manipulation_logit': round(float(m_logit[t].item()), 8),
                    'release_logit': round(float(r_logit[t].item()), 8),
                    'grasp_target': bool(ep.grasp_target[t].item()),
                    'grasp_known_mask': bool(ep.grasp_known_mask[t].item()),
                    'manipulation_target': bool(ep.manipulation_target[t].item()),
                    'manipulation_known_mask': bool(ep.manipulation_known_mask[t].item()),
                    'release_target': bool(ep.release_target[t].item()),
                    'release_known_mask': bool(ep.release_known_mask[t].item()),
                }) + '\n')

        # Event-level aggregation
        step_recs = [json.loads(l) for l in step_lines]
        ev_groups = defaultdict(list)
        for rec in step_recs:
            if rec['event_id'] >= 0 and rec['route_supported']:
                ev_groups[(rec['canonical_parent_key'], rec['event_id'])].append(rec)

        for (identity, eid), steps in ev_groups.items():
            first_step = steps[0]
            g_km = [s['grasp_known_mask'] for s in steps]
            m_km = [s['manipulation_known_mask'] for s in steps]
            r_km = [s['release_known_mask'] for s in steps]
            g_probs = [s['grasp_prob'] for i, s in enumerate(steps) if g_km[i]]
            m_probs = [s['manipulation_prob'] for i, s in enumerate(steps) if m_km[i]]
            r_probs = [s['release_prob'] for i, s in enumerate(steps) if r_km[i]]

            g_max = max(g_probs) if g_probs else 0.0
            m_max = max(m_probs) if m_probs else 0.0
            r_max = max(r_probs) if r_probs else 0.0

            def first_crossing(probs, threshold=0.5):
                for i, p in enumerate(probs):
                    if p >= threshold:
                        return i
                return -1

            def coverage(probs, threshold=0.5):
                return sum(1 for p in probs if p >= threshold) / max(1, len(probs))

            event_lines.append(json.dumps({
                'candidate_id': candidate, 'outer_fold': outer_fold,
                'inner_fold': inner_fold, 'seed': seed,
                'canonical_parent_key': identity, 'event_id': eid,
                'mechanism_route': first_step['mechanism_route'],
                'event_ordinal': first_step['event_ordinal'],
                'is_later_event': first_step['is_later_event'],
                'event_duration': first_step['event_duration'],
                'release_positive_duration': first_step['release_positive_duration'],
                'grasp_event_score': round(g_max, 8),
                'manipulation_event_score': round(m_max, 8),
                'release_event_score': round(r_max, 8),
                'grasp_target': any(s['grasp_target'] and s['grasp_known_mask'] for s in steps),
                'manipulation_target': any(s['manipulation_target'] and s['manipulation_known_mask'] for s in steps),
                'release_target': any(s['release_target'] and s['release_known_mask'] for s in steps),
                'grasp_known_steps': sum(g_km),
                'manipulation_known_steps': sum(m_km),
                'release_known_steps': sum(r_km),
                'grasp_first_crossing_05': first_crossing(g_probs),
                'manipulation_first_crossing_05': first_crossing(m_probs),
                'release_first_crossing_05': first_crossing(r_probs),
                'grasp_coverage_05': round(coverage(g_probs), 8),
                'manipulation_coverage_05': round(coverage(m_probs), 8),
                'release_coverage_05': round(coverage(r_probs), 8),
            }) + '\n')

    manifest = {'candidate': candidate, 'outer_fold': outer_fold, 'inner_fold': inner_fold,
                'seed': seed, 'total_steps': len(step_lines), 'total_events': len(event_lines),
                'total_episodes': len(val_eps)}

    _atomic_text(staging / 'heldout_step_predictions.jsonl', ''.join(step_lines))
    _atomic_text(staging / 'heldout_event_predictions.jsonl', ''.join(event_lines))
    _atomic_text(staging / 'prediction_manifest.json', json.dumps(manifest, indent=2))
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'checkpoint_dir': str(ckpt_dir), 'checkpoint_seal': sha256_file(ckpt_dir / 'SHA256SUMS'),
        'inner_cv_splits_root': str(args.inner_cv_splits_root),
        'candidate': candidate, 'outer_fold': outer_fold, 'inner_fold': inner_fold, 'seed': seed,
    }, indent=2))
    _atomic_text(staging / 'environment.json', json.dumps({
        'python_version': platform.python_version(), 'torch': torch.__version__,
        'host': platform.node(),
    }, indent=2))
    write_seal(staging)
    os.replace(staging, out)
    print(f'Prediction sealed: {out}')
    print(f'  Steps: {len(step_lines)} Events: {len(event_lines)} Eps: {len(val_eps)}')


if __name__ == '__main__':
    main()
