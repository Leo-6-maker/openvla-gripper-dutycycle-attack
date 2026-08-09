#!/usr/bin/env python3
"""V2 inner-CV prediction audit — full step-level parity verification.

Reloads original validation episodes and compares every step:
identity, step_index, event_id, event_role, mechanism_route, route_supported,
all three head targets and known masks, event ordinal, event duration.
"""
import argparse, csv, hashlib, json, os, sys, uuid, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_dataset import (
    load_factorized_episodes, verify_factorized_source_roots,
)
from gripper_attack.b3_training_protocol import sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids, get_outer_val_ids

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
S1 = OPS / 'OFFICIAL_V3_S1_FIT_V1_d31187f'
TEACHER = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'
SUPPORTED_ROUTES = ['single_object_pick_place', 'multi_object_transfer']


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


def audit_full_parity(pred_dir, split_bundle, outer_fold, inner_fold):
    """Full step-level parity audit against original episodes."""
    issues = []
    stats = {}

    # Verify seal
    try:
        verify_sealed_directory(pred_dir)
        stats['seal_ok'] = True
    except Exception as e:
        issues.append(f'SEAL_FAIL: {e}')
        return issues, stats

    # Load predictions
    step_file = pred_dir / 'heldout_step_predictions.jsonl'
    event_file = pred_dir / 'heldout_event_predictions.jsonl'
    if not step_file.is_file():
        issues.append('MISSING_STEP_PREDICTIONS')
        return issues, stats
    if not event_file.is_file():
        issues.append('MISSING_EVENT_PREDICTIONS')
        return issues, stats

    steps = [json.loads(l) for l in step_file.read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in event_file.read_text().splitlines() if l.strip()]

    stats['pred_step_count'] = len(steps)
    stats['pred_event_count'] = len(events)

    # Build prediction index: (identity, step_index) -> prediction
    pred_by_step = {}
    for s in steps:
        key = (s['canonical_parent_key'], s['step_index'])
        if key in pred_by_step:
            issues.append(f'DUPLICATE_STEP: {key}')
        pred_by_step[key] = s

    # Get expected inner validation identities
    _, inner_val_ids = resolve_inner_train_val_ids(split_bundle, outer_fold, inner_fold)
    outer_val_ids = get_outer_val_ids(split_bundle, outer_fold)

    # Identity checks
    predicted_ids = set(s['canonical_parent_key'] for s in steps)
    extra = predicted_ids - inner_val_ids
    if extra:
        issues.append(f'PREDICTED_NOT_INNER_VAL: {len(extra)}')
    missing_ids = inner_val_ids - predicted_ids
    if missing_ids:
        issues.append(f'MISSING_INNER_VAL: {len(missing_ids)}')
    outer_contam = predicted_ids & outer_val_ids
    if outer_contam:
        issues.append(f'OUTER_VAL_CONTAMINATION: {len(outer_contam)}')

    stats['expected_identities'] = len(inner_val_ids)
    stats['predicted_identities'] = len(predicted_ids)

    # Load original episodes for step-level comparison
    verify_factorized_source_roots(S1, TEACHER)
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    id_to_row = {r['canonical_parent_key']: r for r in fit_rows}
    val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]
    val_eps = load_factorized_episodes(S1, TEACHER, val_rows)

    stats['loaded_episodes'] = len(val_eps)

    # Per-episode parity
    step_mismatches = 0
    target_mismatches = 0
    mask_mismatches = 0
    route_mismatches = 0
    nan_inf_count = 0
    prob_range_issues = 0
    sigmoid_issues = 0
    unsupported_emit = 0
    route_sup_mismatches = 0
    event_id_mismatches = 0
    event_role_mismatches = 0

    for ep in val_eps:
        ident = ep.canonical_parent_key
        T = len(ep.features_25d)

        # Check step count
        ep_steps = [s for s in steps if s['canonical_parent_key'] == ident]
        if len(ep_steps) != T:
            issues.append(f'STEP_COUNT: {ident}: pred={len(ep_steps)} expected={T}')
            step_mismatches += abs(len(ep_steps) - T)

        # Check step_index continuity
        ep_step_indices = sorted([s['step_index'] for s in ep_steps])
        expected_indices = list(range(T))
        if ep_step_indices != expected_indices:
            missing = set(expected_indices) - set(ep_step_indices)
            extra_idx = set(ep_step_indices) - set(expected_indices)
            if missing:
                issues.append(f'MISSING_STEPS: {ident}: {sorted(missing)[:5]}...')
            if extra_idx:
                issues.append(f'EXTRA_STEPS: {ident}: {sorted(extra_idx)[:5]}...')
            step_mismatches += len(missing) + len(extra_idx)

        # Per-step comparison
        eids = ep.event_id
        unique_event_ids = sorted(set(int(eids[t].item()) for t in range(T) if eids[t].item() >= 0))
        eid_to_ordinal = {e: i for i, e in enumerate(unique_event_ids)}
        event_dur = {}
        release_pos_dur = {}
        for eid in unique_event_ids:
            event_dur[eid] = sum(1 for t in range(T) if int(eids[t].item()) == eid)
            release_pos_dur[eid] = sum(1 for t in range(T)
                                       if int(eids[t].item()) == eid
                                       and ep.release_target[t].item()
                                       and ep.release_known_mask[t].item())

        for t in range(T):
            key = (ident, t)
            if key not in pred_by_step:
                continue
            s = pred_by_step[key]
            eid = int(eids[t].item())
            ordinal = eid_to_ordinal.get(eid, -1)

            # Route
            if s.get('mechanism_route') != ep.mechanism_route:
                route_mismatches += 1
            if s.get('route_supported') != ep.route_supported:
                route_sup_mismatches += 1

            # Event ID
            if s.get('event_id') != eid:
                event_id_mismatches += 1
            if s.get('event_role') != ep.event_role[t]:
                event_role_mismatches += 1
            if s.get('event_ordinal') != ordinal:
                event_id_mismatches += 1
            if s.get('event_duration') != event_dur.get(eid, 0):
                event_id_mismatches += 1
            if s.get('release_positive_duration') != release_pos_dur.get(eid, 0):
                event_id_mismatches += 1

            # Targets
            for head, tgt, msk in [('grasp', ep.grasp_target, ep.grasp_known_mask),
                                    ('manipulation', ep.manipulation_target, ep.manipulation_known_mask),
                                    ('release', ep.release_target, ep.release_known_mask)]:
                if s.get(f'{head}_target') != bool(tgt[t].item()):
                    target_mismatches += 1
                if s.get(f'{head}_known_mask') != bool(msk[t].item()):
                    mask_mismatches += 1

            # Numerical checks
            for h in ['grasp', 'manipulation', 'release']:
                prob = s.get(f'{h}_prob', 0)
                logit = s.get(f'{h}_logit', 0)

                if math.isnan(prob) or math.isinf(prob):
                    nan_inf_count += 1
                if not (0.0 <= prob <= 1.0):
                    prob_range_issues += 1

                # sigmoid(logit) ≈ prob
                try:
                    sig = 1.0 / (1.0 + math.exp(-logit)) if logit < 0 else math.exp(-logit) / (1.0 + math.exp(-logit)) if logit > 700 else 1.0 / (1.0 + math.exp(-logit))
                except OverflowError:
                    sig = 0.0 if logit < -700 else 1.0
                if abs(sig - prob) > 0.02:
                    sigmoid_issues += 1

            # Unsupported route
            if not s.get('route_supported', True):
                for h in ['grasp', 'manipulation', 'release']:
                    if abs(s.get(f'{h}_prob', 0)) > 1e-8:
                        unsupported_emit += 1

    # Compile stats
    stats['step_mismatches'] = step_mismatches
    stats['target_mismatches'] = target_mismatches
    stats['mask_mismatches'] = mask_mismatches
    stats['route_mismatches'] = route_mismatches
    stats['route_sup_mismatches'] = route_sup_mismatches
    stats['event_id_mismatches'] = event_id_mismatches
    stats['event_role_mismatches'] = event_role_mismatches
    stats['nan_inf'] = nan_inf_count
    stats['prob_range_issues'] = prob_range_issues
    stats['sigmoid_issues'] = sigmoid_issues
    stats['unsupported_emit'] = unsupported_emit

    # Event-level checks
    pred_events_by_key = {}
    for e in events:
        key = (e['canonical_parent_key'], e['event_id'])
        if key in pred_events_by_key:
            issues.append(f'DUPLICATE_EVENT: {key}')
        pred_events_by_key[key] = e

    stats['total_step_issues'] = (step_mismatches + target_mismatches + mask_mismatches +
                                   route_mismatches + route_sup_mismatches + event_id_mismatches +
                                   event_role_mismatches + nan_inf_count + prob_range_issues +
                                   sigmoid_issues + unsupported_emit)

    if stats['total_step_issues'] > 0:
        issues.append(f'TOTAL_STEP_ISSUES: {stats["total_step_issues"]}')
    else:
        stats['full_parity'] = True

    return issues, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-dir', type=Path, required=True)
    ap.add_argument('--inner-cv-splits-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    verify_sealed_directory(args.inner_cv_splits_root)
    split_bundle = json.loads((args.inner_cv_splits_root / 'inner_cv_splits.json').read_text())

    manifest = json.loads((args.prediction_dir / 'prediction_manifest.json').read_text())
    outer_fold = manifest.get('outer_fold', 0)
    inner_fold = manifest.get('inner_fold', 0)

    issues, stats = audit_full_parity(args.prediction_dir, split_bundle, outer_fold, inner_fold)

    result = {
        'prediction_dir': str(args.prediction_dir),
        'outer_fold': outer_fold, 'inner_fold': inner_fold,
        'issues': issues, 'stats': stats,
        'status': 'PASS' if len(issues) == 0 else 'HOLD',
    }

    if args.output:
        out = args.output
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, out)

    print(f'Status: {result["status"]}')
    print(f'Step issues: {stats.get("total_step_issues", "N/A")}')
    print(f'Target mismatches: {stats.get("target_mismatches", "N/A")}')
    print(f'Mask mismatches: {stats.get("mask_mismatches", "N/A")}')
    print(f'Route mismatches: {stats.get("route_mismatches", "N/A")}')
    print(f'Event mismatches: {stats.get("event_id_mismatches", "N/A")}')
    print(f'NaN/Inf: {stats.get("nan_inf", "N/A")}')
    print(f'Unsupported emit: {stats.get("unsupported_emit", "N/A")}')
    print(f'Full parity: {stats.get("full_parity", False)}')

    if issues:
        sys.exit(1)


if __name__ == '__main__':
    main()
