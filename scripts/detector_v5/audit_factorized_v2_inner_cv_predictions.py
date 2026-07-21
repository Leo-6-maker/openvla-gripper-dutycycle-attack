#!/usr/bin/env python3
"""V2 inner-CV prediction audit: identity integrity, route correctness, seal verification."""
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.b3_training_protocol import sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids, get_outer_val_ids

SUPPORTED_ROUTES = ['single_object_pick_place', 'multi_object_transfer']


def audit_prediction(pred_dir, split_bundle, outer_fold, inner_fold):
    """Audit a single prediction shard."""
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
    manifest_file = pred_dir / 'prediction_manifest.json'

    if not step_file.is_file():
        issues.append('MISSING_STEP_PREDICTIONS')
        return issues, stats
    if not event_file.is_file():
        issues.append('MISSING_EVENT_PREDICTIONS')
        return issues, stats

    steps = [json.loads(l) for l in step_file.read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in event_file.read_text().splitlines() if l.strip()]
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}

    stats['step_count'] = len(steps)
    stats['event_count'] = len(events)

    # Identity checks
    predicted_ids = set()
    predicted_ids_by_episode = {}
    for s in steps:
        ident = s['canonical_parent_key']
        predicted_ids.add(ident)
        predicted_ids_by_episode.setdefault(ident, set()).add(s['step_index'])

    # Expected inner validation identities
    _, inner_val_ids = resolve_inner_train_val_ids(split_bundle, outer_fold, inner_fold)
    outer_val_ids = get_outer_val_ids(split_bundle, outer_fold)

    # All predicted must be inner val
    extra = predicted_ids - inner_val_ids
    if extra:
        issues.append(f'PREDICTED_NOT_INNER_VAL: {len(extra)} identities')
    missing = inner_val_ids - predicted_ids
    if missing:
        issues.append(f'MISSING_INNER_VAL: {len(missing)} identities')

    # No outer val contamination
    outer_contam = predicted_ids & outer_val_ids
    if outer_contam:
        issues.append(f'OUTER_VAL_CONTAMINATION: {len(outer_contam)}')

    # No inner train contamination
    inner_train_ids, _ = resolve_inner_train_val_ids(split_bundle, outer_fold, inner_fold)
    train_contam = predicted_ids & inner_train_ids
    if train_contam:
        issues.append(f'INNER_TRAIN_CONTAMINATION: {len(train_contam)}')

    # NaN/Inf check
    nan_count = 0
    for s in steps:
        for k in ['grasp_prob', 'manipulation_prob', 'release_prob',
                   'grasp_logit', 'manipulation_logit', 'release_logit']:
            v = s.get(k, 0)
            if v != v or v == float('inf') or v == float('-inf'):
                nan_count += 1
                break
    if nan_count > 0:
        issues.append(f'NaN/Inf: {nan_count} steps')

    # Unsupported route: probabilities must be exactly 0
    unsupported_emit = 0
    for s in steps:
        if not s.get('route_supported', True):
            if (abs(s.get('grasp_prob', 0)) > 1e-8 or
                abs(s.get('manipulation_prob', 0)) > 1e-8 or
                abs(s.get('release_prob', 0)) > 1e-8):
                unsupported_emit += 1
    if unsupported_emit > 0:
        issues.append(f'UNSUPPORTED_EMIT: {unsupported_emit} steps')
    stats['unsupported_emit'] = unsupported_emit

    # Route consistency: step route must match event route
    route_mismatch = 0
    for s in steps:
        if s.get('route_supported') and s.get('mechanism_route') not in SUPPORTED_ROUTES:
            route_mismatch += 1
    if route_mismatch > 0:
        issues.append(f'ROUTE_MISMATCH: {route_mismatch}')

    # Event ordinal consistency
    ordinal_issues = 0
    for e in events:
        if e.get('is_later_event') and e.get('event_ordinal', -1) < 1:
            ordinal_issues += 1
    if ordinal_issues > 0:
        issues.append(f'EVENT_ORDINAL_INCONSISTENCY: {ordinal_issues}')

    stats['unique_identities'] = len(predicted_ids)
    stats['expected_identities'] = len(inner_val_ids)
    stats['issues'] = len(issues)

    return issues, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-dir', type=Path, required=True)
    ap.add_argument('--inner-cv-splits-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    verify_sealed_directory(args.inner_cv_splits_root)
    split_bundle = json.loads((args.inner_cv_splits_root / 'inner_cv_splits.json').read_text())

    # Detect outer/inner fold from prediction directory name or manifest
    manifest = json.loads((args.prediction_dir / 'prediction_manifest.json').read_text())
    outer_fold = manifest.get('outer_fold', 0)
    inner_fold = manifest.get('inner_fold', 0)

    issues, stats = audit_prediction(args.prediction_dir, split_bundle, outer_fold, inner_fold)

    result = {
        'prediction_dir': str(args.prediction_dir),
        'outer_fold': outer_fold, 'inner_fold': inner_fold,
        'issues': issues,
        'stats': stats,
        'status': 'PASS' if len(issues) == 0 else 'HOLD',
    }

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    if issues:
        print(f'HOLD: {len(issues)} issues')
        for issue in issues:
            print(f'  {issue}')
        sys.exit(1)
    else:
        print(f'PASS: {stats["step_count"]} steps, {stats["event_count"]} events, {stats["unique_identities"]} identities')


if __name__ == '__main__':
    main()
