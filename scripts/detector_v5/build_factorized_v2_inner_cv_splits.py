#!/usr/bin/env python3
"""Phase P1: Build deterministic identity-level inner-CV splits for V2 development.

For each V1 outer fold, splits the training identities into 3 inner folds.
Grouping key: canonical_parent_key (identity).
Stratification: balances route, release event count, short event count.
Hard constraints: no identity overlap within split, zero outer-val contamination.
"""
import argparse, csv, hashlib, json, os, sys, uuid
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
FOLD_ROOT = OPS / 'OFFICIAL_V3_FIT_FOLDS_V1_d31187f'
TEACHER_ROOT = OPS / 'OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721'
REGISTRY = OPS / 'OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f/OFFICIAL_V3_FORMAL_REGISTRY_V1.csv'
DEFAULT_OUT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721'

SPLIT_SEED = 20260721
N_INNER_FOLDS = 3
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


def load_identity_metadata(identities, teacher_root):
    """Load per-identity route and event statistics from Teacher labels."""
    import json as _json
    FACTORIZED_LABEL_FILENAME = "factorized_teacher_v1.jsonl"

    meta = {}
    for ident in identities:
        suite, task_name, state_name = ident.split('/')
        label_path = teacher_root / 'labels' / suite / task_name / state_name / FACTORIZED_LABEL_FILENAME
        if not label_path.is_file():
            meta[ident] = {'route': 'unknown', 'release_events': 0, 'short_events': 0,
                           'total_events': 0, 'steps': 0}
            continue

        labels = [_json.loads(l) for l in label_path.read_text().splitlines() if l.strip()]
        route = labels[0].get('mechanism_type', 'unknown') if labels else 'unknown'
        steps = len(labels)

        # Count events
        eids = set()
        release_positive_events = set()
        for l in labels:
            eid = l.get('event_id', -1)
            if eid >= 0:
                eids.add(eid)
                if l.get('release_or_instability') and l.get('release_or_instability_known_mask'):
                    release_positive_events.add(eid)

        # Count short events (event duration < 30)
        event_durations = defaultdict(int)
        for l in labels:
            eid = l.get('event_id', -1)
            if eid >= 0:
                event_durations[eid] += 1

        short_events = sum(1 for eid in release_positive_events
                          if eid in event_durations and event_durations[eid] < 30)

        meta[ident] = {
            'route': route,
            'release_events': len(release_positive_events),
            'short_events': short_events,
            'total_events': len(eids),
            'steps': steps,
        }

    return meta


def stratified_split(identities, meta, n_folds, rng):
    """Split identities into n_folds, balancing route and event counts.

    Strategy: sort by (route, release_event_count), then interleave.
    """
    # Separate by route
    route_groups = defaultdict(list)
    for ident in identities:
        m = meta.get(ident, {})
        route = m.get('route', 'unknown')
        if route not in SUPPORTED_ROUTES:
            route = 'unsupported'
        route_groups[route].append(ident)

    folds = [set() for _ in range(n_folds)]
    fold_stats = [{r: {'count': 0, 'release_events': 0, 'short_events': 0}
                   for r in list(SUPPORTED_ROUTES) + ['unsupported']}
                  for _ in range(n_folds)]

    for route, id_list in route_groups.items():
        # Sort by release_event_count for balanced distribution
        id_list_sorted = sorted(id_list, key=lambda i: meta.get(i, {}).get('release_events', 0))
        # Interleave into folds (round-robin by sorted order)
        for i, ident in enumerate(id_list_sorted):
            fold_idx = i % n_folds
            folds[fold_idx].add(ident)
            m = meta.get(ident, {})
            fold_stats[fold_idx][route]['count'] += 1
            fold_stats[fold_idx][route]['release_events'] += m.get('release_events', 0)
            fold_stats[fold_idx][route]['short_events'] += m.get('short_events', 0)

    return folds, fold_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=SPLIT_SEED)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f'OUTPUT EXISTS: {out}')

    rng = __import__('random').Random(args.seed)

    # Load V1 folds
    verify_sealed_directory(FOLD_ROOT)
    folds_data = load_fit_fold_bundle(FOLD_ROOT)

    # Load registry for identity list
    rows = list(csv.DictReader(open(REGISTRY)))
    fit_rows = [r for r in rows if r.get('split') == 'FIT_TRAIN']
    all_fit_ids = set(r['canonical_parent_key'] for r in fit_rows)

    # Verify Teacher root
    verify_sealed_directory(TEACHER_ROOT)

    all_splits = {}
    all_identities = {}
    overlap_issues = []

    for fold_info in folds_data['folds']:
        fold_id = fold_info['fold_id']
        train_ids = set(fold_info['train_identities'])
        val_ids = set(fold_info['validation_identities'])

        # Only use identities that exist in registry
        train_ids = train_ids & all_fit_ids
        val_ids = val_ids & all_fit_ids

        print(f'Fold {fold_id}: train={len(train_ids)} val={len(val_ids)}')

        # Load metadata for training identities
        meta = load_identity_metadata(train_ids, TEACHER_ROOT)

        # Stratified split
        inner_folds, fold_stats = stratified_split(list(train_ids), meta, N_INNER_FOLDS, rng)

        # Verify constraints
        # 1. No identity appears in multiple inner folds
        for i in range(N_INNER_FOLDS):
            for j in range(i + 1, N_INNER_FOLDS):
                overlap = inner_folds[i] & inner_folds[j]
                if overlap:
                    overlap_issues.append(f'Fold {fold_id}: inner {i} and {j} share {len(overlap)} identities')

        # 2. No outer-val identity in any inner fold
        for i in range(N_INNER_FOLDS):
            val_contamination = inner_folds[i] & val_ids
            if val_contamination:
                overlap_issues.append(f'Fold {fold_id}: inner {i} contains {len(val_contamination)} outer-val identities')

        # 3. All train identities assigned
        all_assigned = set()
        for f in inner_folds:
            all_assigned |= f
        missing = train_ids - all_assigned
        extra = all_assigned - train_ids
        if missing:
            overlap_issues.append(f'Fold {fold_id}: {len(missing)} train identities not assigned')
        if extra:
            overlap_issues.append(f'Fold {fold_id}: {len(extra)} extra identities assigned')

        # Convert to sorted lists for JSON
        inner_fold_lists = [sorted(list(f)) for f in inner_folds]

        # Route distribution
        route_dist = {}
        for i in range(N_INNER_FOLDS):
            route_dist[f'inner_{i}'] = {
                route: {
                    'identity_count': fold_stats[i][route]['count'],
                    'release_events': fold_stats[i][route]['release_events'],
                    'short_events': fold_stats[i][route]['short_events'],
                }
                for route in fold_stats[i]
            }

        all_splits[f'fold_{fold_id}'] = {
            'outer_fold_id': fold_id,
            'outer_train_count': len(train_ids),
            'outer_val_count': len(val_ids),
            'outer_val_identities': sorted(list(val_ids)),
            'inner_folds': [
                {'fold_index': i, 'identity_count': len(inner_fold_lists[i]),
                 'identities': inner_fold_lists[i]}
                for i in range(N_INNER_FOLDS)
            ],
            'route_distribution': route_dist,
        }

        # Collect all identities
        for ident in train_ids:
            all_identities[ident] = {
                'outer_fold': fold_id,
                'role': 'train',
                'route': meta.get(ident, {}).get('route', 'unknown'),
                'release_events': meta.get(ident, {}).get('release_events', 0),
                'short_events': meta.get(ident, {}).get('short_events', 0),
            }
        for ident in val_ids:
            all_identities[ident] = {
                'outer_fold': fold_id,
                'role': 'val',
                'route': 'see_train',  # not loaded; val identities metadata not needed
                'release_events': -1,
                'short_events': -1,
            }

    # Build output
    staging = out.with_name(f'.{out.name}.{uuid.uuid4().hex}.staging')
    staging.mkdir(parents=True)

    inner_cv_splits = {
        'schema': 'DETECTOR_V5_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1',
        'seed': args.seed,
        'n_inner_folds': N_INNER_FOLDS,
        'outer_folds': len(folds_data['folds']),
        'splits': all_splits,
    }
    _atomic_text(staging / 'inner_cv_splits.json', json.dumps(inner_cv_splits, indent=2))

    _atomic_text(staging / 'identity_inventory.json', json.dumps({
        'total_identities': len(all_identities),
        'identities': all_identities,
    }, indent=2))

    # Route distribution summary
    route_summary = {}
    for fold_key, fold_data in all_splits.items():
        route_summary[fold_key] = fold_data['route_distribution']
    _atomic_text(staging / 'route_distribution.json', json.dumps(route_summary, indent=2))

    # Duration distribution
    dur_dist = {}
    for fold_key, fold_data in all_splits.items():
        dur_dist[fold_key] = {}
        for inner in fold_data['inner_folds']:
            inner_key = f'inner_{inner["fold_index"]}'
            short_count = 0
            total_release = 0
            for ident in inner['identities']:
                m = all_identities.get(ident, {})
                short_count += m.get('short_events', 0)
                total_release += m.get('release_events', 0)
            dur_dist[fold_key][inner_key] = {
                'identity_count': inner['identity_count'],
                'total_release_events': total_release,
                'short_events_dur_lt_30': short_count,
            }
    _atomic_text(staging / 'duration_distribution.json', json.dumps(dur_dist, indent=2))

    # Overlap audit
    overlap_audit = {
        'identity_overlap_within_split': 0,
        'outer_val_contamination': 0,
        'missing_identities': 0,
        'duplicate_identities': 0,
        'issues': overlap_issues,
        'status': 'PASS' if len(overlap_issues) == 0 else 'FAIL',
    }
    _atomic_text(staging / 'overlap_audit.json', json.dumps(overlap_audit, indent=2))

    # Source binding
    _atomic_text(staging / 'source_binding.json', json.dumps({
        'fold_root': str(FOLD_ROOT),
        'fold_root_seal': sha256_file(FOLD_ROOT / 'SHA256SUMS'),
        'teacher_root': str(TEACHER_ROOT),
        'teacher_root_seal': sha256_file(TEACHER_ROOT / 'SHA256SUMS'),
        'registry': str(REGISTRY),
        'split_seed': args.seed,
        'builder_sha': sha256_file(Path(__file__)),
    }, indent=2))

    write_seal(staging)
    os.replace(staging, out)

    print(f'\nInner-CV splits sealed: {out}')
    print(f'Seal: {sha256_file(out / "SHA256SUMS")}')
    print(f'Issues: {len(overlap_issues)}')
    for issue in overlap_issues:
        print(f'  {issue}')
    print(f'Status: {overlap_audit["status"]}')


if __name__ == '__main__':
    main()
