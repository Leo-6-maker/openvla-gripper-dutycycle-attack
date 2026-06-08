#!/usr/bin/env python3
"""Stage-B v1.1 master label aggregator — from detector feature tables + silver stability.

Produces one row per unique parent window (task_key, state_id, seed, window_start, window_end).
Merges labels from bronze / silver_override / rescue_override tiers.
Adds provenance columns (all RC1a constants, spot-checked against traces).

Usage:
  python scripts/stageb/build_master_label_table_v1_1.py \
    --bronze-features /path/to/fixed_bronze_feature_table.csv \
    --silver-features /path/to/fixed_silver_override_feature_table.csv \
    --rescue-features /path/to/fixed_rescue_override_feature_table.csv \
    --silver-stability /path/to/silver_p1a_stability_by_parent.csv \
    --out tables/stageb_v1_1_all_pair_labels_aggregated_rc1a_d4a3827.csv
"""
import csv, os, sys, argparse
from collections import defaultdict

# ── RC1a provenance constants (verified 2026-06-08 against trace samples) ──
PROVENANCE = {
    'trace_version': 'corrected_stageb_v1_1',
    'source_snapshot_id': 'f9840cb1',
    'prompt_style': 'official_in_out',
    'image_preprocess_style': 'official_rot180_only',
    'qpos_source': 'obs_robot0_gripper_qpos',
    'open_convention': 'env_action_6_lt_neg_0p5_means_OPEN',
}

# ── Hard filters ──
REQUIRED_FIELDS = [
    'pair_id', 'task_key', 'state_id', 'seed',
    'window_start', 'window_end',
    'target_cmd_any', 'target_cmd_specific', 'target_phys', 'target_rand',
    'label_tier',
]

# ── Target explanation ──
# target_cmd_any in feature table = cmd_susceptible from pair labels,
#   which already excludes random-confounded (vis_meets AND NOT rand_meets).
#   So it IS cmd_specific, not raw cmd_any.
# target_phys = vis_specific_physical_response (phys_sensitive AND rand_qpos < 0.01)
# target_rand = random_confounded (rand meets open threshold)


def make_key(row):
    return (row['task_key'], row['state_id'], row['seed'],
            int(row['window_start']), int(row['window_end']))


def load_feature_table(path, label):
    """Load a detector feature table, return dict keyed by unique window key."""
    rows = {}
    with open(path, 'r', newline='') as f:
        for r in csv.DictReader(f):
            for fld in REQUIRED_FIELDS:
                if fld not in r:
                    print('HARD_FAIL_MISSING_FIELD: %s missing %s in %s' % (path, fld, label))
                    sys.exit(1)
            key = make_key(r)
            if key in rows:
                print('WARNING: duplicate key %s in %s (%s vs %s)' %
                      (key, label, rows[key].get('pair_id', '?'), r.get('pair_id', '?')))
                continue
            rows[key] = dict(r)
            rows[key]['source_feature_table'] = label
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bronze-features', required=True)
    ap.add_argument('--silver-features', required=True)
    ap.add_argument('--rescue-features', required=True)
    ap.add_argument('--silver-stability', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # ── Load feature tables ──
    bronze = load_feature_table(args.bronze_features, 'bronze')
    silver = load_feature_table(args.silver_features, 'silver_override')
    rescue = load_feature_table(args.rescue_features, 'rescue_override')

    print('Loaded: bronze=%d silver=%d rescue=%d' % (len(bronze), len(silver), len(rescue)))

    # ── Load silver stability ──
    stability = {}
    with open(args.silver_stability, 'r', newline='') as f:
        for r in csv.DictReader(f):
            pid = r['parent_pair_id']
            stability[pid] = {
                'n_silver_repeats': int(r.get('n_silver_repeats', 0)),
                'silver_status': r.get('silver_status', '?'),
                'vis_cmd_rate': float(r.get('vis_cmd_rate', 0)),
                'rand_cmd_rate': float(r.get('rand_cmd_rate', 0)),
                'vis_phys_rate': float(r.get('vis_phys_rate', 0)),
                'bronze_cmd': int(r.get('bronze_cmd', -1)),
                'bronze_phys': int(r.get('bronze_phys', -1)),
                'bronze_rand': int(r.get('bronze_rand', -1)),
            }
    print('Silver stability entries: %d' % len(stability))

    # ── Merge: start with bronze, override with silver/rescue ──
    # All 3 tables use same key (task_key, state_id, seed, ws, we)
    all_keys = set(bronze.keys()) | set(silver.keys()) | set(rescue.keys())
    print('Union keys: %d (bronze-only=%d silver-only=%d rescue-only=%d)' %
          (len(all_keys),
           len(all_keys - set(silver.keys()) - set(rescue.keys())),
           len(set(silver.keys()) - set(bronze.keys())),
           len(set(rescue.keys()) - set(bronze.keys()))))

    master_rows = []
    tier_counts = defaultdict(int)
    for key in sorted(all_keys):
        # Priority: rescue > silver > bronze, but only if the row is a REAL override
        # (rescue table includes all 45 bronze keys with bronze_only as filler)
        src = None; src_batch = 'unknown'
        if key in rescue and rescue[key].get('label_tier', '') not in ('bronze_only', 'bronze'):
            src = rescue[key]; src_batch = 'rescue'
        elif key in silver and silver[key].get('label_tier', '') not in ('bronze_only', 'bronze'):
            src = silver[key]; src_batch = 'silver'
        elif key in bronze:
            src = bronze[key]; src_batch = 'bronze'
        if src is None:
            continue

        pair_id = src['pair_id']
        task_key_str = src['task_key']

        # Stability data
        stab = stability.get(pair_id, {})

        # Derive label flags
        cmd_specific = int(src.get('target_cmd_any', '0'))
        # Note: target_cmd_any in feature table = cmd_susceptible from pair labels,
        # which already excludes random_confounded. So cmd_specific = cmd_any here.
        random_sensitive = int(src.get('target_rand', '0'))
        vis_specific_phys = int(src.get('target_phys', '0'))

        # cmd_any_raw and phys_any_raw are NOT in current pair labels (only
        # the random-excluded versions are). Mark as -1 (unavailable).
        cmd_any_raw = -1
        phys_any_raw = -1

        label_tier = src.get('label_tier', '?')
        tier_counts[label_tier] += 1

        row = {
            'parent_id': pair_id,
            'pair_id': pair_id,
            'task_key': task_key_str,
            'state_id': src['state_id'],
            'seed': src['seed'],
            'window_start': src['window_start'],
            'window_end': src['window_end'],
            'source_batch': src_batch,
            # ── Labels ──
            'cmd_any_raw': str(cmd_any_raw),          # NOT AVAILABLE in current data
            'cmd_specific': str(cmd_specific),
            'random_sensitive': str(random_sensitive),
            'phys_any_raw': str(phys_any_raw),          # NOT AVAILABLE in current data
            'vis_specific_physical': str(vis_specific_phys),
            'label_tier': label_tier,
            # ── Silver stability ──
            'silver_status': stab.get('silver_status', ''),
            'n_silver_repeats': str(stab.get('n_silver_repeats', 0)),
            'vis_cmd_rate': str(stab.get('vis_cmd_rate', '')),
            'rand_cmd_rate': str(stab.get('rand_cmd_rate', '')),
            'vis_phys_rate': str(stab.get('vis_phys_rate', '')),
            'bronze_cmd_orig': str(stab.get('bronze_cmd', '')),
            'bronze_phys_orig': str(stab.get('bronze_phys', '')),
            'bronze_rand_orig': str(stab.get('bronze_rand', '')),
            # ── Clean features ──
            'clean_open_count': src.get('clean_open_count', ''),
            'clean_open_frac': src.get('clean_open_frac', ''),
            'raw_gripper_mean': src.get('raw_gripper_mean', ''),
            'raw_gripper_max': src.get('raw_gripper_max', ''),
            'qpos_pre': src.get('qpos_pre', ''),
            'qpos_mean': src.get('qpos_mean', ''),
            'qpos_max': src.get('qpos_max', ''),
            'qpos_slope': src.get('qpos_slope', ''),
            'eef_disp': src.get('eef_disp', ''),
            'stratum': src.get('stratum', ''),
            'actual_max_step': src.get('actual_max_step', ''),
            # ── Provenance ──
            'trace_version': PROVENANCE['trace_version'],
            'source_snapshot_id': PROVENANCE['source_snapshot_id'],
            'prompt_style': PROVENANCE['prompt_style'],
            'image_preprocess_style': PROVENANCE['image_preprocess_style'],
            'qpos_source': PROVENANCE['qpos_source'],
            'open_convention': PROVENANCE['open_convention'],
            'freeze_sha': 'd4a3827',
            'source_feature_table': src.get('source_feature_table', src_batch),
        }
        master_rows.append(row)

    # ── Write ──
    if not master_rows:
        print('ERROR: 0 master rows')
        sys.exit(1)

    fieldnames = list(master_rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(master_rows)

    # ── Audit summary ──
    print('\n=== MASTER TABLE AUDIT ===')
    print('Total parent windows: %d' % len(master_rows))
    print('\nBy label_tier:')
    for tier, n in sorted(tier_counts.items()):
        print('  %s: %d' % (tier, n))

    print('\nBy task_key:')
    task_counts = Counter(r['task_key'] for r in master_rows)
    for tk, n in task_counts.most_common():
        cmd = sum(1 for r in master_rows if r['task_key'] == tk and r['cmd_specific'] == '1')
        phys = sum(1 for r in master_rows if r['task_key'] == tk and r['vis_specific_physical'] == '1')
        rand = sum(1 for r in master_rows if r['task_key'] == tk and r['random_sensitive'] == '1')
        print('  %-20s total=%2d  cmd=%2d  phys=%2d  rand=%2d' % (tk, n, cmd, phys, rand))

    print('\nLabel summary:')
    print('  cmd_specific:        %d' % sum(1 for r in master_rows if r['cmd_specific'] == '1'))
    print('  vis_specific_phys:   %d' % sum(1 for r in master_rows if r['vis_specific_physical'] == '1'))
    print('  random_sensitive:    %d' % sum(1 for r in master_rows if r['random_sensitive'] == '1'))
    print('  With silver status:  %d' % sum(1 for r in master_rows if r['silver_status']))
    print('  Stable (non-unstable) silver: %d' %
          sum(1 for r in master_rows if r['silver_status']
              and r['silver_status'] not in ('', 'unstable')))

    # Gap report
    print('\n=== DATA GAPS ===')
    print('  cmd_any_raw set to -1: raw cmd_any (including rand-confounded) not in current pair labels')
    print('  phys_any_raw set to -1: raw phys_any (including rand-confounded) not in current pair labels')
    print('  P1b windows not tracked separately: P1b contributed 18 pairs merged into silver feature table')

    print('\nOutput: %s' % args.out)


if __name__ == '__main__':
    from collections import Counter
    main()
