#!/usr/bin/env python3
"""CQFR Label Validator & Aggregator V3.5 — complete."""
import os, json, csv, math, sys
import numpy as np
from collections import defaultdict

LEGAL_TO = {'success', 'failure', 'ambiguous', 'video_invalid'}
LEGAL_CQ = {'yes', 'no', 'ambiguous'}
LEGAL_YN = {'yes', 'no', 'ambiguous', 'not_applicable'}
LEGAL_CAUSE = {'gripper', 'arm', 'mixed', 'other', 'unclear', 'not_applicable'}
LEGAL_CONF = {'high', 'medium', 'low'}
REQUIRED_FIELDS = ['task_outcome', 'contact_quality_failure',
                   'task_outcome_confidence', 'contact_quality_confidence']
SUBTYPE_FIELDS = ['premature_release', 'drop_after_lift', 'unstable_transport',
                  'uncontrolled_final_drop']
PLACEMENT_FIELD = 'controlled_placement'
CAUSE_FIELD = 'primary_contact_failure_cause'

def validate_labels(path, role):
    """Full validator: all fields checked, blanks rejected."""
    errors = []; warnings = []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return rows, ['empty file'], 0, 0

    n_filled = 0; n_blank = 0
    for i, row in enumerate(rows):
        bid = row.get('blind_id', f'row_{i}')
        # Check if all required fields are filled
        filled = all(row.get(f, '').strip() != '' for f in REQUIRED_FIELDS)
        all_blank = all(row.get(f, '').strip() == '' for f in REQUIRED_FIELDS)
        if all_blank:
            n_blank += 1; continue
        if not filled:
            errors.append(f'{bid}: incomplete — required fields must all be filled')
            continue
        n_filled += 1

        to = row.get('task_outcome', '').strip()
        cq = row.get('contact_quality_failure', '').strip()
        tc = row.get('task_outcome_confidence', '').strip()
        cc = row.get('contact_quality_confidence', '').strip()

        if to not in LEGAL_TO: errors.append(f'{bid}: illegal task_outcome={to}')
        if cq not in LEGAL_CQ: errors.append(f'{bid}: illegal contact_quality_failure={cq}')
        if tc not in LEGAL_CONF: errors.append(f'{bid}: illegal task_outcome_confidence={tc}')
        if cc not in LEGAL_CONF: errors.append(f'{bid}: illegal contact_quality_confidence={cc}')

        if to == 'video_invalid' and cq not in ('', 'ambiguous'):
            errors.append(f'{bid}: video_invalid with CQ={cq}')

        # Check subtypes
        for sf in SUBTYPE_FIELDS:
            v = row.get(sf, '').strip()
            if v and v not in LEGAL_YN:
                errors.append(f'{bid}: illegal {sf}={v}')

        # Check placement
        v = row.get(PLACEMENT_FIELD, '').strip()
        if v and v not in LEGAL_YN:
            errors.append(f'{bid}: illegal {PLACEMENT_FIELD}={v}')

        # Check cause
        cause = row.get(CAUSE_FIELD, '').strip()
        if cause and cause not in LEGAL_CAUSE:
            errors.append(f'{bid}: illegal {CAUSE_FIELD}={cause}')

        # CQ=no with subtypes/cause
        if cq == 'no':
            for sf in SUBTYPE_FIELDS:
                v = row.get(sf, '').strip()
                if v == 'yes':
                    errors.append(f'{bid}: CQ=no but {sf}=yes')
            if cause not in ('', 'not_applicable'):
                errors.append(f'{bid}: CQ=no but cause={cause}')

        # CQ=yes should have at least one subtype or cause
        if cq == 'yes':
            has_subtype = any(row.get(sf, '').strip() == 'yes' for sf in SUBTYPE_FIELDS)
            has_cause = cause not in ('', 'not_applicable')
            if not has_subtype and not has_cause:
                warnings.append(f'{bid}: CQ=yes but no subtype and no cause')

    print(f'{role}: {len(rows)} rows, {n_filled} filled, {n_blank} blank, {len(errors)} errors, {len(warnings)} warnings')
    for e in errors[:15]: print(f'  ERROR: {e}')
    for w in warnings[:5]: print(f'  WARN: {w}')
    return rows, errors, n_filled, n_blank

def cohens_kappa(table):
    """Cohen's kappa from 2x2 confusion matrix [[a,b],[c,d]]."""
    a, b = table[0]; c, d = table[1]
    n = a + b + c + d
    if n == 0: return 0.0
    p_o = (a + d) / n
    p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if p_e == 1.0: return 1.0
    return (p_o - p_e) / (1.0 - p_e)

def gwets_ac1(table):
    """Gwet's AC1 from 2x2 confusion matrix."""
    a, b = table[0]; c, d = table[1]
    n = a + b + c + d
    if n == 0: return 0.0
    p_o = (a + d) / n
    # Marginal proportions
    p1_pos = (a + b) / n; p1_neg = (c + d) / n
    p2_pos = (a + c) / n; p2_neg = (b + d) / n
    pi_pos = (p1_pos + p2_pos) / 2
    pi_neg = (p1_neg + p2_neg) / 2
    p_e = pi_pos * (1 - pi_pos) + pi_neg * (1 - pi_neg)
    p_e = 2 * p_e  # Gwet's formula
    if p_e == 1.0: return 1.0
    return (p_o - p_e) / (1.0 - p_e)

def cluster_bootstrap(video_hashes, cond_runs, metric_fn, n_iter=10000, seed=42):
    """Cluster bootstrap by video hash (preserving multiplicity)."""
    hashes = list(video_hashes)
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_iter):
        sampled_hashes = rng.choice(hashes, size=len(hashes), replace=True)
        # Collect ALL runs for each sampled hash (preserving multiplicity)
        bs_runs = []
        for h in sampled_hashes:
            bs_runs.extend([r for r in cond_runs if r['unique_video_sha256'] == h])
        v = metric_fn(bs_runs)
        if v is not None:
            values.append(v)
    if not values: return 0.0, 0.0, 0.0
    return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)

def aggregate(r1_rows, r2_rows, run_mapping, blind_key, output_dir):
    """Full aggregation with adjudication, all metrics, cluster bootstrap."""
    os.makedirs(output_dir, exist_ok=True)

    r1_map = {r['blind_id']: r for r in r1_rows}
    r2_map = {r['blind_id']: r for r in r2_rows}
    overlap = sorted(set(r1_map.keys()) & set(r2_map.keys()))
    r1_only = sorted(set(r1_map.keys()) - overlap)

    if not overlap:
        print('FATAL: No overlap between reviewers')
        sys.exit(1)

    # ===== Adjudication =====
    adjudicated = {}; disagreements = []
    for bid in overlap:
        r1_cq = r1_map[bid].get('contact_quality_failure', '').strip()
        r2_cq = r2_map[bid].get('contact_quality_failure', '').strip()
        if r1_cq == r2_cq:
            adjudicated[bid] = r1_cq
        else:
            disagreements.append({'blind_id': bid, 'reviewer1_cq': r1_cq, 'reviewer2_cq': r2_cq, 'adjudicated_cq': 'PENDING'})
    for bid in r1_only:
        adjudicated[bid] = r1_map[bid].get('contact_quality_failure', '').strip()

    # Write adjudication CSV
    adj_path = os.path.join(output_dir, 'CQFR_ADJUDICATION.csv')
    if disagreements:
        with open(adj_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['blind_id', 'reviewer1_cq', 'reviewer2_cq', 'adjudicated_cq'])
            w.writeheader(); w.writerows(disagreements)
        print(f'FATAL: {len(disagreements)} disagreements require third-person adjudication')
        print(f'  See: {adj_path}')
        print(f'  Fill adjudicated_cq column, then re-run.')
        sys.exit(1)
    else:
        with open(adj_path, 'w', newline='') as f:
            f.write('No disagreements.\n')
        print('Adjudication: 0 disagreements')

    # ===== Agreement on overlap =====
    # Build 2x2 table for CQ (yes vs no, excluding ambiguous)
    r1_cq_list = []; r2_cq_list = []
    for bid in overlap:
        r1c = r1_map[bid].get('contact_quality_failure', '').strip()
        r2c = r2_map[bid].get('contact_quality_failure', '').strip()
        if r1c in ('yes', 'no') and r2c in ('yes', 'no'):
            r1_cq_list.append(r1c); r2_cq_list.append(r2c)
    a = sum(1 for x, y in zip(r1_cq_list, r2_cq_list) if x == 'yes' and y == 'yes')
    b = sum(1 for x, y in zip(r1_cq_list, r2_cq_list) if x == 'yes' and y == 'no')
    c = sum(1 for x, y in zip(r1_cq_list, r2_cq_list) if x == 'no' and y == 'yes')
    d = sum(1 for x, y in zip(r1_cq_list, r2_cq_list) if x == 'no' and y == 'no')
    n_pair = a + b + c + d
    raw_agree = (a + d) / n_pair if n_pair else 1.0
    kappa_val = cohens_kappa([[a, b], [c, d]]) if n_pair else 0.0
    ac1_val = gwets_ac1([[a, b], [c, d]]) if n_pair else 0.0
    pos_agree = 2*a / (2*a + b + c) if (2*a + b + c) else 0.0
    neg_agree = 2*d / (2*d + b + c) if (2*d + b + c) else 0.0
    print(f'Agreement on {n_pair} CQ-labeled pairs: raw={raw_agree:.3f}, kappa={kappa_val:.3f}, AC1={ac1_val:.3f}')
    print(f'  Positive agree={pos_agree:.3f}, Negative agree={neg_agree:.3f}')

    # ===== Map to 108 rows =====
    mapping_rows = []
    for rm in run_mapping:
        bid = rm['unique_blind_id']
        cq_label = adjudicated.get(bid, 'ambiguous')
        to_label = r1_map.get(bid, {}).get('task_outcome', '').strip()
        mapping_rows.append({**rm, 'cq_label': cq_label, 'task_outcome': to_label})

    # ===== Per-condition metrics =====
    conditions = sorted(set(rm['condition'] for rm in run_mapping))
    cond_summary = []

    print('\n=== PER-CONDITION (conditional denominator) ===')
    for cond in conditions:
        cr = [r for r in mapping_rows if r['condition'] == cond]
        valid_cq = [r for r in cr if r['cq_label'] in ('yes', 'no')]
        valid_to = [r for r in cr if r['task_outcome'] in ('success', 'failure')]
        n_itt = len(cr); n_cq = len(valid_cq); n_to = len(valid_to)
        ambig = sum(1 for r in cr if r['cq_label'] == 'ambiguous')
        invid = sum(1 for r in cr if r['task_outcome'] == 'video_invalid')

        # CQFR
        cqfr_yes = sum(1 for r in valid_cq if r['cq_label'] == 'yes')
        cqfr_val = cqfr_yes / n_cq if n_cq else 0

        # CQSR
        cqsr_n = sum(1 for r in valid_cq if r['simulator_task_success'] == 'True' and r['cq_label'] == 'no')
        cqsr_val = cqsr_n / n_cq if n_cq else 0

        # SR-CQ mismatch
        mismatch_n = sum(1 for r in valid_cq if r['simulator_task_success'] == 'True' and r['cq_label'] == 'yes')
        mismatch_val = mismatch_n / n_cq if n_cq else 0

        # Human-simulator disagreement
        hs_n = sum(1 for r in valid_to if r['task_outcome'] != r['simulator_task_success'])
        hs_val = hs_n / n_to if n_to else 0

        # Cluster bootstrap CIs
        video_hashes = list(set(r['unique_video_sha256'] for r in cr))
        _, cqfr_lo, cqfr_hi = cluster_bootstrap(video_hashes, cr,
            lambda runs: (lambda vc=[r for r in runs if r['cq_label'] in ('yes','no')]:
                sum(1 for r in vc if r['cq_label']=='yes')/len(vc) if vc else None)())
        _, cqsr_lo, cqsr_hi = cluster_bootstrap(video_hashes, cr,
            lambda runs: (lambda vc=[r for r in runs if r['cq_label'] in ('yes','no')]:
                sum(1 for r in vc if r['simulator_task_success']=='True' and r['cq_label']=='no')/len(vc) if vc else None)())
        _, mis_lo, mis_hi = cluster_bootstrap(video_hashes, cr,
            lambda runs: (lambda vc=[r for r in runs if r['cq_label'] in ('yes','no')]:
                sum(1 for r in vc if r['simulator_task_success']=='True' and r['cq_label']=='yes')/len(vc) if vc else None)())

        n_videos = len(video_hashes)
        print(f'{cond}:')
        print(f'  CQFR={cqfr_yes}/{n_cq}={cqfr_val:.3f} [{cqfr_lo:.3f},{cqfr_hi:.3f}]')
        print(f'  CQSR={cqsr_val:.3f} [{cqsr_lo:.3f},{cqsr_hi:.3f}]')
        print(f'  SR-CQ mismatch={mismatch_val:.3f} [{mis_lo:.3f},{mis_hi:.3f}]')
        print(f'  human-sim disagree={hs_val:.3f}')
        print(f'  ITT N={n_itt}, valid CQ={n_cq}, ambiguous={ambig}, video_invalid={invid}, unique videos={n_videos}')

        cond_summary.append({
            'condition': cond, 'ITT_N': n_itt, 'valid_CQ_N': n_cq,
            'valid_TO_N': n_to, 'ambiguous_CQ': ambig, 'video_invalid': invid,
            'unique_videos': n_videos,
            'CQFR': round(cqfr_val, 4), 'CQFR_ci_lo': round(cqfr_lo, 4), 'CQFR_ci_hi': round(cqfr_hi, 4),
            'CQSR': round(cqsr_val, 4), 'CQSR_ci_lo': round(cqsr_lo, 4), 'CQSR_ci_hi': round(cqsr_hi, 4),
            'SR_CQ_mismatch': round(mismatch_val, 4), 'mismatch_ci_lo': round(mis_lo, 4), 'mismatch_ci_hi': round(mis_hi, 4),
            'human_sim_disagree': round(hs_val, 4),
        })

    # Pooled
    valid_all = [r for r in mapping_rows if r['cq_label'] in ('yes', 'no')]
    n_yes_all = sum(1 for r in valid_all if r['cq_label'] == 'yes')
    ambig_all = sum(1 for r in mapping_rows if r['cq_label'] == 'ambiguous')
    invid_all = sum(1 for r in mapping_rows if r['task_outcome'] == 'video_invalid')
    print(f'\nPooled: CQFR={n_yes_all}/{len(valid_all)}, ambiguous={ambig_all}/{len(mapping_rows)}, invalid={invid_all}/{len(mapping_rows)}')

    # ITT versions
    for cond in conditions:
        cr = [r for r in mapping_rows if r['condition'] == cond]
        valid_cq = [r for r in cr if r['cq_label'] in ('yes', 'no')]
        n_itt = len(cr)
        # ITT: treat ambiguous/invalid as NOT a CQ failure
        itt_yes = sum(1 for r in valid_cq if r['cq_label'] == 'yes')
        itt_cqfr = itt_yes / n_itt if n_itt else 0
        itt_cqsr = sum(1 for r in valid_cq if r['simulator_task_success'] == 'True' and r['cq_label'] == 'no') / n_itt if n_itt else 0
        itt_mismatch = sum(1 for r in valid_cq if r['simulator_task_success'] == 'True' and r['cq_label'] == 'yes') / n_itt if n_itt else 0
        for cs in cond_summary:
            if cs['condition'] == cond:
                cs['ITT_CQFR'] = round(itt_cqfr, 4)
                cs['ITT_CQSR'] = round(itt_cqsr, 4)
                cs['ITT_mismatch'] = round(itt_mismatch, 4)

    # Write condition summary CSV
    cs_path = os.path.join(output_dir, 'CQFR_CONDITION_SUMMARY.csv')
    with open(cs_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cond_summary[0].keys()))
        w.writeheader(); w.writerows(cond_summary)

    # Write audit JSON
    audit = {
        'n_runs': len(mapping_rows),
        'n_unique_videos': len(set(r['unique_video_sha256'] for r in mapping_rows)),
        'n_overlap': len(overlap),
        'n_disagreements': len(disagreements),
        'agreement_raw': round(raw_agree, 4),
        'cohens_kappa': round(kappa_val, 4),
        'gwets_ac1': round(ac1_val, 4),
        'positive_agreement': round(pos_agree, 4),
        'negative_agreement': round(neg_agree, 4),
        'pooled_ambiguous': ambig_all,
        'pooled_video_invalid': invid_all,
        'pooled_CQFR': round(n_yes_all / len(valid_all), 4) if valid_all else 0,
        'condition_summary': cond_summary,
    }
    with open(os.path.join(output_dir, 'CQFR_AGGREGATION_AUDIT.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    # Write 108-row mapped output
    map_path = os.path.join(output_dir, 'CQFR_108_ROW_AGGREGATED.csv')
    with open(map_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        w.writeheader(); w.writerows(mapping_rows)

    print(f'\nOutputs: {cs_path}, {map_path}, {os.path.join(output_dir, "CQFR_AGGREGATION_AUDIT.json")}')
    return mapping_rows

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--r1', required=True)
    ap.add_argument('--r2', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--output-dir', default='/tmp/cqfr_aggregated')
    args = ap.parse_args()

    with open(args.key) as f: blind_key = list(csv.DictReader(f))
    with open(args.mapping) as f: run_mapping = list(csv.DictReader(f))

    r1_rows, e1, f1, b1 = validate_labels(args.r1, 'Reviewer 1')
    r2_rows, e2, f2, b2 = validate_labels(args.r2, 'Reviewer 2')

    if b1 > 0:
        print(f'FATAL: Reviewer 1 has {b1} blank rows. All rows must be filled.')
        sys.exit(1)
    if b2 > 0:
        print(f'FATAL: Reviewer 2 has {b2} blank rows. All rows must be filled.')
        sys.exit(1)
    if e1 or e2:
        print(f'FATAL: {len(e1)+len(e2)} validation errors. Fix before aggregating.')
        sys.exit(1)

    aggregate(r1_rows, r2_rows, run_mapping, blind_key, args.output_dir)
