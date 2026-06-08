#!/usr/bin/env python3
"""CPU-only interim audit of completed expansion pairs. No torch, no CUDA.

Output: interim audit CSV + classification summary.
"""
import csv, json, os, sys
from collections import Counter

SMOKE_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827'
EXP_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827_interim'

os.makedirs(OUT_DIR, exist_ok=True)

CMD_THRESHOLD = 6
PHYS_THRESHOLD = 0.01


def classify(vis_open, rand_open, vis_streak, rand_streak, vis_qpos, rand_qpos,
             n_total_steps, category_label):
    """Apply expanded label taxonomy."""
    # Unstable / edge
    if n_total_steps <= 50:
        return 'unstable_or_edge', 'episode too short (steps=%d)' % n_total_steps

    vis_cmd = (vis_open >= CMD_THRESHOLD or vis_streak >= CMD_THRESHOLD)
    rand_cmd = (rand_open >= CMD_THRESHOLD or rand_streak >= CMD_THRESHOLD)
    vis_phys = abs(vis_qpos) >= PHYS_THRESHOLD
    rand_phys = abs(rand_qpos) >= PHYS_THRESHOLD

    if vis_cmd and not rand_cmd:
        cmd_label = 'cmd_specific'
    elif rand_cmd and not vis_cmd:
        cmd_label = 'rand_command_sensitive'
    elif vis_cmd and rand_cmd:
        cmd_label = 'random_command_confounded'
    else:
        cmd_label = 'no_command_effect'

    if vis_phys and not rand_phys:
        phys_label = 'vis_specific_phys'
    elif rand_phys and not vis_phys:
        phys_label = 'rand_phys_confound'
    elif vis_phys and rand_phys:
        phys_label = 'shared_qpos_response'
    else:
        phys_label = 'no_phys_effect'

    # Combined label
    if cmd_label == 'cmd_specific' and phys_label == 'vis_specific_phys':
        combined = 'cmd+phys_specific'
    elif cmd_label == 'cmd_specific':
        combined = 'cmd_specific'
    elif cmd_label == 'rand_command_sensitive':
        combined = 'rand_command_sensitive'
    elif cmd_label == 'random_command_confounded':
        combined = 'random_confounded'
    elif phys_label == 'vis_specific_phys':
        combined = 'vis_specific_phys'
    elif phys_label == 'rand_phys_confound':
        combined = 'rand_phys_confound'
    elif phys_label == 'shared_qpos_response':
        combined = 'shared_qpos'
    elif category_label.startswith('hard_neg'):
        if vis_open < 3 and rand_open < 3 and abs(vis_qpos) < 0.005 and abs(rand_qpos) < 0.005:
            combined = 'hard_negative_confirmed'
        else:
            combined = 'hard_neg_candidate_unconfirmed'
    else:
        combined = 'negative'

    details = 'cmd=%s phys=%s' % (cmd_label, phys_label)
    return combined, details


def load_pairs(summary_dir):
    """Load all VIS/RAND pairs from summary JSONs."""
    pairs = {}
    for f in sorted(os.listdir(summary_dir)):
        if not f.startswith('summary_') or not f.endswith('.json'):
            continue
        with open(os.path.join(summary_dir, f)) as fp:
            d = json.load(fp)
        pid = d.get('pair_id', '?')
        c = 'VIS' if '_vis_pgd_' in f else 'RAND'
        pairs.setdefault(pid, {})[c] = d
        pairs[pid]['task'] = d.get('task_key', '?')
        pairs[pid]['state'] = d.get('state_id', '?')
        pairs[pid]['ws'] = d.get('window_start', 0)
        pairs[pid]['we'] = d.get('window_end', 0)
        pairs[pid]['seed'] = d.get('seed', 0)
    return pairs


def category_from_pair_id(pid):
    if 'cmd_expansion' in pid: return 'cmd_expansion'
    if 'phys_enrichment' in pid: return 'phys_enrichment'
    if 'hard_neg_candidate' in pid: return 'hard_neg_candidate'
    if 'rand_abstain' in pid: return 'rand_abstain'
    if 'sentinel' in pid: return 'sentinel_repeat'
    return 'unknown'


# ── Load smoke + expansion ──
smoke_pairs = load_pairs(SMOKE_DIR)
exp_pairs = load_pairs(EXP_DIR)
all_pairs = {}
all_pairs.update(smoke_pairs)
all_pairs.update(exp_pairs)

print('Smoke: %d pairs  Expansion: %d pairs (completed)' %
      (len(smoke_pairs), len(exp_pairs)))

# ── Classify each pair ──
rows = []
for pid in sorted(all_pairs.keys()):
    p = all_pairs[pid]
    v = p.get('VIS', {}); r = p.get('RAND', {})
    if not v or not r:
        continue  # incomplete

    vo = v.get('decoded_open_count', 0); vs = v.get('decoded_longest_open_streak', 0)
    ro = r.get('decoded_open_count', 0); rs = r.get('decoded_longest_open_streak', 0)
    vq = v.get('qpos_delta', 0); rq = r.get('qpos_delta', 0)
    v_steps = v.get('n_total_steps', 299)
    cat = category_from_pair_id(pid)
    source = 'smoke' if 'smoke_' in pid else 'expansion'

    combined, details = classify(vo, ro, vs, rs, vq, rq, v_steps, cat)

    rows.append({
        'source': source,
        'pair_id': pid,
        'task_key': p['task'],
        'state_id': str(p['state']),
        'seed': str(p['seed']),
        'window_start': str(p['ws']),
        'window_end': str(p['we']),
        'category_label': cat,
        'vis_open_count': str(vo),
        'vis_streak': str(vs),
        'vis_qpos_delta': str(round(vq, 6)),
        'rand_open_count': str(ro),
        'rand_streak': str(rs),
        'rand_qpos_delta': str(round(rq, 6)),
        'vis_total_steps': str(v_steps),
        'rand_total_steps': str(r.get('n_total_steps', 299)),
        'interim_label': combined,
        'label_details': details,
    })

# ── Write audit CSV ──
if rows:
    with open(os.path.join(OUT_DIR, 'interim_pair_audit.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ── Summary ──
label_counts = Counter(r['interim_label'] for r in rows)
source_counts = Counter(r['source'] for r in rows)
cat_label = Counter()
for r in rows:
    cat_label['%s→%s' % (r['category_label'], r['interim_label'])] += 1

print('\n=== INTERIM AUDIT: %d pairs ===' % len(rows))
print('Sources: smoke=%d expansion=%d' % (source_counts.get('smoke', 0),
                                           source_counts.get('expansion', 0)))
print('\nLabel distribution:')
for label, n in label_counts.most_common():
    print('  %-30s %d' % (label, n))

print('\nCategory → Interim label:')
for cl, n in sorted(cat_label.items()):
    print('  %-50s %d' % (cl, n))

print('\nBy task:')
task_labels = {}
for r in rows:
    tk = r['task_key']
    task_labels.setdefault(tk, Counter())[r['interim_label']] += 1
for tk in sorted(task_labels):
    counts = ' '.join('%s=%d' % (l, c) for l, c in task_labels[tk].most_common())
    print('  %-20s %s' % (tk, counts))

# Hard_neg conversion
hn_rows = [r for r in rows if r['category_label'] == 'hard_neg_candidate']
if hn_rows:
    hn_conv = Counter(r['interim_label'] for r in hn_rows)
    print('\nHard neg candidate conversion:')
    for l, c in hn_conv.most_common():
        print('  %-30s %d' % (l, c))

print('\nOutput: %s/interim_pair_audit.csv' % OUT_DIR)
print('  (CPU-only, no GPU, no torch)')
