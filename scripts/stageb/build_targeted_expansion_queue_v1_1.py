#!/usr/bin/env python3
"""Stage-B v1.1 targeted expansion queue builder.

Designs a balanced expansion queue from the 1198 reachable window candidates,
excluding already-labeled windows.  Applies task diversity, butter cap,
and same-task contrast heuristics.

Usage:
  python scripts/stageb/build_targeted_expansion_queue_v1_1.py \
    --candidates /data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv \
    --master-labels tables/stageb_v1_1_all_pair_labels_aggregated_rc1a_d4a3827.csv \
    --out tables/stageb_v1_1_targeted_expansion_queue_rc1a_d4a3827.csv \
    --plan-out reports/STAGEB_RC1A_D4A3827_TARGETED_EXPANSION_PLAN.md \
    --max-parents 28
"""
import csv, os, sys, argparse
from collections import defaultdict, Counter

# ── Task metadata ──
ALL_TASKS = ['alphabet_soup', 'bbq_sauce', 'butter', 'cream_cheese',
             'milk', 'orange_juice', 'salad_dressing', 'tomato_sauce']

# Tasks that currently lack or have very few certain label types
NEED_CMD = ['alphabet_soup', 'bbq_sauce', 'cream_cheese', 'orange_juice',
            'salad_dressing']  # 0-1 cmd_specific currently
NEED_PHYS = ['cream_cheese', 'orange_juice', 'salad_dressing',
             'bbq_sauce', 'alphabet_soup']  # 0 phys currently
NEED_RAND = ['alphabet_soup', 'cream_cheese', 'bbq_sauce',
             'tomato_sauce', 'orange_juice']  # 0-1 rand currently
NEED_HARD_NEG = ['alphabet_soup', 'bbq_sauce', 'orange_juice',
                 'salad_dressing', 'cream_cheese']  # few confirmed HNs

# Task cap: max 3-4 parent windows per task overall
MAX_PER_TASK = 4
# butter cap: ≤ 20% of total expansion
BUTTER_MAX = 5  # 20% of 25 ≈ 5


def make_key(r):
    return (r['task_key'], r['state_id'], r.get('seed', '0'),
            int(r['window_start']), int(r['window_end']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--master-labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plan-out', required=True)
    ap.add_argument('--max-parents', type=int, default=28)
    args = ap.parse_args()

    # ── Load candidates ──
    all_candidates = []
    with open(args.candidates, 'r', newline='') as f:
        for r in csv.DictReader(f):
            # Filter provenance
            if r.get('trace_version', '') != 'corrected_stageb_v1_1':
                continue
            if r.get('source_snapshot_id', '') != 'f9840cb1':
                continue
            all_candidates.append(r)
    print('Loaded %d provenance-passing candidates' % len(all_candidates))

    # ── Load already-labeled windows ──
    labeled_keys = set()
    labeled_by_task = defaultdict(set)
    with open(args.master_labels, 'r', newline='') as f:
        for r in csv.DictReader(f):
            key = make_key(r)
            labeled_keys.add(key)
            labeled_by_task[r['task_key']].add(key)
    print('Already labeled: %d windows across %d tasks' %
          (len(labeled_keys), len(labeled_by_task)))

    # ── Exclude already-labeled ──
    pool = []
    for c in all_candidates:
        key = make_key(c)
        if key not in labeled_keys:
            pool.append(c)
    print('Available pool after exclusion: %d windows' % len(pool))

    # ── Pool statistics ──
    pool_by_task = defaultdict(list)
    for c in pool:
        pool_by_task[c['task_key']].append(c)
    print('\nPool by task:')
    for tk in ALL_TASKS:
        n = len(pool_by_task[tk])
        high = sum(1 for c in pool_by_task[tk] if c.get('candidate_stratum') == 'high_opportunity')
        med = sum(1 for c in pool_by_task[tk] if c.get('candidate_stratum') == 'medium_opportunity')
        low = sum(1 for c in pool_by_task[tk] if c.get('candidate_stratum') == 'hard_negative_or_idle')
        print('  %-20s %3d  (high=%d med=%d low=%d)' % (tk, n, high, med, low))

    # ── Selection ──
    selected = []  # list of (category, candidate_dict, reason)
    task_counts = Counter()
    used_keys = set(labeled_keys)  # block re-selection

    def select_windows(category, task_list, n_target, heuristic_fn):
        """Select up to n_target windows from task_list using heuristic_fn for ranking."""
        nonlocal selected
        n_selected = 0
        for tk in task_list:
            if n_selected >= n_target:
                break
            if task_counts[tk] >= MAX_PER_TASK:
                continue
            if tk == 'butter' and task_counts.get('butter', 0) >= BUTTER_MAX:
                continue
            candidates_tk = pool_by_task.get(tk, [])
            # Rank by heuristic
            scored = [(c, heuristic_fn(c)) for c in candidates_tk
                      if make_key(c) not in used_keys]
            scored.sort(key=lambda x: x[1], reverse=True)  # higher score = better
            for c, score in scored:
                if n_selected >= n_target:
                    break
                if task_counts[tk] >= MAX_PER_TASK:
                    break
                if tk == 'butter' and task_counts.get('butter', 0) >= BUTTER_MAX:
                    break
                selected.append((category, c,
                                 '%s heuristic=%.3f' % (tk, score)))
                used_keys.add(make_key(c))
                task_counts[tk] += 1
                n_selected += 1
        return n_selected

    # ── Heuristic functions ──
    def cmd_heuristic(c):
        """Prefer closed-gripper windows in medium/high opportunity strata.
        Low clean_open_count means attack can induce OPEN."""
        open_c = int(c.get('clean_open_count', 0))
        stratum = c.get('candidate_stratum', '')
        score = 0.0
        # Prefer closed gripper (0 opens) or very few opens
        if open_c == 0:
            score += 3.0
        elif open_c <= 2:
            score += 1.5
        elif open_c <= 4:
            score += 0.5
        # Prefer high then medium opportunity
        if stratum == 'high_opportunity':
            score += 2.0
        elif stratum == 'medium_opportunity':
            score += 1.0
        # Prefer later windows (grasp phase)
        ws = int(c.get('window_start', 0))
        actual_max = int(c.get('actual_max_step', 299))
        score += 0.5 * (ws / max(actual_max, 1))
        # Diversity: prefer windows farther from already-used
        return score

    def phys_heuristic(c):
        """Prefer windows with meaningful qpos (interaction)."""
        qpos_pre = float(c.get('qpos_abs_sum_pre', 0))
        qpos_mean = float(c.get('qpos_abs_sum_window_mean', 0))
        eef = float(c.get('eef_displacement', 0))
        stratum = c.get('candidate_stratum', '')
        score = 0.0
        # Medium qpos range (0.01-0.05) suggests object interaction
        if 0.01 <= qpos_pre <= 0.05:
            score += 3.0
        elif 0.005 <= qpos_pre < 0.01:
            score += 1.5
        elif qpos_pre > 0.05:
            score += 1.0
        # EEF displacement shows arm movement
        if eef > 0.01:
            score += 2.0
        elif eef > 0.005:
            score += 1.0
        # Prefer high/medium opportunity
        if stratum == 'high_opportunity':
            score += 1.5
        elif stratum == 'medium_opportunity':
            score += 0.8
        return score

    def hard_neg_heuristic(c):
        """Prefer windows with zero clean opens, low qpos, idle stratum."""
        open_c = int(c.get('clean_open_count', 0))
        open_frac = float(c.get('clean_open_frac', 0))
        qpos_pre = float(c.get('qpos_abs_sum_pre', 0))
        stratum = c.get('candidate_stratum', '')
        score = 0.0
        if open_c == 0:
            score += 4.0
        elif open_c <= 2:
            score += 1.0
        if open_frac == 0.0:
            score += 2.0
        if qpos_pre < 0.005:
            score += 2.0
        elif qpos_pre < 0.01:
            score += 1.0
        if stratum == 'hard_negative_or_idle':
            score += 2.0
        elif stratum == 'medium_opportunity':
            score += 0.5
        return score

    def rand_heuristic(c):
        """Prefer windows in intermediate states (some opens, some qpos)."""
        open_c = int(c.get('clean_open_count', 0))
        qpos_pre = float(c.get('qpos_abs_sum_pre', 0))
        stratum = c.get('candidate_stratum', '')
        score = 0.0
        if 1 <= open_c <= 3:
            score += 3.0
        elif open_c >= 4:
            score += 1.5
        if qpos_pre > 0.005:
            score += 2.0
        if stratum in ('medium_opportunity', 'high_opportunity'):
            score += 1.0
        return score

    # ── Execute selection ──
    TARGETS = [
        ('cmd_expansion', NEED_CMD, 9, cmd_heuristic,
         'Non-butter/tomato cmd_specific candidates'),
        ('phys_enrichment', NEED_PHYS, 7, phys_heuristic,
         'Physical bridge enrichment for underrepresented tasks'),
        ('hard_negative', NEED_HARD_NEG, 7, hard_neg_heuristic,
         'Confirmed hard negatives (zero VIS, zero RAND expected)'),
        ('rand_abstain', NEED_RAND, 5, rand_heuristic,
         'Random_sensitive abstain calibration (non-butter tasks)'),
    ]

    total_selected = 0
    plan_sections = []
    for cat, task_list, n_target, hfn, desc in TARGETS:
        n = select_windows(cat, task_list, n_target, hfn)
        total_selected += n
        plan_sections.append('### %s (%d selected, target %d)' % (cat, n, n_target))
        plan_sections.append(desc)
        plan_sections.append('')
        sel = [s for s in selected if s[0] == cat]
        for _, c, reason in sel:
            plan_sections.append('- `%s` s%s seed=%s [%s,%s] %s' %
                                 (c['task_key'], c['state_id'], c.get('seed', '?'),
                                  c['window_start'], c['window_end'], reason))
        plan_sections.append('')

    # ── Sentinel repeats ──
    # Pick 2 known stable positive windows that are NOT butter
    sentinel_plan = []
    print('\nSentinel repeat candidates:')
    with open(args.master_labels, 'r', newline='') as f:
        master = list(csv.DictReader(f))
    # Find stable cmd+phys windows not in butter
    for r in master:
        if r['task_key'] == 'butter':
            continue
        if r['cmd_specific'] == '1' and r['vis_specific_physical'] == '1':
            if r.get('silver_status', '') == 'stable_cmd+phys':
                sentinel_plan.append(r)
                print('  SENTINEL CANDIDATE: %s s%s [%s,%s] %s qpos_pre=%s' %
                      (r['task_key'], r['state_id'], r['window_start'],
                       r['window_end'], r['silver_status'], r.get('qpos_pre', '')))

    # ── Burn-in sentinel: pick up to 3 non-butter stable cmd+phys ──
    sentinel_selected = []
    for r in sentinel_plan[:3]:
        key = make_key(r)
        # Find in original candidates for full metadata
        cand = None
        for c in all_candidates:
            if make_key(c) == key:
                cand = c
                break
        if cand:
            sentinel_selected.append(('sentinel_repeat', cand,
                                      '%s stable_cmd+phys' % r['task_key']))
            used_keys.add(key)
            task_counts[r['task_key']] += 1
            total_selected += 1
    plan_sections.append('### sentinel_repeats (%d selected)' % len(sentinel_selected))
    plan_sections.append('Known stable cmd+phys windows for pipeline health check.')
    for _, c, reason in sentinel_selected:
        plan_sections.append('- `%s` s%s [%s,%s] %s' %
                             (c['task_key'], c['state_id'], c['window_start'],
                              c['window_end'], reason))

    # ── Output queue CSV ──
    all_selected = selected + sentinel_selected
    if not all_selected:
        print('ERROR: 0 windows selected')
        sys.exit(1)

    fieldnames = ['category', 'pair_id', 'task_key', 'state_id', 'seed',
                  'window_start', 'window_end', 'n_window_steps',
                  'actual_max_step', 'candidate_stratum',
                  'clean_open_count', 'clean_open_frac',
                  'raw_gripper_mean', 'raw_gripper_max',
                  'qpos_pre', 'qpos_mean', 'qpos_max', 'qpos_slope',
                  'eef_disp', 'heuristic_reason']

    queue_rows = []
    for cat, c, reason in all_selected:
        pair_id = '%s_%s_s%s_w%s_%s' % (cat, c['task_key'], c['state_id'],
                                         c['window_start'], c['window_end'])
        # Ensure unique by appending seed
        pair_id = '%s_seed%s' % (pair_id, c.get('seed', '0'))
        queue_rows.append({
            'category': cat,
            'pair_id': pair_id,
            'task_key': c['task_key'],
            'state_id': c['state_id'],
            'seed': c.get('seed', '0'),
            'window_start': c['window_start'],
            'window_end': c['window_end'],
            'n_window_steps': c.get('n_window_steps', ''),
            'actual_max_step': c.get('actual_max_step', ''),
            'candidate_stratum': c.get('candidate_stratum', ''),
            'clean_open_count': c.get('clean_open_count', ''),
            'clean_open_frac': c.get('clean_open_frac', ''),
            'raw_gripper_mean': c.get('raw_gripper_mean', ''),
            'raw_gripper_max': c.get('raw_gripper_max', ''),
            'qpos_pre': c.get('qpos_abs_sum_pre', ''),
            'qpos_mean': c.get('qpos_abs_sum_window_mean', ''),
            'qpos_max': c.get('qpos_abs_sum_window_max', ''),
            'qpos_slope': c.get('qpos_abs_sum_slope', ''),
            'eef_disp': c.get('eef_displacement', ''),
            'heuristic_reason': reason,
        })

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(queue_rows)

    # ── Write plan ──
    plan_text = '# Stage-B RC1a d4a3827 Targeted Expansion Plan\n\n'
    plan_text += '**Date**: 2026-06-08\n'
    plan_text += '**Anchor**: d4a3827\n'
    plan_text += '**Total parent windows**: %d\n' % total_selected
    plan_text += '**GPU jobs**: %d (each parent = VIS + RAND)\n\n' % (total_selected * 2)
    plan_text += '## Task distribution\n\n'
    plan_text += '| Task | Selected | cmd | phys | HN | rand | sentinel |\n'
    plan_text += '|------|----------|-----|------|-----|------|----------|\n'
    for tk in ALL_TASKS:
        total_tk = task_counts.get(tk, 0)
        if total_tk > 0:
            cmd_tk = sum(1 for s in all_selected if s[1]['task_key'] == tk and s[0] == 'cmd_expansion')
            phys_tk = sum(1 for s in all_selected if s[1]['task_key'] == tk and s[0] == 'phys_enrichment')
            hn_tk = sum(1 for s in all_selected if s[1]['task_key'] == tk and s[0] == 'hard_negative')
            rand_tk = sum(1 for s in all_selected if s[1]['task_key'] == tk and s[0] == 'rand_abstain')
            sen_tk = sum(1 for s in all_selected if s[1]['task_key'] == tk and s[0] == 'sentinel_repeat')
            plan_text += '| %s | %d | %d | %d | %d | %d | %d |\n' % (
                tk, total_tk, cmd_tk, phys_tk, hn_tk, rand_tk, sen_tk)
    plan_text += '\n## Category breakdown\n\n'
    plan_text += '| Category | Count | Purpose |\n'
    plan_text += '|----------|-------|--------|\n'
    for cat in ['cmd_expansion', 'phys_enrichment', 'hard_negative', 'rand_abstain', 'sentinel_repeat']:
        n = sum(1 for s in all_selected if s[0] == cat)
        purposes = {'cmd_expansion': 'Non-butter/tomato cmd_specific',
                    'phys_enrichment': 'Physical bridge for underrepresented tasks',
                    'hard_negative': 'Confirmed hard negatives',
                    'rand_abstain': 'Random-sensitive abstain calibration',
                    'sentinel_repeat': 'Pipeline health check'}
        plan_text += '| %s | %d | %s |\n' % (cat, n, purposes.get(cat, ''))

    plan_text += '\n## Butter cap check\n'
    butter_n = task_counts.get('butter', 0)
    butter_pct = butter_n / max(total_selected, 1) * 100
    plan_text += 'Butter windows: %d / %d = %.0f%% (cap: %d/%d = 20%%)\n' % (
        butter_n, total_selected, butter_pct, BUTTER_MAX, total_selected)

    plan_text += '\n## Per-task cap check\n'
    plan_text += 'Max per task: %d\n' % MAX_PER_TASK
    for tk in ALL_TASKS:
        n = task_counts.get(tk, 0)
        if n > MAX_PER_TASK:
            plan_text += '  WARNING: %s has %d > cap %d\n' % (tk, n, MAX_PER_TASK)
    plan_text += '  All tasks within cap.\n' if all(
        task_counts.get(tk, 0) <= MAX_PER_TASK for tk in ALL_TASKS
    ) else '  Some tasks exceed cap.\n'

    plan_text += '\n## Constraints checklist\n'
    plan_text += '- [ ] butter ≤ 20%%: %d/%d (%.0f%%)\n' % (butter_n, total_selected, butter_pct)
    plan_text += '- [ ] Each task ≤ %d parents\n' % MAX_PER_TASK
    plan_text += '- [ ] ≤ %d total parents\n' % args.max_parents
    plan_text += '- [ ] Every parent in reachable candidate pool\n'
    plan_text += '- [ ] No overlap with already-labeled windows\n'
    plan_text += '- [ ] random_sensitive → abstain, NOT negative\n'
    plan_text += '- [ ] All provenance: corrected_stageb_v1_1 + f9840cb1\n'
    plan_text += '- [ ] Smoke first (6 parents) before sleep expansion\n'

    plan_text += '\n## Window details\n\n'
    for section in plan_sections:
        plan_text += section + '\n'

    plan_text += '\n## Output\n\n'
    plan_text += 'Queue CSV: `%s`\n' % args.out
    plan_text += 'Next step: 6-parent smoke, then sleep expansion.\n'

    os.makedirs(os.path.dirname(args.plan_out) or '.', exist_ok=True)
    with open(args.plan_out, 'w') as f:
        f.write(plan_text)

    # ── Summary ──
    print('\n=== EXPANSION QUEUE SUMMARY ===')
    print('Total parents: %d (%d jobs)' % (total_selected, total_selected * 2))
    for cat in ['cmd_expansion', 'phys_enrichment', 'hard_negative', 'rand_abstain', 'sentinel_repeat']:
        n = sum(1 for s in all_selected if s[0] == cat)
        print('  %s: %d' % (cat, n))
    print('Task counts:', dict(task_counts))
    print('Butter: %d/%d = %.0f%%' % (butter_n, total_selected, butter_pct))
    print('Output: %s' % args.out)
    print('Plan: %s' % args.plan_out)


if __name__ == '__main__':
    main()
