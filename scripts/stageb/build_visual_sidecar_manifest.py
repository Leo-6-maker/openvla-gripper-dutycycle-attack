#!/usr/bin/env python3
"""Build visual sidecar manifest: windows whose clean frames we want CLIP/DINOv2 embeddings for.

Includes:
1. All 45 master table windows (already labeled — training/eval for detector)
2. All 27 targeted expansion queue windows (about to be labeled)
3. Smoke windows (subset of expansion)

Output: tables/stageb_v1_1_visual_sidecar_manifest_d4a3827.csv
"""
import csv, os, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master-labels', required=True)
    ap.add_argument('--expansion-queue', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []

    # Master table windows
    with open(args.master_labels, 'r', newline='') as f:
        for r in csv.DictReader(f):
            ws = int(r['window_start'])
            we = int(r['window_end'])
            wc = (ws + we) // 2
            rows.append({
                'source': 'master_labeled',
                'task_key': r['task_key'],
                'state_id': r['state_id'],
                'seed': r['seed'],
                'window_start': str(ws),
                'window_end': str(we),
                'window_center': str(wc),
                'frame_start': str(max(0, ws - 2)),      # 2 steps before window
                'frame_center': str(wc),
                'frame_end': str(min(int(r.get('actual_max_step', '299')), we + 2)),  # 2 steps after
                'label_tier': r.get('label_tier', '?'),
                'cmd_specific': r.get('cmd_specific', '?'),
                'vis_specific_phys': r.get('vis_specific_physical', '?'),
                'random_sensitive': r.get('random_sensitive', '?'),
                'clean_open_count': r.get('clean_open_count', ''),
                'qpos_pre': r.get('qpos_pre', ''),
            })

    # Expansion queue windows (avoid duplicates with master)
    master_keys = set()
    with open(args.master_labels, 'r', newline='') as f:
        for r in csv.DictReader(f):
            master_keys.add((r['task_key'], r['state_id'], r['seed'],
                            r['window_start'], r['window_end']))

    with open(args.expansion_queue, 'r', newline='') as f:
        for r in csv.DictReader(f):
            key = (r['task_key'], r['state_id'], r['seed'],
                   r['window_start'], r['window_end'])
            if key in master_keys:
                continue
            ws = int(r['window_start'])
            we = int(r['window_end'])
            wc = (ws + we) // 2
            max_s = int(r.get('actual_max_step', '299'))
            rows.append({
                'source': 'expansion_queue',
                'task_key': r['task_key'],
                'state_id': r['state_id'],
                'seed': r['seed'],
                'window_start': str(ws),
                'window_end': str(we),
                'window_center': str(wc),
                'frame_start': str(max(0, ws - 2)),
                'frame_center': str(wc),
                'frame_end': str(min(max_s, we + 2)),
                'label_tier': 'pending',
                'cmd_specific': '?',
                'vis_specific_phys': '?',
                'random_sensitive': '?',
                'clean_open_count': r.get('clean_open_count', ''),
                'qpos_pre': r.get('qpos_pre', ''),
            })

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print('Manifest: %d windows (%d labeled + %d pending)' %
          (len(rows),
           sum(1 for r in rows if r['source'] == 'master_labeled'),
           sum(1 for r in rows if r['source'] == 'expansion_queue')))
    print('Output: %s' % args.out)


if __name__ == '__main__':
    main()
