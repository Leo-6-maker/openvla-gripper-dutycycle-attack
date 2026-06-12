#!/usr/bin/env python3
"""Generate parameterized one-step smoke scripts from validated baseline."""
import csv

TOP30 = '/data/liuyu/outputs/stageb_v5_critical_close_overnight_20260613_0100/tables/phase2_smoke_top30.csv'
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'

rows = list(csv.DictReader(open(TOP30)))

with open(f'{REPO}/scripts/stageb/run_s20d_v5_token_pgd_one_step_smoke.py') as f:
    src = f.read()

# Find CASES block bounds
start = src.find('CASES = [')
end = src.find(']\n\n#', start) + 1  # include the ]

# Generate split scripts
seeds = [99, 199, 299]
splits = [('gpu01', rows[:15], '0,1', '0'), ('gpu45', rows[15:], '4,5', '4')]

for name, split_rows, gpu, render in splits:
    lines = []
    for r in split_rows:
        for seed in seeds:
            lines.append("    {'candidate_id': '%s', 'task': '%s', 'state_id': %s, "
                         "'window_start': %s, 'window_end': %s, 'attack_seed': %s, "
                         "'purpose': 'phase2_smoke'}," % (
                         r['candidate_id'], r['task'], r['state_id'],
                         r['event_center_step'], r['event_center_step'], seed))

    new_cases = 'CASES = [\n' + '\n'.join(lines) + '\n]'
    new_src = src[:start] + new_cases + src[end:]
    new_src = new_src.replace("GPU = '0'", "GPU = '%s'" % gpu)
    new_src = new_src.replace("RENDER = '0'", "RENDER = '%s'" % render)
    # Unique output path per GPU pair
    new_src = new_src.replace(
        "s20d_v5_token_pgd_one_step_smoke.csv",
        "phase2_smoke_%s_output.csv" % name)
    # Set CUDA_VISIBLE_DEVICES BEFORE torch import (critical for GPU isolation)
    new_src = new_src.replace(
        "import numpy as np, torch",
        "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '%s'\nimport numpy as np, torch" % gpu)

    out_path = f'{REPO}/scripts/stageb/run_phase2_smoke_{name}.py'
    with open(out_path, 'w') as f:
        f.write(new_src)
    print('Wrote %s: %d candidates x %d seeds = %d cases' % (
        out_path, len(split_rows), len(seeds), len(lines)))
