#!/usr/bin/env python3
"""Generate exact replay script for 3 C2O candidates."""
import os
REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
os.chdir(REPO)
src = open('scripts/stageb/run_s20d_v5_token_pgd_one_step_smoke.py').read()
start = src.find('CASES = [')
end = src.find(']\n\n#', start) + 1

c2o_cases = [
    ('chocolate_pudding', 21, 44, 299),
    ('cream_cheese', 35, 80, 99),
    ('cream_cheese', 35, 80, 299),
]
lines = []
for task, sid, ws, seed in c2o_cases:
    cid = '%s_s%d_w%d_%d_c%d_close_streak' % (task, sid, ws-3, ws+3, ws)
    lines.append("    {'candidate_id': '%s', 'task': '%s', 'state_id': %d, "
                "'window_start': %d, 'window_end': %d, 'attack_seed': %d, "
                "'purpose': 'c2o_exact_replay'}," % (cid, task, sid, ws, ws, seed))

new_cases = 'CASES = [\n' + '\n'.join(lines) + '\n]'
new_src = src[:start] + new_cases + src[end:]
new_src = new_src.replace("GPU = '0'", "GPU = '4,5'")
new_src = new_src.replace("RENDER = '0'", "RENDER = '4'")
new_src = new_src.replace(
    'import numpy as np, torch',
    "import os as _os; _os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'\nimport numpy as np, torch")
new_src = new_src.replace(
    's20d_v5_token_pgd_one_step_smoke.csv',
    'c2o_exact_replay_output.csv')

out = 'scripts/stageb/run_c2o_replay.py'
with open(out, 'w') as f:
    f.write(new_src)
print('Generated %s with %d cases' % (out, len(c2o_cases)))
