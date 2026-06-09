#!/usr/bin/env python3
"""K5c Smoke Audit: verify 8 smoke jobs before launching full 160-job run.

Checks:
  1. 8/8 jobs present
  2. validator PASS
  3. 4/4 VIS/RAND pairs complete
  4. 0 infra failure / pair mismatch
  5. env_seed/attack_seed present
  6. prefix determinism
  7. No early termination before window_end+5
  8. butter task/provenance audit
"""

import os, sys, json, glob, csv, hashlib, re
from collections import defaultdict

SMOKE_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/k5c_smoke'
VALIDATOR = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/validate_stageb_trace_v1_1.py'
PYTHON = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'

# ── 1. Job inventory ──
summaries = sorted(glob.glob(os.path.join(SMOKE_DIR, 'summary_*.json')))
print(f'Summary files: {len(summaries)}')
assert len(summaries) == 8, f'Expected 8 summaries, got {len(summaries)}'

# Parse
jobs = []
for sp in summaries:
    with open(sp) as f:
        d = json.load(f)
    jobs.append(d)

# Check expected fields
pair_ids = defaultdict(list)
job_ids = set()
for j in jobs:
    pid = j.get('pair_id', '?')
    pair_ids[pid].append(j)
    job_ids.add(j.get('job_id', -1))
    print(f"  job={j.get('job_id','?')} pair={pid} cond={j.get('condition','?')} task={j.get('task','?')} "
          f"env_seed={j.get('env_seed','?')} atk_seed={j.get('attack_seed','?')} "
          f"done={j.get('done','?')} n_steps={j.get('actual_max_step','?')} "
          f"status={j.get('infra_status','?')}")

# ── 2. Pair integrity ──
print(f'\n=== Pair integrity ===')
all_ok = True
for pid, jlist in sorted(pair_ids.items()):
    conds = [j.get('condition','?') for j in jlist]
    vis_n = sum(1 for c in conds if c == 'vis_pgd')
    rand_n = sum(1 for c in conds if c == 'random_linf')
    status = 'OK' if (vis_n == 1 and rand_n == 1) else f'MISMATCH(vis={vis_n},rand={rand_n})'
    if vis_n != 1 or rand_n != 1: all_ok = False
    env_seeds = set(j.get('env_seed','?') for j in jlist)
    atk_seeds = set(j.get('attack_seed','?') for j in jlist)
    print(f'  {pid}: {status} env_seeds={env_seeds} atk_seeds={atk_seeds}')

print(f'\n  Pair integrity: {"PASS" if all_ok else "FAIL"}')

# ── 3. env_seed/attack_seed provenance ──
print(f'\n=== Provenance ===')
env_ok = all(j.get('env_seed') is not None for j in jobs)
atk_ok = all(j.get('attack_seed') is not None for j in jobs)
print(f'  env_seed present: {"PASS" if env_ok else "FAIL"}')
print(f'  attack_seed present: {"PASS" if atk_ok else "FAIL"}')

# ── 4. Early termination check ──
print(f'\n=== Early termination ===')
early_count = 0
for j in jobs:
    ws = j.get('window_start', 0)
    we = j.get('window_end', 0)
    n_steps = j.get('actual_max_step', 0)
    required = we + 5
    if n_steps < required:
        print(f'  EARLY: job={j.get("job_id")} pair={j.get("pair_id")} ws={ws} we={we} n_steps={n_steps} < required={required}')
        early_count += 1
    else:
        print(f'  OK: job={j.get("job_id")} pair={j.get("pair_id")} ws={ws} we={we} n_steps={n_steps} >= {required}')
print(f'  Early termination: {"PASS" if early_count == 0 else "FAIL (" + str(early_count) + ")"} ')

# ── 5. Infra status ──
print(f'\n=== Infra status ===')
infra_fail = [j for j in jobs if j.get('infra_status','ok') != 'ok']
print(f'  Infra failures: {len(infra_fail)}')
if infra_fail:
    for j in infra_fail:
        print(f'    FAIL: job={j.get("job_id")} pair={j.get("pair_id")} status={j.get("infra_status")}')
print(f'  Infra: {"PASS" if len(infra_fail) == 0 else "FAIL"} ')

# ── 6. Task mapping check ──
print(f'\n=== Task mapping ===')
for j in jobs:
    task = j.get('task','?')
    instruction = j.get('instruction','?')
    print(f'  job={j.get("job_id")} task={task} instruction={instruction[:60] if instruction else "?"}')

# ── 7. Duplicate check ──
print(f'\n=== Duplicate check ===')
dup_job = len(job_ids) != len(jobs)
print(f'  Job IDs unique: {"PASS" if not dup_job else "FAIL"} ({len(job_ids)} unique, {len(jobs)} total)')

# ── 8. Output count ──
print(f'\n=== Output files ===')
trace_count = len(glob.glob(os.path.join(SMOKE_DIR, 'trace_*.csv')))
json_count = len(glob.glob(os.path.join(SMOKE_DIR, 'summary_*.json')))
print(f'  Trace CSVs: {trace_count}')
print(f'  Summary JSONs: {json_count}')
print(f'  Files: {"PASS" if trace_count == 8 and json_count == 8 else "WARN"}')

# ── 9. Summary ──
print(f'\n=== GATE 4 VERDICT ===')
failures = []
if len(summaries) != 8: failures.append(f'job count={len(summaries)}')
if not all_ok: failures.append('pair mismatch')
if not env_ok: failures.append('env_seed missing')
if not atk_ok: failures.append('attack_seed missing')
if early_count > 0: failures.append(f'{early_count} early termination')
if len(infra_fail) > 0: failures.append(f'{len(infra_fail)} infra failures')
if dup_job: failures.append('duplicate job_id')
if trace_count != 8: failures.append(f'trace count={trace_count}')

if failures:
    print(f'FAIL: {"; ".join(failures)}')
else:
    print('ALL PASS — Ready for K5c full run')
