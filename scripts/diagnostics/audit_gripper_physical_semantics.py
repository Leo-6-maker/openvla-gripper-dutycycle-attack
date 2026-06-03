#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal LIBERO env smoke to confirm gripper physical OPEN/CLOSE semantics.

Applies known raw_gripper values through the production pipeline and records
the resulting env_gripper and qpos response.

Expected behavior:
  - raw_gripper ≈ 0.0  →  env_gripper = +1  →  qpos moves toward open (decreases)
  - raw_gripper ≈ 0.996 →  env_gripper = -1  →  qpos stays closed or closes (increases)

Outputs:
  reports/GRIPPER_PHYSICAL_SEMANTICS_AUDIT.md
  tables/gripper_physical_semantics_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get(
    'VLA_REPO',
    '/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524',
)
sys.path.insert(0, f'{REPO}/src')

from gripper_attack.gripper_semantics import (
    raw_gripper_is_open,
    decoded_action_to_env_gripper,
    env_gripper_is_open,
    CANONICAL_OPEN_SEMANTICS_VERSION,
)


# ── Production pipeline (must match vis_rollout_adaptive_v3.py) ──

def normalize_gripper_action(action, binarize=True):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action


def invert_gripper_action(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


# ── Test values representing canonical OPEN and CLOSE ──
OPEN_RAW = 0.0          # neutral bin → env=+1 → OPEN
CLOSE_RAW = 0.996094    # saturation bin → env=-1 → CLOSE/HOLD
NEUTRAL_RAW = 0.5       # boundary


def parse_args():
    ap = argparse.ArgumentParser(
        description='Smoke-test gripper physical semantics via LIBERO env')
    ap.add_argument('--task', default='ketchup',
                     choices=['ketchup', 'cream_cheese', 'salad_dressing', 'tomato_sauce'])
    ap.add_argument('--gpu', type=int, default=0, help='GPU for rendering')
    ap.add_argument('--steps', type=int, default=10,
                     help='steps per test condition')
    ap.add_argument('--output-dir', default='tables')
    ap.add_argument('--report', default='reports/GRIPPER_PHYSICAL_SEMANTICS_AUDIT.md')
    return ap.parse_args()


def init_libero_env(task_name, gpu_id, seed=0):
    """Minimal LIBERO env init."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    TASK_IDS = {
        'cream_cheese': 1, 'salad_dressing': 2, 'ketchup': 4, 'tomato_sauce': 5,
    }

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_object']()
    task = task_suite.get_task(TASK_IDS[task_name])
    bddl = os.path.join(get_libero_path('bddl_files'),
                        task.problem_folder, task.bddl_file)
    initial_states = task_suite.get_task_init_states(TASK_IDS[task_name])

    env_args = {
        'bddl_file_name': bddl,
        'camera_heights': 256, 'camera_widths': 256,
        'has_renderer': False, 'has_offscreen_renderer': True,
        'use_camera_obs': True, 'camera_names': ['agentview'],
        'control_freq': 20,
        'render_gpu_device_id': gpu_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, initial_states[0]


def run_condition(env, init_state, raw_gripper_val, n_steps, label):
    """Apply raw_gripper_val for n_steps and record qpos trend."""
    env.reset()
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    env.set_init_state(init_state)

    # 5 warmup steps
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7))

    rows = []
    for step in range(n_steps):
        # Build action: arm zeros, gripper = raw_gripper_val
        action = np.zeros(7, dtype=np.float32)
        action[-1] = raw_gripper_val

        # Apply production pipeline
        env_action = normalize_gripper_action(action, binarize=True)
        env_action = invert_gripper_action(env_action)

        obs, reward, done, info = env.step(env_action)
        qpos = obs['robot0_gripper_qpos'].copy()
        qpos_val = float(qpos[0]) if len(qpos) > 0 else 0.0

        env_grip = decoded_action_to_env_gripper(raw_gripper_val)
        rows.append({
            'label': label,
            'raw_gripper': raw_gripper_val,
            'env_gripper': env_grip,
            'env_is_open': env_gripper_is_open(env_grip),
            'canonical_is_open': raw_gripper_is_open(raw_gripper_val),
            'step': step,
            'qpos': qpos_val,
        })

    # NOTE: do NOT close env here — caller reuses the same env for both conditions.
    return rows


def main():
    args = parse_args()
    print(f'[0] Physical semantics smoke: task={args.task} steps={args.steps}')
    print(f'    Canonical version: {CANONICAL_OPEN_SEMANTICS_VERSION}')

    # Stage 0: verify pipeline equivalence WITHOUT env (no GPU needed)
    print('[1] Stage 0: Pipeline equivalence check (no env)')
    for raw_val, expected_env, expected_open in [
        (OPEN_RAW, 1.0, True),
        (CLOSE_RAW, -1.0, False),
        (0.0, 1.0, True),
        (0.3, 1.0, True),
        (0.8, -1.0, False),
        (-0.5, 1.0, True),
    ]:
        env_g = decoded_action_to_env_gripper(raw_val)
        is_o = raw_gripper_is_open(raw_val)
        assert env_g == expected_env, \
            f"raw={raw_val}: expected env={expected_env}, got {env_g}"
        assert is_o == expected_open, \
            f"raw={raw_val}: expected open={expected_open}, got {is_o}"
        assert is_o == env_gripper_is_open(env_g), \
            f"raw={raw_val}: raw_is_open={is_o} != env_is_open={env_gripper_is_open(env_g)}"
    print('    PASS: All equivalence checks passed.')

    # Stage 1: LIBERO env smoke (requires GPU)
    print(f'[2] Stage 1: LIBERO env smoke (GPU {args.gpu})')
    env, init_state = init_libero_env(args.task, args.gpu)

    all_rows = []
    results = {}  # label -> {pass, qpos_start, qpos_end, qpos_delta}
    for raw_val, label in [
        (OPEN_RAW, 'open_raw_0.0'),
        (CLOSE_RAW, 'close_raw_0.996'),
    ]:
        print(f'    Testing: {label} (raw={raw_val})')
        rows = run_condition(env, init_state, raw_val, args.steps, label)
        all_rows.extend(rows)

        qpos_vals = [r['qpos'] for r in rows]
        qpos_start = qpos_vals[0]
        qpos_end = qpos_vals[-1]
        qpos_delta = qpos_end - qpos_start

        if raw_gripper_is_open(raw_val):
            passed = qpos_delta < 0 or qpos_end < 0.02
            status = 'PASS' if passed else 'FAIL'
            print(f'      qpos: {qpos_start:.4f} -> {qpos_end:.4f} (delta={qpos_delta:.4f}) [{status}]')
        else:
            passed = qpos_delta >= -0.001
            status = 'PASS' if passed else 'FAIL'
            print(f'      qpos: {qpos_start:.4f} -> {qpos_end:.4f} (delta={qpos_delta:.4f}) [{status}]')
        results[label] = {'passed': passed, 'qpos_start': qpos_start,
                          'qpos_end': qpos_end, 'qpos_delta': qpos_delta}

    env.close()

    # Hard assert: both conditions must pass.
    failed_any = False
    for label, r in results.items():
        if not r['passed']:
            failed_any = True
            print(f'FATAL: {label} FAILED: qpos {r["qpos_start"]:.4f} -> {r["qpos_end"]:.4f} (delta={r["qpos_delta"]:.4f})')
    if failed_any:
        print('FATAL: Physical semantics smoke FAILED. Canonical OPEN/CLOSE mapping not verified in env.')
        sys.exit(1)

    # Write CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'gripper_physical_semantics_smoke.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f'    CSV saved: {csv_path}')

    # Write report with actual measured values
    _open_r = results.get('open_raw_0.0', {})
    _close_r = results.get('close_raw_0.996', {})
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(f"""# Gripper Physical Semantics Audit

**Date**: 2026-06-03
**Canonical version**: `{CANONICAL_OPEN_SEMANTICS_VERSION}`
**Task**: `{args.task}`
**Steps per condition**: {args.steps}

## Stage 0: Pipeline Equivalence (no env)

All parameterized raw→env→open mappings verified:
- `raw=0.0` → `env=+1` → OPEN
- `raw=0.996` → `env=-1` → CLOSE
- `raw_is_open ⇔ env_is_open` for all tested values

**PASS**

## Stage 1: LIBERO Physical Response

### OPEN condition (raw={OPEN_RAW})
- Expected: `env_gripper=+1` → qpos decreases (gripper opens)
- Measured: qpos {_open_r.get('qpos_start', '?')} → {_open_r.get('qpos_end', '?')} (delta={_open_r.get('qpos_delta', '?')})
- Status: **{'PASS' if _open_r.get('passed') else 'FAIL'}**

### CLOSE/HOLD condition (raw={CLOSE_RAW})
- Expected: `env_gripper=-1` → qpos stays high or increases
- Measured: qpos {_close_r.get('qpos_start', '?')} → {_close_r.get('qpos_end', '?')} (delta={_close_r.get('qpos_delta', '?')})
- Status: **{'PASS' if _close_r.get('passed') else 'FAIL'}**

## Verdict

{'**PASS**: Canonical semantics confirmed in LIBERO env.' if not failed_any else '**FAIL**: Physical response does not match canonical semantics.'}

## Claim

The canonical semantics (`raw_gripper < 0.5 ⇔ OPEN ⇔ env=+1 ⇔ qpos decreases`)
is CORRECT for the LIBERO-Object gripper action space.
""")
    print(f'    Report saved: {args.report}')

    print()
    print('[3] Done. Physical semantics audit complete.')


if __name__ == '__main__':
    main()
