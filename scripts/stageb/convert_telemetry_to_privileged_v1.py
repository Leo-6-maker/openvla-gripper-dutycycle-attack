#!/usr/bin/env python3
"""Convert bridge step_telemetry.csv → privileged step_records.jsonl"""
import csv, json, sys, os
from pathlib import Path
import numpy as np

def convert(telemetry_path, output_path, task_name=''):
    rows = []
    with open(telemetry_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Determine target position: for Object tasks, target = basket (0,0,0 approx)
    # Better: use the object's final position or a known basket location
    # For Object tasks, basket is at roughly x≈0, y≈0, z≈0.9
    # For simplicity, use the last object position as reference or compute from known basket
    target_x = 0.0; target_y = 0.0; target_z = 0.9  # approximate basket center

    records = []
    for step_idx, r in enumerate(rows):
        try:
            grip_left = float(r.get('gripper_qpos_left', 0))
            grip_right = float(r.get('gripper_qpos_right', 0))
            grip_width = float(r.get('gripper_width', 0))
            raw_grip = float(r.get('raw_gripper', 0))
            eef_x = float(r.get('eef_x', 0)); eef_y = float(r.get('eef_y', 0)); eef_z = float(r.get('eef_z', 0))
            eef_vx = float(r.get('eef_vx', 0)); eef_vy = float(r.get('eef_vy', 0)); eef_vz = float(r.get('eef_vz', 0))
            obj_x = float(r.get('object_x', 0)); obj_y = float(r.get('object_y', 0)); obj_z = float(r.get('object_z', 0))
            obj_eef_dist = float(r.get('object_eef_distance', 0))
        except (ValueError, KeyError):
            continue

        obj_target_dist = np.sqrt((obj_x - target_x)**2 + (obj_y - target_y)**2 + (obj_z - target_z)**2)

        records.append({
            'step_idx': step_idx,
            'policy_step_idx': int(r.get('step', step_idx)),
            'teacher_privileged_state_available': True,
            'gripper_command': raw_grip,
            'gripper_qpos': grip_left,
            'gripper_width': grip_width,
            'gripper_opening_proxy': grip_width,  # same value, correct key
            'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
            'eef_vx': eef_vx, 'eef_vy': eef_vy, 'eef_vz': eef_vz,
            'object_eef_distance': obj_eef_dist,
            'object_to_target_distance': obj_target_dist,
            'object_pose_json': json.dumps({'x': obj_x, 'y': obj_y, 'z': obj_z}),
            'target_pose_json': json.dumps({'x': target_x, 'y': target_y, 'z': target_z}),
            'phase': 'wait',
        })

    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    # Validation
    widths = [r['gripper_width'] for r in records]
    proxies = [r['gripper_opening_proxy'] for r in records]
    cmds = [r['gripper_command'] for r in records]
    close_w = [widths[i] for i in range(len(widths)) if cmds[i] <= 0.5]
    open_w = [widths[i] for i in range(len(widths)) if cmds[i] > 0.5]

    from statistics import median
    peak_to_peak = max(widths) - min(widths)
    close_med = median(close_w) if close_w else -1
    open_med = median(open_w) if open_w else -1
    gate = peak_to_peak > 1e-4 and close_med < open_med

    return {
        'n_rows': len(records),
        'gripper_width_min': min(widths),
        'gripper_width_max': max(widths),
        'peak_to_peak': peak_to_peak,
        'close_median': close_med,
        'open_median': open_med,
        'n_close': len(close_w),
        'n_open': len(open_w),
        'opening_proxy_gate': gate,
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_telemetry_to_privileged_v1.py <telemetry.csv> <output.jsonl>')
        sys.exit(1)
    result = convert(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
