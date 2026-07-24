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

    # Check if telemetry has real target fields (patched bridge) or needs approximate fallback
    has_target = all(k in rows[0] for k in ['target_x', 'target_y', 'target_z']) if rows else False
    has_gripper_qpos = 'gripper_qpos' in rows[0] if rows else False

    records = []
    skipped = 0
    for step_idx, r in enumerate(rows):
        # Fail closed: all fields must exist and be finite
        try:
            grip_left = float(r['gripper_qpos_left'])
            grip_right = float(r['gripper_qpos_right'])
            grip_width = float(r['gripper_width'])
            raw_grip = float(r['raw_gripper'])
            eef_x = float(r['eef_x']); eef_y = float(r['eef_y']); eef_z = float(r['eef_z'])
            eef_vx = float(r['eef_vx']); eef_vy = float(r['eef_vy']); eef_vz = float(r['eef_vz'])
            obj_x = float(r['object_x']); obj_y = float(r['object_y']); obj_z = float(r['object_z'])
            obj_eef_dist = float(r['object_eef_distance'])

            if has_target:
                tgt_x = float(r['target_x']); tgt_y = float(r['target_y']); tgt_z = float(r['target_z'])
                obj_tgt_dist = float(r.get('object_to_target_distance', np.sqrt((obj_x-tgt_x)**2 + (obj_y-tgt_y)**2 + (obj_z-tgt_z)**2)))
            else:
                tgt_x = 0.0; tgt_y = 0.0; tgt_z = 0.9
                obj_tgt_dist = np.sqrt((obj_x-tgt_x)**2 + (obj_y-tgt_y)**2 + (obj_z-tgt_z)**2)

            grip_qpos = float(r['gripper_qpos']) if has_gripper_qpos else (grip_left + grip_right)
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue

        records.append({
            'step_idx': step_idx,
            'policy_step_idx': int(r.get('step', step_idx)),
            'teacher_privileged_state_available': True,
            'gripper_command': raw_grip,
            'gripper_qpos': grip_qpos,
            'gripper_width': grip_width,
            'gripper_opening_proxy': grip_width,
            'eef_x': eef_x, 'eef_y': eef_y, 'eef_z': eef_z,
            'eef_vx': eef_vx, 'eef_vy': eef_vy, 'eef_vz': eef_vz,
            'object_eef_distance': obj_eef_dist,
            'object_to_target_distance': obj_tgt_dist,
            'object_pose_json': json.dumps([obj_x, obj_y, obj_z]),
            'target_pose_json': json.dumps([tgt_x, tgt_y, tgt_z]),
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
