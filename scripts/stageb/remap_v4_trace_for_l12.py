#!/usr/bin/env python3
"""Remap V4 trace CSV to L12-expected field format.

The V4 runner captures 34 fields but lacks some L12-expected fields.
This script fills in computable fields and marks missing privileged
fields as empty/NA (rather than defaulting to 0).
"""

import csv
import sys


def remap_v4_to_l12(input_path: str, output_path: str):
    """Read V4-format trace, write L12-format CSV."""
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        v4_rows = list(reader)

    prev_clean_close = 0
    close_streak = 0

    l12_rows = []
    for t, r in enumerate(v4_rows):
        # ── Compute from raw V4 fields ──
        clean_gripper_env = float(r.get("clean_gripper_env", 0))
        decoded_open_bool = int(r.get("decoded_open_bool", 1))
        gripper_qpos_before = float(r.get("gripper_qpos_before", 0))
        eef_x = float(r.get("eef_x", 0))
        eef_y = float(r.get("eef_y", 0))
        eef_z = float(r.get("eef_z", 0))
        obj_x = float(r.get("obj_x", 0))
        obj_y = float(r.get("obj_y", 0))
        obj_z = float(r.get("obj_z", 0))

        # Compute raw from env: env=+1→raw>0.5 (OPEN), env=-1→raw<0.5 (CLOSE)
        if clean_gripper_env > 0:
            clean_gripper_raw = 0.7  # OPEN
        else:
            clean_gripper_raw = 0.0  # CLOSE

        clean_close = int(clean_gripper_raw <= 0.5)
        close_onset = int(clean_close and not prev_clean_close)

        if clean_close:
            close_streak += 1
        else:
            close_streak = 0

        qpos_abs_before = abs(gripper_qpos_before)

        # Distances
        eef_to_obj_distance = ((eef_x - obj_x)**2 + (eef_y - obj_y)**2 + (eef_z - obj_z)**2)**0.5

        # Target coordinates — NOT in V4 trace, mark as empty
        # (libt_target object exists but coordinates not captured by V4 runner)

        row = {
            "step": t,
            "clean_gripper_env": clean_gripper_env,
            "clean_gripper_raw": clean_gripper_raw,
            "gripper_qpos_before": gripper_qpos_before,
            "gripper_qpos_after": float(r.get("gripper_qpos_after", 0)),
            "qpos_abs_before": qpos_abs_before,
            "qpos_abs_after": abs(float(r.get("gripper_qpos_after", 0))),
            "eef_x": eef_x,
            "eef_y": eef_y,
            "eef_z": eef_z,
            "clean_close": clean_close,
            "close_onset": close_onset,
            "close_streak": close_streak,
            "decoded_open_bool": decoded_open_bool,
            "obj_x": obj_x,
            "obj_y": obj_y,
            "obj_z": obj_z,
            # Target coords NOT available in V4 trace — honest empty
            "target_obj_x": "",
            "target_obj_y": "",
            "target_obj_z": "",
            "eef_to_obj_distance": eef_to_obj_distance,
            "obj_to_target_distance": "",  # NOT available
            "success": int(r.get("success_done", 0)),
            "done": int(r.get("success_done", 0)),
            "source_trace": input_path,
        }
        l12_rows.append(row)
        prev_clean_close = clean_close

    fieldnames = list(l12_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(l12_rows)

    return l12_rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: remap_v4_trace_for_l12.py <input_v4.csv> <output_l12.csv>")
        sys.exit(1)
    rows = remap_v4_to_l12(sys.argv[1], sys.argv[2])
    print(f"Remapped {len(rows)} rows to {sys.argv[2]}")
