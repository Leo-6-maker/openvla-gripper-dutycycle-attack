#!/usr/bin/env python3
"""Scan server for all existing LIBERO Object clean trace CSVs and produce an inventory.

Output: object_legacy_trace_inventory.csv
Each row = one trace_file with columns for task, state_id, available fields.
"""
import csv, json, os, sys, hashlib, re
from pathlib import Path
from collections import defaultdict, Counter

OUTPUTS_ROOT = "/data/liuyu/outputs"
TEN_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
]
SEEN_KEYS = set()
INVENTORY = []

def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def parse_trace_filename(fname):
    """Parse '{task}_s{state_id}_...' from filename."""
    for tk in TEN_TASKS:
        if fname.startswith(tk):
            rest = fname[len(tk):]
            m = re.match(r'_s(\d+)', rest)
            if m:
                return tk, int(m.group(1))
    return None, None

def scan_directory(dirpath):
    """Scan a directory for trace CSVs."""
    if not os.path.isdir(dirpath):
        return
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith('.csv'):
            continue
        if not any(fname.startswith(tk) for tk in TEN_TASKS):
            continue
        task, sid = parse_trace_filename(fname)
        if task is None:
            continue
        fpath = os.path.join(dirpath, fname)
        key = (task, sid)
        # Dedup: prefer first occurrence per task-state
        if key in SEEN_KEYS:
            continue
        SEEN_KEYS.add(key)

        # Read header to check available fields
        try:
            with open(fpath) as f:
                reader = csv.reader(f)
                header = next(reader)
                nrows = sum(1 for _ in reader) + 1  # +1 for header already read
        except Exception:
            header = []
            nrows = -1

        has_privileged = all(f in header for f in ['obj_x', 'obj_z', 'eef_x'])
        has_eef = 'eef_x' in header
        has_qpos = 'gripper_qpos_before' in header or 'gripper_qpos' in header
        has_raw_gripper = 'clean_gripper_raw' in header or 'raw_gripper' in header
        has_env_gripper = 'clean_gripper_env' in header or 'env_gripper' in header
        has_open = 'decoded_open_bool' in header or 'decoded_open' in header
        has_success = any(f in header for f in ['success_primary', 'success_check'])
        has_eef_to_obj = 'eef_to_obj' in str(header) or 'obj_to_eef' in str(header)
        has_obj_init = 'obj_init_z' in header
        has_pre_post = 'obj_post_z' in header and 'obj_pre_z' in header
        has_obj_z_delta = 'obj_z_delta' in str(header)
        has_step = 'step' in header
        has_split = 'split' in header

        # Teacher label usable: privileged object + step-aligned + EEF
        teacher_ok = has_privileged and has_step
        student_ok = has_raw_gripper and has_env_gripper and has_qpos and has_step

        INVENTORY.append({
            "task": task,
            "state_id": sid,
            "file_path": fpath,
            "dir_name": os.path.basename(dirpath),
            "n_rows": nrows,
            "has_privileged_object": int(has_privileged),
            "has_eef": int(has_eef),
            "has_qpos": int(has_qpos),
            "has_raw_gripper": int(has_raw_gripper),
            "has_env_gripper": int(has_env_gripper),
            "has_decoded_open": int(has_open),
            "has_success": int(has_success),
            "has_eef_to_obj": int(has_eef_to_obj),
            "has_obj_init": int(has_obj_init),
            "has_pre_post": int(has_pre_post),
            "has_obj_z_delta": int(has_obj_z_delta),
            "has_step": int(has_step),
            "teacher_label_usable": int(teacher_ok and has_eef),
            "student_train_usable": int(student_ok),
            "file_sha256": "",  # too slow for bulk scan
        })

def main():
    # Scan stageb directories first (largest collections)
    stageb_dirs = []
    for d in os.listdir(OUTPUTS_ROOT):
        dp = os.path.join(OUTPUTS_ROOT, d)
        if os.path.isdir(dp) and 'stageb' in d.lower():
            stageb_dirs.append(dp)

    # Also scan milestone and other key dirs
    for d in os.listdir(OUTPUTS_ROOT):
        dp = os.path.join(OUTPUTS_ROOT, d)
        if os.path.isdir(dp) and any(k in d.lower() for k in ['milestone', 'object', 'libero', 'd4', 'd5']):
            if dp not in stageb_dirs:
                stageb_dirs.append(dp)

    print("Scanning {} directories...".format(len(stageb_dirs)))
    for dp in sorted(stageb_dirs):
        scan_directory(dp)

    # Write inventory
    out = "/data/liuyu/outputs/object_legacy_trace_inventory.csv"
    if INVENTORY:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(INVENTORY[0].keys()))
            w.writeheader()
            w.writerows(INVENTORY)

    # Summary
    total = len(INVENTORY)
    teacher_n = sum(1 for r in INVENTORY if r["teacher_label_usable"])
    student_n = sum(1 for r in INVENTORY if r["student_train_usable"])
    priv_n = sum(1 for r in INVENTORY if r["has_privileged_object"])

    print("Total unique task-state traces: {}".format(total))
    print("With privileged object pose: {}".format(priv_n))
    print("Teacher-label usable: {}".format(teacher_n))
    print("Student-train usable: {}".format(student_n))

    # Per-task counts
    task_counts = Counter()
    for r in INVENTORY:
        task_counts[r["task"]] += 1
    for tk in TEN_TASKS:
        print("  {}: {}".format(tk, task_counts.get(tk, 0)))

    print("Output: {}".format(out))


if __name__ == "__main__":
    main()
