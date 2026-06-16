#!/usr/bin/env python3
"""Scan server for all existing LIBERO Object clean trace CSVs and produce an inventory.

Upgraded (2026-06-16): episode-level discovery, no first-occurrence dedup,
tier classification, step continuity, fast SHA, gap queue.

Output:
  data_inventory/object500_episode_inventory.csv  — every episode copy ranked
  data_inventory/object500_usage_qualification.csv — tier counts + notes
"""
import csv, hashlib, os, re, sys
from collections import defaultdict, Counter
from pathlib import Path

OUTPUTS_ROOT = "/data/liuyu/outputs"
TEN_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
]

D4_REQUIRED_FIELDS = [
    "step", "raw_gripper", "env_gripper", "gripper_qpos_before",
    "eef_x", "eef_y", "eef_z", "decoded_open",
    "raw_valid", "env_valid", "qpos_valid", "eef_valid", "semantics_ok",
    "success_done", "success_check",
]

D4_PRIVILEGED_FIELDS = [
    "obj_pre_x", "obj_pre_y", "obj_pre_z",
    "obj_post_x", "obj_post_y", "obj_post_z",
    "eef_pre_x", "eef_pre_y", "eef_pre_z",
    "eef_post_x", "eef_post_y", "eef_post_z",
    "eef_to_obj_pre", "eef_to_obj_post",
    "privileged_valid", "obj_z_delta_post",
    "obj_init_x", "obj_init_y", "obj_init_z",
    "target_object_name",
]


def sha256_file_fast(path):
    """Fast SHA256: file size + first 64KB."""
    if not os.path.isfile(path):
        return ""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def sha256_file(path):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_episode_name(dirname):
    for tk in TEN_TASKS:
        if tk in dirname.lower():
            m = re.search(rf'{tk}_s(\d+)', dirname)
            if m:
                return tk, int(m.group(1))
    return None, None


def check_step_continuity(rows):
    if not rows:
        return False, "empty_trace"
    expected = 0
    for r in rows:
        try:
            s = int(r.get("step", -1))
        except (ValueError, TypeError):
            return False, f"bad_step_value_at_expected_{expected}"
        if s != expected:
            return False, f"step_gap_expected_{expected}_got_{s}"
        expected += 1
    return True, f"{expected}_steps_ok"


def classify_episode(has_fields, has_files, step_ok, n_rows):
    if not has_files.get("step_trace"):
        return "quarantined", "missing_step_trace"
    if not step_ok:
        return "quarantined", "step_continuity_fail"

    d4_basic = all(has_fields.get(f, False) for f in [
        "step", "raw_gripper", "env_gripper", "eef_x", "eef_y", "eef_z",
        "decoded_open", "success_done",
    ])
    if not d4_basic:
        missing = [f for f in D4_REQUIRED_FIELDS if not has_fields.get(f, False)]
        return "baseline_only", f"missing_fields:{','.join(missing[:5])}"

    has_privileged = all(has_fields.get(f, False) for f in [
        "obj_pre_x", "obj_pre_z", "eef_to_obj_pre", "privileged_valid",
    ])
    has_pre_post = has_fields.get("obj_post_z") and has_fields.get("obj_pre_z")
    has_candidates = has_files.get("detector_candidates")
    has_provenance = has_files.get("provenance") and has_files.get("artifact_hashes")

    if has_provenance and has_candidates and has_privileged and has_pre_post:
        return "streaming_replay_usable", "full_d4_schema_with_candidates"
    if has_privileged and has_pre_post:
        return "teacher_label_usable", "privileged_with_pre_post"
    if has_privileged:
        return "teacher_label_usable", "privileged_no_pre_post"
    if has_candidates:
        return "student_train_usable", "has_candidates_no_privileged"
    return "student_train_usable", "d4_basic_fields"


def discover_episodes():
    """Walk OUTPUTS_ROOT for episode directories containing step_trace.csv."""
    episodes = []
    for root, dirs, files in os.walk(OUTPUTS_ROOT):
        depth = root[len(OUTPUTS_ROOT):].count(os.sep)
        if depth > 5:
            dirs.clear()
            continue
        if "step_trace.csv" in files:
            task, sid = parse_episode_name(os.path.basename(root))
            if task is None:
                task, sid = parse_episode_name(os.path.basename(os.path.dirname(root)))
            if task is not None:
                episodes.append((root, task, sid))
    return episodes


def scan_directory(dirpath):
    """Legacy CSV-file-level scanner (kept for backward compatibility)."""
    results = []
    if not os.path.isdir(dirpath):
        return results
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith('.csv'):
            continue
        task, sid = parse_episode_name(fname)
        if task is None:
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            with open(fpath) as f:
                reader = csv.reader(f)
                header = next(reader)
                nrows = sum(1 for _ in reader)
        except Exception:
            header = []
            nrows = -1

        has_privileged = all(f in header for f in ['obj_x', 'obj_z', 'eef_x'])
        has_eef = 'eef_x' in header
        has_qpos = 'gripper_qpos_before' in header or 'gripper_qpos' in header
        has_raw_gripper = 'clean_gripper_raw' in header or 'raw_gripper' in header
        has_env_gripper = 'clean_gripper_env' in header or 'env_gripper' in header
        has_open = 'decoded_open_bool' in header or 'decoded_open' in header
        has_success = any(f in header for f in ['success_primary', 'success_check', 'success_done'])
        has_step = 'step' in header
        has_obj_init = 'obj_init_z' in header
        has_pre_post = 'obj_post_z' in header and 'obj_pre_z' in header

        teacher_ok = has_privileged and has_step
        student_ok = has_raw_gripper and has_env_gripper and has_qpos and has_step

        results.append({
            "task": task, "state_id": sid,
            "file_path": fpath, "dir_name": os.path.basename(dirpath),
            "n_rows": nrows,
            "has_privileged_object": int(has_privileged),
            "has_eef": int(has_eef), "has_qpos": int(has_qpos),
            "has_raw_gripper": int(has_raw_gripper),
            "has_env_gripper": int(has_env_gripper),
            "has_decoded_open": int(has_open),
            "has_success": int(has_success),
            "has_obj_init": int(has_obj_init),
            "has_pre_post": int(has_pre_post),
            "has_step": int(has_step),
            "teacher_label_usable": int(teacher_ok and has_eef),
            "student_train_usable": int(student_ok),
            "file_sha256": "",
        })
    return results


def run_episode_level():
    """Episode-level inventory: discover step_trace.csv dirs, tier-classify, rank copies."""
    print("Discovering episodes with step_trace.csv...")
    episodes = discover_episodes()
    print(f"Found {len(episodes)} episode directories")

    grouped = defaultdict(list)
    for ep_dir, task, sid in episodes:
        grouped[(task, sid)].append(ep_dir)

    n_unique = len(grouped)
    total_copies = len(episodes)
    print(f"Unique (task, state_id): {n_unique}  Total copies: {total_copies}")

    inventory = []
    task_counts = defaultdict(int)

    for (task, sid), dirs in sorted(grouped.items()):
        candidates = []
        for ep_dir in dirs:
            st_path = os.path.join(ep_dir, "step_trace.csv")
            if not os.path.exists(st_path):
                continue
            try:
                rows = list(csv.DictReader(open(st_path)))
            except Exception:
                continue
            n_rows = len(rows)
            header_fields = list(rows[0].keys()) if rows else []

            has_fields = {f: (f in header_fields) for f in D4_REQUIRED_FIELDS + D4_PRIVILEGED_FIELDS}
            has_files = {
                "step_trace": True,
                "detector_candidates": os.path.exists(os.path.join(ep_dir, "detector_candidates.csv")),
                "action_identity": os.path.exists(os.path.join(ep_dir, "action_identity.csv")),
                "detector_emission": os.path.exists(os.path.join(ep_dir, "detector_emission.json")),
                "teacher_sidecar": os.path.exists(os.path.join(ep_dir, "teacher_sidecar.json")),
                "provenance": os.path.exists(os.path.join(ep_dir, "provenance.csv")),
                "artifact_hashes": os.path.exists(os.path.join(ep_dir, "artifact_hashes.csv")),
                "latency": os.path.exists(os.path.join(ep_dir, "latency.csv")),
            }
            step_ok, step_note = check_step_continuity(rows)
            st_sha = sha256_file_fast(st_path) if step_ok else ""
            cands_sha = sha256_file_fast(os.path.join(ep_dir, "detector_candidates.csv")) if has_files["detector_candidates"] else ""
            tier, reason = classify_episode(has_fields, has_files, step_ok, n_rows)

            prov_commit = ""
            prov_gpu = ""
            if has_files["provenance"]:
                try:
                    for pr in csv.DictReader(open(os.path.join(ep_dir, "provenance.csv"))):
                        if pr.get("key") == "git_HEAD":
                            prov_commit = pr.get("value", "")[:16]
                        if pr.get("key") == "cuda_visible_devices":
                            prov_gpu = pr.get("value", "")
                except Exception:
                    pass

            completeness = (
                sum(1 for v in has_fields.values() if v)
                + sum(1 for v in has_files.values() if v) * 2
                + (10 if step_ok else 0)
                + (5 if prov_commit else 0)
            )

            candidates.append({
                "task": task, "state_id": sid,
                "episode_dir": ep_dir, "n_rows": n_rows,
                "step_ok": step_ok, "step_note": step_note,
                "tier": tier, "tier_reason": reason,
                "completeness": completeness,
                "prov_commit": prov_commit, "prov_gpu": prov_gpu,
                "step_trace_sha256_fast": st_sha,
                "candidates_sha256_fast": cands_sha,
                "n_d4_basic_fields": sum(1 for f in D4_REQUIRED_FIELDS if has_fields.get(f)),
                "n_d4_privileged_fields": sum(1 for f in D4_PRIVILEGED_FIELDS if has_fields.get(f)),
                "has_detector_candidates": int(has_files["detector_candidates"]),
                "has_provenance": int(has_files["provenance"]),
                "has_artifact_hashes": int(has_files["artifact_hashes"]),
                "has_teacher_sidecar": int(has_files["teacher_sidecar"]),
            })

        candidates.sort(key=lambda x: x["completeness"], reverse=True)
        best = candidates[0]
        best["n_copies"] = len(candidates)
        best["copy_rank"] = 1
        best["tie_breaker"] = "highest_completeness"
        inventory.append(best)
        task_counts[task] += 1

        for rank, c in enumerate(candidates[1:], 2):
            c["n_copies"] = len(candidates)
            c["copy_rank"] = rank
            c["tie_breaker"] = f"completeness_rank_{rank}"
            inventory.append(c)

    # Write episode inventory
    out_dir = os.path.join(OUTPUTS_ROOT, "data_inventory")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "object500_episode_inventory.csv")
    if inventory:
        fields = [
            "task", "state_id", "tier", "tier_reason", "completeness",
            "copy_rank", "n_copies", "tie_breaker",
            "n_rows", "step_ok", "step_note",
            "n_d4_basic_fields", "n_d4_privileged_fields",
            "has_detector_candidates", "has_provenance", "has_artifact_hashes",
            "has_teacher_sidecar",
            "prov_commit", "prov_gpu",
            "step_trace_sha256_fast", "candidates_sha256_fast",
            "episode_dir",
        ]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(inventory)
    print(f"Episode inventory: {out_csv} ({len(inventory)} rows)")

    # Tier distribution
    tier_counts = defaultdict(int)
    for r in inventory:
        if r["copy_rank"] == 1:
            tier_counts[r["tier"]] += 1

    print("\n=== Tier Distribution (best copy per task-state) ===")
    for tier in ["streaming_replay_usable", "teacher_label_usable",
                  "student_train_usable", "baseline_only", "quarantined"]:
        print(f"  {tier}: {tier_counts.get(tier, 0)}")

    # Usage qualification
    usage = defaultdict(int)
    for r in inventory:
        if r["copy_rank"] != 1:
            continue
        t = r["tier"]
        if t == "streaming_replay_usable":
            usage["teacher_label_usable"] += 1
            usage["student_train_usable"] += 1
            usage["streaming_replay_usable"] += 1
        elif t == "teacher_label_usable":
            usage["teacher_label_usable"] += 1
            usage["student_train_usable"] += 1
        elif t == "student_train_usable":
            usage["student_train_usable"] += 1
        elif t == "baseline_only":
            usage["baseline_only"] += 1
        elif t == "quarantined":
            usage["quarantined"] += 1

    usage_csv = os.path.join(out_dir, "object500_usage_qualification.csv")
    with open(usage_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "note"])
        w.writerow(["teacher_label_usable", usage["teacher_label_usable"],
                     "Has privileged object pose + EEF, can generate Teacher-P labels"])
        w.writerow(["student_train_usable", usage["student_train_usable"],
                     "Has gripper/env/qpos/step, can train detector"])
        w.writerow(["streaming_replay_usable", usage["streaming_replay_usable"],
                     "Full D4 schema + candidates + provenance, can replay"])
        w.writerow(["baseline_only", usage["baseline_only"],
                     "Limited fields, baseline comparison only"])
        w.writerow(["quarantined", usage["quarantined"],
                     "Schema/provenance/continuity failure"])
        w.writerow(["total_unique", n_unique, ""])

    print("\n=== Usage Qualification ===")
    for cat in ["teacher_label_usable", "student_train_usable", "streaming_replay_usable",
                "baseline_only", "quarantined"]:
        print(f"  {cat}: {usage[cat]}")

    # Per-task
    print("\n=== Per-Task ===")
    for tk in TEN_TASKS:
        n = task_counts.get(tk, 0)
        n_stream = sum(1 for r in inventory
                       if r["task"] == tk and r["copy_rank"] == 1
                       and r["tier"] == "streaming_replay_usable")
        print(f"  {tk}: {n} states, streaming={n_stream}")

    # Gap queue
    print("\n=== Gap Queue ===")
    gaps = []
    for (task, sid), dirs in sorted(grouped.items()):
        has_streaming = False
        for d in dirs:
            st_path = os.path.join(d, "step_trace.csv")
            if not os.path.exists(st_path):
                continue
            try:
                rows = list(csv.DictReader(open(st_path)))
            except Exception:
                continue
            if not rows:
                continue
            hf = {f: (f in list(rows[0].keys())) for f in D4_REQUIRED_FIELDS + D4_PRIVILEGED_FIELDS}
            hfiles = {
                "step_trace": True,
                "detector_candidates": os.path.exists(os.path.join(d, "detector_candidates.csv")),
                "provenance": os.path.exists(os.path.join(d, "provenance.csv")),
                "artifact_hashes": os.path.exists(os.path.join(d, "artifact_hashes.csv")),
            }
            sk, _ = check_step_continuity(rows)
            tier, _ = classify_episode(hf, hfiles, sk, len(rows))
            if tier == "streaming_replay_usable":
                has_streaming = True
                break
        if not has_streaming:
            gaps.append(f"{task}_s{sid}: no streaming_replay_usable, best={dirs[0]}")

    for g in gaps[:20]:
        print(f"  {g}")
    if len(gaps) > 20:
        print(f"  ... and {len(gaps) - 20} more")
    print(f"Total gap states: {len(gaps)}")
    print(f"\nUsage CSV: {usage_csv}")

    return inventory, usage, n_unique


def run_legacy_scan():
    """Legacy directory-level CSV scanner (original behavior)."""
    stageb_dirs = []
    for d in os.listdir(OUTPUTS_ROOT):
        dp = os.path.join(OUTPUTS_ROOT, d)
        if os.path.isdir(dp) and 'stageb' in d.lower():
            stageb_dirs.append(dp)
    for d in os.listdir(OUTPUTS_ROOT):
        dp = os.path.join(OUTPUTS_ROOT, d)
        if os.path.isdir(dp) and any(k in d.lower() for k in ['milestone', 'object', 'libero', 'd4', 'd5']):
            if dp not in stageb_dirs:
                stageb_dirs.append(dp)

    print(f"Scanning {len(stageb_dirs)} directories (legacy mode)...")
    seen = set()
    inventory = []
    for dp in sorted(stageb_dirs):
        for row in scan_directory(dp):
            key = (row["task"], row["state_id"])
            if key in seen:
                continue
            seen.add(key)
            inventory.append(row)

    out_csv = os.path.join(OUTPUTS_ROOT, "data_inventory", "object_legacy_trace_inventory.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if inventory:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
            w.writeheader()
            w.writerows(inventory)

    total = len(inventory)
    teacher_n = sum(1 for r in inventory if r["teacher_label_usable"])
    student_n = sum(1 for r in inventory if r["student_train_usable"])
    print(f"Total unique task-state traces: {total}")
    print(f"Teacher-label usable: {teacher_n}")
    print(f"Student-train usable: {student_n}")
    for tk in TEN_TASKS:
        n = sum(1 for r in inventory if r["task"] == tk)
        print(f"  {tk}: {n}")
    print(f"Output: {out_csv}")


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument("--mode", choices=["episode", "legacy", "both"], default="episode",
                        help="episode=step_trace.csv dirs (default), legacy=CSV files, both=run both")
    args = parser.parse_args()

    if args.mode in ("episode", "both"):
        run_episode_level()
    if args.mode in ("legacy", "both"):
        run_legacy_scan()


if __name__ == "__main__":
    main()
