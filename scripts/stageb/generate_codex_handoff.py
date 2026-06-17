#!/usr/bin/env python3
"""C4: Generate Codex VIS handoff from Object frame capture."""
import csv, json, hashlib, os

FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
OUT_DIR = "tables"
HANDOFF_LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


labels = {}
for r in csv.DictReader(open(HANDOFF_LABELS)):
    labels[(r["task"], int(r["state_id"]))] = r

# Selected 3 parents with their target frame steps
SELECTED = [
    {
        "parent_id": "butter_s11",
        "task": "butter", "state_id": 11,
        "timing_class": "exact", "clean_success": 1,
        "teacher_ws": 58, "teacher_anchor": 60, "teacher_we": 68,
        "d5_emit": 60,
        "target_steps": [58, 60, 68],
        "step_roles": {58: "teacher_ws", 60: "teacher_anchor+d5_emit", 68: "teacher_we"},
    },
    {
        "parent_id": "tomato_sauce_s23",
        "task": "tomato_sauce", "state_id": 23,
        "timing_class": "early", "clean_success": 1,
        "teacher_ws": 139, "teacher_anchor": 141, "teacher_we": 149,
        "d5_emit": 69,
        "target_steps": [69, 139, 141],
        "step_roles": {69: "d5_emit", 139: "teacher_ws", 141: "teacher_anchor"},
        "missing": "teacher_we=149 (episode terminates before this step)",
    },
    {
        "parent_id": "salad_dressing_s11",
        "task": "salad_dressing", "state_id": 11,
        "timing_class": "late", "clean_success": 1,
        "teacher_ws": 57, "teacher_anchor": 59, "teacher_we": 67,
        "d5_emit": 128,
        "target_steps": [57, 59, 67, 128],
        "step_roles": {57: "teacher_ws", 59: "teacher_anchor", 67: "teacher_we", 128: "d5_emit"},
    },
]

# Generate frame table
frame_rows = []
for sel in SELECTED:
    tag = sel["task"] + "_s" + str(sel["state_id"]) + "_frame"
    ep_dir = os.path.join(FRAME_DIR, tag)

    for step, role in sel["step_roles"].items():
        npy_path = os.path.join(ep_dir, f"frame_{step:04d}.npy")
        pt_path = os.path.join(ep_dir, f"processor_{step:04d}.pt")
        png_path = os.path.join(ep_dir, f"frame_{step:04d}.png")

        npy_sha = sha256_file(npy_path)
        pt_sha = sha256_file(pt_path)
        png_sha = sha256_file(png_path)

        inside_win = sel["teacher_ws"] <= step < sel["teacher_we"]
        d5_rel = "emit" if step == sel["d5_emit"] else "other"

        frame_rows.append({
            "parent_id": sel["parent_id"], "task": sel["task"],
            "state_id": str(sel["state_id"]),
            "timing_class": sel["timing_class"],
            "clean_success": str(sel["clean_success"]),
            "frame_step": str(step), "frame_role": role,
            "inside_teacher_window": str(inside_win),
            "d5_emit_relation": d5_rel,
            "raw_frame_path": npy_path, "raw_frame_sha256": npy_sha,
            "processor_tensor_path": pt_path, "processor_tensor_sha256": pt_sha,
            "png_path": png_path, "png_sha256": png_sha,
            "prompt_instruction": "",
            "unnorm_key": "libero_object",
            "target_token": "31744",
            "attack_lambda": "2.0",
            "attack_seeds": "81,82",
            "gpu": "1,5",
        })

# Write frames table
frame_csv = os.path.join(OUT_DIR, "l3_vis_selected_frames_v1.csv")
with open(frame_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
    w.writeheader(); w.writerows(frame_rows)
print(f"Frames: {frame_csv} ({len(frame_rows)} frames)")

# Write parents table
parent_rows = []
for sel in SELECTED:
    parent_rows.append({
        "parent_id": sel["parent_id"], "task": sel["task"], "state_id": str(sel["state_id"]),
        "timing_class": sel["timing_class"], "clean_success": str(sel["clean_success"]),
        "teacher_ws": str(sel["teacher_ws"]), "teacher_anchor": str(sel["teacher_anchor"]),
        "teacher_we": str(sel["teacher_we"]), "d5_emit": str(sel["d5_emit"]),
        "n_frames": str(len(sel["target_steps"])),
        "repeatability": "PASS",
        "missing_frames": sel.get("missing", ""),
    })
parent_csv = os.path.join(OUT_DIR, "l3_vis_selected_parents_v1.csv")
with open(parent_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(parent_rows[0].keys()))
    w.writeheader(); w.writerows(parent_rows)
print(f"Parents: {parent_csv}")

# Job plan
job_rows = []
for sel in SELECTED:
    for step, role in sel["step_roles"].items():
        for seed in [81, 82]:
            job_rows.append({
                "parent_id": sel["parent_id"], "frame_step": str(step),
                "frame_role": role, "seed": str(seed),
                "condition": "TRUE_PGD_TRAJECTORY21_SELECTIVE",
                "target_token": "31744", "lambda": "2.0",
                "gpu": "1,5",
            })
job_csv = os.path.join(OUT_DIR, "l3_vis_job_plan_v1.csv")
with open(job_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(job_rows[0].keys()))
    w.writeheader(); w.writerows(job_rows)
print(f"Job plan: {job_csv} ({len(job_rows)} jobs)")

# Summary
print(f"\n=== VIS Handoff Summary ===")
print(f"Selected parents: {len(SELECTED)}")
print(f"Selected frames: {len(frame_rows)}")
print(f"Job plan entries: {len(job_rows)} (2 seeds × {len(frame_rows)} frames)")
print(f"Lambda: 2.0, Seeds: 81+82, GPU: 1,5")
for sel in SELECTED:
    n_teacher = sum(1 for f in frame_rows if f["parent_id"] == sel["parent_id"] and f["inside_teacher_window"] == "True")
    n_comp = sum(1 for f in frame_rows if f["parent_id"] == sel["parent_id"] and f["d5_emit_relation"] == "emit")
    print(f"  {sel['parent_id']}: {len(sel['target_steps'])} frames ({n_teacher} in-window, {n_comp} comparator)")
    if sel.get("missing"):
        print(f"    NOTE: {sel['missing']}")
