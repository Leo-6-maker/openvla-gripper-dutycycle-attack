#!/usr/bin/env python3
"""C0: Audit Object frame capture — 14 parents, 65 frames."""
import csv, json, hashlib, os, re

FRAME_DIR = "/data/liuyu/outputs/l12_frame_handoff_v2_r1"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
PANEL = "/data/liuyu/outputs/l12_timing_panel_v2"
RESUME = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"


def sha256_file(p):
    if not os.path.isfile(p): return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


acc = {}
for r in csv.DictReader(open(MANIFEST)):
    if r.get("status") == "BOUND":
        acc[(r["task"], int(r["state_id"]))] = r

labels = {}
for r in csv.DictReader(open(LABELS)):
    labels[(r["task"], int(r["state_id"]))] = r


def get_timing_emit(task, sid):
    for base in [PANEL, RESUME]:
        p = os.path.join(base, task + "_s" + str(sid) + "_shadow_attempt1", "detector_emission.json")
        if os.path.exists(p):
            return json.load(open(p)).get("emit_step", -1)
    return -1


parents = []
total_frames = 0
for d in sorted(os.listdir(FRAME_DIR)):
    dp = os.path.join(FRAME_DIR, d)
    if not os.path.isdir(dp): continue
    m = re.match(r"(.+)_s(\d+)_frame", d)
    if not m: continue
    task = m.group(1)
    sid = int(m.group(2))

    lr = labels.get((task, sid), {})
    teacher_ws = int(lr.get("ws", -1))
    teacher_anchor = int(lr.get("anchor", -1))
    teacher_we = int(lr.get("we", -1))
    d5_emit = get_timing_emit(task, sid)

    frames_data = []
    for fname in sorted(os.listdir(dp)):
        if not fname.endswith(".npy"): continue
        step = int(fname.replace("frame_", "").replace(".npy", ""))
        npy_path = os.path.join(dp, fname)
        pt_path = os.path.join(dp, fname.replace(".npy", ".pt"))
        npy_sha = sha256_file(npy_path)
        pt_sha = sha256_file(pt_path)

        role = "other"
        if step == d5_emit and d5_emit >= 0:
            role = "d5_emit"
        if step == teacher_anchor:
            role = "teacher_anchor"
        elif step == teacher_ws:
            role = "teacher_ws"
        elif step == teacher_we:
            role = "teacher_we"

        frames_data.append({"step": step, "role": role, "npy_sha": npy_sha, "pt_sha": pt_sha,
                            "npy_path": npy_path, "pt_path": pt_path})

    if d5_emit < 0: tcls = "miss"
    elif d5_emit == teacher_anchor: tcls = "exact"
    elif d5_emit < teacher_ws: tcls = "early"
    elif d5_emit < teacher_we: tcls = "in_window"
    else: tcls = "late"

    sp = acc.get((task, sid), {}).get("split", "?")

    parents.append({
        "task": task, "state_id": sid, "split": sp, "timing_class": tcls,
        "teacher_ws": teacher_ws, "teacher_anchor": teacher_anchor, "teacher_we": teacher_we,
        "d5_emit": d5_emit, "n_frames": len(frames_data), "frames": frames_data,
    })
    total_frames += len(frames_data)

print("C0 AUDIT: " + str(len(parents)) + " parents, " + str(total_frames) + " frames")
for p in parents:
    roles = sorted(set(f["role"] for f in p["frames"]))
    print("  " + p["task"] + "_s" + str(p["state_id"]) + ": " + p["timing_class"]
          + " emit=" + str(p["d5_emit"]) + " frames=" + str(p["n_frames"])
          + " roles=" + str(roles))

selected = [("butter", "11"), ("tomato_sauce", "23"), ("salad_dressing", "11")]
print("\n=== Selected 3 parents ===")
for t, s in selected:
    p = next((x for x in parents if x["task"] == t and x["state_id"] == int(s)), None)
    if not p:
        print("  " + t + "_s" + s + ": MISSING")
        continue
    actual_roles = set(f["role"] for f in p["frames"])
    exp_roles = {"teacher_ws", "teacher_anchor", "teacher_we", "d5_emit"}
    missing = exp_roles - actual_roles
    print("  " + t + "_s" + s + ": frames=" + str(p["n_frames"])
          + " roles_actual=" + str(sorted(actual_roles))
          + " missing=" + str(sorted(missing) if missing else "NONE"))
    for f in p["frames"]:
        inside = "IN" if p["teacher_ws"] <= f["step"] < p["teacher_we"] else "OUT"
        print("    step=" + str(f["step"]) + " role=" + f["role"] + " " + inside)
