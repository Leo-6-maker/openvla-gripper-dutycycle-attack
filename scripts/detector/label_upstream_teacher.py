#!/usr/bin/env python3
"""Run V2PrivilegedTeacher on upstream artifact-rich C1 corpus. Output labeled CSV."""
import csv, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, TeacherConfig

teacher = V2PrivilegedTeacher(TeacherConfig())

train_dirs = [
    os.path.join(REPO, "evidence/c1_train_a_fp32_gpu2"),
    os.path.join(REPO, "evidence/c1_train_b_fp32_gpu3"),
    os.path.join(REPO, "evidence/c1_val_cal_xfer_fp32_gpu4"),
]

rows = []
n_eps = 0
n_corridor = 0

for d in train_dirs:
    if not os.path.isdir(d):
        continue
    for ep in sorted(os.listdir(d)):
        ep_dir = os.path.join(d, ep)
        if not os.path.isdir(ep_dir):
            continue
        tf = os.path.join(ep_dir, "trace.csv")
        if not os.path.exists(tf):
            continue
        trace = list(csv.DictReader(open(tf)))
        if not trace:
            continue
        n_eps += 1

        # Determine split
        ii = int(trace[0]["init_idx"])
        dname = os.path.basename(d)
        if "train_a" in dname or "train_b" in dname:
            split = "train"
        elif "val_cal_xfer" in dname:
            if ii == 9: split = "val"
            elif ii == 10: split = "cal"
            elif ii >= 11: split = "xfer_test"
            else: split = "held_out"
        else:
            split = "held_out"

        # Build teacher trajectory records
        traj_recs = []
        for i, row in enumerate(trace):
            rec = {
                "object_pose_json": json.dumps([
                    float(row.get("object_pose_x", "nan")),
                    float(row.get("object_pose_y", "nan")),
                    float(row.get("object_pose_z", "nan")),
                ]),
                "target_pose_json": json.dumps([
                    float(row.get("target_pose_x", "nan")),
                    float(row.get("target_pose_y", "nan")),
                    float(row.get("target_pose_z", "nan")),
                ]),
                "object_to_target_distance": row.get("object_to_target_distance", "nan"),
                "object_eef_distance": row.get("object_eef_distance", "nan"),
                "gripper_command": row.get("raw_gripper", row.get("gripper_command", "nan")),
                "eef_x": row.get("eef_x", "nan"),
                "eef_y": row.get("eef_y", "nan"),
                "eef_z": row.get("eef_z", "nan"),
                "step": i,
            }
            traj_recs.append(rec)

        try:
            teacher_results = teacher.label_trajectory(traj_recs)
            corridor_phases = {"stable_carry", "pre_place_unsupported"}
            has_corridor = any(r.get("phase", "") in corridor_phases for r in teacher_results)
            if has_corridor:
                n_corridor += 1
        except Exception:
            teacher_results = [{"phase": "abstain_unsupported"} for _ in trace]

        for i, tr in enumerate(teacher_results):
            phase = tr.get("phase", "abstain_unsupported")
            trace[i]["teacher_phase"] = phase
            trace[i]["teacher_sc5_corridor_active"] = str(phase in corridor_phases)
            trace[i]["split"] = split
            trace[i]["is_held_out"] = "False"
            trace[i]["episode_id"] = ep
            rows.append(trace[i])

out_path = os.path.join(REPO, "tables/upstream_sc5_train_val_cal_labeled.csv")
keys = list(rows[0].keys())
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

print(f"Episodes: {n_eps}  Corridor: {n_corridor}")
print(f"Rows: {len(rows)}")
print(f"Saved: {out_path}")
print(f"has teacher_phase: {'teacher_phase' in keys}")
print(f"has teacher_sc5_corridor_active: {'teacher_sc5_corridor_active' in keys}")
print(f"has split: {'split' in keys}")
