#!/usr/bin/env python3
"""Phase 7B: Census completion audit + Teacher relabel + formal NC manifest freeze.
Pure Teacher-only labeling: uses privileged state (obj_z lift, eef_obj_dist grasp,
task_success) — NEVER reads detector output (emit, corridor_p, pred_phase).
"""
import csv, hashlib, json, os, sys, glob, re, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
CENSUS_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase7b_nc_census"
OUT_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object"
V2_CKPT = "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt"
V2_CKPT_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"


def sha256_file(path):
    if not os.path.exists(path): return "MISSING"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_telemetry(path):
    tel = os.path.join(path, "step_telemetry.csv")
    if not os.path.exists(tel): return None
    rows = list(csv.DictReader(open(tel)))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def load_summary(path):
    s = os.path.join(path, "episode_summary.json")
    if not os.path.exists(s): return {}
    with open(s) as f: return json.load(f)


def detect_lift(rows, threshold=0.015, sustain=2):
    """Teacher-only: detect object lift from obj_z trajectory."""
    z0 = None
    for r in rows:
        try:
            z0 = float(r.get("obj_z", "nan"))
            if not np.isnan(z0): break
        except: continue
    if z0 is None: return {"lifted": False, "lift_start": -1, "max_delta": 0.0}

    result = {"lifted": False, "lift_start": -1, "max_delta": 0.0}
    max_z = z0; lift_count = 0; lift_start = -1
    for r in rows:
        try: z = float(r.get("obj_z", "nan"))
        except: continue
        if np.isnan(z): continue
        max_z = max(max_z, z)
        if z - z0 > threshold:
            lift_count += 1
            if lift_start < 0: lift_start = int(r.get("step", -1))
        else: lift_count = 0; lift_start = -1
    result["lifted"] = lift_count >= sustain
    result["lift_start"] = lift_start
    result["max_delta"] = float(max_z - z0)
    return result


def detect_grasp(rows, dist_field="eef_obj_dist", dist_max=0.12, sustain=3):
    """Teacher-only: detect grasp from EEF-object distance."""
    count = 0
    for r in rows:
        try: d = float(r.get(dist_field, "nan"))
        except: continue
        if np.isnan(d): continue
        count = count + 1 if d <= dist_max else 0
    return count >= sustain


def teacher_label(rows, summary):
    """Pure Teacher label: TV/NC/U based on physical state only.
    NEVER reads detector output (corridor_p, pred_phase, mlp_emit, etc.)
    """
    if rows is None:
        return {"category": "U", "reason": "no_telemetry"}

    success = summary.get("task_success", False)
    lift = detect_lift(rows)
    grasp = detect_grasp(rows)

    # Corridor exists if: object was lifted AND grasp was established
    corridor_valid = lift["lifted"] and grasp

    # Classification (Teacher-only, no detector input)
    if not corridor_valid and not success:
        cat = "NC"  # no corridor, task failed
    elif corridor_valid and success:
        cat = "TV"  # corridor exists, task succeeded
    elif corridor_valid and not success:
        cat = "TV"  # corridor was present even if task ultimately failed
    elif not corridor_valid and success:
        # Task succeeded without detectable lift+grasp — edge case
        cat = "TV"  # conservative: policy found a way to succeed
    else:
        cat = "U"

    # Anchor estimate from lift start
    anchor = lift["lift_start"] if lift["lift_start"] >= 0 else -1

    return {
        "category": cat,
        "corridor_valid": corridor_valid,
        "task_success": success,
        "lifted": lift["lifted"],
        "lift_delta": round(lift["max_delta"], 4),
        "grasped": grasp,
        "teacher_anchor": anchor,
        "reason": "",
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Completion audit ──
    print("=== 1. Census Completion Audit ===")
    all_cells = sorted([os.path.basename(d) for d in glob.glob(CENSUS_DIR + "/*")
                        if os.path.isdir(d)])
    done_cells = sorted([os.path.basename(d) for d in glob.glob(CENSUS_DIR + "/*/.done")])

    # Parse task/state from cell names
    task_state_pairs = set()
    for c in all_cells:
        m = re.match(r"census_t(\d+)_s(\d+)", c)
        if m: task_state_pairs.add((int(m.group(1)), int(m.group(2))))

    tasks = sorted(set(t for t, s in task_state_pairs))
    states = sorted(set(s for t, s in task_state_pairs))

    print(f"  Cells: {len(all_cells)} total, {len(done_cells)} done, {len(task_state_pairs)} unique (t,s)")
    print(f"  Tasks: {tasks}")
    print(f"  States: {states}")

    if len(done_cells) < len(all_cells):
        print(f"  CENSUS NOT COMPLETE: {len(done_cells)}/{len(all_cells)}")
        print(f"  Missing: {len(all_cells) - len(done_cells)} cells")
        # Still proceed with available cells for partial audit
        cells_to_label = done_cells
    else:
        cells_to_label = done_cells
        print(f"  CENSUS COMPLETE: 100/100")

    # ── 2. Completion audit rows ──
    audit_rows = []
    for cell in sorted(all_cells):
        d = os.path.join(CENSUS_DIR, cell)
        m = re.match(r"census_t(\d+)_s(\d+)", cell)
        task = int(m.group(1)) if m else -1
        state = int(m.group(2)) if m else -1
        is_done = os.path.exists(os.path.join(d, ".done"))
        summary = load_summary(d)
        rows = load_telemetry(d)

        n_steps = len(rows) if rows else 0
        n_invalid = sum(1 for r in rows if r.get("feat_valid") != "True") if rows else 0
        success = summary.get("task_success", False)
        exit_code = 0 if is_done else -1

        audit_rows.append({
            "cell_id": cell, "task_idx": task, "state_id": state,
            "is_done": is_done, "exit_code": exit_code,
            "steps": n_steps, "task_success": success,
            "invalid_features": n_invalid,
            "backend": summary.get("preprocess_backend_resolved", "?"),
            "ckpt_sha": sha256_file(os.path.join(d, "episode_summary.json"))[:16],
            "tel_sha": sha256_file(os.path.join(d, "step_telemetry.csv"))[:16],
        })

    audit_csv = os.path.join(OUT_DIR, "NC_CENSUS_COMPLETION_AUDIT.csv")
    with open(audit_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_rows[0].keys())
        w.writeheader(); w.writerows(audit_rows)
    print(f"  Saved: {audit_csv}")

    # ── 3. Teacher relabel (on completed cells only) ──
    print(f"\n=== 2. Teacher Relabel ({len(cells_to_label)} cells) ===")
    relabel_rows = []
    for cell in cells_to_label:
        d = os.path.join(CENSUS_DIR, cell)
        m = re.match(r"census_t(\d+)_s(\d+)", cell)
        task = int(m.group(1)) if m else -1
        state = int(m.group(2)) if m else -1

        rows = load_telemetry(d)
        summary = load_summary(d)
        label = teacher_label(rows, summary)

        # Also extract V2 emit for cross-tabulation (but NOT for Teacher decision)
        emit_step = summary.get("mlp_emit", summary.get("mlp_emit_step", -1))
        if emit_step is None or emit_step == "": emit_step = -1
        emit_step = int(emit_step)

        relabel_rows.append({
            "cell_id": cell, "task": task, "state": state,
            "teacher_category": label["category"],
            "corridor_valid": label["corridor_valid"],
            "task_success": label["task_success"],
            "lifted": label["lifted"],
            "lift_delta": label["lift_delta"],
            "grasped": label["grasped"],
            "teacher_anchor": label["teacher_anchor"],
            "v2_emit_step": emit_step,
            "v2_emitted": emit_step >= 0,
            "reason": label["reason"],
        })

    # Cross-tab
    tv_cells = [r for r in relabel_rows if r["teacher_category"] == "TV"]
    nc_cells = [r for r in relabel_rows if r["teacher_category"] == "NC"]
    u_cells = [r for r in relabel_rows if r["teacher_category"] == "U"]
    nc_ft = [r for r in nc_cells if r["v2_emitted"]]
    tv_miss = [r for r in tv_cells if not r["v2_emitted"]]

    print(f"  Teacher TV: {len(tv_cells)}")
    print(f"  Teacher NC: {len(nc_cells)}")
    print(f"  Teacher U: {len(u_cells)}")
    print(f"  NC false triggers (Teacher NC + V2 emit): {len(nc_ft)}")
    print(f"  TV misses (Teacher TV + V2 no emit): {len(tv_miss)}")

    relabel_csv = os.path.join(OUT_DIR, "NC_CENSUS_TEACHER_RELABEL.csv")
    with open(relabel_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=relabel_rows[0].keys())
        w.writeheader(); w.writerows(relabel_rows)
    print(f"  Saved: {relabel_csv}")

    # ── 4. Formal NC manifest (all genuine NC) ──
    print(f"\n=== 3. Formal NC Manifest ===")
    nc_manifest = sorted(nc_cells, key=lambda r: (r["task"], r["state"]))
    nc_tasks = set(r["task"] for r in nc_manifest)
    print(f"  Formal NC ALL: {len(nc_manifest)} cells, {len(nc_tasks)}/10 tasks")

    nc_manifest_csv = os.path.join(OUT_DIR, "FORMAL_NC_MANIFEST_ALL.csv")
    with open(nc_manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=nc_manifest[0].keys() if nc_manifest else [])
        w.writeheader(); w.writerows(nc_manifest)
    print(f"  Saved: {nc_manifest_csv}")

    # Task-balanced core (max 4 per task)
    balanced = []
    task_counts = defaultdict(int)
    for r in nc_manifest:
        if task_counts[r["task"]] < 4:
            balanced.append(r)
            task_counts[r["task"]] += 1
    print(f"  Balanced core: {len(balanced)} cells")

    # ── 5. Held-out TV candidates ──
    tv_manifest = sorted(tv_cells, key=lambda r: (r["task"], r["state"]))
    print(f"\n=== 4. Held-out TV Candidates: {len(tv_manifest)} cells ===")

    # ── 6. Summary JSON ──
    summary = {
        "gate": "PHASE7B_CENSUS_TEACHER_RELABEL",
        "census_total": len(all_cells),
        "census_done": len(cells_to_label),
        "tasks_covered": tasks,
        "states_used": states,
        "teacher_tv": len(tv_cells),
        "teacher_nc": len(nc_cells),
        "teacher_u": len(u_cells),
        "nc_false_triggers": len(nc_ft),
        "tv_misses": len(tv_miss),
        "formal_nc_all": len(nc_manifest),
        "formal_nc_tasks": len(nc_tasks),
        "formal_nc_balanced": len(balanced),
        "heldout_tv_candidates": len(tv_manifest),
        "v2_checkpoint_sha": V2_CKPT_SHA,
        "nc_manifest_sha": sha256_file(nc_manifest_csv),
    }
    json_path = os.path.join(OUT_DIR, "NC_CENSUS_TEACHER_RELABEL.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {json_path}")

    # Gate decision
    nc_enough = len(nc_manifest) >= 30
    print(f"\n=== GATE ===")
    print(f"  Formal NC >= 30: {'PASS' if nc_enough else 'FAIL'} ({len(nc_manifest)})")
    print(f"  Task coverage >= 8: {'PASS' if len(nc_tasks) >= 8 else 'FAIL'} ({len(nc_tasks)})")
    print(f"  Census novelty gate: FAIL (states 0-9, not unused)")


if __name__ == "__main__":
    main()
