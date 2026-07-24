#!/usr/bin/env python3
"""P4 v2: Privileged Teacher labeling with real target positions.

Fixes from P4_RUN_1 audit:
- Target pose resolved from LIBERO BDDL + MuJoCo sim state (not default-zero)
- Anchor key: anchor["anchor"] (not anchor["anchor_candidate"])
- teacher_valid=True implies anchor>=0 (assertion)
- Uses frozen C16 Teacher config
- CPU-only after initial env queries
"""
import os, sys, json, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

TEACHER_CONFIG_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json"
CORPUS_ROOT = REPO / "evidence/m1c/object_clean_corpus"
OUT_BASE = REPO / "evidence/m1c/object_teacher_labels_v2"

HARD_NEGATIVE_CATEGORIES = [
    "close_no_grasp", "lift_no_carry", "pseudo_carry", "drop_after_lift",
    "regrasp_or_recovery", "closed_gripper_no_object_follow",
    "unsupported_phase", "no_stable_carry", "other_clean_negative",
]


def sha256_file(p):
    if isinstance(p, str): p = Path(p)
    if not p.exists(): return "MISSING"
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_teacher_config():
    tc_raw = json.load(open(TEACHER_CONFIG_PATH))
    from gripper_attack.v2_privileged_teacher import TeacherConfig
    tc = TeacherConfig()
    tc.version = tc_raw["version"]
    tc.calibrated_from = tc_raw["calibrated_from"]
    tc.guard = tc_raw["guard"]
    tc.K = tc_raw["K"]
    for k, v in tc_raw["thresholds"].items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    return tc


def resolve_target_position(task_idx, state_id):
    """Resolve target body position from LIBERO BDDL + MuJoCo sim state.
    Returns (target_x, target_y, target_z) or None if unresolvable."""
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait

    try:
        bm = benchmark.get_benchmark_dict()
        suite = bm["libero_object"]()
        task_obj = suite.get_task(task_idx)
        init_states = suite.get_task_init_states(task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

        env, _ = build_v4_exact_env(bddl, 0, 400, 10)
        available = set(env.sim.model.body_names)

        basket_body = None
        for line in open(bddl).read().split('\n'):
            line = line.strip()
            if line and not line.startswith('(:') and ' - ' in line:
                parts = line.split(' - ')
                main_name = parts[0].strip() + "_main"
                if parts[1].strip() in ['basket', 'bin']:
                    if main_name in available:
                        basket_body = main_name
                        break

        if basket_body:
            env.set_init_state(init_states[state_id])
            env, _ = apply_dummy_wait(env, None, 10)
            bid = env.sim.model.body_name2id(basket_body)
            target_pos = np.array(env.sim.data.body_xpos[bid])
            env.close()
            return float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    except Exception:
        pass
    return None


_target_cache = {}

def get_target(task_idx, state_id):
    key = (task_idx, state_id)
    if key not in _target_cache:
        _target_cache[key] = resolve_target_position(task_idx, state_id)
    return _target_cache[key]


def build_records_from_telemetry(tel_path, task_idx, state_id):
    """Convert step_telemetry.csv to privileged Teacher input records."""
    if isinstance(tel_path, str) and '\n' in tel_path:
        rows = list(csv.DictReader(tel_path.splitlines()))
    else:
        rows = list(csv.DictReader(open(tel_path)))

    target_pos = get_target(task_idx, state_id)
    if target_pos is None:
        return None  # fail-closed: cannot label without target

    tx, ty, tz = target_pos
    records = []
    prev_eef = None
    for t, r in enumerate(rows):
        obj_x = float(r.get("obj_x", 0)); obj_y = float(r.get("obj_y", 0)); obj_z = float(r.get("obj_z", 0))
        eef_x = float(r.get("eef_x", 0)); eef_y = float(r.get("eef_y", 0)); eef_z = float(r.get("eef_z", 0))
        obj_to_target = float(np.linalg.norm(np.array([obj_x, obj_y, obj_z]) - np.array([tx, ty, tz])))
        eef_to_obj = float(np.linalg.norm(np.array([obj_x, obj_y, obj_z]) - np.array([eef_x, eef_y, eef_z])))
        qpos_sum = float(r.get("qpos_sum", 0))
        raw_grip = float(r.get("raw_gripper", 0))
        # Use feature gripper_opening_proxy if available and non-empty, else qpos_sum
        f_grip = r.get("f_gripper_opening_proxy", "")
        if f_grip and f_grip != "":
            grip_open = float(f_grip)
        else:
            grip_open = qpos_sum

        rec = {
            "step_idx": t, "policy_step_idx": t, "phase": "policy",
            "teacher_privileged_state_available": True,
            "object_pose_json": json.dumps([obj_x, obj_y, obj_z]),
            "target_pose_json": json.dumps([tx, ty, tz]),
            "object_to_target_distance": obj_to_target,
            "object_eef_distance": eef_to_obj,
            "gripper_qpos": qpos_sum,
            "gripper_width": grip_open,
            "gripper_opening_proxy": grip_open,
            "gripper_command": raw_grip,
            "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
            "eef_vx": float("nan") if prev_eef is None else eef_x - prev_eef[0],
            "eef_vy": float("nan") if prev_eef is None else eef_y - prev_eef[1],
            "eef_vz": float("nan") if prev_eef is None else eef_z - prev_eef[2],
        }
        records.append(rec)
        prev_eef = np.array([eef_x, eef_y, eef_z])
    return records


def label_one(cell_dir, tc, labeler_sha, task_idx, state_id):
    """Label a single cell with real target positions."""
    tel = cell_dir / "step_telemetry.csv"
    if not tel.exists():
        return None, "missing_telemetry"

    records = build_records_from_telemetry(tel, task_idx, state_id)
    if records is None:
        return None, "target_unresolvable"

    from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, find_sc5_anchor_v2
    try:
        teacher = V2PrivilegedTeacher(config=tc)
        labels = teacher.label_trajectory(records)
        anchor = find_sc5_anchor_v2(labels, K=tc.K, guard=tc.guard)
    except Exception as e:
        return None, f"teacher_error:{e}"

    stable_carry = anchor.get("stable_carry_start", -1) >= 0
    teacher_valid = anchor.get("valid", False)
    teacher_anchor = anchor.get("anchor", -1)  # CORRECTED: was "anchor_candidate"
    k10_reason = anchor.get("reason", "")
    release_step = -1
    for l in reversed(labels):
        if l.get("phase") == "release_safe":
            release_step = l.get("step_idx", -1)
            break

    # Invariant: teacher_valid implies anchor >= 0
    if teacher_valid:
        assert teacher_anchor >= 0, f"teacher_valid=True but anchor={teacher_anchor}"

    if not stable_carry:
        phases = [l.get("phase","") for l in labels]
        if "grasp_close" in phases and "first_lift" not in phases:
            hard_cat = "close_no_grasp"
        elif "first_lift" in phases and "stable_carry" not in phases:
            hard_cat = "drop_after_lift" if "release_safe" in phases else "lift_no_carry"
        elif "stable_grasp" in phases and "stable_carry" not in phases:
            hard_cat = "pseudo_carry"
        else:
            hard_cat = "no_stable_carry"
    else:
        hard_cat = "other"

    return {
        "stable_carry_present": stable_carry,
        "teacher_valid": teacher_valid,
        "teacher_anchor": teacher_anchor,
        "release_step": release_step,
        "full_k10_valid": teacher_valid,
        "k10_invalid_reason": k10_reason,
        "no_corridor": not teacher_valid,
        "hard_negative_category": hard_cat,
        "teacher_config_sha256": sha256_file(TEACHER_CONFIG_PATH),
        "labeler_sha256": labeler_sha,
        "label_error": "",
    }, None


def main():
    ap = argparse.ArgumentParser(description="P4 Teacher Labeling v2")
    ap.add_argument("--gpu", type=int, default=7, help="GPU for MuJoCo env (brief, CPU mostly)")
    ap.add_argument("--output", default=str(OUT_BASE))
    ap.add_argument("--corpus", default=str(CORPUS_ROOT))
    ap.add_argument("--pools", default="train,validation")
    ap.add_argument("--max-cells", type=int, default=0, help="Limit cells for smoke")
    ap.add_argument("--smoke", action="store_true", help="Smoke: 2 cells only")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"

    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    labeler_sha = sha256_file(__file__)
    tc = load_teacher_config()

    pools = [p.strip() for p in args.pools.split(",")]
    cells = []
    for pool in pools:
        pool_dir = corpus / pool
        if not pool_dir.exists(): continue
        for cell_dir in sorted(pool_dir.iterdir()):
            if not cell_dir.is_dir(): continue
            try:
                parts = cell_dir.name.split("_")
                task = int(parts[0].replace("task",""))
                state = int(parts[1].replace("state",""))
            except (ValueError, IndexError): continue
            cells.append({"pool": pool, "task": task, "state": state, "path": cell_dir})

    if args.smoke:
        cells = cells[:2]
    elif args.max_cells > 0:
        cells = cells[:args.max_cells]

    print(f"P4 v2 Labeler: {len(cells)} cells  Teacher={tc.version} K={tc.K} guard={tc.guard}")

    # Pre-resolve targets
    target_failures = 0
    resolved = 0
    for cell in cells:
        tgt = get_target(cell["task"], cell["state"])
        if tgt is None:
            target_failures += 1
            print(f"  TARGET_UNRESOLVABLE: task={cell['task']} state={cell['state']}")
        else:
            resolved += 1
    print(f"  Targets: {resolved} resolved, {target_failures} unresolvable")
    if target_failures > 0:
        print(f"  FAIL-CLOSED: cannot label {target_failures} cells without target position")
        sys.exit(1)

    rows = []
    errors = []
    for i, cell in enumerate(cells):
        print(f"[{i+1}/{len(cells)}] {cell['pool']}/task{cell['task']}_state{cell['state']} ...", end=" ", flush=True)
        result, err = label_one(cell["path"], tc, labeler_sha, cell["task"], cell["state"])
        if err:
            print(f"ERROR: {err}")
            errors.append({**cell, "error": err})
            continue
        result.update(cell)
        ep = cell["path"] / "episode_summary.json"
        if ep.exists():
            s = json.load(open(ep))
            result["task_success"] = s.get("task_success", None)
            result["n_steps"] = s.get("n_steps", -1)
        else:
            result["task_success"] = None; result["n_steps"] = -1
        rows.append(result)
        tag = "valid" if result["teacher_valid"] else "no_corr"
        print(f"{tag} anchor={result['teacher_anchor']}")

    if rows:
        fieldnames = ["pool","task","state","stable_carry_present","teacher_valid",
                      "teacher_anchor","release_step","full_k10_valid","k10_invalid_reason",
                      "no_corridor","hard_negative_category","task_success","n_steps",
                      "teacher_config_sha256","labeler_sha256","label_error"]
        with open(out_base / "per_cell_teacher_labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    tv_by_pool = defaultdict(lambda: defaultdict(int))
    nc_by_pool = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["teacher_valid"]: tv_by_pool[r["pool"]][r["task"]] += 1
        else: nc_by_pool[r["pool"]][r["task"]] += 1
    for pool_name in pools:
        print(f"\n{pool_name}: teacher_valid={sum(tv_by_pool[pool_name].values())}, no_corridor={sum(nc_by_pool[pool_name].values())}")

    if rows:
        with open(out_base / "denominator_check.json", "w") as f:
            json.dump({
                "train_tv": sum(tv_by_pool["train"].values()),
                "train_nc": sum(nc_by_pool["train"].values()),
                "val_tv": sum(tv_by_pool["validation"].values()),
                "val_nc": sum(nc_by_pool["validation"].values()),
            }, f, indent=2)

    manifest = {
        "gate": "P4_TEACHER_LABELING_V2",
        "labeler_sha256": labeler_sha,
        "teacher_config_sha256": sha256_file(TEACHER_CONFIG_PATH),
        "n_cells": len(cells), "n_errors": len(errors),
        "target_resolution": f"{resolved}/{len(cells)} resolved",
    }
    with open(out_base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if errors:
        print(f"\n  ERRORS: {len(errors)}"); sys.exit(1)
    print(f"\n  P4 v2 COMPLETE: {out_base}")


if __name__ == "__main__":
    main()
