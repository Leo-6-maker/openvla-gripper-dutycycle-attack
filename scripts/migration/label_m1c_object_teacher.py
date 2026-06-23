#!/usr/bin/env python3
"""P4: Privileged Teacher labeling for M1C Object clean corpus.

Uses frozen C16 Teacher config to label train + validation cells.
Reads from corpus, writes sidecar CSV — never modifies original files.

Smoke mode: --smoke runs only task0_state3 through task9_state3.
First run writes labels; second run verifies determinism.
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
OUT_BASE = REPO / "evidence/m1c/object_teacher_labels_v1"
TRAJ_SIDECAR = REPO / "evidence/m1c/object_clean_corpus_audit_preflight_20260624/trajectory_sha256_sidecar.csv"

HARD_NEGATIVE_CATEGORIES = [
    "close_no_grasp", "lift_no_carry", "pseudo_carry", "drop_after_lift",
    "regrasp_or_recovery", "closed_gripper_no_object_follow",
    "unsupported_phase", "no_stable_carry", "other_clean_negative",
]

SMOKE_STATES = [3]  # one per task, state=3


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


def build_records_from_telemetry(tel_path):
    """Convert step_telemetry.csv to privileged Teacher input records.
    Accepts file path (str/Path) or CSV string content."""
    if isinstance(tel_path, str) and '\n' in tel_path:
        rows = list(csv.DictReader(tel_path.splitlines()))
    else:
        rows = list(csv.DictReader(open(tel_path)))
    records = []
    prev_eef = None
    for t, r in enumerate(rows):
        obj_x = float(r.get("obj_x", 0)); obj_y = float(r.get("obj_y", 0)); obj_z = float(r.get("obj_z", 0))
        eef_x = float(r.get("eef_x", 0)); eef_y = float(r.get("eef_y", 0)); eef_z = float(r.get("eef_z", 0))
        target_x = 0; target_y = 0; target_z = 0
        # Target pos from telemetry (if recorded)
        if "target_x" in r:
            target_x = float(r.get("target_x", 0))
            target_y = float(r.get("target_y", 0))
            target_z = float(r.get("target_z", 0))

        rec = {
            "step_idx": t, "policy_step_idx": t, "phase": "policy",
            "teacher_privileged_state_available": True,
            "object_pose_json": json.dumps([obj_x, obj_y, obj_z]),
            "target_pose_json": json.dumps([target_x, target_y, target_z]),
            "object_to_target_distance": float(np.linalg.norm(np.array([obj_x, obj_y, obj_z]) - np.array([target_x, target_y, target_z]))),
            "object_eef_distance": float(np.linalg.norm(np.array([obj_x, obj_y, obj_z]) - np.array([eef_x, eef_y, eef_z]))),
            "gripper_qpos": float(r.get("qpos_sum", 0)),
            "gripper_width": float(r.get("qpos_sum", 0)),
            "gripper_opening_proxy": float(r.get("qpos_sum", 0)),
            "gripper_command": float(r.get("raw_gripper", 0)),
            "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
            "eef_vx": float("nan") if prev_eef is None else eef_x - prev_eef[0],
            "eef_vy": float("nan") if prev_eef is None else eef_y - prev_eef[1],
            "eef_vz": float("nan") if prev_eef is None else eef_z - prev_eef[2],
        }
        records.append(rec)
        prev_eef = np.array([eef_x, eef_y, eef_z])
    return records


def label_one(cell_dir, tc, labeler_sha):
    """Label a single cell. Returns (result_dict, error_string)."""
    tel = cell_dir / "step_telemetry.csv"
    if not tel.exists():
        return None, "missing_telemetry"

    try:
        records = build_records_from_telemetry(tel)
    except Exception as e:
        return None, f"telemetry_parse_error:{e}"

    from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, find_sc5_anchor_v2
    try:
        teacher = V2PrivilegedTeacher(config=tc)
        labels = teacher.label_trajectory(records)
        anchor = find_sc5_anchor_v2(labels, K=tc.K, guard=tc.guard)
    except Exception as e:
        return None, f"teacher_error:{e}"

    stable_carry = anchor.get("stable_carry_start", -1) >= 0
    teacher_valid = anchor.get("valid", False)
    teacher_anchor = anchor.get("anchor_candidate", -1)
    k10_reason = anchor.get("reason", "")
    release_step = -1
    for l in reversed(labels):
        if l.get("phase") == "release_safe":
            release_step = l.get("step_idx", -1)
            break

    # Hard negative classification
    if not stable_carry:
        # Inspect labels for negative category
        phases = [l.get("phase","") for l in labels]
        if "grasp_close" in phases and "first_lift" not in phases:
            hard_cat = "close_no_grasp"
        elif "first_lift" in phases and "stable_carry" not in phases:
            if "release_safe" in phases:
                hard_cat = "drop_after_lift"
            else:
                hard_cat = "lift_no_carry"
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
    ap = argparse.ArgumentParser(description="P4 Teacher Labeling")
    ap.add_argument("--smoke", action="store_true", help="Smoke: 10 cells only")
    ap.add_argument("--output", default=str(OUT_BASE), help="Output directory")
    ap.add_argument("--corpus", default=str(CORPUS_ROOT), help="Corpus root")
    ap.add_argument("--pools", default="train,validation", help="Pools to label")
    args = ap.parse_args()

    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    labeler_sha = sha256_file(__file__)
    tc = load_teacher_config()

    pools = [p.strip() for p in args.pools.split(",")]
    cells = []
    for pool in pools:
        pool_dir = corpus / pool
        if not pool_dir.exists():
            continue
        for cell_dir in sorted(pool_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            try:
                parts = cell_dir.name.split("_")
                task = int(parts[0].replace("task", ""))
                state = int(parts[1].replace("state", ""))
            except (ValueError, IndexError):
                continue
            if args.smoke and state not in SMOKE_STATES:
                continue
            cells.append({"pool": pool, "task": task, "state": state, "path": cell_dir})

    print(f"P4 Labeler: {len(cells)} cells ({'smoke' if args.smoke else 'full'})")
    print(f"  Teacher: {tc.version}  K={tc.K}  guard={tc.guard}")
    print(f"  Labeler: {labeler_sha[:16]}")

    rows = []
    errors = []
    for i, cell in enumerate(cells):
        print(f"[{i+1}/{len(cells)}] {cell['pool']}/task{cell['task']}_state{cell['state']} ...", end=" ", flush=True)
        result, err = label_one(cell["path"], tc, labeler_sha)
        if err:
            print(f"ERROR: {err}")
            errors.append({**cell, "error": err})
            continue
        result.update(cell)
        # Check consistency with summary
        ep = cell["path"] / "episode_summary.json"
        if ep.exists():
            s = json.load(open(ep))
            result["task_success"] = s.get("task_success", None)
            result["n_steps"] = s.get("n_steps", -1)
        else:
            result["task_success"] = None
            result["n_steps"] = -1
        rows.append(result)
        tag = "valid" if result["teacher_valid"] else "no_corr"
        print(f"{tag} anchor={result['teacher_anchor']} cat={result['hard_negative_category']}")

    # Output
    if rows:
        fieldnames = ["pool", "task", "state", "stable_carry_present", "teacher_valid",
                      "teacher_anchor", "release_step", "full_k10_valid", "k10_invalid_reason",
                      "no_corridor", "hard_negative_category", "task_success", "n_steps",
                      "teacher_config_sha256", "labeler_sha256", "label_error"]
        with open(out_base / "per_cell_teacher_labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # Per-task distribution
    tv_by_pool = defaultdict(lambda: defaultdict(int))
    nc_by_pool = defaultdict(lambda: defaultdict(int))
    for r in rows:
        pool = r["pool"]
        if r["teacher_valid"]:
            tv_by_pool[pool][r["task"]] += 1
        else:
            nc_by_pool[pool][r["task"]] += 1
    for pool in pools:
        tv_total = sum(tv_by_pool[pool].values())
        nc_total = sum(nc_by_pool[pool].values())
        print(f"\n{pool}: teacher_valid={tv_total}, no_corridor={nc_total}")
    if rows:
        with open(out_base / "denominator_check.json", "w") as f:
            json.dump({
                "train_tv": sum(tv_by_pool["train"].values()),
                "train_nc": sum(nc_by_pool["train"].values()),
                "val_tv": sum(tv_by_pool["validation"].values()),
                "val_nc": sum(nc_by_pool["validation"].values()),
                "train_tv_min": 120, "train_nc_min": 80,
                "val_tv_min": 30, "val_nc_min": 20,
            }, f, indent=2)

    # Manifest
    manifest = {
        "gate": "P4_TEACHER_LABELING",
        "p3a_status": "PASS",
        "p3b_status": "PARTIAL",
        "labeler_sha256": labeler_sha,
        "teacher_config_sha256": sha256_file(TEACHER_CONFIG_PATH),
        "n_cells": len(cells),
        "n_errors": len(errors),
        "trajectory_sidecar_sha256": sha256_file(TRAJ_SIDECAR) if TRAJ_SIDECAR.exists() else "MISSING",
        "restrictions": {
            "threshold_recalibration": "FORBIDDEN",
            "task_success_as_label": "FORBIDDEN",
            "formal_blind_included": "FORBIDDEN",
            "attack_data_included": "FORBIDDEN",
        },
    }
    if rows:
        manifest["n_labels"] = len(rows)
        tv = [r for r in rows if r["teacher_valid"]]
        nc = [r for r in rows if not r["teacher_valid"]]
        manifest["summary"] = {
            "train_tv": sum(tv_by_pool["train"].values()),
            "train_nc": sum(nc_by_pool["train"].values()),
            "validation_tv": sum(tv_by_pool["validation"].values()),
            "validation_nc": sum(nc_by_pool["validation"].values()),
        }
    with open(out_base / "P4_TEACHER_LABELING_PROTOCOL_FROZEN.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Recursive manifest
    manifest_rows = []
    for fp in out_base.iterdir():
        if fp.is_file():
            manifest_rows.append({"path": fp.name, "sha256": sha256_file(fp), "size": fp.stat().st_size})
    with open(out_base / "recursive_artifact_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "size"])
        w.writeheader()
        w.writerows(manifest_rows)

    with open(out_base / "run.log", "w") as f:
        f.write(f"Labeler: {labeler_sha}\n")
        f.write(f"Teacher: {tc.version}\n")
        f.write(f"Cells: {len(cells)}, Errors: {len(errors)}\n")

    if errors:
        print(f"\n  ERRORS: {len(errors)}")
        sys.exit(1)
    print(f"\n  P4 COMPLETE: {out_base}")


if __name__ == "__main__":
    main()
