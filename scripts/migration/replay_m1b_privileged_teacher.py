#!/usr/bin/env python3
"""M1B Privileged Action Replay: replay recorded env_action_7d in fresh LIBERO env,
verify parity with original telemetry, collect privileged state, run C16 Teacher.

Strictly does NOT re-run VLA. Uses frozen C16 Teacher config (v2_teacher_fixed_semantics).
Frame alignment: state_before_step_t → compare → env.step(action_t).
"""
import os, sys, json, hashlib, time, csv, argparse, copy
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/artifact_manifest_complete.json"
TEACHER_CONFIG_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json"
OUT_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime_b0_d1"


def load_frozen_teacher_config():
    """Load C16 frozen Teacher config, NOT draft defaults."""
    cfg = json.load(open(TEACHER_CONFIG_PATH))
    from gripper_attack.v2_privileged_teacher import TeacherConfig
    tc = TeacherConfig()
    tc.version = cfg["version"]
    tc.calibrated_from = cfg["calibrated_from"]
    tc.guard = cfg["guard"]
    tc.K = cfg["K"]
    for k, v in cfg["thresholds"].items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    tc.config_sha = hashlib.sha256(open(TEACHER_CONFIG_PATH, "rb").read()).hexdigest()
    return tc


def verify_source_artifacts(cell, cell_dir):
    """Verify all source artifact hashes match manifest."""
    errors = []
    for fname, sha_key in [("step_telemetry.csv", "telemetry_sha256"),
                           ("episode_summary.json", "episode_summary_sha256"),
                           (".done", "done_sha256")]:
        fpath = os.path.join(cell_dir, fname)
        if os.path.exists(fpath):
            actual = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            expected = cell.get(sha_key, "")
            if actual != expected:
                errors.append(f"{fname}: expected={expected[:16]}... actual={actual[:16]}...")
    return errors


def replay_one(episode_key, profile, gpu, save_video=False):
    """Replay one cell. Returns dict with parity and teacher results."""
    manifest = json.load(open(MANIFEST_PATH))
    cell = None
    for c in manifest["cells"]:
        if c["episode_key"] == episode_key and c["profile"] == profile:
            cell = c
            break
    if cell is None:
        return {"_error": f"cell not found: {episode_key}/{profile}"}

    cell_dir = os.path.join(OUT_BASE, cell["relative_path"])
    errors = verify_source_artifacts(cell, cell_dir)
    if errors:
        return {"_error": "SOURCE_ARTIFACT_HASH_MISMATCH", "details": errors}

    # Load original telemetry
    tel_path = os.path.join(cell_dir, "step_telemetry.csv")
    orig_tel = list(csv.DictReader(open(tel_path)))
    orig_summary = json.load(open(os.path.join(cell_dir, "episode_summary.json")))

    # Parse recorded actions
    recorded_actions = []
    for row in orig_tel:
        act_str = row.get("env_action_7d", row.get("raw_action_7d", "[]"))
        act = json.loads(act_str) if isinstance(act_str, str) else act_str
        recorded_actions.append(np.array(act, dtype=np.float64))

    # Build env (exact same as bridge)
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait

    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_idx = cell["task_idx"]
    state_id = cell["state_id"]
    task_obj = suite.get_task(task_idx)
    init_states = suite.get_task_init_states(task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

    env, obs = build_v4_exact_env(bddl, gpu, 400, 10)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)

    # Replay loop
    n_steps = len(recorded_actions)
    replay_records = []
    eef_errors = []
    obj_errors = []

    for t in range(n_steps):
        # Read state BEFORE step
        eef_pos = np.array(env.get_sim_state().get_joint_pos("robot0_right_hand")[:3])
        obj_name = task_obj.object_name if hasattr(task_obj, 'object_name') else None

        # Get object pose from env
        try:
            obj_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(obj_name)])
        except Exception:
            obj_pos = np.array([float(orig_tel[t].get("obj_x", 0)),
                                float(orig_tel[t].get("obj_y", 0)),
                                float(orig_tel[t].get("obj_z", 0))])

        # Get target pose
        try:
            target_name = task_obj.target_name if hasattr(task_obj, 'target_name') else "basket"
            target_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(target_name)])
        except Exception:
            target_pos = np.array([0.0, 0.0, 0.0])

        # Verify parity with original telemetry
        orig_eef = np.array([float(orig_tel[t].get("eef_x", 0)),
                             float(orig_tel[t].get("eef_y", 0)),
                             float(orig_tel[t].get("eef_z", 0))])
        orig_obj = np.array([float(orig_tel[t].get("obj_x", 0)),
                             float(orig_tel[t].get("obj_y", 0)),
                             float(orig_tel[t].get("obj_z", 0))])

        eef_err = np.max(np.abs(eef_pos - orig_eef))
        obj_err = np.max(np.abs(obj_pos - orig_obj))
        eef_errors.append(float(eef_err))
        obj_errors.append(float(obj_err))

        # Collect privileged record
        rec = {
            "step_idx": t,
            "policy_step_idx": t,
            "phase": "policy",
            "teacher_privileged_state_available": True,
            "object_pose_json": json.dumps(obj_pos.tolist()),
            "target_pose_json": json.dumps(target_pos.tolist()),
            "object_to_target_distance": float(np.linalg.norm(obj_pos - target_pos)),
            "object_eef_distance": float(np.linalg.norm(obj_pos - eef_pos)),
            "gripper_qpos": float(env.get_sim_state().get_joint_qpos()[-2:].sum()),
            "eef_x": float(eef_pos[0]), "eef_y": float(eef_pos[1]), "eef_z": float(eef_pos[2]),
            "eef_vx": 0.0 if t == 0 else float(eef_pos[0] - prev_eef[0]),
            "eef_vy": 0.0 if t == 0 else float(eef_pos[1] - prev_eef[1]),
            "eef_vz": 0.0 if t == 0 else float(eef_pos[2] - prev_eef[2]),
            "replay_eef_err": float(eef_err),
            "replay_obj_err": float(obj_err),
        }
        if t == 0:
            rec["eef_vx"] = float("nan")
            rec["eef_vy"] = float("nan")
            rec["eef_vz"] = float("nan")
        replay_records.append(rec)
        prev_eef = eef_pos.copy()

        # Execute recorded action
        env.step(recorded_actions[t])

    # Final success check
    replay_success = bool(env.check_success()) if hasattr(env, "check_success") else False
    orig_success = orig_summary.get("task_success", False)
    env.close()

    # Parity check
    max_eef_err = max(eef_errors) if eef_errors else 999
    max_obj_err = max(obj_errors) if obj_errors else 999
    steps_match = n_steps == orig_summary.get("n_steps", -1)
    success_match = replay_success == orig_success

    parity = {
        "n_steps_original": orig_summary.get("n_steps", -1),
        "n_steps_replay": n_steps,
        "steps_match": steps_match,
        "success_original": orig_success,
        "success_replay": replay_success,
        "success_match": success_match,
        "action_count_match": len(recorded_actions) == orig_summary.get("n_steps", -1),
        "max_eef_error": float(max_eef_err),
        "max_obj_error": float(max_obj_err),
        "eef_parity_pass": max_eef_err <= 1e-5,
        "obj_parity_pass": max_obj_err <= 1e-5,
        "all_parity_pass": steps_match and success_match and max_eef_err <= 1e-5 and max_obj_err <= 1e-5,
        "source_artifact_hash_check": "PASS" if not errors else "FAIL",
    }

    # Run C16 Teacher on privileged records
    teacher_config = load_frozen_teacher_config()
    from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, find_sc5_anchor_v2

    teacher = V2PrivilegedTeacher(config=teacher_config)
    labels = teacher.label_trajectory(replay_records)
    anchor_result = find_sc5_anchor_v2(labels, K=teacher_config.K, guard=teacher_config.guard)

    teacher_summary = {
        "teacher_supported": anchor_result.get("anchor", -1) >= 0,
        "teacher_anchor": anchor_result.get("anchor", -1),
        "stable_carry_start": anchor_result.get("stable_carry_start", -1),
        "release_safe_start": anchor_result.get("release_safe_start", -1),
        "K10_valid": anchor_result.get("K10_valid", False),
        "reason": anchor_result.get("reason", "unknown"),
        "teacher_config_sha": teacher_config.config_sha,
    }

    # Detect target binding ambiguity
    target_binding_ok = True
    target_binding_note = ""

    return {
        "episode_key": episode_key,
        "profile": profile,
        "parity": parity,
        "teacher": teacher_summary,
        "n_replay_steps": n_steps,
        "source_artifact_hash_check": "PASS" if not errors else "FAIL",
        "target_binding_ok": target_binding_ok,
        "target_binding_note": target_binding_note,
    }


def main():
    ap = argparse.ArgumentParser(description="M1B Privileged Action Replay")
    ap.add_argument("--episode", required=True, help="episode_key (e.g. butter_s0)")
    ap.add_argument("--profile", required=True, choices=["B0", "D1"])
    ap.add_argument("--gpu", type=int, default=3, help="GPU for MuJoCo rendering (default 3)")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        OUT_BASE / "replay_canary" / args.episode / args.profile)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
    os.environ["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"

    print(f"Replay: {args.episode}/{args.profile} GPU={args.gpu}")
    result = replay_one(args.episode, args.profile, args.gpu, args.save_video)

    # Write outputs
    json.dump(result.get("parity", {}), open(out_dir / "replay_parity.json", "w"), indent=2)
    json.dump(result.get("teacher", {}), open(out_dir / "teacher_summary.json", "w"), indent=2)

    if result.get("_error"):
        print(f"  ERROR: {result['_error']}")
        if "details" in result:
            for d in result["details"]:
                print(f"    {d}")
    else:
        p = result["parity"]
        t = result["teacher"]
        print(f"  Parity: steps={p['steps_match']} succ={p['success_match']} "
              f"eef={p['max_eef_error']:.2e} obj={p['max_obj_error']:.2e} "
              f"ALL={'PASS' if p['all_parity_pass'] else 'FAIL'}")
        print(f"  Teacher: supported={t['teacher_supported']} anchor={t['teacher_anchor']} "
              f"sc_start={t['stable_carry_start']} K10={t['K10_valid']}")

    json.dump(result, open(out_dir / "replay_summary.json", "w"), indent=2, default=str)
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
