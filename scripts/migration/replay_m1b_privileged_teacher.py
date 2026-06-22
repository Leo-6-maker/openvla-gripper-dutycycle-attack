#!/usr/bin/env python3
"""M1B Privileged Action Replay — replay recorded env_action_7d, verify parity,
collect privileged state, run frozen C16 Teacher.  Does NOT re-run VLA."""
import os, sys, json, hashlib, time, csv, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/artifact_manifest_complete.json"
TEACHER_CONFIG_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json"
OUT_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime_b0_d1"


def load_frozen_teacher():
    from gripper_attack.v2_privileged_teacher import TeacherConfig
    cfg_raw = json.load(open(TEACHER_CONFIG_PATH))
    tc = TeacherConfig()
    tc.version = cfg_raw["version"]
    tc.calibrated_from = cfg_raw["calibrated_from"]
    tc.guard = cfg_raw["guard"]
    tc.K = cfg_raw["K"]
    for k, v in cfg_raw["thresholds"].items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    tc.config_sha = hashlib.sha256(open(TEACHER_CONFIG_PATH, "rb").read()).hexdigest()
    return tc


def replay_one(episode_key, profile, gpu):
    manifest = json.load(open(MANIFEST_PATH))
    cell = next((c for c in manifest["cells"] if c["episode_key"] == episode_key and c["profile"] == profile), None)
    if cell is None:
        return {"_error": f"not in manifest: {episode_key}/{profile}"}

    cell_dir = os.path.join(OUT_BASE, cell["relative_path"])

    # Verify source hashes
    for fname, sha_key in [("step_telemetry.csv", "telemetry_sha256"),
                           ("episode_summary.json", "episode_summary_sha256")]:
        fpath = os.path.join(cell_dir, fname)
        if os.path.exists(fpath):
            actual = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            if actual != cell.get(sha_key, ""):
                return {"_error": f"SOURCE_HASH_MISMATCH: {fname}"}

    # Load originals
    orig_tel = list(csv.DictReader(open(os.path.join(cell_dir, "step_telemetry.csv"))))
    orig_summary = json.load(open(os.path.join(cell_dir, "episode_summary.json")))
    recorded_actions = [np.array(json.loads(r.get("env_action_7d", r.get("raw_action_7d", "[]")), dtype=np.float64))
                        for r in orig_tel]

    # Build env (same as bridge)
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait

    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_obj = suite.get_task(cell["task_idx"])
    init_states = suite.get_task_init_states(cell["task_idx"])
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)

    env, obs = build_v4_exact_env(bddl, gpu, 400, 10)
    available = set(env.sim.model.body_names)

    # Resolve object & basket body names from BDDL + available MuJoCo bodies
    obj_body = None
    basket_body = None
    for line in open(bddl).read().split('\n'):
        line = line.strip()
        if line and not line.startswith('(:') and ' - ' in line:
            parts = line.split(' - ')
            main_name = parts[0].strip() + "_main"
            if parts[1].strip() in ['basket', 'bin']:
                if main_name in available:
                    basket_body = main_name
            elif not obj_body and main_name in available:
                obj_body = main_name

    if not obj_body:
        env.close()
        return {"_error": "TARGET_BINDING_AMBIGUOUS", "detail": "no object body"}
    if not basket_body:
        env.close()
        return {"_error": "TARGET_BINDING_AMBIGUOUS", "detail": "no basket body"}

    obs = env.set_init_state(init_states[cell["state_id"]])
    env, obs = apply_dummy_wait(env, obs, 10)

    # Replay loop — frame-aligned with bridge
    n = len(recorded_actions)
    records = []
    eef_errs = []
    obj_errs = []
    prev_eef = None
    grip_site_id = env.sim.model.site_name2id("gripper0_grip_site")
    obj_body_id = env.sim.model.body_name2id(obj_body)
    basket_body_id = env.sim.model.body_name2id(basket_body)

    for t in range(n):
        # EEF from grip SITE (same as bridge)
        eef_pos = np.array(env.sim.data.site_xpos[grip_site_id])
        obj_pos = np.array(env.sim.data.body_xpos[obj_body_id])
        target_pos = np.array(env.sim.data.body_xpos[basket_body_id])
        grip_qpos = float(env.sim.data.qpos[-2:].sum())

        # Parity
        orig_eef = np.array([float(orig_tel[t].get("eef_x", 0)),
                             float(orig_tel[t].get("eef_y", 0)),
                             float(orig_tel[t].get("eef_z", 0))])
        orig_obj = np.array([float(orig_tel[t].get("obj_x", 0)),
                             float(orig_tel[t].get("obj_y", 0)),
                             float(orig_tel[t].get("obj_z", 0))])
        eef_errs.append(float(np.max(np.abs(eef_pos - orig_eef))))
        obj_errs.append(float(np.max(np.abs(obj_pos - orig_obj))))

        # Privileged record (matching _extract_state field names)
        rec = {
            "step_idx": t, "policy_step_idx": t, "phase": "policy",
            "teacher_privileged_state_available": True,
            "object_pose_json": json.dumps([float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])]),
            "target_pose_json": json.dumps([float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]),
            "object_to_target_distance": float(np.linalg.norm(obj_pos - target_pos)),
            "object_eef_distance": float(np.linalg.norm(obj_pos - eef_pos)),
            "gripper_qpos": grip_qpos,
            "gripper_width": grip_qpos,
            "gripper_opening_proxy": grip_qpos,
            "gripper_command": float(orig_tel[t].get("raw_gripper", 0)),
            "eef_x": float(eef_pos[0]), "eef_y": float(eef_pos[1]), "eef_z": float(eef_pos[2]),
            "eef_vx": float("nan") if prev_eef is None else float(eef_pos[0] - prev_eef[0]),
            "eef_vy": float("nan") if prev_eef is None else float(eef_pos[1] - prev_eef[1]),
            "eef_vz": float("nan") if prev_eef is None else float(eef_pos[2] - prev_eef[2]),
        }
        records.append(rec)
        prev_eef = eef_pos.copy()
        env.step(recorded_actions[t])

    replay_succ = bool(env.check_success()) if hasattr(env, "check_success") else False
    orig_succ = orig_summary.get("task_success", False)
    env.close()

    max_eef = max(eef_errs)
    max_obj = max(obj_errs)
    parity_pass = (n == orig_summary.get("n_steps", -1) and replay_succ == orig_succ and max_eef <= 1e-5 and max_obj <= 1e-5)

    parity = {"n_steps_match": n == orig_summary.get("n_steps", -1),
              "success_match": replay_succ == orig_succ,
              "max_eef_error": float(max_eef), "max_obj_error": float(max_obj),
              "eef_parity": max_eef <= 1e-5, "obj_parity": max_obj <= 1e-5,
              "all_parity_pass": parity_pass}

    # Teacher
    tc = load_frozen_teacher()
    from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, find_sc5_anchor_v2
    teacher = V2PrivilegedTeacher(config=tc)
    labels = teacher.label_trajectory(records)
    anchor = find_sc5_anchor_v2(labels, K=tc.K, guard=tc.guard)

    return {"parity": parity, "teacher": {"teacher_supported": anchor.get("anchor", -1) >= 0,
                                          "teacher_anchor": anchor.get("anchor", -1),
                                          "stable_carry_start": anchor.get("stable_carry_start", -1),
                                          "K10_valid": anchor.get("K10_valid", False),
                                          "reason": anchor.get("reason", "unknown"),
                                          "teacher_config_sha": tc.config_sha},
            "target_binding": {"obj_body": obj_body, "basket_body": basket_body}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--profile", required=True, choices=["B0", "D1"])
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
    os.environ["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"

    out_dir = OUT_BASE / "replay_canary" / args.episode / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Replay: {args.episode}/{args.profile} GPU={args.gpu}")
    result = replay_one(args.episode, args.profile, args.gpu)

    if result.get("_error"):
        print(f"  ERROR: {result['_error']}")
    else:
        p = result["parity"]
        print(f"  Parity: steps={p['n_steps_match']} succ={p['success_match']} "
              f"eef={p['max_eef_error']:.2e} obj={p['max_obj_error']:.2e} ALL={'PASS' if p['all_parity_pass'] else 'FAIL'}")
        t = result["teacher"]
        print(f"  Teacher: supported={t['teacher_supported']} anchor={t['teacher_anchor']} sc_start={t['stable_carry_start']} K10={t['K10_valid']}")

    json.dump(result.get("parity", {}), open(out_dir / "replay_parity.json", "w"), indent=2)
    json.dump(result.get("teacher", {}), open(out_dir / "teacher_summary.json", "w"), indent=2)
    json.dump(result, open(out_dir / "replay_summary.json", "w"), indent=2, default=str)
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
