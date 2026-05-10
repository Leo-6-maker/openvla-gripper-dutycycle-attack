#!/usr/bin/env python3
from __future__ import annotations
import argparse, getpass, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_openvla_compat import patch_openvla
from gripper_attack.io import read_json, sha256_jsonable, write_csv, write_json, write_jsonl
from gripper_attack.grasp import GraspPhaseTracker, compute_grasp_metadata, object_pos, proxy_grasp_metadata
from gripper_attack.uncertainty import arm_entropy, gripper_entropy, motion_weighted_arm_entropy, motion_weighted_xyz_entropy, prefix_entropy, prefix_top2_margin, xyz_entropy
from v4_run_eval_openvla import (
    decode_with_scores,
    fake_action,
    fake_logits,
    get_instruction,
    load_model,
    load_yaml,
    postprocess_openvla_action_for_libero,
    resolve_task_index,
)


def rho_key(rho: float) -> str:
    return f"rho_{float(rho):.2f}"


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(args, cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True).strip() or default
    except Exception:
        return default


def summarize(rows: list[dict], task_id: str) -> dict:
    H = [float(r["entropy"]) for r in rows if r.get("signal_available", True)]
    M = [float(r["margin"]) for r in rows if r.get("signal_available", True)]
    X = [float(r["xyz_entropy"]) for r in rows if r.get("signal_available", True)]
    A = [float(r["arm_entropy"]) for r in rows if r.get("signal_available", True)]
    WX = [float(r["motion_weighted_xyz_entropy"]) for r in rows if r.get("signal_available", True)]
    WA = [float(r["motion_weighted_arm_entropy"]) for r in rows if r.get("signal_available", True)]
    G = [float(r["gripper_entropy"]) for r in rows if r.get("signal_available", True)]
    C = [float(r["grasp_composite_entropy"]) for r in rows if r.get("signal_available", True)]
    return {
        "task_id": task_id,
        "num_steps": len(H),
        "entropy_mean": float(np.mean(H)) if H else 0.0,
        "entropy_p90": float(np.quantile(H, 0.90)) if H else 0.0,
        "xyz_entropy_mean": float(np.mean(X)) if X else 0.0,
        "arm_entropy_mean": float(np.mean(A)) if A else 0.0,
        "motion_weighted_xyz_entropy_mean": float(np.mean(WX)) if WX else 0.0,
        "motion_weighted_arm_entropy_mean": float(np.mean(WA)) if WA else 0.0,
        "gripper_entropy_mean": float(np.mean(G)) if G else 0.0,
        "grasp_composite_entropy_mean": float(np.mean(C)) if C else 0.0,
        "grasp_gate_rate": float(np.mean([bool(r.get("grasp_gate_active")) for r in rows])) if rows else 0.0,
        "proxy_grasp_gate_rate": float(np.mean([bool(r.get("proxy_grasp_gate_active")) for r in rows])) if rows else 0.0,
        "margin_mean": float(np.mean(M)) if M else 0.0,
        "margin_p10": float(np.quantile(M, 0.10)) if M else 0.0,
        "signal_availability_rate": float(len(H) / max(len(rows), 1)),
    }


def score_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    keys = ["entropy", "xyz_entropy", "arm_entropy", "motion_weighted_xyz_entropy", "motion_weighted_arm_entropy", "gripper_entropy", "grasp_composite_entropy", "margin"]
    return {
        key: np.asarray([float(r[key]) for r in rows if r.get("signal_available", True)], dtype=np.float64)
        for key in keys
    }


def uncertainty_scores(prefix_logits, clean_action) -> dict:
    return {
        "entropy": float(prefix_entropy(prefix_logits)),
        "xyz_entropy": float(xyz_entropy(prefix_logits)),
        "arm_entropy": float(arm_entropy(prefix_logits)),
        "motion_weighted_xyz_entropy": float(motion_weighted_xyz_entropy(prefix_logits, clean_action)),
        "motion_weighted_arm_entropy": float(motion_weighted_arm_entropy(prefix_logits, clean_action)),
        "gripper_entropy": float(gripper_entropy(prefix_logits)),
        "margin": float(prefix_top2_margin(prefix_logits)),
    }


def add_composite_scores(rows: list[dict]) -> None:
    x = np.asarray([float(r.get("xyz_entropy", 0.0)) for r in rows if r.get("signal_available", True)], dtype=np.float64)
    g = np.asarray([float(r.get("gripper_entropy", 0.0)) for r in rows if r.get("signal_available", True)], dtype=np.float64)
    xm = float(np.mean(x)) if x.size else 0.0; xs = float(np.std(x)) if x.size else 1.0
    gm = float(np.mean(g)) if g.size else 0.0; gs = float(np.std(g)) if g.size else 1.0
    xs = max(xs, 1e-6); gs = max(gs, 1e-6)
    for r in rows:
        r["xyz_entropy_mean"] = xm; r["xyz_entropy_std"] = xs
        r["gripper_entropy_mean"] = gm; r["gripper_entropy_std"] = gs
        r["grasp_composite_entropy"] = float(max((float(r.get("xyz_entropy", 0.0)) - xm) / xs, (float(r.get("gripper_entropy", 0.0)) - gm) / gs))


def make_thresholds(args, task, atk_cfg, cal_cfg, rows: list[dict], model_path: str, unnorm_key: str) -> dict:
    add_composite_scores(rows)
    scores = score_arrays(rows)
    if any(len(v) == 0 for v in scores.values()):
        raise SystemExit("no valid uncertainty rows collected; cannot write thresholds")
    dirty = git_value(["git", "status", "--porcelain"], default="")
    thresholds = {
        "version": "v4.3",
        "calibration_source": "rollout_passive_clean_observer",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_seed": int(args.seed),
        "K_trigger": int(atk_cfg["uncertainty"]["K_trigger"]),
        "num_episodes": int(args.episodes),
        "num_steps_total": int(len(scores["entropy"])),
        "model_path": model_path,
        "unnorm_key": unnorm_key,
        "task_id": task["task_id"],
        "preprocess": {
            "libero_official_preprocess": bool(args.libero_official_preprocess),
            "center_crop": bool(args.center_crop),
            "image_size": int(args.image_size),
            "openvla_resize_size": int(args.openvla_resize_size),
            "postprocess_gripper": bool(args.postprocess_gripper),
            "num_steps_wait": int(args.num_steps_wait),
            "success_metric": args.success_metric,
            "deterministic_init_states": bool(args.deterministic_init_states),
        },
        "command": " ".join(sys.argv),
        "code_git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "code_dirty": bool(dirty.strip()),
        "tasks": {task["task_id"]: {}},
    }
    rho_values = sorted(set(float(x) for x in cal_cfg.get("thresholds_for_rho", [0.01, 0.05, 0.10, 0.20])) | {0.03, 0.05})
    for rho in rho_values:
        rho = float(rho)
        thresholds["tasks"][task["task_id"]][rho_key(rho)] = {
            "entropy": float(np.quantile(scores["entropy"], 1.0 - rho)),
            "xyz_entropy": float(np.quantile(scores["xyz_entropy"], 1.0 - rho)),
            "arm_entropy": float(np.quantile(scores["arm_entropy"], 1.0 - rho)),
            "motion_weighted_xyz_entropy": float(np.quantile(scores["motion_weighted_xyz_entropy"], 1.0 - rho)),
            "motion_weighted_arm_entropy": float(np.quantile(scores["motion_weighted_arm_entropy"], 1.0 - rho)),
            "gripper_entropy": float(np.quantile(scores["gripper_entropy"], 1.0 - rho)),
            "grasp_composite_entropy": float(np.quantile(scores["grasp_composite_entropy"], 1.0 - rho)),
            "xyz_entropy_mean": float(np.mean(scores["xyz_entropy"])),
            "xyz_entropy_std": float(np.std(scores["xyz_entropy"])),
            "gripper_entropy_mean": float(np.mean(scores["gripper_entropy"])),
            "gripper_entropy_std": float(np.std(scores["gripper_entropy"])),
            "margin": float(np.quantile(scores["margin"], rho)),
            "num_steps": int(len(scores["entropy"])),
            "num_episodes": int(args.episodes),
        }
    return thresholds


def write_progress(out: Path, args, task, status: str, rows: list[dict], episodes_done: int, error: str = "") -> None:
    write_json(str(out / "progress.json"), {
        "version": "v4.3",
        "status": status,
        "error": error,
        "task_id": task["task_id"],
        "episodes_requested": int(args.episodes),
        "episodes_completed": int(episodes_done),
        "steps_written": int(len(rows)),
        "max_steps": int(args.max_steps_override or task["max_steps"]),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    })


def run_dry(args, task, atk_cfg, cal_cfg, out: Path) -> None:
    rng = np.random.RandomState(args.seed)
    rows = []
    k = int(atk_cfg["uncertainty"]["K_trigger"])
    max_steps = int(args.max_steps_override or 5)
    for ep in range(args.episodes):
        for t in range(max_steps):
            logits = fake_logits(rng, k=k, v=64)
            action = fake_action(rng)
            scores = uncertainty_scores(logits, action)
            rows_tmp = [{"xyz_entropy": scores["xyz_entropy"], "gripper_entropy": scores["gripper_entropy"], "signal_available": True}]
            add_composite_scores(rows_tmp); scores["grasp_composite_entropy"] = rows_tmp[0]["grasp_composite_entropy"]
            rows.append({
                "version": "v4.3", "calibration_source": "rollout_passive_clean_observer_dry_run",
                "task_id": task["task_id"], "suite": task["suite"], "episode_id": ep, "step_idx": t,
                "seed": int(args.seed), "max_steps": max_steps, **scores,
                "prefix_logits_shape": list(logits.shape), "action": action.tolist(),
                "success_so_far": False, "Tclean_decode": 0.0, "Ttrig": 0.0,
                "grasp_gate_active": False, "proxy_grasp_gate_active": False,
                "signal_available": True, "camera_obs_key": args.camera_obs_key,
                "model_path": args.model_path or "dry_run", "unnorm_key": args.unnorm_key,
            })
        write_progress(out, args, task, "running", rows, ep + 1)
    thresholds = make_thresholds(args, task, atk_cfg, cal_cfg, rows, args.model_path or "dry_run", args.unnorm_key)
    thresholds["calibration_source"] = "rollout_passive_clean_observer_dry_run"
    thresholds["dry_run"] = True
    write_jsonl(str(out / "passive_uncertainty_steps.jsonl"), rows)
    write_csv(str(out / "calibration_summary.csv"), [summarize(rows, task["task_id"])])
    write_json(str(out / "thresholds.json"), thresholds)
    write_progress(out, args, task, "done", rows, args.episodes)
    print(f"[ok] rollout-passive calibration dry_run -> {out}", flush=True)


def run_real(args, task, atk_cfg, cal_cfg, out: Path) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    model_path = args.model_path or atk_cfg.get("model_paths", {}).get(task["suite"]) or atk_cfg.get("model_paths", {}).get("libero_goal")
    if args.auto_patch_compat:
        patch_openvla(Path(args.base_model_code_dir), Path(model_path))
    model, processor, device = load_model(model_path, model_gpu_device_id=int(args.model_gpu_device_id))
    keys = list(getattr(model, "norm_stats", {}).keys())
    unnorm = args.unnorm_key if args.unnorm_key in keys else (task.get("default_unnorm_key") if task.get("default_unnorm_key") in keys else (keys[0] if keys else args.unnorm_key))
    bench = get_benchmark(task["suite"])()
    idx = resolve_task_index(bench, task["task_name"])
    instruction = get_instruction(bench, idx, task["task_name"])
    init_states = bench.get_task_init_states(idx)
    max_steps = int(args.max_steps_override or task["max_steps"])
    env = OffScreenRenderEnv(
        bddl_file_name=bench.get_task_bddl_file_path(idx),
        camera_heights=int(args.image_size), camera_widths=int(args.image_size),
        render_gpu_device_id=args.render_gpu_device_id, horizon=max_steps,
    )
    try:
        env.seed(0)
    except Exception:
        pass
    rng = np.random.RandomState(args.seed)
    state_ids = np.arange(args.episodes) % len(init_states) if args.deterministic_init_states else rng.choice(len(init_states), size=args.episodes, replace=args.episodes > len(init_states))
    rows = []
    episodes = []
    k = int(atk_cfg["uncertainty"]["K_trigger"])
    write_progress(out, args, task, "running", rows, 0)
    try:
        for ep, sid in enumerate(state_ids):
            obs = env.reset(); obs = env.set_init_state(init_states[int(sid)])
            if args.num_steps_wait > 0:
                dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                for _ in range(int(args.num_steps_wait)):
                    obs, _, _, _ = env.step(dummy_action)
            bowl0=object_pos(env, "akita_black_bowl_1")
            grasp_tracker=GraspPhaseTracker()
            grasp_tracker.reset(float(bowl0[2]) if bowl0 is not None else 0.0)
            proxy_gripper_history=[]
            ep_start = len(rows); success = False; invalid = False; invalid_reason = ""
            for t in range(max_steps):
                if args.camera_obs_key not in obs:
                    invalid = True; invalid_reason = f"missing camera {args.camera_obs_key}; keys={list(obs.keys())}"; break
                clean, prefix_logits, tclean, _ = decode_with_scores(
                    model, processor, device, obs[args.camera_obs_key], instruction, unnorm, k,
                    libero_official_preprocess=args.libero_official_preprocess,
                    center_crop=args.center_crop,
                    resize_size=args.openvla_resize_size,
                    drop_attention_mask=(not args.keep_attention_mask),
                )
                t0 = time.time()
                signal_available = prefix_logits is not None
                if signal_available:
                    scores = uncertainty_scores(prefix_logits, clean); shape = list(np.asarray(prefix_logits).shape)
                else:
                    scores = {"entropy": 0.0, "xyz_entropy": 0.0, "arm_entropy": 0.0, "motion_weighted_xyz_entropy": 0.0, "motion_weighted_arm_entropy": 0.0, "gripper_entropy": 0.0, "grasp_composite_entropy": 0.0, "margin": 0.0}; shape = []
                ttrig = time.time() - t0
                env_action = postprocess_openvla_action_for_libero(clean, enabled=args.postprocess_gripper)
                grasp_meta=compute_grasp_metadata(env,t,clean,env_action,grasp_tracker,gate_dist_threshold=float(args.grasp_gate_dist))
                proxy_meta=proxy_grasp_metadata(t,clean,env_action,proxy_gripper_history)
                obs, reward, done, info = env.step(env_action)
                proxy_gripper_history.append(float(env_action[-1]) if len(env_action) else 0.0)
                proxy_gripper_history=proxy_gripper_history[-5:]
                success_done = bool(done); success_check = bool(env.check_success())
                success = success_done if args.success_metric == "done" else success_check
                rows.append({
                    "version": "v4.3", "calibration_source": "rollout_passive_clean_observer",
                    "task_id": task["task_id"], "suite": task["suite"], "episode_id": int(ep),
                    "step_idx": int(t), "seed": int(args.seed), "max_steps": max_steps,
                    "init_state_id": int(sid), **scores,
                    "prefix_logits_shape": shape, "action": np.asarray(clean, dtype=np.float32).tolist(),
                    "env_action": np.asarray(env_action, dtype=np.float32).tolist(),
                    "success_so_far": bool(success), "success_done": success_done, "success_check": success_check,
                    "Tclean_decode": float(tclean), "Ttrig": float(ttrig), "signal_available": bool(signal_available),
                    "camera_obs_key": args.camera_obs_key, "model_path": model_path, "unnorm_key": unnorm,
                    **grasp_meta, **proxy_meta,
                })
                if success or bool(done):
                    break
            add_composite_scores(rows)
            ep_rows = rows[ep_start:]
            episodes.append({
                "episode_id": int(ep), "init_state_id": int(sid), "num_steps": len(ep_rows),
                "success": bool(success), "invalid": bool(invalid), "invalid_reason": invalid_reason,
                "signal_availability_rate": float(np.mean([bool(r.get("signal_available")) for r in ep_rows])) if ep_rows else 0.0,
                "entropy_mean": float(np.mean([r["entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "xyz_entropy_mean": float(np.mean([r["xyz_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "arm_entropy_mean": float(np.mean([r["arm_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "motion_weighted_xyz_entropy_mean": float(np.mean([r["motion_weighted_xyz_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "motion_weighted_arm_entropy_mean": float(np.mean([r["motion_weighted_arm_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "margin_mean": float(np.mean([r["margin"] for r in ep_rows])) if ep_rows else 0.0,
                "gripper_entropy_mean": float(np.mean([r["gripper_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "grasp_composite_entropy_mean": float(np.mean([r["grasp_composite_entropy"] for r in ep_rows])) if ep_rows else 0.0,
                "grasp_gate_rate": float(np.mean([bool(r.get("grasp_gate_active")) for r in ep_rows])) if ep_rows else 0.0,
                "proxy_grasp_gate_rate": float(np.mean([bool(r.get("proxy_grasp_gate_active")) for r in ep_rows])) if ep_rows else 0.0,
            })
            write_jsonl(str(out / "passive_uncertainty_steps.jsonl"), rows)
            write_csv(str(out / "calibration_episodes.csv"), episodes)
            write_csv(str(out / "calibration_summary.csv"), [summarize(rows, task["task_id"])])
            write_progress(out, args, task, "running", rows, ep + 1)
            print(f"[progress] calibration episode={ep+1}/{args.episodes} steps={len(ep_rows)} total_steps={len(rows)} success={success} invalid={invalid}", flush=True)
    except Exception as e:
        write_progress(out, args, task, "error", rows, len(episodes), error=repr(e))
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass
    add_composite_scores(rows)
    thresholds = make_thresholds(args, task, atk_cfg, cal_cfg, rows, model_path, unnorm)
    write_json(str(out / "thresholds.json"), thresholds)
    manifest = {
        "version": "v4.3", "run_id": out.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(), "user": getpass.getuser(), "cwd": os.getcwd(), "command": " ".join(sys.argv),
        "code_git_commit": thresholds.get("code_git_commit", "unknown"), "code_dirty": thresholds.get("code_dirty", "unknown"),
        "config_hash": sha256_jsonable({"attack": atk_cfg, "calibration": cal_cfg}),
        "attack_config_path": args.attack_config, "tasks_config_path": args.tasks_config,
        "directions_config_path": "", "thresholds_path": str(out / "thresholds.json"),
        "model_id": atk_cfg.get("victim", "openvla_7b"), "model_checkpoint_path": model_path,
        "dataset_manifest_hash": "", "task_id": task["task_id"], "suite": task["suite"],
        "seed": int(args.seed), "trigger_name": "passive_uncertainty_logging", "rho": 0.0,
        "grasp_gate_dist_threshold": float(args.grasp_gate_dist),
        "episodes": int(args.episodes), "max_steps": max_steps,
        "output_files": {"steps": str(out / "passive_uncertainty_steps.jsonl"), "thresholds": str(out / "thresholds.json"), "summary": str(out / "calibration_summary.csv")},
        "status": "done", "error": "",
    }
    write_json(str(out / "run_manifest.json"), manifest)
    write_progress(out, args, task, "done", rows, args.episodes)
    print(f"[ok] rollout-passive calibration -> {out / 'thresholds.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--calibration_config", default="configs/v4_calibration.yaml")
    ap.add_argument("--attack_config", default="configs/v4_attack.yaml")
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--model_path", default="")
    ap.add_argument("--base_model_code_dir", default="${OPENVLA_BASE_MODEL_DIR}")
    ap.add_argument("--unnorm_key", default="libero_goal")
    ap.add_argument("--camera_obs_key", default="agentview_image")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps_override", type=int, default=0)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--output_dir", default="outputs/v4/calibration_rollout")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--model_gpu_device_id", type=int, default=-1)
    ap.add_argument("--grasp_gate_dist", type=float, default=0.10, help="Privileged grasp gate eef-bowl distance threshold in meters")
    ap.add_argument("--auto_patch_compat", action="store_true")
    ap.add_argument("--libero_official_preprocess", action="store_true")
    ap.add_argument("--center_crop", action="store_true")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--openvla_resize_size", type=int, default=224)
    ap.add_argument("--num_steps_wait", type=int, default=0)
    ap.add_argument("--postprocess_gripper", action="store_true")
    ap.add_argument("--success_metric", choices=["done", "check_success"], default="done")
    ap.add_argument("--deterministic_init_states", action="store_true")
    ap.add_argument("--keep_attention_mask", action="store_true")
    args = ap.parse_args()
    tasks = load_yaml(args.tasks_config)["tasks"]
    task = next(t for t in tasks if t["task_id"] == args.task_id)
    atk_cfg = load_yaml(args.attack_config)
    cal_cfg = load_yaml(args.calibration_config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        run_dry(args, task, atk_cfg, cal_cfg, out)
    else:
        run_real(args, task, atk_cfg, cal_cfg, out)


if __name__ == "__main__":
    main()
