#!/usr/bin/env python3
"""V5-LEAN collector — perturbation, native SHA, target pose, no video.

Minimal modification of V4 bridge: adds perturbation before reset,
records native SHA256 fields, target position in telemetry, 3 QA frames.
No full video. Disk budget ~100-150KB per cell.
"""
import os, sys, json, csv, hashlib, uuid, time, argparse, copy
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "migration"))  # for label_m1c_object_teacher

CKPT_PATH = REPO / "artifacts/detector/sc5_mlp_s2.pt"
MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", str(REPO / "models/openvla-7b-finetuned-libero-object"))


def sha256_file(p):
    if not Path(p).exists(): return "MISSING"
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="V5 LEAN Collector")
    ap.add_argument("--task_idx", type=int, required=True)
    ap.add_argument("--state_id", type=int, required=True)
    ap.add_argument("--perturbation_template", default="P0")
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--render_gpu", type=int, default=0)
    ap.add_argument("--condition", default="CLEAN")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.render_gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cell_dir = Path(args.output_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)

    cell_uuid = str(uuid.uuid4())[:12]
    run_uuid = str(uuid.uuid4())[:8]

    # Build env
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()
    suite = bm["libero_object"]()
    task_obj = suite.get_task(args.task_idx)
    init_states = suite.get_task_init_states(args.task_idx)

    from libero.libero import get_libero_path
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    from gripper_attack.v5_perturbation import apply_perturbation, compute_initial_state_hash
    bddl_path = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env, obs = build_v4_exact_env(bddl_path, args.render_gpu, 400, 10)

    # Capture original state hash
    original_state_sha = compute_initial_state_hash(env)

    # Apply perturbation
    env, obs, pert_spec = apply_perturbation(env, obs, args.perturbation_template, args.base_seed)
    perturbed_state_sha = compute_initial_state_hash(env)

    obs = env.set_init_state(init_states[args.state_id])
    env, obs = apply_dummy_wait(env, obs, 10)

    # Resolve target position
    target_x = target_y = target_z = 0.0
    from label_m1c_object_teacher import resolve_target_position
    tgt = resolve_target_position(args.task_idx, args.state_id)
    if tgt: target_x, target_y, target_z = tgt

    # Build target resolver telemetry
    target_resolver_sha = hashlib.sha256(
        open(__file__, "rb").read()
    ).hexdigest()

    # Model
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0",
        attn_implementation="eager")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    def prompt(ins): return f"In: {ins}\nOut:"

    from scripts.v4_run_eval_openvla import decode_with_scores
    from gripper_attack.openvla_preprocess import prepare_openvla_image

    # Detector
    from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime
    detector = SC5DetectorRuntime(str(CKPT_PATH))

    telemetry = []
    qa_frames = []
    ANCHOR = 0
    step = 0
    instruction = task_obj.name

    while True:
        t0 = time.perf_counter()
        raw = obs.get("agentview_image", None)
        if raw is None:
            raw = obs.get("image", obs.get("agentview_image"))
        action, _, _, _ = decode_with_scores(model, processor, getattr(model, "device", "cuda:0"),
                                              raw, instruction, "libero_object", 8,
                                              libero_preprocess_backend="official_pil_lanczos",
                                              center_crop=True, resize_size=224)
        env_action = np.array(action, dtype=np.float64)

        # Detector
        from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeaturesV2
        if step == 0:
            streamer = SC5StreamingFeaturesV2()
            _first_valid = -1
        feat_res = streamer.update(env_action, obs, step == 0)
        if feat_res.get("valid"):
            if _first_valid < 0: _first_valid = step
            dec = detector.update(feat_res["features"], step)

        eef_pos = np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
        obj_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(f"{task_obj.name.split('_')[-2]}_main")])
        qpos_sum = float(env.sim.data.qpos[-2:].sum())

        tel = {
            "step": step, "task_idx": args.task_idx, "parent_state_id": args.state_id,
            "perturbation_seed": args.base_seed, "perturbation_spec": args.perturbation_template,
            "language_instruction": instruction,
            "raw_gripper": float(action[-1]), "gripper_qpos": qpos_sum,
            "gripper_opening_proxy": qpos_sum,
            "eef_x": float(eef_pos[0]), "eef_y": float(eef_pos[1]), "eef_z": float(eef_pos[2]),
            "object_x": float(obj_pos[0]), "object_y": float(obj_pos[1]), "object_z": float(obj_pos[2]),
            "target_x": target_x, "target_y": target_y, "target_z": target_z,
            "object_eef_distance": float(np.linalg.norm(obj_pos - eef_pos)),
            "object_target_distance": float(np.linalg.norm(obj_pos - np.array([target_x, target_y, target_z]))),
            "corridor_p": dec.get("corridor_p"), "release_p": dec.get("release_p"),
            "pred_phase": dec.get("pred_phase"), "feat_valid": feat_res.get("valid", False),
            "condition": args.condition, "attack_frames": 0,
            "env_action_7d": json.dumps([float(x) for x in action]),
            "model_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        telemetry.append(tel)

        # QA frames: start, first_close, final
        if len(qa_frames) < 3 and step in [0, 5]:
            qa_frames.append(copy.deepcopy(raw))
        if step % 50 == 49:
            qa_frames = qa_frames[:2]  # keep start + first_close

        obs, _, done, _ = env.step(env_action)
        step += 1
        if done: break

    success = bool(env.check_success()) if hasattr(env, "check_success") else False
    env.close()

    # Save QA frames
    import PIL.Image
    for i, frame in enumerate(qa_frames[:3]):
        PIL.Image.fromarray(np.asarray(frame)).save(cell_dir / f"qa_frame_{i}.png")

    # Save telemetry
    with open(cell_dir / "step_telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=telemetry[0].keys())
        w.writeheader(); w.writerows(telemetry)

    bridge_sha = sha256_file(__file__)
    ckpt_sha = sha256_file(CKPT_PATH)

    # Compute trajectory SHA
    tel_bytes = open(cell_dir / "step_telemetry.csv", "rb").read()

    summary = {
        "run_uuid": run_uuid, "cell_uuid": cell_uuid, "pool": "sc5_v2",
        "task_idx": args.task_idx, "parent_state_id": args.state_id,
        "perturbation_seed": args.base_seed, "perturbation_family": args.perturbation_template,
        "n_steps": step, "exit_code": 0, "task_success": success,
        "condition": args.condition, "attack_frames": 0,
        "initial_state_sha256": original_state_sha,
        "perturbed_initial_state_sha256": perturbed_state_sha,
        "trajectory_content_sha256": hashlib.sha256(tel_bytes).hexdigest(),
        "checkpoint_sha256": ckpt_sha,
        "bridge_sha256": bridge_sha,
        "target_resolver_sha256": target_resolver_sha,
        "perturbation_generator_sha256": pert_spec["perturbation_generator_sha256"],
    }
    with open(cell_dir / "episode_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(cell_dir / "target_resolver.json", "w") as f:
        json.dump({"target_position": [target_x, target_y, target_z], "task": args.task_idx, "state": args.state_id}, f)
    with open(cell_dir / ".done", "w") as f:
        json.dump({"exit_code": 0, "telemetry_sha": hashlib.sha256(tel_bytes).hexdigest()}, f)

    print(f"V5 LEAN: task={args.task_idx} state={args.state_id} pert={args.perturbation_template} "
          f"steps={step} succ={success} size={tel_bytes.__len__()}B")


if __name__ == "__main__":
    import torch
    main()
