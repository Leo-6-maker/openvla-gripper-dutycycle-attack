#!/usr/bin/env python3
"""C2f online canary worker — TRUE_T10 vs RAND_T10 paired using D Full SigLIP detector.

Reuses D7 episode infrastructure, replacing C2e3 with C2fSigLIPDetectorRuntime.
Minimal: Object 12 + L10 12 parents.  Records per-step detector decisions and
attack delivery for offline paired analysis.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

# ── Attack config (mirrors D7) ──
EPSILON = 6.0 / 255.0
PGD_STEPS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-key", required=True)
    ap.add_argument("--condition", required=True, choices=["CLEAN", "TRUE_T10", "RAND_T10"])
    ap.add_argument("--checkpoint", required=True, help="C2fDetector .pt path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--tau-emit", type=float, default=0.33)
    ap.add_argument("--tau-suppress", type=float, default=0.67)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--git-commit", default="unknown")
    args = ap.parse_args()

    # GPU binding is handled by launcher via CUDA_VISIBLE_DEVICES.
    # Worker does not override it — always uses cuda:0.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Attack protocol constants (mirrors D7 T10) ──
    ATTACK_HORIZON = 10
    EPSILON = 6.0 / 255.0
    PGD_STEPS = 10

    # ── Parse parent key ──
    suite, task_str, state_str, _, _ = args.parent_key.split("/")
    task_idx = int(task_str.replace("task_", ""))
    state_id = int(state_str.replace("state_", ""))

    out_dir = Path(args.output_dir) / args.parent_key / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load OpenVLA model ──
    from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS, _visible_gpu_id
    model_path = SUITE_MODELS.get(suite, SUITE_MODELS["libero_10"])
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    vla_model = AutoModelCls.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                                              torch_dtype=torch.bfloat16, device_map=device).eval()

    # ── Load C2f detector ──
    from gripper_attack.c2f_siglip_detector_runtime import C2fSigLIPDetectorRuntime
    detector = C2fSigLIPDetectorRuntime(
        checkpoint_path=args.checkpoint,
        openvla_model=vla_model,
        openvla_processor=processor,
        device=device,
        window=args.window,
        tau_emit=args.tau_emit,
        tau_suppress=args.tau_suppress,
    )

    # ── Build env ──
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()
    task_suite = bm[suite]()
    task = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    task_language = str(getattr(task, "language", "") or task.name or "")
    if not task_language:
        task_language = str(Path(task.bddl_file).stem).replace("_", " ")
    task_bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)

    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    env, obs = build_v4_exact_env(task_bddl, _visible_gpu_id(), args.max_steps, 10)
    obs = env.set_init_state(init_states[state_id])
    env, obs = apply_dummy_wait(env, obs, 10)

    from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, physical_gripper_state
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from gripper_attack.c2f_siglip_detector_runtime import CANONICAL_25D_FEATURES
    _streamer = SC5StreamingFeatureAdapterV2()
    eef_sid = env.sim.model.site_name2id("gripper0_grip_site")

    # ── Episode loop ──
    step_records = []
    buffer_25d = []
    attack_window_start = -1
    attack_window_end = -1
    delivery_count = 0
    success = False
    prev_eef = None

    try:
     for step in range(args.max_steps):
        # RGB — mirror _rgb_from_obs() in C2f adapter for feature parity
        rgb = np.asarray(obs["agentview_image"])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb[..., :3].copy()
        if rgb.dtype != np.uint8:
            if np.nanmax(rgb) <= 1.0:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.size == 0 or np.max(rgb[..., :3]) < 5:
            raise RuntimeError(f"C2f RGB capture failed at step {step}: blank image")

        # 25D features with proper EEF velocity (mirrors C2f adapter)
        gs = physical_gripper_state(env, obs)
        gq_raw = gs.get("qpos", np.zeros(2)) if isinstance(gs, dict) else np.zeros(2)
        # Use correct unnorm_key for model (libero_goal model lost, using libero_10)
        unnorm_key = suite if suite != "libero_goal" else "libero_10"
        action, _, _, _ = decode_with_scores(
            vla_model, processor, device, rgb, task_language, unnorm_key, 8,
            libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
            resize_size=224, drop_attention_mask=True,
        )
        env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

        eef_pos = env.sim.data.site_xpos[eef_sid]
        eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
        eef_vx = eef_x - prev_eef[0] if prev_eef is not None else 0.0
        eef_vy = eef_y - prev_eef[1] if prev_eef is not None else 0.0
        eef_vz = eef_z - prev_eef[2] if prev_eef is not None else 0.0
        prev_eef = (eef_x, eef_y, eef_z)

        raw_grip = float(action[-1])
        gq = float(gq_raw[0] + gq_raw[1]) if len(gq_raw) >= 2 else raw_grip
        gw = float(abs(gq_raw[0]) + abs(gq_raw[1])) if len(gq_raw) >= 2 else 0.0
        _res = _streamer.update(
            step_id=step,
            raw_gripper=raw_grip,
            env_gripper=-1.0 if raw_grip > 0.5 else 1.0,
            gripper_qpos=gq,
            gripper_opening_proxy=gw,
            eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
            eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
            action_dx=float(env_action[0]), action_dy=float(env_action[1]),
            action_dz=float(env_action[2]), action_gripper=raw_grip,
        )
        # Extract 25D features from streamer response (mirrors C2f adapter)
        fv = {f: float(_res["features"].get(f, 0.0) or 0.0) for f in CANONICAL_25D_FEATURES}
        feat_25d = [fv[f] for f in CANONICAL_25D_FEATURES]
        buffer_25d.append(np.asarray(feat_25d, dtype=np.float32))

        # C2f detector prediction — full 4-condition gate
        det_out = detector.predict(buffer_25d, rgb, task_language, suite, task_idx)

        rec = {"step": step,
               "emit_p": det_out["emit_p"], "suppress_p": det_out["suppress_p"],
               "abstain_p": det_out["abstain_p"], "primary_p": det_out["primary_p"],
               "emitted": det_out["emitted"], "ready": det_out["ready"],
               "attack_delivered": False, "success": False}

        # ── T10 attack delivery (D7 protocol: K=10 consecutive steps) ──
        attack_active = (attack_window_start >= 0 and step < attack_window_end)
        if det_out["emitted"] and attack_window_start < 0 and args.condition in ("TRUE_T10", "RAND_T10"):
            attack_window_start = step
            attack_window_end = step + ATTACK_HORIZON

        if attack_window_start >= 0 and step == attack_window_start:
            rec["attack_window_start"] = True

        if attack_active or step == attack_window_start:
            # TRUE_T10: force gripper open by setting action[-1] to 1.0
            # RAND_T10: add random perturbation scaled to EPSILON
            attack_action = np.asarray(action, dtype=np.float32).copy()
            if args.condition == "TRUE_T10":
                attack_action[-1] = 1.0  # force gripper open
            else:
                noise = np.random.randn(*attack_action.shape).astype(np.float32)
                noise = noise / (np.linalg.norm(noise) + 1e-8) * EPSILON
                attack_action = np.clip(attack_action + noise, -1.0, 1.0)
            env_action = postprocess_openvla_action_for_libero(attack_action, enabled=True)
            delivery_count += 1
            rec["attack_delivered"] = True

        step_records.append(rec)
        obs, reward, done, info = env.step(env_action)
        if done:
            success = bool(info.get("success", False) or reward > 0.5)
            step_records[-1]["success"] = success
            break

    except Exception as _e:
        step_records.append({"step": -1, "error": str(_e), "emitted": False, "attack_delivered": False, "success": False})

    # Write metadata BEFORE env.close() to survive EGL teardown crashes
    import hashlib as _hashlib
    ckpt_sha = _hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest()
    meta = {
        "parent_key": args.parent_key, "condition": args.condition,
        "suite": suite, "task_index": task_idx, "state_id": state_id,
        "task_language": task_language,
        "total_steps": len(step_records), "success": success,
        "attack_window_start": attack_window_start,
        "attack_window_end": attack_window_end,
        "delivery_count": delivery_count,
        "detector_checkpoint": args.checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "tau_emit": args.tau_emit, "tau_suppress": args.tau_suppress,
        "tau_abstain": 0.5, "tau_primary": 0.5,
        "attack_horizon": ATTACK_HORIZON,
        "epsilon": EPSILON, "pgd_steps": PGD_STEPS,
        "git_commit": args.git_commit,
    }
    with open(out_dir / "episode_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "step_records.jsonl", "w") as f:
        for rec in step_records:
            f.write(json.dumps(rec) + "\n")

    try:
        env.close()
    except Exception:
        pass

    print(f"{args.parent_key}/{args.condition}: steps={len(step_records)} "
          f"emit={attack_window_start >= 0} "
          f"attack_window=[{attack_window_start},{attack_window_end}) "
          f"delivery_count={delivery_count} success={success}")


if __name__ == "__main__":
    main()
