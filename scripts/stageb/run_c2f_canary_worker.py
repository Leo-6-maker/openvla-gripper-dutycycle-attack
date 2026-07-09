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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

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
    _streamer = SC5StreamingFeatureAdapterV2()
    eef_sid = env.sim.model.site_name2id("gripper0_grip_site")

    # ── Episode loop ──
    step_records = []
    buffer_25d = []
    attack_delivered = False
    attack_step = -1
    success = False

    for step in range(args.max_steps):
        rgb = np.asarray(obs["agentview_image"])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3].copy()

        # 25D features
        gs = physical_gripper_state(env, obs)
        gq_raw = gs.get("qpos", np.zeros(2)) if isinstance(gs, dict) else np.zeros(2)
        action, _, _, _ = decode_with_scores(
            vla_model, processor, device, rgb, task_language, suite, 8,
            libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
            resize_size=224, drop_attention_mask=True,
        )
        env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

        eef_pos = env.sim.data.site_xpos[eef_sid]
        _streamer.update(
            gripper_command=float(action[-1]),
            gripper_qpos=float(gq_raw[0] + gq_raw[1]) if len(gq_raw) >= 2 else float(action[-1]),
            gripper_opening_proxy=float(abs(gq_raw[0]) + abs(gq_raw[1])) if len(gq_raw) >= 2 else 0.0,
            eef_x=float(eef_pos[0]), eef_y=float(eef_pos[1]), eef_z=float(eef_pos[2]),
            eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
            action_dx=float(env_action[0]), action_dy=float(env_action[1]),
            action_dz=float(env_action[2]), action_gripper=float(env_action[-1]),
        )
        feat_25d = _streamer.get_canonical_25d()
        buffer_25d.append(np.asarray(feat_25d, dtype=np.float32))

        # C2f detector prediction
        emit_p, supp_p, emitted = detector.predict(buffer_25d, rgb, task_language, suite, task_idx)

        rec = {"step": step, "emit_p": emit_p, "suppress_p": supp_p, "emitted": emitted,
               "attack_delivered": False, "success": False}

        # Attack injection (mirrors D7 TRUE_T10 / RAND_T10 protocol)
        if emitted and not attack_delivered and args.condition in ("TRUE_T10", "RAND_T10"):
            from v4_run_eval_openvla import decode_with_scores as decode_attack
            if args.condition == "TRUE_T10":
                # targeted: force gripper-open token
                attack_action, _, _, _ = decode_attack(
                    vla_model, processor, device, rgb, task_language, suite, 8,
                    epsilon=EPSILON, pgd_steps=PGD_STEPS, target_gripper_open=True,
                    libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
                    resize_size=224, drop_attention_mask=True,
                )
            else:
                # RAND_T10: random direction perturbation
                attack_action, _, _, _ = decode_attack(
                    vla_model, processor, device, rgb, task_language, suite, 8,
                    epsilon=EPSILON, pgd_steps=PGD_STEPS, target_gripper_open=None,
                    libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
                    resize_size=224, drop_attention_mask=True,
                )
            env_action = postprocess_openvla_action_for_libero(
                np.asarray(attack_action, dtype=np.float32), enabled=True,
            )
            attack_delivered = True
            attack_step = step
            rec["attack_delivered"] = True
            rec["attack_step"] = step

        step_records.append(rec)
        obs, reward, done, info = env.step(env_action)
        if done:
            success = bool(info.get("success", False) or reward > 0.5)
            step_records[-1]["success"] = success
            break

    env.close()

    # ── Write outputs ──
    meta = {
        "parent_key": args.parent_key, "condition": args.condition,
        "suite": suite, "task_index": task_idx, "state_id": state_id,
        "task_language": task_language,
        "total_steps": len(step_records), "success": success,
        "attack_delivered": attack_delivered, "attack_step": attack_step,
        "detector_checkpoint": args.checkpoint,
        "tau_emit": args.tau_emit, "tau_suppress": args.tau_suppress,
        "git_commit": args.git_commit,
    }
    with open(out_dir / "episode_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "step_records.jsonl", "w") as f:
        for rec in step_records:
            f.write(json.dumps(rec) + "\n")

    print(f"{args.parent_key}/{args.condition}: steps={len(step_records)} "
          f"emit={attack_delivered} attack_step={attack_step} "
          f"success={success}")


if __name__ == "__main__":
    main()
