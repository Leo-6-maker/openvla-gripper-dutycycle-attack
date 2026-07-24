#!/usr/bin/env python3
"""D7 persistent worker: load model once, process many episodes from manifest.

One worker = one GPU = one model load = many episodes sequentially.
Eliminates per-episode model reloading (4 min each) for D7 rollout.

Usage: python run_d7_table1_persistent_worker.py --gpu N --manifest queue.csv
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
os.environ["OPENVLA_ATTN_IMPLEMENTATION"] = "eager"
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["NUMEXPR_NUM_THREADS"] = "1"

SUITE_MODELS = {
    "libero_10": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10",
    "libero_goal": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal",
    "libero_object": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "libero_spatial": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620",
}
K = 10; EPSILON = 0.023529411764705882; TARGET_TOKEN = 31744; ARM_GATE = 5; PGD_STEPS = 20; MAX_STEPS = 300


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

def load_c2e3(package_dir: str):
    """Load C2e3 GRU detector with normalization + context contract."""
    from gripper_attack.c2e3_gru_detector_runtime import C2e3GRUDetectorRuntime
    return C2e3GRUDetectorRuntime(package_dir)


def run_episode(model, processor, vla_model, device, model_dtype, action_dim, unnorm_key, bm_key, task_idx, state_id, seed, condition, trigger_override, out_dir, source_commit, detector):
    """Execute one episode with pre-loaded models (persistent)."""
    import torch
    from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero, physical_gripper_state
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    from libero.libero import benchmark, get_libero_path

    is_attack = condition != "CLEAN"; is_rand = condition == "RAND_T10"; is_oracle = condition == "COMMAND_OPEN_ORACLE"
    attack_frames = K if (is_attack or is_oracle) else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    result = {"suite": bm_key, "condition": condition, "task_idx": task_idx, "state_id": state_id, "seed": seed,
        "task_success": False, "n_steps": 0, "detector_emitted": False, "emit_step": -1,
        "attack_frames": 0, "failure_taxonomy": "unknown", "error": "", "runtime_seconds": 0.0,
        "source_commit": source_commit}

    try:
        bm = benchmark.get_benchmark_dict(); suite_obj = bm[bm_key]()
        task_obj = suite_obj.get_task(task_idx); init_states = suite_obj.get_task_init_states(task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
        instruction = task_obj.language
        env, obs = build_v4_exact_env(bddl, int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")), MAX_STEPS, 10)
        obs = env.set_init_state(init_states[state_id]); env, obs = apply_dummy_wait(env, obs, 10)

        _streamer = SC5StreamingFeatureAdapterV2()
        _gr_emit = -1; _gr_ep = 0.0; _gr_sp = 0.0
        _eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        _prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
        _feature_buffer = []; W = 16
        SC5_F = ["gripper_command","gripper_qpos","gripper_opening_proxy","eef_x","eef_y","eef_z",
                  "eef_vx","eef_vy","eef_vz","action_dx","action_dy","action_dz","action_gripper",
                  "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
                  "close_onset","time_since_close","eef_speed","eef_z_delta_since_close",
                  "qpos_delta_1","qpos_delta_3","opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5"]

        # Attacker (persistent, created once per condition change)
        attacker = None
        if is_attack and not is_rand and not is_oracle:
            from gripper_attack.attack_adapter import OpenVLAVisualAttacker
            opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                   "epsilon": EPSILON, "num_steps": PGD_STEPS, "step_size": EPSILON * 0.075,
                   "random_start": True, "prefix_refresh_interval": 1, "surrogate_score_path": "cached_autoregressive_generate_v1",
                   "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
                   "target_token_id": TARGET_TOKEN, "target_execution_class": "CLIP_MEDIATED_OPEN",
                   "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": ARM_GATE}
            attacker = OpenVLAVisualAttacker(model=vla_model, processor=processor, config={"attack_optimizer": opt},
                seed=seed, preprocess_kwargs={"libero_preprocess_backend": "upstream_tf_jpeg", "center_crop": True, "resize_size": 224}, device=device)

        telemetry = []; attack_count = 0

        for step in range(MAX_STEPS):
            if "agentview_image" not in obs: break
            raw = np.asarray(obs["agentview_image"]).copy()
            gs = physical_gripper_state(env, obs)
            q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos",[])) > 0 else float("nan")
            q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos",[])) > 1 else float("nan")
            qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
            eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

            action, _, _, gen = decode_with_scores(vla_model, processor, device, raw, instruction, unnorm_key, 8,
                libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224, drop_attention_mask=True)
            raw_grip = float(action[-1])
            env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

            # GRU trigger
            if _gr_emit < 0:
                eef_valid = np.all(np.isfinite([eef_x, eef_y, eef_z]))
                _vx = eef_x - _prev_eef[0] if _prev_eef and eef_valid else float("nan")
                _vy = eef_y - _prev_eef[1] if _prev_eef and eef_valid else float("nan")
                _vz = eef_z - _prev_eef[2] if _prev_eef and eef_valid else float("nan")
                if eef_valid: _prev_eef = (eef_x, eef_y, eef_z)
                gw = abs(q7) + abs(q8) if not (np.isnan(q7) or np.isnan(q8) or np.isnan(qpos_sum)) else float("nan")
                gq = float(qpos_sum) if not np.isnan(qpos_sum) else float("nan")
                try:
                    _res = _streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=-1.0 if raw_grip>0.5 else 1.0,
                        gripper_qpos=gq, gripper_opening_proxy=gw, eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                        eef_vx=_vx, eef_vy=_vy, eef_vz=_vz, action_dx=float(action[0]), action_dy=float(action[1]),
                        action_dz=float(action[2]), action_gripper=raw_grip)
                except: _res = {"valid": False}
                if _res.get("valid"):
                    fv = [float(_res["features"].get(f, 0.0) or 0.0) for f in SC5_F]
                    _feature_buffer.append(fv)
                    if len(_feature_buffer) >= W:
                        window = np.array(_feature_buffer[-W:], dtype=np.float32)
                        _gr_ep, _gr_sp, _gr_emitted = detector.predict(window, suite=unnorm_key, task_index=task_idx)
                        if _gr_emitted: _gr_emit = step

            # Attack
            attack_this = False; adv_tokens = None
            _trigger_step = trigger_override if trigger_override >= 0 else _gr_emit
            if is_attack and _trigger_step >= 0 and step >= _trigger_step and attack_count < attack_frames:
                if is_rand:
                    from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
                    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
                    proc = prepare_openvla_image_for_attack(raw, libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224)
                    inputs = processor(prompt(instruction), proc, return_tensors="pt")
                    inputs.pop("attention_mask", None)
                    iids = inputs["input_ids"].to(device)
                    if not torch.all(iids[:,-1]==29871): iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=device)], dim=1)
                    x = inputs["pixel_values"].to(device=device, dtype=model_dtype)
                    delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=seed+100000+attack_count, dtype=torch.float32, device=device)
                    proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
                    with torch.inference_mode():
                        go = vla_model.generate(input_ids=iids, pixel_values=proj.detach().to(dtype=model_dtype), max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
                elif is_oracle:
                    oracle_action = np.asarray(action, dtype=np.float32).copy(); oracle_action[-1] = -1.0
                    env_action_final = postprocess_openvla_action_for_libero(oracle_action, enabled=True)
                    attack_this = True; attack_count += 1
                else:
                    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
                    clean_action_np = np.asarray(action, dtype=np.float32)
                    proc = prepare_openvla_image_for_attack(raw, libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224)
                    inputs = processor(prompt(instruction), proc, return_tensors="pt")
                    inputs.pop("attention_mask", None)
                    iids = inputs["input_ids"].to(device)
                    if not torch.all(iids[:,-1]==29871): iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=device)], dim=1)
                    pv = inputs["pixel_values"].to(device=device, dtype=model_dtype)
                    with torch.inference_mode():
                        go = vla_model.generate(input_ids=iids, pixel_values=pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
                    clean_gen = type("CG",(),{})(); clean_gen.sequences = torch.tensor([iids[0].detach().cpu().tolist()+[int(t) for t in clean_tokens]], dtype=torch.long, device=device); clean_gen.scores = []
                    attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen, unnorm_key=unnorm_key)
                    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                    with torch.inference_mode():
                        go2 = vla_model.generate(input_ids=iids, pixel_values=adv_inputs["pixel_values"], max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    adv_tokens = extract_exact_new_tokens(go2.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)

                if adv_tokens is not None:
                    vocab_size = int(vla_model.config.text_config.vocab_size - vla_model.config.pad_to_multiple_of)
                    disc = np.clip(vocab_size - np.array([int(t) for t in adv_tokens]) - 1, 0, vla_model.bin_centers.shape[0] - 1)
                    na = vla_model.bin_centers[disc]; s = vla_model.get_action_stats(unnorm_key)
                    lo = np.asarray(s["q01"], np.float32); hi = np.asarray(s["q99"], np.float32)
                    mk = np.asarray(s.get("mask", np.ones_like(lo, dtype=bool)), dtype=bool)
                    attack_action = np.where(mk, 0.5*(na+1)*(hi-lo)+lo, na).astype(np.float32)
                    env_action_final = postprocess_openvla_action_for_libero(attack_action, enabled=True)
                    attack_this = True; attack_count += 1

            obs, reward, done, info = env.step(env_action_final)
            telemetry.append({"step": step, "emitted": _gr_emit>=0, "attack_this": attack_this, "attack_count": attack_count,
                "raw_gripper": raw_grip, "gr_emit_p": _gr_ep, "gr_suppress_p": _gr_sp})
            if done: break

        success = bool(reward > 0 if isinstance(reward, (int,float)) else False)
        result.update({"task_success": success, "n_steps": len(telemetry), "detector_emitted": _gr_emit>=0,
            "emit_step": _gr_emit, "attack_frames": attack_count, "failure_taxonomy": "success" if success else ("no_trigger" if _gr_emit<0 else "attack_ineffective"),
            "runtime_seconds": time.time()-start_time})
        if telemetry:
            with open(out_dir / "step_telemetry.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
        env.close()
    except Exception as e:
        import traceback; traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"; result["failure_taxonomy"] = "runtime_error"

    prov = detector.provenance
    result["detector_checkpoint_sha256"] = prov["checkpoint_sha256"]
    result["normalization_stats_sha256"] = prov["normalization_sha256"]
    result["config_sha256"] = prov["config_sha256"]
    result["context_lookup_sha256"] = prov["context_lookup_sha256"]
    result["normalization_applied"] = True
    result["context_policy"] = prov["context_policy"]
    result["tau_emit"] = prov["tau_emit"]
    result["tau_suppress"] = prov["tau_suppress"]
    write_json(out_dir / "episode_summary.json", result)
    write_json(out_dir / "artifact_sha256.json", {
        "detector_checkpoint_sha256": prov["checkpoint_sha256"],
        "normalization_stats_sha256": prov["normalization_sha256"],
        "config_sha256": prov["config_sha256"],
        "context_lookup_sha256": prov["context_lookup_sha256"],
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--detector-package", required=True,
                    help="C2e3 package directory (contains model, config, normalization stats, context lookup)")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--start-row", type=int, default=0)
    ap.add_argument("--end-row", type=int, default=999999)
    args = ap.parse_args()

    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # Read manifest
    rows = list(csv.DictReader(open(args.manifest)))
    rows = rows[args.start_row:min(args.end_row, len(rows))]
    print(f"Worker GPU {args.gpu}: {len(rows)} episodes to process", flush=True)

    # Load C2e3 detector runtime once (includes model + normalization + context lookup)
    detector = load_c2e3(args.detector_package)
    print(f"Detector loaded: checkpoint={detector.checkpoint_sha256[:16]}... "
          f"norm={detector.normalization_sha256[:16]}... "
          f"tau_emit={detector.tau_emit} tau_suppress={detector.tau_suppress}", flush=True)

    # Track current suite to avoid reloading
    current_suite = None
    vla_model = None; processor = None; model_dtype = None

    completed = 0; errors = 0
    for idx, row in enumerate(rows):
        suite = row["suite"]; condition = row["condition"]
        task_idx = int(row.get("task_index", 0) or 0)
        state_id = int(row.get("state_id", 0) or 0)
        seed = int(row.get("seed", row.get("seed_id", 0)) or 0)
        parent_key = row.get("parent_key", f"ep_{idx}")
        trigger_override = int(row.get("detector_trigger_step", -1) or -1)

        model_path = SUITE_MODELS[suite]; unnorm_key = suite; bm_key = suite

        # Load/reuse OpenVLA model
        if suite != current_suite:
            print(f"  Loading {suite} model...", flush=True)
            from transformers import AutoProcessor
            try: from transformers import AutoModelForImageTextToText as AutoModelCls
            except: from transformers import AutoModelForVision2Seq as AutoModelCls
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            vla_model = AutoModelCls.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device)
            model_dtype = next(vla_model.parameters()).dtype
            action_dim = int(vla_model.get_action_dim(unnorm_key))
            current_suite = suite
            print(f"  {suite} loaded dim={action_dim}", flush=True)

        out_dir = Path(args.output_root) / suite / condition / parent_key
        print(f"  [{idx+1}/{len(rows)}] {suite}/{condition} task={task_idx}", flush=True, end=" ")
        result = run_episode(vla_model, processor, vla_model, device, model_dtype, action_dim,
            unnorm_key, bm_key, task_idx, state_id, seed, condition, trigger_override,
            out_dir, args.source_commit, detector)
        if result.get("error"): errors += 1
        completed += 1
        print(f"success={result['task_success']} steps={result['n_steps']} attack={result['attack_frames']} "
              f"runtime={result['runtime_seconds']:.0f}s {'ERR:'+result['error'][:80] if result.get('error') else ''}", flush=True)

    print(f"Worker GPU {args.gpu} DONE: {completed} episodes, {errors} errors", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
