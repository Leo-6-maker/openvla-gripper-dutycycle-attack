#!/usr/bin/env python3
"""D7 Table1 cross-suite bridge — real episode runner.

Parameterizes run_v2_vis_sc5_mlp_bridge.py for four LIBERO suites.
Loads OpenVLA per suite, runs LIBERO env, applies C2e3 GRU detector,
executes VIS/RAND/ORACLE attack, writes telemetry.

Execution patterns copied from run_v2_vis_sc5_mlp_bridge.py (verified).
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

# ============ Suite Registry ============
SUITE_MODELS = {
    "libero_10": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-10",
    "libero_goal": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-goal",
    "libero_object": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "libero_spatial": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-spatial",
}
SUITE_UNNORM = {s: s for s in SUITE_MODELS}
SUITE_BM = {s: s for s in SUITE_MODELS}
K = 10; EPSILON = 0.023529411764705882; TARGET_TOKEN = 31744; ARM_GATE = 5; PGD_STEPS = 20; MAX_STEPS = 400


def sha256_str(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_c2e3_detector(checkpoint_path: str):
    """Load C2e3 frozen GRU detector. Returns (model, tau_emit, tau_suppress, ckpt)."""
    import numpy as np, torch
    from torch import nn

    class GRU(nn.Module):
        def __init__(self, nf=25, nc=108, hidden=128):
            super().__init__()
            self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
            self.head = nn.Linear(hidden + nc, 2)
        def forward(self, xt, xc):
            _, h = self.gru(xt); return self.head(torch.cat([h[-1], xc], dim=1))

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"Checkpoint keys: {list(ckpt.keys())} — need model_state_dict or state_dict")
    cfg = ckpt["config"]
    cfg_th = ckpt.get("threshold", {})
    model = GRU(25, 108, cfg.get("channels", cfg.get("hidden", 128)))
    model.load_state_dict(state)
    model.cpu().eval()
    return model, float(cfg_th.get("tau_emit", 0.33)), float(cfg_th.get("tau_suppress", 0.67))


def run_episode(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute one episode: OpenVLA + LIBERO + C2e3 GRU + attack (copied from v2 bridge patterns)."""
    import numpy as np, torch

    model_path = args.model_path or SUITE_MODELS[args.suite]
    unnorm_key = args.unnorm_key or SUITE_UNNORM[args.suite]
    bm_key = SUITE_BM[args.suite]

    out_dir = Path(args.output_dir) / args.suite / args.condition / args.parent_key
    out_dir.mkdir(parents=True, exist_ok=True)

    is_attack = args.condition != "CLEAN"
    is_rand = args.condition == "RAND_T10"
    is_oracle = args.condition == "COMMAND_OPEN_ORACLE"
    attack_frames = K if (is_attack or is_oracle) else 0

    start_time = time.time()
    result = {"suite": args.suite, "condition": args.condition, "parent_key": args.parent_key,
        "task_idx": args.task_idx, "state_id": args.state_id, "seed": args.seed,
        "task_success": False, "n_steps": 0, "detector_emitted": False, "emit_step": -1,
        "attack_frames": 0, "failure_taxonomy": "unknown", "error": "", "runtime_seconds": 0.0,
        "source_commit": args.source_commit}

    try:
        # ── Load C2e3 GRU detector ──
        det_model, tau_emit, tau_suppress = load_c2e3_detector(args.detector_checkpoint)

        # ── OpenVLA model (copied from v2 bridge) ──
        from transformers import AutoProcessor
        try: from transformers import AutoModelForImageTextToText as AutoModelCls
        except Exception: from transformers import AutoModelForVision2Seq as AutoModelCls
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        _dtype_name = os.environ.get("OPENVLA_DTYPE", "bfloat16")
        _attn_name = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager")
        _dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        _torch_dtype = _dtype_map.get(_dtype_name, torch.bfloat16)
        model = AutoModelCls.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=_torch_dtype, low_cpu_mem_usage=True, device_map="cuda:0", attn_implementation=_attn_name)
        device = "cuda:0"; model_dtype = next(model.parameters()).dtype
        action_dim = int(model.get_action_dim(unnorm_key))
        print(f"OpenVLA: {model_path} dim={action_dim} suite={args.suite}")

        # ── Persistent attacker (copied from v2 bridge) ──
        attacker = None
        if is_attack and not is_rand and not is_oracle:
            from gripper_attack.attack_adapter import OpenVLAVisualAttacker
            opt = {"method": "token_prefix_pgd", "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                   "epsilon": EPSILON, "num_steps": PGD_STEPS, "step_size": EPSILON * 0.075,
                   "random_start": True, "prefix_refresh_interval": 1,
                   "surrogate_score_path": "cached_autoregressive_generate_v1",
                   "strict_route": True, "allow_fallback": False, "temporal_init": "prev_delta",
                   "target_token_id": TARGET_TOKEN, "target_execution_class": "CLIP_MEDIATED_OPEN",
                   "gripper_margin": 5.0, "arm_preserve_weight": 0.5, "arm_gate_min_match_count": ARM_GATE}
            attacker = OpenVLAVisualAttacker(model=model, processor=processor, config={"attack_optimizer": opt},
                seed=args.seed, preprocess_kwargs={"libero_preprocess_backend": "upstream_tf_jpeg", "center_crop": True, "resize_size": 224}, device=device)

        # ── LIBERO env (copied from v2 bridge) ──
        from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero
        from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
        from libero.libero import benchmark, get_libero_path
        bm = benchmark.get_benchmark_dict(); suite_obj = bm[bm_key]()
        task_obj = suite_obj.get_task(args.task_idx)
        init_states = suite_obj.get_task_init_states(args.task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
        instruction = task_obj.language
        env, obs = build_v4_exact_env(bddl, args.render_gpu, MAX_STEPS, 10)
        obs = env.set_init_state(init_states[args.state_id])
        env, obs = apply_dummy_wait(env, obs, 10)

        # Object site (best-effort, copied from v2 bridge)
        _obj_key = task_obj.name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")
        try:
            obj_sid = env.sim.model.site_name2id(f"{_obj_key}_1_default_site")
        except Exception:
            obj_sid = None

        # ── Streaming 25D features (copied from v2 bridge) ──
        from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
        from v4_run_eval_openvla import physical_gripper_state
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        _streamer = SC5StreamingFeatureAdapterV2()
        _gr_emit = -1; _gr_ep = 0.0; _gr_sp = 0.0
        _eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        _prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
        W = 16; _feature_buffer = []
        _invalid_steps = 0; _first_valid_step = -1

        telemetry = []; attack_count = 0

        for step in range(MAX_STEPS):
            if "agentview_image" not in obs: break
            raw = np.asarray(obs["agentview_image"]).copy()

            # Physical gripper state (copied from v2 bridge)
            gs = physical_gripper_state(env, obs)
            q7 = float(gs["qpos"][0]) if gs and len(gs.get("qpos", [])) > 0 else float("nan")
            q8 = float(gs["qpos"][1]) if gs and len(gs.get("qpos", [])) > 1 else float("nan")
            qpos_sum = q7 + q8 if not (np.isnan(q7) or np.isnan(q8)) else float("nan")
            eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

            # Clean decode (copied from v2 bridge)
            action, _, _, gen = decode_with_scores(model, processor, device, raw, instruction, unnorm_key, 8,
                libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224, drop_attention_mask=True)
            raw_grip = float(action[-1])
            env_action_final = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)

            # ── C2e3 GRU ONLINE TRIGGER (replaces MLP) ──
            if not (bool)(_gr_emit >= 0):
                # Causal EEF velocity
                eef_valid = np.all(np.isfinite([eef_x, eef_y, eef_z]))
                if _prev_eef is not None and eef_valid:
                    _vx, _vy, _vz = eef_x - _prev_eef[0], eef_y - _prev_eef[1], eef_z - _prev_eef[2]
                else:
                    _vx = _vy = _vz = float("nan")
                if eef_valid: _prev_eef = (eef_x, eef_y, eef_z)

                gripper_w = abs(q7) + abs(q8) if not (np.isnan(q7) or np.isnan(q8) or np.isnan(qpos_sum)) else float("nan")
                gripper_q = float(qpos_sum) if not np.isnan(qpos_sum) else float("nan")

                # Streamer update (copied from v2 bridge)
                try:
                    _res = _streamer.update(step_id=step, raw_gripper=raw_grip, env_gripper=-1.0 if raw_grip > 0.5 else 1.0,
                        gripper_qpos=gripper_q, gripper_opening_proxy=gripper_w,
                        eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                        eef_vx=_vx, eef_vy=_vy, eef_vz=_vz,
                        action_dx=float(action[0]), action_dy=float(action[1]),
                        action_dz=float(action[2]), action_gripper=raw_grip)
                except ValueError as e:
                    _res = {"valid": False, "error": f"step_sequence:{str(e)[:80]}"}
                except Exception as e:
                    _res = {"valid": False, "error": f"streamer_error:{type(e).__name__}"}

                _feat_valid = _res.get("valid", False)
                if _feat_valid: _first_valid_step = step if _first_valid_step < 0 else _first_valid_step
                else: _invalid_steps += 1

                # GRU trigger check (W=16 window)
                if _feat_valid:
                    SC5_F = ["gripper_command","gripper_qpos","gripper_opening_proxy","eef_x","eef_y","eef_z",
                              "eef_vx","eef_vy","eef_vz","action_dx","action_dy","action_dz","action_gripper",
                              "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
                              "close_onset","time_since_close","eef_speed","eef_z_delta_since_close",
                              "qpos_delta_1","qpos_delta_3","opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5"]
                    fv = [float(_res["features"].get(f, 0.0) or 0.0) for f in SC5_F]
                    _feature_buffer.append(fv)
                    if len(_feature_buffer) >= W:
                        window = np.array(_feature_buffer[-W:], dtype=np.float32).reshape(1, W, 25)
                        ctx = np.zeros((1, 108), dtype=np.float32)
                        with torch.no_grad():
                            logits = det_model(torch.from_numpy(window), torch.from_numpy(ctx)).numpy()[0]
                        _gr_ep = 1.0 / (1.0 + np.exp(-np.clip(float(logits[0]), -50, 50)))
                        _gr_sp = 1.0 / (1.0 + np.exp(-np.clip(float(logits[1]), -50, 50)))
                        if _gr_ep >= tau_emit and _gr_sp <= tau_suppress:
                            _gr_emit = step

            # ── ATTACK (copied from v2 bridge, only trigger changed) ──
            attack_this = False; adv_token = None; adv_arm = 0
            _trigger_step = args.trigger_step_override if args.trigger_step_override >= 0 else _gr_emit
            _clean_candidate = np.asarray(action, dtype=np.float32)

            if is_attack and _trigger_step >= 0 and step >= _trigger_step and attack_count < attack_frames:
                if is_rand:
                    from gripper_attack.m3_controls import sample_processor_delta, project_and_cast_processor_values
                    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack
                    proc_image = prepare_openvla_image_for_attack(raw, libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224)
                    inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
                    inputs.pop("attention_mask", None)
                    iids = inputs["input_ids"].to(device)
                    if not torch.all(iids[:, -1] == 29871):
                        iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
                    x = inputs["pixel_values"].to(device=device, dtype=model_dtype)
                    delta = sample_processor_delta(x.shape, epsilon=EPSILON, seed=args.seed + 100000 + attack_count, dtype=torch.float32, device=x.device)
                    proj, _ = project_and_cast_processor_values(x, delta, epsilon=EPSILON, candidate_is_delta=True)
                    adv_pv = proj.detach().to(dtype=model_dtype)
                    with torch.inference_mode():
                        go = model.generate(input_ids=iids, pixel_values=adv_pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    adv_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
                elif is_oracle:
                    # Command-open oracle: set env action gripper to -1.0 (open)
                    oracle_action = list(action)
                    oracle_action[-1] = -1.0
                    env_action_final = np.array(oracle_action, dtype=np.float32)
                    attack_this = True
                else:
                    # TRUE_T10: VIS targeted attack (copied from v2 bridge)
                    from gripper_attack.attack_adapter import prepare_openvla_image_for_attack, get_adv_inputs_from_attack_result
                    clean_action_np = np.asarray(action, dtype=np.float32)
                    proc_image = prepare_openvla_image_for_attack(raw, libero_preprocess_backend="upstream_tf_jpeg", center_crop=True, resize_size=224)
                    inputs = processor(prompt(instruction), proc_image, return_tensors="pt")
                    inputs.pop("attention_mask", None)
                    iids = inputs["input_ids"].to(device)
                    if not torch.all(iids[:, -1] == 29871):
                        iids = torch.cat([iids, torch.tensor([[29871]], dtype=torch.long, device=iids.device)], dim=1)
                    pv = inputs["pixel_values"].to(device=device, dtype=model_dtype)
                    with torch.inference_mode():
                        go = model.generate(input_ids=iids, pixel_values=pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    clean_tokens = extract_exact_new_tokens(go.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)
                    clean_gen = type("CleanGen", (), {})()
                    clean_gen.sequences = torch.tensor([iids[0].detach().cpu().tolist() + [int(t) for t in clean_tokens]], dtype=torch.long, device=device)
                    clean_gen.scores = []
                    attack_result = attacker.attack(raw, instruction, clean_action_np, clean_action_np, clean_gen, unnorm_key=unnorm_key)
                    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
                    adv_pv = adv_inputs["pixel_values"]
                    with torch.inference_mode():
                        go2 = model.generate(input_ids=iids, pixel_values=adv_pv, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    adv_tokens = extract_exact_new_tokens(go2.sequences, prompt_len=int(iids.shape[1]), expected_new_tokens=action_dim)

                if adv_tokens is not None:
                    from gripper_attack.m3_controls import decode_action_tokens, compute_arm_selection_mask
                    adv_decoded = decode_action_tokens(adv_tokens, unnorm_key=unnorm_key)
                    _clean_6 = np.asarray(action[:6], dtype=np.float32)
                    _adv_all = np.asarray(adv_decoded, dtype=np.float32)
                    _arm_mask = compute_arm_selection_mask(_clean_6, _adv_all[:6], ARM_GATE)
                    adv_arm = int(_arm_mask.sum())
                    _final_6 = np.where(_arm_mask, _adv_all[:6], _clean_6)
                    _final = np.concatenate([_final_6, _adv_all[6:]])
                    env_action_final = postprocess_openvla_action_for_libero(_final, enabled=True)
                    attack_this = True
                    attack_count += 1

            # ── env.step (exactly once per timestep, copied from v2 bridge) ──
            obs, reward, done, info = env.step(env_action_final)

            # Telemetry
            obj_z = float(env.sim.data.site_xpos[obj_sid][2]) if obj_sid is not None else float("nan")
            telemetry.append({"step": step, "emitted": _gr_emit >= 0, "emit_step": _gr_emit,
                "attack_this": attack_this, "attack_count": attack_count,
                "raw_gripper": raw_grip, "env_gripper": -1.0 if raw_grip > 0.5 else 1.0,
                "eef_x": eef_x, "eef_y": eef_y, "eef_z": eef_z,
                "object_z": obj_z,
                "gr_emit_p": _gr_ep, "gr_suppress_p": _gr_sp})

            if done: break

        # ── Post-episode ──
        success = bool(reward > 0 if isinstance(reward, (int, float)) else False)
        result.update({"task_success": success, "n_steps": len(telemetry),
            "detector_emitted": _gr_emit >= 0, "emit_step": _gr_emit,
            "attack_frames": attack_count, "failure_taxonomy": "success" if success else ("no_trigger" if _gr_emit < 0 else "attack_ineffective"),
            "runtime_seconds": time.time() - start_time})
        if len(telemetry) > 0:
            result["token_open_duty"] = sum(1 for t in telemetry if t.get("raw_gripper", 0) > 0.5) / len(telemetry)
            result["env_open_duty"] = sum(1 for t in telemetry if t.get("env_gripper", 0) < -0.5) / len(telemetry)

        # Write telemetry
        if telemetry:
            with open(out_dir / "step_telemetry.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys())); w.writeheader(); w.writerows(telemetry)
        env.close()

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["failure_taxonomy"] = "runtime_error"
        traceback.print_exc()

    # ── Write outputs ──
    result["detector_checkpoint_sha256"] = sha256_file(Path(args.detector_checkpoint)) if Path(args.detector_checkpoint).exists() else ""
    result["threshold_sha256"] = sha256_file(Path(args.threshold_json)) if Path(args.threshold_json).exists() else ""
    write_json(out_dir / "episode_summary.json", result)
    write_json(out_dir / "episode_manifest.json", {"suite": args.suite, "condition": args.condition, "parent_key": args.parent_key, "task_idx": args.task_idx, "state_id": args.state_id, "seed": args.seed, "model_path": model_path, "detector_checkpoint": args.detector_checkpoint, "source_commit": args.source_commit, "timestamp_unix": time.time()})
    artifact_sha = {}
    for fn in sorted(out_dir.glob("*")):
        if fn.is_file() and fn.name != "artifact_sha256.json": artifact_sha[fn.name] = sha256_file(fn)
    write_json(out_dir / "artifact_sha256.json", artifact_sha)
    return result


def main():
    ap = argparse.ArgumentParser(description="D7 Table1 cross-suite bridge (single episode)")
    ap.add_argument("--suite", required=True, choices=list(SUITE_MODELS.keys()))
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--task-idx", type=int, required=True)
    ap.add_argument("--state-id", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", required=True, choices=["CLEAN","TRUE_T10","RAND_T10","COMMAND_OPEN_ORACLE"])
    ap.add_argument("--parent-key", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--detector-checkpoint", required=True); ap.add_argument("--threshold-json", required=True)
    ap.add_argument("--trigger-step-override", type=int, default=-1)
    ap.add_argument("--source-commit", required=True); ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "suite": args.suite, "condition": args.condition, "parent_key": args.parent_key, "model_path": args.model_path or SUITE_MODELS.get(args.suite, "")}, indent=2)); return 0

    print(f"D7 Bridge: {args.suite}/{args.condition} parent={args.parent_key} task={args.task_idx}")
    result = run_episode(args)
    status = "SUCCESS" if not result.get("error") else "ERROR"
    print(f"  {status}: success={result['task_success']} emitted={result['detector_emitted']} steps={result['n_steps']} attack_frames={result['attack_frames']} runtime={result['runtime_seconds']:.0f}s")
    if result.get("error"): print(f"  ERROR: {result['error'][:200]}")
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
