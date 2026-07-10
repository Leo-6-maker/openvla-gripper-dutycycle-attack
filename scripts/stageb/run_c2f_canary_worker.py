#!/usr/bin/env python3
"""C2f Track A online canary worker: action-space command intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

PROTOCOL_NAME = "C2F_TRACK_A_CMDOPEN_ACTION_SPACE"
PROTOCOL_VERSION = "2026-07-10.v2"
ATTACK_SPACE = "action_space_command_intervention"
COND_CLEAN = "CLEAN"
COND_TRUE = "TRUE_CMDOPEN_T10_C2F"
COND_RAND = "RAND_ACTION_NOISE_T10_C2F"
ATTACK_HORIZON = 10
EPSILON = 6.0 / 255.0
HASHED_MODEL_FILES = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "processing_prismatic.py", "configuration_prismatic.py",
    "modeling_prismatic.py",
}


def _git_provenance(repo: Path, enforce_clean: bool = True, expected_commit: str = "") -> Dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if expected_commit and commit != expected_commit:
        raise RuntimeError(f"Expected git commit {expected_commit}, got {commit}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    clean = status.strip() == ""
    if enforce_clean and not clean:
        raise RuntimeError(f"Refusing C2f Track A run from dirty worktree: {status.splitlines()[:20]}")
    return {"repo_path": str(repo), "repo_commit": commit, "repo_clean": clean, "repo_status_porcelain": status.splitlines()}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _selected_model_hashes(model_path: Path) -> Dict[str, str]:
    return {
        p.name: _sha256_file(p)
        for p in sorted(model_path.iterdir() if model_path.is_dir() else [])
        if p.is_file() and p.name in HASHED_MODEL_FILES
    }


def _norm_stats_keys(model: Any) -> List[str]:
    stats = getattr(model, "norm_stats", None)
    if stats is None and hasattr(model, "config"):
        stats = getattr(model.config, "norm_stats", None)
    return sorted(str(k) for k in stats.keys()) if isinstance(stats, dict) else []


def _resolve_unnorm_key(suite: str, keys: List[str]) -> str:
    if suite in keys:
        return suite
    matches = [k for k in keys if suite in k or suite.replace("_", "-") in k]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Cannot resolve unnorm_key for suite={suite}; norm_stats_keys={keys}")


def _rand_seed(parent_key: str, condition: str, attack_start: int) -> int:
    material = f"{PROTOCOL_VERSION}|{parent_key}|{condition}|{attack_start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def _rand_seed_material(parent_key: str, condition: str, attack_start: int) -> str:
    return f"{PROTOCOL_VERSION}|{parent_key}|{condition}|{attack_start}"


def _validate_goal_manifest(manifest_path: str, model_path: Path, unnorm_key: str) -> Dict[str, Any]:
    if not manifest_path:
        return {"policy_model_manifest_path": None, "policy_model_manifest_sha256": None}
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED":
        raise RuntimeError(f"Goal manifest is not PASS: {manifest.get('status')}")
    if Path(manifest.get("model_path", "")).resolve() != model_path:
        raise RuntimeError(f"Goal manifest model_path mismatch: {manifest.get('model_path')} != {model_path}")
    if manifest.get("unnorm_key") != unnorm_key:
        raise RuntimeError(f"Goal manifest unnorm_key mismatch: {manifest.get('unnorm_key')} != {unnorm_key}")
    if manifest.get("missing_referenced_shards"):
        raise RuntimeError(f"Goal manifest has missing shards: {manifest.get('missing_referenced_shards')}")
    return {"policy_model_manifest_path": str(path), "policy_model_manifest_sha256": _sha256_file(path)}


def _protocol_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "attack_space": ATTACK_SPACE,
        "direct_command_override": args.condition == COND_TRUE,
        "action_noise_semantics": "deterministic sha256-seeded normalized gaussian action-space noise" if args.condition == COND_RAND else None,
        "condition_semantics": {
            COND_CLEAN: "clean rollout, no intervention",
            COND_TRUE: "detector-triggered T10 action-space force-open gripper command",
            COND_RAND: "detector-triggered T10 action-space deterministic random action noise",
        },
        "tau_emit": args.tau_emit,
        "tau_suppress": args.tau_suppress,
        "tau_abstain": 0.5,
        "tau_primary": 0.5,
        "attack_horizon": ATTACK_HORIZON,
        "epsilon": EPSILON,
    }


def _action_evidence(clean_raw_action: np.ndarray, intervened_raw_action: np.ndarray,
                     clean_env_action: np.ndarray, executed_env_action: np.ndarray,
                     rand_noise: np.ndarray, condition: str) -> Dict[str, Any]:
    return {
        "clean_raw_action": clean_raw_action.astype(float).tolist(),
        "intervened_raw_action": intervened_raw_action.astype(float).tolist(),
        "executed_env_action": np.asarray(executed_env_action, dtype=np.float32).astype(float).tolist(),
        "action_delta": (intervened_raw_action - clean_raw_action).astype(float).tolist(),
        "rand_noise_vector": rand_noise.astype(float).tolist() if condition == COND_RAND else [],
        "rand_noise_norm": float(np.linalg.norm(rand_noise)) if condition == COND_RAND else 0.0,
        "clean_gripper_raw": float(clean_raw_action[-1]),
        "intervened_gripper_raw": float(intervened_raw_action[-1]),
        "clean_gripper_env": float(clean_env_action[-1]),
        "executed_gripper_env": float(np.asarray(executed_env_action)[-1]),
    }


def _failure_meta(args: argparse.Namespace, suite: str, task_idx: int, state_id: int,
                  error: BaseException, step_records: List[Dict[str, Any]],
                  extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "parent_key": args.parent_key,
        "condition": args.condition,
        "suite": suite,
        "task_index": task_idx,
        "state_id": state_id,
        "success": None,
        "runtime_valid": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "total_steps": len(step_records),
        **_protocol_meta(args),
        **extra,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-key", required=True)
    ap.add_argument("--condition", required=True, choices=[COND_CLEAN, COND_TRUE, COND_RAND])
    ap.add_argument("--checkpoint", required=True, help="C2fDetector .pt path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--tau-emit", type=float, default=0.33)
    ap.add_argument("--tau-suppress", type=float, default=0.67)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--expected-git-commit", required=True)
    ap.add_argument("--policy-model-manifest", default="")
    args = ap.parse_args()

    suite, task_str, state_str, _, _ = args.parent_key.split("/")
    task_idx = int(task_str.replace("task_", ""))
    state_id = int(state_str.replace("state_", ""))
    out_dir = Path(args.output_dir) / args.parent_key / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = None
    step_records: List[Dict[str, Any]] = []
    extra_meta: Dict[str, Any] = {}
    rc = 0

    try:
        extra_meta["git_provenance"] = _git_provenance(REPO, enforce_clean=True, expected_commit=args.expected_git_commit)
        extra_meta["git_commit"] = extra_meta["git_provenance"]["repo_commit"]
        from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS, _visible_gpu_id
        model_path = Path(SUITE_MODELS[suite]).resolve()
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForVision2Seq as AutoModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelCls

        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        vla_model = AutoModelCls.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map=device,
        ).eval()
        norm_keys = _norm_stats_keys(vla_model)
        unnorm_key = _resolve_unnorm_key(suite, norm_keys)
        if suite == "libero_goal":
            if not args.policy_model_manifest:
                raise RuntimeError("Goal episodes require --policy-model-manifest")
            extra_meta.update(_validate_goal_manifest(args.policy_model_manifest, model_path, unnorm_key))
        else:
            extra_meta.update({"policy_model_manifest_path": None, "policy_model_manifest_sha256": None})

        from gripper_attack.c2f_siglip_detector_runtime import C2fSigLIPDetectorRuntime, CANONICAL_25D_FEATURES
        detector = C2fSigLIPDetectorRuntime(
            checkpoint_path=args.checkpoint,
            openvla_model=vla_model,
            openvla_processor=processor,
            device=device,
            window=args.window,
            tau_emit=args.tau_emit,
            tau_suppress=args.tau_suppress,
        )

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
        streamer = SC5StreamingFeatureAdapterV2()
        eef_sid = env.sim.model.site_name2id("gripper0_grip_site")

        buffer_25d: List[np.ndarray] = []
        attack_window_start = -1
        attack_window_end = -1
        delivery_steps: List[int] = []
        rand_rng: Optional[np.random.Generator] = None
        success = False
        prev_eef = None

        ckpt_path = Path(args.checkpoint).resolve()
        worker_path = Path(__file__).resolve()
        runtime_path = REPO / "src" / "gripper_attack" / "c2f_siglip_detector_runtime.py"
        extra_meta.update({
            "task_language": task_language,
            "policy_model_path": str(model_path),
            "policy_model_file_hashes": _selected_model_hashes(model_path),
            "processor_path": str(model_path),
            "norm_stats_keys": norm_keys,
            "unnorm_key": unnorm_key,
            "detector_checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "worker_path": str(worker_path),
            "worker_sha256": _sha256_file(worker_path),
            "runtime_path": str(runtime_path),
            "runtime_sha256": _sha256_file(runtime_path),
                "random_seed": None,
                "random_seed_material": None,
            })

        for step in range(args.max_steps):
            rgb = np.asarray(obs["agentview_image"])
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
                rgb = np.moveaxis(rgb, 0, -1)
            rgb = rgb[..., :3].copy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb * 255.0, 0, 255) if np.nanmax(rgb) <= 1.0 else np.clip(rgb, 0, 255)).astype(np.uint8)
            if rgb.size == 0 or np.max(rgb[..., :3]) < 5:
                raise RuntimeError(f"C2f RGB capture failed at step {step}: blank image")

            gs = physical_gripper_state(env, obs)
            gq_raw = gs.get("qpos", np.zeros(2)) if isinstance(gs, dict) else np.zeros(2)
            action, _, _, _ = decode_with_scores(
                vla_model, processor, device, rgb, task_language, unnorm_key, 8,
                libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
                resize_size=224, drop_attention_mask=True,
            )
            clean_raw_action = np.asarray(action, dtype=np.float32)
            clean_env_action = postprocess_openvla_action_for_libero(clean_raw_action, enabled=True)
            env_action = clean_env_action.copy()

            eef_pos = env.sim.data.site_xpos[eef_sid]
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
            eef_vx = eef_x - prev_eef[0] if prev_eef is not None else 0.0
            eef_vy = eef_y - prev_eef[1] if prev_eef is not None else 0.0
            eef_vz = eef_z - prev_eef[2] if prev_eef is not None else 0.0
            prev_eef = (eef_x, eef_y, eef_z)

            raw_grip = float(action[-1])
            gq = float(gq_raw[0] + gq_raw[1]) if len(gq_raw) >= 2 else raw_grip
            gw = float(abs(gq_raw[0]) + abs(gq_raw[1])) if len(gq_raw) >= 2 else 0.0
            res = streamer.update(
                step_id=step, raw_gripper=raw_grip, env_gripper=-1.0 if raw_grip > 0.5 else 1.0,
                gripper_qpos=gq, gripper_opening_proxy=gw,
                eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
                action_dx=float(env_action[0]), action_dy=float(env_action[1]),
                action_dz=float(env_action[2]), action_gripper=raw_grip,
            )
            fv = {f: float(res["features"].get(f, 0.0) or 0.0) for f in CANONICAL_25D_FEATURES}
            buffer_25d.append(np.asarray([fv[f] for f in CANONICAL_25D_FEATURES], dtype=np.float32))
            det_out = detector.predict(buffer_25d, rgb, task_language, suite, task_idx)

            rec = {
                "step": step, "emit_p": det_out["emit_p"], "suppress_p": det_out["suppress_p"],
                "abstain_p": det_out["abstain_p"], "primary_p": det_out["primary_p"],
                "emitted": det_out["emitted"], "ready": det_out["ready"],
                "attack_delivered": False, "success": False,
            }

            if det_out["emitted"] and attack_window_start < 0 and args.condition in (COND_TRUE, COND_RAND):
                attack_window_start = step
                attack_window_end = step + ATTACK_HORIZON
                if args.condition == COND_RAND:
                    seed = _rand_seed(args.parent_key, args.condition, attack_window_start)
                    rand_rng = np.random.default_rng(seed)
                    extra_meta["random_seed"] = seed
                    extra_meta["random_seed_material"] = _rand_seed_material(args.parent_key, args.condition, attack_window_start)

            if attack_window_start >= 0 and step == attack_window_start:
                rec["attack_window_start"] = True

            if attack_window_start >= 0 and step < attack_window_end:
                attack_action = clean_raw_action.copy()
                noise = np.zeros_like(attack_action, dtype=np.float32)
                if args.condition == COND_TRUE:
                    attack_action[-1] = 1.0
                elif args.condition == COND_RAND:
                    assert rand_rng is not None
                    noise = rand_rng.standard_normal(attack_action.shape).astype(np.float32)
                    noise = noise / (np.linalg.norm(noise) + 1e-8) * EPSILON
                    attack_action = np.clip(attack_action + noise, -1.0, 1.0)
                env_action = postprocess_openvla_action_for_libero(attack_action, enabled=True)
                delivery_steps.append(step)
                rec["attack_delivered"] = True
                rec.update(_action_evidence(clean_raw_action, attack_action, clean_env_action, env_action, noise, args.condition))

            step_records.append(rec)
            obs, reward, done, info = env.step(env_action)
            if done:
                success = bool(info.get("success", False) or reward > 0.5)
                step_records[-1]["success"] = success
                break

        meta = {
            "parent_key": args.parent_key, "condition": args.condition,
            "suite": suite, "task_index": task_idx, "state_id": state_id,
            "total_steps": len(step_records), "success": success, "runtime_valid": True,
            "error_type": None, "error_message": None,
            "attack_window_start": attack_window_start, "attack_window_end": attack_window_end,
            "delivery_count": len(delivery_steps), "delivery_steps": delivery_steps,
            **_protocol_meta(args), **extra_meta,
        }
    except Exception as exc:
        rc = 1
        step_records.append({"step": -1, "error_type": type(exc).__name__, "error": str(exc),
                             "emitted": False, "attack_delivered": False,
                             "success": None, "runtime_valid": False})
        meta = _failure_meta(args, suite, task_idx, state_id, exc, step_records, extra_meta)
    finally:
        _write_json(out_dir / "episode_metadata.json", meta)
        with (out_dir / "step_records.jsonl").open("w", encoding="utf-8") as f:
            for rec in step_records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    print(f"{args.parent_key}/{args.condition}: steps={len(step_records)} "
          f"emit={meta.get('attack_window_start', -1) >= 0} "
          f"attack_window=[{meta.get('attack_window_start', -1)},{meta.get('attack_window_end', -1)}) "
          f"delivery_count={meta.get('delivery_count', 0)} success={meta.get('success')} "
          f"runtime_valid={meta.get('runtime_valid')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
