#!/usr/bin/env python3
"""Suite-agnostic SC5 clean rollout collector.

This runner is intentionally CLEAN-only. It loads the suite-matched OpenVLA
policy, records deployment-safe 25D SC5 features, runs the frozen Object-trained
SC5 detector, and writes auditable artifacts. It never constructs a visual
attacker, PGD, RAND, shuffled-gradient, or Teacher anchor.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

SUPPORTED_SUITES = {"libero_spatial", "libero_goal", "libero_10"}
DEFAULT_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_goal": 400,
    "libero_10": 400,
}
NUM_STEPS_WAIT = 10
PREPROCESS_KWARGS = {
    "libero_official_preprocess": False,
    "libero_preprocess_backend": "official_pil_lanczos",
    "center_crop": True,
    "resize_size": 224,
    "drop_attention_mask": True,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_jsonable(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(data)


def run_text(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(REPO), stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def git_status() -> dict[str, str]:
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "dirty_status": run_text(["git", "status", "--short"]),
    }


def gpu_snapshot() -> dict[str, str]:
    query = "index,uuid,name,memory.used,memory.total"
    return {
        "nvidia_smi_query": run_text(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            timeout=20,
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "openvla_attn_implementation": os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", ""),
    }


def fail_if_output_exists(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"--output_dir exists and is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=sorted(SUPPORTED_SUITES))
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--unnorm_key", required=True)
    ap.add_argument("--task_idx", type=int, required=True)
    ap.add_argument("--state_id", type=int, required=True)
    ap.add_argument("--eval_seed", type=int, required=True)
    ap.add_argument("--detector_path", required=True)
    ap.add_argument("--source_commit", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--render_gpu", type=int, required=True)
    ap.add_argument("--max_steps_override", type=int, default=0)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Validate arguments and write a plan manifest only.")
    ap.add_argument("--manifest_only", action="store_true", help="Alias for --dry_run.")
    ap.add_argument("--no_gpu", action="store_true", help="Require dry-run mode; fail for actual rollouts.")
    return ap.parse_args()


def max_steps_for_suite(suite: str, override: int = 0) -> tuple[int, str]:
    if override:
        return int(override), "cli_override"
    return int(DEFAULT_MAX_STEPS[suite]), "suite_default"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_base_manifest(args: argparse.Namespace) -> dict[str, Any]:
    max_steps, max_steps_source = max_steps_for_suite(args.suite, args.max_steps_override)
    return {
        "runner": "run_sc5_cross_suite_clean.py",
        "condition": "CLEAN",
        "attack_enabled": False,
        "vis_enabled": False,
        "rand_enabled": False,
        "teacher_anchor_required": False,
        "suite": args.suite,
        "task_idx": int(args.task_idx),
        "state_id": int(args.state_id),
        "eval_seed": int(args.eval_seed),
        "model_path": str(args.model_path),
        "unnorm_key": str(args.unnorm_key),
        "detector_path": str(args.detector_path),
        "source_commit": str(args.source_commit),
        "render_gpu": int(args.render_gpu),
        "max_steps": int(max_steps),
        "max_steps_source": max_steps_source,
        "num_steps_wait": NUM_STEPS_WAIT,
        "preprocess_kwargs": dict(PREPROCESS_KWARGS),
        "git": git_status(),
        "gpu_snapshot": gpu_snapshot(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
    }


def tensor_sha256(tensor: Any) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    files = []
    if model_path.exists():
        for path in sorted(model_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".safetensors", ".bin", ".model", ".txt"}:
                files.append({
                    "path": str(path.relative_to(model_path)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    return {
        "path": str(model_path),
        "exists": model_path.exists(),
        "file_count_hashed": len(files),
        "sha256": sha256_jsonable(files),
        "files": files[:200],
        "truncated": len(files) > 200,
    }


def write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> str:
    if not frames:
        return "NO_FRAMES"
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return f"IMAGEIO_UNAVAILABLE:{type(exc).__name__}"
    imageio.mimwrite(path, frames, fps=fps)
    return "WROTE"


def make_invalid_privileged_sidecar(reason: str) -> dict[str, Any]:
    return {
        "privileged_valid": False,
        "teacher_abstain": True,
        "reason": reason,
        "object_id_used": "",
        "object_pose_used": False,
        "target_pose_used": False,
        "claim": "no_privileged_teacher_anchor_available_for_cross_suite_clean_collection",
    }


def build_detector_rows(step_row: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    out = {
        "step": step_row["step"],
        "feat_valid": step_row["feat_valid"],
        "feat_error": step_row["feat_error"],
        "detector_state": step_row["detector_state"],
        "corridor_p": step_row["corridor_p"],
        "release_p": step_row["release_p"],
        "pred_phase": step_row["pred_phase"],
        "emit_step": step_row["mlp_emit"],
        "emitted": step_row["mlp_triggered"],
    }
    for name in feature_names:
        out[name] = step_row.get("f_" + name, "")
    return out


def run_clean_collection(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    fail_if_output_exists(out)
    manifest = build_base_manifest(args)

    if args.no_gpu and not (args.dry_run or args.manifest_only):
        raise SystemExit("--no_gpu requires --dry_run or --manifest_only")
    if args.dry_run or args.manifest_only:
        manifest["dry_run"] = True
        write_json(out / "episode_manifest.json", manifest)
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    from v4_run_eval_openvla import (
        decode_with_scores,
        physical_gripper_state,
        postprocess_openvla_action_for_libero,
        prompt,
    )
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens

    np.random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.eval_seed))

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    visible = torch.cuda.device_count()
    if visible <= 0:
        raise SystemExit("No CUDA device visible for actual clean rollout.")
    model = AutoModelCls.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={idx: "10000MiB" for idx in range(visible)} | {"cpu": "128GiB"},
        attn_implementation="eager",
    )
    model.eval()
    model_dtype = next(model.parameters()).dtype
    device = "cuda:0"
    for v in getattr(model, "hf_device_map", {}).values():
        if isinstance(v, int):
            device = f"cuda:{v}"
            break
    action_dim = int(model.get_action_dim(args.unnorm_key))
    action_stats = model.get_action_stats(args.unnorm_key)

    detector = SC5DetectorRuntime(args.detector_path, tau_corridor=0.3, tau_release=0.3, guard=5)
    streamer = SC5StreamingFeatureAdapterV2()

    bm = benchmark.get_benchmark_dict()
    suite_obj = bm[args.suite]()
    task_obj = suite_obj.get_task(int(args.task_idx))
    init_states = suite_obj.get_task_init_states(int(args.task_idx))
    if int(args.state_id) >= len(init_states):
        raise SystemExit(f"state_id {args.state_id} out of range: n={len(init_states)}")
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    instruction = str(task_obj.language)
    max_steps, _ = max_steps_for_suite(args.suite, args.max_steps_override)

    manifest.update({
        "dry_run": False,
        "task_name": getattr(task_obj, "name", ""),
        "instruction": instruction,
        "bddl_file": bddl,
        "action_dim": action_dim,
        "model_dtype": str(model_dtype),
        "hf_device_map": getattr(model, "hf_device_map", {}),
        "model_fingerprint": model_fingerprint(Path(args.model_path)),
        "detector_checkpoint_sha256": detector.checkpoint_sha256,
        "detector_dataset_sha256": detector.dataset_sha256,
        "detector_feature_schema_version": "SC5_FEATURES_V2_25D",
        "detector_feature_order": list(SC5_FEATURES),
        "action_stats_sha256": sha256_jsonable(action_stats),
    })
    write_json(out / "episode_manifest.json", manifest)

    env, obs = build_v4_exact_env(bddl, int(args.render_gpu), max_steps, NUM_STEPS_WAIT)
    obs = env.set_init_state(init_states[int(args.state_id)])
    env, obs = apply_dummy_wait(env, obs, NUM_STEPS_WAIT)

    eef_init = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
    prev_eef = (float(eef_init[0]), float(eef_init[1]), float(eef_init[2]))
    first_valid_step = -1
    invalid_steps = 0
    telemetry: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    frame_index: list[dict[str, Any]] = []
    clean_tokens_first: list[int] | None = None
    prompt_ids_sha = ""
    task_success = False

    for step in range(max_steps):
        if "agentview_image" not in obs:
            break
        raw = np.asarray(obs["agentview_image"]).copy()
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        frames.append(raw)
        frame_index.append({
            "step": step,
            "frame_array_index": len(frames) - 1,
            "raw_agentview_sha256": sha256_bytes(raw.tobytes()),
            "shape": "x".join(map(str, raw.shape)),
            "dtype": str(raw.dtype),
        })

        proc_image = prepare_openvla_image(raw, **{k: v for k, v in PREPROCESS_KWARGS.items() if k != "drop_attention_mask"})
        proc_inputs = processor(prompt(instruction.lower()), proc_image, return_tensors="pt")
        proc_inputs.pop("attention_mask", None)
        if "input_ids" in proc_inputs and not torch.all(proc_inputs["input_ids"][:, -1] == 29871):
            proc_inputs["input_ids"] = torch.cat(
                (proc_inputs["input_ids"], torch.unsqueeze(torch.tensor([29871]).long(), dim=0)),
                dim=1,
            )
        if step == 0:
            prompt_ids_sha = tensor_sha256(proc_inputs["input_ids"])
            manifest["prompt"] = prompt(instruction.lower())
            manifest["prompt_token_ids_sha256"] = prompt_ids_sha
            manifest["processor_pixel_values_shape"] = list(proc_inputs["pixel_values"].shape)
            manifest["processor_pixel_values_dtype"] = str(proc_inputs["pixel_values"].dtype)
            manifest["processor_pixel_values_step0_sha256"] = tensor_sha256(proc_inputs["pixel_values"])
            write_json(out / "episode_manifest.json", manifest)

        t0 = time.perf_counter()
        action, _prefix_logits, _dt, gen = decode_with_scores(
            model,
            processor,
            device,
            raw,
            instruction,
            args.unnorm_key,
            action_dim,
            **PREPROCESS_KWARGS,
        )
        model_ms = (time.perf_counter() - t0) * 1000.0
        tokens = extract_exact_new_tokens(
            gen.sequences,
            prompt_len=int(getattr(gen, "prompt_len", gen.sequences.shape[1] - action_dim)),
            expected_new_tokens=action_dim,
        )
        token_list = [int(t) for t in tokens]
        if clean_tokens_first is None:
            clean_tokens_first = token_list

        env_action = postprocess_openvla_action_for_libero(np.asarray(action, dtype=np.float32), enabled=True)
        raw_grip = float(action[-1])
        env_grip = float(env_action[-1])
        gs = physical_gripper_state(env, obs)
        qpos = gs.get("qpos", []) if gs else []
        q7 = float(qpos[0]) if len(qpos) > 0 else float("nan")
        q8 = float(qpos[1]) if len(qpos) > 1 else float("nan")
        qpos_sum = q7 + q8 if np.isfinite(q7) and np.isfinite(q8) else float("nan")
        opening_proxy = abs(q7) + abs(q8) if np.isfinite(q7) and np.isfinite(q8) else float("nan")
        eef_pos = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
        if prev_eef is not None and np.all(np.isfinite([eef_x, eef_y, eef_z])):
            eef_vx = eef_x - prev_eef[0]
            eef_vy = eef_y - prev_eef[1]
            eef_vz = eef_z - prev_eef[2]
        else:
            eef_vx = eef_vy = eef_vz = float("nan")
        if np.all(np.isfinite([eef_x, eef_y, eef_z])):
            prev_eef = (eef_x, eef_y, eef_z)

        feat_valid = False
        feat_error = ""
        feat_25d: dict[str, float] = {}
        det_state = detector.state
        det_cp = det_rp = det_pp = ""
        try:
            feat_res = streamer.update(
                step_id=step,
                raw_gripper=raw_grip,
                env_gripper=env_grip,
                gripper_qpos=float(qpos_sum),
                gripper_opening_proxy=float(opening_proxy),
                eef_x=eef_x,
                eef_y=eef_y,
                eef_z=eef_z,
                eef_vx=float(eef_vx),
                eef_vy=float(eef_vy),
                eef_vz=float(eef_vz),
                action_dx=float(action[0]),
                action_dy=float(action[1]),
                action_dz=float(action[2]),
                action_gripper=raw_grip,
            )
        except Exception as exc:
            feat_res = {"valid": False, "error": f"streamer_error:{type(exc).__name__}:{str(exc)[:120]}"}
        feat_valid = bool(feat_res.get("valid", False))
        feat_error = str(feat_res.get("error", ""))
        if feat_valid:
            feat_25d = dict(feat_res["features"])
            if first_valid_step < 0:
                first_valid_step = step
            decision = detector.update(feat_25d, step)
            det_state = decision["state"]
            det_cp = decision.get("corridor_p")
            det_rp = decision.get("release_p")
            det_pp = decision.get("pred_phase")
        else:
            invalid_steps += 1

        row: dict[str, Any] = {
            "step": step,
            "condition": "CLEAN",
            "suite": args.suite,
            "task_idx": int(args.task_idx),
            "state_id": int(args.state_id),
            "eval_seed": int(args.eval_seed),
            "raw_gripper": raw_grip,
            "env_gripper": env_grip,
            "gripper_qpos": qpos_sum,
            "gripper_opening_proxy": opening_proxy,
            "eef_x": eef_x,
            "eef_y": eef_y,
            "eef_z": eef_z,
            "eef_vx": eef_vx,
            "eef_vy": eef_vy,
            "eef_vz": eef_vz,
            "action_dx": float(action[0]),
            "action_dy": float(action[1]),
            "action_dz": float(action[2]),
            "action_gripper": raw_grip,
            "action_vector_json": json.dumps([float(x) for x in np.asarray(action).tolist()]),
            "env_action_json": json.dumps([float(x) for x in np.asarray(env_action).tolist()]),
            "exact_new_tokens_json": json.dumps(token_list),
            "gripper_token": token_list[-1] if token_list else "",
            "feat_valid": feat_valid,
            "feat_error": feat_error,
            "detector_state": det_state,
            "corridor_p": det_cp,
            "release_p": det_rp,
            "pred_phase": det_pp,
            "mlp_emit": detector.emit_step,
            "mlp_triggered": detector.emitted,
            "model_ms": round(model_ms, 3),
        }
        for fn in SC5_FEATURES:
            row["f_" + fn] = feat_25d.get(fn, "")
        telemetry.append(row)

        obs, _reward, done, _info = env.step(env_action)
        if bool(done):
            break

    try:
        task_success = bool(env.check_success()) if hasattr(env, "check_success") else False
    finally:
        env.close()

    summary = {
        "condition": "CLEAN",
        "suite": args.suite,
        "task_idx": int(args.task_idx),
        "state_id": int(args.state_id),
        "eval_seed": int(args.eval_seed),
        "n_steps": len(telemetry),
        "task_success": task_success,
        "invalid_feature_steps": invalid_steps,
        "first_valid_step": first_valid_step,
        "mlp_triggered": detector.emitted,
        "mlp_emit_step": detector.emit_step,
        "clean_exact_7_tokens": clean_tokens_first or [],
        "clean_arm_prefix": (clean_tokens_first or [])[:6],
        "clean_gripper_token": (clean_tokens_first or [""])[-1] if clean_tokens_first else "",
        "checkpoint_sha256": detector.checkpoint_sha256,
        "dataset_sha256": detector.dataset_sha256,
        "privileged_valid": False,
        "teacher_abstain": True,
        "vis_or_rand_run": False,
        "manual_anchor_used": False,
    }

    write_csv(out / "step_telemetry.csv", telemetry)
    write_csv(out / "detector_telemetry.csv", [build_detector_rows(r, SC5_FEATURES) for r in telemetry])
    write_csv(out / "frame_index.csv", frame_index)
    write_json(out / "episode_summary.json", summary)
    write_json(out / "privileged_sidecar.json", make_invalid_privileged_sidecar("cross_suite_clean_collector_has_no_reliable_teacher_object_binding"))

    if frames:
        np.savez_compressed(out / "agentview_frames_uint8.npz", agentview=np.stack(frames, axis=0))
    video_status = {}
    if args.save_video:
        video_status["rollout_raw.mp4"] = write_video(out / "rollout_raw.mp4", frames, fps=10)
        video_status["rollout_overlay.mp4"] = write_video(out / "rollout_overlay.mp4", frames, fps=10)
    else:
        video_status["rollout_raw.mp4"] = "DISABLED"
        video_status["rollout_overlay.mp4"] = "DISABLED"
    write_json(out / "video_manifest.json", {"video_status": video_status, "overlay_mode": "raw_duplicate_no_text"})

    manifest["final_summary_sha256"] = sha256_jsonable(summary)
    manifest["artifact_files"] = []
    write_json(out / "episode_manifest.json", manifest)
    seal_artifacts(out)

    print(json.dumps({
        "result": "CLEAN_COLLECTION_DONE",
        "output_dir": str(out),
        "steps": len(telemetry),
        "success": task_success,
        "invalid_feature_steps": invalid_steps,
        "mlp_emit_step": detector.emit_step,
    }, sort_keys=True))


def seal_artifacts(out: Path) -> None:
    rows = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out).as_posix()
        if rel == "artifact_sha256.json":
            continue
        rows.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(out / "artifact_sha256.json", {"files": rows, "recursive_sha256": sha256_jsonable(rows)})


def main() -> None:
    args = parse_args()
    run_clean_collection(args)


if __name__ == "__main__":
    main()
