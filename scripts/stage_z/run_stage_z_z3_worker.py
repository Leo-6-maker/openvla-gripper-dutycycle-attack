#!/usr/bin/env python3
"""Execute sealed-action Z3 branches without loading a policy model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_stage_z_z1_runtime_canary as z1
from gripper_attack.stage_v_m3_5_physical_taxonomy import aperture_metric, bind_object_taxonomy, telemetry_from_env
from stage_z_preparation.action_semantics import validate_action_pair
from stage_z_preparation.z3_contract import H_PHYS, MODEL_M0, MODEL_M1, MODEL_M2, command_open_action, physical_class, physical_label, arm_delta_linf


MODELS = (MODEL_M0, MODEL_M1, MODEL_M2)
ARMS = ("CLEAN_BRANCH_CRITICAL", "COMMAND_OPEN_T3_CRITICAL", "COMMAND_OPEN_T5_CRITICAL", "COMMAND_OPEN_T10_CRITICAL", "COMMAND_OPEN_T5_NONCRITICAL_CONTROL")
DOSE = {"CLEAN_BRANCH_CRITICAL": 0, "COMMAND_OPEN_T3_CRITICAL": 3, "COMMAND_OPEN_T5_CRITICAL": 5, "COMMAND_OPEN_T10_CRITICAL": 10, "COMMAND_OPEN_T5_NONCRITICAL_CONTROL": 5}
MIN_FREE_BYTES = 5 * 1024**3


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def aperture(obs: Any, env: Any) -> float | None:
    if isinstance(obs, Mapping):
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            if key in obs:
                value = aperture_metric(obs[key])
                if value is not None:
                    return value
    try:
        return aperture_metric(np.asarray(env.sim.data.qpos[-2:], dtype=np.float64).tolist())
    except Exception:
        return None


def source_anchor(job: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = root / str(job["receipt_path"])
    if not receipt.is_file() or sha(receipt) != str(job["receipt_sha256"]):
        raise RuntimeError(f"SOURCE_RECEIPT_SHA_MISMATCH:{job['branch_id']}")
    data = load(receipt)
    key = "critical" if job["anchor_class"] == "CRITICAL" else "noncritical"
    anchor = data.get("selected_anchors", {}).get(key)
    if not isinstance(anchor, dict) or anchor.get("status", "").startswith("SELECTED_") is False:
        raise RuntimeError(f"SOURCE_ANCHOR_MISSING:{job['branch_id']}")
    if int(anchor.get("step", -1)) != int(job["anchor_step"]) or str(anchor.get("state_sha256")) != str(job["anchor_state_sha256"]):
        raise RuntimeError(f"SOURCE_ANCHOR_HASH_MISMATCH:{job['branch_id']}")
    state = np.asarray(anchor.get("state"), dtype=np.float64)
    if state.ndim != 1 or hashlib.sha256(state.tobytes()).hexdigest() != str(job["anchor_state_sha256"]):
        raise RuntimeError(f"SOURCE_STATE_HASH_MISMATCH:{job['branch_id']}")
    rows = anchor.get("action_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"SOURCE_ACTION_ROWS_MISSING:{job['branch_id']}")
    return {"state": state, "state_sha256": str(anchor["state_sha256"]), "step": int(anchor["step"]), "rank_digest": anchor.get("rank_digest")}, rows


def prepare_anchor(config: dict[str, Any], job: Mapping[str, Any], anchor: Mapping[str, Any], counters: dict[str, int]):
    task_idx = int(str(job["canonical_parent_key"]).split("/task_")[1].split("/")[0])
    state_id = int(str(job["canonical_parent_key"]).rsplit("_", 1)[1])
    env, task_suite, task = z1.make_libero_env(config, str(job["suite"]), task_idx)
    env.reset()
    initial_states = task_suite.get_task_init_states(task_idx)
    obs = env.set_init_state(initial_states[state_id])
    dummy = [0.0] * 6 + [-1.0]
    for _ in range(int(config["environment"]["dummy_wait_steps"])):
        obs = env.step(dummy)[0]
        counters["env_step_calls"] += 1
    z1.restore_state(env, np.asarray(anchor["state"], dtype=np.float64))
    exact = np.array_equal(np.asarray(anchor["state"], dtype=np.float64), z1.snapshot_state(env))
    if not exact:
        env.close()
        raise RuntimeError(f"ANCHOR_STATE_RESTORE_NOT_EXACT:{job['branch_id']}")
    return env, task, obs


def bddl_binding(env: Any, task: Any) -> tuple[dict[str, Any], str]:
    from libero.libero import get_libero_path  # type: ignore

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    binding = bind_object_taxonomy(env, bddl)
    if binding.get("status") != "PASS" or not binding.get("target_object_ids"):
        raise RuntimeError("OBJECT_TAXONOMY_BINDING_INVALID")
    return binding, str(binding["target_object_ids"][0])


def action_pair(model: str, source: Mapping[str, Any], *, forced: bool, duration: int) -> tuple[list[float], list[float], float, dict[str, Any]]:
    raw = [float(value) for value in source["raw_action"]]
    final = [float(value) for value in source["env_action"]]
    if len(raw) != 7 or len(final) != 7 or not all(math.isfinite(value) for value in raw + final):
        raise RuntimeError("SEALED_ACTION_VECTOR_INVALID")
    if not forced:
        check = validate_action_pair(model, raw, final, raw_gripper=raw[-1], final_gripper=final[-1])
        if not check["accepted"]:
            raise RuntimeError(f"SEALED_ACTION_SEMANTICS_INVALID:{check['reason']}")
        return raw, final, 0.0, check
    opened_raw, opened_final = command_open_action(model, raw, final, duration=duration)
    check = validate_action_pair(model, opened_raw, opened_final, raw_gripper=opened_raw[-1], final_gripper=opened_final[-1])
    if not check["accepted"]:
        raise RuntimeError(f"OPEN_ACTION_SEMANTICS_INVALID:{check['reason']}")
    delta = arm_delta_linf(final, opened_final)
    if delta > 1e-7:
        raise RuntimeError("OPEN_ARM_ISOLATION_FAILED")
    return list(opened_raw), list(opened_final), delta, check


def run_branch(config: dict[str, Any], job: Mapping[str, Any], anchor: Mapping[str, Any], source_rows: list[dict[str, Any]], *, output: Path, counters: dict[str, int], control: Mapping[str, Any] | None = None, manual_video: Path | None = None) -> dict[str, Any]:
    duration = int(DOSE[job["arm"]])
    is_clean = job["arm"] == "CLEAN_BRANCH_CRITICAL"
    required = 20 if is_clean else duration + H_PHYS
    branch: dict[str, Any] = {"schema": "STAGE_Z_Z3_BRANCH_RECEIPT_V1", "status": "RUNNING", "branch_id": job["branch_id"], "model_family": job["model_family"], "suite": job["suite"], "canonical_parent_key": job["canonical_parent_key"], "arm": job["arm"], "duration": duration, "anchor_class": job["anchor_class"], "anchor_step": job["anchor_step"], "anchor_state_sha256": job["anchor_state_sha256"], "source_receipt_path": job["receipt_path"], "source_receipt_sha256": job["receipt_sha256"], "model_inference": False, "source_action_authority": "SEALED_Z2_CLEAN_ANCHOR_ACTION_ROWS", "state_restore_exact": False, "causal_input_binding_pass": False, "control_action_reference_exact": False, "available_horizon_steps": 0, "rows": [], "treatment_receipts": [], "treatment_compliant": False, "treatment_compliance": {"delivered_open_steps": 0}, "manual_audit_id": job.get("manual_audit_id"), "blinded_video_id": job.get("blinded_video_id"), "runtime_counters": {"env_step_calls": 0, "physical_telemetry_reads": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0, "attack_outcome_reads": 0}, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write(output, branch)
    frames: list[np.ndarray] = []
    env = None
    try:
        env, task, obs = prepare_anchor(config, job, anchor, counters)
        branch["state_restore_exact"] = True
        binding, target = bddl_binding(env, task)
        for relative in range(required):
            if relative >= len(source_rows):
                raise RuntimeError(f"SEALED_ACTION_HORIZON_SHORT:{relative}/{required}")
            source = source_rows[relative]
            if int(source.get("step", -1)) != int(anchor["step"]) + relative:
                raise RuntimeError(f"SEALED_ACTION_STEP_MISMATCH:{relative}")
            forced = not is_clean and relative < duration
            raw, env_action, arm_delta, semantics = action_pair(str(job["model_family"]), source, forced=forced, duration=duration)
            if not forced and (raw != [float(value) for value in source["raw_action"]] or env_action != [float(value) for value in source["env_action"]]):
                raise RuntimeError(f"CLEAN_ACTION_REPLAY_MISMATCH:{relative}")
            pre = telemetry_from_env(env, binding, target_object_id=target)
            pre_aperture = aperture(obs, env)
            row: dict[str, Any] = {"step": int(anchor["step"]) + relative, "relative_step": relative, "arm": job["arm"], "raw_policy_action": raw, "normalized_action": raw, "env_action": env_action, "reference_raw_action": [float(value) for value in source["raw_action"]], "reference_env_action": [float(value) for value in source["env_action"]], "reference_action_sha256": digest(source), "action_sha256": digest({"raw": raw, "env": env_action}), "arm_delta_linf": arm_delta, "gripper_delta_env": env_action[-1] - float(source["env_action"][-1]), "pre_aperture": pre_aperture, "post_contact_telemetry_valid": False}
            obs = env.step(env_action)[0]
            counters["env_step_calls"] += 1
            branch["runtime_counters"]["env_step_calls"] += 1
            post = telemetry_from_env(env, binding, target_object_id=target)
            branch["runtime_counters"]["physical_telemetry_reads"] += 1
            row.update({"post_contact_telemetry_valid": post.get("contact_telemetry_valid") is True, "post_object_position": post.get("object_position"), "post_object_gripper_contact": post.get("object_gripper_contact"), "post_object_support_contact": post.get("object_support_contact"), "post_object_eef_distance_m": post.get("object_eef_distance_m"), "post_aperture": aperture(obs, env), "post_telemetry_reason": post.get("telemetry_reason")})
            branch["rows"].append(row)
            if forced:
                receipt = {"requested_dose": duration, "relative_step": relative, "raw_policy_action": raw, "normalized_action": raw, "env_action": env_action, "reference_arm_action": [float(value) for value in source["env_action"][:6]], "actual_arm_action": env_action[:6], "arm_delta_linf": arm_delta, "pre_aperture": pre_aperture, "post_aperture": row["post_aperture"]}
                branch["treatment_receipts"].append(receipt)
            if manual_video is not None and isinstance(obs, Mapping) and obs.get("agentview_image") is not None:
                frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8).copy())
            branch["available_horizon_steps"] = len(branch["rows"])
            atomic_write(output, branch)
        branch["status"] = "PASS"
        branch["control_action_reference_exact"] = True
        branch["causal_input_binding_pass"] = True
        branch["treatment_compliant"] = is_clean or len(branch["treatment_receipts"]) == duration
        branch["treatment_compliance"] = {"delivered_open_steps": len(branch["treatment_receipts"]), "command_delivery_valid": branch["treatment_compliant"]}
        if is_clean:
            branch["physical_class"] = physical_class(branch, required)
        elif job["anchor_class"] == "CRITICAL":
            branch["physical_class"] = physical_class(branch, required, control)
            branch["v_phys_label"] = physical_label(control, branch, duration, str(job["model_family"])) if control is not None else "CONTROL_REFERENCE_MISSING_ABSTAIN"
            branch["runtime_counters"]["v_phys_reads"] = 1
        else:
            branch["physical_class"] = physical_class(branch, required)
            branch["v_phys_label"] = "NOT_APPLICABLE_NONCRITICAL_CONTROL"
        if manual_video is not None and frames:
            import imageio.v2 as imageio

            manual_video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(manual_video), frames, fps=10, codec="libx264")
            branch["manual_video_path"] = str(manual_video)
            branch["manual_video_sha256"] = sha(manual_video)
        atomic_write(output, branch)
        return branch
    except Exception as exc:
        branch.update({"status": "ENGINEERING_INVALID_OR_HORIZON_CENSORED", "error": f"{type(exc).__name__}:{exc}", "next_legal_action": "STOP_FOR_PI"})
        atomic_write(output, branch)
        raise
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODELS, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--m1-manifest", type=Path)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    protocol = load(args.protocol)
    manifest = load(args.manifest)
    config = load(args.config)
    if protocol.get("status") != "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN" or manifest.get("status") != "STAGE_Z_Z3_EXECUTION_MANIFEST_FROZEN_NOT_EXECUTED":
        raise RuntimeError("Z3_SOURCE_OR_EXECUTION_MANIFEST_NOT_FROZEN")
    if shutil.disk_usage("/mnt/sdc").free < MIN_FREE_BYTES:
        raise RuntimeError("Z3_STORAGE_FREE_MARGIN_BREACHED")
    z1.require_single_visible_gpu(args.gpu_id)
    gpu = z1.gpu_snapshot(args.gpu_id)
    jobs = [job for job in manifest["jobs"] if job["model_family"] == args.model_family and job["suite"] == args.suite]
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [job for job in jobs if job["branch_id"] in wanted]
    if not jobs:
        raise RuntimeError("Z3_WORKER_NO_JOBS")
    jobs.sort(key=lambda job: (job["canonical_parent_key"], ARMS.index(job["arm"])))
    counters = {"env_step_calls": 0, "physical_telemetry_reads": 0, "model_inference_calls": 0, "open_intervention_steps": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "pgd_calls": 0, "attack_outcome_reads": 0, "scientific_parent_exposure": len({job["canonical_parent_key"] for job in jobs})}
    control_by_parent: dict[str, dict[str, Any]] = {}
    for job in jobs:
        anchor, source_rows = source_anchor(job, Path.cwd())
        output = args.output_dir / f"{job['branch_id']}.json"
        control = control_by_parent.get(str(job["canonical_parent_key"])) if job["anchor_class"] == "CRITICAL" and job["arm"] != "CLEAN_BRANCH_CRITICAL" else None
        manual_video = args.output_dir / "manual_videos" / f"{job['blinded_video_id']}.mp4" if job.get("blinded_video_id") else None
        result = run_branch(config, job, anchor, source_rows, output=output, counters=counters, control=control, manual_video=manual_video)
        if job["arm"] == "CLEAN_BRANCH_CRITICAL":
            control_by_parent[str(job["canonical_parent_key"])] = result
        if job["arm"] != "CLEAN_BRANCH_CRITICAL":
            counters["open_intervention_steps"] += int(len(result.get("treatment_receipts", [])))
    summary = {"schema": "STAGE_Z_Z3_WORKER_RECEIPT_V1", "status": "PASS_Z3_WORKER_BATCH", "model_family": args.model_family, "suite": args.suite, "gpu": gpu, "jobs_completed": len(jobs), "runtime_counters": counters, "claim_boundary": "Z3 branch execution only; task success is unread and manual labels remain pending.", "next_legal_action": "CONTINUE_Z3_MATRIX_OR_STOP_ON_ANY_FAILURE"}
    atomic_write(args.output_dir / f"WORKER_{args.model_family}_{args.suite}.json", summary)
    print(json.dumps({"status": summary["status"], "jobs": len(jobs), "model_family": args.model_family, "suite": args.suite}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "ENGINEERING_INVALID_Z3_WORKER", "error": f"{type(exc).__name__}:{exc}", "next_legal_action": "STOP_FOR_PI"}, sort_keys=True), file=sys.stderr)
        raise
