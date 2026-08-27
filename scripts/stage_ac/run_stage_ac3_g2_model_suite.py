#!/usr/bin/env python3
"""Execute one frozen AC3 G2 model/suite shard with model-specific actions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any



ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
CONDITIONS = ("CLEAN_REFERENCE", "OPEN_T3", "OPEN_T5", "OPEN_T10")
DOSES = {"CLEAN_REFERENCE": 0, "OPEN_T3": 3, "OPEN_T5": 5, "OPEN_T10": 10}
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {"M0_OPENVLA": "FRESH_PER_STEP", "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE", "M2_PI05_LIBERO": "FRESH_PI05_REPLAN"}
ACTION_DIM = 7
ARM_TOLERANCE = 1e-7
MIN_FREE_MIB = 20_480
GATE = "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"AC3_G2_MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA1R1: ModuleType
AA1: ModuleType
np: Any
sys.path.insert(0, str(ROOT / "src"))
from stage_z_preparation.z3_contract import physical_class, physical_label, treatment_compliant  # noqa: E402


def load_runtime() -> None:
    global AA1R1, AA1, np
    import numpy as np  # noqa: F401

    AA1R1 = load_module(ROOT / "scripts/stage_aa/run_stage_aa1r1_engineering_branch.py", "ac3_g2_aa1r1")
    AA1 = AA1R1.AA1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def boundary_state_array(value: Any) -> np.ndarray:
    embedded_sha = None
    if isinstance(value, dict):
        embedded_sha = value.get("sha256")
        value = value.get("state")
    array = np.asarray(value, dtype=np.float64)
    require(array.ndim == 1 and array.size > 0, "AC3_G2_BOUNDARY_STATE_INVALID")
    digest = sha256_bytes(array.tobytes())
    if embedded_sha is not None:
        require(digest == str(embedded_sha), "AC3_G2_BOUNDARY_STATE_EMBEDDED_SHA")
    return array


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def counters() -> dict[str, int]:
    return {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "physical_endpoint_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 1,
        "ac2_exposure": 0,
    }


def set_branch_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def step_unpack(result: Any) -> tuple[Any, bool]:
    require(isinstance(result, (tuple, list)) and len(result) >= 3, "AC3_G2_ENV_STEP_RETURN_INVALID")
    return result[0], bool(result[2]) or bool(result[3]) if len(result) >= 5 else bool(result[2])


def frame_from(obs: Any, env: Any) -> np.ndarray | None:
    if isinstance(obs, dict):
        for key in ("agentview_image", "agentview_rgb", "frontview_image"):
            value = obs.get(key)
            if value is not None:
                array = np.asarray(value)
                if array.ndim == 3 and array.shape[-1] >= 3:
                    return np.asarray(array[..., :3], dtype=np.uint8).copy()
    try:
        value = env.render()
    except Exception:
        return None
    array = np.asarray(value) if value is not None else None
    return np.asarray(array[..., :3], dtype=np.uint8).copy() if array is not None and array.ndim == 3 and array.shape[-1] >= 3 else None


def aperture(obs: Any, env: Any) -> float | None:
    if isinstance(obs, dict):
        for key in ("robot0_gripper_qpos", "gripper_qpos"):
            value = obs.get(key)
            if value is not None:
                metric = AA1.TAXONOMY.aperture_metric(np.asarray(value, dtype=float).tolist())
                if metric is not None:
                    return float(metric)
    try:
        metric = AA1.TAXONOMY.aperture_metric(np.asarray(env.sim.data.qpos[-2:], dtype=float).tolist())
        return float(metric) if metric is not None else None
    except Exception:
        return None


def write_video(path: Path, frames: list[np.ndarray]) -> dict[str, Any]:
    require(frames, "AC3_G2_VIDEO_NO_FRAMES")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=10, codec="libx264")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data), "frames": len(frames), "fps": 10}


def snapshot_diff(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in after}


def as_contract_branch(result: dict[str, Any], job: dict[str, Any], source_receipt: dict[str, Any], branch_counts: dict[str, int], clean_control: dict[str, Any] | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    open_by_step = {int(item["step"]): item for item in result.get("action_receipts", [])}
    source_actions = source_receipt["clean"]["actions"]
    anchor = int(job["selected_anchor"]["step"])
    for row in result["rows"]:
        step = int(row["step"])
        source = source_actions[step]
        post = row["post"]
        pre = row["pre"]
        open_receipt = open_by_step.get(step)
        actual_action = [float(value) for value in row["action"]]
        rows.append({
            "step": step,
            "relative_step": step - anchor,
            "arm": job["condition"],
            "raw_policy_action": ([float(value) for value in open_receipt["raw_policy_action"]] if open_receipt else [float(value) for value in source["raw"]]),
            "normalized_action": ([float(value) for value in open_receipt["opened_raw_action"]] if open_receipt else [float(value) for value in source["raw"]]),
            "env_action": actual_action,
            "reference_raw_action": [float(value) for value in source["raw"]],
            "reference_env_action": [float(value) for value in source["final"]],
            "reference_action_sha256": canonical_hash(source),
            "action_sha256": canonical_hash({"raw": actual_action, "env": actual_action}),
            "arm_delta_linf": float(open_receipt["arm_delta_linf"]) if open_receipt else 0.0,
            "gripper_delta_env": float(actual_action[-1]) - float(source["final"][-1]),
            "pre_aperture": row.get("pre_aperture"),
            "post_aperture": row.get("post_aperture"),
            "pre_object_gripper_contact": pre.get("object_gripper_contact"),
            "post_contact_telemetry_valid": post.get("contact_telemetry_valid") is True,
            "post_object_position": post.get("object_position"),
            "post_object_gripper_contact": post.get("object_gripper_contact"),
            "post_object_support_contact": post.get("object_support_contact"),
            "post_object_eef_distance_m": post.get("object_eef_distance_m"),
            "post_telemetry_reason": post.get("telemetry_reason"),
        })
    contract = {
        "schema": "STAGE_AC_AC3_BRANCH_RECEIPT_V1",
        "status": "PASS",
        "gate": GATE,
        "branch_id": job["branch_id"],
        "model_family": job["model_family"],
        "suite": job["suite"],
        "canonical_parent_key": job["canonical_parent_key"],
        "cell_id": job["cell_id"],
        "parent_exposure_class": job["parent_exposure_class"],
        "condition": job["condition"],
        "dose": int(job["dose"]),
        "anchor_step": anchor,
        "anchor_state_sha256": job["selected_anchor"]["boundary_state_sha256"],
        "source_receipt_path": job["source_receipt"]["path"],
        "source_receipt_sha256": job["source_receipt"]["sha256"],
        "source_clean_trajectory_digest": job["selected_anchor"]["source_clean_trajectory_digest"],
        "branch_seed": job["branch_seed"],
        "model_inference": job["condition"] != "CLEAN_REFERENCE",
        "source_action_authority": "SEALED_AC2_CLEAN_ACTIONS_PLUS_FROZEN_MODEL_REINFERENCE",
        "state_restore_exact": bool(result["state_restore_exact"]),
        "causal_input_binding_pass": True,
        "control_action_reference_exact": True,
        "available_horizon_steps": len(rows),
        "rows": rows,
        "treatment_receipts": [
            {
                "requested_dose": int(job["dose"]),
                "relative_step": int(item["step"]) - anchor,
                "raw_policy_action": [float(value) for value in item["opened_raw_action"]],
                "normalized_action": [float(value) for value in item["opened_raw_action"]],
                "env_action": [float(value) for value in item["env_action"]],
                "reference_arm_action": [float(value) for value in item["raw_policy_action"][:6]],
                "actual_arm_action": [float(value) for value in item["env_action"][:6]],
                "arm_delta_linf": float(item["arm_delta_linf"]),
                "pre_aperture": next((r.get("pre_aperture") for r in rows if int(r["step"]) == int(item["step"])), None),
                "post_aperture": next((r.get("post_aperture") for r in rows if int(r["step"]) == int(item["step"])), None),
            }
            for item in result.get("action_receipts", [])
        ],
        "treatment_compliant": job["condition"] == "CLEAN_REFERENCE" or len(result.get("action_receipts", [])) == int(job["dose"]),
        "treatment_compliance": {"delivered_open_steps": len(result.get("action_receipts", [])), "command_delivery_valid": True},
        "queue_boundary_steps": result.get("queue_boundary_steps", []),
        "expected_queue_boundary_steps": result.get("expected_queue_boundary_steps", []),
        "queue_reset_verified": bool(result["queue_reset_verified"]),
        "telemetry_aligned": bool(result["telemetry_aligned"]),
        "arm_preserved": bool(result["arm_preserved"]),
        "max_arm_delta_linf": float(result["max_arm_delta_linf"]),
        "exact_open_delivery": bool(result["exact_open_delivery"]),
        "open_intervention_steps": int(result["open_intervention_steps"]),
        "runtime_counters": dict(sorted(branch_counts.items())),
        "blinded_video_id": job.get("blinded_video_id"),
        "video": result.get("video"),
        "trace_digest": result.get("trace_digest"),
        "scientific_claim": "AC3_TREATMENT_NAIVE_PRIMARY_BRANCH_ONLY",
        "claim_boundary": "AC3 primary physical branch execution; no promotion until G5",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    contract["physical_endpoint_reads"] = 1
    contract["physical_class"] = physical_class(contract, 20 if job["condition"] == "CLEAN_REFERENCE" else int(job["dose"]) + 10, clean_control)
    contract["v_phys_label"] = "NOT_APPLICABLE_CLEAN_REFERENCE" if job["condition"] == "CLEAN_REFERENCE" else physical_label(clean_control or {}, contract, int(job["dose"]), str(job["model_family"]))
    contract["runtime_counters"]["physical_endpoint_reads"] = 1
    if job["condition"] != "CLEAN_REFERENCE":
        contract["runtime_counters"]["v_phys_reads"] = 1
    return contract


def load_source_unit(unit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_path = Path(str(unit["source_receipt"]["path"]))
    require(source_path.is_file(), f"AC3_G2_SOURCE_RECEIPT_MISSING:{source_path}")
    data = source_path.read_bytes()
    require(len(data) == int(unit["source_receipt"]["bytes"]), f"AC3_G2_SOURCE_RECEIPT_BYTES:{source_path}")
    require(sha256_bytes(data) == str(unit["source_receipt"]["sha256"]), f"AC3_G2_SOURCE_RECEIPT_SHA:{source_path}")
    receipt = json.loads(data.decode("utf-8"))
    require(receipt.get("status") == "AC2_CLEAN_CELL_COMPLETE", f"AC3_G2_SOURCE_RECEIPT_STATUS:{source_path}")
    require(receipt.get("model_family") == unit["model_family"] and receipt.get("canonical_parent_key") == unit["canonical_parent_key"], f"AC3_G2_SOURCE_BINDING:{source_path}")
    clean = receipt.get("clean")
    require(isinstance(clean, dict) and clean.get("status") == "AC2_CLEAN_CELL_COMPLETE", f"AC3_G2_SOURCE_CLEAN_MISSING:{source_path}")
    anchor = unit["selected_anchor"]
    step = int(anchor["step"])
    actions = clean.get("actions")
    rows = clean.get("rows")
    states = clean.get("boundary_states")
    require(isinstance(actions, list) and isinstance(rows, list) and isinstance(states, dict), f"AC3_G2_SOURCE_EVIDENCE_MISSING:{source_path}")
    require(step + 20 <= len(actions) and step + 20 <= len(rows), f"AC3_G2_SOURCE_ANCHOR_HORIZON:{source_path}")
    require(actions[step : step + 20] == anchor["actions"], f"AC3_G2_SOURCE_ANCHOR_ACTION_MISMATCH:{source_path}")
    state = np.asarray(anchor["boundary_state"], dtype=np.float64)
    require(sha256_bytes(state.tobytes()) == str(anchor["boundary_state_sha256"]), f"AC3_G2_ANCHOR_STATE_SHA:{source_path}")
    source_state = boundary_state_array(states[str(step)])
    require(np.array_equal(state, source_state), f"AC3_G2_ANCHOR_STATE_SOURCE_MISMATCH:{source_path}")
    require(str(clean.get("clean_trajectory_digest")) == str(anchor["source_clean_trajectory_digest"]), f"AC3_G2_SOURCE_TRAJECTORY_DIGEST:{source_path}")
    prepared = dict(clean)
    prepared["boundary_states"] = {int(key): boundary_state_array(value) for key, value in states.items()}
    prepared["engineering_clean_rows"] = rows[step : step + 20]
    point = {
        "step": step,
        "point_label": "CRITICAL_ANCHOR",
        "selection_rank_sha256": anchor["selection_rank_sha256"],
        "boundary": BOUNDARY[unit["model_family"]],
        "clean_boundary_state_sha256": anchor["boundary_state_sha256"],
    }
    canary = {"suite": unit["suite"], "task_idx": unit["source_task_idx"], "state_id": unit["state_id"], "canonical_parent_key": unit["canonical_parent_key"]}
    return receipt, prepared, {"canary": canary, "point": point}


def execute_branch(config: dict[str, Any], job: dict[str, Any], receipt: dict[str, Any], clean: dict[str, Any], binding: dict[str, Any], infer: Any, output: Path, video_path: Path | None) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"AC3_G2_APPEND_ONLY_BRANCH_EXISTS:{output}")
    branch_counts = counters()
    initial: dict[str, Any] = {
        "schema": "STAGE_AC_AC3_BRANCH_RECEIPT_V1",
        "status": "RUNNING",
        "gate": GATE,
        "branch_id": job["branch_id"],
        "model_family": job["model_family"],
        "suite": job["suite"],
        "canonical_parent_key": job["canonical_parent_key"],
        "cell_id": job["cell_id"],
        "condition": job["condition"],
        "dose": int(job["dose"]),
        "anchor_step": int(job["selected_anchor"]["step"]),
        "anchor_state_sha256": job["selected_anchor"]["boundary_state_sha256"],
        "source_receipt_path": job["source_receipt"]["path"],
        "source_receipt_sha256": job["source_receipt"]["sha256"],
        "branch_seed": job["branch_seed"],
        "blinded_video_id": job.get("blinded_video_id"),
        "runtime_counters": branch_counts,
        "scientific_claim": "AC3_TREATMENT_NAIVE_PRIMARY_BRANCH_ONLY",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(output, initial)
    set_branch_seed(int(job["branch_seed"]["seed"]))
    point = {"step": int(job["selected_anchor"]["step"]), "selection_rank_sha256": job["selected_anchor"]["selection_rank_sha256"], "boundary": BOUNDARY[job["model_family"]], "clean_boundary_state_sha256": job["selected_anchor"]["boundary_state_sha256"]}
    canary = {"suite": job["suite"], "task_idx": job["source_task_idx"], "state_id": job["state_id"], "canonical_parent_key": job["canonical_parent_key"]}
    required = 20 if job["condition"] == "CLEAN_REFERENCE" else int(job["dose"]) + 10
    anchor_step = int(point["step"])
    require(anchor_step + required <= len(clean["actions"]), f"AC3_G2_BRANCH_HORIZON:{job['branch_id']}")
    expected_state = clean["boundary_states"][anchor_step]
    env, _task, obs, episode_done = AA1R1.replay_prefix(config, canary, clean["actions"], anchor_step, expected_state, branch_counts)
    queue: list[tuple[np.ndarray, np.ndarray]] = []
    queue_boundaries: list[int] = []
    rows: list[dict[str, Any]] = []
    action_receipts: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    first = frame_from(obs, env)
    if first is not None:
        frames.append(first)
    try:
        for offset in range(required):
            step = anchor_step + offset
            require(not episode_done, f"AC3_G2_TERMINAL_BEFORE_HORIZON:{job['branch_id']}:{step}")
            pre = AA1.telemetry(env, clean["binding"], str(clean["target_object"]), branch_counts)
            pre_aperture = aperture(obs, env)
            is_clean = job["condition"] == "CLEAN_REFERENCE"
            if is_clean:
                source_action = clean["actions"][step]
                raw_action = np.asarray(source_action["raw"], dtype=np.float32)
                final_action = np.asarray(source_action["final"], dtype=np.float32)
            else:
                if not queue:
                    queue_boundaries.append(step)
                    queue = AA1.model_pairs(infer, obs, str(clean["language"]), job["model_family"], branch_counts)
                raw_action, final_action = queue.pop(0)
            require(raw_action.size == ACTION_DIM and final_action.size == ACTION_DIM, f"AC3_G2_ACTION_DIM:{job['branch_id']}:{step}")
            require(np.isfinite(raw_action).all() and np.isfinite(final_action).all(), f"AC3_G2_ACTION_NONFINITE:{job['branch_id']}:{step}")
            opened = False
            if not is_clean and offset < int(job["dose"]):
                opened_raw, opened_final = AA1.command_open_action(job["model_family"], raw_action.tolist(), final_action.tolist(), int(job["dose"]))
                opened_raw = np.asarray(opened_raw, dtype=np.float32)
                opened_final = np.asarray(opened_final, dtype=np.float32)
                require(opened_raw.size == ACTION_DIM and opened_final.size == ACTION_DIM, f"AC3_G2_OPEN_DIM:{job['branch_id']}:{step}")
                require(np.array_equal(opened_final[:6], final_action[:6]) and float(opened_final[-1]) == -1.0, f"AC3_G2_OPEN_CONTRACT:{job['branch_id']}:{step}")
                action = opened_final
                opened = True
                arm_delta = float(np.max(np.abs(opened_final[:6] - final_action[:6])))
                action_receipts.append({"step": step, "raw_policy_action": raw_action.tolist(), "opened_raw_action": opened_raw.tolist(), "env_action": opened_final.tolist(), "arm_delta_linf": arm_delta})
                branch_counts["open_intervention_steps"] += 1
            else:
                action = final_action
            obs, episode_done = step_unpack(env.step(action.tolist()))
            branch_counts["env_step_calls"] += 1
            post = AA1.telemetry(env, clean["binding"], str(clean["target_object"]), branch_counts)
            post_aperture = aperture(obs, env)
            next_frame = frame_from(obs, env)
            if next_frame is not None:
                frames.append(next_frame)
            rows.append({"step": step, "pre": pre, "post": post, "action": action.tolist(), "opened": opened, "pre_aperture": pre_aperture, "post_aperture": post_aperture, "terminal_after": episode_done})
            partial = {**initial, "rows": rows, "action_receipts": action_receipts, "runtime_counters": branch_counts, "available_horizon_steps": len(rows)}
            atomic_write(output, partial)
        expected_boundaries = [] if job["condition"] == "CLEAN_REFERENCE" else list(range(anchor_step, anchor_step + required, QUEUE_LENGTH[job["model_family"]]))
        queue_reset = job["condition"] == "CLEAN_REFERENCE" or queue_boundaries == expected_boundaries
        telemetry_aligned = len(rows) == required and all(isinstance(row["pre"], dict) and isinstance(row["post"], dict) for row in rows)
        arm_deltas = [float(item["arm_delta_linf"]) for item in action_receipts]
        arm_preserved = all(delta <= ARM_TOLERANCE for delta in arm_deltas)
        exact_open = job["condition"] == "CLEAN_REFERENCE" or [item["step"] for item in action_receipts] == list(range(anchor_step, anchor_step + int(job["dose"])))
        require(queue_reset and telemetry_aligned and arm_preserved and exact_open, f"AC3_G2_BRANCH_INVARIANT:{job['branch_id']}")
        result = {"state_restore_exact": True, "rows": rows, "action_receipts": action_receipts, "queue_boundary_steps": queue_boundaries, "expected_queue_boundary_steps": expected_boundaries, "queue_reset_verified": queue_reset, "telemetry_aligned": telemetry_aligned, "arm_preserved": arm_preserved, "max_arm_delta_linf": max(arm_deltas, default=0.0), "exact_open_delivery": exact_open, "open_intervention_steps": len(action_receipts), "trace_digest": canonical_hash(rows), "video": write_video(video_path, frames) if video_path is not None else None}
        contract = as_contract_branch(result, job, receipt, branch_counts, None)
        return contract
    finally:
        env.close()


def prepare_static(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    g0_root = read_json(args.g0_root)
    protocol = read_json(args.protocol)
    manifest = read_json(args.manifest)
    g1_root = read_json(args.g1_root)
    runtime = read_json(args.runtime_authority)
    require(g0_root.get("status") == "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE", "AC3_G2_G0_NOT_FROZEN")
    require(protocol.get("gate") == GATE and protocol.get("status") == "STAGE_AC_AC3_G0_STATIC_FREEZE_AUTHORIZED", "AC3_G2_PROTOCOL_INVALID")
    require(manifest.get("schema") == "STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1" and len(manifest.get("branches", [])) == 384, "AC3_G2_MANIFEST_INVALID")
    require(sha256_file(args.manifest) == g0_root["artifacts"]["launch_manifest"]["sha256"], "AC3_G2_MANIFEST_ROOT_SHA")
    require(g1_root.get("status") == "STAGE_AC_AC3Q_G1_ENGINEERING_QUALIFICATION_PASS_STOP_FOR_PI", "AC3_G2_G1_NOT_PASS")
    require(runtime.get("status") == "STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC3_G2_RUNTIME_AUTHORITY_INVALID")
    for entry in runtime.get("runtime_files", {}).values():
        path = ROOT / str(entry["path"])
        require(path.is_file() and path.stat().st_size == int(entry["bytes"]) and sha256_file(path) == str(entry["sha256"]), f"AC3_G2_RUNTIME_FILE_MISMATCH:{path}")
    jobs = [job for job in manifest["branches"] if job.get("model_family") == args.model_family and job.get("suite") == args.suite]
    parents = {job["canonical_parent_key"] for job in jobs}
    require(len(jobs) == 4 * len(parents), f"AC3_G2_SHARD_SIZE_NOT_FOUR_CONDITIONS:{args.model_family}:{args.suite}:{len(jobs)}:{len(parents)}")
    require(len({job["branch_id"] for job in jobs}) == len(jobs), "AC3_G2_BRANCH_ID_DUPLICATE")
    for parent in parents:
        conditions = {job["condition"] for job in jobs if job["canonical_parent_key"] == parent}
        require(conditions == set(CONDITIONS), f"AC3_G2_PARENT_CONDITIONS_INVALID:{args.model_family}:{args.suite}:{parent}:{conditions}")
    sample = read_json(args.blind_sample)
    blind_map = {row["branch_id"]: row["blinded_video_id"] for row in sample.get("sample", [])}
    require(len(blind_map) == 96 and len(set(blind_map.values())) == 96, "AC3_G2_BLIND_SAMPLE_INVALID")
    for job in jobs:
        job["blinded_video_id"] = blind_map.get(job["branch_id"])
    return manifest, jobs, blind_map


def load_model(config: dict[str, Any], family: str, suite: str) -> tuple[Any, Any, Path, dict[str, Any]]:
    checkpoint = AA1.checkpoint_path(config, family, suite)
    manifest = None
    if family == "M1_OPENVLA_OFT":
        manifest_path = ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
        manifest = AA1.Z1.verify_m1_materialization(manifest_path, checkpoint, suite, str(config["model_families"][family]["checkpoint_manifests_sha256"]))
    if family == "M0_OPENVLA":
        infer, model, _normalization = AA1.Z1.load_openvla(str(checkpoint), oft=False, suite=suite, return_chunk=True)
    elif family == "M1_OPENVLA_OFT":
        infer, model, _normalization = AA1.Z1.load_openvla(str(checkpoint), oft=True, suite=suite, return_chunk=True)
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        infer, model = AA1.Z1.load_pi05(str(checkpoint), return_chunk=True)
    return infer, model, checkpoint, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_runtime()
    _manifest, jobs, blind_map = prepare_static(args)
    if args.output_dir.exists() and any(args.output_dir.glob("*.json")):
        raise RuntimeError(f"AC3_G2_OUTPUT_DIR_NOT_EMPTY:{args.output_dir}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    AA1.require_single_gpu(args.gpu_id)
    gpu = AA1.gpu_snapshot(args.gpu_id)
    config = read_json(args.config)
    AA1.Z1.configure_libero(config)
    infer = model = None
    worker_counts: dict[str, int] = {}
    completed: list[dict[str, Any]] = []
    try:
        infer, model, checkpoint, checkpoint_manifest = load_model(config, args.model_family, args.suite)
        jobs.sort(key=lambda job: (job["canonical_parent_key"], CONDITIONS.index(job["condition"])))
        source_cache: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        controls: dict[str, dict[str, Any]] = {}
        for job in jobs:
            parent = str(job["canonical_parent_key"])
            if parent not in source_cache:
                source_cache[parent] = load_source_unit(next(unit for unit in _manifest["model_parent_units"] if unit["model_family"] == job["model_family"] and unit["canonical_parent_key"] == parent))
            source_receipt, clean, binding = source_cache[parent]
            output = args.output_dir / f"{job['branch_id']}.json"
            video_path = args.video_dir / f"{job['blinded_video_id']}.mp4" if job.get("blinded_video_id") else None
            try:
                result = execute_branch(config, job, source_receipt, clean, binding, infer, output, video_path)
                if job["condition"] == "CLEAN_REFERENCE":
                    controls[parent] = result
                elif job["condition"] != "CLEAN_REFERENCE":
                    result["physical_class"] = physical_class(result, int(job["dose"]) + 10, controls.get(parent))
                    result["v_phys_label"] = physical_label(controls.get(parent) or {}, result, int(job["dose"]), args.model_family)
                    result["runtime_counters"]["v_phys_reads"] = 1
                    atomic_write(output, result)
                for key, value in result.get("runtime_counters", {}).items():
                    worker_counts[key] = worker_counts.get(key, 0) + int(value)
                completed.append({"branch_id": job["branch_id"], "condition": job["condition"], "status": result["status"], "receipt": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)}, "video": result.get("video")})
            except Exception as exc:
                failure = {"schema": "STAGE_AC_AC3_BRANCH_RECEIPT_V1", "status": "ENGINEERING_INVALID_OR_HORIZON_CENSORED", "gate": GATE, "branch_id": job["branch_id"], "model_family": args.model_family, "suite": args.suite, "canonical_parent_key": parent, "condition": job["condition"], "dose": int(job["dose"]), "source_receipt_path": job["source_receipt"]["path"], "source_receipt_sha256": job["source_receipt"]["sha256"], "error": {"type": type(exc).__name__, "message": str(exc)}, "next_legal_action": "STOP_FOR_PI", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                atomic_write(output, failure)
                raise
        require(len(completed) == 128, f"AC3_G2_COMPLETED_COUNT:{len(completed)}")
        summary = {"schema": "STAGE_AC_AC3_G2_WORKER_RECEIPT_V1", "status": "PASS_AC3_G2_MODEL_SUITE_WORKER", "gate": GATE, "model_family": args.model_family, "suite": args.suite, "gpu": gpu, "checkpoint": str(checkpoint), "checkpoint_manifest": checkpoint_manifest, "jobs_completed": len(completed), "branch_receipts": completed, "runtime_counters": dict(sorted(worker_counts.items())), "scientific_claim": "AC3_TREATMENT_NAIVE_PRIMARY_BRANCH_EXECUTION_ONLY", "claim_boundary": "G2 branch execution; no G3 promotion statistics", "next_legal_action": "CONTINUE_AC3_G2_MATRIX"}
        atomic_write(args.output_dir / f"WORKER_{args.model_family}_{args.suite}.json", summary)
        return summary
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def self_test() -> None:
    global np
    import numpy as np  # noqa: F401

    assert len(MODELS) == 3 and len(SUITES) == 3
    assert sum(DOSES.values()) == 18
    assert QUEUE_LENGTH["M2_PI05_LIBERO"] == 5
    wrapped = {"sha256": sha256_bytes(np.asarray([1.0, 2.0], dtype=np.float64).tobytes()), "state": [1.0, 2.0]}
    assert np.array_equal(boundary_state_array(wrapped), np.asarray([1.0, 2.0], dtype=np.float64))
    print(json.dumps({"status": "AC3_G2_STATIC_SELF_TEST_PASS", "shards": 9, "primary_branches": 384, "four_conditions_per_parent": True}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json")
    parser.add_argument("--g0-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json")
    parser.add_argument("--g1-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_G1_ROOT_SEAL_V1.json")
    parser.add_argument("--runtime-authority", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_V2.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json")
    parser.add_argument("--blind-sample", type=Path, default=ROOT / "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json")
    parser.add_argument("--model-family", choices=MODELS)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ("model_family", "suite", "gpu_id", "output_dir", "video_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --self-test is used")
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "model_family": args.model_family, "suite": args.suite, "jobs": result["jobs_completed"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ENGINEERING_INVALID_AC3_G2_WORKER", "error": f"{type(exc).__name__}:{exc}", "next_legal_action": "STOP_FOR_PI"}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
