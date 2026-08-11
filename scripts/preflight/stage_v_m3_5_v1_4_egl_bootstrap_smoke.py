#!/usr/bin/env python3
"""Non-rollout V1.4 EGL/PIL bootstrap smoke; no policy or model load."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _uuid(value: Any) -> str:
    return str(value or "").strip().lower().removeprefix("gpu-")


def _inventory() -> list[dict[str, Any]]:
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in result.splitlines():
        index, uuid, free_mib, total_mib = (part.strip() for part in line.split(",", 3))
        rows.append({"index": int(index), "uuid": _uuid(uuid), "free_memory_mib": int(free_mib), "total_memory_mib": int(total_mib)})
    return rows


def _compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GPU_PROCESS_QUERY_FAILED:{result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            pid, name, uuid, used_mib = (part.strip() for part in line.split(",", 3))
            rows.append({"pid": int(pid), "process_name": name, "gpu_uuid": _uuid(uuid), "used_memory_mib": int(used_mib)})
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--protected-pid", type=int, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_official_v3_20260716"))
    parser.add_argument("--upstream-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4"))
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    authorization_path = args.authorization.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{output}")
    receipt: dict[str, Any] = {
        "schema": "STAGE_V_M3_5_V1_4_EGL_BOOTSTRAP_SMOKE_V1",
        "version": "V1.4",
        "status": "FAIL",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_script": str(Path(__file__).resolve()),
        "smoke_script_sha256": _sha(Path(__file__).resolve()),
        "runtime_python": sys.executable,
        "repo_root": str(repo),
        "protocol": str(protocol_path),
        "authorization": str(authorization_path),
        "physical_gpu_index": args.gpu,
        "protected_pid": args.protected_pid,
        "protected_counters": dict(COUNTERS),
        "outcome_data_observed": False,
        "simulator_steps": 0,
        "smoke_reset_calls": 0,
        "smoke_step_calls": 0,
        "policy_loaded": False,
        "model_loaded": False,
        "rollout_started": False,
        "eval160_read": False,
    }
    env = None
    failure = None
    try:
        _require(sys.executable == "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python", "RUNTIME_PYTHON_MISMATCH")
        protocol = _load(protocol_path)
        authorization = _load(authorization_path)
        static_path = Path(str(authorization.get("static_audit_report", ""))).resolve()
        static_audit = _load(static_path)
        actual_commit = _git(repo, "rev-parse", "HEAD")
        actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
        _require(_git(repo, "status", "--porcelain") == "", "SOURCE_WORKTREE_NOT_CLEAN")
        _require(protocol.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A" and protocol.get("status") == "FROZEN_RUNTIME_AUTHORIZED", "PROTOCOL_NOT_FROZEN")
        _require(authorization.get("status") == "PASS" and authorization.get("protocol_sha256") == _sha(protocol_path), "AUTHORIZATION_NOT_BOUND")
        _require(authorization.get("source_commit") == actual_commit and authorization.get("source_tree") == actual_tree, "AUTHORIZATION_SOURCE_MISMATCH")
        _require(static_audit.get("status") == "PASS_STATIC_DESIGN_ONLY" and authorization.get("static_audit_sha256") == _sha(static_path), "STATIC_AUDIT_NOT_BOUND")
        _require(protocol.get("protected_counters") == COUNTERS and authorization.get("protected_counters") == COUNTERS, "PROTECTED_COUNTERS_NONZERO")
        binding = protocol["source_binding"]
        _require(actual_commit == binding["runtime_commit"] and actual_tree == binding["runtime_tree"], "SOURCE_BINDING_MISMATCH")
        resource = protocol["resource_contract"]
        admitted = [int(value) for value in resource["admitted_gpu_indices"]]
        excluded = [int(value) for value in resource["excluded_gpu_indices"]]
        _require(args.gpu in admitted and args.gpu not in excluded and args.gpu != 3, "GPU_NOT_ADMITTED")
        inventory = _inventory()
        row = next((item for item in inventory if item["index"] == args.gpu), None)
        _require(row is not None, "GPU_INVENTORY_MISSING")
        expected_uuid = _uuid(resource["gpu_uuid_by_index"][str(args.gpu)])
        _require(row["uuid"] == expected_uuid and row["free_memory_mib"] >= int(resource["minimum_free_memory_mib"]), "GPU_IDENTITY_OR_MEMORY_INVALID")
        processes = _compute_processes()
        _require(not any(item["gpu_uuid"] == expected_uuid for item in processes), "GPU_ALREADY_HAS_COMPUTE_PROCESS")
        for name, file_binding in resource["runtime_egl_files"].items():
            _require(_sha(Path(str(file_binding["path"])).resolve()) == file_binding["sha256"], f"RUNTIME_EGL_FILE_SHA_MISMATCH:{name}")
        receipt.update({"source_commit": actual_commit, "source_tree": actual_tree, "source_status": "", "protocol_sha256": _sha(protocol_path), "authorization_sha256": _sha(authorization_path), "static_audit_report": str(static_path), "static_audit_sha256": _sha(static_path), "gpu_inventory_before": inventory, "gpu_compute_processes_before": processes, "gpu_uuid": expected_uuid, "gpu_free_memory_mib_before": row["free_memory_mib"], "minimum_free_memory_mib": int(resource["minimum_free_memory_mib"]), "protected_pid_alive_before": _alive(args.protected_pid)})
        os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(args.gpu)})
        sys.path[:0] = [str(repo), str(repo / "src")]
        from PIL import Image
        from gripper_attack.stage_v_canonical_execution_core import canonical_value

        descriptor = canonical_value(Image.frombytes("RGB", (2, 1), bytes((0, 1, 2, 3, 4, 5))))
        _require(descriptor == {"kind": "image", "mode": "RGB", "size": [2, 1], "raw_sha256": "17e88db187afd62c16e5debf3e6527cd006bc012bc90b51a810cd80c2d511f43"}, "PIL_TRACE_DESCRIPTOR_MISMATCH")
        receipt["pil_trace_descriptor"] = descriptor
        receipt["pil_trace_descriptor_status"] = "PASS"
        from scripts.detector_v5.run_stage_v_canonical_clean import _load_external_modules
        from scripts.detector_v5.stage_v_gpu_resource_contract import resolve_cuda_physical_uuid
        _, _, _, _, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
        get_libero_path, OffScreenRenderEnv = libero_runtime
        task = benchmark.get_benchmark_dict()["libero_10"]().get_task(0)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        _require(bddl.is_file(), "BDDL_NOT_FOUND")
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, camera_names=["agentview"], has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, control_freq=20, render_gpu_device_id=args.gpu, horizon=1)
        render_id = getattr(env, "render_gpu_device_id", None)
        if render_id is None:
            render_id = getattr(getattr(env, "env", None), "render_gpu_device_id", None)
        _require(int(render_id) == args.gpu, "ENV_RENDER_GPU_PHYSICAL_BINDING_MISMATCH")
        import torch

        _require(torch.cuda.is_available() and torch.cuda.current_device() == 0, "TORCH_LOGICAL_DEVICE_INVALID")
        properties = torch.cuda.get_device_properties(0)
        torch_uuid, uuid_source = resolve_cuda_physical_uuid(args.gpu, torch_device_uuid=getattr(properties, "uuid", None), inventory=[{"gpu_id": item["index"], "gpu_uuid": item["uuid"], "memory_free_mib": item["free_memory_mib"]} for item in inventory])
        _require(torch_uuid == expected_uuid, "TORCH_PHYSICAL_GPU_UUID_MISMATCH")
        from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import _model_binding_receipt

        binding_path = output.parent / "M35_RUNTIME_BINDING_RECEIPT.json"
        _require(not binding_path.exists(), "RUNTIME_BINDING_RECEIPT_ALREADY_EXISTS")
        binding_args = argparse.Namespace(gpu=args.gpu, source_commit=actual_commit, source_tree=actual_tree, parent_key="libero_10/task_00/state_00", runtime_input_binding={"runtime_inputs": {"gpu": {"physical_gpu_index": args.gpu, "gpu_uuid": expected_uuid}}})
        _model_binding_receipt(binding_args, env, output.parent)
        binding_receipt = _load(binding_path)
        _require(binding_receipt.get("status") == "PASS" and binding_receipt.get("episode_started") is False, "RUNTIME_BINDING_RECEIPT_INVALID")
        sim = getattr(env, "sim", None) or getattr(getattr(env, "env", None), "sim", None)
        sim_time = float(getattr(getattr(sim, "data", None), "time", 0.0))
        _require(sim is not None and sim_time == 0.0, "SIM_TIME_ADVANCED_DURING_BOOTSTRAP")
        contexts = [value for value in (getattr(sim, "render_context", None), getattr(sim, "_render_context_offscreen", None)) if value is not None]
        receipt.update({"runtime_binding_receipt": str(binding_path), "runtime_binding_receipt_sha256": _sha(binding_path), "runtime_binding_status": binding_receipt.get("status"), "runtime_binding_checked_without_model_or_policy": True, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "mujoco_gl": os.environ["MUJOCO_GL"], "mujoco_egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"], "env_render_gpu_device_id": int(render_id), "torch_current_device": int(torch.cuda.current_device()), "torch_device_uuid": torch_uuid, "torch_device_uuid_source": uuid_source, "torch_device_name": torch.cuda.get_device_name(0), "offscreen_context_types": [f"{type(value).__module__}.{type(value).__name__}" for value in contexts], "sim_time_before_close": sim_time, "bddl_file": str(bddl), "bddl_sha256": _sha(bddl), "selected_parent_identity": "libero_10/task_00/state_00", "status": "PASS"})
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if env is not None:
            try:
                env.close()
                receipt["environment_closed"] = True
            except BaseException as exc:
                receipt["environment_closed"] = False
                failure = failure or {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        else:
            receipt["environment_closed"] = False
    receipt["protected_pid_alive_after"] = _alive(args.protected_pid)
    receipt["source_status_after"] = _git(repo, "status", "--porcelain")
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if failure is not None:
        receipt["failure"] = failure
        receipt["status"] = "FAIL"
    if receipt["source_status_after"] or receipt["protected_counters"] != COUNTERS or not receipt["protected_pid_alive_after"]:
        receipt["status"] = "FAIL"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(f"{_sha(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "gpu": args.gpu, "simulator_steps": receipt["simulator_steps"], "reset_calls": receipt["smoke_reset_calls"], "step_calls": receipt["smoke_step_calls"], "uuid_source": receipt.get("torch_device_uuid_source")}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
