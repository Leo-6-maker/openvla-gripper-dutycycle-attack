#!/usr/bin/env python3
"""Fail-closed eight-GPU Stage V M1-V2 supervisor."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PREFIX = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800"
IDENTITY = "libero_10/task_08/state_47"
GPU_IDS = tuple(range(8))
PHASES = {
    "Q1": ("CLEAN_QUALIFICATION", "rep_01"),
    "C1": ("COUNTERFACTUAL_CLEAN_PREFIX", "rep_01"),
    "Q2": ("CLEAN_QUALIFICATION", "rep_02"),
    "C2": ("COUNTERFACTUAL_CLEAN_PREFIX", "rep_02"),
}
LABELS = tuple(PHASES)
BOUNDARIES = (
    "eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts",
    "attack_rollouts", "intervention_applied_steps", "counterfactual_open_steps",
)


class V2Error(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V2Error(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _git_blob_sha(relative_path: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative_path}"], check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def _number(value: str) -> int | float | None:
    token = value.strip().split()[0] if value.strip() else ""
    try:
        number = float(token)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _query_nvidia_smi(*args: str) -> str:
    completed = subprocess.run(["nvidia-smi", *args], check=False, capture_output=True, text=True, timeout=20)
    if completed.returncode != 0:
        raise V2Error(f"NVIDIA_SMI_FAILED:{completed.returncode}:{completed.stderr[-300:]}")
    return completed.stdout


def _pid_detail(pid: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ps", "-o", "user=,pid=,ppid=,etime=,args=", "-p", str(pid)],
            check=False, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {"pid": pid, "owner": None, "command": None}
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    fields = line.split(None, 4)
    return {
        "pid": pid,
        "owner": fields[0] if fields else None,
        "ppid": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None,
        "elapsed": fields[3] if len(fields) > 3 else None,
        "command": fields[4] if len(fields) > 4 else None,
    }


def _compute_processes(text: str, uuid_to_gpu: Mapping[str, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 4 or not fields[1].strip().isdigit():
            continue
        pid = int(fields[1].strip())
        result.append({
            **_pid_detail(pid),
            "gpu_id": uuid_to_gpu.get(fields[0].strip()),
            "gpu_uuid": fields[0].strip(),
            "process_name": fields[2].strip(),
            "used_memory_mib": _number(fields[3]),
            "kind": "COMPUTE",
        })
    return result


def _pmon_processes(text: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        result.append({
            **_pid_detail(int(fields[1])),
            "gpu_id": int(fields[0]),
            "kind": fields[2].upper(),
            "process_name": fields[-1],
        })
    return result


def validate_binding_receipt(receipt: Mapping[str, Any], gpu: int) -> None:
    expected = {
        "logical_worker_id": f"worker_{gpu}",
        "requested_physical_gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "torch_current_device": 0,
        "mujoco_gl": "egl",
        "egl_device_identifier": gpu,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise V2Error(f"GPU_BINDING_RECEIPT_INVALID:{key}:gpu_{gpu:02d}")
    if not str(receipt.get("gpu_uuid", "")).strip():
        raise V2Error(f"GPU_BINDING_RECEIPT_GPU_UUID_MISSING:gpu_{gpu:02d}")
    if not isinstance(receipt.get("renderer_device_information"), Mapping):
        raise V2Error(f"GPU_BINDING_RECEIPT_RENDERER_INFO_MISSING:gpu_{gpu:02d}")


def gpu_preflight(*, gpu_ids: tuple[int, ...] = GPU_IDS, idle_memory_max_mib: int = 1024) -> dict[str, Any]:
    """Take one conservative all-GPU snapshot; any telemetry gap is unsafe."""
    gpu_text = _query_nvidia_smi(
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    gpu_rows: list[dict[str, Any]] = []
    for fields in csv.reader(gpu_text.splitlines()):
        if len(fields) < 7 or not fields[0].strip().isdigit():
            continue
        gpu_rows.append({
            "index": int(fields[0]), "uuid": fields[1].strip(), "name": fields[2].strip(),
            "memory_total_mib": _number(fields[3]), "memory_used_mib": _number(fields[4]),
            "memory_free_mib": _number(fields[5]), "utilization_gpu_percent": _number(fields[6]),
        })
    if {row["index"] for row in gpu_rows} != set(gpu_ids):
        raise V2Error("GPU_INVENTORY_MISMATCH")
    uuid_to_gpu = {str(row["uuid"]): int(row["index"]) for row in gpu_rows}
    compute = _compute_processes(_query_nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ), uuid_to_gpu)
    pmon = _pmon_processes(_query_nvidia_smi("pmon", "-c", "1"))
    by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpu_ids}
    seen: set[tuple[int | None, int]] = set()
    unmapped: list[dict[str, Any]] = []
    for process in compute + pmon:
        key = (process.get("gpu_id"), int(process["pid"]))
        if key in seen:
            continue
        seen.add(key)
        if process.get("gpu_id") in by_gpu:
            by_gpu[int(process["gpu_id"])].append(process)
        else:
            unmapped.append(process)
    decisions: list[dict[str, Any]] = []
    for row in sorted(gpu_rows, key=lambda item: int(item["index"])):
        gpu = int(row["index"])
        reasons: list[str] = []
        used = row.get("memory_used_mib")
        if not isinstance(used, (int, float)) or used > idle_memory_max_mib:
            reasons.append("MEMORY_NOT_IDLE")
        if by_gpu[gpu]:
            reasons.append("FOREIGN_PROCESS_PRESENT")
        if unmapped:
            reasons.append("UNMAPPED_PROCESS_TELEMETRY")
        decisions.append({**row, "compute_pids": [p["pid"] for p in by_gpu[gpu] if p["kind"] == "COMPUTE"],
                          "foreign_processes": by_gpu[gpu], "safe": not reasons, "reasons": reasons})
    all_safe = not unmapped and all(bool(row["safe"]) for row in decisions)
    return {
        "schema": "STAGE_V_M1_V2_GPU_PREFLIGHT_V1",
        "status": "PASS" if all_safe else "HOLD_WAIT_FOR_8GPU_SAFE",
        "gpu_ids": list(gpu_ids), "all_8_safe": all_safe,
        "idle_memory_max_mib": idle_memory_max_mib, "gpu_rows": decisions,
        "unmapped_processes": unmapped,
        "captured_utc": _now(), "foreign_processes_touched": False, "gpu5_touched": False,
    }


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = _load(path)
    required = {
        "schema": "STAGE_V_M1_VISUAL_DETERMINISM_PROTOCOL_V2_8GPU",
        "status": "FROZEN_DIAGNOSTIC_ONLY_NO_SCIENCE_AUTHORIZATION",
        "gpu_ids": list(GPU_IDS), "workers": 8, "runs_per_gpu": 4,
        "total_r1_runs": 32, "parallelism": 8, "seed": 7,
        "worker_gpu_mapping": "FIXED_WORKER_I_TO_GPU_I",
        "fresh_subprocess_per_run": True, "lockstep_barriers": True,
        "renderer_binding_canary_required": True, "tolerance_allowed": False,
        "gpu5_authorized": True,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise V2Error(f"V2_PROTOCOL_INVALID:{key}")
    if protocol.get("phase_order") != list(PHASES):
        raise V2Error("V2_PHASE_ORDER_INVALID")
    v1_path = REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v1.json"
    v1 = _load(v1_path)
    if _git_blob_sha("configs/stage_v_m1_visual_determinism_protocol_v1.json") != protocol.get("base_v1_visual_protocol_sha256"):
        raise V2Error("V1_VISUAL_PROTOCOL_CHANGED")
    if v1.get("rb1_v1_protocol_sha256") != protocol.get("base_rb1_v1_protocol_sha256"):
        raise V2Error("V1_RB1_PROTOCOL_BINDING_CHANGED")
    if v1.get("gpu5_authorized") is not False or v1.get("rb1_v1_modified") is not False:
        raise V2Error("V1_BOUNDARY_INVALID")
    if set(protocol.get("classification_enum", [])) != {
        "RAW_OBSERVATION_NON_POLICY_DIFFERENCE", "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM",
        "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE", "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE",
        "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM", "SIMULATOR_RUNTIME_NONDETERMINISM",
        "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE", "HETEROGENEOUS_MULTI_GPU_DIVERGENCE",
        "MULTI_LAYER_NONDETERMINISM", "UNCLASSIFIED",
    }:
        raise V2Error("V2_CLASSIFICATION_ENUM_INVALID")
    return protocol


def _verify_source_binding(manifest: Mapping[str, Any]) -> None:
    if _git("status", "--porcelain"):
        raise V2Error("V2_SOURCE_WORKTREE_DIRTY")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")


def _reject_v1_root(root: Path) -> None:
    if "M1_VISUAL_DETERMINISM" in root.name or (root / "M1_MANIFEST.json").exists():
        raise V2Error("V2_MUST_NOT_TOUCH_V1_ROOT")


def prepare_root(root: Path, protocol: Mapping[str, Any], *, source_commit: str, source_tree: str, model_path: str) -> None:
    if root.exists():
        raise V2Error("V2_ROOT_MUST_BE_NEW")
    if _git("rev-parse", "HEAD") != source_commit or _git("rev-parse", "HEAD^{tree}") != source_tree:
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")
    root.mkdir(parents=True)
    manifest = {
        "schema": "STAGE_V_M1_V2_8GPU_MANIFEST_V1", "status": "PREPARED_NO_RUNTIME_STARTED",
        "created_utc": _now(), "protocol": "M1_V2_8GPU", "protocol_sha256": sha256_file(REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json"),
        "diagnostic_identity": IDENTITY, "source_commit": source_commit, "source_tree": source_tree,
        "gpu_ids": list(GPU_IDS), "workers": 8, "runs_per_gpu": 4, "total_r1_runs": 32,
        "phase_order": list(PHASES), "seed": 7, "model_path": model_path,
        "new_science_rollouts_authorized": False, "formal_parent_promotion_authorized": False,
        "eval160_authorized": False, "protected_evaluation_authorized": False,
        "vis_pgd_authorized": False, "student_training_authorized": False,
        "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0,
        "intervention_applied_steps": 0, "counterfactual_open_steps": 0,
    }
    _write(root / "M1_V2_MANIFEST.json", manifest)
    _write(root / "M1_V2_STATUS.json", {"schema": "STAGE_V_M1_V2_STATUS_V1", "status": "PREPARED_NO_RUNTIME_STARTED", "r1_started": False, "r2_started": False, "classification": "UNCLASSIFIED"})


def _require_manifest(root: Path) -> dict[str, Any]:
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise V2Error("V2_ROOT_ALREADY_CONSUMED")
    if manifest.get("diagnostic_identity") != IDENTITY:
        raise V2Error("V2_IDENTITY_MISMATCH")
    if any(manifest.get(field) is not False for field in ("new_science_rollouts_authorized", "formal_parent_promotion_authorized", "eval160_authorized", "protected_evaluation_authorized", "vis_pgd_authorized", "student_training_authorized")):
        raise V2Error("V2_BOUNDARY_INVALID")
    if any(manifest.get(field, 0) != 0 for field in BOUNDARIES):
        raise V2Error("V2_PROTECTED_BOUNDARY_NONZERO")
    _verify_source_binding(manifest)
    return manifest


def _preflight_file(root: Path) -> dict[str, Any]:
    value = _load(root / "M1_V2_GPU_PREFLIGHT.json")
    if value.get("status") != "PASS" or value.get("all_8_safe") is not True:
        raise V2Error("HOLD_WAIT_FOR_8GPU_SAFE")
    return value


def _run_renderer_canary_child(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch
    if not torch.cuda.is_available():
        raise V2Error("CANARY_CUDA_UNAVAILABLE")
    candidate = _load(args.candidate)
    sys.path.insert(0, str(args.upstream_root))
    official_src = args.official_snapshot_root / "src"
    import gripper_attack
    gripper_attack.__path__.append(str(official_src / "gripper_attack"))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite_instance = benchmark.get_benchmark_dict()[args.suite]()
    task = suite_instance.get_task(int(candidate["task_index"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=8, camera_widths=8, has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, render_gpu_device_id=int(args.gpu), horizon=1)
    try:
        observed = None
        for obj in (env, getattr(env, "sim", None), getattr(getattr(env, "sim", None), "render_context", None)):
            for name in ("render_gpu_device_id", "gpu_device_id", "device_id"):
                value = getattr(obj, name, None) if obj is not None else None
                if value is not None:
                    observed = int(value)
                    break
            if observed is not None:
                break
        if observed != int(args.gpu):
            raise V2Error("EGL_DEVICE_BINDING_MISMATCH")
        properties = torch.cuda.get_device_properties(0)
        gpu_uuid = str(getattr(properties, "uuid", "")).strip()
        if not gpu_uuid:
            raise V2Error("CANARY_GPU_UUID_UNAVAILABLE")
        renderer_device_information = {
            "env_class": type(env).__name__,
            "sim_class": type(getattr(env, "sim", None)).__name__,
            "render_context_class": type(getattr(getattr(env, "sim", None), "render_context", None)).__name__,
            "observed_device_id": observed,
        }
        result = {
            "schema": "STAGE_V_M1_V2_RENDERER_BINDING_CANARY_V1", "status": "PASS",
            "logical_worker_id": f"worker_{int(args.gpu)}", "requested_physical_gpu": int(args.gpu),
            "physical_gpu_index": int(args.gpu), "gpu_uuid": gpu_uuid,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_current_device": int(torch.cuda.current_device()), "cuda_logical_device": 0,
            "cuda_device_name": torch.cuda.get_device_name(0), "mujoco_gl": os.environ["MUJOCO_GL"],
            "egl_device_identifier": observed, "egl_device_id": observed,
            "egl_binding_source": "OffScreenRenderEnv.render_gpu_device_id",
            "renderer_device_information": renderer_device_information,
            "episode_started": False,
        }
        validate_binding_receipt(result, int(args.gpu))
    finally:
        env.close()
    _write(args.canary_output, result)
    return 0


def run_renderer_canary(root: Path, args: argparse.Namespace) -> None:
    _require_manifest(root)
    _preflight_file(root)
    output_root = root / "renderer_canary"
    output_root.mkdir(parents=True, exist_ok=True)

    def one(gpu: int) -> dict[str, Any]:
        output = output_root / f"gpu_{gpu:02d}.json"
        command = [str(sys.executable), str(Path(__file__).resolve()), "--renderer-canary", "--gpu", str(gpu), "--canary-output", str(output), "--candidate", str(args.candidate), "--suite", args.suite, "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root)]
        result = subprocess.run(command, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu)}, capture_output=True, text=True)
        return {"gpu": gpu, "returncode": result.returncode, "stderr": result.stderr[-1000:], "output": str(output)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, GPU_IDS))
    if any(item["returncode"] != 0 or not Path(item["output"]).is_file() or _load(Path(item["output"])).get("status") != "PASS" for item in results):
        _write(root / "M1_V2_RENDERER_CANARY.json", {"schema": "STAGE_V_M1_V2_RENDERER_CANARY_AGGREGATE_V1", "status": "HOLD_RENDER_DEVICE_BINDING_MISMATCH", "results": results})
        raise V2Error("HOLD_RENDER_DEVICE_BINDING_MISMATCH")
    for gpu in GPU_IDS:
        validate_binding_receipt(_load(output_root / f"gpu_{gpu:02d}.json"), gpu)
    _write(root / "M1_V2_RENDERER_CANARY.json", {"schema": "STAGE_V_M1_V2_RENDERER_CANARY_AGGREGATE_V1", "status": "PASS", "gpu_ids": list(GPU_IDS), "results": results})


def _run_one(root: Path, args: argparse.Namespace, gpu: int, label: str, run_set: str) -> dict[str, Any]:
    mode, replicate = PHASES[label]
    base = root / ("runs" if run_set == "r1" else "raw_runs") / f"gpu_{gpu:02d}"
    output = base / label
    if output.exists():
        return {"gpu": gpu, "label": label, "status": "HOLD_PARTIAL_R1_ARTIFACT", "reason": "OUTPUT_ALREADY_EXISTS"}
    log_root = root / "logs" / f"gpu_{gpu:02d}"
    log_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "7", "PYTHONUNBUFFERED": "1"}
    command = [str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/run_stage_v_canonical_clean.py"), "--candidate", str(args.candidate), "--contract", str(args.contract), "--output-dir", str(output), "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root), "--model-path", str(args.model_path), "--suite", args.suite, "--gpu", str(gpu), "--seed", "7", "--mode", mode, "--source-commit", str(args.source_commit), "--source-tree", str(args.source_tree), "--enable-runtime"]
    if run_set == "r2":
        command.extend(["--raw-capture-plan", str(args.raw_capture_plan)])
    with (log_root / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout, (log_root / f"{label}.stderr.log").open("w", encoding="utf-8") as stderr:
        run = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
    if run.returncode != 0:
        return {"gpu": gpu, "label": label, "status": "FAIL", "returncode": run.returncode}
    producer = output / "RB1_PRODUCER_RECEIPT.json"
    independent = output / "RB1_INDEPENDENT_RECEIPT.json"
    audit = subprocess.run([str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/audit_stage_v_rb1_receipt.py"), "--protocol", str(REPO_ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json"), "--receipt", str(producer), "--artifact-root", str(output / "trace"), "--core", str(REPO_ROOT / "src/gripper_attack/stage_v_canonical_execution_core.py"), "--output", str(independent), "--repo", str(REPO_ROOT)], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    if audit.returncode != 0 or not independent.is_file():
        return {"gpu": gpu, "label": label, "status": "FAIL_AUDIT", "returncode": audit.returncode, "stderr": audit.stderr[-1000:]}
    try:
        canary = _load(root / "renderer_canary" / f"gpu_{gpu:02d}.json")
        validate_binding_receipt(canary, gpu)
        _write(output / "M1_V2_WORKER_BINDING_RECEIPT.json", {
            "schema": "STAGE_V_M1_V2_WORKER_BINDING_RECEIPT_V1", "status": "PASS",
            "logical_worker_id": canary["logical_worker_id"], "requested_physical_gpu": gpu,
            "gpu_uuid": canary["gpu_uuid"], "cuda_visible_devices": canary["cuda_visible_devices"],
            "torch_current_device": canary["torch_current_device"], "mujoco_gl": canary["mujoco_gl"],
            "egl_device_identifier": canary["egl_device_identifier"],
            "renderer_device_information": canary["renderer_device_information"],
            "run_set": run_set, "phase": label,
            "renderer_canary_receipt": f"renderer_canary/gpu_{gpu:02d}.json",
        })
    except (OSError, ValueError, KeyError, V2Error) as exc:
        return {"gpu": gpu, "label": label, "status": "FAIL_BINDING_RECEIPT", "reason": str(exc)}
    return {"gpu": gpu, "label": label, "status": "PASS", "output": str(output), "mode": mode, "replicate": replicate}


def run_matrix(root: Path, args: argparse.Namespace, run_set: str) -> None:
    manifest = _require_manifest(root)
    if args.source_commit != manifest.get("source_commit") or args.source_tree != manifest.get("source_tree"):
        raise V2Error("V2_SOURCE_BINDING_ARGUMENT_MISMATCH")
    _preflight_file(root)
    if not _load(root / "M1_V2_RENDERER_CANARY.json").get("status") == "PASS":
        raise V2Error("HOLD_RENDER_DEVICE_BINDING_MISMATCH")
    if run_set == "r2" and not args.raw_capture_plan:
        raise V2Error("V2_RAW_CAPTURE_PLAN_REQUIRED")
    status_path = root / "M1_V2_STATUS.json"
    status = _load(status_path)
    status.update({"status": "RUNNING_R1" if run_set == "r1" else "RUNNING_R2", "phase": "M1-R1" if run_set == "r1" else "M1-R2", "source_commit": manifest["source_commit"], "source_tree": manifest["source_tree"], "r1_started": bool(status.get("r1_started")) or run_set == "r1", "r2_started": bool(status.get("r2_started")) or run_set == "r2", "protected_boundaries": {key: 0 for key in BOUNDARIES}})
    _write(status_path, status)
    for label in PHASES:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda gpu: _run_one(root, args, gpu, label, run_set), GPU_IDS))
        _write(root / f"M1_V2_{run_set.upper()}_{label}_RECEIPTS.json", {"schema": "STAGE_V_M1_V2_PHASE_RECEIPTS_V1", "phase": label, "results": results})
        if any(item.get("status") != "PASS" for item in results):
            status.update({"status": "HOLD_PARTIAL_R1", "failed_phase": label, "results": results})
            _write(status_path, status)
            raise V2Error(f"HOLD_PARTIAL_R1:{label}")
    status.update({"status": "R1_COMPLETE_PENDING_AUDIT" if run_set == "r1" else "R2_COMPLETE_PENDING_AUDIT", "completed_run_set": run_set})
    _write(status_path, status)
    audit_command = [str(sys.executable), str(REPO_ROOT / "scripts/detector_v5/audit_stage_v_m1_v2_8gpu.py"), "--root", str(root)]
    if run_set == "r2":
        audit_command.append("--final")
    audit = subprocess.run(audit_command, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if audit.returncode != 0:
        status.update({"status": "HOLD_AUDIT_DISAGREEMENT", "audit_stdout": audit.stdout[-2000:], "audit_stderr": audit.stderr[-2000:]})
        _write(status_path, status)
        raise V2Error("HOLD_AUDIT_DISAGREEMENT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-root", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--run-canary", action="store_true")
    modes.add_argument("--run-r1", action="store_true")
    modes.add_argument("--run-r2", action="store_true")
    modes.add_argument("--renderer-canary", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_8gpu.json")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    parser.add_argument("--model-path")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--official-snapshot-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--canary-output", type=Path)
    parser.add_argument("--raw-capture-plan", type=Path)
    args = parser.parse_args(argv)
    try:
        protocol = validate_protocol(args.protocol.resolve())
        if args.renderer_canary:
            if args.gpu not in GPU_IDS or not args.canary_output or not args.candidate or not args.official_snapshot_root or not args.upstream_root:
                raise V2Error("CANARY_ARGUMENTS_REQUIRED")
            return _run_renderer_canary_child(args)
        if sys.prefix != PYTHON_PREFIX:
            raise V2Error(f"V2_PYTHON_PREFIX_MISMATCH:{sys.prefix}")
        root = args.root.resolve()
        _reject_v1_root(root)
        if args.prepare_root:
            if not args.source_commit or not args.source_tree or not args.model_path:
                raise V2Error("PREPARE_ROOT_BINDING_ARGUMENTS_REQUIRED")
            prepare_root(root, protocol, source_commit=args.source_commit, source_tree=args.source_tree, model_path=args.model_path)
            return 0
        if args.preflight_only:
            root.mkdir(parents=True, exist_ok=True)
            _write(root / "M1_V2_GPU_PREFLIGHT.json", gpu_preflight(idle_memory_max_mib=int(protocol["idle_memory_max_mib"])))
            return 0
        if not args.candidate or not args.official_snapshot_root or not args.upstream_root or not args.source_commit or not args.source_tree:
            raise V2Error("RUNTIME_ARGUMENTS_REQUIRED")
        if args.run_canary:
            run_renderer_canary(root, args)
        elif args.run_r1:
            if not args.contract or not args.model_path:
                raise V2Error("R1_ARGUMENTS_REQUIRED")
            run_matrix(root, args, "r1")
        else:
            if not args.contract or not args.model_path:
                raise V2Error("R2_ARGUMENTS_REQUIRED")
            run_matrix(root, args, "r2")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, V2Error) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
