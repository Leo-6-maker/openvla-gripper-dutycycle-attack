#!/usr/bin/env python3
"""Run one Stage-AC2 treatment-naive clean-only model-parent cell."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_ac import eligibility_v2 as ELIGIBILITY
from stage_ac import m1_manifest_authority as M1_MANIFEST


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA2R2 = load_module(ROOT / "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py", "ac2_aa2r2_helpers")
AA1 = AA2R2.AA1
Z1 = AA1.Z1
SEMANTICS = AA2R2.SEMANTICS

MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {"M0_OPENVLA": "FRESH_PER_STEP", "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE", "M2_PI05_LIBERO": "FRESH_PI05_REPLAN"}
ACTION_DIM = 7
MIN_FREE_MIB = 20_480
M1_MANIFEST_SOURCE = ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
M1_RECONCILIATION = ROOT / "reports/STAGE_AC_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_V1.json"
AC2_PROTOCOL_STATUSES = frozenset({
    "STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE",
    "STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_AUTHORIZED_PRE_RESUME",
})
AC2_SOURCE_STATUSES = frozenset({
    "STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
    "STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
})


def load_json(path: Path) -> dict[str, Any]:
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
    return sha256_bytes(json.dumps(safe_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return safe_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_value(item) for key, item in value.items()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isfinite(number):
            return number
        return "NaN" if np.isnan(number) else ("Infinity" if number > 0 else "-Infinity")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(safe_value(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def set_clean_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def step_unpack(result: Any) -> tuple[Any, bool]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise RuntimeError("AC2_ENV_STEP_RETURN_INVALID")
    return result[0], bool(result[2]) if len(result) == 3 else bool(result[2] or result[3])


def checkpoint_path(config: dict[str, Any], family: str, suite: str) -> Path:
    spec = config["model_families"][family]
    if family == "M0_OPENVLA":
        return Path(spec["paths"][suite])
    if family == "M1_OPENVLA_OFT":
        return Path(spec["checkpoint_root"]) / suite
    return Path(spec["checkpoint"])


def binding_path(root: Path, binding: dict[str, Any]) -> Path:
    path = Path(str(binding["path"]))
    return path if path.is_absolute() else root / path


def verify_binding(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = binding_path(root, binding)
    if not path.is_file():
        raise RuntimeError(f"AC2_SOURCE_FILE_MISSING:{label}:{path}")
    if int(path.stat().st_size) != int(binding["bytes"]) or sha256_file(path) != str(binding["sha256"]):
        raise RuntimeError(f"AC2_SOURCE_FILE_HASH_MISMATCH:{label}:{path}")
    return path


def find_cell(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("cells", []) if row.get("cell_id") == cell_id]
    if len(rows) != 1:
        raise RuntimeError(f"AC2_CELL_ID_NOT_UNIQUE:{cell_id}")
    return rows[0]


def find_parent(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("parents", []) if row.get("canonical_parent_key") == key]
    if len(rows) != 1:
        raise RuntimeError(f"AC2_PARENT_NOT_UNIQUE:{key}")
    return rows[0]


def validate_manifest_source_binding(root: Path, manifest: dict[str, Any], source: dict[str, Any], source_path: Path, launch_manifest_path: Path) -> None:
    manifest_binding = manifest.get("source_bindings", {}).get("runtime_source_authority")
    if source.get("status") == "STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        # The immutable V1 launch manifest points at the superseded V1 source.
        # AC2R2 keeps that manifest byte-identical and binds it through its own
        # versioned source authority instead of rewriting the historical link.
        launch_binding = source.get("input_authorities", {}).get("launch_manifest")
        if not isinstance(launch_binding, dict):
            raise RuntimeError("AC2R2_LAUNCH_MANIFEST_BINDING_MISSING")
        bound_path = binding_path(root, launch_binding)
        if bound_path.resolve() != launch_manifest_path.resolve():
            raise RuntimeError("AC2R2_LAUNCH_MANIFEST_PATH_MISMATCH")
        if int(bound_path.stat().st_size) != int(launch_binding["bytes"]) or sha256_file(bound_path) != str(launch_binding["sha256"]):
            raise RuntimeError("AC2R2_LAUNCH_MANIFEST_BINDING_INVALID")
        return
    if manifest_binding is None or sha256_file(source_path) != str(manifest_binding["sha256"]):
        raise RuntimeError("AC2_SOURCE_AUTHORITY_MANIFEST_BINDING_INVALID")


def m1_manifest_source(args: argparse.Namespace) -> Path:
    value = getattr(args, "m1_manifest", None)
    return Path(value) if value is not None else M1_MANIFEST_SOURCE


def prepare_m1_runtime_manifest(args: argparse.Namespace) -> Path:
    target = args.output.resolve().parent.parent / "authority/STAGE_AC_AC2R1_M1_MANIFEST_HISTORICAL_CRLF_V1.json"
    M1_MANIFEST.materialize_historical_runtime_manifest(m1_manifest_source(args), M1_RECONCILIATION, args.z1_config, target)
    return target


def official_final_action_adapter(infer: Any, family: str) -> Any:
    """Apply the already-frozen PI05 clip at the AC2 action boundary.

    The Z1 loader permits tiny float32 overshoots within its input tolerance;
    the official LIBERO boundary still clips those values before delivery.
    """

    if family != "M2_PI05_LIBERO":
        return infer

    def infer_with_clip(obs: dict[str, Any], language: str) -> tuple[np.ndarray, dict[str, Any]]:
        chunk, meta = infer(obs, language)
        values = np.asarray(chunk, dtype=np.float32)
        clipped = np.clip(values, -1.0, 1.0).astype(np.float32)
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["ac2_official_final_clip_applied"] = bool(not np.array_equal(values, clipped))
        return clipped, meta

    return infer_with_clip


def validate_static(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any] | None]:
    protocol = load_json(args.protocol)
    source = load_json(args.source_authority)
    manifest = load_json(args.launch_manifest)
    config = load_json(args.z1_config)
    cell = find_cell(manifest, args.cell_id)
    if protocol.get("status") not in AC2_PROTOCOL_STATUSES:
        raise RuntimeError("AC2_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("clean_only") is not True or protocol.get("open_intervention_allowed") is not False or protocol.get("attack_or_pgd_allowed") is not False:
        raise RuntimeError("AC2_CLEAN_ONLY_FIREWALL_INVALID")
    if any(protocol.get(key) is not False for key in ("physical_endpoint_read_allowed", "v_phys_read_allowed", "task_success_read_allowed", "protected_or_eval160_allowed")):
        raise RuntimeError("AC2_FORBIDDEN_READ_FIREWALL_INVALID")
    if source.get("status") not in AC2_SOURCE_STATUSES:
        raise RuntimeError("AC2_SOURCE_AUTHORITY_NOT_FROZEN")
    if manifest.get("status") != "STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE" or manifest.get("cell_count") != 720 or len(manifest.get("cells", [])) != 720:
        raise RuntimeError("AC2_LAUNCH_MANIFEST_INVALID")
    if manifest.get("population", {}).get("parent_count") != 240 or len(manifest.get("parents", [])) != 240:
        raise RuntimeError("AC2_PARENT_MANIFEST_INVALID")
    if cell.get("model_family") not in MODELS or cell.get("suite") not in SUITES:
        raise RuntimeError("AC2_CELL_MODEL_OR_SUITE_INVALID")
    if cell.get("clean_only_authorization") != "AC2_CLEAN_ONLY_NO_OPEN_NO_ATTACK_NO_PROTECTED":
        raise RuntimeError("AC2_CELL_AUTHORIZATION_INVALID")
    parent = find_parent(manifest, str(cell["canonical_parent_key"]))
    if parent.get("exposure_class") not in {"H0_UNTOUCHED", "HC_CLEAN_ONLY"} or parent.get("treatment_naive") is not True:
        raise RuntimeError("AC2_PARENT_NOT_TREATMENT_NAIVE")
    if parent.get("suite") != cell.get("suite") or parent.get("task") != cell.get("task") or parent.get("state") != cell.get("state") or parent.get("state_sha256") != cell.get("state_sha256"):
        raise RuntimeError("AC2_PARENT_CELL_BINDING_INVALID")
    if config.get("status") != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_RUNTIME_AUTHORITY_NOT_FROZEN")
    for expected in source.get("runtime_files", []):
        name = str(expected["path"])
        verify_binding(ROOT, expected, name)
    input_bindings = source.get("input_authorities", {})
    for name, expected in input_bindings.items():
        verify_binding(ROOT, expected, name)
    validate_manifest_source_binding(ROOT, manifest, source, args.source_authority, args.launch_manifest)
    config_binding = input_bindings.get("z1_protocol")
    if config_binding is None or sha256_file(args.z1_config) != str(config_binding["sha256"]):
        raise RuntimeError("AC2_Z1_PROTOCOL_BINDING_INVALID")
    m1_reconciliation_binding = input_bindings.get("m1_manifest_reconciliation")
    if m1_reconciliation_binding is None:
        raise RuntimeError("AC2R1_M1_RECONCILIATION_BINDING_MISSING")
    reconciliation_path = binding_path(ROOT, m1_reconciliation_binding)
    M1_MANIFEST.validate_reconciliation(m1_manifest_source(args), reconciliation_path, args.z1_config)
    if args.cell_id not in {row["cell_id"] for row in manifest["cells"]}:
        raise RuntimeError("AC2_CELL_NOT_IN_MANIFEST")
    if int(cell["state_id"]) != int(str(cell["state"]).split("_")[-1]) or int(cell["source_task_idx"]) != int(str(cell["task"]).split("_")[-1]):
        raise RuntimeError("AC2_OFFICIAL_INDEX_BINDING_INVALID")
    if cell.get("seed") is None or int(cell["seed"]) < 0:
        raise RuntimeError("AC2_SEED_INVALID")
    checkpoint = checkpoint_path(config, str(cell["model_family"]), str(cell["suite"]))
    if not checkpoint.is_dir():
        raise RuntimeError(f"AC2_CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
    checkpoint_manifest = None
    if cell["model_family"] == "M1_OPENVLA_OFT":
        manifest_path = prepare_m1_runtime_manifest(args)
        checkpoint_manifest = Z1.verify_m1_materialization(manifest_path, checkpoint, str(cell["suite"]), str(config["model_families"]["M1_OPENVLA_OFT"]["checkpoint_manifests_sha256"]))
    return protocol, source, cell, checkpoint, checkpoint_manifest


def capture_clean(config: dict[str, Any], cell: dict[str, Any], infer: Any, counters: dict[str, int], context: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    family = str(cell["model_family"])
    suite = str(cell["suite"])
    parent_key = str(cell["canonical_parent_key"])
    env, _task_suite, task, obs, _initial_states = AA1.make_env(config, suite, int(cell["source_task_idx"]), int(cell["state_id"]), counters)
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    boundary_states: dict[str, Any] = {}
    baseline_z: float | None = None
    done = False
    pair_infer = official_final_action_adapter(infer, family)
    try:
        binding = AA1.TAXONOMY.bind_object_taxonomy(env, AA1.bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {
                "status": "AC2_CLEAN_CELL_COMPLETE",
                "eligibility_status": "INELIGIBLE_OBJECT_BINDING",
                "binding": binding,
                "rows": rows,
                "actions": actions,
                "eligibility_rows": eligibility_rows,
                "candidate_audit": [],
                "selected_critical": None,
                "selected_noncritical": None,
            }
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        horizon = HORIZONS[suite]
        for step in range(horizon):
            if done:
                break
            fresh = not queue
            if fresh:
                context["current_step"] = step
                snapshot = Z1.snapshot_state(env)
                snapshot_value = safe_value(snapshot)
                boundary_states[str(step)] = {"sha256": sha256_bytes(np.asarray(snapshot, dtype=np.float64).tobytes()), "state": snapshot_value}
                context["partial_boundary_states"] = boundary_states
                queue = AA2R2.model_pairs_v2(pair_infer, obs, language, family, counters, context)
            raw_action, final_action = queue.pop(0)
            if raw_action.size != ACTION_DIM or final_action.size != ACTION_DIM:
                raise RuntimeError("AC2_FINAL_ACTION_NOT_EXACTLY_SEVEN")
            pre = AA1.telemetry(env, binding, target, counters)
            if baseline_z is None and isinstance(pre.get("object_position"), list) and len(pre["object_position"]) == 3:
                baseline_z = float(pre["object_position"][2])
            semantics = SEMANTICS.validate_action_pair(family, raw_action.tolist(), final_action.tolist(), raw_gripper=float(raw_action[-1]), final_gripper=float(final_action[-1]))
            if not semantics.get("accepted"):
                raise RuntimeError(f"AC2_ACTION_SEMANTICS_INVALID:{semantics.get('reason')}")
            obs, done = step_unpack(env.step(final_action.tolist()))
            counters["env_step_calls"] += 1
            post = AA1.telemetry(env, binding, target, counters)
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "model_boundary": fresh,
                "terminal_before": False,
                "terminal_after": done,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                "action_semantics": semantics,
                "boundary_state_sha256": boundary_states[str(step)]["sha256"] if fresh else None,
                "pre": pre,
                "post": post,
            }
            rows.append(row)
            actions.append({"step": step, "boundary": fresh, "raw": raw_action.tolist(), "final": final_action.tolist(), "boundary_state_sha256": row["boundary_state_sha256"]})
            eligibility_rows.append({**post, "step": step, "model_boundary": fresh, "terminal_before": False, "terminal_after": done})
            context["partial_rows"] = rows
            context["partial_actions"] = actions
            context["partial_boundary_states"] = boundary_states
        candidate_audit: list[dict[str, Any]] = []
        critical_candidates: list[dict[str, Any]] = []
        noncritical_candidates: list[dict[str, Any]] = []
        for anchor_class, salt, output in (
            ("CRITICAL", str(eligibility["critical_selection_salt"]), critical_candidates),
            ("NONCRITICAL", str(eligibility["noncritical_selection_salt"]), noncritical_candidates),
        ):
            for step in range(max(0, len(eligibility_rows) - ELIGIBILITY.CLEAN_CONTINUATION_STEPS + 1)):
                result = ELIGIBILITY.evaluate_candidate(eligibility_rows, actions, step, baseline_z, anchor_class, 0)
                rank = ELIGIBILITY.rank_candidate(salt, family, parent_key, step)
                evidence = eligibility_rows[step : step + ELIGIBILITY.CLEAN_CONTINUATION_STEPS]
                audit = {
                    "step": step,
                    "anchor_class": anchor_class,
                    "boundary": bool(actions[step].get("boundary")),
                    "selection_rank_sha256": rank,
                    "eligible": bool(result["eligible"]),
                    "reason_codes": list(result["reason_codes"]),
                    "metrics": safe_value(result.get("metrics", {})),
                    "continuation_steps": [int(row["step"]) for row in evidence],
                    "continuation_digest": canonical_hash(evidence),
                    "boundary_state_sha256": actions[step].get("boundary_state_sha256"),
                }
                candidate_audit.append(audit)
                if result["eligible"]:
                    output.append({**audit, "evidence_rows": safe_value(evidence)})
        critical_candidates.sort(key=lambda row: (row["selection_rank_sha256"], row["step"]))
        noncritical_candidates.sort(key=lambda row: (row["selection_rank_sha256"], row["step"]))
        return {
            "status": "AC2_CLEAN_CELL_COMPLETE",
            "eligibility_status": "ELIGIBLE_CRITICAL" if critical_candidates else "INELIGIBLE_CRITICAL",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "steps_captured": len(rows),
            "complete_trajectory": len(rows) == horizon,
            "baseline_z_m": baseline_z,
            "telemetry_valid_rows": sum(ELIGIBILITY.telemetry_valid(row) for row in eligibility_rows),
            "clean_trajectory_digest": canonical_hash({"rows": rows, "actions": actions, "boundary_states": boundary_states}),
            "boundary_count": len(boundary_states),
            "boundary_state_sha256": {key: value["sha256"] for key, value in boundary_states.items()},
            "rows": rows,
            "actions": actions,
            "eligibility_rows": eligibility_rows,
            "boundary_states": boundary_states,
            "candidate_audit": sorted(candidate_audit, key=lambda row: (row["anchor_class"], row["step"])),
            "critical_reason_counts": _reason_counts(candidate_audit, "CRITICAL"),
            "noncritical_reason_counts": _reason_counts(candidate_audit, "NONCRITICAL"),
            "critical_candidates": critical_candidates,
            "noncritical_candidates": noncritical_candidates,
            "selected_critical": critical_candidates[0] if critical_candidates else None,
            "selected_noncritical": noncritical_candidates[0] if noncritical_candidates else None,
        }
    finally:
        env.close()


def _reason_counts(audit: list[dict[str, Any]], anchor_class: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in audit:
        if row["anchor_class"] != anchor_class:
            continue
        for reason in row["reason_codes"]:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def empty_counters() -> dict[str, int]:
    return {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "dummy_wait_env_step_calls": 0,
        "clean_telemetry_reads": 0,
        "physical_telemetry_reads": 0,
        "physical_endpoint_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "v_phys_reads": 0,
        "aa_v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 1,
    }


def load_model(config: dict[str, Any], family: str, suite: str, runtime_manifest: Path | None = None) -> tuple[Any, Any, dict[str, Any], Path, dict[str, Any] | None]:
    checkpoint = checkpoint_path(config, family, suite)
    checkpoint_manifest = None
    if family == "M1_OPENVLA_OFT":
        manifest_path = runtime_manifest or M1_MANIFEST_SOURCE
        checkpoint_manifest = Z1.verify_m1_materialization(manifest_path, checkpoint, suite, str(config["model_families"][family]["checkpoint_manifests_sha256"]))
    if family == "M0_OPENVLA":
        infer, model, normalization = Z1.load_openvla(str(checkpoint), oft=False, suite=suite, return_chunk=True)
    elif family == "M1_OPENVLA_OFT":
        infer, model, normalization = Z1.load_openvla(str(checkpoint), oft=True, suite=suite, return_chunk=True)
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        infer, model = Z1.load_pi05(str(checkpoint), return_chunk=True)
        normalization = {"checkpoint_mutated": False}
    return infer, model, normalization, checkpoint, checkpoint_manifest


def run_loaded_cell(args: argparse.Namespace, protocol: dict[str, Any], source: dict[str, Any], cell: dict[str, Any], config: dict[str, Any], infer: Any, normalization: dict[str, Any], checkpoint: Path, checkpoint_manifest: dict[str, Any] | None, gpu: dict[str, Any]) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"AC2_APPEND_ONLY_OUTPUT_EXISTS:{output}")
    counters = empty_counters()
    counters["scientific_parent_exposure"] = 1
    receipt: dict[str, Any] = {
        "schema": "STAGE_AC_AC2_CLEAN_SCREEN_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "cell_id": cell["cell_id"],
        "model_family": cell["model_family"],
        "canonical_parent_key": cell["canonical_parent_key"],
        "suite": cell["suite"],
        "task": cell["task"],
        "source_task_idx": cell["source_task_idx"],
        "state": cell["state"],
        "state_id": cell["state_id"],
        "state_sha256": cell["state_sha256"],
        "parent_exposure_class": cell["parent_exposure_class"],
        "seed": cell["seed"],
        "gpu_id": args.gpu_id,
        "gpu_admission_snapshot": gpu,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": checkpoint_manifest,
        "clean_only": True,
        "runtime_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "UNSET"),
        },
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(output, receipt)
    context: dict[str, Any] = {
        "receipt": receipt,
        "output": output,
        "counters": counters,
        "pair_audits": [],
        "family": cell["model_family"],
        "current_step": 0,
        "failure_persisted": False,
        "partial_rows": [],
        "partial_actions": [],
        "partial_boundary_states": {},
    }
    set_clean_seed(int(cell["seed"]))
    try:
        Z1.configure_libero(config)
        clean = capture_clean(config, cell, infer, counters, context, protocol["eligibility"])
        receipt.update(
            {
                "status": "AC2_CLEAN_CELL_COMPLETE",
                "normalization": normalization,
                "clean": clean,
                "action_pair_audit": context["pair_audits"],
                "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
                "runtime_counters": counters,
                "scientific_claim": "NONE_AC2_TREATMENT_ONLY_CLEAN_ELIGIBILITY",
                "claim_boundary": "AC2 clean-only screening and model-specific eligibility evidence; no OPEN, endpoint, V_phys, protected read, or AC3 outcome.",
                "next_legal_action": "STOP_FOR_PI_AFTER_FULL_AC2_CENSUS",
            }
        )
        atomic_write(output, receipt)
        return receipt
    except Exception as exc:
        partial = {
            "rows": context.get("partial_rows", []),
            "actions": context.get("partial_actions", []),
            "boundary_states": context.get("partial_boundary_states", {}),
            "action_pair_audit": context.get("pair_audits", []),
        }
        receipt.update(
            {
                "status": "AC2_ENGINEERING_HOLD_RUNTIME_ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc), "partial_evidence_digest": canonical_hash(partial)},
                "partial_clean_evidence": safe_value(partial),
                "action_pair_audit_sha256": canonical_hash(context["pair_audits"]),
                "runtime_counters": counters,
                "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
                "next_legal_action": "STOP_FOR_PI",
            }
        )
        atomic_write(output, receipt)
        raise


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol, source, cell, checkpoint, checkpoint_manifest = validate_static(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        raise RuntimeError("AC2_SINGLE_GPU_BINDING_INVALID")
    AA1.require_single_gpu(args.gpu_id)
    gpu = AA1.gpu_snapshot(args.gpu_id)
    config = load_json(args.z1_config)
    infer = model = None
    try:
        runtime_manifest = prepare_m1_runtime_manifest(args) if cell["model_family"] == "M1_OPENVLA_OFT" else None
        infer, model, normalization, checkpoint, checkpoint_manifest = load_model(config, str(cell["model_family"]), str(cell["suite"]), runtime_manifest)
        return run_loaded_cell(args, protocol, source, cell, config, infer, normalization, checkpoint, checkpoint_manifest, gpu)
    finally:
        if model is not None:
            del model
            gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=False)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "cell_id": args.cell_id, "eligibility_status": result.get("clean", {}).get("eligibility_status")}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AC2_ENGINEERING_HOLD_RUNTIME_ERROR", "cell_id": args.cell_id, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
