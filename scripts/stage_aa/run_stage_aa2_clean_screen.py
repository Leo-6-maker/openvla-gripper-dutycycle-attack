#!/usr/bin/env python3
"""Run one pre-registered AA2 clean-only model-parent cell.

AA2 evaluates clean eligibility only.  It never opens the gripper, reads an
attack endpoint, reads task success, or enters a protected evaluation path.
The receipt is written before runtime work and is updated on every failure so
an interrupted cell cannot disappear from the census.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
QUEUE_LENGTH = {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5}
BOUNDARY = {
    "M0_OPENVLA": "FRESH_PER_STEP",
    "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE",
    "M2_PI05_LIBERO": "FRESH_PI05_REPLAN",
}
CRITICAL_SALT = "STAGE_AA_AA2_CRITICAL_ANCHOR_V1_20260826"
NONCRITICAL_SALT = "STAGE_AA_AA2_NONCRITICAL_ANCHOR_V1_20260826"
CLEAN_SEED_SALT = "STAGE_AA_AA2_CLEAN_SCREEN_SEED_V1_20260826"
ACTION_DIM = 7


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AA1 = load_module(ROOT / "scripts/stage_aa/run_stage_aa1_engineering_canary.py", "aa2_aa1_runtime")


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
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def step_unpack(result: Any) -> tuple[Any, bool]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise RuntimeError("AA2_ENV_STEP_RETURN_INVALID")
    return result[0], bool(result[2]) if len(result) == 4 else bool(result[2] or result[3])


def set_clean_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def finite_vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(np.isfinite(float(item)) for item in value)


def telemetry_valid(row: dict[str, Any]) -> bool:
    return (
        row.get("contact_telemetry_valid") is True
        and isinstance(row.get("object_identity"), str)
        and bool(row["object_identity"])
        and finite_vector(row.get("object_position"), 3)
        and finite_vector(row.get("eef_position"), 3)
        and isinstance(row.get("object_eef_distance_m"), (int, float))
        and np.isfinite(float(row["object_eef_distance_m"]))
        and isinstance(row.get("object_gripper_contact"), bool)
        and isinstance(row.get("object_support_contact"), bool)
    )


def stable_grasp(row: dict[str, Any], baseline_z: float) -> bool:
    if not telemetry_valid(row):
        return False
    return (
        row["object_gripper_contact"] is True
        and row["object_support_contact"] is False
        and float(row["object_position"][2]) - baseline_z >= 0.015
        and float(row["object_eef_distance_m"]) <= 0.12
        and float(row["object_eef_distance_m"]) <= 0.04
    )


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "step",
            "remaining_horizon",
            "model_boundary",
            "terminal_before",
            "terminal_after",
            "raw_gripper",
            "env_gripper",
            "contact_telemetry_valid",
            "object_identity",
            "object_position",
            "eef_position",
            "object_eef_distance_m",
            "object_gripper_contact",
            "object_support_contact",
        )
    }


def rank_candidate(salt: str, family: str, parent_key: str, step: int) -> str:
    return sha256_bytes(f"{salt}|{family}|{parent_key}|{step}".encode())


def scan_candidates(
    rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    family: str,
    parent_key: str,
    baseline_z: float | None,
    complete: bool,
    anchor_class: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    def add_reason(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for step in range(max(0, len(rows) - 19)):
        if not actions[step].get("boundary"):
            continue
        window = rows[step : step + 3]
        continuation = rows[step : step + 20]
        local: list[str] = []
        if len(continuation) != 20 or not complete:
            local.append("CLEAN_TRAJECTORY_INCOMPLETE_OR_HORIZON_CENSORED")
        if baseline_z is None:
            local.append("OBJECT_BASELINE_INVALID")
        if not all(telemetry_valid(row) for row in continuation):
            local.append("TELEMETRY_INVALID_IN_ELIGIBILITY_WINDOW")
        identities = {row.get("object_identity") for row in continuation if telemetry_valid(row)}
        if len(identities) != 1:
            local.append("OBJECT_IDENTITY_NOT_CONSTANT")
        if any(row.get("terminal_before") is True or row.get("terminal_after") is True for row in continuation):
            local.append("TERMINAL_OR_NATURAL_RELEASE_IN_ELIGIBILITY_WINDOW")

        if anchor_class == "CRITICAL":
            if baseline_z is not None and not all(stable_grasp(row, baseline_z) for row in window):
                local.append("STABLE_GRASP_WINDOW_INVALID")
            if baseline_z is not None and not all(stable_grasp(row, baseline_z) for row in continuation):
                local.append("CLEAN_STABLE_CONTINUATION_INVALID")
        else:
            current = rows[step]
            if current.get("object_gripper_contact") is not False or current.get("object_support_contact") is not False:
                local.append("NOT_PRE_CONTACT")
            if baseline_z is not None and any(stable_grasp(row, baseline_z) for row in rows[:step]):
                local.append("ESTABLISHED_GRASP_ALREADY_PRESENT")

        if local:
            for reason in sorted(set(local)):
                add_reason(reason)
            continue

        salt = CRITICAL_SALT if anchor_class == "CRITICAL" else NONCRITICAL_SALT
        candidates.append(
            {
                "step": step,
                "anchor_class": anchor_class,
                "selection_rank_sha256": rank_candidate(salt, family, parent_key, step),
                "boundary": BOUNDARY[family],
                "remaining_horizon_steps": len(rows) - step,
                "candidate_digest": canonical_hash([compact_row(row) for row in continuation]),
                "evidence_rows": [compact_row(row) for row in continuation],
            }
        )
    return sorted(candidates, key=lambda row: (row["selection_rank_sha256"], row["step"])), reasons


def checkpoint_path(config: dict[str, Any], family: str, suite: str) -> Path:
    spec = config["model_families"][family]
    if family == "M0_OPENVLA":
        return Path(spec["paths"][suite])
    if family == "M1_OPENVLA_OFT":
        return Path(spec["checkpoint_root"]) / suite
    return Path(spec["checkpoint"])


def find_cell(manifest: dict[str, Any], cell_id: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("cells", []) if row.get("cell_id") == cell_id]
    if len(rows) != 1:
        raise RuntimeError(f"AA2_CELL_ID_NOT_UNIQUE:{cell_id}")
    return rows[0]


def validate_static(
    protocol: dict[str, Any],
    source: dict[str, Any],
    manifest: dict[str, Any],
    cell: dict[str, Any],
    aa0: dict[str, Any],
    capacity: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if protocol.get("status") != "STAGE_AA_AA2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE":
        raise RuntimeError("AA2_PROTOCOL_NOT_AUTHORIZED")
    if protocol.get("clean_only") is not True or protocol.get("open_intervention_allowed") is not False:
        raise RuntimeError("AA2_CLEAN_ONLY_FIREWALL_INVALID")
    if source.get("status") != "STAGE_AA_AA2_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("AA2_SOURCE_AUTHORITY_NOT_FROZEN")
    if manifest.get("status") != "STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE":
        raise RuntimeError("AA2_LAUNCH_MANIFEST_NOT_FROZEN")
    if manifest.get("cell_count") != 324 or len(manifest.get("cells", [])) != 324:
        raise RuntimeError("AA2_LAUNCH_MANIFEST_CELL_COUNT_INVALID")
    if aa0.get("status") != "STAGE_AA_AA0_PROSPECTIVE_PROTOCOL_FROZEN_STOP_FOR_PI":
        raise RuntimeError("AA0_PROTOCOL_NOT_FROZEN")
    if cell.get("model_family") not in MODELS or cell.get("suite") not in SUITES:
        raise RuntimeError("AA2_CELL_MODEL_OR_SUITE_INVALID")
    pool = set(capacity["analysis_pool_after_aa1_reservation"]["keys"])
    if cell.get("canonical_parent_key") not in pool:
        raise RuntimeError("AA2_PARENT_NOT_IN_FROZEN_POOL")
    if cell.get("canonical_parent_key") in {row["canonical_parent_key"] for row in capacity["aa1_engineering_canary_reservation"]["reserved_rows"]}:
        raise RuntimeError("AA2_CANARY_EXPOSURE_FORBIDDEN")
    if cell.get("clean_only_authorization") != "AA2_CLEAN_ONLY_NO_OPEN_NO_ATTACK_NO_PROTECTED":
        raise RuntimeError("AA2_CELL_AUTHORIZATION_INVALID")
    if cell.get("seed") is None or cell.get("eligibility_implementation_sha256") != sha256_file(ROOT / "scripts/stage_aa/run_stage_aa2_clean_screen.py"):
        raise RuntimeError("AA2_CELL_SOURCE_BINDING_INVALID")
    if config.get("environment", {}).get("dummy_wait_steps") != 10:
        raise RuntimeError("AA2_ENVIRONMENT_BINDING_INVALID")


def capture_clean(config: dict[str, Any], cell: dict[str, Any], infer: Any, counters: dict[str, int]) -> dict[str, Any]:
    family = str(cell["model_family"])
    suite = str(cell["suite"])
    parent_key = str(cell["canonical_parent_key"])
    task_idx = int(cell["source_task_idx"])
    state_id = int(str(cell["state"]).split("_")[-1])
    env, task_suite, task, obs, _initial_states = AA1.make_env(config, suite, task_idx, state_id, counters)
    try:
        binding = AA1.TAXONOMY.bind_object_taxonomy(env, AA1.bddl_path(env, task))
        if binding.get("status") != "PASS":
            return {"status": "INELIGIBLE_CLEAN_OBJECT_BINDING", "reason_codes": ["OBJECT_BINDING_INVALID"], "binding": binding, "rows": [], "actions": []}
        target = str(binding["target_object_ids"][0])
        language = str(task.language)
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        boundary_states: dict[int, np.ndarray] = {}
        done = False
        baseline_z: float | None = None
        horizon = HORIZONS[suite]
        for step in range(horizon):
            if done:
                break
            fresh = not queue
            if fresh:
                boundary_states[step] = AA1.Z1.snapshot_state(env)
                queue = AA1.model_pairs(infer, obs, language, family, counters)
            raw_action, final_action = queue.pop(0)
            current = AA1.telemetry(env, binding, target, counters)
            if baseline_z is None and finite_vector(current.get("object_position"), 3):
                baseline_z = float(current["object_position"][2])
            row = {
                "step": step,
                "remaining_horizon": horizon - step,
                "terminal_before": done,
                "model_boundary": fresh,
                "raw_action_7d": raw_action.tolist(),
                "env_action_7d": final_action.tolist(),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(final_action[-1]),
                **current,
            }
            rows.append(row)
            actions.append({"boundary": fresh, "raw": raw_action.tolist(), "final": final_action.tolist()})
            obs, done = step_unpack(env.step(final_action.tolist()))
            counters["env_step_calls"] += 1
            row["terminal_after"] = done
        complete = len(rows) == horizon
        critical, critical_reasons = scan_candidates(rows, actions, family, parent_key, baseline_z, complete, "CRITICAL")
        noncritical, noncritical_reasons = scan_candidates(rows, actions, family, parent_key, baseline_z, complete, "NONCRITICAL")
        public_candidates = lambda values: [{key: value for key, value in item.items() if key != "evidence_rows"} for item in values]
        telemetry_rows = sum(telemetry_valid(row) for row in rows)
        trajectory_digest = canonical_hash(
            {
                "rows": [compact_row(row) for row in rows],
                "actions": [{"boundary": row["boundary"], "raw": row["raw"], "final": row["final"]} for row in actions],
                "boundary_state_sha256": {str(step): sha256_bytes(state.tobytes()) for step, state in boundary_states.items()},
            }
        )
        return {
            "status": "PASS_AA2_CLEAN_TRAJECTORY_CAPTURED",
            "binding": binding,
            "target_object": target,
            "language": language,
            "horizon": horizon,
            "steps_captured": len(rows),
            "complete_trajectory": complete,
            "baseline_z_m": baseline_z,
            "telemetry_valid_rows": telemetry_rows,
            "clean_trajectory_digest": trajectory_digest,
            "boundary_count": len(boundary_states),
            "critical_candidates": public_candidates(critical),
            "noncritical_candidates": public_candidates(noncritical),
            "critical_reason_counts": critical_reasons,
            "noncritical_reason_counts": noncritical_reasons,
            "selected_critical": critical[0] if critical else None,
            "selected_noncritical": noncritical[0] if noncritical else None,
            "rows": rows,
            "actions": actions,
        }
    finally:
        env.close()


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(args.protocol)
    source = load_json(args.source_authority)
    manifest = load_json(args.launch_manifest)
    aa0 = load_json(args.aa0)
    capacity = load_json(args.capacity)
    config = load_json(args.z1_config)
    cell = find_cell(manifest, args.cell_id)
    validate_static(protocol, source, manifest, cell, aa0, capacity, config)
    counters = {
        "model_inference_calls": 0,
        "env_step_calls": 0,
        "physical_telemetry_reads": 0,
        "open_intervention_steps": 0,
        "pgd_calls": 0,
        "attacked_env_steps": 0,
        "aa_v_phys_reads": 0,
        "task_success_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
        "scientific_parent_exposure": 0,
        "aa2_exposure": 0,
    }
    receipt: dict[str, Any] = {
        "schema": "STAGE_AA_AA2_CLEAN_SCREEN_CELL_RECEIPT_V1",
        "status": "RUNNING",
        "gate": protocol["gate"],
        "cell_id": args.cell_id,
        "model_family": cell["model_family"],
        "canonical_parent_key": cell["canonical_parent_key"],
        "suite": cell["suite"],
        "source_task_idx": cell["source_task_idx"],
        "state": cell["state"],
        "seed": cell["seed"],
        "clean_only": True,
        "runtime_counters": counters,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(args.output, receipt)
    model = None
    try:
        AA1.require_single_gpu(args.gpu_id)
        receipt["gpu"] = AA1.gpu_snapshot(args.gpu_id)
        set_clean_seed(int(cell["seed"]))
        AA1.Z1.configure_libero(config)
        checkpoint = checkpoint_path(config, str(cell["model_family"]), str(cell["suite"]))
        if not checkpoint.exists():
            raise RuntimeError(f"CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
        checkpoint_manifest = None
        if cell["model_family"] == "M1_OPENVLA_OFT":
            checkpoint_manifest = AA1.Z1.verify_m1_materialization(
                Path(args.m1_manifest), checkpoint, str(cell["suite"]), str(config["model_families"]["M1_OPENVLA_OFT"]["checkpoint_manifests_sha256"])
            )
        receipt.update({"checkpoint": str(checkpoint), "checkpoint_manifest": checkpoint_manifest})
        if cell["model_family"] == "M2_PI05_LIBERO":
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if cell["model_family"] == "M0_OPENVLA":
            infer, model, normalization = AA1.Z1.load_openvla(str(checkpoint), oft=False, suite=str(cell["suite"]), return_chunk=True)
        elif cell["model_family"] == "M1_OPENVLA_OFT":
            infer, model, normalization = AA1.Z1.load_openvla(str(checkpoint), oft=True, suite=str(cell["suite"]), return_chunk=True)
        else:
            infer, model = AA1.Z1.load_pi05(str(checkpoint), return_chunk=True)
            normalization = {"checkpoint_mutated": False}
        counters["aa2_exposure"] = 1
        counters["scientific_parent_exposure"] = 1
        clean = capture_clean(config, cell, infer, counters)
        selected_critical = clean.get("selected_critical")
        selected_noncritical = clean.get("selected_noncritical")
        receipt.update(
            {
                "status": "AA2_CLEAN_CELL_COMPLETE",
                "normalization": normalization,
                "clean": {key: value for key, value in clean.items() if key not in {"rows", "actions"}},
                "eligibility": {
                    "critical": selected_critical is not None,
                    "noncritical": selected_noncritical is not None,
                    "critical_anchor": selected_critical,
                    "noncritical_anchor": selected_noncritical,
                    "noncritical_affects_primary_denominator": False,
                },
                "scientific_claim": "NONE_AA2_CLEAN_DENOMINATOR_ONLY",
                "next_legal_action": "STOP_FOR_PI_AFTER_FULL_CENSUS",
            }
        )
        receipt["runtime_counters"] = counters
        atomic_write(args.output, receipt)
        return receipt
    except Exception as exc:
        receipt.update(
            {
                "status": "AA2_ENGINEERING_HOLD_RUNTIME_ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "scientific_claim": "NONE_DUE_TO_ENGINEERING_HOLD",
                "next_legal_action": "STAGE_AA_AA2_ENGINEERING_HOLD_STOP_FOR_PI",
            }
        )
        receipt["runtime_counters"] = counters
        atomic_write(args.output, receipt)
        raise
    finally:
        if model is not None:
            del model


def self_test() -> None:
    rows = []
    actions = []
    for step in range(25):
        rows.append(
            {
                "step": step,
                "remaining_horizon": 25 - step,
                "terminal_before": False,
                "terminal_after": False,
                "contact_telemetry_valid": True,
                "object_identity": "object",
                "object_position": [0.1, 0.2, 0.3],
                "eef_position": [0.1, 0.2, 0.3],
                "object_eef_distance_m": 0.01,
                "object_gripper_contact": True,
                "object_support_contact": False,
            }
        )
        actions.append({"boundary": step == 0, "raw": [0.0] * 7, "final": [0.0] * 6 + [1.0]})
    critical, reasons = scan_candidates(rows, actions, "M0_OPENVLA", "mock/key", 0.2, True, "CRITICAL")
    assert critical and not reasons
    assert critical[0]["step"] == 0
    rows[0]["object_gripper_contact"] = False
    critical, reasons = scan_candidates(rows, actions, "M0_OPENVLA", "mock/key", 0.2, True, "CRITICAL")
    assert not critical and "STABLE_GRASP_WINDOW_INVALID" in reasons
    print(json.dumps({"status": "AA2_STATIC_MOCK_PASS"}, sort_keys=True))


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--aa0", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--z1-config", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_cell(args)
        print(json.dumps({"status": result["status"], "cell_id": args.cell_id}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "AA2_ENGINEERING_HOLD_RUNTIME_ERROR", "cell_id": args.cell_id, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
