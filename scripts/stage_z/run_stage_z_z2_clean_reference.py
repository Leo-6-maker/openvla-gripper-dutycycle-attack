#!/usr/bin/env python3
"""Run one Z2 clean reference and outcome-blind anchor selection cell.

The cell records model clean actions and privileged instantaneous geometry only.
It never reads task success, reward, terminal outcome, attack state, or the
protected evaluation namespace.  Z3 treatment branches consume the selected
anchor snapshot and clean action prefix from this receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
Z1_RUNNER = ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
MODEL_FAMILIES = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
H_PHYS = 10
T10 = 10
OFT_QUEUE = 8
PI05_REPLAN = 5
CRITICAL_PHASES = frozenset({"CONTACT_MANIPULATION", "ENGAGED_LIFT", "CARRY"})
R1_RECEIPT_SCHEMA = "STAGE_Z_Z2R1_CLEAN_REFERENCE_CELL_RECEIPT_V1"


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_value(path: str, *args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", path, *args], text=True).strip()


def panel_row(panel: dict[str, Any], parent_key: str) -> dict[str, Any]:
    rows = [row for row in panel.get("rows", []) if row.get("canonical_parent_key") == parent_key]
    if len(rows) != 1:
        raise RuntimeError(f"PANEL_PARENT_NOT_UNIQUE:{parent_key}")
    row = rows[0]
    if row.get("state") != "FROZEN_SHARED_FRESH":
        raise RuntimeError("PANEL_PARENT_NOT_FRESH")
    return row


def verify_static_authority(
    config: dict[str, Any],
    panel: dict[str, Any],
    parent_key: str,
    suite: str,
    *,
    canary_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config.get("status") != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_SOURCE_AUTHORITY_NOT_FROZEN")
    if config["z0r2"]["root_seal_sha256"] != "e6e3db5a9f5e7641d2c09c0b0ca225ca99763494430cdda22830edab1853053d":
        raise RuntimeError("Z0R2_ROOT_BINDING_INVALID")
    if config["z0r2"]["panel_sha256"] != "2d0066eba451006e81db490665e7822c35521caca4e96f7a7d7980f021506ef1":
        raise RuntimeError("Z0R1_PANEL_BINDING_INVALID")
    if canary_ledger is None:
        row = panel_row(panel, parent_key)
    else:
        if canary_ledger.get("status") != "STAGE_Z_Z1_ENGINEERING_CANARY_LEDGER_FROZEN":
            raise RuntimeError("CANARY_LEDGER_NOT_FROZEN")
        source = canary_ledger.get("source_authority", {})
        if source.get("scientific_panel", {}).get("overlap_count") != 0:
            raise RuntimeError("CANARY_SCIENTIFIC_PANEL_OVERLAP")
        matches = [
            item
            for item in canary_ledger.get("selected", [])
            if item.get("canonical_parent_key") == parent_key
            and item.get("suite") == suite
            and item.get("role") == "PRIMARY"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"CANARY_PARENT_NOT_UNIQUE:{parent_key}")
        row = matches[0]
        if row.get("permanent_exclusion") is not True or row.get("scientific_use") is not False or row.get("outcome_read") is not False:
            raise RuntimeError("CANARY_EXCLUSION_CONTRACT_INVALID")
        if parent_key in {item.get("canonical_parent_key") for item in panel.get("rows", [])}:
            raise RuntimeError("CANARY_PARENT_IN_SCIENTIFIC_PANEL")
    if row.get("suite") != suite:
        raise RuntimeError("PANEL_SUITE_MISMATCH")
    common = config["environment"]["common_libero_checkout"]
    if git_value(common, "rev-parse", "HEAD") != config["environment"]["common_libero_commit"]:
        raise RuntimeError("COMMON_LIBERO_COMMIT_MISMATCH")
    if git_value(common, "rev-parse", "HEAD^{tree}") != config["environment"]["common_libero_tree"]:
        raise RuntimeError("COMMON_LIBERO_TREE_MISMATCH")
    if git_value(common, "status", "--short"):
        raise RuntimeError("COMMON_LIBERO_SOURCE_DIRTY")
    for family in ("M1_OPENVLA_OFT", "M2_PI05_LIBERO"):
        spec = config["model_families"][family]
        source = spec["source_checkout"]
        if git_value(source, "rev-parse", "HEAD") != spec["source_commit"]:
            raise RuntimeError(f"{family}_SOURCE_COMMIT_MISMATCH")
        if git_value(source, "rev-parse", "HEAD^{tree}") != spec["source_tree"]:
            raise RuntimeError(f"{family}_SOURCE_TREE_MISMATCH")
        if git_value(source, "status", "--short"):
            raise RuntimeError(f"{family}_SOURCE_DIRTY")
    return row


def model_checkpoint(config: dict[str, Any], family: str, suite: str) -> str:
    spec = config["model_families"][family]
    if family == "M0_OPENVLA":
        return str(spec["paths"][suite])
    if family == "M1_OPENVLA_OFT":
        return str(Path(spec["checkpoint_root"]) / suite)
    return str(spec["checkpoint"])


def make_pairs(raw_chunk: Any, env_chunk: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    raw = np.asarray(raw_chunk, dtype=np.float32)
    env = np.asarray(env_chunk, dtype=np.float32)
    if raw.ndim != 2 or env.ndim != 2 or raw.shape != env.shape or raw.shape[1] != 7 or raw.shape[0] < 1:
        raise RuntimeError(f"ACTION_CHUNK_SHAPE_INVALID:{raw.shape}:{env.shape}")
    return [(raw[index].copy(), env[index].copy()) for index in range(raw.shape[0])]


def candidate_hash(row: dict[str, Any]) -> str:
    return canonical_hash({
        "step": row["step"],
        "phase": row["phase"],
        "anchor_class": row["anchor_class"],
        "object_identity": row.get("object_identity"),
        "object_gripper_contact": row.get("object_gripper_contact"),
        "object_support_contact": row.get("object_support_contact"),
    })


def run_cell(config: dict[str, Any], panel: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    z1 = load_module(Z1_RUNNER, "stage_z_z1_runtime_canary_for_z2")
    z1.require_single_visible_gpu(args.gpu_id)
    gpu = z1.gpu_snapshot(args.gpu_id)
    canary_ledger = load_json(args.canary_ledger) if args.population == "engineering_canary" else None
    row = verify_static_authority(config, panel, args.parent_key, args.suite, canary_ledger=canary_ledger)
    is_engineering_canary = args.population == "engineering_canary"
    checkpoint = model_checkpoint(config, args.model_family, args.suite)
    taxonomy = load_module(ROOT / "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py", "stage_z_physical_taxonomy")
    phases = load_module(ROOT / "src/gripper_attack/stage_v_m3_5_phase_classifier.py", "stage_z_phase_classifier")
    sys.path.insert(0, str(ROOT / "src"))
    action_semantics = importlib.import_module("stage_z_preparation.action_semantics")
    anchors = importlib.import_module("stage_z_preparation.anchors")
    z1.configure_libero(config)
    env, task_suite, task = z1.make_libero_env(config, args.suite, args.task_idx)
    counters = {"model_inference_calls": 0, "env_step_calls": 0, "anchor_telemetry_reads": 0, "physical_interventions": 0, "pgd_calls": 0, "attacked_env_steps": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0, "stage_z_scientific_parent_exposure": 0}
    checkpoint_manifest = None
    try:
        bddl = Path(z1.make_libero_env.__globals__["get_libero_path"]("bddl_files")) / task.problem_folder / task.bddl_file if "get_libero_path" in z1.make_libero_env.__globals__ else None
        if bddl is None or not bddl.is_file():
            from libero.libero import get_libero_path  # type: ignore
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        binding = taxonomy.bind_object_taxonomy(env, bddl)
        if binding.get("status") == "INELIGIBLE":
            return {
                "schema": R1_RECEIPT_SCHEMA,
                "status": "ABSTAIN_Z2_NO_LEGAL_ANCHOR",
                "model_family": args.model_family,
                "suite": args.suite,
                "population": args.population,
                "role": "PRIMARY",
                "canonical_parent_key": args.parent_key,
                "task_idx": args.task_idx,
                "state_id": args.state_id,
                "task_language": str(task.language),
                "checkpoint": checkpoint,
                "checkpoint_manifest": None,
                "normalization": None,
                "gpu": gpu,
                "panel_row": row,
                "object_binding": binding,
                "anchor_rule": {
                    "critical_phases": sorted(CRITICAL_PHASES),
                    "noncritical_phases": ["PRE_CONTACT"],
                    "phase_classifier": phases.specification(),
                    "critical_salt": args.critical_salt,
                    "noncritical_salt": args.noncritical_salt,
                    "student_or_detector_used": False,
                    "outcome_fields_used": [],
                },
                "clean_reference": {
                    "horizon": HORIZONS[args.suite],
                    "dummy_wait_steps": 0,
                    "decision_boundary_count": 0,
                    "decision_boundaries": [],
                    "candidate_count": 0,
                    "candidate_digest": canonical_hash([]),
                    "critical_candidates": [],
                    "noncritical_candidates": [],
                    "abstention_reason": binding.get("reason"),
                },
                "selected_anchors": {"critical": None, "noncritical": None},
                "runtime_counters": counters,
                "scientific_claim": "NONE_ENGINEERING_ONLY" if is_engineering_canary else "Z2_CLEAN_REFERENCE_AND_ANCHOR_ONLY",
                "protected_boundary": {"eval160": "UNREAD", "protected": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "attacked_env_steps": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0},
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        if binding.get("status") != "PASS":
            raise RuntimeError(f"OBJECT_TAXONOMY_BINDING_INVALID:{binding.get('reason')}")
        target_object = str(binding["target_object_ids"][0])
        if args.model_family == "M1_OPENVLA_OFT":
            checkpoint_manifest = z1.verify_m1_materialization(
                Path(args.m1_manifest),
                Path(checkpoint),
                args.suite,
                str(config["model_families"]["M1_OPENVLA_OFT"]["checkpoint_manifests_sha256"]),
            )
        if not Path(checkpoint).exists():
            raise RuntimeError(f"CHECKPOINT_NOT_MATERIALIZED:{checkpoint}")
        counters["stage_z_scientific_parent_exposure"] = 0 if is_engineering_canary else 1
        if args.model_family == "M0_OPENVLA":
            infer, _model, normalization = z1.load_openvla(checkpoint, oft=False, suite=args.suite, return_chunk=True)
        elif args.model_family == "M1_OPENVLA_OFT":
            infer, _model, normalization = z1.load_openvla(checkpoint, oft=True, suite=args.suite, return_chunk=True)
        else:
            infer, _model = z1.load_pi05(checkpoint, return_chunk=True)
            normalization = {"checkpoint_mutated": False}

        env.reset()
        initial_states = task_suite.get_task_init_states(args.task_idx)
        obs = env.set_init_state(initial_states[args.state_id])
        dummy = [0.0] * 6 + [-1.0]
        for _ in range(int(config["environment"]["dummy_wait_steps"])):
            obs = env.step(dummy)[0]
            counters["env_step_calls"] += 1

        horizon = HORIZONS[args.suite]
        queue: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[dict[str, Any]] = []
        boundary_states: dict[int, np.ndarray] = {}
        boundary_meta: dict[int, dict[str, Any]] = {}
        clean_actions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for step in range(horizon):
            fresh = not queue
            if fresh:
                chunk, meta = infer(obs, str(task.language))
                counters["model_inference_calls"] += 1
                raw_chunk = meta.get("raw_action_chunk")
                if raw_chunk is None:
                    raw_chunk = np.asarray(chunk, dtype=np.float32).tolist()
                queue = make_pairs(raw_chunk, chunk)
                if args.model_family == "M1_OPENVLA_OFT":
                    queue = queue[:OFT_QUEUE]
                elif args.model_family == "M2_PI05_LIBERO":
                    queue = queue[:PI05_REPLAN]
                boundary_states[step] = z1.snapshot_state(env)
                boundary_meta[step] = {
                    "step": step,
                    "model_boundary": "FRESH_PER_STEP" if args.model_family == "M0_OPENVLA" else ("FRESH_OFT_ACTION_QUEUE" if args.model_family == "M1_OPENVLA_OFT" else "FRESH_PI05_REPLAN"),
                    "chunk_length": len(queue),
                    "state_sha256": sha256_bytes(boundary_states[step].tobytes()),
                    "action_chunk_sha256": canonical_hash([{"raw": raw.tolist(), "env": action.tolist()} for raw, action in queue]),
                }
            raw_action, env_action = queue.pop(0)
            clean_actions[step] = (raw_action.copy(), env_action.copy())
            raw_values = raw_action.tolist()
            env_values = env_action.tolist()
            action_semantics_check = action_semantics.validate_action_pair(
                args.model_family,
                raw_values,
                env_values,
                raw_gripper=float(raw_action[-1]),
                final_gripper=float(env_action[-1]),
            )
            if args.model_family == "M2_PI05_LIBERO" and not action_semantics_check["accepted"]:
                raise RuntimeError(f"M2_ACTION_SEMANTICS_INVALID:{action_semantics_check['reason']}")
            telemetry = taxonomy.telemetry_from_env(env, binding, target_object_id=target_object)
            counters["anchor_telemetry_reads"] += 1
            rows.append({
                "step": step,
                "clean_record_valid": True,
                "clean_terminal": False,
                "remaining_horizon": horizon - step,
                "object_identity": telemetry.get("object_identity"),
                "object_position": telemetry.get("object_position"),
                "eef_position": telemetry.get("eef_position"),
                "object_eef_distance_m": telemetry.get("object_eef_distance_m"),
                "object_gripper_contact": telemetry.get("object_gripper_contact"),
                "object_support_contact": telemetry.get("object_support_contact"),
                "contact_telemetry_valid": telemetry.get("contact_telemetry_valid"),
                "raw_gripper": float(raw_action[-1]),
                "env_gripper": float(env_action[-1]),
                "raw_action_7d": raw_values,
                "env_action_7d": env_values,
                "action_semantics": action_semantics_check,
                "model_boundary": fresh,
            })
            obs = env.step(env_action.tolist())[0]
            counters["env_step_calls"] += 1

        phase_rows, action_semantics_diagnostics = action_semantics.classify_trajectory_with_action_semantics(
            rows, args.model_family, phases
        )
        candidates: list[dict[str, Any]] = []
        anchor_objects: list[Any] = []
        for clean_row, phase in zip(rows, phase_rows):
            if not clean_row["model_boundary"] or not phase.get("phase_eligible"):
                continue
            label = str(phase["clean_only_phase_label"])
            if label in CRITICAL_PHASES:
                anchor_class = "CRITICAL"
            elif label == "PRE_CONTACT":
                anchor_class = "NONCRITICAL"
            else:
                continue
            step = int(clean_row["step"])
            if clean_row["remaining_horizon"] < T10 + H_PHYS:
                continue
            candidate_record = {
                "parent_key": args.parent_key,
                "model_id": args.model_family,
                "step": step,
                "anchor_class": anchor_class,
                "phase": label,
                "candidate_digest": candidate_hash({**clean_row, "phase": label, "anchor_class": anchor_class}),
                "state_sha256": boundary_meta[step]["state_sha256"],
            }
            candidates.append(candidate_record)
            anchor_objects.append(anchors.AnchorCandidate(
                parent_key=args.parent_key,
                model_id=args.model_family,
                step=step,
                anchor_class=anchor_class,
                metadata={"phase": label, "candidate_digest": candidate_record["candidate_digest"]},
            ))
        critical = anchors.select_anchor(anchor_objects, salt=args.critical_salt, model_id=args.model_family, parent_key=args.parent_key, anchor_class="CRITICAL")
        noncritical = anchors.select_anchor(anchor_objects, salt=args.noncritical_salt, model_id=args.model_family, parent_key=args.parent_key, anchor_class="NONCRITICAL")

        def materialize(selection: Any) -> dict[str, Any] | None:
            if selection.selected is None:
                return None
            step = int(selection.selected.step)
            action_rows = []
            for index in range(step, min(step + T10 + H_PHYS, horizon)):
                raw, action = clean_actions[index]
                action_rows.append({"step": index, "raw_action": raw.tolist(), "env_action": action.tolist()})
            return {
                "status": selection.status,
                "anchor_class": selection.anchor_class,
                "step": step,
                "rank_digest": selection.rank_digest,
                "phase": str(selection.selected.metadata["phase"]),
                "state": boundary_states[step].tolist(),
                "state_sha256": boundary_meta[step]["state_sha256"],
                "action_rows": action_rows,
            }

        critical_record = materialize(critical)
        noncritical_record = materialize(noncritical)
        status = "PASS_Z2_CLEAN_REFERENCE_WITH_BOTH_ANCHORS" if critical_record and noncritical_record else "ABSTAIN_Z2_NO_LEGAL_ANCHOR"
        return {
            "schema": R1_RECEIPT_SCHEMA,
            "status": status,
            "model_family": args.model_family,
            "suite": args.suite,
            "population": args.population,
            "role": "PRIMARY",
            "canonical_parent_key": args.parent_key,
            "task_idx": args.task_idx,
            "state_id": args.state_id,
            "task_language": str(task.language),
            "checkpoint": checkpoint,
            "checkpoint_manifest": checkpoint_manifest,
            "normalization": normalization,
            "gpu": gpu,
            "panel_row": row,
            "object_binding": {key: value for key, value in binding.items() if key not in {"body_names"}},
            "anchor_rule": {
                "critical_phases": sorted(CRITICAL_PHASES),
                "noncritical_phases": ["PRE_CONTACT"],
                "phase_classifier": phases.specification(),
                "action_semantics_adapter": {
                    "module": "stage_z_preparation.action_semantics",
                    "diagnostics": action_semantics_diagnostics,
                },
                "critical_salt": args.critical_salt,
                "noncritical_salt": args.noncritical_salt,
                "student_or_detector_used": False,
                "outcome_fields_used": [],
            },
            "clean_reference": {
                "horizon": horizon,
                "dummy_wait_steps": int(config["environment"]["dummy_wait_steps"]),
                "decision_boundary_count": len(boundary_meta),
                "decision_boundaries": list(boundary_meta.values()),
                "candidate_count": len(candidates),
                "candidate_digest": canonical_hash(candidates),
                "critical_candidates": [item for item in candidates if item["anchor_class"] == "CRITICAL"],
                "noncritical_candidates": [item for item in candidates if item["anchor_class"] == "NONCRITICAL"],
                "action_semantics": action_semantics_diagnostics,
            },
            "selected_anchors": {"critical": critical_record, "noncritical": noncritical_record},
            "runtime_counters": counters,
            "scientific_claim": "NONE_ENGINEERING_ONLY" if is_engineering_canary else "Z2_CLEAN_REFERENCE_AND_ANCHOR_ONLY",
            "protected_boundary": {"eval160": "UNREAD", "protected": "UNREAD", "pgd_calls": 0, "physical_interventions": 0, "attacked_env_steps": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "protected_reads": 0},
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--task-idx", type=int, required=True)
    parser.add_argument("--state-id", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--critical-salt", default="STAGE_Z_Z2_CRITICAL_ANCHOR_V1_20260823")
    parser.add_argument("--noncritical-salt", default="STAGE_Z_Z2_NONCRITICAL_CONTROL_V1_20260823")
    parser.add_argument("--population", choices=("scientific", "engineering_canary"), default="scientific")
    parser.add_argument("--canary-ledger", type=Path)
    args = parser.parse_args()
    if args.population == "engineering_canary" and args.canary_ledger is None:
        parser.error("--canary-ledger is required for engineering_canary")
    try:
        result = run_cell(load_json(args.config), load_json(args.panel), args)
    except Exception as exc:
        result = {
            "schema": R1_RECEIPT_SCHEMA,
            "status": "ENGINEERING_INVALID_Z2_CLEAN_REFERENCE",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "model_family": args.model_family,
            "suite": args.suite,
            "population": args.population,
            "canonical_parent_key": args.parent_key,
            "runtime_counters": {"model_inference_calls": 0, "env_step_calls": 0, "anchor_telemetry_reads": 0, "physical_interventions": 0, "pgd_calls": 0, "attacked_env_steps": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "eval160_reads": 0, "protected_reads": 0, "stage_z_scientific_parent_exposure": 0},
            "scientific_claim": "NONE_DUE_TO_ENGINEERING_INVALID",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "model_family": args.model_family, "suite": args.suite, "output": str(args.output)}))


if __name__ == "__main__":
    main()
