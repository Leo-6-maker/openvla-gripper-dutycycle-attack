#!/usr/bin/env python3
"""Prepare and run the exact 40x24 clean plan/snapshot gate.

This wrapper reuses Gate A for clean rollout, probe selection, snapshots, and
non-intervention restore canaries.  It never imports or calls a treatment arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot  # noqa: E402


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
CORRIDOR_COMMIT = "3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2"
CORRIDOR_TREE = "2492a075e782a112d1e857248956b2647e751039"
CORRIDOR_RUNNER_SHA = "26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279"
MIN_FREE_MEMORY_MIB = 20480
PLAN_SCHEMA = "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1"
MODEL_RELATIVE_PATHS = {
    "libero_10": Path("libero-10/openvla-7b-finetuned-libero-10"),
    "libero_goal": Path("libero-goal"),
    "libero_object": Path("openvla-7b-finetuned-libero-object"),
    "libero_spatial": Path("libero-spatial/spatial_c8f03f4_20260620"),
}
GATE_A_RUNNER = REPO_ROOT / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py"


class PlanGateError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PlanGateError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _sha_file(path: Path) -> str:
    if not path.is_file():
        raise PlanGateError(f"FILE_MISSING:{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _slug(parent_key: str) -> str:
    return parent_key.replace("/", "__")


def _git_snapshot() -> tuple[str, str]:
    def run(*parts: str) -> str:
        return subprocess.check_output(["git", *parts], cwd=REPO_ROOT, text=True).strip()

    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _model_path(model_root: Path, suite: str) -> Path:
    try:
        path = (model_root / MODEL_RELATIVE_PATHS[suite]).resolve()
    except KeyError as exc:
        raise PlanGateError(f"SUITE_MODEL_MAPPING_MISSING:{suite}") from exc
    if not path.is_dir():
        raise PlanGateError(f"MODEL_PATH_MISSING:{path}")
    return path


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != "STAGE_V_M4_EXACT_40X24_PLAN_ONLY_PROTOCOL_V1" or protocol.get("status") != "FROZEN":
        raise PlanGateError("PLAN_PROTOCOL_NOT_FROZEN")
    matrix = protocol.get("matrix", {})
    expected = {"parents": 40, "probes_per_parent": 24, "probe_count_total": 960, "planned_branch_authorities_total": 3840, "h_phys": 10}
    for key, value in expected.items():
        if matrix.get(key) != value:
            raise PlanGateError(f"PLAN_MATRIX_INVALID:{key}")
    operation = protocol.get("operation", {})
    forbidden = {"intervention_executed": False, "outcomes_read": False, "v_phys_generated": False, "teacher_predictions_read": False, "student_predictions_read": False, "attack_rollouts": False, "vis_rollouts": False, "eval160_reads": False}
    for key, value in forbidden.items():
        if operation.get(key) is not value:
            raise PlanGateError(f"PLAN_OPERATION_BOUNDARY_INVALID:{key}")
    if operation.get("fresh_render_primary_consumption") != "HARD_STOP":
        raise PlanGateError("PLAN_FRESH_RENDER_BOUNDARY_INVALID")
    if protocol.get("selection", {}).get("selection_version") != "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1":
        raise PlanGateError("PLAN_SELECTION_VERSION_INVALID")
    if protocol.get("protected_counters") != COUNTERS:
        raise PlanGateError("PLAN_PROTECTED_COUNTERS_NONZERO")
    source = protocol.get("source_binding", {})
    if source.get("corridor_science_commit") != CORRIDOR_COMMIT or source.get("corridor_science_tree") != CORRIDOR_TREE or source.get("corridor_runner_sha256") != CORRIDOR_RUNNER_SHA:
        raise PlanGateError("PLAN_CORRIDOR_SOURCE_BINDING_INVALID")


def _validate_population(final_manifest: Mapping[str, Any], split: Mapping[str, Any], attempt_registry: Mapping[str, Any], *, final_sha: str, split_sha: str) -> list[dict[str, Any]]:
    if final_manifest.get("schema") != "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" or final_manifest.get("status") != "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" or final_manifest.get("parent_count") != 40:
        raise PlanGateError("FINAL40_NOT_FROZEN")
    if final_manifest.get("formal_m4_authorized") is not False or final_manifest.get("outcomes_read") is not False:
        raise PlanGateError("FINAL40_DOWNSTREAM_BOUNDARY_INVALID")
    parents = final_manifest.get("parents")
    if not isinstance(parents, list) or len(parents) != 40:
        raise PlanGateError("FINAL40_PARENT_COUNT_INVALID")
    parent_keys = [str(row.get("canonical_parent_key")) for row in parents if isinstance(row, Mapping)]
    if len(parent_keys) != 40 or len(set(parent_keys)) != 40:
        raise PlanGateError("FINAL40_IDENTITY_INVALID")
    suite_counts = Counter(str(row.get("suite")) for row in parents if isinstance(row, Mapping))
    if suite_counts != Counter({"libero_10": 10, "libero_goal": 10, "libero_object": 10, "libero_spatial": 10}):
        raise PlanGateError(f"FINAL40_SUITE_COUNTS_INVALID:{dict(suite_counts)}")
    if split.get("schema") != "STAGE_V_M4_FINAL_PARENT_SPLIT_V2" or split.get("status") != "FROZEN" or split.get("final_manifest_sha256") != final_sha:
        raise PlanGateError("FINAL_SPLIT_BINDING_INVALID")
    split_rows = split.get("parents")
    if not isinstance(split_rows, list) or len(split_rows) != 40:
        raise PlanGateError("FINAL_SPLIT_PARENT_COUNT_INVALID")
    split_keys = [str(row.get("canonical_parent_key")) for row in split_rows if isinstance(row, Mapping)]
    if set(split_keys) != set(parent_keys) or len(set(split_keys)) != 40:
        raise PlanGateError("FINAL_SPLIT_IDENTITY_INVALID")
    if split.get("counts") != {"TRAIN": 24, "VAL": 8, "TEST": 8}:
        raise PlanGateError("FINAL_SPLIT_COUNTS_INVALID")
    if final_manifest.get("split_counts") != {"TRAIN": 24, "VAL": 8, "TEST": 8}:
        raise PlanGateError("FINAL40_SPLIT_COUNTS_INVALID")
    manifest_split_path = Path(str(final_manifest.get("final_split_path", "")))
    if manifest_split_path.is_file() and _sha_file(manifest_split_path) != split_sha:
        raise PlanGateError("FINAL_MANIFEST_SPLIT_PATH_HASH_INVALID")
    if attempt_registry.get("schema") != "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1" or attempt_registry.get("status") != "FROZEN_EXACT55_CORRIDOR_ATTEMPT_FIREWALL" or attempt_registry.get("attempted_identity_count") != 55 or attempt_registry.get("unique_identity_count") != 55 or attempt_registry.get("duplicate_count") != 0 or attempt_registry.get("outcomes_read") is not False or attempt_registry.get("protected_counters") != COUNTERS:
        raise PlanGateError("EXACT55_FIREWALL_INVALID")
    attempted = {str(row.get("canonical_parent_key")) for row in attempt_registry.get("attempted_identities", []) if isinstance(row, Mapping)}
    if len(attempted) != 55 or not set(parent_keys).issubset(attempted):
        raise PlanGateError("EXACT55_FINAL40_UNION_INVALID")
    source = final_manifest.get("source_binding", {})
    if source.get("science_commit") != CORRIDOR_COMMIT or source.get("science_tree") != CORRIDOR_TREE or source.get("runner_sha256") != CORRIDOR_RUNNER_SHA:
        raise PlanGateError("FINAL40_CORRIDOR_SOURCE_BINDING_INVALID")
    return [dict(row) for row in parents]


def _validate_authorization(root: Path, protocol: Mapping[str, Any], authorization: Mapping[str, Any], final_manifest: Mapping[str, Any], split: Mapping[str, Any], attempt_registry: Mapping[str, Any]) -> tuple[str, str]:
    source_commit, source_tree = _git_snapshot()
    required = {
        "status": "PASS_PRELAUNCH",
        "scope": "EXACT_40X24_PLAN_AND_SNAPSHOT_ONLY",
        "protocol_sha256": _sha_file(root / "PLAN_PROTOCOL.json"),
        "final_manifest_sha256": _sha_file(root / "inputs/FINAL_PARENT_MANIFEST.json"),
        "final_split_sha256": _sha_file(root / "inputs/FINAL_PARENT_SPLIT.json"),
        "attempt_registry_sha256": _sha_file(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json"),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "gate_a_runner_sha256": _sha_file(GATE_A_RUNNER),
        "protected_counters": COUNTERS,
        "intervention_executed": False,
        "outcomes_read": False,
    }
    for key, value in required.items():
        if authorization.get(key) != value:
            raise PlanGateError(f"PLAN_AUTHORIZATION_BINDING_INVALID:{key}")
    runtime = authorization.get("runtime", {})
    for key in ("official_snapshot_root", "upstream_root", "model_root", "python_executable"):
        if not runtime.get(key):
            raise PlanGateError(f"PLAN_RUNTIME_BINDING_MISSING:{key}")
    if authorization.get("matrix") != {"parents": 40, "probes_per_parent": 24, "probe_count_total": 960, "planned_branch_authorities_total": 3840}:
        raise PlanGateError("PLAN_AUTHORIZATION_MATRIX_INVALID")
    _validate_protocol(protocol)
    _validate_population(final_manifest, split, attempt_registry, final_sha=_sha_file(root / "inputs/FINAL_PARENT_MANIFEST.json"), split_sha=_sha_file(root / "inputs/FINAL_PARENT_SPLIT.json"))
    return source_commit, source_tree


def _gpu_snapshot(requested: list[int]) -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append({"gpu": int(parts[0]), "memory_free_mib": int(float(parts[1])), "memory_used_mib": int(float(parts[2])), "utilization_gpu_percent": int(float(parts[3]))})
        except ValueError:
            continue
    eligible = [row["gpu"] for row in rows if row["gpu"] in requested and row["memory_free_mib"] > MIN_FREE_MEMORY_MIB]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    return {"status": "PASS" if result.returncode == 0 and eligible else "HOLD", "requested_gpu_pool": requested, "eligible_gpu_pool": eligible, "minimum_free_memory_mib": MIN_FREE_MEMORY_MIB, "strict_rule": "memory_free_mib > 20480", "gpu_rows": rows, "compute_apps_raw": apps.stdout.splitlines(), "foreign_workload_allowed": True, "foreign_process_interference": False, "query_returncode": result.returncode}


def _prepare(args: argparse.Namespace) -> int:
    protocol = _load(args.protocol.resolve())
    final_manifest = _load(args.final_manifest.resolve())
    split = _load(args.final_split.resolve())
    attempt_registry = _load(args.attempt_registry.resolve())
    _validate_protocol(protocol)
    final_sha = _sha_file(args.final_manifest.resolve())
    split_sha = _sha_file(args.final_split.resolve())
    _validate_population(final_manifest, split, attempt_registry, final_sha=final_sha, split_sha=split_sha)
    source_commit, source_tree = _git_snapshot()
    root = args.output_root.resolve()
    if root.exists():
        raise PlanGateError(f"REFUSE_OVERWRITE:{root}")
    root.mkdir(parents=True)
    _copy(args.protocol.resolve(), root / "PLAN_PROTOCOL.json")
    _copy(args.final_manifest.resolve(), root / "inputs/FINAL_PARENT_MANIFEST.json")
    _copy(args.final_split.resolve(), root / "inputs/FINAL_PARENT_SPLIT.json")
    _copy(args.attempt_registry.resolve(), root / "inputs/EXACT55_ATTEMPT_REGISTRY.json")
    selected = [{"canonical_parent_key": row["canonical_parent_key"], "suite": row["suite"], "split": row["split"]} for row in final_manifest["parents"]]
    _write(root / "SELECTION_MANIFEST.json", {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_SELECTION_MANIFEST_V1", "status": "FROZEN_BEFORE_PLAN_RUNTIME", "selection_reads": {"branch_results_read": False, "counterfactual_outcomes_read": False, "v_phys_read": False}, "selected_parents": selected, "parent_count": 40, "final_manifest_sha256": final_sha, "final_split_sha256": split_sha, "protected_counters": COUNTERS})
    gate_a_protocol = {"schema": "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A", "version": "V1.4.2-GATE-A", "status": "FROZEN_RUNTIME_AUTHORIZED", "runtime_authorized": True, "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree}, "probe_plan_selection_version": "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1", "operation": {"fresh_render_primary_consumption": "HARD_STOP", "fresh_render_equality_gate_used": False, "intervention_executed": False}, "protected_counters": COUNTERS}
    _write(root / "GATE_A_PROTOCOL.json", gate_a_protocol)
    runtime = {"official_snapshot_root": str(args.official_snapshot_root.resolve()), "upstream_root": str(args.upstream_root.resolve()), "model_root": str(args.model_root.resolve()), "python_executable": str(Path(sys.executable).resolve()), "model_paths": {suite: str(_model_path(args.model_root.resolve(), suite)) for suite in MODEL_RELATIVE_PATHS}}
    authorization = {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_AUTHORIZATION_V1", "status": "PASS_PRELAUNCH", "scope": "EXACT_40X24_PLAN_AND_SNAPSHOT_ONLY", "protocol_sha256": _sha_file(root / "PLAN_PROTOCOL.json"), "final_manifest_sha256": final_sha, "final_split_sha256": split_sha, "attempt_registry_sha256": _sha_file(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json"), "source_commit": source_commit, "source_tree": source_tree, "gate_a_runner_sha256": _sha_file(GATE_A_RUNNER), "runtime": runtime, "matrix": {"parents": 40, "probes_per_parent": 24, "probe_count_total": 960, "planned_branch_authorities_total": 3840}, "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS, "failure_action": "HOLD_SEALED_NO_RESERVE_SUBSTITUTION_NO_RERUN"}
    _write(root / "PLAN_AUTHORIZATION.json", authorization)
    _write(root / "PREPARE_RECEIPT.json", {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_PREPARE_RECEIPT_V1", "status": "PASS_PRELAUNCH_READY", "root": str(root), "source_commit": source_commit, "source_tree": source_tree, "protocol_sha256": _sha_file(root / "PLAN_PROTOCOL.json"), "final_manifest_sha256": final_sha, "final_split_sha256": split_sha, "attempt_registry_sha256": _sha_file(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json"), "protected_counters": COUNTERS})
    print(json.dumps({"status": "PASS_PRELAUNCH_READY", "root": str(root), "source_commit": source_commit, "source_tree": source_tree}, sort_keys=True))
    return 0


def _run_one(root: Path, parent: Mapping[str, Any], gpu: int, runtime: Mapping[str, Any], source_commit: str, source_tree: str) -> dict[str, Any]:
    key = str(parent["canonical_parent_key"])
    slug = _slug(key)
    output_dir = root / "parents" / slug
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [str(Path(runtime["python_executable"])), str(GATE_A_RUNNER), "--protocol", str(root / "GATE_A_PROTOCOL.json"), "--selection-manifest", str(root / "SELECTION_MANIFEST.json"), "--parent-key", key, "--output-dir", str(output_dir), "--official-snapshot-root", str(runtime["official_snapshot_root"]), "--upstream-root", str(runtime["upstream_root"]), "--model-path", str(runtime["model_paths"][str(parent["suite"])]), "--gpu", str(gpu), "--source-commit", source_commit, "--source-tree", source_tree, "--enable-runtime"]
    command_path = log_dir / f"{slug}.COMMAND.json"
    log_path = log_dir / f"{slug}.log"
    _write(command_path, {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_COMMAND_V1", "canonical_parent_key": key, "gpu": gpu, "command": command, "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS})
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False, text=True)
    return {"canonical_parent_key": key, "suite": parent["suite"], "split": parent["split"], "gpu": gpu, "return_code": process.returncode, "output_dir": str(output_dir.relative_to(root).as_posix()), "log": str(log_path.relative_to(root).as_posix()), "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS}


def _build_manifest(root: Path, parents: list[dict[str, Any]], source_commit: str, source_tree: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    parent_entries: list[dict[str, Any]] = []
    probe_authorities: list[dict[str, Any]] = []
    branch_authorities: list[dict[str, Any]] = []
    errors: list[str] = []
    for parent in parents:
        key = str(parent["canonical_parent_key"])
        parent_root = root / "parents" / _slug(key)
        entry = {"canonical_parent_key": key, "suite": parent["suite"], "split": parent["split"], "output_dir": str(parent_root.relative_to(root).as_posix()), "status": "HOLD"}
        try:
            receipt = _load(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
            audit = _load(parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json")
            clean = _load(parent_root / "CLEAN_TRAJECTORY_V1_4.json")
            plan = _load(parent_root / "PROBE_PLAN_V1_4.json")
            taxonomy_path = parent_root / "TAXONOMY_BINDING.json"
            taxonomy = _load(taxonomy_path)
            if receipt.get("status") != "PASS" or audit.get("status") != "PASS" or receipt.get("snapshot_count") != 24 or receipt.get("intervention_executed") is not False or receipt.get("outcomes_read") is not False or receipt.get("protected_counters") != COUNTERS or taxonomy.get("status") != "PASS":
                raise PlanGateError("PARENT_GATE_A_NOT_PASS")
            rows = clean.get("rows")
            probes = plan.get("probe_steps")
            if not isinstance(rows, list) or not isinstance(probes, list) or len(probes) != 24 or plan.get("outcomes_read") is not False or plan.get("selection_version") != "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1":
                raise PlanGateError("PARENT_PLAN_INVALID")
            action_sequence = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in rows]
            action_sha = canonical_sha256(canonical_value(action_sequence))
            entry.update({"status": "PASS", "receipt_sha256": _sha_file(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json"), "audit_sha256": _sha_file(parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"), "clean_trajectory_path": str((parent_root / "CLEAN_TRAJECTORY_V1_4.json").relative_to(root).as_posix()), "clean_trajectory_sha256": _sha_file(parent_root / "CLEAN_TRAJECTORY_V1_4.json"), "clean_reference_action_sequence_sha256": action_sha, "probe_plan_path": str((parent_root / "PROBE_PLAN_V1_4.json").relative_to(root).as_posix()), "probe_plan_sha256": _sha_file(parent_root / "PROBE_PLAN_V1_4.json"), "taxonomy_binding_path": str(taxonomy_path.relative_to(root).as_posix()), "taxonomy_binding_sha256": _sha_file(taxonomy_path), "taxonomy_status": taxonomy.get("status"), "model_path": runtime["model_paths"][str(parent["suite"])], "probe_count": 24, "intervention_executed": False, "outcomes_read": False})
            for probe, snapshot_row in zip(probes, receipt.get("snapshots", [])):
                probe_id = str(probe["probe_id"])
                snapshot_path = parent_root / str(snapshot_row["path"])
                loaded = load_snapshot(snapshot_path, materialize_torch=False)
                payload = loaded["payload"]
                snapshot_manifest = loaded["manifest"]
                if snapshot_manifest.get("binding", {}).get("parent_key") != key or snapshot_manifest.get("binding", {}).get("source_commit") != source_commit or snapshot_manifest.get("binding", {}).get("source_tree") != source_tree:
                    raise PlanGateError(f"SNAPSHOT_BINDING_INVALID:{key}:{probe_id}")
                window = payload.get("clean_reference_action_window")
                window_sha = canonical_sha256(canonical_value(window))
                authority = {"canonical_parent_key": key, "suite": parent["suite"], "task_index": parent["task_index"], "state_index": parent["state_index"], "split": parent["split"], "probe_id": probe_id, "probe_step": int(probe["step"]), "sim_state_sha256": probe.get("state_sha256"), "raw_observation_sha256": payload.get("raw_observation_sha256"), "policy_rgb_224_sha256": probe.get("policy_rgb_224_sha256"), "policy_input_sha256": probe.get("policy_input_sha256"), "snapshot_path": str(snapshot_path.relative_to(root).as_posix()), "snapshot_manifest_sha256": snapshot_row.get("manifest_sha256"), "clean_reference_action_sequence_sha256": action_sha, "clean_reference_action_window_sha256": window_sha, "object_taxonomy": {"status": taxonomy.get("status"), "object_identity": probe.get("object_identity"), "target_object_ids": taxonomy.get("target_object_ids", []), "binding_sha256": _sha_file(taxonomy_path)}, "H_phys": 10, "source_binding": {"downstream_source_commit": source_commit, "downstream_source_tree": source_tree, "corridor_science_commit": CORRIDOR_COMMIT, "corridor_science_tree": CORRIDOR_TREE, "corridor_runner_sha256": CORRIDOR_RUNNER_SHA}, "intervention_executed": False, "outcomes_read": False}
                probe_authorities.append(authority)
                for arm in ("CONTROL", "T3", "T5", "T10"):
                    branch_authorities.append({"canonical_parent_key": key, "probe_id": probe_id, "probe_step": int(probe["step"]), "arm": arm, "branch_id": "m4-v2-plan-" + hashlib.sha256(f"M4_V2_PLAN::{key}::{probe_id}::R0::{arm}".encode()).hexdigest(), "snapshot_manifest_sha256": snapshot_row.get("manifest_sha256"), "execution_status": "PLANNED_NOT_EXECUTED", "outcomes_read": False, "protected_counters": COUNTERS})
        except Exception as exc:
            errors.append(f"{key}:{type(exc).__name__}:{exc}")
            entry["error"] = str(exc)
        parent_entries.append(entry)
    manifest = {"schema": PLAN_SCHEMA, "version": "EXACT-40X24-PLAN-ONLY-V1", "status": "FROZEN_PLAN_ONLY_PENDING_INDEPENDENT_AUDIT", "sealed": False, "final40_manifest_sha256": _sha_file(root / "inputs/FINAL_PARENT_MANIFEST.json"), "final_split_sha256": _sha_file(root / "inputs/FINAL_PARENT_SPLIT.json"), "exact55_attempt_registry_sha256": _sha_file(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json"), "downstream_source": {"commit": source_commit, "tree": source_tree, "gate_a_runner_sha256": _sha_file(GATE_A_RUNNER)}, "corridor_source": {"commit": CORRIDOR_COMMIT, "tree": CORRIDOR_TREE, "runner_sha256": CORRIDOR_RUNNER_SHA}, "parent_count": 40, "probe_count_per_parent": 24, "probe_count_total": len(probe_authorities), "planned_branch_authority_count": len(branch_authorities), "planned_branch_authority_expected": 3840, "parents": parent_entries, "probe_authorities": probe_authorities, "branch_authorities": branch_authorities, "selection_outcomes_read": False, "intervention_executed": False, "v_phys_generated": False, "teacher_predictions_read": False, "student_predictions_read": False, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0, "protected_counters": COUNTERS, "errors": sorted(errors), "failure_action": "HOLD_SEALED_NO_RESERVE_SUBSTITUTION_NO_RERUN"}
    return manifest


def _seal(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.sha256"}:
            lines.append(f"{_sha_file(path)}  {path.relative_to(root).as_posix()}\n")
    sums = root / "SHA256SUMS"
    sums.write_text("".join(lines), encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{_sha_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if not root.is_dir():
        raise PlanGateError(f"PREPARE_ROOT_MISSING:{root}")
    protocol = _load(root / "PLAN_PROTOCOL.json")
    authorization = _load(root / "PLAN_AUTHORIZATION.json")
    final_manifest = _load(root / "inputs/FINAL_PARENT_MANIFEST.json")
    split = _load(root / "inputs/FINAL_PARENT_SPLIT.json")
    attempt_registry = _load(root / "inputs/EXACT55_ATTEMPT_REGISTRY.json")
    source_commit, source_tree = _validate_authorization(root, protocol, authorization, final_manifest, split, attempt_registry)
    runtime = authorization["runtime"]
    requested = sorted(set(int(item) for item in str(args.gpus).split(",") if item.strip()))
    resource = _gpu_snapshot(requested)
    _write(root / "RESOURCE_PRELAUNCH.json", resource)
    eligible = resource["eligible_gpu_pool"]
    if not eligible:
        raise PlanGateError("NO_GPU_FREE_MEMORY_ABOVE_20G")
    parents = [dict(row) for row in final_manifest["parents"]]
    max_workers = min(int(args.max_workers or len(eligible)), len(eligible))
    assignments = [eligible[index % len(eligible)] for index in range(len(parents))]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_one, root, parent, assignments[index], runtime, source_commit, source_tree) for index, parent in enumerate(parents)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["canonical_parent_key"]))
    _write(root / "PARENT_RUN_REGISTRY.json", {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_PARENT_RUN_REGISTRY_V1", "status": "COMPLETE" if all(row["return_code"] == 0 for row in results) else "HOLD", "parent_count": len(results), "results": results, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS})
    manifest = _build_manifest(root, parents, source_commit, source_tree, runtime)
    _write(root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json", manifest)
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m4_exact_40x24_plan_only.py"
    first = subprocess.run([str(Path(runtime["python_executable"])), str(auditor), "--root", str(root)], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    audit = _load(root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json") if (root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json").is_file() else {"status": "FAIL", "errors": [first.stderr[-1000:]]}
    if audit.get("status") == "PASS":
        manifest["status"] = "PASS_EXACT_40X24_PLAN_ONLY"
        manifest["sealed"] = True
        manifest["independent_audit_sha256"] = _sha_file(root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json")
        _write(root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json", manifest)
        second = subprocess.run([str(Path(runtime["python_executable"])), str(auditor), "--root", str(root)], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        audit = _load(root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json")
        audit_status = audit.get("status") if second.returncode == 0 else "FAIL"
    else:
        manifest["status"] = "HOLD_SEALED_EXACT_40X24_PLAN_ONLY"
        manifest["sealed"] = True
        _write(root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json", manifest)
        audit_status = "FAIL"
    _write(root / "PLAN_RESULT.json", {"schema": "STAGE_V_M4_EXACT_40X24_PLAN_RESULT_V1", "status": audit_status, "manifest_status": manifest["status"], "parent_count": manifest["parent_count"], "probe_count_total": manifest["probe_count_total"], "planned_branch_authority_count": manifest["planned_branch_authority_count"], "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS, "resource_prelaunch": resource, "audit_sha256": _sha_file(root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json") if (root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json").is_file() else None})
    _seal(root)
    print(json.dumps({"status": audit_status, "root": str(root), "parents": manifest["parent_count"], "probes": manifest["probe_count_total"], "planned_branches": manifest["planned_branch_authority_count"]}, sort_keys=True))
    return 0 if audit_status == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--final-manifest", type=Path, required=True)
    prep.add_argument("--final-split", type=Path, required=True)
    prep.add_argument("--attempt-registry", type=Path, required=True)
    prep.add_argument("--official-snapshot-root", type=Path, required=True)
    prep.add_argument("--upstream-root", type=Path, required=True)
    prep.add_argument("--model-root", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    run.add_argument("--max-workers", type=int)
    args = parser.parse_args(argv)
    try:
        return _prepare(args) if args.mode == "prepare" else _run(args)
    except (OSError, KeyError, ValueError, PlanGateError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "HOLD_SEALED", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
