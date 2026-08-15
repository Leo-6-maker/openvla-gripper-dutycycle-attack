#!/usr/bin/env python3
"""Diagnostic-only FIT670/R3 telemetry reconstruction from frozen M4 clean actions.

This runner never loads the policy, reads reward/done/info, runs an
intervention, or materializes a V_phys label.  It replays only the already
sealed M4 clean action sequence and validates the resulting current-step
telemetry against the frozen FIT670/R3 contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gripper_attack.stage_v_canonical_execution_core import (  # noqa: E402
    canonical_initial_state_sha256,
    canonical_sha256,
    canonical_value,
)
from gripper_attack.v5_r3_teacher import (  # noqa: E402
    canonicalize_fit670_episode,
    validate_contact_row,
)
from scripts.detector_v5.run_stage_v_canonical_clean import _load_external_modules  # noqa: E402
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import _new_env  # noqa: E402
from gripper_attack.stage_v_m3_5_physical_taxonomy import _body_is_gripper  # noqa: E402
from n5.phase2_labels.run_r5f_full40_materialize import (  # noqa: E402
    collect_entity,
    load_resolutions,
    sha256_file,
    verify_entity_identity,
)


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _finite_vector(value: Any, size: int) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != size or not all(math.isfinite(item) for item in values):
        raise ValueError("NONFINITE_VECTOR")
    return values


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _inventory(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        rows.append({"index": int(fields[0]), "uuid": fields[1], "free_mib": int(fields[2]), "used_mib": int(fields[3]), "utilization_gpu": int(fields[4])})
    row = next((item for item in rows if item["index"] == int(gpu)), None)
    if row is None or int(row["free_mib"]) <= 20480:
        raise RuntimeError("GPU_RESOURCE_CONTRACT_NOT_SATISFIED")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    return {"requested_gpu": int(gpu), "selected_gpu": row, "compute_apps": [line.strip() for line in apps.stdout.splitlines() if line.strip()], "query_returncode": result.returncode}


def _contact_pairs(env: Any, object_body_ids: set[int]) -> list[dict[str, Any]]:
    model = env.sim.model
    data = env.sim.data
    body_names = [str(model.body(index).name or "") for index in range(int(model.nbody))]
    geom_names = [str(model.geom(index).name or "") for index in range(int(model.ngeom))]
    efc_force = getattr(data, "efc_force", None)
    pairs = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        address = int(contact.efc_address)
        if address < 0 or efc_force is None or address >= len(efc_force):
            raise ValueError("CONTACT_FORCE_BINDING_UNAVAILABLE")
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        other = body2 if body1 in object_body_ids else body1
        object_gripper = bool((body1 in object_body_ids or body2 in object_body_ids) and 0 <= other < len(body_names) and _body_is_gripper(body_names[other]))
        pairs.append({
            "body1": body_names[body1], "body1_id": body1,
            "body2": body_names[body2], "body2_id": body2,
            "dist": float(contact.dist), "efc_address": address,
            "geom1": geom_names[int(contact.geom1)], "geom1_id": int(contact.geom1),
            "geom2": geom_names[int(contact.geom2)], "geom2_id": int(contact.geom2),
            "is_object_gripper_contact": object_gripper,
            "normal": _finite_vector(contact.frame[:3], 3),
            "normal_constraint_force_scalar": float(efc_force[address]),
            "position": _finite_vector(contact.pos, 3),
        })
    return pairs


def _entities(env: Any, resolutions: Mapping[tuple[str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    entities = []
    for key in sorted(resolutions, key=lambda item: (str(item[0]), int(item[1]))):
        resolution = resolutions[key]
        verify_entity_identity(env.sim.model, str(resolution["entity_type"]), int(resolution["entity_id"]), str(resolution.get("alias_to") or resolution.get("name") or ""))
        entity = collect_entity(env.sim.model, env.sim.data, resolution)
        entity.update({
            "logical_name": str(resolution.get("name") or ""),
            "alias_to": str(resolution.get("alias_to") or ""),
            "role": str(resolution.get("semantic_role") or ""),
            "resolution": str(resolution.get("resolution") or ""),
            "resolution_kind": str(resolution.get("resolution") or ""),
            "binding_identity": canonical_sha256({"entity_type": entity["entity_type"], "entity_id": entity["entity_id"], "entity_name": entity["entity_name"]}),
        })
        entities.append(entity)
    return entities


def _episode_row(env: Any, *, step: int, action: Mapping[str, Any], horizon: int, entities: list[dict[str, Any]], relations: list[Mapping[str, Any]], object_body_ids: set[int], eef_site_id: int) -> dict[str, Any]:
    env.sim.forward()
    data = env.sim.data
    eef_pos = _finite_vector(data.site_xpos[eef_site_id], 3)
    eef_quat = _finite_vector(data.site_xmat[eef_site_id], 9)
    from n5.phase2_labels.run_r5f_full40_materialize import mat_to_quat
    qpos = _finite_vector(data.qpos[-2:], 2)
    return {
        "step": int(step),
        "raw_action_7d": _finite_vector(action["raw_action"], 7),
        "action_env_7d": _finite_vector(action["env_action"], 7),
        "executed_action_7d": _finite_vector(action["env_action"], 7),
        "entities": entities,
        "contact_pairs": _contact_pairs(env, object_body_ids),
        "contact_ncon_total": int(data.ncon),
        "contact_truncated": False,
        "forward_before_capture": True,
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": mat_to_quat(eef_quat),
        "robot0_gripper_qpos": qpos,
        "horizon": int(horizon),
        "relations": relations,
    }


def _seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums = "".join(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    digest = _sha(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def run(args: argparse.Namespace) -> int:
    if not args.enable_runtime:
        raise RuntimeError("RUNTIME_DISABLED_UNTIL_EXPLICIT_DIAGNOSTIC_AUTHORIZATION")
    actual_commit = _git(ROOT, "rev-parse", "HEAD")
    actual_tree = _git(ROOT, "rev-parse", "HEAD^{tree}")
    if actual_commit != args.source_commit or actual_tree != args.source_tree:
        raise RuntimeError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    inventory = _inventory(args.gpu)

    exact_root = args.exact_plan_root.resolve()
    manifest = _load(exact_root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    parent = next((row for row in manifest.get("parents", []) if isinstance(row, Mapping) and row.get("canonical_parent_key") == args.parent_key), None)
    if not isinstance(parent, Mapping):
        raise ValueError("M4_PARENT_NOT_FOUND")
    clean_path = (exact_root / str(parent["clean_trajectory_path"])).resolve()
    clean = _load(clean_path)
    if clean.get("outcomes_read") is not False or not isinstance(clean.get("rows"), list):
        raise ValueError("M4_CLEAN_INPUT_BOUNDARY_INVALID")
    rows = sorted(clean["rows"], key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in rows] != list(range(len(rows))):
        raise ValueError("M4_CLEAN_ACTION_STEP_CLOSURE_INVALID")
    actions = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in rows]
    if canonical_sha256(canonical_value(actions)) != str(parent["clean_reference_action_sequence_sha256"]):
        raise ValueError("M4_CLEAN_ACTION_SHA_MISMATCH")
    if args.max_steps is not None:
        rows = rows[: int(args.max_steps)]
    if not rows:
        raise ValueError("M4_CLEAN_ACTIONS_EMPTY")

    suite, task_part, state_part = args.parent_key.split("/")
    task_index = int(task_part.removeprefix("task_"))
    state_index = int(state_part.removeprefix("state_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    _get_image, _get_processor, _get_model, _adapter, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root.resolve(), args.upstream_root.resolve())
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = (Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).resolve()
    registry = _load(args.registry_path.resolve())
    legacy = registry.get("legacy", registry)
    resolutions, relations = load_resolutions(str(args.registry_path.resolve()), allow_articulated=True)
    if legacy.get("task_key") != f"{suite}/task_{task_index:02d}" or _sha(bddl) != str(legacy.get("bddl_sha256")):
        raise ValueError("REGISTRY_OR_BDDL_BINDING_MISMATCH")
    object_body_ids = {int(resolution["entity_id"]) for resolution in resolutions.values() if resolution.get("entity_type") == "body" and resolution.get("semantic_role") == "MANIPULATED_OBJECT"}
    model_env = None
    env = None
    captured = []
    try:
        env, _obs = _new_env(OffScreenRenderEnv, str(bddl), HORIZONS[suite], int(args.gpu), init_state, args, output, write_binding=False)
        eef_site_id = int(env.sim.model.site_name2id("gripper0_grip_site"))
        for row in rows:
            captured.append(_episode_row(env, step=int(row["step"]), action=row, horizon=HORIZONS[suite], entities=_entities(env, resolutions), relations=relations, object_body_ids=object_body_ids, eef_site_id=eef_site_id))
            # Deliberately ignore reward, done, and info: this is clean telemetry only.
            _obs, _reward, _done, _info = env.step(list(row["env_action"]))
    finally:
        if env is not None:
            env.close()

    episode = {
        "schema": "FIT670_EPISODE_V2",
        "schema_version": "FIT670_FEATURE_SCHEMA_V1",
        "episode_id": args.parent_key,
        "suite": suite,
        "task_id": task_index,
        "state_id": state_index,
        "collection_seed": None,
        "initial_state_sha256": canonical_initial_state_sha256(init_state, {"canonical_parent_key": args.parent_key, "suite": suite, "task_index": task_index, "state_index": state_index}),
        "step_count": len(captured),
        "n_steps": len(captured),
        "official_horizon": HORIZONS[suite],
        "steps": captured,
        "telemetry": captured,
        "relations": relations,
        "geometry_status": "OK" if resolutions else "NOT_APPLICABLE",
        "forward_before_capture": True,
        "model_inference": False,
        "attack_enabled": False,
        "detector_loaded": False,
        "teacher_labels_generated": False,
        "outcomes_read": False,
    }
    canonical_rows = canonicalize_fit670_episode(episode)
    for index, row in enumerate(canonical_rows):
        validate_contact_row(row, expected_step=index)

    output.mkdir(parents=True)
    _write(output / "RECONSTRUCTED_FIT670_EPISODE.json", episode)
    _write(output / "R3_COVERAGE_VALIDATION.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_VALIDATION_V1",
        "status": "PASS_FULL_CLEAN_R3_COVERAGE" if len(rows) == len(clean["rows"]) else "PASS_PARTIAL_CLEAN_R3_COVERAGE",
        "canonical_parent_key": args.parent_key,
        "rows_replayed": len(rows),
        "rows_available": len(clean["rows"]),
        "r3_rows_validated": len(canonical_rows),
        "fit670_schema": episode["schema"],
        "intervention_executed": False,
        "labels_generated": 0,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    })
    _write(output / "PROVENANCE.json", {
        "schema": "STAGE_VI_M4_R3_CLEAN_RECONSTRUCTION_PROVENANCE_V1",
        "status": "PASS_DIAGNOSTIC_ONLY",
        "source_commit": actual_commit,
        "source_tree": actual_tree,
        "source_status": _git(ROOT, "status", "--porcelain"),
        "official_snapshot_root": str(args.official_snapshot_root.resolve()),
        "upstream_root": str(args.upstream_root.resolve()),
        "exact_plan_root": str(exact_root),
        "exact_plan_manifest_sha256": _sha(exact_root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"),
        "clean_trajectory_path": str(clean_path),
        "clean_trajectory_sha256": _sha(clean_path),
        "registry_path": str(args.registry_path.resolve()),
        "registry_sha256": _sha(args.registry_path.resolve()),
        "bddl_path": str(bddl),
        "bddl_sha256": _sha(bddl),
        "gpu_inventory": inventory,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
    })
    digest = _seal(output)
    print(json.dumps({"status": "PASS", "rows": len(canonical_rows), "sha256sums_sha256": digest, "output": str(output)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-plan-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
