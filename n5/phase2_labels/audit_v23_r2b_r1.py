"""FIT-only recursive source and head-wise reference audit for C3-T1D-R2B-R1.

This reads only the frozen pilot's three bound source files.  It does not
load a policy, train a student, or use labels/outcomes as a reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


class AuditHold(RuntimeError):
    pass


EXPECTED_SOURCE_FILES = (
    "episode_metadata.json",
    "step_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
)
OBJECT_WIDTH = 14
FORBIDDEN_COMPONENTS = {"cal", "check", "g10", "t2r", "t2r-d", "protected"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(path: Path) -> Path:
    path = path.resolve()
    if any(part.lower() in FORBIDDEN_COMPONENTS for part in path.parts):
        raise AuditHold(f"protected path rejected: {path}")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditHold(f"non-object JSONL row: {path}:{line_no}")
            rows.append(value)
    return rows


def verify_seal(root: Path) -> dict[str, Any]:
    root = safe_path(root)
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise AuditHold(f"unsealed root: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if len(side) != 2 or side[1] != "SHA256SUMS" or side[0] != sha256_file(sums):
        raise AuditHold(f"seal sidecar mismatch: {root}")
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        rel = Path(name.strip())
        if rel.is_absolute() or ".." in rel.parts or rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise AuditHold(f"unsafe sealed member: {name}")
        target = root / rel
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise AuditHold(f"sealed member mismatch: {target}")
        expected[rel.as_posix()] = digest
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if actual != set(expected):
        raise AuditHold(f"sealed closure mismatch: {root}")
    return {"root": str(root), "sha256sums_sha256": sha256_file(sums), "file_count": len(expected)}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, list):
        if not value:
            return (0,)
        child = shape(value[0])
        return (len(value),) + child if child is not None and all(shape(x) == child for x in value) else (len(value),)
    return () if finite(value) else None


def flatten_profile(value: Any, path: str, rows: dict[str, dict[str, Any]]) -> None:
    item = rows.setdefault(path, {"types": set(), "shapes": set(), "rows": 0, "finite_rows": 0,
                                  "finite_values": 0, "nonfinite_values": 0})
    item["rows"] += 1
    item["types"].add(type(value).__name__)
    s = shape(value)
    if s is not None:
        item["shapes"].add("x".join(map(str, s)) or "scalar")
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_profile(child, f"{path}.{key}" if path else str(key), rows)
    elif isinstance(value, list):
        scalar_values = [x for x in value if not isinstance(x, (dict, list))]
        if scalar_values:
            good = sum(finite(x) for x in scalar_values)
            item["finite_values"] += good
            item["nonfinite_values"] += len(scalar_values) - good
            if good == len(scalar_values) and scalar_values:
                item["finite_rows"] += 1
        for child in value:
            if isinstance(child, dict):
                flatten_profile(child, f"{path}[]", rows)
            elif isinstance(child, list):
                flatten_profile(child, f"{path}[]", rows)
    elif finite(value):
        item["finite_rows"] += 1
        item["finite_values"] += 1
    elif value is not None:
        item["nonfinite_values"] += 1


def profile_file(name: str, rows: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    values = rows if isinstance(rows, list) else [rows]
    profiles: dict[str, dict[str, Any]] = {}
    for value in values:
        flatten_profile(value, "", profiles)
    total = len(values)
    out = []
    for path, item in sorted(profiles.items()):
        out.append({
            "file": name,
            "field_path": path or "$",
            "types": ",".join(sorted(item["types"])),
            "shapes": ",".join(sorted(item["shapes"])),
            "row_coverage": f"{item['rows']}/{total}",
            "finite_row_coverage": f"{item['finite_rows']}/{total}",
            "finite_value_count": item["finite_values"],
            "nonfinite_value_count": item["nonfinite_values"],
        })
    return out


def parse_bddl(text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    obj = re.search(r"\(:objects\s*(.*?)\n\s*\)\s*\n", text, flags=re.S)
    objects: list[str] = []
    if obj:
        for line in obj.group(1).splitlines():
            m = re.fullmatch(r"\s*([A-Za-z0-9_ ]+)\s+-\s+([A-Za-z0-9_]+)\s*", line)
            if m:
                objects.extend(m.group(1).split())
    goal = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n", text, flags=re.S)
    relations = []
    if goal:
        relations = [(a, b, c) for a, b, c in re.findall(
            r"\(([A-Za-z_]+)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", goal.group(1)
        )]
    return objects, relations


def task_specs(suites: list[str], libero_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    # Importing LIBERO here only loads benchmark/BDDL metadata; no policy or rollout.
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    specs: dict[tuple[str, int], dict[str, Any]] = {}
    for suite in suites:
        benchmark = get_benchmark(suite)(0)
        for task_idx in range(10):
            task = benchmark.get_task(task_idx)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            text = bddl.read_text(encoding="utf-8")
            objects, relations = parse_bddl(text)
            specs[(suite, task_idx)] = {
                "task_name": task.name,
                "task_language": task.language,
                "bddl_path": str(bddl),
                "bddl_sha256": sha256_file(bddl),
                "objects": objects,
                "relations": relations,
                "expected_width": len(objects) * OBJECT_WIDTH,
                "libero_root": str(libero_root.resolve()),
            }
    return specs


def quat_geodesic(left: list[float], right: list[float]) -> float:
    def norm(q: list[float]) -> list[float]:
        n = math.sqrt(sum(float(x) * float(x) for x in q))
        if n == 0 or not math.isfinite(n):
            raise AuditHold("invalid quaternion in reference comparison")
        return [float(x) / n for x in q]
    a, b = norm(left), norm(right)
    dot = min(1.0, max(-1.0, abs(sum(x * y for x, y in zip(a, b)))))
    return 2.0 * math.acos(dot)


def l2(left: list[float], right: list[float]) -> float:
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    if not math.isfinite(value):
        raise AuditHold("nonfinite reference error")
    return value


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def compare_reference_geometry(args: argparse.Namespace, records: list[dict[str, Any]],
                               specs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    if args.geometry_root is None:
        return {"status": "HOLD_REFERENCE_MISSING", "reason": "geometry replay root not supplied"}
    threshold = json_load(args.threshold_config)
    if threshold.get("schema") != "C3_S3_NUMERICAL_THRESHOLDS_V1" or threshold.get("status") != "FROZEN":
        raise AuditHold("numerical threshold contract is not frozen")
    position_limit = float(threshold["dynamic_position_p99_m"])
    rotation_limit = float(threshold["dynamic_rotation_p99_rad"])
    object_positions: list[float] = []
    object_rotations: list[float] = []
    target_positions: list[float] = []
    target_rotations: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    missing_geometry_steps = 0
    relation_rows = 0
    pilot_by_id = {str(r["episode_id"]): r for r in records}
    for episode_id, record in pilot_by_id.items():
        spec = specs[(str(record["suite"]), int(record["task_id"]))]
        sidecar = read_jsonl(safe_path(Path(str(record["source_episode_root"]))) / "privileged_teacher_sidecar.jsonl")
        geometry_path = args.geometry_root / "episodes" / episode_id.replace("/", "__") / "geometry_cases.jsonl"
        if not geometry_path.is_file():
            raise AuditHold(f"geometry episode missing: {episode_id}")
        geometry_rows = read_jsonl(geometry_path)
        if len(sidecar) != len(geometry_rows):
            raise AuditHold(f"geometry/source length mismatch: {episode_id}")
        local_object, local_rotation, local_target, local_target_rotation = [], [], [], []
        for step, (source_row, geometry_row) in enumerate(zip(sidecar, geometry_rows)):
            if int(geometry_row.get("step", -1)) != step:
                raise AuditHold(f"geometry step closure failed: {episode_id}:{step}")
            for relation_index, (_, object_name, target_name) in enumerate(spec["relations"]):
                relations = geometry_row.get("relations", [])
                if relation_index >= len(relations):
                    missing_geometry_steps += 1
                    continue
                geometry_relation = relations[relation_index]
                object_pose = geometry_relation.get("object", {}).get("pose", {})
                base = spec["objects"].index(object_name) * OBJECT_WIDTH if object_name in spec["objects"] else None
                if base is None or not isinstance(source_row.get("object_state"), list):
                    continue
                state = source_row["object_state"]
                source_pos = [float(x) for x in state[base:base + 3]]
                source_quat = [float(x) for x in state[base + 3:base + 7]]
                replay_pos = object_pose.get("pos")
                replay_quat = object_pose.get("quat")
                if not (isinstance(replay_pos, list) and isinstance(replay_quat, list) and len(replay_pos) == 3 and len(replay_quat) == 4):
                    raise AuditHold(f"malformed geometry object pose: {episode_id}:{step}")
                local_object.append(l2(source_pos, replay_pos))
                local_rotation.append(quat_geodesic(source_quat, [float(x) for x in replay_quat]))
                relation_rows += 1
                target_base = spec["objects"].index(target_name) * OBJECT_WIDTH if target_name in spec["objects"] else None
                target_pose = geometry_relation.get("target", {}).get("pose", {})
                if target_base is not None and isinstance(target_pose.get("pos"), list) and isinstance(target_pose.get("quat"), list):
                    local_target.append(l2([float(x) for x in state[target_base:target_base + 3]], target_pose["pos"]))
                    local_target_rotation.append(quat_geodesic([float(x) for x in state[target_base + 3:target_base + 7]], [float(x) for x in target_pose["quat"]]))
        object_positions.extend(local_object); object_rotations.extend(local_rotation)
        target_positions.extend(local_target); target_rotations.extend(local_target_rotation)
        episode_rows.append({"episode_id": episode_id, "object_position_count": len(local_object),
                             "object_rotation_count": len(local_rotation), "target_position_count": len(local_target),
                             "target_rotation_count": len(local_target_rotation),
                             "object_position_p99_m": percentile(local_object, .99),
                             "object_rotation_p99_rad": percentile(local_rotation, .99)})
    metrics = {
        "object_position": {"count": len(object_positions), "p50": percentile(object_positions, .50),
                             "p95": percentile(object_positions, .95), "p99": percentile(object_positions, .99),
                             "max": max(object_positions) if object_positions else None},
        "object_rotation": {"count": len(object_rotations), "p50": percentile(object_rotations, .50),
                             "p95": percentile(object_rotations, .95), "p99": percentile(object_rotations, .99),
                             "max": max(object_rotations) if object_rotations else None},
        "target_position": {"count": len(target_positions), "p50": percentile(target_positions, .50),
                             "p95": percentile(target_positions, .95), "p99": percentile(target_positions, .99),
                             "max": max(target_positions) if target_positions else None},
        "target_rotation": {"count": len(target_rotations), "p50": percentile(target_rotations, .50),
                             "p95": percentile(target_rotations, .95), "p99": percentile(target_rotations, .99),
                             "max": max(target_rotations) if target_rotations else None},
    }
    known = metrics["object_position"]["count"] > 0 and metrics["object_rotation"]["count"] > 0
    pass_numeric = known and metrics["object_position"]["p99"] <= position_limit and metrics["object_rotation"]["p99"] <= rotation_limit
    return {"status": "PASS" if pass_numeric else "FAIL_NUMERICAL_FIDELITY", "threshold_config": str(args.threshold_config.resolve()),
            "threshold_config_sha256": sha256_file(args.threshold_config), "thresholds": {"position_p99_m": position_limit, "rotation_p99_rad": rotation_limit},
            "metrics": metrics, "relation_rows": relation_rows, "missing_geometry_steps": missing_geometry_steps,
            "qpos_classification_flips": None, "comotion_classification_flips": None, "predicate_decision_flips": None,
            "near_boundary_policy": "UNKNOWN", "episode_rows": episode_rows,
            "independent_chains": {"source": "recorded privileged object_state", "replay": "deterministic MuJoCo geometry_cases DIRECT_SIM_STATE",
                                   "same_action_replay_pose_chain": False}, "protected_payload_read": False}


def source_binding(args: argparse.Namespace, metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [args.collector_source, args.domain_source, args.robosuite_source,
             args.protocol_config, args.schema_doc]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise AuditHold(f"source binding file missing or unsafe: {path}")
    collector_sha = sha256_file(args.collector_source)
    declared = {str(m.get("collector_source_sha256", {}).get("official_clean_worker.py", "")) for m in metadata_rows}
    missing_variants = sorted(x for x in declared if x and x != collector_sha)
    collector_text = args.collector_source.read_text(encoding="utf-8")
    domain_text = args.domain_source.read_text(encoding="utf-8")
    robosuite_text = args.robosuite_source.read_text(encoding="utf-8")
    assertions = {
        "collector_reads_object_state": 'obs.get("object-state", [])' in collector_text,
        "collector_writes_qpos_eef_contact": all(k in collector_text for k in ("robot0_gripper_qpos", "robot0_eef_pos", "mujoco_contact_pairs")),
        "domain_object_order": "for (i, obj) in enumerate(self.objects):" in domain_text,
        "domain_four_components": "sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]" in domain_text,
        "robosuite_modality_concat": "obs_by_modality[modality].append" in robosuite_text and "np.concatenate(obs, axis=-1)" in robosuite_text,
    }
    if not all(assertions.values()):
        raise AuditHold(f"source assertions incomplete: {assertions}")
    return {
        "collector_source": str(args.collector_source.resolve()),
        "collector_source_sha256": collector_sha,
        "declared_collector_source_sha256_variants": sorted(declared),
        "unresolved_collector_source_sha256_variants": missing_variants,
        "status": "PASS" if not missing_variants else "HOLD_SOURCE_VARIANTS_UNRESOLVED",
        "domain_source": str(args.domain_source.resolve()),
        "domain_source_sha256": sha256_file(args.domain_source),
        "robosuite_source": str(args.robosuite_source.resolve()),
        "robosuite_source_sha256": sha256_file(args.robosuite_source),
        "protocol_config": str(args.protocol_config.resolve()),
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "schema_doc": str(args.schema_doc.resolve()),
        "schema_doc_sha256": sha256_file(args.schema_doc),
        "object_state_layout": ["pos[3]", "quat_xyzw[4]", "to_eef_pos[3]", "to_eef_quat_xyzw[4]"],
        "object_state_width": OBJECT_WIDTH,
        "assertions": assertions,
        "basis": "official collector + LIBERO source + robosuite modality order; no label/outcome inference",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    pilot = safe_path(args.pilot_root)
    pilot_seal = verify_seal(pilot)
    manifest = json_load(pilot / "PILOT_INPUT_MANIFEST.json")
    records = manifest.get("records")
    if manifest.get("episode_count") != 40 or not isinstance(records, list) or len(records) != 40:
        raise AuditHold("frozen pilot manifest is not exactly 40 records")
    ids = [str(r.get("episode_id")) for r in records]
    if len(set(ids)) != 40 or any(not r.get("source_episode_root") for r in records):
        raise AuditHold("pilot identity closure failed")
    all_profiles: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    total_steps = 0
    for record in records:
        episode_id = str(record["episode_id"])
        source = safe_path(Path(str(record["source_episode_root"])))
        declared = {str(x.get("name")): x for x in record.get("source_files", []) if isinstance(x, dict)}
        if set(declared) != set(EXPECTED_SOURCE_FILES):
            raise AuditHold(f"bound source file set mismatch: {episode_id}")
        actual_steps: dict[str, Any] = {}
        for name in EXPECTED_SOURCE_FILES:
            path = source / name
            spec = declared[name]
            if path.is_symlink() or not path.is_file() or sha256_file(path) != spec.get("sha256"):
                raise AuditHold(f"source file seal mismatch: {episode_id}/{name}")
            if name.endswith(".jsonl"):
                actual_steps[name] = read_jsonl(path)
            else:
                actual_steps[name] = json_load(path)
        metadata = actual_steps["episode_metadata.json"]
        steps = actual_steps["step_records.jsonl"]
        sidecar = actual_steps["privileged_teacher_sidecar.jsonl"]
        if len(steps) != len(sidecar) or len(steps) != int(record["observed_step_count"]):
            raise AuditHold(f"step count mismatch: {episode_id}")
        for row_set, name in ((steps, "step_records.jsonl"), (sidecar, "privileged_teacher_sidecar.jsonl")):
            if [row.get("step") for row in row_set] != list(range(len(row_set))):
                raise AuditHold(f"step identity closure failed: {episode_id}/{name}")
            for row in row_set:
                if row.get("suite") != record.get("suite") or int(row.get("task_idx", -1)) != int(record.get("task_id", -2)):
                    raise AuditHold(f"task identity mismatch: {episode_id}/{name}")
        if metadata.get("canonical_parent_key") != episode_id or metadata.get("suite") != record.get("suite"):
            raise AuditHold(f"metadata identity mismatch: {episode_id}")
        total_steps += len(steps)
        metadata_rows.append(metadata)
        all_profiles.extend(profile_file(f"{episode_id}/episode_metadata.json", metadata))
        all_profiles.extend(profile_file(f"{episode_id}/step_records.jsonl", steps))
        all_profiles.extend(profile_file(f"{episode_id}/privileged_teacher_sidecar.jsonl", sidecar))
        per_episode.append({"episode_id": episode_id, "suite": record["suite"], "task_idx": int(record["task_id"]),
                            "state_id": int(record["state_id"]), "step_count": len(steps),
                            "source_files_sha256": {name: sha256_file(source / name) for name in EXPECTED_SOURCE_FILES},
                            "metadata_schema": metadata.get("schema"), "collector_git_head": metadata.get("collector_git_head"),
                            "collector_source_sha256": metadata.get("collector_source_sha256", {}),
                            "initial_state_sha256": metadata.get("initial_state_sha256")})
    if total_steps != 9422:
        raise AuditHold(f"frozen pilot step count mismatch: {total_steps}")
    binding = source_binding(args, metadata_rows)
    suites = sorted({str(r["suite"]) for r in records})
    specs = task_specs(suites, args.libero_root)
    coverage: list[dict[str, Any]] = []
    relation_counts = Counter()
    for episode in per_episode:
        spec = specs[(episode["suite"], episode["task_idx"])]
        expected_width = spec["expected_width"]
        sidecar = read_jsonl(safe_path(Path(str(next(r for r in records if r["episode_id"] == episode["episode_id"])["source_episode_root"]))) / "privileged_teacher_sidecar.jsonl")
        width_ok = all(isinstance(row.get("object_state"), list) and len(row["object_state"]) == expected_width and all(finite(x) for x in row["object_state"]) for row in sidecar)
        qpos_ok = all(isinstance(row.get("robot0_gripper_qpos"), list) and len(row["robot0_gripper_qpos"]) == 2 and all(finite(x) for x in row["robot0_gripper_qpos"]) for row in sidecar)
        eef_ok = all(isinstance(row.get("robot0_eef_pos"), list) and len(row["robot0_eef_pos"]) == 3 and all(finite(x) for x in row["robot0_eef_pos"]) for row in sidecar)
        contact_ok = all(row.get("contact_capture_valid") is True for row in sidecar)
        object_names = set(spec["objects"])
        for predicate, obj, target in spec["relations"]:
            relation_counts[predicate] += 1
            object_ok = obj in object_names
            target_object = target in object_names
            articulated = any(token in target.lower() for token in ("drawer", "door", "cabinet", "handle"))
            physical = "AVAILABLE" if width_ok and qpos_ok and eef_ok and contact_ok and object_ok else "PARTIAL"
            instability = "AVAILABLE" if width_ok and eef_ok and contact_ok and object_ok else "PARTIAL"
            if articulated:
                placement = safe_release = "UNKNOWN"
                placement_reason = "articulated target unsupported by bound object-state/static-site evidence"
            elif target_object and width_ok:
                placement = "AVAILABLE"
                placement_reason = "object target has exact task-conditional object-state slice"
                safe_release = "AVAILABLE" if qpos_ok and contact_ok and eef_ok else "PARTIAL"
            else:
                placement = safe_release = "PARTIAL"
                placement_reason = "region/static target pose is absent from the three bound source files"
            row_base = {"episode_id": episode["episode_id"], "suite": episode["suite"], "task_idx": episode["task_idx"],
                        "state_id": episode["state_id"], "predicate": predicate, "object": obj, "target": target,
                        "object_state_width": expected_width, "object_state_layout_bound": width_ok}
            for head, status, reason in (
                ("gripper_closing_state", "AVAILABLE" if qpos_ok else "MISSING", "recorded robot0_gripper_qpos"),
                ("physical_criticality", physical, "qpos + EEF + object-state + contact sidecar"),
                ("instability", instability, "contact transition + object/EEF relative pose fields"),
                ("placement", placement, placement_reason),
                ("safe_release", safe_release, "placement + qpos + contact + causal EEF fields"),
            ):
                coverage.append({**row_base, "head": head, "status": status, "reason": reason,
                                 "reference_source": "RECORDED_TELEMETRY_OR_SCHEMA_ONLY",
                                 "independent_from_action_replay": True})
    relation_parse_ok = bool(relation_counts)
    status = "R1B_PARTIAL_REFERENCE_HOLD" if (
        binding["status"] != "PASS" or not relation_parse_ok or any(x["status"] in {"PARTIAL", "MISSING", "UNKNOWN"} for x in coverage)
    ) else "R1B_REFERENCE_RECOVERY_PASS"
    reference = {
        "schema": "C3_T1D_R2B_R1_REFERENCE_AUDIT_V1",
        "status": status,
        "episode_count": len(per_episode), "step_count": total_steps,
        "relation_count": sum(relation_counts.values()), "relation_predicate_counts": dict(sorted(relation_counts.items())),
        "relation_parse_ok": relation_parse_ok,
        "object_state_semantics": {"source": "LIBERO_BDDL_OBJECT_ORDER", "width_per_object": OBJECT_WIDTH,
                                   "components": ["pos[3]", "quat_xyzw[4]", "to_eef_pos[3]", "to_eef_quat_xyzw[4]"],
                                   "coordinate_frame": "world_pose_for_pos_quat; EEF-relative for to_eef fields",
                                   "source_bound": True},
        "head_counts": {head: Counter(x["status"] for x in coverage if x["head"] == head) for head in sorted({x["head"] for x in coverage})},
        "supported_relation_reference_coverage": all(x["status"] == "AVAILABLE" for x in coverage if x["status"] != "UNKNOWN"),
        "unknown_to_false": 0,
        "source_binding": binding,
        "pilot_seal": pilot_seal,
        "coverage_rows": coverage,
        "episode_audit": per_episode,
        "protected_payload_read": False, "model_inference": False, "student_training": False, "rollout": False, "attack": False,
    }
    geometry = verify_seal(args.geometry_root) if args.geometry_root else None
    r1d = {"schema": "C3_T1D_R2B_R1D_REFERENCE_COMPARISON_V1",
           **compare_reference_geometry(args, records, specs), "geometry_root": geometry}
    return {"field_audit": {"schema": "C3_T1D_R2B_R1A_FIELD_AUDIT_V1", "status": "PASS" if binding["status"] == "PASS" else "HOLD_SCHEMA_UNBOUND", "episode_count": len(per_episode),
                             "step_count": total_steps, "source_file_set": list(EXPECTED_SOURCE_FILES), "pilot_seal": pilot_seal,
                             "field_profiles": all_profiles, "episodes": per_episode, "source_binding": binding,
                             "suite_counts": dict(Counter(x["suite"] for x in per_episode)),
                             "task_counts": dict(Counter(f"{x['suite']}/task_{x['task_idx']:02d}" for x in per_episode)),
                             "protected_payload_read": False},
            "reference_audit": reference, "r1d": r1d}


def write_output(parent: Path, name: str, result: dict[str, Any]) -> dict[str, Any]:
    parent = safe_path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / name
    if final.exists() or final.is_symlink():
        raise AuditHold(f"non-overwrite violation: {final}")
    staging = parent / f".staging_{name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "R1A_FIELD_AUDIT.json").write_text(json.dumps(result["field_audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "R1B_REFERENCE_AUDIT.json").write_text(json.dumps(result["reference_audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "R1D_REFERENCE_DECISION.json").write_text(json.dumps(result["r1d"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "R1A_FIELD_PATHS.csv").open("w", newline="", encoding="utf-8") as f:
            rows = result["field_audit"]["field_profiles"]
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file", "field_path"])
            writer.writeheader(); writer.writerows(rows)
        with (staging / "HEAD_REFERENCE_COVERAGE.csv").open("w", newline="", encoding="utf-8") as f:
            rows = result["reference_audit"]["coverage_rows"]
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["episode_id", "head", "status"])
            writer.writeheader(); writer.writerows(rows)
        report = {
            "schema": "C3_T1D_R2B_R1_AUDIT_RECEIPT_V1", "R1A": result["field_audit"]["status"],
            "R1B": result["reference_audit"]["status"], "R1D": result["r1d"]["status"],
            "protected_payload_read": False, "model_inference": False, "student_training": False,
            "rollout": False, "attack": False,
        }
        (staging / "R2B_R1_RECEIPT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(staging / n)}  {n}\n" for n in files), encoding="utf-8")
        sums_sha = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"root": str(final), "sha256sums_sha256": sums_sha, **report}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-root", type=Path, required=True)
    p.add_argument("--geometry-root", type=Path)
    p.add_argument("--threshold-config", type=Path, required=True)
    p.add_argument("--collector-source", type=Path, required=True)
    p.add_argument("--domain-source", type=Path, required=True)
    p.add_argument("--robosuite-source", type=Path, required=True)
    p.add_argument("--protocol-config", type=Path, required=True)
    p.add_argument("--schema-doc", type=Path, required=True)
    p.add_argument("--libero-root", type=Path, required=True)
    p.add_argument("--output-parent", type=Path, required=True)
    p.add_argument("--output-name", required=True)
    args = p.parse_args()
    try:
        print(json.dumps(write_output(args.output_parent, args.output_name, audit(args)), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "HOLD_SCHEMA_UNBOUND", "reason": f"{type(exc).__name__}:{exc}",
                          "protected_payload_read": False, "model_inference": False, "student_training": False,
                          "rollout": False, "attack": False}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
