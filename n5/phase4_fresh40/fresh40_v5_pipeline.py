#!/usr/bin/env python3
"""Minimal, sealed, FIT-only Fresh40 V5 development pipeline.

The only episode input is the already sealed R5-F Run-B telemetry root.  This
module deliberately treats missing object-gripper contact pairs as a
development proxy limitation; it never upgrades the proxy to final Teacher
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
STUDENT = ROOT / "n5" / "phase3_student"
for _p in (SRC, STUDENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HEADS = ("physical_criticality", "k10_feasible", "safe_release", "instability", "gripper_closing_state")
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FORBIDDEN = {"task_success", "terminal", "terminal_state", "reward", "outcome", "future", "attack_result"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite_vector(value: Any, length: int | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if length is not None and len(value) != length:
        return None
    try:
        result = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def pose(entity: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(entity, Mapping):
        return None
    world = entity.get("world_pose")
    if not isinstance(world, Mapping):
        return None
    pos = finite_vector(world.get("position"), 3)
    quat = finite_vector(world.get("quaternion"), 4)
    if pos is None or quat is None:
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        return None
    return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64) / norm


def tri(value: Any) -> str:
    if value in ("TRUE", True, 1):
        return "TRUE"
    if value in ("FALSE", False, 0):
        return "FALSE"
    return "UNKNOWN"


def label(value: str, reason: str, source: str = "RECORDED_TELEMETRY_DEVELOPMENT_PROXY") -> dict[str, Any]:
    if value not in ("TRUE", "FALSE", "UNKNOWN"):
        raise ValueError(value)
    return {"value": value, "mask": value != "UNKNOWN", "reason": reason, "source": source}


def aggregate_or(values: Iterable[str]) -> str:
    vals = list(values)
    if any(v == "TRUE" for v in vals):
        return "TRUE"
    if not vals or any(v == "UNKNOWN" for v in vals):
        return "UNKNOWN"
    return "FALSE"


def aggregate_and(values: Iterable[str]) -> str:
    vals = list(values)
    if any(v == "FALSE" for v in vals):
        return "FALSE"
    if not vals or any(v == "UNKNOWN" for v in vals):
        return "UNKNOWN"
    return "TRUE"


def hash_listed_files(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    if not sums.is_file() or not (root / "SHA256SUMS.sha256").is_file():
        raise RuntimeError(f"missing source seal: {root}")
    declared_side = (root / "SHA256SUMS.sha256").read_text().strip().split()[0]
    actual_side = sha256_file(sums)
    if declared_side != actual_side:
        raise RuntimeError(f"SHA256SUMS sidecar mismatch: {root}")
    declared: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        declared[rel.strip().lstrip("* ")] = digest
    for rel, digest in declared.items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"sealed file mismatch: {root / rel}")
    return declared


def _seal(root: Path) -> dict[str, str]:
    payload = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        payload.append((p.relative_to(root).as_posix(), sha256_file(p)))
    (root / "SHA256SUMS").write_text("".join(f"{digest}  {rel}\n" for rel, digest in payload))
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n")
    return {"sha256sums_sha256": sums_sha, "files": len(payload)}


def _publish(staging: Path, final: Path) -> None:
    if final.exists():
        raise RuntimeError(f"refusing to overwrite existing root: {final}")
    # ponytail: unique staging plus an existence check is sufficient here; the
    # caller uses a unique run root and never mutates a published root.
    os.rename(staging, final)


def _load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != "FRESH40_V5_DEVELOPMENT_PROTOCOL_V1":
        raise RuntimeError("unexpected protocol schema")
    return cfg


def _episode_paths(source_root: Path) -> list[Path]:
    return sorted((source_root / "episodes").glob("*/episode.json"))


def audit_source(source_root: Path, output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    declared = hash_listed_files(source_root)
    manifest = json.loads((source_root / "MANIFEST.json").read_text())
    paths = _episode_paths(source_root)
    if len(paths) != 40:
        raise RuntimeError(f"expected 40 episode files, found {len(paths)}")
    identities: set[str] = set()
    suite_tasks: dict[str, set[int]] = {s: set() for s in SUITES}
    total_steps = 0
    telemetry_steps = 0
    contact_pairs_steps = 0
    invalid_steps = 0
    direct_entities = 0
    per_suite_steps: dict[str, int] = {s: 0 for s in SUITES}
    rows = []
    for path in paths:
        ep = json.loads(path.read_text())
        ident = str(ep["episode_id"])
        if ident in identities:
            raise RuntimeError(f"duplicate identity: {ident}")
        identities.add(ident)
        suite = str(ep["suite"])
        task = int(ep["task_id"])
        if suite not in SUITES or task not in range(10):
            raise RuntimeError(f"unexpected identity: {suite} task {task}")
        suite_tasks[suite].add(task)
        steps, telemetry = ep.get("steps", []), ep.get("telemetry", [])
        if len(steps) != len(telemetry) or len(steps) != int(ep.get("step_count", -1)):
            raise RuntimeError(f"step/telemetry mismatch: {ident}")
        for idx, (step, tel) in enumerate(zip(steps, telemetry)):
            if step.get("step") != idx or tel.get("step") != idx:
                raise RuntimeError(f"non-contiguous step sequence: {ident}:{idx}")
            if not isinstance(step.get("action_raw_7d"), list) or len(step["action_raw_7d"]) != 7:
                invalid_steps += 1
            if not isinstance(tel.get("robot0_eef_pos"), list) or len(tel["robot0_eef_pos"]) != 3:
                invalid_steps += 1
            if not isinstance(tel.get("robot0_gripper_qpos"), list) or len(tel["robot0_gripper_qpos"]) != 2:
                invalid_steps += 1
            if "mujoco_contact_pairs" in tel:
                contact_pairs_steps += 1
            if isinstance(tel.get("entities"), list):
                direct_entities += len(tel["entities"])
        total_steps += len(steps)
        telemetry_steps += len(telemetry)
        per_suite_steps[suite] += len(steps)
        rows.append({"episode_id": ident, "suite": suite, "task_id": task, "steps": len(steps), "relations": len(ep.get("relations", []))})
        if ep.get("teacher_labels_generated") is not False or ep.get("attack_enabled") is not False:
            raise RuntimeError(f"source flags are not clean/non-teacher: {ident}")
    if identities != {r["episode_id"] for r in rows}:
        raise RuntimeError("identity closure failed")
    if any(suite_tasks[s] != set(range(10)) for s in SUITES):
        raise RuntimeError(f"suite/task closure failed: {suite_tasks}")
    report = {
        "schema": "FRESH40_V5_SOURCE_AUDIT_V1",
        "status": "PASS_SOURCE_INTEGRITY_WITH_CONTACT_LIMITATION",
        "source_root": str(source_root),
        "source_sha256sums_sha256": sha256_file(source_root / "SHA256SUMS"),
        "source_manifest_sha256": sha256_file(source_root / "MANIFEST.json"),
        "source_manifest_gate": manifest.get("gate"),
        "source_manifest_consumer_eligible": manifest.get("consumer_eligible"),
        "identity_count": len(identities),
        "total_steps": total_steps,
        "telemetry_steps": telemetry_steps,
        "per_suite_steps": per_suite_steps,
        "direct_entity_pose_rows": direct_entities,
        "mujoco_contact_pairs_steps": contact_pairs_steps,
        "global_contact_count_only": contact_pairs_steps == 0,
        "invalid_required_rows": invalid_steps,
        "teacher_labels_generated": False,
        "attack_enabled": False,
        "protected_payload_read": False,
        "new_openvla_inference": False,
        "source_files_checked": len(declared),
        "episodes": rows,
        "quality_limitation": "No exact object-gripper contact pairs; contact-dependent heads are development distance/global-contact proxies and non-consumable.",
        "authorization": config["authorization"],
    }
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (staging / "source_binding.json").write_text(json.dumps({"source_root": str(source_root), "source_manifest_sha256": report["source_manifest_sha256"], "source_sha256sums_sha256": report["source_sha256sums_sha256"], "source_commit": manifest.get("source_commit"), "source_tree": manifest.get("source_tree")}, indent=2, sort_keys=True))
    (staging / "runtime_audit.json").write_text(json.dumps({"model_inference": "historical source only", "new_inference": False, "protected_reads": 0, "attack": False}, indent=2, sort_keys=True))
    seal = _seal(staging)
    _publish(staging, output_root)
    report["output_root"] = str(output_root)
    report["output_sha256s_sha256"] = seal["sha256sums_sha256"]
    return report


def _entity_lookup(telemetry: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    out = {}
    for entity in telemetry.get("entities", []):
        if not isinstance(entity, Mapping):
            continue
        try:
            key = (str(entity["entity_type"]), int(entity["entity_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in out:
            raise RuntimeError(f"duplicate entity key {key}")
        out[key] = entity
    return out


def _relation_pose(lookup: Mapping[tuple[str, int], Mapping[str, Any]], resolution: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        key = (str(resolution["entity_type"]), int(resolution["entity_id"]))
    except (KeyError, TypeError, ValueError):
        return None
    entity = lookup.get(key)
    if entity is None:
        return None
    return pose(entity)


def _persistence(values: list[str], minimum: int) -> list[str]:
    result, streak = [], 0
    for value in values:
        if value == "TRUE":
            streak += 1
            result.append("TRUE" if streak >= minimum else "UNKNOWN")
        else:
            streak = 0
            result.append(value)
    return result


def _teacher_episode(ep: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tc = cfg["teacher_contract"]
    steps, telemetry, relations = ep["steps"], ep["telemetry"], ep.get("relations", [])
    thresholds = {k: float(v) for k, v in tc.items() if isinstance(v, (int, float))}
    relation_history: dict[int, dict[str, Any]] = {}
    raw: dict[str, list[str]] = {h: [] for h in HEADS if h != "k10_feasible"}
    for idx, (step, tel) in enumerate(zip(steps, telemetry)):
        q = finite_vector(tel.get("robot0_gripper_qpos"), 2)
        eef = finite_vector(tel.get("robot0_eef_pos"), 3)
        width = float(abs(q[0]) + abs(q[1])) if q is not None else None
        q_close = "UNKNOWN" if width is None else ("TRUE" if width <= thresholds["qpos_close_threshold"] else "FALSE" if width >= thresholds["qpos_open_threshold"] else "UNKNOWN")
        per_phys, per_safe, per_inst = [], [], []
        for ridx, relation in enumerate(relations):
            lookup = _entity_lookup(tel)
            obj_res = relation.get("object_resolution", {})
            target_res = relation.get("target_resolution", {})
            obj_pose = _relation_pose(lookup, obj_res)
            target_pose = _relation_pose(lookup, target_res)
            previous = relation_history.get(ridx)
            if obj_pose is None or target_pose is None or eef is None:
                per_phys.append("UNKNOWN"); per_safe.append("UNKNOWN"); per_inst.append("UNKNOWN")
                relation_history[ridx] = {"object": None, "target": None, "contact": None}
                continue
            obj_pos, _ = obj_pose; target_pos, _ = target_pose
            dist_eef = float(np.linalg.norm(obj_pos - eef))
            dist_target = float(np.linalg.norm(obj_pos - target_pos))
            contact_proxy = bool(dist_eef <= thresholds["contact_distance_proxy_threshold_m"] and int(tel.get("contact_count", 0)) > 0)
            if previous and previous.get("object") is not None:
                obj_delta = obj_pos - previous["object"]
                eef_delta = eef - previous["eef"]
                od, ed = float(np.linalg.norm(obj_delta)), float(np.linalg.norm(eef_delta))
                if od < thresholds["comotion_min_displacement_m"] or ed < thresholds["comotion_min_displacement_m"]:
                    comotion = "UNKNOWN"
                else:
                    comotion = "TRUE" if float(np.dot(obj_delta, eef_delta) / (od * ed)) >= thresholds["comotion_cosine_threshold"] else "FALSE"
                slip = "TRUE" if contact_proxy and od > thresholds["slip_relative_motion_threshold_m"] else "FALSE"
                contact_loss = "TRUE" if previous.get("contact") is True and not contact_proxy else "FALSE"
            else:
                comotion, slip, contact_loss = "UNKNOWN", "UNKNOWN", "UNKNOWN"
            lift = "TRUE" if float(obj_pos[2] - (previous.get("initial_z", obj_pos[2]) if previous else obj_pos[2])) >= thresholds["lift_threshold_m"] else "FALSE"
            stable_grasp = "TRUE" if contact_proxy and (comotion in ("TRUE", "UNKNOWN") or lift == "TRUE") else "FALSE"
            physical = "TRUE" if stable_grasp == "TRUE" and (lift == "TRUE" or comotion == "TRUE") else "UNKNOWN" if stable_grasp == "UNKNOWN" or comotion == "UNKNOWN" else "FALSE"
            placement = "TRUE" if dist_target <= thresholds["placement_distance_threshold_m"] else "FALSE"
            if previous and previous.get("target") is not None:
                stable = "TRUE" if float(np.linalg.norm(target_pos - previous["target"])) <= thresholds["placement_stability_translation_threshold_m"] else "FALSE"
            else:
                stable = "UNKNOWN"
            released = "UNKNOWN" if width is None else "TRUE" if width >= thresholds["qpos_open_threshold"] and not contact_proxy else "FALSE" if width < thresholds["qpos_open_threshold"] else "UNKNOWN"
            safe = aggregate_and((placement, released, stable))
            instability = aggregate_or((slip, contact_loss))
            per_phys.append(physical); per_safe.append(safe); per_inst.append(instability)
            relation_history[ridx] = {"object": obj_pos, "eef": np.asarray(eef, dtype=np.float64), "target": target_pos, "contact": contact_proxy, "initial_z": previous.get("initial_z", float(obj_pos[2])) if previous else float(obj_pos[2])}
        raw["physical_criticality"].append(aggregate_or(per_phys))
        raw["safe_release"].append(aggregate_and(per_safe))
        raw["instability"].append(aggregate_or(per_inst))
        raw["gripper_closing_state"].append(q_close)
    persistent = {
        "physical_criticality": _persistence(raw["physical_criticality"], 2),
        "safe_release": _persistence(raw["safe_release"], 2),
        "instability": _persistence(raw["instability"], 1),
        "gripper_closing_state": _persistence(raw["gripper_closing_state"], 1),
    }
    rows = []
    for idx, step in enumerate(steps):
        safe = persistent["safe_release"][idx]
        remaining = HORIZONS[str(ep["suite"])] - idx - 1 if str(ep["suite"]) in HORIZONS else -1
        k10 = "UNKNOWN" if remaining < 0 or safe == "UNKNOWN" else "TRUE" if remaining >= 10 and safe == "FALSE" else "FALSE"
        row = {"step": int(step["step"])}
        for head in ("physical_criticality", "k10_feasible", "safe_release", "instability", "gripper_closing_state"):
            value = k10 if head == "k10_feasible" else persistent[head][idx]
            source = "RECORDED_TELEMETRY_DEVELOPMENT_PROXY" if head != "k10_feasible" else "PROTOCOL_HORIZON_AND_SAFE_RELEASE_PROXY"
            row[head] = label(value, "UNKNOWN_CONTACT_OR_GEOMETRY" if value == "UNKNOWN" else "CAUSAL_RECORDED_TELEMETRY", source)
        rows.append(row)
    summary = {"steps": len(rows), "head_counts": {h: {v: sum(r[h]["value"] == v for r in rows) for v in ("TRUE", "FALSE", "UNKNOWN")} for h in HEADS}, "relation_count": len(relations), "contact_source": tc["contact_source"]}
    return rows, summary


def build_teacher(source_root: Path, output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    hash_listed_files(source_root)
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or output_root.exists():
        raise RuntimeError(f"output already exists: {output_root}")
    staging.mkdir(parents=True)
    (staging / "episodes").mkdir()
    source_manifest_sha = sha256_file(source_root / "MANIFEST.json")
    total = {h: {v: 0 for v in ("TRUE", "FALSE", "UNKNOWN")} for h in HEADS}
    episode_summaries = []
    for ep_path in _episode_paths(source_root):
        ep = json.loads(ep_path.read_text())
        rows, summary = _teacher_episode(ep, config)
        ep_dir = staging / "episodes" / str(ep["episode_id"])
        ep_dir.mkdir()
        with (ep_dir / "labels.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        (ep_dir / "episode_manifest.json").write_text(json.dumps({"episode_id": ep["episode_id"], "suite": ep["suite"], "task_id": ep["task_id"], "state_id": ep["state_id"], "step_count": len(rows), "source_episode_sha256": sha256_file(ep_path), "teacher_quality_status": config["teacher_contract"]["quality_status"], "exact_contact_available": False}, indent=2, sort_keys=True))
        for h in HEADS:
            for value in total[h]:
                total[h][value] += summary["head_counts"][h][value]
        episode_summaries.append({"episode_id": ep["episode_id"], **summary})
    manifest = {"schema": "FRESH40_V5_TEACHER_PROXY_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "source_root": str(source_root), "source_manifest_sha256": source_manifest_sha, "teacher_protocol_sha256": canonical_sha(config), "identity_count": len(episode_summaries), "total_steps": sum(x["steps"] for x in episode_summaries), "heads": list(HEADS), "head_counts": total, "episodes": episode_summaries, "contact_pairs_available": False, "quality_status": config["teacher_contract"]["quality_status"], "formal_training_authorized": False, "formal_inference_authorized": False, "attack_authorized": False, "protected_reads": False}
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (staging / "QUALITY_LIMITATION.json").write_text(json.dumps({"status": "PROXY_NONCONSUMABLE", "reason": "r5f telemetry has global contact_count but no exact object-gripper contact pairs", "safe_release_is_not_final": True}, indent=2, sort_keys=True))
    seal = _seal(staging)
    _publish(staging, output_root)
    return {"output_root": str(output_root), "sha256sums_sha256": seal["sha256sums_sha256"], "manifest_sha256": sha256_file(output_root / "MANIFEST.json"), "head_counts": total}


def _stream_features(ep: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2, FEATURE_NAMES
    adapter = SC5StreamingFeatureAdapterV2()
    values, candidates = [], []
    previous_eef = None
    for step, tel in zip(ep["steps"], ep["telemetry"]):
        raw = finite_vector(step.get("action_raw_7d"), 7)
        env = finite_vector(step.get("action_env_7d"), 7)
        q = finite_vector(tel.get("robot0_gripper_qpos"), 2)
        eef = finite_vector(tel.get("robot0_eef_pos"), 3)
        if raw is None or env is None or q is None or eef is None:
            raise RuntimeError(f"missing causal feature source at {ep['episode_id']}:{step.get('step')}")
        if previous_eef is None:
            vel = [0.0, 0.0, 0.0]
        else:
            vel = [float(a - b) for a, b in zip(eef, previous_eef)]
        rec = adapter.update(int(step["step"]), raw[6], env[6], sum(q), sum(abs(x) for x in q), *eef, *vel, *raw[:3], raw[6])
        if not rec["valid"]:
            raise RuntimeError(f"invalid SC5 feature row at {ep['episode_id']}:{step.get('step')}: {rec.get('error')}")
        values.append([float(rec["features"][name]) for name in FEATURE_NAMES])
        candidates.append(bool(raw[6] <= 0.5 and env[6] > 0))
        previous_eef = eef
    return np.asarray(values, dtype=np.float32), np.asarray(candidates, dtype=bool)


def _select_split(eps: list[str], seed: int) -> tuple[list[str], list[str]]:
    by_suite = {suite: [] for suite in SUITES}
    for ident in eps:
        by_suite[ident.split("/", 1)[0]].append(ident)
    dev, train = [], []
    for suite, values in by_suite.items():
        ordered = sorted(values, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
        dev.extend(ordered[:2]); train.extend(ordered[2:])
    return sorted(train), sorted(dev)


def build_dataset(source_root: Path, teacher_root: Path, output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    hash_listed_files(source_root); hash_listed_files(teacher_root)
    identities = [str(json.loads(p.read_text())["episode_id"]) for p in _episode_paths(source_root)]
    train, dev = _select_split(identities, int(config["student_contract"]["seed"]))
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or output_root.exists():
        raise RuntimeError(f"output already exists: {output_root}")
    staging.mkdir(parents=True); (staging / "episodes").mkdir()
    train_values = []
    all_manifest = []
    teacher_manifest_sha = sha256_file(teacher_root / "MANIFEST.json")
    for ep_path in _episode_paths(source_root):
        ep = json.loads(ep_path.read_text()); ident = str(ep["episode_id"])
        features, candidate = _stream_features(ep)
        label_path = teacher_root / "episodes" / ident / "labels.jsonl"
        rows = [json.loads(line) for line in label_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(features):
            raise RuntimeError(f"feature/label mismatch: {ident}")
        records = []
        for idx, (feat, cand, row) in enumerate(zip(features, candidate, rows)):
            records.append({"episode_id": ident, "step": idx, "features_25d": feat.tolist(), "student_valid": True, "candidate_close": bool(cand), "labels": {h: row[h] for h in HEADS}})
        safe_ident = ident.replace("/", "__")
        (staging / "episodes" / f"{safe_ident}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        if ident in train:
            train_values.append(features)
        all_manifest.append({"episode_id": ident, "step_count": len(records), "split": "train" if ident in train else "dev"})
    concat = np.concatenate(train_values, axis=0)
    mean, std = concat.mean(0).astype(np.float32), (concat.std(0) + 1e-8).astype(np.float32)
    (staging / "normalizer.json").write_text(json.dumps({"schema": "FRESH40_V5_NORMALIZER_V1", "fit_split": "train", "feature_count": 25, "mean": mean.tolist(), "std": std.tolist()}, indent=2, sort_keys=True))
    manifest = {"schema": "FRESH40_V5_CAUSAL_STUDENT_DATASET_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "source_root": str(source_root), "source_manifest_sha256": sha256_file(source_root / "MANIFEST.json"), "teacher_root": str(teacher_root), "teacher_manifest_sha256": teacher_manifest_sha, "feature_schema": "SC5StreamingFeatureAdapterV2", "feature_count": 25, "identities": all_manifest, "train_identity_sha256": canonical_sha(train), "dev_identity_sha256": canonical_sha(dev), "train_identities": train, "dev_identities": dev, "teacher_fields_in_student_input": False, "future_frames": 0, "protected_reads": False, "formal_training_authorized": False, "attack_authorized": False}
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (staging / "split_identities.json").write_text(json.dumps({"train": train, "dev": dev}, indent=2, sort_keys=True))
    seal = _seal(staging); _publish(staging, output_root)
    return {"output_root": str(output_root), "sha256sums_sha256": seal["sha256sums_sha256"], "manifest_sha256": sha256_file(output_root / "MANIFEST.json"), "train": len(train), "dev": len(dev)}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("audit", "teacher", "dataset", "train", "shadow", "canary"))
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--teacher-root", type=Path)
    ap.add_argument("--dataset-root", type=Path)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "FRESH40_V5_DEVELOPMENT_PROTOCOL_V1.json")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variant", choices=("critical_only", "three_head", "full_five"), default="full_five")
    ap.add_argument("--epochs", type=int, default=12)
    return ap.parse_args()


def main() -> int:
    args = _parse_args(); cfg = _load_config(args.config)
    if args.command == "audit":
        if not args.source_root: raise SystemExit("--source-root required")
        result = audit_source(args.source_root.resolve(), args.output_root.resolve(), cfg)
    elif args.command == "teacher":
        if not args.source_root: raise SystemExit("--source-root required")
        result = build_teacher(args.source_root.resolve(), args.output_root.resolve(), cfg)
    elif args.command == "dataset":
        if not args.source_root or not args.teacher_root: raise SystemExit("--source-root and --teacher-root required")
        result = build_dataset(args.source_root.resolve(), args.teacher_root.resolve(), args.output_root.resolve(), cfg)
    elif args.command == "train":
        raise SystemExit("train is intentionally gated until CPU audit confirms active-head coverage; use the dedicated training stage after dataset review")
    else:
        raise SystemExit(f"{args.command} is not enabled in the initial E0-E2 CPU pass")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
