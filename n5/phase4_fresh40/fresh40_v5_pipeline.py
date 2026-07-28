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
VARIANT_ACTIVE_HEADS = {
    "critical_only": ("physical_criticality",),
    "three_head": ("physical_criticality", "instability", "gripper_closing_state"),
    "full_five": HEADS,
}
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


def variant_decision(variant: str, candidate_close: bool, probabilities: Mapping[str, float], threshold: float = 0.5) -> bool:
    """Frozen deploy decision; inactive heads are never read."""
    if variant not in VARIANT_ACTIVE_HEADS:
        raise ValueError(f"unknown variant: {variant}")
    if not candidate_close or float(probabilities["physical_criticality"]) < threshold:
        return False
    if variant == "critical_only":
        return True
    if float(probabilities["instability"]) >= threshold or float(probabilities["gripper_closing_state"]) < threshold:
        return False
    if variant == "three_head":
        return True
    return float(probabilities["k10_feasible"]) >= threshold and float(probabilities["safe_release"]) < threshold


def _rank_auc(scores: list[float], values: list[bool]) -> float | None:
    pos = sum(values); neg = len(values) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: (float(scores[i]), i))
    rank_sum = 0.0; i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for k in order[i:j] if values[k])
        i = j
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _average_precision(scores: list[float], values: list[bool]) -> float | None:
    pos = sum(values)
    if not pos:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    hit = 0; total = 0.0
    for rank, idx in enumerate(order, 1):
        if values[idx]:
            hit += 1; total += hit / rank
    return float(total / pos)


def _head_metrics(scores: list[float], values: list[bool]) -> dict[str, Any]:
    pos = sum(values); neg = len(values) - pos
    pred = [x >= 0.5 for x in scores]
    tp = sum(p and y for p, y in zip(pred, values)); fp = sum(p and not y for p, y in zip(pred, values))
    fn = sum((not p) and y for p, y in zip(pred, values)); tn = sum((not p) and not y for p, y in zip(pred, values))
    balanced_accuracy = ((tp / pos) + (tn / neg)) / 2.0 if pos and neg else None
    return {"known_steps": len(values), "positive": pos, "negative": neg, "auroc": _rank_auc(scores, values), "auprc": _average_precision(scores, values), "recall": tp / pos if pos else None, "precision": tp / (tp + fp) if tp + fp else None, "balanced_accuracy": balanced_accuracy}


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
        ep_dir.mkdir(parents=True)
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


def _dataset_episode_path(dataset_root: Path, identity: str) -> Path:
    return dataset_root / "episodes" / f"{identity.replace('/', '__')}.jsonl"


def _load_student_data(dataset_root: Path, split: str) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    manifest = json.loads((dataset_root / "MANIFEST.json").read_text())
    split_data = json.loads((dataset_root / "split_identities.json").read_text())
    normalizer = json.loads((dataset_root / "normalizer.json").read_text())
    mean, std = np.asarray(normalizer["mean"], dtype=np.float32), np.asarray(normalizer["std"], dtype=np.float32)
    episodes = []
    for identity in split_data[split]:
        rows = [json.loads(line) for line in _dataset_episode_path(dataset_root, identity).read_text().splitlines() if line.strip()]
        if not rows or [r["step"] for r in rows] != list(range(len(rows))):
            raise RuntimeError(f"invalid dataset step closure: {identity}")
        features = np.asarray([r["features_25d"] for r in rows], dtype=np.float32)
        if features.shape[1] != 25 or not np.isfinite(features).all():
            raise RuntimeError(f"invalid 25D input: {identity}")
        features = (features - mean) / std
        episodes.append({"identity": identity, "rows": rows, "features": features})
    return episodes, mean, std


def train_student(dataset_root: Path, output_root: Path, config: Mapping[str, Any], variant: str, epochs: int, device_name: str = "cpu") -> dict[str, Any]:
    import torch
    from n5_student_model import N5MultiHeadStudent

    if output_root.exists():
        raise RuntimeError(f"output already exists: {output_root}")
    active = VARIANT_ACTIVE_HEADS[variant]
    train_eps, mean, std = _load_student_data(dataset_root, "train")
    coverage = {h: {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0} for h in HEADS}
    for ep in train_eps:
        for row in ep["rows"]:
            for h in HEADS:
                coverage[h][row["labels"][h]["value"]] += 1
    for head in active:
        if coverage[head]["TRUE"] == 0 or coverage[head]["FALSE"] == 0:
            raise RuntimeError(f"active head lacks both classes: {head}: {coverage[head]}")
    seed = int(config["student_contract"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    model = N5MultiHeadStudent(input_dim=25, hidden=int(config["student_contract"]["hidden"]), short_rf=int(config["student_contract"]["short_rf"]), long_rf=int(config["student_contract"]["long_rf"]), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["student_contract"]["learning_rate"]), weight_decay=float(config["student_contract"]["weight_decay"]))
    history = []
    model.train()
    for epoch in range(int(epochs)):
        epoch_order = sorted(range(len(train_eps)), key=lambda i: hashlib.sha256(f"{seed}:{epoch}:{train_eps[i]['identity']}".encode()).hexdigest())
        losses = []
        for ep_idx in epoch_order:
            ep = train_eps[ep_idx]
            x = torch.from_numpy(ep["features"]).unsqueeze(0).to(device)
            outputs = model(x, timestep_mask=torch.ones((1, x.shape[1]), dtype=torch.bool, device=device))
            terms = []
            for head in active:
                values = torch.tensor([1.0 if r["labels"][head]["value"] == "TRUE" else 0.0 for r in ep["rows"]], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor([bool(r["labels"][head]["mask"]) for r in ep["rows"]], dtype=torch.bool, device=device).unsqueeze(0)
                if bool(mask.any()):
                    terms.append(torch.nn.functional.binary_cross_entropy_with_logits(outputs[head][mask], values[mask]))
            if not terms:
                continue
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["student_contract"]["gradient_clip"]))
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)) if losses else None, "episodes": len(losses)})
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or output_root.exists():
        raise RuntimeError(f"output already exists: {output_root}")
    staging.mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict(), "input_dim": 25, "hidden": int(config["student_contract"]["hidden"]), "short_rf": int(config["student_contract"]["short_rf"]), "long_rf": int(config["student_contract"]["long_rf"]), "heads": list(HEADS), "variant": variant, "seed": seed}, staging / "checkpoint.pt")
    manifest = {"schema": "FRESH40_V5_STUDENT_DEVELOPMENT_CHECKPOINT_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "dataset_root": str(dataset_root), "dataset_manifest_sha256": sha256_file(dataset_root / "MANIFEST.json"), "variant": variant, "active_heads": list(active), "checkpoint_sha256": sha256_file(staging / "checkpoint.pt"), "train_identity_sha256": json.loads((dataset_root / "MANIFEST.json").read_text())["train_identity_sha256"], "coverage": coverage, "history": history, "formal_training_authorized": False, "formal_inference_authorized": False, "attack_authorized": False, "protected_reads": False}
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (staging / "normalizer.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(), "dataset_manifest_sha256": manifest["dataset_manifest_sha256"]}, indent=2, sort_keys=True))
    seal = _seal(staging); _publish(staging, output_root)
    return {"output_root": str(output_root), "variant": variant, "active_heads": list(active), "checkpoint_sha256": manifest["checkpoint_sha256"], "sha256sums_sha256": seal["sha256sums_sha256"], "history": history, "coverage": coverage}


def shadow_student(dataset_root: Path, checkpoint_root: Path, output_root: Path, config: Mapping[str, Any], variant: str) -> dict[str, Any]:
    import torch
    from n5_student_model import N5MultiHeadStudent

    if output_root.exists():
        raise RuntimeError(f"output already exists: {output_root}")
    checkpoint = torch.load(checkpoint_root / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = N5MultiHeadStudent(input_dim=25, hidden=int(checkpoint["hidden"]), short_rf=int(checkpoint["short_rf"]), long_rf=int(checkpoint["long_rf"]), dropout=0.1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    dev_eps, _, _ = _load_student_data(dataset_root, "dev")
    pred_rows, events, event_rows = [], [], []
    counts = {"steps": 0, "emits": 0, "critical_known": 0, "critical_true": 0, "critical_correct": 0, "critical_fp": 0, "unknown_emits": 0, "safe_true_emits": 0, "instability_true_emits": 0}
    head_scores = {h: [] for h in HEADS}; head_values = {h: [] for h in HEADS}
    known_critical_true_total = known_critical_true_candidate = 0
    with torch.no_grad():
        for ep in dev_eps:
            x = torch.from_numpy(ep["features"]).unsqueeze(0)
            probs = {h: torch.sigmoid(v).squeeze(0).cpu().numpy() for h, v in model(x, timestep_mask=torch.ones((1, x.shape[1]), dtype=torch.bool)).items()}
            prior_candidate = False; event_id = -1; emitted_event = False; current_event = []
            for idx, row in enumerate(ep["rows"]):
                candidate = bool(row["candidate_close"])
                if candidate and not prior_candidate:
                    if current_event:
                        event_rows.append(current_event)
                    current_event = []
                    event_id += 1; emitted_event = False
                if not candidate:
                    if current_event:
                        event_rows.append(current_event); current_event = []
                    emitted_event = False
                probability_row = {h: float(probs[h][idx]) for h in HEADS}
                critical = probability_row["physical_criticality"]
                emit = bool(not emitted_event and variant_decision(variant, candidate, probability_row))
                if emit:
                    emitted_event = True; counts["emits"] += 1; events.append({"episode_id": ep["identity"], "step": idx, "event_id": event_id, "critical_probability": critical, "safe_release_probability": probability_row["safe_release"], "instability_probability": probability_row["instability"]})
                    if row["labels"]["physical_criticality"]["value"] == "UNKNOWN": counts["unknown_emits"] += 1
                    if row["labels"]["safe_release"]["value"] == "TRUE": counts["safe_true_emits"] += 1
                    if row["labels"]["instability"]["value"] == "TRUE": counts["instability_true_emits"] += 1
                target = row["labels"]["physical_criticality"]
                current_event.append({"step": idx, "emit": emit, "target": target}) if candidate else None
                if target["mask"]:
                    counts["critical_known"] += 1; counts["critical_true"] += int(target["value"] == "TRUE")
                    if emit and target["value"] == "TRUE": counts["critical_correct"] += 1
                    if emit and target["value"] == "FALSE": counts["critical_fp"] += 1
                for h in HEADS:
                    head = row["labels"][h]
                    if head["mask"]:
                        head_scores[h].append(probability_row[h]); head_values[h].append(head["value"] == "TRUE")
                known_critical_true_total += int(target["mask"] and target["value"] == "TRUE")
                known_critical_true_candidate += int(target["mask"] and target["value"] == "TRUE" and candidate)
                pred_rows.append({"episode_id": ep["identity"], "step": idx, "candidate_close": candidate, "probabilities": probability_row, "active_heads": list(VARIANT_ACTIVE_HEADS[variant]), "emit": emit, "event_id": event_id, "action_mutation": False})
                counts["steps"] += 1; prior_candidate = candidate
            if current_event:
                event_rows.append(current_event)
    positive_events = negative_events = predicted_positive_events = predicted_negative_events = 0; latencies = []
    for event in event_rows:
        known = [x["target"] for x in event if x["target"]["mask"]]
        if not known or any(x["target"]["value"] == "UNKNOWN" for x in event):
            continue
        has_true = any(x["value"] == "TRUE" for x in known)
        predicted = any(x["emit"] for x in event)
        if has_true:
            positive_events += 1; predicted_positive_events += int(predicted)
            if predicted:
                latencies.append(next(x["step"] for x in event if x["emit"]) - event[0]["step"])
        else:
            negative_events += 1; predicted_negative_events += int(predicted)
    head_report = {h: _head_metrics(head_scores[h], head_values[h]) for h in HEADS}
    event_report = {"positive_events": positive_events, "predicted_positive_events": predicted_positive_events, "positive_event_recall": predicted_positive_events / positive_events if positive_events else None, "known_negative_events": negative_events, "predicted_negative_events": predicted_negative_events, "known_negative_event_fp": predicted_negative_events / negative_events if negative_events else None, "mean_emit_latency_steps": float(np.mean(latencies)) if latencies else None, "emitted_positive_latency_samples": len(latencies)}
    metrics = {"schema": "FRESH40_V5_CAUSAL_SHADOW_V2", "status": "DEVELOPMENT_NONCONSUMABLE", "variant": variant, "active_heads": list(VARIANT_ACTIVE_HEADS[variant]), "variant_equation": "candidate AND active head predicates; one emit per candidate event", "checkpoint_sha256": sha256_file(checkpoint_root / "checkpoint.pt"), "dataset_manifest_sha256": sha256_file(dataset_root / "MANIFEST.json"), "counts": counts, "candidate_gate_ceiling": {"known_critical_true_steps": known_critical_true_total, "known_critical_true_candidate_steps": known_critical_true_candidate, "recall_ceiling": known_critical_true_candidate / known_critical_true_total if known_critical_true_total else None}, "head_metrics": head_report, "event_metrics": event_report, "unknown_label_handling": {"unknown_emit_count": counts["unknown_emits"], "unknown_labels_masked_from_metrics": True, "unknown_as_negative": False}, "action_mutation": False, "protected_reads": False, "attack_enabled": False}
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"; staging.mkdir(parents=True)
    (staging / "prediction_records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in pred_rows))
    (staging / "event_records.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in events))
    (staging / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (staging / "MANIFEST.json").write_text(json.dumps({"schema": "FRESH40_V5_CAUSAL_SHADOW_BUNDLE_V2", "status": "DEVELOPMENT_NONCONSUMABLE", "variant": variant, "active_heads": list(VARIANT_ACTIVE_HEADS[variant]), "dataset_manifest_sha256": metrics["dataset_manifest_sha256"], "checkpoint_sha256": metrics["checkpoint_sha256"], "prediction_count": len(pred_rows), "event_count": len(events), "action_mutation": False, "protected_reads": False, "attack_enabled": False}, indent=2, sort_keys=True))
    seal = _seal(staging); _publish(staging, output_root)
    return {"output_root": str(output_root), "sha256sums_sha256": seal["sha256sums_sha256"], **metrics}


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
        if not args.dataset_root: raise SystemExit("--dataset-root required")
        result = train_student(args.dataset_root.resolve(), args.output_root.resolve(), cfg, args.variant, args.epochs, args.device)
    elif args.command == "shadow":
        if not args.dataset_root or not args.checkpoint: raise SystemExit("--dataset-root and --checkpoint required")
        result = shadow_student(args.dataset_root.resolve(), args.checkpoint.resolve(), args.output_root.resolve(), cfg, args.variant)
    else:
        if not args.dataset_root or not args.checkpoint: raise SystemExit("--dataset-root and --checkpoint required")
        result = shadow_student(args.dataset_root.resolve(), args.checkpoint.resolve(), args.output_root.resolve(), cfg, args.variant)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
