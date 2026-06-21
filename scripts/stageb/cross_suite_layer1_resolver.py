#!/usr/bin/env python3
"""CPU-only Layer 1 resolver utilities for Cross-Suite CLEAN300.

This module intentionally stays offline. It reads CLEAN artifacts and ontology
metadata, never loads OpenVLA, never opens detector telemetry, and never uses
Layer 2 emit fields as Teacher inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]

ONTOLOGY_VERSION = "cross_suite_task_ontology_v1"
TEACHER_VERSION = "cross_suite_teacher_v1"
RESOLVER_VERSION = "cross_suite_resolver_v1"
PHYSICS_VERSION = "cross_suite_teacher_physics_v1"
PRIMARY_MECHANISM = "single_object_pick_place"
SUPPLEMENTARY_MECHANISMS = {"multi_object_transfer", "mixed_articulated_pick_place"}
VALID_BINDING_STATUSES = {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY", "BOUND_STRUCTURED_FALLBACK"}
REGION_TARGET_ALIASES = {
    "back_compartment_of_caddy",
    "bottom_drawer",
    "cabinet_inside",
    "cabinet_top",
    "front_of_stove",
    "left_plate",
    "right_of_plate",
    "right_plate",
    "stove_front",
    "top_drawer",
}
ROOT_REGISTRY = REPO / "evidence" / "manifests" / "cross_suite_clean300_root_registry.json"
PHYSICS_CONFIG = REPO / "configs" / "cross_suite_teacher_physics_v1.yaml"
RESOLVER_NOT_IMPLEMENTED = "RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM"
FORBIDDEN_INPUT_FIELDS = {
    "mlp_emit_step",
    "mlp_triggered",
    "mlp_emit",
    "corridor_p",
    "release_p",
    "pred_phase",
    "detector_state",
    "emitted",
    "emit_step",
}
ALLOWED_STEP_FIELDS = {
    "step",
    "raw_gripper",
    "env_gripper",
    "gripper_qpos",
    "gripper_opening_proxy",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
    "action_vector_json",
    "env_action_json",
    "exact_new_tokens_json",
    "gripper_token",
    "feat_valid",
    "feat_error",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(parts: Iterable[Any]) -> str:
    text = "|".join(str(p) for p in parts)
    return sha256_bytes(text.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


@dataclass(frozen=True)
class OntologyTask:
    suite: str
    task_idx: int
    task_name: str
    instruction: str
    mechanism_type: str
    teacher_applicable: str
    expected_event_count_class: str
    manipulated_object_aliases: tuple[str, ...]
    target_aliases: tuple[str, ...]
    articulated_body_aliases: tuple[str, ...]


@dataclass(frozen=True)
class BindingResult:
    name: str
    index: int
    status: str
    source: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalEvent:
    status: str
    close_onset_step: int | str
    grasp_established_step: int | str
    lift_onset_step: int | str
    stable_carry_start: int | str
    teacher_window_start: int | str
    teacher_anchor_step: int | str
    teacher_window_end: int | str
    release_onset_step: int | str
    target_proximity_step: int | str
    object_gripper_separation_step: int | str
    placement_complete: bool
    object_gripper_min_distance: float | str
    object_target_min_distance: float | str
    event_valid: bool
    event_invalid_reason: str


def load_ontology(path: Path) -> dict[tuple[str, int], OntologyTask]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks: dict[tuple[str, int], OntologyTask] = {}
    for row in data["tasks"]:
        task = OntologyTask(
            suite=str(row["suite"]),
            task_idx=int(row["task_idx"]),
            task_name=str(row["task_name"]),
            instruction=str(row["instruction"]),
            mechanism_type=str(row["mechanism_type"]),
            teacher_applicable=str(row["teacher_applicable"]),
            expected_event_count_class=str(row["expected_event_count_class"]),
            manipulated_object_aliases=tuple(str(x) for x in (row.get("manipulated_object_aliases") or [])),
            target_aliases=tuple(str(x) for x in (row.get("target_aliases") or [])),
            articulated_body_aliases=tuple(str(x) for x in (row.get("articulated_body_aliases") or [])),
        )
        tasks[(task.suite, task.task_idx)] = task
    return tasks


def load_physics_config(path: Path = PHYSICS_CONFIG) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data)


def canonical_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row["suite"]),
            str(int(row["task_idx"])),
            str(int(row["state_id"])),
            str(int(row.get("eval_seed", 0))),
            str(row.get("condition", "CLEAN")),
        ]
    )


def classify_mechanism(task: OntologyTask) -> str:
    if task.mechanism_type == PRIMARY_MECHANISM:
        return "primary_single_object"
    if task.mechanism_type in SUPPLEMENTARY_MECHANISMS:
        return "supplementary_event_audit"
    return "semantic_abstain"


def select_manifest_rows(
    ledger_rows: list[dict[str, str]],
    ontology: dict[tuple[str, int], OntologyTask],
    *,
    exclude_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exclude_keys = exclude_keys or set()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ledger_rows:
        if row.get("condition") != "CLEAN" or row.get("status") != "COMPLETE_VALID":
            rejected.append({"canonical_key": canonical_key(row), "reason": "not_complete_clean"})
            continue
        key = (row.get("suite", ""), int(row.get("task_idx", -1)))
        task = ontology.get(key)
        if task is None:
            rejected.append({"canonical_key": canonical_key(row), "reason": "ontology_missing"})
            continue
        ck = canonical_key(row)
        if ck in exclude_keys:
            rejected.append({"canonical_key": ck, "reason": "excluded"})
            continue
        source_sha = row.get("artifact_recursive_sha256") or row.get("source_episode_sha") or ""
        candidates.append(
            {
                "canonical_key": ck,
                "suite": task.suite,
                "task_idx": task.task_idx,
                "state_id": int(row["state_id"]),
                "eval_seed": int(row.get("eval_seed", 0)),
                "condition": "CLEAN",
                "episode_path": row["episode_path"],
                "source_episode_sha": source_sha,
                "task_success": str(row.get("task_success", "")),
                "n_steps": int(float(row.get("n_steps") or 0)),
                "mechanism_type": task.mechanism_type,
                "teacher_applicable": task.teacher_applicable,
                "mechanism_group": classify_mechanism(task),
                "selection_hash": stable_hash([task.suite, task.task_idx, row["state_id"], row.get("eval_seed", 0), source_sha]),
            }
        )
    return candidates, rejected


def deterministic_pick(candidates: list[dict[str, Any]], predicates: list[tuple[str, int]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for label, count in predicates:
        pool = [
            r
            for r in candidates
            if r["canonical_key"] not in selected_keys and _matches_selection_label(r, label)
        ]
        pool.sort(key=lambda r: (r["selection_hash"], r["canonical_key"]))
        for row in pool[:count]:
            selected_keys.add(row["canonical_key"])
            out = dict(row)
            out["selection_bucket"] = label
            out["selection_rank"] = len(selected)
            selected.append(out)
    return selected


def _matches_selection_label(row: dict[str, Any], label: str) -> bool:
    suite = row["suite"]
    group = row["mechanism_group"]
    success = str(row.get("task_success", "")).lower() == "true"
    if label == "spatial_primary":
        return suite == "libero_spatial" and group == "primary_single_object"
    if label == "goal_primary_success":
        return suite == "libero_goal" and group == "primary_single_object" and success
    if label == "goal_negative_or_abstain":
        return suite == "libero_goal" and group == "semantic_abstain"
    if label == "libero10_single_event":
        return suite == "libero_10" and group == "primary_single_object"
    if label == "libero10_multi_or_mixed":
        return suite == "libero_10" and group == "supplementary_event_audit"
    return False


def build_dev_canary_manifest(ledger_rows: list[dict[str, str]], ontology: dict[tuple[str, int], OntologyTask]) -> dict[str, Any]:
    candidates, rejected = select_manifest_rows(ledger_rows, ontology)
    selected = deterministic_pick(
        candidates,
        [
            ("spatial_primary", 4),
            ("goal_primary_success", 2),
            ("goal_negative_or_abstain", 2),
            ("libero10_single_event", 2),
            ("libero10_multi_or_mixed", 2),
        ],
    )
    return {
        "manifest_type": "cross_suite_layer1_dev_canary_manifest_v1",
        "selection_rule": "stable sha256 over suite|task_idx|state_id|eval_seed|source_episode_sha within preregistered buckets",
        "source_commit": git_commit(),
        "selected_count": len(selected),
        "expected_count": 12,
        "selected": selected,
        "rejected_count": len(rejected),
    }


def build_blind_review_manifest(
    ledger_rows: list[dict[str, str]],
    ontology: dict[tuple[str, int], OntologyTask],
    *,
    exclude_keys: set[str],
    count: int = 24,
) -> dict[str, Any]:
    candidates, rejected = select_manifest_rows(ledger_rows, ontology, exclude_keys=exclude_keys)
    candidates.sort(key=lambda r: (r["selection_hash"], r["canonical_key"]))
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        success = "success" if str(row.get("task_success", "")).lower() == "true" else "failure"
        strata.setdefault((row["suite"], row["mechanism_group"], success), []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in sorted(strata):
            if len(selected) >= count:
                break
            if strata[key]:
                row = dict(strata[key].pop(0))
                row["selection_bucket"] = "|".join(key)
                row["selection_rank"] = len(selected)
                selected.append(row)
                progressed = True
        if not progressed:
            break
    return {
        "manifest_type": "resolver_diagnostic_holdout_v1",
        "holdout_role": "diagnostic_only_observed_during_resolver_development_not_final_unbiased_validation",
        "selection_rule": "disjoint deterministic round-robin over suite|mechanism_group|task_success strata",
        "source_commit": git_commit(),
        "selected_count": len(selected),
        "requested_count": count,
        "excluded_key_count": len(exclude_keys),
        "selected": selected,
        "rejected_count": len(rejected),
    }


def normalize_entity_name(name: str) -> str:
    low = name.lower()
    for suffix in ["_main", "_default_site", "_contain_region", "_init_region", "_joint0"]:
        if low.endswith(suffix):
            low = low[: -len(suffix)]
    parts = low.split("_")
    if len(parts) > 2 and parts[-1].isdigit():
        low = "_".join(parts[:-1])
    return low


def bind_unique(names: list[str], aliases: tuple[str, ...]) -> BindingResult:
    """Bind by exact or structured names only.

    This deliberately rejects final-positive arbitrary substring matching. If
    more than one candidate remains, the binding is ambiguous.
    """

    candidates: list[tuple[str, int, str, str]] = []
    alias_set = {a.lower() for a in aliases}
    for idx, name in enumerate(names):
        low = name.lower()
        norm = normalize_entity_name(name)
        for alias in alias_set:
            if low == alias:
                candidates.append((name, idx, "BOUND_EXACT", alias))
            elif norm == alias:
                candidates.append((name, idx, "BOUND_BDDL_ONTOLOGY", alias))
            elif (
                (low.startswith(alias + "_") or norm.endswith("_" + alias))
                and normalize_entity_name(alias) == alias
            ):
                candidates.append((name, idx, "BOUND_STRUCTURED_FALLBACK", alias))
    unique: dict[str, tuple[str, int, str, str]] = {c[0]: c for c in candidates}
    vals = list(unique.values())
    if len(vals) == 1:
        name, idx, status, source = vals[0]
        return BindingResult(name=name, index=idx, status=status, source=source, candidates=tuple(unique))
    if len(vals) > 1:
        return BindingResult(name="", index=-1, status="AMBIGUOUS", source="multiple_structured_candidates", candidates=tuple(sorted(unique)))
    return BindingResult(name="", index=-1, status="FAILED", source="no_structured_candidate", candidates=())


def bind_candidates(names: list[str], aliases: tuple[str, ...]) -> list[BindingResult]:
    out: list[BindingResult] = []
    seen: set[str] = set()
    for idx, name in enumerate(names):
        result = bind_unique([name], aliases)
        if result.status in VALID_BINDING_STATUSES and name not in seen:
            out.append(BindingResult(name=name, index=idx, status=result.status, source=result.source, candidates=(name,)))
            seen.add(name)
    return out


def bind_many(names: list[str], aliases: tuple[str, ...]) -> list[BindingResult]:
    specific_aliases = [a for a in aliases if any(ch.isdigit() for ch in a)]
    use_aliases = specific_aliases or list(aliases)
    results: list[BindingResult] = []
    seen: set[str] = set()
    for alias in use_aliases:
        result = bind_unique(names, (alias,))
        if result.status in VALID_BINDING_STATUSES and result.name not in seen:
            results.append(result)
            seen.add(result.name)
        elif result.status == "AMBIGUOUS":
            results.append(result)
    return results


def region_specific_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(alias for alias in aliases if alias in REGION_TARGET_ALIASES)


def bind_target(site_names: list[str], body_names: list[str], aliases: tuple[str, ...]) -> tuple[BindingResult, str]:
    region_aliases = region_specific_aliases(aliases)
    if region_aliases:
        site_result = bind_unique(site_names, region_aliases)
        if site_result.status in VALID_BINDING_STATUSES or site_result.status == "AMBIGUOUS":
            return site_result, "site"
        body_result = bind_unique(body_names, region_aliases)
        if body_result.status in VALID_BINDING_STATUSES or body_result.status == "AMBIGUOUS":
            return body_result, "body"
        return BindingResult("", -1, "FAILED", "region_specific_target_missing:" + "|".join(region_aliases), ()), "site"
    site_result = bind_unique(site_names, aliases)
    if site_result.status in VALID_BINDING_STATUSES or site_result.status == "AMBIGUOUS":
        return site_result, "site"
    body_result = bind_unique(body_names, aliases)
    return body_result, "body"


def load_step_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
        for raw in reader:
            row = {k: raw.get(k, "") for k in ALLOWED_STEP_FIELDS if k in raw}
            rows.append(row)
    return rows


def load_episode_context(episode_path: Path) -> dict[str, Any]:
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)
    manifest = read_json(episode_path / "episode_manifest.json")
    summary = read_json(episode_path / "episode_summary.json")
    sim_manifest = read_json(episode_path / "sim_state_manifest.json")
    step_rows = load_step_rows(episode_path / "step_telemetry.csv")
    with np.load(episode_path / "sim_state_stream.npz", allow_pickle=False) as npz:
        sim_arrays = {k: np.asarray(npz[k]) for k in npz.files}
    return {
        "episode_path": episode_path,
        "manifest": manifest,
        "summary": summary,
        "sim_manifest": sim_manifest,
        "sim_arrays": sim_arrays,
        "step_rows": step_rows,
    }


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def find_first_close_onset(step_rows: list[dict[str, Any]]) -> int | None:
    onsets = find_close_onsets(step_rows)
    return onsets[0] if onsets else None


def find_close_onsets(step_rows: list[dict[str, Any]]) -> list[int]:
    onsets: list[int] = []
    prev_close = False
    for row in step_rows:
        step = int(float(row.get("step", -1)))
        env_grip = _float(row, "env_gripper")
        raw_grip = _float(row, "raw_gripper")
        is_close = (math.isfinite(env_grip) and env_grip > 0) or (math.isfinite(raw_grip) and raw_grip < 0.5)
        if is_close and not prev_close:
            onsets.append(step)
        prev_close = is_close
    return onsets


def resolve_gripper_site(site_names: list[str]) -> BindingResult:
    candidates = [idx for idx, name in enumerate(site_names) if name == "gripper0_grip_site"]
    if len(candidates) == 1:
        idx = candidates[0]
        return BindingResult(site_names[idx], idx, "BOUND_EXACT", "gripper0_grip_site", (site_names[idx],))
    if len(candidates) > 1:
        return BindingResult("", -1, "AMBIGUOUS", "multiple_gripper_sites", tuple(site_names[idx] for idx in candidates))
    return BindingResult("", -1, "FAILED", "missing_gripper0_grip_site", ())


def propose_timing(step_rows: list[dict[str, Any]], n_steps: int) -> dict[str, int | str]:
    close = find_first_close_onset(step_rows)
    if close is None:
        return {
            "close_onset_step": "",
            "grasp_established_step": "",
            "lift_onset_step": "",
            "stable_carry_start": "",
            "teacher_window_start": "",
            "teacher_anchor_step": "",
            "teacher_window_end": "",
            "release_onset_step": "",
            "timing_status": "NO_RELEVANT_CLOSE_ONSET",
        }
    start = max(0, close - 2)
    end = min(max(n_steps - 1, 0), close + 8)
    return {
        "close_onset_step": close,
        "grasp_established_step": close,
        "lift_onset_step": min(close + 1, max(n_steps - 1, 0)),
        "stable_carry_start": min(close + 2, max(n_steps - 1, 0)),
        "teacher_window_start": start,
        "teacher_anchor_step": close,
        "teacher_window_end": end,
        "release_onset_step": "",
        "timing_status": "HEURISTIC_CLOSE_ONSET_WINDOW",
    }


def first_index(mask: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(mask[max(start, 0) :])
    if len(idx) == 0:
        return None
    return int(idx[0] + max(start, 0))


def contiguous_start(mask: np.ndarray, start: int = 0, length: int = 3) -> int | None:
    arr = np.asarray(mask, dtype=bool)
    for idx in range(max(start, 0), max(len(arr) - length + 1, 0)):
        if bool(arr[idx : idx + length].all()):
            return idx
    return None


def _gripper_closed_mask(step_rows: list[dict[str, Any]], n: int) -> np.ndarray:
    closed = np.zeros(n, dtype=bool)
    for idx, row in enumerate(step_rows[:n]):
        env_grip = _float(row, "env_gripper")
        raw_grip = _float(row, "raw_gripper")
        closed[idx] = (math.isfinite(env_grip) and env_grip > 0) or (math.isfinite(raw_grip) and raw_grip < 0.5)
    return closed


def _attempt_end_from_close(closed: np.ndarray, close_idx: int) -> int:
    """Return exclusive attempt end for one uninterrupted close segment."""
    n = len(closed)
    for idx in range(close_idx + 1, n):
        if not bool(closed[idx]):
            return idx
    return n


def _coupled_motion_mask(obj: np.ndarray, grip: np.ndarray, max_delta_m: float) -> np.ndarray:
    if len(obj) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(obj), dtype=bool)
    if len(obj) > 1:
        obj_delta = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        grip_delta = np.linalg.norm(np.diff(grip, axis=0), axis=1)
        mask[1:] = np.abs(obj_delta - grip_delta) <= max_delta_m
    return mask


def _orientation_jump_mask(body_xquat: np.ndarray | None, object_index: int, n: int, max_jump: float) -> np.ndarray:
    if body_xquat is None or object_index < 0 or body_xquat.ndim != 3 or body_xquat.shape[0] < n:
        return np.ones(n, dtype=bool)
    quat = np.asarray(body_xquat[:n, object_index, :], dtype=float)
    if quat.shape[1] != 4:
        return np.ones(n, dtype=bool)
    norm = np.linalg.norm(quat, axis=1)
    if not np.isfinite(norm).all() or float(np.nanmax(norm)) <= 0:
        return np.ones(n, dtype=bool)
    quat = quat / np.maximum(norm[:, None], 1e-9)
    jumps = np.zeros(n, dtype=float)
    if n > 1:
        # q and -q represent the same orientation; use absolute dot product.
        dots = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
        jumps[1:] = 1.0 - np.clip(dots, 0.0, 1.0)
    return jumps <= max_jump


def detect_physical_event(
    *,
    step_rows: list[dict[str, Any]],
    sim_arrays: dict[str, np.ndarray],
    site_names: list[str],
    object_binding: BindingResult,
    target_binding: BindingResult,
    target_kind: str,
    physics: dict[str, Any] | None = None,
) -> PhysicalEvent:
    cfg = physics or load_physics_config()
    thresholds = cfg.get("thresholds", {})
    object_gripper_near = float(thresholds.get("object_gripper_near_m", 0.12))
    object_gripper_separated = float(thresholds.get("object_gripper_separated_m", 0.18))
    object_lift_delta = float(thresholds.get("object_lift_delta_m", 0.025))
    stable_carry_min = int(thresholds.get("stable_carry_min_frames", 3))
    grasp_min = int(thresholds.get("grasp_min_frames", 2))
    object_target_near = float(thresholds.get("object_target_near_m", 0.14))
    motion_coupling_max_delta = float(thresholds.get("motion_coupling_max_delta_m", 0.06))
    orientation_jump_max = float(thresholds.get("orientation_jump_max", 0.25))
    close_candidates = find_close_onsets(step_rows)
    if not close_candidates:
        return PhysicalEvent("NO_CLOSE_ONSET", "", "", "", "", "", "", "", "", "", "", False, "", "", False, "no_close_onset")
    body_xpos = sim_arrays.get("body_xpos")
    site_xpos = sim_arrays.get("site_xpos")
    body_xquat = sim_arrays.get("body_xquat")
    if body_xpos is None or site_xpos is None:
        return PhysicalEvent("SIM_ARRAY_MISSING", close_candidates[0], "", "", "", "", "", "", "", "", "", False, "", "", False, "required_sim_arrays_missing")
    if object_binding.index < 0:
        return PhysicalEvent("BINDING_INVALID", close_candidates[0], "", "", "", "", "", "", "", "", "", False, "", "", False, "object_binding_invalid")
    gripper_binding = resolve_gripper_site(site_names)
    if gripper_binding.index < 0:
        return PhysicalEvent("GRIPPER_SITE_INVALID", close_candidates[0], "", "", "", "", "", "", "", "", "", False, "", "", False, gripper_binding.source)
    obj = np.asarray(body_xpos[:, object_binding.index, :], dtype=float)
    grip = np.asarray(site_xpos[:, gripper_binding.index, :], dtype=float)
    target = None
    target_valid = target_binding.index >= 0
    if target_valid and target_kind == "site":
        target = np.asarray(site_xpos[:, target_binding.index, :], dtype=float)
    elif target_valid:
        target = np.asarray(body_xpos[:, target_binding.index, :], dtype=float)
    n = min(len(obj), len(grip), len(target) if target is not None else len(obj), len(step_rows))
    obj, grip = obj[:n], grip[:n]
    if target is not None:
        target = target[:n]
    if n == 0:
        return PhysicalEvent("EMPTY_TRAJECTORY", close_candidates[0], "", "", "", "", "", "", "", "", "", False, "", "", False, "empty_trajectory")
    obj_grip = np.linalg.norm(obj - grip, axis=1)
    obj_target = np.linalg.norm(obj - target, axis=1) if target is not None else None
    near_grip = obj_grip < object_gripper_near
    lift_mask = obj[:, 2] > (obj[0, 2] + object_lift_delta)
    closed = _gripper_closed_mask(step_rows, n)
    coupled_motion = _coupled_motion_mask(obj, grip, motion_coupling_max_delta)
    orientation_ok = _orientation_jump_mask(body_xquat, object_binding.index, n, orientation_jump_max)
    carry_evidence_mask = near_grip & lift_mask & closed & coupled_motion & orientation_ok
    target_step_by_close: int | None = None
    best_incomplete: PhysicalEvent | None = None
    for close in close_candidates:
        close_idx = min(int(close), n - 1)
        attempt_end = _attempt_end_from_close(closed, close_idx)
        attempt_mask = np.zeros(n, dtype=bool)
        attempt_mask[close_idx:attempt_end] = True
        if attempt_end <= close_idx + 1:
            grasp = None
            lift = None
            carry = None
        else:
            # Require causal separation: close_onset < grasp < lift <= stable_carry.
            grasp_mask = near_grip & closed & attempt_mask
            grasp = contiguous_start(grasp_mask, close_idx + 1, grasp_min)
            lift = first_index(lift_mask & closed & attempt_mask, (grasp + 1) if grasp is not None else close_idx + 1)
            carry = contiguous_start(
                carry_evidence_mask & attempt_mask,
                lift if lift is not None else close_idx + 1,
                stable_carry_min,
            )
        target_step = None
        if obj_target is not None:
            target_step = first_index(obj_target < object_target_near, carry if carry is not None else close_idx)
            target_step_by_close = target_step_by_close if target_step_by_close is not None else target_step
        release = None
        separation = None
        if carry is not None:
            for row in step_rows[carry:n]:
                step = int(float(row.get("step", -1)))
                env_grip = _float(row, "env_gripper")
                raw_grip = _float(row, "raw_gripper")
                is_open = (math.isfinite(env_grip) and env_grip < 0) or (math.isfinite(raw_grip) and raw_grip > 0.5)
                if is_open:
                    release = step
                    break
            sep = first_index(obj_grip > object_gripper_separated, carry)
            separation = sep
        valid = all(x is not None for x in [grasp, lift, carry]) and int(close) < int(grasp) < int(lift) <= int(carry)
        invalid = []
        if grasp is None:
            invalid.append("no_grasp_proximity_after_close")
        if lift is None:
            invalid.append("no_object_lift")
        if carry is None:
            invalid.append("no_stable_carry")
        if all(x is not None for x in [grasp, lift, carry]) and not (int(close) < int(grasp) < int(lift) <= int(carry)):
            invalid.append("phase_order_violation")
        if carry is not None and not bool((coupled_motion & orientation_ok)[int(carry)]):
            invalid.append("collision_or_orientation_jump_not_carry")
        placement_complete = bool(target_step is not None and release is not None and release >= target_step)
        start = max(0, int(grasp if grasp is not None else close_idx) - 2)
        anchor = int(grasp if grasp is not None else close_idx)
        end_candidates = [x for x in [release, separation, target_step, n - 1] if x is not None]
        end = int(min(end_candidates)) if end_candidates else min(n - 1, close_idx + 3)
        candidate = PhysicalEvent(
            status="PHYSICAL_EVENT_VALID" if valid else "PHYSICAL_EVENT_INCOMPLETE",
            close_onset_step=int(close),
            grasp_established_step="" if grasp is None else int(grasp),
            lift_onset_step="" if lift is None else int(lift),
            stable_carry_start="" if carry is None else int(carry),
            teacher_window_start=start if valid else "",
            teacher_anchor_step=anchor if valid else "",
            teacher_window_end=end if valid else "",
            release_onset_step="" if release is None else int(release),
            target_proximity_step="" if target_step is None else int(target_step),
            object_gripper_separation_step="" if separation is None else int(separation),
            placement_complete=placement_complete,
            object_gripper_min_distance=float(np.nanmin(obj_grip)),
            object_target_min_distance="" if obj_target is None else float(np.nanmin(obj_target)),
            event_valid=bool(valid),
            event_invalid_reason="|".join(invalid),
        )
        if valid:
            return candidate
        if best_incomplete is None or len(str(candidate.event_invalid_reason).split("|")) < len(str(best_incomplete.event_invalid_reason).split("|")):
            best_incomplete = candidate
    if best_incomplete is not None:
        return best_incomplete
    return PhysicalEvent(
        "PHYSICAL_EVENT_INCOMPLETE",
        close_candidates[0],
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "" if target_step_by_close is None else int(target_step_by_close),
        "",
        False,
        float(np.nanmin(obj_grip)),
        "" if obj_target is None else float(np.nanmin(obj_target)),
        False,
        "no_candidate_close_supported_physical_event",
    )


def event_to_row(
    *,
    episode_key: str,
    event_idx: int,
    object_binding: BindingResult,
    target_binding: BindingResult,
    event: PhysicalEvent,
    supplementary: bool = False,
) -> dict[str, Any]:
    return {
        "episode_key": episode_key,
        "event_id": f"{episode_key}|event{event_idx}",
        "object_body_name": object_binding.name,
        "object_joint_name": "",
        "target_body_or_site_name": target_binding.name,
        "binding_source": f"{object_binding.source}->{target_binding.source}",
        "binding_confidence_class": "high" if object_binding.status == "BOUND_EXACT" and target_binding.status == "BOUND_EXACT" else "medium",
        "close_onset_step": event.close_onset_step,
        "grasp_established_step": event.grasp_established_step,
        "lift_onset_step": event.lift_onset_step,
        "stable_carry_start": event.stable_carry_start,
        "teacher_window_start": event.teacher_window_start,
        "teacher_anchor_step": event.teacher_anchor_step,
        "teacher_window_end": event.teacher_window_end,
        "release_onset_step": event.release_onset_step,
        "target_proximity_step": event.target_proximity_step,
        "object_gripper_separation_step": event.object_gripper_separation_step,
        "placement_complete": bool(event.placement_complete),
        "object_gripper_min_distance": event.object_gripper_min_distance,
        "object_target_min_distance": event.object_target_min_distance,
        "timing_status": event.status,
        "event_valid": bool(event.event_valid),
        "event_invalid_reason": event.event_invalid_reason,
        "supplementary_event": bool(supplementary),
    }


def source_episode_relpath(episode_path: Path) -> tuple[str, str]:
    abspath = episode_path.resolve()
    roots = []
    if ROOT_REGISTRY.exists():
        try:
            registry = read_json(ROOT_REGISTRY)
            roots = [Path(r["path"]) for r in registry.get("official_roots", [])]
        except Exception:
            roots = []
    for root in roots:
        try:
            return str(abspath.relative_to(root)), str(abspath)
        except ValueError:
            continue
    return episode_path.name, str(abspath)


def resolve_episode(row: dict[str, Any], task: OntologyTask, *, teacher_run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_path = Path(row["episode_path"])
    ctx = load_episode_context(episode_path)
    sim_meta = ctx["sim_manifest"].get("metadata", {})
    body_names = [str(x) for x in sim_meta.get("body_names", [])]
    site_names = [str(x) for x in sim_meta.get("site_names", [])]
    source_sha = str(row.get("source_episode_sha") or row.get("artifact_recursive_sha256") or "")
    episode_key = str(row.get("canonical_key") or canonical_key(row))
    n_steps = int(ctx["summary"].get("n_steps") or row.get("n_steps") or len(ctx["step_rows"]))
    relpath, abspath_audit = source_episode_relpath(episode_path)

    base = {
        "teacher_executed": True,
        "teacher_run_id": teacher_run_id,
        "teacher_version": TEACHER_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "resolver_version": RESOLVER_VERSION,
        "episode_key": episode_key,
        "suite": task.suite,
        "task_idx": task.task_idx,
        "state_id": int(row["state_id"]),
        "source_episode_relpath": relpath,
        "source_episode_abspath_audit_only": abspath_audit,
        "source_episode_sha": source_sha,
        "mechanism_type": task.mechanism_type,
    }

    if task.mechanism_type == PRIMARY_MECHANISM:
        object_candidates = bind_candidates(body_names, task.manipulated_object_aliases)
        obj_binding = bind_unique(body_names, task.manipulated_object_aliases)
        target_binding, target_kind = bind_target(site_names, body_names, task.target_aliases)
        if object_candidates and target_binding.status in VALID_BINDING_STATUSES:
            valid_candidates: list[tuple[BindingResult, PhysicalEvent]] = []
            incomplete_candidates: list[tuple[BindingResult, PhysicalEvent]] = []
            for candidate in object_candidates:
                event = detect_physical_event(
                    step_rows=ctx["step_rows"],
                    sim_arrays=ctx["sim_arrays"],
                    site_names=site_names,
                    object_binding=candidate,
                    target_binding=target_binding,
                    target_kind=target_kind,
                )
                if event.event_valid:
                    valid_candidates.append((candidate, event))
                else:
                    incomplete_candidates.append((candidate, event))
            if len(valid_candidates) > 1:
                episode = {
                    **base,
                    "mechanism_eligible": False,
                    "object_binding_status": "AMBIGUOUS",
                    "target_binding_status": target_binding.status,
                    "teacher_status": "OBJECT_BINDING_AMBIGUOUS",
                    "teacher_semantic_abstain": True,
                    "abstain_reason": "multiple_object_candidates_have_physical_events",
                    "event_count": 0,
                    "manual_review_required": True,
                    "object_binding_candidates_json": json.dumps([c.name for c, _ in valid_candidates]),
                    "target_binding_candidates_json": json.dumps(list(target_binding.candidates)),
                    "canonical_instance_candidates_json": json.dumps([normalize_entity_name(c.name) for c, _ in valid_candidates]),
                    "binding_decision_reason": "ambiguous_after_physical_event_filter",
                }
                return episode, []
            if len(valid_candidates) == 1:
                obj_binding, event = valid_candidates[0]
                episode = {
                    **base,
                    "mechanism_eligible": True,
                    "object_binding_status": obj_binding.status,
                    "target_binding_status": target_binding.status,
                    "teacher_status": "ELIGIBLE_EVENT",
                    "teacher_semantic_abstain": False,
                    "abstain_reason": "",
                    "event_count": 1,
                    "manual_review_required": True,
                    "object_binding_candidates_json": json.dumps([c.name for c in object_candidates]),
                    "target_binding_candidates_json": json.dumps(list(target_binding.candidates)),
                    "canonical_instance_candidates_json": json.dumps([normalize_entity_name(c.name) for c in object_candidates]),
                    "binding_decision_reason": "unique_candidate_with_grasp_lift_stable_carry",
                }
                return episode, [
                    event_to_row(
                        episode_key=episode_key,
                        event_idx=0,
                        object_binding=obj_binding,
                        target_binding=target_binding,
                        event=event,
                        supplementary=False,
                    )
                ]
            fallback_event = incomplete_candidates[0][1] if incomplete_candidates else None
            episode = {
                **base,
                "mechanism_eligible": False,
                "object_binding_status": "FAILED" if not object_candidates else "BOUND_STRUCTURED_FALLBACK",
                "target_binding_status": target_binding.status,
                "teacher_status": "NO_RELEVANT_GRASP_EVENT",
                "teacher_semantic_abstain": True,
                "abstain_reason": fallback_event.event_invalid_reason if fallback_event is not None else "no_physical_event_candidate",
                "event_count": 0,
                "manual_review_required": True,
                "object_binding_candidates_json": json.dumps([c.name for c in object_candidates]),
                "target_binding_candidates_json": json.dumps(list(target_binding.candidates)),
                "canonical_instance_candidates_json": json.dumps([normalize_entity_name(c.name) for c in object_candidates]),
                "binding_decision_reason": "no_candidate_has_grasp_lift_stable_carry",
            }
            return episode, []
        if obj_binding.status in VALID_BINDING_STATUSES and target_binding.status in VALID_BINDING_STATUSES:
            event = detect_physical_event(
                step_rows=ctx["step_rows"],
                sim_arrays=ctx["sim_arrays"],
                site_names=site_names,
                object_binding=obj_binding,
                target_binding=target_binding,
                target_kind=target_kind,
            )
            if not event.event_valid:
                status = "NO_RELEVANT_GRASP_EVENT" if event.status != "SIM_ARRAY_MISSING" else "SCHEMA_INVALID"
                episode = {
                    **base,
                    "mechanism_eligible": False,
                    "object_binding_status": obj_binding.status,
                    "target_binding_status": target_binding.status,
                    "teacher_status": status,
                    "teacher_semantic_abstain": True,
                    "abstain_reason": event.event_invalid_reason or event.status,
                    "event_count": 0,
                    "manual_review_required": True,
                }
                return episode, []
            episode = {
                **base,
                "mechanism_eligible": True,
                "object_binding_status": obj_binding.status,
                "target_binding_status": target_binding.status,
                "teacher_status": "ELIGIBLE_EVENT",
                "teacher_semantic_abstain": False,
                "abstain_reason": "",
                "event_count": 1,
                "manual_review_required": True,
            }
            return episode, [
                event_to_row(
                    episode_key=episode_key,
                    event_idx=0,
                    object_binding=obj_binding,
                    target_binding=target_binding,
                    event=event,
                    supplementary=False,
                )
            ]
        status = "OBJECT_BINDING_AMBIGUOUS" if obj_binding.status in {"FAILED", "AMBIGUOUS"} else "TARGET_BINDING_AMBIGUOUS"
        episode = {
            **base,
            "mechanism_eligible": False,
            "object_binding_status": obj_binding.status,
            "target_binding_status": target_binding.status,
            "teacher_status": status,
            "teacher_semantic_abstain": True,
            "abstain_reason": "required object or target binding failed",
            "event_count": 0,
            "manual_review_required": True,
        }
        return episode, []

    if task.mechanism_type in SUPPLEMENTARY_MECHANISMS:
        object_bindings = [b for b in bind_many(body_names, task.manipulated_object_aliases) if b.status in VALID_BINDING_STATUSES]
        target_binding, target_kind = bind_target(site_names, body_names, task.target_aliases)
        supplementary_events: list[dict[str, Any]] = []
        if object_bindings and target_binding.status in VALID_BINDING_STATUSES:
            for event_idx, object_binding in enumerate(object_bindings):
                event = detect_physical_event(
                    step_rows=ctx["step_rows"],
                    sim_arrays=ctx["sim_arrays"],
                    site_names=site_names,
                    object_binding=object_binding,
                    target_binding=target_binding,
                    target_kind=target_kind,
                )
                if event.event_valid:
                    supplementary_events.append(
                        event_to_row(
                            episode_key=episode_key,
                            event_idx=event_idx,
                            object_binding=object_binding,
                            target_binding=target_binding,
                            event=event,
                            supplementary=True,
                        )
                    )
        episode = {
            **base,
            "mechanism_eligible": False,
            "object_binding_status": "NOT_APPLICABLE",
            "target_binding_status": "NOT_APPLICABLE" if target_binding.status in VALID_BINDING_STATUSES else target_binding.status,
            "teacher_status": "MULTI_EVENT_AUDIT_ONLY" if supplementary_events else RESOLVER_NOT_IMPLEMENTED,
            "teacher_semantic_abstain": True,
            "abstain_reason": "supplementary_event_level_audit_not_primary_denominator"
            if supplementary_events
            else "supplementary_event_segmentation_not_reliably_resolved",
            "event_count": len(supplementary_events),
            "manual_review_required": True,
        }
        return episode, supplementary_events

    episode = {
        **base,
        "mechanism_eligible": False,
        "object_binding_status": "NOT_APPLICABLE",
        "target_binding_status": "NOT_APPLICABLE",
        "teacher_status": "CORRECT_SEMANTIC_ABSTAIN",
        "teacher_semantic_abstain": True,
        "abstain_reason": f"mechanism_not_supported_for_primary_teacher:{task.mechanism_type}",
        "event_count": 0,
        "manual_review_required": False,
    }
    return episode, []


def validate_episode_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        status = row.get("teacher_status")
        if row.get("teacher_executed") is not True:
            errors.append(f"{row.get('episode_key')}: teacher_executed must be true")
        if status == "ELIGIBLE_EVENT":
            if not row.get("mechanism_eligible") or row.get("teacher_semantic_abstain") or int(row.get("event_count", 0)) < 1:
                errors.append(f"{row.get('episode_key')}: invalid ELIGIBLE_EVENT invariant")
        elif status == "CORRECT_SEMANTIC_ABSTAIN":
            if row.get("mechanism_eligible") or not row.get("teacher_semantic_abstain") or int(row.get("event_count", -1)) != 0:
                errors.append(f"{row.get('episode_key')}: invalid abstain invariant")
        elif status == "MULTI_EVENT_AUDIT_ONLY":
            if row.get("mechanism_type") not in SUPPLEMENTARY_MECHANISMS or not row.get("manual_review_required"):
                errors.append(f"{row.get('episode_key')}: invalid multi-event invariant")
        elif status == RESOLVER_NOT_IMPLEMENTED:
            if row.get("mechanism_type") not in SUPPLEMENTARY_MECHANISMS or int(row.get("event_count", -1)) != 0:
                errors.append(f"{row.get('episode_key')}: invalid resolver-not-implemented invariant")
        elif status in {"OBJECT_BINDING_AMBIGUOUS", "TARGET_BINDING_AMBIGUOUS", "RESOLVER_FAILED", "SCHEMA_INVALID", "NO_RELEVANT_GRASP_EVENT"}:
            if row.get("mechanism_eligible"):
                errors.append(f"{row.get('episode_key')}: invalid failed/ambiguous invariant")
    return errors


def rows_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    obj = read_json(manifest_path)
    return list(obj.get("selected", []))


def run_resolver(manifest_path: Path, ontology_path: Path, output_dir: Path, *, teacher_run_id: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ontology = load_ontology(ontology_path)
    rows = rows_from_manifest(manifest_path)
    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        task = ontology[(row["suite"], int(row["task_idx"]))]
        try:
            ep, events = resolve_episode(row, task, teacher_run_id=teacher_run_id)
            episode_rows.append(ep)
            event_rows.extend(events)
        except Exception as exc:
            failures.append({"canonical_key": row.get("canonical_key", ""), "error": f"{type(exc).__name__}:{exc}"})
    validation_errors = validate_episode_rows(episode_rows)
    write_csv(output_dir / "teacher_episode_labels_v1.csv", episode_rows)
    write_csv(output_dir / "teacher_event_labels_v1.csv", event_rows)
    sidecars = {
        "teacher_run_id": teacher_run_id,
        "manifest": str(manifest_path),
        "episode_count": len(episode_rows),
        "event_count": len(event_rows),
        "failure_count": len(failures),
        "validation_error_count": len(validation_errors),
        "failures": failures,
        "validation_errors": validation_errors,
        "claim_boundary": {
            "full_clean300_batch": "NOT_RUN",
            "detector_telemetry": "NOT_READ",
            "gpu_libero_vis_rand_attack": "NOT_RUN",
            "manual_review": "NOT_COMPLETE",
        },
    }
    write_json(output_dir / "privileged_sidecar_resolved_v1.json", sidecars)
    return sidecars


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def teacher_timeline_rows(label: dict[str, str], event: dict[str, str]) -> list[dict[str, Any]]:
    marker_fields = [
        ("window_start", "proposed Teacher window start", "teacher_window_start"),
        ("close_onset", "first accepted close onset", "close_onset_step"),
        ("grasp_established", "grasp evidence established", "grasp_established_step"),
        ("lift_onset", "object lift onset", "lift_onset_step"),
        ("stable_carry_start", "stable carry starts", "stable_carry_start"),
        ("anchor", "Teacher anchor step", "teacher_anchor_step"),
        ("target_proximity", "target proximity, if observed", "target_proximity_step"),
        ("release_onset", "release onset, if observed", "release_onset_step"),
        ("window_end", "proposed Teacher window end", "teacher_window_end"),
    ]
    rows: list[dict[str, Any]] = []
    for marker, description, field in marker_fields:
        step = optional_int(event.get(field))
        if step is None:
            continue
        rows.append(
            {
                "episode_key": label.get("episode_key", ""),
                "teacher_status": label.get("teacher_status", ""),
                "event_id": event.get("event_id", ""),
                "marker": marker,
                "description": description,
                "step": step,
                "object_body": event.get("object_body_name", ""),
                "target_body_or_site": event.get("target_body_or_site_name", ""),
            }
        )
    if not rows:
        rows.append(
            {
                "episode_key": label.get("episode_key", ""),
                "teacher_status": label.get("teacher_status", ""),
                "event_id": event.get("event_id", ""),
                "marker": "no_teacher_event",
                "description": "no accepted Teacher event row",
                "step": "",
                "object_body": "",
                "target_body_or_site": "",
            }
        )
    return rows


def write_teacher_timeline(path: Path, label: dict[str, str], event: dict[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, teacher_timeline_rows(label, event))
    return "WROTE"


def draw_teacher_overlay(frame: np.ndarray, label: dict[str, str], event: dict[str, str], step: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    window_start = optional_int(event.get("teacher_window_start"))
    window_end = optional_int(event.get("teacher_window_end"))
    close_step = optional_int(event.get("close_onset_step"))
    anchor_step = optional_int(event.get("teacher_anchor_step"))
    release_step = optional_int(event.get("release_onset_step"))
    if window_start is not None and window_end is not None and window_start <= step <= window_end:
        draw.rectangle([0, 0, width, 10], fill=(40, 180, 80))
    if close_step is not None and step == close_step:
        draw.rectangle([0, 10, width, 20], fill=(220, 60, 50))
    if anchor_step is not None and step == anchor_step:
        draw.rectangle([0, 20, width, 30], fill=(255, 220, 0))
    if release_step is not None and step == release_step:
        draw.rectangle([0, 30, width, 40], fill=(80, 160, 255))
    status = label.get("teacher_status", "")
    event_id = event.get("event_id", "")
    obj = event.get("object_body_name", "")
    target = event.get("target_body_or_site_name", "")
    draw.rectangle([0, max(0, height - 42), width, height], fill=(0, 0, 0))
    draw.text((4, height - 40), f"step={step} status={status}", fill=(255, 255, 255))
    draw.text((4, height - 24), f"{event_id} {obj}->{target}", fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def write_teacher_overlay_video(raw_video: Path, output_path: Path, label: dict[str, str], event: dict[str, str]) -> str:
    if not raw_video.exists():
        return "RAW_VIDEO_MISSING"
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return f"IMAGEIO_UNAVAILABLE:{type(exc).__name__}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reader = imageio.get_reader(raw_video)
        meta = reader.get_meta_data()
        fps = int(meta.get("fps") or 10)
        frames = [draw_teacher_overlay(frame, label, event, idx) for idx, frame in enumerate(reader)]
        reader.close()
        if not frames:
            return "NO_FRAMES"
        imageio.mimwrite(output_path, frames, fps=fps)
        return "WROTE"
    except Exception as exc:
        return f"OVERLAY_FAILED:{type(exc).__name__}:{exc}"


def build_review_package(manifest_path: Path, resolver_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_manifest(manifest_path)
    label_rows = read_csv_rows(resolver_dir / "teacher_episode_labels_v1.csv")
    event_rows = read_csv_rows(resolver_dir / "teacher_event_labels_v1.csv")
    labels_by_key = {r["episode_key"]: r for r in label_rows}
    events_by_key: dict[str, list[dict[str, str]]] = {}
    for event in event_rows:
        events_by_key.setdefault(event["episode_key"], []).append(event)
    package_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        episode_path = Path(row["episode_path"])
        label = labels_by_key.get(row["canonical_key"], {})
        event_candidates = events_by_key.get(row["canonical_key"], [{}])
        raw_video = episode_path / "rollout_raw.mp4"
        dest_video = output_dir / "videos" / f"review_{idx:03d}_{row['suite']}_t{int(row['task_idx']):02d}_s{int(row['state_id']):02d}.mp4"
        video_status = "missing"
        if raw_video.exists():
            dest_video.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dest_video.exists():
                    dest_video.unlink()
                os.symlink(raw_video, dest_video)
                video_status = "symlink"
            except Exception:
                shutil.copy2(raw_video, dest_video)
                video_status = "copied"
        hidden_rows.append(
            {
                "review_id_prefix": f"review_{idx:03d}",
                "episode_key": row["canonical_key"],
                "task_success": row.get("task_success", ""),
                "source_episode_abspath_audit_only": row["episode_path"],
                "artifact_recursive_sha256": row.get("artifact_recursive_sha256", ""),
            }
        )
        for event_idx, event in enumerate(event_candidates):
            review_id = f"review_{idx:03d}_event_{event_idx:02d}"
            timeline_path = output_dir / "teacher_timelines" / f"{review_id}_teacher_timeline.csv"
            timeline_status = write_teacher_timeline(timeline_path, label, event)
            overlay_path = output_dir / "teacher_overlays" / f"{review_id}_teacher_overlay.mp4"
            overlay_status = write_teacher_overlay_video(raw_video, overlay_path, label, event)
            overlay_public_path = str(overlay_path) if overlay_status == "WROTE" else ""
            package_rows.append(
                {
                    "review_id": review_id,
                    "episode_key": row["canonical_key"],
                    "suite": row["suite"],
                    "task_idx": row["task_idx"],
                    "state_id": row["state_id"],
                    "mechanism_type": row.get("mechanism_type", label.get("mechanism_type", "")),
                    "teacher_status": label.get("teacher_status", ""),
                    "event_id": event.get("event_id", ""),
                    "proposed_object_body": event.get("object_body_name", ""),
                    "proposed_target_body_or_site": event.get("target_body_or_site_name", ""),
                    "proposed_close_onset": event.get("close_onset_step", ""),
                    "proposed_grasp_established": event.get("grasp_established_step", ""),
                    "proposed_lift_onset": event.get("lift_onset_step", ""),
                    "proposed_stable_carry_start": event.get("stable_carry_start", ""),
                    "proposed_window_start": event.get("teacher_window_start", ""),
                    "proposed_anchor": event.get("teacher_anchor_step", ""),
                    "proposed_window_end": event.get("teacher_window_end", ""),
                    "proposed_release_onset": event.get("release_onset_step", ""),
                    "blind_video_path": str(dest_video) if raw_video.exists() else "",
                    "teacher_only_timeline_path": str(timeline_path),
                    "teacher_only_timeline_status": timeline_status,
                    "teacher_only_overlay_path": overlay_public_path,
                    "teacher_only_overlay_status": overlay_status,
                    "video_status": video_status,
                    "reviewer_id": "",
                    "object_binding_correct": "",
                    "target_binding_correct": "",
                    "mechanism_correct": "",
                    "event_count_correct": "",
                    "timing_window_correct": "",
                    "abstention_correct": "",
                    "corrected_window_start": "",
                    "corrected_window_end": "",
                    "disagreement_reason": "",
                    "review_timestamp": "",
                }
            )
    write_csv(output_dir / "blind_review_queue.csv", package_rows)
    write_csv(output_dir / "blind_review_hidden_audit_manifest.csv", hidden_rows)
    instructions = {
        "package_type": "resolver_diagnostic_holdout_review_package_v1",
        "holdout_role": "diagnostic_only_observed_during_resolver_development_not_final_unbiased_validation",
        "manifest": str(manifest_path),
        "resolver_dir": str(resolver_dir),
        "review_count": len(package_rows),
        "blindness": "No detector telemetry, detector overlay, VIS/RAND/shuffled, or attack outputs are included. Teacher-only overlays contain only resolver proposal markers.",
        "required_human_fields": [
            "reviewer_id",
            "object_binding_correct",
            "target_binding_correct",
            "mechanism_correct",
            "event_count_correct",
            "timing_window_correct",
            "abstention_correct",
            "corrected_window_start",
            "corrected_window_end",
            "disagreement_reason",
            "review_timestamp",
        ],
        "manual_review_complete": False,
    }
    write_json(output_dir / "blind_review_instructions.json", instructions)
    return instructions


def _cmd_build_manifests(args: argparse.Namespace) -> None:
    ontology = load_ontology(Path(args.ontology))
    ledger = read_csv_rows(Path(args.ledger))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = build_dev_canary_manifest(ledger, ontology)
    write_json(out / "layer1_dev_canary_manifest_v1.json", dev)
    blind = build_blind_review_manifest(
        ledger,
        ontology,
        exclude_keys={r["canonical_key"] for r in dev["selected"]},
        count=int(args.blind_count),
    )
    write_json(out / "layer1_blind_review_manifest_v1.json", blind)
    write_csv(out / "layer1_dev_canary_manifest_v1.csv", dev["selected"])
    write_csv(out / "layer1_blind_review_manifest_v1.csv", blind["selected"])
    write_json(
        out / "layer1_manifest_summary.json",
        {
            "dev_selected_count": dev["selected_count"],
            "blind_selected_count": blind["selected_count"],
            "full_clean300_batch": "NOT_RUN",
            "detector_telemetry": "NOT_READ",
        },
    )


def _cmd_resolve(args: argparse.Namespace) -> None:
    result = run_resolver(
        Path(args.manifest),
        Path(args.ontology),
        Path(args.output_dir),
        teacher_run_id=str(args.teacher_run_id),
    )
    if result["failure_count"] or result["validation_error_count"]:
        raise SystemExit(f"resolver validation failed: {result}")


def _cmd_review_package(args: argparse.Namespace) -> None:
    build_review_package(Path(args.manifest), Path(args.resolver_dir), Path(args.output_dir))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common_ontology = str(REPO / "configs" / "cross_suite_task_ontology_v1.yaml")
    m = sub.add_parser("build-manifests")
    m.add_argument("--ledger", required=True)
    m.add_argument("--ontology", default=common_ontology)
    m.add_argument("--output_dir", required=True)
    m.add_argument("--blind_count", type=int, default=24)
    r = sub.add_parser("resolve")
    r.add_argument("--manifest", required=True)
    r.add_argument("--ontology", default=common_ontology)
    r.add_argument("--output_dir", required=True)
    r.add_argument("--teacher_run_id", required=True)
    p = sub.add_parser("review-package")
    p.add_argument("--manifest", required=True)
    p.add_argument("--resolver_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "build-manifests":
        _cmd_build_manifests(args)
    elif args.cmd == "resolve":
        _cmd_resolve(args)
    elif args.cmd == "review-package":
        _cmd_review_package(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
