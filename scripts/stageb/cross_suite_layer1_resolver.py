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

REPO = Path(__file__).resolve().parents[2]

ONTOLOGY_VERSION = "cross_suite_task_ontology_v1"
TEACHER_VERSION = "cross_suite_teacher_v1"
RESOLVER_VERSION = "cross_suite_resolver_v1"
PRIMARY_MECHANISM = "single_object_pick_place"
SUPPLEMENTARY_MECHANISMS = {"multi_object_transfer", "mixed_articulated_pick_place"}
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
        "manifest_type": "cross_suite_layer1_blind_review_manifest_v1",
        "selection_rule": "disjoint deterministic round-robin over suite|mechanism_group|task_success strata",
        "source_commit": git_commit(),
        "selected_count": len(selected),
        "requested_count": count,
        "excluded_key_count": len(exclude_keys),
        "selected": selected,
        "rejected_count": len(rejected),
    }


def match_alias(names: list[str], aliases: tuple[str, ...]) -> tuple[str, str]:
    for alias in aliases:
        normalized = alias.lower()
        for name in names:
            low = name.lower()
            if low == normalized or low.startswith(normalized + "_") or normalized in low:
                return name, "BOUND_EXACT" if low == normalized else "BOUND_BDDL_ONTOLOGY"
    return "", "FAILED"


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
    return {
        "episode_path": episode_path,
        "manifest": manifest,
        "summary": summary,
        "sim_manifest": sim_manifest,
        "step_rows": step_rows,
    }


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def find_first_close_onset(step_rows: list[dict[str, Any]]) -> int | None:
    prev_close = False
    for row in step_rows:
        step = int(float(row.get("step", -1)))
        env_grip = _float(row, "env_gripper")
        raw_grip = _float(row, "raw_gripper")
        is_close = (math.isfinite(env_grip) and env_grip > 0) or (math.isfinite(raw_grip) and raw_grip < 0.5)
        if is_close and not prev_close:
            return step
        prev_close = is_close
    return None


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


def resolve_episode(row: dict[str, Any], task: OntologyTask, *, teacher_run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_path = Path(row["episode_path"])
    ctx = load_episode_context(episode_path)
    sim_meta = ctx["sim_manifest"].get("metadata", {})
    body_names = [str(x) for x in sim_meta.get("body_names", [])]
    site_names = [str(x) for x in sim_meta.get("site_names", [])]
    source_sha = str(row.get("source_episode_sha") or row.get("artifact_recursive_sha256") or "")
    episode_key = str(row.get("canonical_key") or canonical_key(row))
    n_steps = int(ctx["summary"].get("n_steps") or row.get("n_steps") or len(ctx["step_rows"]))

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
        "source_episode_relpath": str(episode_path),
        "source_episode_sha": source_sha,
        "mechanism_type": task.mechanism_type,
    }

    if task.mechanism_type == PRIMARY_MECHANISM:
        obj, obj_status = match_alias(body_names, task.manipulated_object_aliases)
        target, target_status = match_alias(site_names + body_names, task.target_aliases)
        if obj_status in {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY", "BOUND_STRUCTURED_FALLBACK"} and target_status in {
            "BOUND_EXACT",
            "BOUND_BDDL_ONTOLOGY",
            "BOUND_STRUCTURED_FALLBACK",
        }:
            timing = propose_timing(ctx["step_rows"], n_steps)
            if timing["timing_status"] == "NO_RELEVANT_CLOSE_ONSET":
                episode = {
                    **base,
                    "mechanism_eligible": False,
                    "object_binding_status": obj_status,
                    "target_binding_status": target_status,
                    "teacher_status": "NO_RELEVANT_GRASP_EVENT",
                    "teacher_semantic_abstain": True,
                    "abstain_reason": "no_close_onset_found_in_allowed_clean_step_fields",
                    "event_count": 0,
                    "manual_review_required": True,
                }
                return episode, []
            episode = {
                **base,
                "mechanism_eligible": True,
                "object_binding_status": obj_status,
                "target_binding_status": target_status,
                "teacher_status": "ELIGIBLE_EVENT",
                "teacher_semantic_abstain": False,
                "abstain_reason": "",
                "event_count": 1,
                "manual_review_required": True,
            }
            event = {
                "episode_key": episode_key,
                "event_id": f"{episode_key}|event0",
                "object_body_name": obj,
                "object_joint_name": "",
                "target_body_or_site_name": target,
                "binding_source": "ontology_alias_body_site_match",
                "binding_confidence_class": "high" if obj_status == "BOUND_EXACT" and target_status == "BOUND_EXACT" else "medium",
                **timing,
                "event_valid": True,
                "event_invalid_reason": "",
            }
            return episode, [event]
        status = "OBJECT_BINDING_AMBIGUOUS" if obj_status == "FAILED" else "TARGET_BINDING_AMBIGUOUS"
        episode = {
            **base,
            "mechanism_eligible": False,
            "object_binding_status": obj_status,
            "target_binding_status": target_status,
            "teacher_status": status,
            "teacher_semantic_abstain": True,
            "abstain_reason": "required object or target binding failed",
            "event_count": 0,
            "manual_review_required": True,
        }
        return episode, []

    if task.mechanism_type in SUPPLEMENTARY_MECHANISMS:
        episode = {
            **base,
            "mechanism_eligible": False,
            "object_binding_status": "NOT_APPLICABLE",
            "target_binding_status": "NOT_APPLICABLE",
            "teacher_status": "MULTI_EVENT_AUDIT_ONLY",
            "teacher_semantic_abstain": True,
            "abstain_reason": "supplementary_event_level_audit_not_primary_denominator",
            "event_count": 0,
            "manual_review_required": True,
        }
        return episode, []

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


def build_review_package(manifest_path: Path, resolver_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_manifest(manifest_path)
    label_rows = read_csv_rows(resolver_dir / "teacher_episode_labels_v1.csv")
    labels_by_key = {r["episode_key"]: r for r in label_rows}
    package_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        episode_path = Path(row["episode_path"])
        label = labels_by_key.get(row["canonical_key"], {})
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
        package_rows.append(
            {
                "review_id": f"review_{idx:03d}",
                "episode_key": row["canonical_key"],
                "suite": row["suite"],
                "task_idx": row["task_idx"],
                "state_id": row["state_id"],
                "task_success": row.get("task_success", ""),
                "mechanism_type": row.get("mechanism_type", label.get("mechanism_type", "")),
                "teacher_status": label.get("teacher_status", ""),
                "mechanism_eligible": label.get("mechanism_eligible", ""),
                "object_binding_status": label.get("object_binding_status", ""),
                "target_binding_status": label.get("target_binding_status", ""),
                "event_count": label.get("event_count", ""),
                "source_episode_relpath": row["episode_path"],
                "blind_video_path": str(dest_video) if raw_video.exists() else "",
                "video_status": video_status,
                "human_binding_accept": "",
                "human_timing_accept": "",
                "human_notes": "",
            }
        )
    write_csv(output_dir / "blind_review_queue.csv", package_rows)
    instructions = {
        "package_type": "cross_suite_layer1_blind_review_package_v1",
        "manifest": str(manifest_path),
        "resolver_dir": str(resolver_dir),
        "review_count": len(package_rows),
        "blindness": "No detector telemetry, detector overlay, VIS/RAND/shuffled, or attack outputs are included.",
        "required_human_fields": ["human_binding_accept", "human_timing_accept", "human_notes"],
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
