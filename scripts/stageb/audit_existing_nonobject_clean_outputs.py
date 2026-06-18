#!/usr/bin/env python3
"""Read-only, metadata-only audit of existing non-Object CLEAN outputs.

This script intentionally performs no GPU work, loads no OpenVLA model, runs no
rollout, and never modifies files under --outputs-root. It scans directory
metadata, small JSON files, CSV headers/streamed rows, JSONL row counts, and NPZ
member headers only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from numpy.lib import format as np_format
except Exception:  # pragma: no cover
    np = None
    np_format = None

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

try:
    from gripper_attack.sc5_detector_runtime import SC5_FEATURES
except Exception:  # pragma: no cover
    SC5_FEATURES = [
        "gripper_command", "gripper_qpos", "gripper_opening_proxy",
        "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
        "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
        "close_onset", "time_since_close", "eef_speed",
        "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
        "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
    ]

SENTINELS = {
    "episode_manifest.json",
    "episode_summary.json",
    "run_manifest.json",
    "video_manifest.json",
    "step_telemetry.csv",
    "detector_telemetry.csv",
    "step_records.jsonl",
    "episode_records.jsonl",
    "summary.json",
    "results.json",
}
NONOBJECT_SUITES = {"libero_spatial", "libero_goal", "libero_10"}
SMALL_JSON_LIMIT = 8 * 1024 * 1024


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_text(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(REPO), stderr=subprocess.STDOUT, timeout=timeout).decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def read_json_small(path: Path) -> dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size > SMALL_JSON_LIMIT:
            return {}
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else {"_json_type": type(obj).__name__}
    except Exception as exc:
        return {"_json_error": f"{type(exc).__name__}:{exc}"}


def first_existing_json(ep: Path, names: list[str]) -> dict[str, Any]:
    for name in names:
        data = read_json_small(ep / name)
        if data:
            return data
    return {}


def infer_suite(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> tuple[str, str, str]:
    for source, obj in [("manifest", manifest), ("summary", summary)]:
        val = str(obj.get("suite", "") or "")
        if val in NONOBJECT_SUITES:
            return val, source, "HIGH"
    for source, obj in [("manifest_unnorm_key", manifest), ("summary_unnorm_key", summary)]:
        val = str(obj.get("unnorm_key", "") or "")
        if val in NONOBJECT_SUITES:
            return val, source, "HIGH"
    text = " ".join(str(x) for x in [
        manifest.get("model_path", ""), summary.get("model_path", ""),
        manifest.get("checkpoint_path", ""), path.as_posix(),
    ]).lower()
    for suite in NONOBJECT_SUITES:
        if suite in text or suite.replace("libero_", "libero-") in text:
            return suite, "model_or_path", "MEDIUM" if "model_path" in text else "LOW"
    if "spatial" in text:
        return "libero_spatial", "path_heuristic", "LOW"
    if "goal" in text:
        return "libero_goal", "path_heuristic", "LOW"
    if "libero10" in text or "libero_10" in text:
        return "libero_10", "path_heuristic", "LOW"
    return "", "unresolved", "NONE"


def infer_condition(path: Path, manifest: dict[str, Any], summary: dict[str, Any], telemetry_header: list[str]) -> tuple[str, str, str]:
    for source, obj in [("manifest", manifest), ("summary", summary)]:
        val = str(obj.get("condition", "") or "")
        if val:
            return val, source, "HIGH"
    if "condition" in telemetry_header:
        # Do not scan rows here. Phase B checks telemetry values for CLEAN candidates.
        return "CONDITION_COLUMN_PRESENT", "telemetry_header", "MEDIUM"
    text = path.as_posix().lower()
    if "clean" in text:
        return "CLEAN", "path_heuristic", "LOW"
    if any(k in text for k in ["vis", "rand", "attack", "true"]):
        return "NON_CLEAN_OR_ATTACK", "path_heuristic", "LOW"
    return "", "unresolved", "NONE"


def csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return next(csv.reader(f), [])
    except Exception:
        return []


def stream_step_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    header = csv_header(path)
    feature_cols = [f"f_{x}" for x in SC5_FEATURES]
    bare_feature_cols = list(SC5_FEATURES)
    has_prefixed = all(c in header for c in feature_cols)
    has_bare = all(c in header for c in bare_feature_cols)
    rows = 0
    attack_true = 0
    nonclean_conditions = set()
    finite_feature_rows = 0
    last_step = None
    continuous = True
    invalid_feature_steps = 0
    clean_values = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            cond = str(row.get("condition", "") or "")
            if cond:
                clean_values.add(cond)
                if cond != "CLEAN":
                    nonclean_conditions.add(cond)
            if str(row.get("attack_this", "")).strip().lower() in {"true", "1", "yes"}:
                attack_true += 1
            try:
                step = int(float(row.get("step", row.get("step_idx", rows - 1))))
                if last_step is not None and step != last_step + 1:
                    continuous = False
                last_step = step
            except Exception:
                continuous = False
            if str(row.get("feat_valid", "true")).strip().lower() in {"false", "0"}:
                invalid_feature_steps += 1
            cols = feature_cols if has_prefixed else bare_feature_cols
            if all(c in row for c in cols):
                ok = True
                for c in cols:
                    try:
                        if not math.isfinite(float(row[c])):
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break
                if ok:
                    finite_feature_rows += 1
    return {
        "exists": True,
        "header": header,
        "row_count": rows,
        "attack_this_true_count": attack_true,
        "condition_values": "|".join(sorted(clean_values)),
        "nonclean_condition_values": "|".join(sorted(nonclean_conditions)),
        "step_sequence_continuous": continuous,
        "invalid_feature_steps_from_rows": invalid_feature_steps,
        "features_25d_columns_complete": has_prefixed or has_bare,
        "features_25d_finite_rows": finite_feature_rows,
        "features_25d_finite_all_rows": rows > 0 and finite_feature_rows == rows,
    }


def jsonl_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows += 1
    return {"exists": True, "row_count": rows}


def npz_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    out = {"exists": True, "members": [], "arrays": {}, "error": ""}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                out["members"].append(info.filename)
                if np_format is None or not info.filename.endswith(".npy"):
                    continue
                with zf.open(info, "r") as raw:
                    bio = io.BytesIO(raw.read(512))
                    try:
                        version = np_format.read_magic(bio)
                        shape, fortran, dtype = np_format._read_array_header(bio, version)
                        out["arrays"][info.filename[:-4]] = {
                            "shape": list(shape),
                            "dtype": str(dtype),
                            "fortran_order": bool(fortran),
                        }
                    except Exception as exc:
                        out["arrays"][info.filename] = {"error": f"{type(exc).__name__}:{exc}"}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def dir_stats(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                st = (Path(root) / fn).stat()
                count += 1
                size += int(st.st_size)
            except OSError:
                continue
    return count, size


def discover_episode_dirs(outputs_root: Path) -> tuple[list[Path], dict[str, int]]:
    found = []
    counters = {"dirs_scanned": 0, "files_seen": 0, "bytes_seen": 0}
    for root, _dirs, files in os.walk(outputs_root):
        counters["dirs_scanned"] += 1
        file_set = set(files)
        counters["files_seen"] += len(files)
        if file_set & SENTINELS:
            found.append(Path(root))
        for fn in files:
            try:
                counters["bytes_seen"] += int((Path(root) / fn).stat().st_size)
            except OSError:
                pass
    return found, counters


def active_progress(active_root: Path, stability_window_sec: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    status_path = active_root / "queue_status.csv"
    queue_pid_path = Path(str(active_root) + ".queue.pid")
    pid = queue_pid_path.read_text(encoding="utf-8").strip() if queue_pid_path.exists() else ""
    if status_path.exists():
        with status_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f))
    now = time.time()
    out = []
    for row in rows:
        ep = Path(row.get("output_dir", ""))
        summary = ep / "episode_summary.json"
        artifact = ep / "artifact_sha256.json"
        stable = False
        if summary.exists() and artifact.exists():
            newest = max(summary.stat().st_mtime, artifact.stat().st_mtime)
            stable = newest < now - stability_window_sec
        out.append({
            "job_id": row.get("job_id", ""),
            "wave": row.get("wave", ""),
            "suite": row.get("suite", ""),
            "task_idx": row.get("task_idx", ""),
            "state_id": row.get("state_id", ""),
            "status": row.get("status", ""),
            "episode_path": str(ep),
            "summary_exists": summary.exists(),
            "artifact_exists": artifact.exists(),
            "files_stable": stable,
            "eligible_active_complete_snapshot": row.get("status") == "COMPLETE" and summary.exists() and artifact.exists() and stable,
        })
    meta = {
        "active_root": str(active_root),
        "queue_pid": pid,
        "queue_status_exists": status_path.exists(),
        "rows": len(rows),
        "complete_count": sum(1 for r in rows if r.get("status") == "COMPLETE"),
        "running_count": sum(1 for r in rows if r.get("status") == "RUNNING"),
        "output_root_mtime": active_root.stat().st_mtime if active_root.exists() else "",
    }
    return out, meta


def classify_episode(ep: Path, outputs_root: Path, active_root: Path, stability_window_sec: int) -> tuple[dict[str, Any], dict[str, Any]]:
    files = {p.name for p in ep.iterdir() if p.is_file()} if ep.exists() else set()
    manifest = read_json_small(ep / "episode_manifest.json")
    summary = first_existing_json(ep, ["episode_summary.json", "summary.json", "results.json", "run_manifest.json"])
    video = read_json_small(ep / "video_manifest.json")
    sidecar = read_json_small(ep / "privileged_sidecar.json")
    header = csv_header(ep / "step_telemetry.csv")
    suite, suite_source, suite_conf = infer_suite(ep, manifest, summary)
    condition, condition_source, condition_conf = infer_condition(ep, manifest, summary, header)
    file_count, total_size = dir_stats(ep)
    mtime = ep.stat().st_mtime
    is_active = active_root and (ep == active_root or active_root in ep.parents)
    is_growing = is_active and mtime > time.time() - stability_window_sec
    active_status = ""
    if is_active:
        # Prefer queue status when available.
        status_path = active_root / "queue_status.csv"
        if status_path.exists():
            try:
                rows = list(csv.DictReader(status_path.open("r", encoding="utf-8", errors="replace")))
                by_path = {r.get("output_dir", ""): r.get("status", "") for r in rows}
                active_status = by_path.get(str(ep), "")
            except Exception:
                active_status = ""
    flags = {
        "manifest_attack_enabled": manifest.get("attack_enabled", ""),
        "manifest_vis_enabled": manifest.get("vis_enabled", ""),
        "manifest_rand_enabled": manifest.get("rand_enabled", ""),
        "summary_vis_or_rand_run": summary.get("vis_or_rand_run", ""),
    }
    is_partial = (
        is_growing
        or active_status == "RUNNING"
        or ("episode_summary.json" not in files and "summary.json" not in files and "results.json" not in files)
    )
    root = {
        "episode_path": str(ep),
        "root_path": str(outputs_root),
        "directory_mtime": dt.datetime.fromtimestamp(mtime).isoformat(),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "sentinel_files": "|".join(sorted(files & SENTINELS)),
        "possible_suite": suite,
        "suite_source": suite_source,
        "suite_confidence": suite_conf,
        "possible_condition": condition,
        "condition_source": condition_source,
        "condition_confidence": condition_conf,
        "is_active_or_growing": bool(is_active or is_growing),
        "active_status": active_status,
        "is_partial": bool(is_partial),
    }
    root.update(flags)
    return root, {"manifest": manifest, "summary": summary, "video": video, "sidecar": sidecar}


def bool_true(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def audit_clean_candidate(ep: Path, root_row: dict[str, Any], docs: dict[str, dict[str, Any]], reference_commit: str) -> dict[str, Any]:
    manifest = docs["manifest"]
    summary = docs["summary"]
    sidecar = docs["sidecar"]
    telemetry = stream_step_telemetry(ep / "step_telemetry.csv")
    detector_header = csv_header(ep / "detector_telemetry.csv")
    frame_header = csv_header(ep / "frame_index.csv")
    step_jsonl = jsonl_stats(ep / "step_records.jsonl")
    episode_jsonl = jsonl_stats(ep / "episode_records.jsonl")
    frames_npz = npz_metadata(ep / "agentview_frames_uint8.npz")
    sim_npz = npz_metadata(ep / "sim_state_stream.npz")
    sim_manifest = read_json_small(ep / "sim_state_manifest.json")
    exact_tokens_present = any(h in telemetry.get("header", []) for h in ["exact_new_tokens_json", "gripper_token", "clean_gripper_token"])
    core_fields = {
        "step": "step" in telemetry.get("header", []),
        "action_vector": any(h in telemetry.get("header", []) for h in ["raw_action", "action_vector", "action_vector_json"]),
        "env_action": any(h in telemetry.get("header", []) for h in ["env_action", "env_action_json"]),
        "raw_gripper": "raw_gripper" in telemetry.get("header", []),
        "env_gripper": "env_gripper" in telemetry.get("header", []),
        "gripper_qpos": "gripper_qpos" in telemetry.get("header", []) or "qpos_sum" in telemetry.get("header", []),
        "gripper_opening_proxy": "gripper_opening_proxy" in telemetry.get("header", []),
        "eef_xyz": all(h in telemetry.get("header", []) for h in ["eef_x", "eef_y", "eef_z"]),
        "eef_vxyz": all(h in telemetry.get("header", []) for h in ["eef_vx", "eef_vy", "eef_vz"]),
        "action_dxyz": all(h in telemetry.get("header", []) for h in ["action_dx", "action_dy", "action_dz"]),
        "action_gripper": "action_gripper" in telemetry.get("header", []),
        "exact_tokens": exact_tokens_present,
    }
    detector_fields = all(h in detector_header or h in telemetry.get("header", []) for h in ["detector_state", "corridor_p", "release_p", "pred_phase"])
    visual_fields = {
        "agentview_frames_uint8_npz": frames_npz.get("exists", False),
        "rollout_raw_mp4": (ep / "rollout_raw.mp4").exists(),
        "rollout_overlay_mp4": (ep / "rollout_overlay.mp4").exists(),
        "frame_index_csv": (ep / "frame_index.csv").exists(),
    }
    sim_arrays = sim_npz.get("arrays", {}) if isinstance(sim_npz.get("arrays"), dict) else {}
    sim_state_complete = all(k in sim_arrays for k in ["qpos", "qvel", "body_xpos", "body_xquat", "site_xpos", "ctrl"]) and bool(sim_manifest)
    privileged_fields_present = any(k in sidecar for k in [
        "object_id", "primary_object", "target_id", "primary_target", "object_pose", "target_pose",
        "teacher_anchor", "teacher_labels", "privileged_valid", "teacher_abstain",
    ])
    privileged_valid = bool_true(sidecar.get("privileged_valid", False))
    teacher_abstain = bool_true(sidecar.get("teacher_abstain", False))

    clean_denominator = (
        root_row["possible_suite"] in NONOBJECT_SUITES
        and root_row["possible_condition"] == "CLEAN"
        and not bool_true(root_row.get("manifest_attack_enabled"))
        and not bool_true(root_row.get("manifest_vis_enabled"))
        and not bool_true(root_row.get("manifest_rand_enabled"))
        and telemetry.get("attack_this_true_count", 0) == 0
        and not root_row.get("is_partial")
    )
    step_core_complete = all(core_fields.values()) and telemetry.get("step_sequence_continuous", False)
    feature_complete = telemetry.get("features_25d_columns_complete", False) and telemetry.get("features_25d_finite_all_rows", False)
    detector_complete = feature_complete and detector_fields
    same_schema = str(manifest.get("source_commit", summary.get("source_commit", ""))) == str(reference_commit)
    visual_complete = all(visual_fields.values())
    artifact_sealed = (ep / "artifact_sha256.json").exists()
    usable = {
        "usable_clean_denominator": clean_denominator,
        "usable_detector_transfer_analysis": clean_denominator and step_core_complete and feature_complete and detector_complete,
        "usable_teacher_relabeling": clean_denominator and sim_state_complete,
        "usable_visual_feature_extraction": clean_denominator and frames_npz.get("exists", False),
        "usable_visual_replay": clean_denominator and frames_npz.get("exists", False) and visual_fields["frame_index_csv"],
        "usable_video_manual_audit": clean_denominator and visual_fields["rollout_raw_mp4"],
        "usable_same_schema_comparison_with_6379397": clean_denominator and same_schema and step_core_complete and feature_complete and detector_complete,
    }
    if not clean_denominator:
        tier = "TIER_X_REJECTED"
    elif step_core_complete and feature_complete and detector_complete and (sim_state_complete or frames_npz.get("exists", False)) and artifact_sealed:
        tier = "TIER_A_FULL_CURRENT_COMPATIBLE" if same_schema else "TIER_B_DETECTOR_COMPATIBLE"
    elif step_core_complete and feature_complete and detector_complete:
        tier = "TIER_B_DETECTOR_COMPATIBLE"
    elif root_row["possible_suite"] in NONOBJECT_SUITES and root_row["possible_condition"] == "CLEAN":
        tier = "TIER_C_CLEAN_DENOMINATOR_ONLY"
    elif (ep / "rollout_raw.mp4").exists() or (ep / "episode_summary.json").exists():
        tier = "TIER_D_LEGACY_REFERENCE_ONLY"
    else:
        tier = "TIER_X_REJECTED"
    if tier.startswith("TIER_A") and not same_schema:
        tier = "TIER_B_DETECTOR_COMPATIBLE"

    row = {
        **root_row,
        "suite": root_row["possible_suite"],
        "condition": root_row["possible_condition"],
        "task_idx": manifest.get("task_idx", summary.get("task_idx", "")),
        "task_name": manifest.get("task_name", summary.get("task_name", "")),
        "instruction": manifest.get("instruction", summary.get("instruction", "")),
        "state_id": manifest.get("state_id", summary.get("state_id", "")),
        "eval_seed": manifest.get("eval_seed", summary.get("eval_seed", "")),
        "source_commit": manifest.get("source_commit", summary.get("source_commit", "")),
        "runner": manifest.get("runner", summary.get("runner", "")),
        "model_path": manifest.get("model_path", summary.get("model_path", "")),
        "unnorm_key": manifest.get("unnorm_key", summary.get("unnorm_key", "")),
        "detector_checkpoint_sha": manifest.get("detector_checkpoint_sha256", summary.get("checkpoint_sha256", "")),
        "detector_dataset_sha": manifest.get("detector_dataset_sha256", summary.get("dataset_sha256", "")),
        "cuda_visible_devices": (manifest.get("gpu_snapshot", {}) or {}).get("cuda_visible_devices", ""),
        "python": manifest.get("python", ""),
        "task_success": summary.get("task_success", ""),
        "n_steps": summary.get("n_steps", telemetry.get("row_count", "")),
        "max_steps": manifest.get("max_steps", ""),
        "invalid_feature_steps": summary.get("invalid_feature_steps", telemetry.get("invalid_feature_steps_from_rows", "")),
        "first_valid_step": summary.get("first_valid_step", ""),
        "mlp_triggered": summary.get("mlp_triggered", ""),
        "mlp_emit_step": summary.get("mlp_emit_step", ""),
        "telemetry_rows": telemetry.get("row_count", 0),
        "step_sequence_continuous": telemetry.get("step_sequence_continuous", False),
        "attack_this_true_count": telemetry.get("attack_this_true_count", 0),
        "features_25d_columns_complete": telemetry.get("features_25d_columns_complete", False),
        "features_25d_finite_all_rows": telemetry.get("features_25d_finite_all_rows", False),
        "detector_fields_complete": detector_fields,
        "step_core_complete": step_core_complete,
        "visual_frames_npz": frames_npz.get("exists", False),
        "frame_index_csv": visual_fields["frame_index_csv"],
        "raw_video_available": visual_fields["rollout_raw_mp4"],
        "overlay_video_available": visual_fields["rollout_overlay_mp4"],
        "sim_state_complete": sim_state_complete,
        "sim_state_arrays": "|".join(sorted(sim_arrays.keys())),
        "privileged_fields_present": privileged_fields_present,
        "privileged_valid": privileged_valid,
        "teacher_abstain": teacher_abstain,
        "teacher_label_count": int(privileged_valid),
        "artifact_sealed": artifact_sealed,
        "same_schema_commit_6379397": same_schema,
        "legacy_compatible_but_not_same_freeze": clean_denominator and not same_schema,
        "tier": tier,
        **usable,
        "step_jsonl_rows": step_jsonl.get("row_count", 0),
        "episode_jsonl_rows": episode_jsonl.get("row_count", 0),
        "npz_member_count": len(frames_npz.get("members", [])) + len(sim_npz.get("members", [])),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def aggregate_task_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        suite = str(r.get("suite") or r.get("possible_suite") or "")
        task = str(r.get("task_idx") or "")
        if suite:
            groups[(suite, task)].append(r)
    out = []
    for (suite, task), rs in sorted(groups.items()):
        clean = [r for r in rs if r.get("usable_clean_denominator") in {True, "True", "true"}]
        tiers = Counter(r.get("tier", "") for r in rs)
        out.append({
            "suite": suite,
            "task_idx": task,
            "discovered_episode_count": len(rs),
            "unique_states": len({str(r.get("state_id")) for r in clean if str(r.get("state_id")) != ""}),
            "unique_eval_seeds": len({str(r.get("eval_seed")) for r in clean if str(r.get("eval_seed")) != ""}),
            "clean_success_count": sum(str(r.get("task_success")).lower() == "true" for r in clean),
            "clean_failure_count": sum(str(r.get("task_success")).lower() == "false" for r in clean),
            "TIER_A_count": tiers.get("TIER_A_FULL_CURRENT_COMPATIBLE", 0),
            "TIER_B_count": tiers.get("TIER_B_DETECTOR_COMPATIBLE", 0),
            "TIER_C_count": tiers.get("TIER_C_CLEAN_DENOMINATOR_ONLY", 0),
            "TIER_D_count": tiers.get("TIER_D_LEGACY_REFERENCE_ONLY", 0),
            "TIER_X_count": tiers.get("TIER_X_REJECTED", 0),
            "25D_complete_count": sum(str(r.get("features_25d_finite_all_rows")).lower() == "true" for r in clean),
            "sim_state_complete_count": sum(str(r.get("sim_state_complete")).lower() == "true" for r in clean),
            "video_available_count": sum(str(r.get("raw_video_available")).lower() == "true" for r in clean),
            "privileged_teacher_label_count": sum(str(r.get("privileged_valid")).lower() == "true" for r in clean),
        })
    return out


def duplicate_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get("suite", "")), str(r.get("task_idx", "")), str(r.get("state_id", "")),
            str(r.get("eval_seed", "")), str(r.get("condition", "")),
        )
        if all(key):
            groups[key].append(r)
    out = []
    rank = {
        "TIER_A_FULL_CURRENT_COMPATIBLE": 0,
        "TIER_B_DETECTOR_COMPATIBLE": 1,
        "TIER_C_CLEAN_DENOMINATOR_ONLY": 2,
        "TIER_D_LEGACY_REFERENCE_ONLY": 3,
        "TIER_X_REJECTED": 4,
    }
    for key, rs in sorted(groups.items()):
        if len(rs) <= 1:
            continue
        primary = sorted(rs, key=lambda r: (rank.get(str(r.get("tier")), 9), not bool_true(r.get("artifact_sealed")), str(r.get("directory_mtime", ""))))[0]
        successes = {str(r.get("task_success")) for r in rs if str(r.get("task_success"))}
        out.append({
            "suite": key[0],
            "task_idx": key[1],
            "state_id": key[2],
            "eval_seed": key[3],
            "condition": key[4],
            "duplicate_count": len(rs),
            "recommended_primary_path": primary.get("episode_path", ""),
            "success_values": "|".join(sorted(successes)),
            "has_success_failure_conflict": len(successes & {"True", "False", "true", "false"}) > 1,
            "all_paths": " | ".join(r.get("episode_path", "") for r in rs),
        })
    return out


def reusability_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for suite in sorted(NONOBJECT_SUITES):
        rs = [r for r in rows if r.get("suite") == suite]
        clean = [r for r in rs if bool_true(r.get("usable_clean_denominator"))]
        task_states = {(str(r.get("task_idx")), str(r.get("state_id"))) for r in clean if str(r.get("task_idx")) != "" and str(r.get("state_id")) != ""}
        row = {
            "suite": suite,
            "discovered": len(rs),
            "usable_clean_denominator": len(clean),
            "usable_detector_transfer_analysis": sum(bool_true(r.get("usable_detector_transfer_analysis")) for r in rs),
            "usable_teacher_relabeling": sum(bool_true(r.get("usable_teacher_relabeling")) for r in rs),
            "usable_visual_feature_extraction": sum(bool_true(r.get("usable_visual_feature_extraction")) for r in rs),
            "usable_visual_replay": sum(bool_true(r.get("usable_visual_replay")) for r in rs),
            "usable_video_manual_audit": sum(bool_true(r.get("usable_video_manual_audit")) for r in rs),
            "usable_same_schema_comparison_with_6379397": sum(bool_true(r.get("usable_same_schema_comparison_with_6379397")) for r in rs),
            "unique_clean_task_states": len(task_states),
            "missing_to_10x10_task_states": max(0, 100 - len(task_states)),
            "missing_to_10x50_task_states": max(0, 500 - len(task_states)),
        }
        out.append(row)
    return out


def write_report(path: Path, meta: dict[str, Any], root_rows: list[dict[str, Any]], master: list[dict[str, Any]], current: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    tier_counts = Counter(r.get("tier", "") for r in master)
    clean_count = sum(bool_true(r.get("usable_clean_denominator")) for r in master)
    detector_count = sum(bool_true(r.get("usable_detector_transfer_analysis")) for r in master)
    teacher_count = sum(bool_true(r.get("usable_teacher_relabeling")) for r in master)
    lines = [
        "# Non-Object Existing CLEAN Data Audit - 2026-06-19",
        "",
        "## Scope",
        "",
        "Read-only metadata-only audit of `/data/liuyu/outputs`. No GPU, OpenVLA load, rollout, file mutation, file move, or large MP4/NPZ full SHA was performed.",
        "",
        "## Run Metadata",
        "",
        "```json",
        json.dumps(meta, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        f"- Candidate sentinel directories discovered: {len(root_rows)}",
        f"- Clean denominator usable episodes: {clean_count}",
        f"- Detector-transfer usable episodes: {detector_count}",
        f"- Teacher/offline relabeling usable episodes: {teacher_count}",
        f"- Tier counts: {dict(tier_counts)}",
        "",
        "## Current 300 Queue Snapshot",
        "",
        f"- Active rows observed: {len(current)}",
        f"- Active COMPLETE stable rows: {sum(bool_true(r.get('eligible_active_complete_snapshot')) for r in current)}",
        f"- Active RUNNING rows: {sum(str(r.get('status')) == 'RUNNING' for r in current)}",
        "",
        "Current active root is reported separately and not mixed with legacy counts.",
        "",
        "## Reusability By Suite",
        "",
        "| Suite | Clean usable | Detector usable | Teacher relabel usable | Unique task-states | Missing 10x10 | Missing 10x50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['suite']} | {r['usable_clean_denominator']} | {r['usable_detector_transfer_analysis']} | "
            f"{r['usable_teacher_relabeling']} | {r['unique_clean_task_states']} | "
            f"{r['missing_to_10x10_task_states']} | {r['missing_to_10x50_task_states']} |"
        )
    lines += [
        "",
        "## Decision Table",
        "",
        "- Directly reusable: `TIER_A_FULL_CURRENT_COMPATIBLE` and `TIER_B_DETECTOR_COMPATIBLE`, depending on whether same-freeze comparison is required.",
        "- Detector analysis but not Teacher: episodes with `usable_detector_transfer_analysis=true` and `usable_teacher_relabeling=false`.",
        "- Clean SR only: `TIER_C_CLEAN_DENOMINATOR_ONLY`.",
        "- Rerun required: `TIER_X_REJECTED`, partial/current-running episodes, attack-contaminated outputs, and any episode with unresolved condition or suite.",
        "- Offline relabel without rerun: episodes with generic sim-state or complete frames but `privileged_valid=false`.",
        "",
        "## Claim Limits",
        "",
        "- This audit does not prove detector timing transfer.",
        "- Anonymous pose streams were not treated as object identity evidence.",
        "- Different source commits are not same-freeze compatible unless explicitly marked.",
        "- Active queue COMPLETE snapshots are separated from legacy inventory.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-root", required=True)
    ap.add_argument("--active-root", required=True)
    ap.add_argument("--reference-schema-commit", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--stability-window-sec", type=int, default=60)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.metadata_only or not args.no_gpu:
        raise SystemExit("This audit requires --metadata-only --no-gpu")
    outputs_root = Path(args.outputs_root)
    active_root = Path(args.active_root)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output-dir is non-empty: {out}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    start = now_iso()
    active_start, active_meta_start = active_progress(active_root, args.stability_window_sec)
    episode_dirs, scan_counters = discover_episode_dirs(outputs_root)
    root_rows: list[dict[str, Any]] = []
    docs_by_ep: dict[str, dict[str, dict[str, Any]]] = {}
    for ep in episode_dirs:
        root_row, docs = classify_episode(ep, outputs_root, active_root, args.stability_window_sec)
        root_rows.append(root_row)
        docs_by_ep[str(ep)] = docs
    master = []
    schema = []
    for row in root_rows:
        clean_candidate = (
            row.get("possible_suite") in NONOBJECT_SUITES
            and row.get("possible_condition") == "CLEAN"
            and not bool_true(row.get("manifest_attack_enabled"))
            and not bool_true(row.get("manifest_vis_enabled"))
            and not bool_true(row.get("manifest_rand_enabled"))
            and not row.get("is_partial")
        )
        if clean_candidate:
            audited = audit_clean_candidate(Path(row["episode_path"]), row, docs_by_ep[row["episode_path"]], args.reference_schema_commit)
        else:
            audited = {
                **row,
                "suite": row.get("possible_suite", ""),
                "condition": row.get("possible_condition", "") or "CONDITION_UNVERIFIED",
                "tier": "TIER_X_REJECTED" if row.get("is_partial") or row.get("possible_condition") != "CLEAN" else "TIER_D_LEGACY_REFERENCE_ONLY",
                "usable_clean_denominator": False,
                "usable_detector_transfer_analysis": False,
                "usable_teacher_relabeling": False,
                "usable_visual_feature_extraction": False,
                "usable_visual_replay": False,
                "usable_video_manual_audit": False,
                "usable_same_schema_comparison_with_6379397": False,
            }
        master.append(audited)
        schema.append({
            "episode_path": audited.get("episode_path", ""),
            "suite": audited.get("suite", ""),
            "condition": audited.get("condition", ""),
            "tier": audited.get("tier", ""),
            "step_core_complete": audited.get("step_core_complete", False),
            "features_25d_columns_complete": audited.get("features_25d_columns_complete", False),
            "features_25d_finite_all_rows": audited.get("features_25d_finite_all_rows", False),
            "detector_fields_complete": audited.get("detector_fields_complete", False),
            "sim_state_complete": audited.get("sim_state_complete", False),
            "visual_frames_npz": audited.get("visual_frames_npz", False),
            "raw_video_available": audited.get("raw_video_available", False),
            "privileged_valid": audited.get("privileged_valid", False),
            "teacher_abstain": audited.get("teacher_abstain", False),
            "same_schema_commit_6379397": audited.get("same_schema_commit_6379397", False),
        })
    active_end, active_meta_end = active_progress(active_root, args.stability_window_sec)
    task_matrix = aggregate_task_matrix(master)
    dupes = duplicate_conflicts(master)
    summary = reusability_summary(master)
    end = now_iso()
    meta = {
        "audit_source_commit": run_text(["git", "rev-parse", "HEAD"]),
        "reference_schema_commit": args.reference_schema_commit,
        "server_hostname": socket.gethostname(),
        "audit_start_time": start,
        "audit_end_time": end,
        "outputs_root": str(outputs_root),
        "active_root": str(active_root),
        "active_start": active_meta_start,
        "active_end": active_meta_end,
        "scan_counters": scan_counters,
        "sentinels": sorted(SENTINELS),
        "rules": {
            "metadata_only": True,
            "no_gpu": True,
            "condition_unverified_not_clean": True,
            "anonymous_pose_not_identity": True,
            "active_root_separate": True,
            "no_large_mp4_npz_full_sha": True,
        },
    }
    write_csv(out / "tables" / "nonobject_output_root_inventory_20260619.csv", root_rows)
    write_csv(out / "tables" / "nonobject_episode_master_ledger_20260619.csv", master)
    write_csv(out / "tables" / "nonobject_schema_coverage_20260619.csv", schema)
    write_csv(out / "tables" / "nonobject_duplicate_conflicts_20260619.csv", dupes)
    write_csv(out / "tables" / "nonobject_task_state_matrix_20260619.csv", task_matrix)
    write_csv(out / "tables" / "nonobject_reusability_summary_20260619.csv", summary)
    write_csv(out / "tables" / "current_cross_suite_300_progress_snapshot.csv", active_end)
    report_json = {"metadata": meta, "reusability_summary": summary, "tier_counts": dict(Counter(r.get("tier", "") for r in master))}
    (out / "reports" / "nonobject_existing_data_audit_20260619.json").write_text(json.dumps(report_json, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_report(out / "reports" / "NONOBJECT_EXISTING_DATA_AUDIT_20260619.md", meta, root_rows, master, active_end, summary)
    print(json.dumps({
        "result": "NONOBJECT_METADATA_AUDIT_DONE",
        "output_dir": str(out),
        "candidate_dirs": len(root_rows),
        "clean_usable": sum(bool_true(r.get("usable_clean_denominator")) for r in master),
        "detector_usable": sum(bool_true(r.get("usable_detector_transfer_analysis")) for r in master),
        "active_complete_stable": sum(bool_true(r.get("eligible_active_complete_snapshot")) for r in active_end),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
