#!/usr/bin/env python3
"""Inventory server artifacts and recover clean-state provenance for Stage-B L3.

This script is intentionally read-only with respect to /data/liuyu/outputs.  It
only writes compact CSV/Markdown audit artifacts under the current repository.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OUTPUTS = Path("/data/liuyu/outputs")
R1 = OUTPUTS / "milestone_r1_official_eval_20260526"
R2 = OUTPUTS / "milestone_r2_official_v4_object_alignment_20260526"
FULL10 = OUTPUTS / "milestone_1d_object_mujoco237_compat_20260526/object_full_10x10"
K5C = OUTPUTS / "stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e"
REPO = Path.cwd()

ARTIFACT_INDEX = REPO / "tables/server_artifact_index_20260611.csv"
RUNNER_TAXONOMY = REPO / "tables/runner_taxonomy_20260611.csv"
OBJECT100_INVENTORY = REPO / "tables/object100_clean_success_state_inventory_verified_20260611.csv"
LEVEL3_SHORTLIST = REPO / "tables/level3_clean_success_state_shortlist_20260611.csv"
STALE_ARTIFACTS = REPO / "tables/stale_or_dangerous_artifacts_20260611.csv"
TAXONOMY_REPORT = REPO / "reports/RUNNER_TAXONOMY_AND_CLAIM_BOUNDARY_20260611.md"
OBJECT100_REPORT = REPO / "reports/OBJECT100_CLEAN_STATE_RECOVERY_20260611.md"
REGISTRY_REPORT = REPO / "reports/SERVER_SOURCE_OF_TRUTH_REGISTRY_20260611.md"


TASK_RE = re.compile(r"pick_up_the_(.*?)_and_place_it_in_the_basket")
OBJ_RUN_RE = re.compile(r"obj_(?P<task>.+)_s(?P<state>\d+)$")
WIN_RE = re.compile(r"(?:^|[_-])w(?P<ws>\d+)_(?P<we>\d+)(?:[_-]|$)")
STATE_RE = re.compile(r"(?:^|[_-])s(?P<state>\d+)(?:[_-]|$)")
JOB_RE = re.compile(r"(?:^|[_-])job(?P<job>\d+)(?:[_-]|$)")
COND_PATTERNS = [
    ("vis_pgd", re.compile(r"vis[_-]?pgd|vis_pgd", re.I)),
    ("random_linf", re.compile(r"random[_-]?linf|random_linf|rand", re.I)),
    ("oracle", re.compile(r"oracle", re.I)),
    ("clean", re.compile(r"clean", re.I)),
]
KEYWORD_RE = re.compile(
    r"S16R|S17|S18|S19|S20|official|object100|clean|center_crop|"
    r"libero_full4_clean_official_aligned_eager_10states_20260525|"
    r"object_official_script_100_manifest|object_official_corrected_100_manifest_reconstructed|"
    r"object_v4_100_manifest_reconstructed|object_official_vs_v4_per_episode_diff",
    re.I,
)


def ensure_dirs() -> None:
    (REPO / "tables").mkdir(exist_ok=True)
    (REPO / "reports").mkdir(exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return {}


def read_jsonl_first(path: Path) -> Dict[str, Any]:
    try:
        with path.open(errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        pass
    return {}


def read_csv_rows(path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with path.open(newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(dict(row))
                if limit is not None and i + 1 >= limit:
                    break
    except Exception:
        pass
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fieldnames})


def task_key_from_task_name(task_name: str) -> str:
    m = TASK_RE.search(str(task_name))
    if m:
        return m.group(1)
    text = str(task_name)
    if text.startswith("object_pick_up_the_"):
        return text.replace("object_pick_up_the_", "").replace("_and_place_it_in_the_basket", "")
    return text.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "")


def task_name_from_key(task: str) -> str:
    return f"pick_up_the_{task}_and_place_it_in_the_basket"


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def as_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def detect_stage(path: Path) -> str:
    s = str(path)
    for key in ["S20", "S19", "S18", "S17", "S16R"]:
        if key.lower() in s.lower():
            return key
    if "milestone_r1_official_eval" in s:
        return "R1_official"
    if "milestone_r2_official_v4_object_alignment" in s:
        return "R2_official_v4_alignment"
    if "object_full_10x10" in s:
        return "object100_v4_full10x10"
    if "stageb_v1_1" in s:
        return "stageb_v1_1"
    return ""


def detect_runner_family(path: Path, data: Dict[str, Any]) -> str:
    text = " ".join([str(path), str(data.get("command", "")), str(data.get("runner_type", "")), str(data.get("version", ""))]).lower()
    if "run_s9b_phase1_runner_attack_port" in text or "phase1_runner" in text:
        return "phase_runner"
    if "run_s20c_official_l3_runner" in text:
        return "s20c_official_fixed_window_l3"
    if "v4_run_eval_openvla" in text or "v4_runner" in text or "object_full_10x10" in text:
        return "official_v4_runner"
    if "official_corrected" in text or "milestone_r1_official_eval" in text:
        return "official_corrected_runner"
    return ""


def metadata_from_name(path: Path) -> Dict[str, Any]:
    name = path.name
    out: Dict[str, Any] = {}
    m = OBJ_RUN_RE.search(path.parent.name)
    if not m:
        m = OBJ_RUN_RE.search(path.stem)
    if m:
        out["task"] = m.group("task")
        out["state_id"] = m.group("state")
    m = WIN_RE.search(name)
    if m:
        out["window_start"] = m.group("ws")
        out["window_end"] = m.group("we")
    m = STATE_RE.search(name)
    if m and "state_id" not in out:
        out["state_id"] = m.group("state")
    m = JOB_RE.search(name)
    if m:
        out["job_id"] = m.group("job")
    for cond, rx in COND_PATTERNS:
        if rx.search(name):
            out["condition"] = cond
            break
    return out


def enrich_from_data(row: Dict[str, Any], data: Dict[str, Any]) -> None:
    def first(*keys: str) -> Any:
        for k in keys:
            if data.get(k) not in (None, ""):
                return data.get(k)
        return ""

    task = first("actual_task_key", "task_key", "task", "task_name", "task_id")
    if task:
        task_s = str(task)
        row["task"] = row.get("task") or task_key_from_task_name(task_s)
        if "actual_task_key" in data:
            row["actual_task_key"] = data.get("actual_task_key")
    row["state_id"] = row.get("state_id") or first("state_id", "seed")
    row["window_start"] = row.get("window_start") or first("window_start")
    row["window_end"] = row.get("window_end") or first("window_end")
    row["condition"] = row.get("condition") or first("condition", "trigger_name")
    row["seed"] = first("seed")
    row["attack_seed"] = first("attack_seed")
    row["random_control_seed"] = first("random_control_seed", "random_seed")
    row["job_id"] = row.get("job_id") or first("job_id")
    row["git_commit"] = first("git_commit", "code_git_commit")
    command = str(first("command"))
    row["center_crop"] = first("center_crop")
    if not row["center_crop"] and "--center_crop" in command:
        row["center_crop"] = "True"
    row["libero_preprocess_backend"] = first("libero_preprocess_backend", "preprocess_backend")
    if not row["libero_preprocess_backend"] and "official_pil_lanczos" in command:
        row["libero_preprocess_backend"] = "official_pil_lanczos"
    row["postprocess_gripper"] = first("postprocess_gripper")
    if not row["postprocess_gripper"] and "--postprocess_gripper" in command:
        row["postprocess_gripper"] = "True"
    row["success_metric"] = first("success_metric")
    if not row["success_metric"]:
        m = re.search(r"--success_metric\s+(\S+)", command)
        if m:
            row["success_metric"] = m.group(1)
    row["max_steps"] = first("max_steps", "max_steps_override")
    if not row["max_steps"]:
        m = re.search(r"--max_steps(?:_override)?\s+(\d+)", command)
        if m:
            row["max_steps"] = m.group(1)
    row["full_episode"] = "True" if "full" in str(row.get("artifact_path", "")).lower() or first("full_episode") else first("full_episode")


def artifact_roots() -> List[Path]:
    roots = [p for p in [R1, R2, FULL10, K5C] if p.exists()]
    if OUTPUTS.exists():
        for p in OUTPUTS.iterdir():
            if p.is_dir() and KEYWORD_RE.search(p.name) and p not in roots:
                roots.append(p)
    return roots


def build_artifact_index() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    max_files = 50000
    for root in artifact_roots():
        for dirpath, dirnames, filenames in os.walk(root):
            dpath = Path(dirpath)
            # Record frame folders without indexing every frame.
            image_files = [fn for fn in filenames if fn.lower().endswith((".png", ".jpg", ".jpeg"))]
            if image_files:
                image_files_sorted = sorted(image_files)
                rows.append({
                    "artifact_path": str(dpath),
                    "artifact_type": "frame_dir",
                    "file_name": dpath.name,
                    "file_size": "",
                    "modified_time": datetime.fromtimestamp(dpath.stat().st_mtime).isoformat(),
                    "video_dir": str(dpath),
                    "num_frames": len(image_files_sorted),
                    "first_frame": image_files_sorted[0],
                    "last_frame": image_files_sorted[-1],
                    "detected_stage": detect_stage(dpath),
                    "detected_runner_family": "",
                    "source_confidence": "medium",
                    "notes": "Frame directory counted only; frames not read.",
                })
                filenames = [fn for fn in filenames if not fn.lower().endswith((".png", ".jpg", ".jpeg"))]
            for fn in filenames:
                path = dpath / fn
                if path in seen:
                    continue
                seen.add(path)
                if len(rows) >= max_files:
                    break
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in {".pt", ".pth", ".ckpt", ".npy", ".npz", ".mp4", ".avi", ".mov"}:
                    artifact_type = suffix.lstrip(".")
                    data = {}
                elif suffix == ".json":
                    artifact_type = "json"
                    data = read_json(path)
                elif suffix == ".jsonl":
                    artifact_type = "jsonl"
                    data = read_jsonl_first(path)
                elif suffix == ".csv":
                    artifact_type = "csv"
                    csv_rows = read_csv_rows(path, limit=1)
                    data = csv_rows[0] if csv_rows else {}
                elif suffix in {".log", ".txt", ".md", ".yaml", ".yml", ".sh", ".py"}:
                    artifact_type = suffix.lstrip(".")
                    data = {}
                else:
                    artifact_type = suffix.lstrip(".") or "file"
                    data = {}
                meta = metadata_from_name(path)
                row: Dict[str, Any] = {
                    "artifact_path": str(path),
                    "artifact_type": artifact_type,
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "detected_stage": detect_stage(path),
                    "detected_runner_family": detect_runner_family(path, data),
                    "task": meta.get("task", ""),
                    "actual_task_key": "",
                    "state_id": meta.get("state_id", ""),
                    "window_start": meta.get("window_start", ""),
                    "window_end": meta.get("window_end", ""),
                    "condition": meta.get("condition", ""),
                    "seed": "",
                    "attack_seed": "",
                    "random_control_seed": "",
                    "job_id": meta.get("job_id", ""),
                    "git_commit": "",
                    "center_crop": "",
                    "libero_preprocess_backend": "",
                    "postprocess_gripper": "",
                    "success_metric": "",
                    "full_episode": "",
                    "max_steps": "",
                    "video_dir": "",
                    "num_frames": "",
                    "first_frame": "",
                    "last_frame": "",
                    "source_confidence": "medium" if data else "low",
                    "notes": "",
                }
                enrich_from_data(row, data)
                if "task-level" in str(data).lower() or "summary_by_task" in path.name:
                    row["notes"] = "Task-level summary; do not treat as per-state evidence."
                    row["source_confidence"] = "low"
                rows.append(row)
    return rows


def runner_taxonomy_rows() -> List[Dict[str, Any]]:
    rows = [
        {
            "runner": "Phase runner",
            "script_path": "scripts/stageb/run_s9b_phase1_runner_attack_port.py",
            "detected_exists": str((REPO / "scripts/stageb/run_s9b_phase1_runner_attack_port.py").exists()),
            "preprocessing_path": "OpenVLA processor path used by phase runner; not full official/v4 eval contract.",
            "model_decode_path": "OpenVLA generate/decode inside phase runner.",
            "action_postprocess_path": "env action recomputed and stepped by phase runner; supports command/qpos bridge checks.",
            "success_detection_path": "Not accepted for official task-level success claims.",
            "state_selection_logic": "LIBERO object task init state by state_id/env_seed; actual_task_key assertion present.",
            "warmup_wait_steps": "Not official v4 num_steps_wait contract.",
            "output_schema": "Stage-B phase traces/summaries with actual_task_key, qpos/action bridge fields.",
            "valid_for": "generated OPEN command; matched VIS/RAND command comparison; qpos physical bridge",
            "not_valid_for": "official task success/failure; task-level SR; Level-3 task effect unless separately aligned",
            "required_flags_config": "N/A for Level 3 official claims",
            "safe_claim_level": "Level 1/2 only",
            "notes": "S20b full-episode phase-runner videos remain diagnostics only.",
        },
        {
            "runner": "Official/v4 runner",
            "script_path": "scripts/v4_run_eval_openvla.py",
            "detected_exists": str((REPO / "scripts/v4_run_eval_openvla.py").exists()),
            "preprocessing_path": "prepare_openvla_image with --libero_official_preprocess --center_crop --libero_preprocess_backend official_pil_lanczos",
            "model_decode_path": "decode_with_scores/model.generate through v4 runner.",
            "action_postprocess_path": "postprocess_openvla_action_for_libero with --postprocess_gripper",
            "success_detection_path": "env.check_success plus done; current Level-3 contract requires --success_metric check_success.",
            "state_selection_logic": "--state_ids or deterministic_init_states over LIBERO init states",
            "warmup_wait_steps": "--num_steps_wait 10 required for corrected official alignment",
            "output_schema": "run_manifest.json, summary.csv, episode_records.jsonl, step_records.jsonl",
            "valid_for": "official-aligned clean success; state-level success; Level-3 task/contact audit if fixed-window attack is integrated correctly",
            "not_valid_for": "phase-runner-only command bridge unless attack integration logs matched command/qpos fields",
            "required_flags_config": "--center_crop; --libero_preprocess_backend official_pil_lanczos; --postprocess_gripper; --success_metric check_success; --num_steps_wait 10; attention_backend=eager when applicable",
            "safe_claim_level": "Official clean baseline and candidate Level 3 audit source",
            "notes": "Object-100 v4 full10x10 uses center_crop=True, official_pil_lanczos, postprocess_gripper, num_steps_wait=10 in command manifests; its recorded success_metric is inventoried explicitly and must not silently define the next Level-3 contract.",
        },
        {
            "runner": "S20c official fixed-window L3 runner",
            "script_path": "scripts/stageb/run_s20c_official_l3_runner.py",
            "detected_exists": str((REPO / "scripts/stageb/run_s20c_official_l3_runner.py").exists()),
            "preprocessing_path": "prepare_openvla_image official_pil_lanczos; code shows attack adapter preprocess_kwargs center_crop=False but runtime image path uses center_crop=True",
            "model_decode_path": "OpenVLA official/v4-style decode plus fixed-window VIS/RAND branches",
            "action_postprocess_path": "official_postprocess normalize/invert before env.step",
            "success_detection_path": "env.check_success in summary/trace",
            "state_selection_logic": "--state_id/--state_ids explicit",
            "warmup_wait_steps": "NUM_STEPS_WAIT in script",
            "output_schema": "S20c trace/summary JSON/CSV if run",
            "valid_for": "Potential Level-3 fixed-window smoke only after center_crop and preprocess kwargs are proven consistent",
            "not_valid_for": "Any result from center_crop-missing smoke; phase-runner task-effect claim",
            "required_flags_config": "center_crop=True end-to-end; official_pil_lanczos; postprocess_gripper; env.check_success; actual task/state provenance",
            "safe_claim_level": "Pending; audit before use",
            "notes": "Current code contains historical center_crop=False in attacker preprocess kwargs, which must be treated as a danger flag unless patched/proven unused.",
        },
    ]
    return rows


def source_path_for_inventory(runner_type: str, row: Dict[str, Any]) -> str:
    if runner_type == "official_script_raw":
        return str(R1 / "tables/object_official_script_100_manifest.csv")
    if runner_type == "official_corrected_reconstructed":
        return str(R2 / "tables/object_official_corrected_100_manifest_reconstructed.csv")
    if runner_type == "v4_runner":
        return str(R2 / "tables/object_v4_100_manifest_reconstructed.csv")
    if runner_type == "v4_full10x10_raw":
        return str(FULL10 / row.get("run_id", ""))
    return ""


def object100_inventory_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_row(**kw: Any) -> None:
        defaults = {
            "task": "", "task_id": "", "state_id": "", "episode_id": "",
            "runner_type": "", "success": "", "success_check": "", "success_done": "",
            "timeout": "", "num_steps": "", "success_step": "", "failure_phase": "",
            "seed": "", "center_crop": "", "attention_backend": "",
            "libero_preprocess_backend": "", "postprocess_gripper": "",
            "deterministic_init_states": "", "state_id_source": "",
            "source_path": "", "source_confidence": "", "notes": "",
        }
        defaults.update(kw)
        rows.append(defaults)

    # R1 official raw manifest: explicit per-state where present.
    for row in read_csv_rows(R1 / "tables/object_official_script_100_manifest.csv"):
        task = task_key_from_task_name(row.get("task_name", ""))
        add_row(
            task=task,
            task_id="",
            state_id=row.get("state_id", ""),
            episode_id=row.get("state_id", ""),
            runner_type="official_script_raw",
            success=row.get("success", ""),
            success_check=row.get("success", ""),
            success_done="",
            timeout="",
            num_steps=row.get("num_steps", ""),
            success_step=row.get("num_steps", "") if parse_bool(row.get("success")) else "",
            failure_phase=row.get("failure_phase", ""),
            seed="0",
            center_crop="True (from corrected official context; verify logs if needed)",
            attention_backend="eager (from R2 reconstruction context)",
            libero_preprocess_backend="official_pil_lanczos (corrected official context)",
            postprocess_gripper="True (official normalize+invert)",
            deterministic_init_states="True",
            state_id_source="explicit state_id column",
            source_path=str(R1 / "tables/object_official_script_100_manifest.csv"),
            source_confidence="high",
            notes="Raw per-state official manifest row.",
        )

    # Official corrected reconstructed rows: includes low-confidence inferred W45 rows.
    for row in read_csv_rows(R2 / "tables/object_official_corrected_100_manifest_reconstructed.csv"):
        task = task_key_from_task_name(row.get("task_name", ""))
        conf = row.get("source_confidence", "") or "low"
        notes = row.get("notes", "")
        add_row(
            task=task,
            task_id=row.get("task_id", ""),
            state_id=row.get("state_id", ""),
            episode_id=row.get("state_id", ""),
            runner_type="official_corrected_reconstructed",
            success=row.get("success", ""),
            success_check=row.get("success", ""),
            success_done="",
            timeout="",
            num_steps=row.get("num_steps", ""),
            success_step=row.get("num_steps", "") if parse_bool(row.get("success")) else "",
            failure_phase=row.get("failure_phase", ""),
            seed=row.get("seed", ""),
            center_crop=row.get("center_crop", ""),
            attention_backend=row.get("attention_backend", ""),
            libero_preprocess_backend="official_pil_lanczos",
            postprocess_gripper="True",
            deterministic_init_states="True",
            state_id_source="explicit state_id column; some success values inferred from task SR",
            source_path=str(R2 / "tables/object_official_corrected_100_manifest_reconstructed.csv"),
            source_confidence=conf,
            notes=notes or "Reconstructed official corrected row.",
        )

    # V4 reconstructed manifest.
    for row in read_csv_rows(R2 / "tables/object_v4_100_manifest_reconstructed.csv"):
        task = task_key_from_task_name(row.get("task_name", ""))
        add_row(
            task=task,
            task_id=row.get("task_id", ""),
            state_id=row.get("state_id", ""),
            episode_id=row.get("state_id", ""),
            runner_type="v4_runner_reconstructed",
            success=row.get("success", ""),
            success_check=row.get("success", ""),
            success_done="",
            timeout="",
            num_steps="",
            success_step="",
            failure_phase=row.get("failure_phase", ""),
            seed=row.get("seed", ""),
            center_crop=row.get("center_crop", ""),
            attention_backend=row.get("attention_backend", ""),
            libero_preprocess_backend="official_pil_lanczos",
            postprocess_gripper="True",
            deterministic_init_states="True",
            state_id_source="explicit run_id/state_id reconstruction",
            source_path=str(R2 / "tables/object_v4_100_manifest_reconstructed.csv"),
            source_confidence=row.get("source_confidence", ""),
            notes=row.get("notes", ""),
        )

    # Raw full10x10 dirs provide strongest v4 per-state evidence and num_steps.
    if FULL10.exists():
        for d in sorted(FULL10.glob("obj_*_s*")):
            m = OBJ_RUN_RE.search(d.name)
            if not m:
                continue
            task = m.group("task")
            state_id = m.group("state")
            manifest = read_json(d / "run_manifest.json")
            episode = read_jsonl_first(d / "episode_records.jsonl")
            summary = read_csv_rows(d / "summary.csv", limit=1)
            summary_row = summary[0] if summary else {}
            command = str(manifest.get("command", ""))
            success = episode.get("success")
            add_row(
                task=task,
                task_id=episode.get("task_id", summary_row.get("task_id", "")),
                state_id=state_id,
                episode_id=episode.get("episode_id", "0"),
                runner_type="v4_full10x10_raw",
                success=str(success),
                success_check="",
                success_done="",
                timeout=str(episode.get("timeout", "")),
                num_steps=episode.get("num_steps", ""),
                success_step=episode.get("num_steps", "") if success is True else "",
                failure_phase=episode.get("failure_phase", ""),
                seed=manifest.get("seed", summary_row.get("seed", "")),
                center_crop=str("--center_crop" in command),
                attention_backend="eager" if "OPENVLA_ATTN_IMPLEMENTATION=eager" in command or "eager" in command else "unknown",
                libero_preprocess_backend="official_pil_lanczos" if "official_pil_lanczos" in command else "unknown",
                postprocess_gripper=str("--postprocess_gripper" in command),
                deterministic_init_states=str("--deterministic_init_states" in command),
                state_id_source="explicit --state_ids in run_manifest command",
                source_path=str(d),
                source_confidence="high",
                notes="Raw object_full_10x10 run dir with episode_records.jsonl.",
            )
    return rows


def best_clean_state_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    priority = {"v4_full10x10_raw": 4, "official_script_raw": 3, "official_corrected_reconstructed": 2, "v4_runner_reconstructed": 1}
    conf_pri = {"high": 3, "medium": 2, "low": 1}
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("task")), str(r.get("state_id")))
        if not key[0] or not key[1]:
            continue
        score = (conf_pri.get(str(r.get("source_confidence")), 0), priority.get(str(r.get("runner_type")), 0))
        prev = best.get(key)
        if prev is None or score > prev["_score"]:
            rr = dict(r)
            rr["_score"] = score
            best[key] = rr
    return best


def level3_shortlist_rows(inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = best_clean_state_map(inventory)
    candidates: List[Dict[str, Any]] = []
    preferred = {("ketchup", "1"): 100, ("ketchup", "3"): 95, ("tomato_sauce", "3"): 100, ("tomato_sauce", "5"): 95}
    for (task, sid), r in best.items():
        success = parse_bool(r.get("success"))
        if success is not True:
            continue
        steps = as_int(r.get("num_steps"))
        if steps is None:
            step_ok = False
        else:
            step_ok = steps < 350
        if task not in {"ketchup", "tomato_sauce"}:
            continue
        base = preferred.get((task, sid), 50)
        if r.get("source_confidence") == "high":
            base += 20
        if step_ok:
            base += 10
        if r.get("runner_type") == "v4_full10x10_raw":
            base += 10
        candidates.append({
            "task": task,
            "state_id": sid,
            "clean_success": str(success),
            "clean_steps": r.get("num_steps", ""),
            "source_confidence": r.get("source_confidence", ""),
            "recommended_rank": base,
            "reason": f"clean_success={success}; steps={r.get('num_steps','')}; runner={r.get('runner_type','')}; confidence={r.get('source_confidence','')}",
            "suggested_window_source": "Use v4_full10x10 step_records/video to locate grasp/contact/transport windows.",
            "notes": r.get("notes", ""),
        })
    candidates.sort(key=lambda x: (-int(x["recommended_rank"]), x["task"], int(x["state_id"])))
    for i, row in enumerate(candidates, 1):
        row["recommended_rank"] = i
    return candidates


def stale_artifact_rows(artifact_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def flag(path: str, artifact_type: str, reason: str, risk: str, action: str) -> None:
        rows.append({
            "artifact_path": path,
            "artifact_type": artifact_type,
            "risk_category": risk,
            "reason": reason,
            "recommended_action": action,
        })

    for r in artifact_rows:
        p = str(r.get("artifact_path", ""))
        pl = p.lower()
        notes = str(r.get("notes", ""))
        if any(k in pl for k in ["old", "overnight", "pre_v1", "pre-v1", "44-row", "patched_rerun", "hotfix"]):
            flag(p, r.get("artifact_type", ""), "Old/pre-v1.1/patched-rerun artifact name.", "old_or_pre_v1_1", "Quarantine for current Level-3 planning.")
        if "run_s20b" in pl or "s20b" in pl:
            flag(p, r.get("artifact_type", ""), "S20b phase-runner full episode artifact; diagnostic only.", "phase_runner_used_as_level3", "Do not use as official task-effect evidence.")
        if r.get("detected_runner_family") == "phase_runner" and str(r.get("full_episode", "")).lower() == "true":
            flag(p, r.get("artifact_type", ""), "Phase runner full episode cannot support Level-3 task-effect claim.", "phase_runner_used_as_level3", "Use official/v4 runner for Level-3.")
        if r.get("actual_task_key", "") == "" and r.get("detected_runner_family") == "phase_runner":
            flag(p, r.get("artifact_type", ""), "Phase-runner artifact missing actual_task_key.", "missing_actual_task_key", "Do not use for task-label claims.")
        if "official" in pl and str(r.get("center_crop", "")).lower() in {"", "false"}:
            flag(p, r.get("artifact_type", ""), "Artifact appears official-aligned but center_crop is missing/false.", "center_crop_missing", "Treat as non-official-aligned unless separately proven.")
        if "random_seed_str" in notes or "attack_seed + job_id" in notes:
            flag(p, r.get("artifact_type", ""), "Legacy random seed construction risk.", "legacy_random_seed", "Require explicit random_control_seed.")
        if r.get("state_id", "") == "" and any(k in pl for k in ["state", "s20", "stageb", "object"]):
            flag(p, r.get("artifact_type", ""), "State-level-looking artifact has unknown state_id.", "unknown_state_id", "Do not use for state-level claims.")
        if "task-level" in notes.lower() or "summary_by_task" in pl:
            flag(p, r.get("artifact_type", ""), "Task-level SR only, not per-state evidence.", "task_sr_as_state", "Do not fabricate per-state success.")
        if "object_official_corrected_100_manifest_reconstructed" in pl:
            flag(p, r.get("artifact_type", ""), "Reconstructed official rows include W45 overwritten low-confidence task-SR inference.", "reconstructed_low_confidence", "Use source_confidence column; do not treat low-confidence inferred rows as raw.")

    # Add explicit known danger entries even if not indexed.
    explicit = [
        ("old 44-row / old overnight labels / pre-v1.1 traces", "known_class", "Pre-RC1a open semantics / trace schema not current.", "old_or_pre_v1_1", "Quarantine."),
        ("S20c center_crop=False smoke", "known_class", "Initial S20c official-aligned smoke missed --center_crop.", "center_crop_missing", "Do not treat as official-aligned corrected result."),
        ("pre-S16R task-label artifacts", "known_class", "Task mapping bug risk before actual_task_key discipline.", "old_task_mapping_bug_labels", "Do not use for current task/state claims."),
    ]
    for e in explicit:
        flag(*e)
    # Deduplicate.
    unique = {}
    for row in rows:
        key = (row["artifact_path"], row["risk_category"], row["reason"])
        unique[key] = row
    return list(unique.values())


def write_runner_taxonomy_report(rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Runner Taxonomy and Claim Boundary",
        "",
        "**Date**: 2026-06-11",
        "",
        "## Claim Boundary",
        "",
        "- Phase-runner evidence supports Level 1/2 command/qpos bridge claims only.",
        "- Official/v4 runner evidence is required for official clean success and any Level-3 task-effect audit.",
        "- S20c fixed-window L3 runner remains pending until center-crop and preprocessing consistency are proven end to end.",
        "",
        "## Runner Table",
        "",
        "| Runner | Safe claim level | Valid for | Not valid for | Required config |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['runner']} | {r['safe_claim_level']} | {r['valid_for']} | {r['not_valid_for']} | {r['required_flags_config']} |")
    TAXONOMY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_object100_report(inventory: List[Dict[str, Any]], shortlist: List[Dict[str, Any]]) -> None:
    by_runner = Counter(r["runner_type"] for r in inventory)
    best = best_clean_state_map(inventory)
    def state_line(task: str) -> List[str]:
        lines = []
        for sid in range(10):
            r = best.get((task, str(sid)))
            if not r:
                lines.append(f"- {task} s{sid}: missing")
            else:
                lines.append(
                    f"- {task} s{sid}: success={r.get('success')} steps={r.get('num_steps')} "
                    f"runner={r.get('runner_type')} confidence={r.get('source_confidence')} source={r.get('source_path')}"
                )
        return lines
    lines = [
        "# Object-100 Clean State Recovery",
        "",
        "**Date**: 2026-06-11",
        "",
        "## Inputs Audited",
        "",
        f"- `{R1 / 'tables/object_official_script_100_manifest.csv'}`",
        f"- `{R2 / 'tables/object_official_corrected_100_manifest_reconstructed.csv'}`",
        f"- `{R2 / 'tables/object_v4_100_manifest_reconstructed.csv'}`",
        f"- `{R2 / 'tables/object_official_vs_v4_per_episode_diff.csv'}`",
        f"- `{FULL10}`",
        "",
        "## Inventory Counts",
        "",
    ]
    for k, v in sorted(by_runner.items()):
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Ketchup State Provenance",
        "",
        "Ketchup official-corrected R2 rows are low-confidence because the W45 CSV was overwritten and values were inferred from task-level SR. Raw per-state evidence is available from the v4 `object_full_10x10` run dirs.",
        "",
    ] + state_line("ketchup") + [
        "",
        "## Tomato Sauce State Provenance",
        "",
        "Tomato sauce has high-confidence official-corrected raw rows for states 0-9 in R1/R2 and high-confidence v4 full10x10 rows. Official state 7 failed, while v4 state 9 failed; this mismatch must be kept explicit.",
        "",
    ] + state_line("tomato_sauce") + [
        "",
        "## Level-3 Shortlist",
        "",
        "| Rank | Task | State | Success | Steps | Confidence | Reason |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for r in shortlist[:12]:
        lines.append(f"| {r['recommended_rank']} | {r['task']} | {r['state_id']} | {r['clean_success']} | {r['clean_steps']} | {r['source_confidence']} | {r['reason']} |")
    OBJECT100_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_registry_report(shortlist: List[Dict[str, Any]]) -> None:
    lines = [
        "# Server Source-of-Truth Registry",
        "",
        "**Date**: 2026-06-11",
        "",
        "## Accepted Level 1/2 Registry",
        "",
        "- `tomato_sauce_s0_w70-80`: phase-runner command/qpos bridge anchor; 5/6 PHYS_PASS plus 1 borderline.",
        "- `ketchup_s0_w150-160`: S19 2/3 + S20a 3/3 fresh under explicit random_control_seed; combined 5/6 PHYS_PASS plus 1 RAND-confounded.",
        "",
        "These are Level 1/2 physical bridge claims only, not official Level-3 task-effect evidence.",
        "",
        "## Current Level 3 Status",
        "",
        "- Level 3 is **not established**.",
        "- S20b phase-runner full-episode videos are archived as diagnostics only.",
        "- Official-aligned fixed-window L3 remains pending on verified clean-success states.",
        "- S20c center-crop-missing smoke must not be treated as official-aligned.",
        "",
        "## Official Clean Baseline Source of Truth",
        "",
        f"- R1 official raw per-state manifest: `{R1 / 'tables/object_official_script_100_manifest.csv'}`.",
        f"- R2 official/v4 reconstructed comparison: `{R2 / 'tables'}`.",
        f"- V4 raw full10x10 per-state dirs: `{FULL10}`.",
        "- Use `source_confidence=high` rows for state-level claims. Low-confidence W45 reconstructed rows are task-SR inferred only.",
        "",
        "## Runner Claim Boundary",
        "",
        "- Phase runner: Level 1/2 command/qpos bridge only.",
        "- Official/v4 runner: clean state success and future Level-3 official task/contact audit.",
        "- S20c runner: pending until center_crop/preprocess/action/success provenance is verified for the exact run.",
        "",
        "## Next Safe Experiment",
        "",
        "Run an official-aligned fixed-window L3 audit only on verified clean-success states, using the official/v4 path with `center_crop=True`, `official_pil_lanczos`, `postprocess_gripper=True`, `success_metric=check_success`, `num_steps_wait=10`, and explicit state IDs.",
        "",
        "Initial shortlist:",
        "",
    ]
    for r in shortlist[:8]:
        lines.append(f"- rank {r['recommended_rank']}: {r['task']} state{r['state_id']} steps={r['clean_steps']} confidence={r['source_confidence']}")
    REGISTRY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    artifact_rows = build_artifact_index()
    artifact_fields = [
        "artifact_path", "artifact_type", "file_name", "file_size", "modified_time",
        "detected_stage", "detected_runner_family", "task", "actual_task_key",
        "state_id", "window_start", "window_end", "condition", "seed",
        "attack_seed", "random_control_seed", "job_id", "git_commit",
        "center_crop", "libero_preprocess_backend", "postprocess_gripper",
        "success_metric", "full_episode", "max_steps", "video_dir",
        "num_frames", "first_frame", "last_frame", "source_confidence", "notes",
    ]
    write_csv(ARTIFACT_INDEX, artifact_rows, artifact_fields)

    tax_rows = runner_taxonomy_rows()
    tax_fields = [
        "runner", "script_path", "detected_exists", "preprocessing_path",
        "model_decode_path", "action_postprocess_path", "success_detection_path",
        "state_selection_logic", "warmup_wait_steps", "output_schema", "valid_for",
        "not_valid_for", "required_flags_config", "safe_claim_level", "notes",
    ]
    write_csv(RUNNER_TAXONOMY, tax_rows, tax_fields)
    write_runner_taxonomy_report(tax_rows)

    inventory = object100_inventory_rows()
    inv_fields = [
        "task", "task_id", "state_id", "episode_id", "runner_type", "success",
        "success_check", "success_done", "timeout", "num_steps", "success_step",
        "failure_phase", "seed", "center_crop", "attention_backend",
        "libero_preprocess_backend", "postprocess_gripper",
        "deterministic_init_states", "state_id_source", "source_path",
        "source_confidence", "notes",
    ]
    write_csv(OBJECT100_INVENTORY, inventory, inv_fields)

    shortlist = level3_shortlist_rows(inventory)
    shortlist_fields = [
        "task", "state_id", "clean_success", "clean_steps", "source_confidence",
        "recommended_rank", "reason", "suggested_window_source", "notes",
    ]
    write_csv(LEVEL3_SHORTLIST, shortlist, shortlist_fields)
    write_object100_report(inventory, shortlist)

    stale = stale_artifact_rows(artifact_rows)
    stale_fields = ["artifact_path", "artifact_type", "risk_category", "reason", "recommended_action"]
    write_csv(STALE_ARTIFACTS, stale, stale_fields)
    write_registry_report(shortlist)

    print(f"artifact_rows={len(artifact_rows)}")
    print(f"inventory_rows={len(inventory)}")
    print(f"shortlist_rows={len(shortlist)}")
    print(f"stale_rows={len(stale)}")
    for p in [
        ARTIFACT_INDEX, RUNNER_TAXONOMY, OBJECT100_INVENTORY, LEVEL3_SHORTLIST,
        STALE_ARTIFACTS, TAXONOMY_REPORT, OBJECT100_REPORT, REGISTRY_REPORT,
    ]:
        print(p)


if __name__ == "__main__":
    main()
