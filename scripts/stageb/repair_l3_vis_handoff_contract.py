#!/usr/bin/env python3
"""H0 fail-closed contract repair/audit for the Layer3 VIS handoff.

This script does not run OpenVLA inference, PGD, RAND controls, LIBERO rollout,
or any GPU work.  It validates that the Layer1/2 handoff is executable by the
Layer3 V4 selective fixed-frame runner before any GPU job is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_TOKEN = 31744
ATTACK_LAMBDA = "2.0"
ATTACK_SEEDS = ("81", "82")
VIS_GPU = "1,5"
V4_CONDITIONS = (
    "TRUE_PGD_TRAJECTORY21_SELECTIVE",
    "RAND21_SELECTIVE",
    "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
)


@dataclass(frozen=True)
class ExpectedFrame:
    parent_id: str
    task: str
    state_id: int
    timing_class: str
    step: int
    role: str
    primary: bool
    inside_teacher_window: bool
    d5_emit_relation: str


EXPECTED_FRAMES: tuple[ExpectedFrame, ...] = (
    ExpectedFrame("butter_s11", "butter", 11, "exact", 58, "teacher_ws", True, True, "other"),
    ExpectedFrame("butter_s11", "butter", 11, "exact", 60, "teacher_anchor+d5_emit", True, True, "emit"),
    ExpectedFrame("butter_s11", "butter", 11, "exact", 68, "teacher_we", False, False, "other"),
    ExpectedFrame("tomato_sauce_s23", "tomato_sauce", 23, "early", 69, "d5_emit", False, False, "emit"),
    ExpectedFrame("tomato_sauce_s23", "tomato_sauce", 23, "early", 139, "teacher_ws", True, True, "other"),
    ExpectedFrame("tomato_sauce_s23", "tomato_sauce", 23, "early", 141, "teacher_anchor", True, True, "other"),
    ExpectedFrame("salad_dressing_s11", "salad_dressing", 11, "late", 57, "teacher_ws", True, True, "other"),
    ExpectedFrame("salad_dressing_s11", "salad_dressing", 11, "late", 59, "teacher_anchor", True, True, "other"),
    ExpectedFrame("salad_dressing_s11", "salad_dressing", 11, "late", 67, "teacher_we", False, False, "other"),
    ExpectedFrame("salad_dressing_s11", "salad_dressing", 11, "late", 128, "d5_emit", False, False, "emit"),
)

EXPECTED_BY_KEY = {(f.parent_id, f.step): f for f in EXPECTED_FRAMES}
EXPECTED_PARENTS = {f.parent_id for f in EXPECTED_FRAMES}
PRIMARY_FRAME_KEYS = {(f.parent_id, f.step) for f in EXPECTED_FRAMES if f.primary}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames or []})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def status_from_failures(failures: list[str]) -> str:
    return "PASS" if not failures else "FAIL"


def package_paths(package_root: Path, parent_id: str, step: int) -> dict[str, Path]:
    base = package_root / parent_id / f"step_{step:04d}"
    return {
        "package_dir": base,
        "raw_agentview": base / "raw_agentview.npy",
        "canonical_processor": base / "processor_inputs_attack.pt",
        "clean_generation": base / "clean_generation.json",
        "manifest": base / "frame_package_manifest.json",
    }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def validate_selected_frame_set(rows: list[Mapping[str, str]]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        try:
            key = (str(row.get("parent_id", "")), int(row.get("frame_step", "")))
        except Exception:
            problems.append(f"bad frame_step row={row}")
            continue
        if key in seen:
            problems.append(f"duplicate frame {key[0]}:{key[1]}")
        seen.add(key)
        exp = EXPECTED_BY_KEY.get(key)
        if exp is None:
            problems.append(f"unexpected frame {key[0]}:{key[1]}")
            continue
        checks = {
            "task": exp.task,
            "state_id": str(exp.state_id),
            "timing_class": exp.timing_class,
            "frame_role": exp.role,
            "inside_teacher_window": str(exp.inside_teacher_window),
            "d5_emit_relation": exp.d5_emit_relation,
            "target_token": str(TARGET_TOKEN),
            "attack_lambda": ATTACK_LAMBDA,
            "gpu": VIS_GPU,
        }
        for field, expected in checks.items():
            if str(row.get(field, "")) != expected:
                problems.append(f"{key[0]}:{key[1]} {field}={row.get(field, '')!r} expected {expected!r}")
        if split_csv(str(row.get("attack_seeds", ""))) != ATTACK_SEEDS:
            problems.append(f"{key[0]}:{key[1]} attack_seeds={row.get('attack_seeds', '')!r} expected 81,82")
    expected_keys = set(EXPECTED_BY_KEY)
    missing = expected_keys - seen
    extra = seen - expected_keys
    for parent_id, step in sorted(missing):
        problems.append(f"missing frame {parent_id}:{step}")
    for parent_id, step in sorted(extra):
        problems.append(f"extra frame {parent_id}:{step}")
    return (not problems), problems


def validate_job_plan(rows: list[Mapping[str, str]]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    expected_jobs = {(f.parent_id, f.step, seed) for f in EXPECTED_FRAMES for seed in ATTACK_SEEDS}
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        try:
            key = (str(row.get("parent_id", "")), int(row.get("frame_step", "")), str(row.get("seed", "")))
        except Exception:
            problems.append(f"bad job row={row}")
            continue
        if key in seen:
            problems.append(f"duplicate job {key}")
        seen.add(key)
        if key not in expected_jobs:
            problems.append(f"unexpected job {key}")
            continue
        if str(row.get("condition", "")) != "TRUE_PGD_TRAJECTORY21_SELECTIVE":
            problems.append(f"{key} condition={row.get('condition', '')!r} expected TRUE_PGD_TRAJECTORY21_SELECTIVE")
        if str(row.get("target_token", "")) != str(TARGET_TOKEN):
            problems.append(f"{key} target_token={row.get('target_token', '')!r}")
        if str(row.get("lambda", "")) != ATTACK_LAMBDA:
            problems.append(f"{key} lambda={row.get('lambda', '')!r}")
        if str(row.get("gpu", "")) != VIS_GPU:
            problems.append(f"{key} gpu={row.get('gpu', '')!r}")
    for missing in sorted(expected_jobs - seen):
        problems.append(f"missing job {missing}")
    for extra in sorted(seen - expected_jobs):
        problems.append(f"extra job {extra}")
    return (not problems), problems


def hash_status(path_text: str, expected_sha: str, *, require_files: bool) -> tuple[str, str, str]:
    if not path_text:
        return "MISSING_PATH", "", "path empty"
    if not expected_sha:
        return "MISSING_EXPECTED_SHA", "", "expected sha empty"
    path = Path(path_text)
    if not path.is_file():
        return ("MISSING_FILE" if require_files else "NOT_CHECKED_FILE_ABSENT"), "", str(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        return "SHA_MISMATCH", actual, f"expected {expected_sha}"
    return "PASS", actual, ""


def audit_frame_rows(
    selected_rows: list[Mapping[str, str]],
    *,
    package_root: Path | None,
    require_files: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        parent_id = str(row.get("parent_id", ""))
        try:
            step = int(row.get("frame_step", ""))
        except Exception:
            step = -1
        exp = EXPECTED_BY_KEY.get((parent_id, step))
        failures: list[str] = []

        raw_status, raw_actual, raw_note = hash_status(
            str(row.get("raw_frame_path", "")),
            str(row.get("raw_frame_sha256", "")),
            require_files=require_files,
        )
        proc_status, proc_actual, proc_note = hash_status(
            str(row.get("processor_tensor_path", "")),
            str(row.get("processor_tensor_sha256", "")),
            require_files=require_files,
        )
        if raw_status != "PASS":
            failures.append(f"raw_frame_{raw_status}")
        if proc_status != "PASS":
            failures.append(f"processor_tensor_{proc_status}")
        if not str(row.get("prompt_instruction", "")).strip():
            failures.append("prompt_instruction_missing")
        if not str(row.get("unnorm_key", "")).strip():
            failures.append("unnorm_key_missing")

        clean_generation_status = "NOT_CONFIGURED"
        canonical_tensor_status = "NOT_CONFIGURED"
        clean_gripper_token = ""
        clean_exact_7_tokens = ""
        clean_arm_prefix = ""
        prompt_token_sha256 = ""
        model_fingerprint_status = "NOT_CONFIGURED"
        package_manifest_status = "NOT_CONFIGURED"
        canonical_tensor_sha256 = ""
        if package_root is not None:
            paths = package_paths(package_root, parent_id, step)
            if not paths["package_dir"].is_dir():
                failures.append("frame_package_missing")
                clean_generation_status = "MISSING_PACKAGE"
                canonical_tensor_status = "MISSING_PACKAGE"
                package_manifest_status = "MISSING_PACKAGE"
            else:
                package_manifest = load_json_if_exists(paths["manifest"])
                package_manifest_status = "PASS" if package_manifest else "MISSING_OR_INVALID"
                if not package_manifest:
                    failures.append("frame_package_manifest_missing")
                if paths["canonical_processor"].is_file():
                    canonical_tensor_sha256 = sha256_file(paths["canonical_processor"])
                    expected = str(package_manifest.get("canonical_processor_tensor_sha256", ""))
                    if expected and expected != canonical_tensor_sha256:
                        canonical_tensor_status = "SHA_MISMATCH"
                        failures.append("canonical_tensor_sha_mismatch")
                    else:
                        canonical_tensor_status = "PASS"
                else:
                    canonical_tensor_status = "MISSING_FILE"
                    failures.append("canonical_processor_tensor_missing")
                clean_generation = load_json_if_exists(paths["clean_generation"])
                if clean_generation:
                    tokens = clean_generation.get("clean_exact_7_tokens") or clean_generation.get("exact_7_tokens") or []
                    clean_exact_7_tokens = json.dumps(tokens)
                    clean_arm_prefix = json.dumps(tokens[:6]) if isinstance(tokens, list) else ""
                    clean_gripper_token = str(clean_generation.get("clean_gripper_token", clean_generation.get("gripper_token", "")))
                    prompt_token_sha256 = str(clean_generation.get("prompt_token_ids_sha256", ""))
                    model_fp = clean_generation.get("model_fingerprint") or clean_generation.get("model_fingerprint_sha256")
                    clean_generation_status = "PASS"
                    if not isinstance(tokens, list) or len(tokens) != 7:
                        failures.append("clean_exact_7_tokens_missing_or_bad")
                        clean_generation_status = "BAD_TOKENS"
                    if not prompt_token_sha256:
                        failures.append("prompt_token_sha_missing")
                    if not model_fp:
                        failures.append("model_fingerprint_missing")
                        model_fingerprint_status = "MISSING"
                    else:
                        model_fingerprint_status = "PASS"
                    if exp and exp.primary and clean_gripper_token != "31872":
                        failures.append(f"primary_clean_gripper_not_31872:{clean_gripper_token}")
                else:
                    clean_generation_status = "MISSING_OR_INVALID"
                    model_fingerprint_status = "MISSING"
                    failures.append("clean_generation_missing")

        out.append(
            {
                "parent_id": parent_id,
                "task": row.get("task", ""),
                "state_id": row.get("state_id", ""),
                "frame_step": step,
                "frame_role": row.get("frame_role", ""),
                "frame_denominator": "PRIMARY" if exp and exp.primary else "DIAGNOSTIC",
                "raw_frame_status": raw_status,
                "raw_frame_actual_sha256": raw_actual,
                "raw_frame_note": raw_note,
                "processor_tensor_status": proc_status,
                "processor_tensor_actual_sha256": proc_actual,
                "processor_tensor_note": proc_note,
                "clean_generation_status": clean_generation_status,
                "canonical_tensor_status": canonical_tensor_status,
                "canonical_tensor_sha256": canonical_tensor_sha256,
                "package_manifest_status": package_manifest_status,
                "prompt_instruction_present": bool(str(row.get("prompt_instruction", "")).strip()),
                "prompt_token_sha256": prompt_token_sha256,
                "model_fingerprint_status": model_fingerprint_status,
                "clean_exact_7_tokens": clean_exact_7_tokens,
                "clean_arm_prefix": clean_arm_prefix,
                "clean_gripper_token": clean_gripper_token,
                "frame_package_status": status_from_failures(failures),
                "failures": ";".join(failures),
            }
        )
    return out


def load_frame_manifest(frame_manifest_path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    data = load_json_if_exists(frame_manifest_path)
    results = data.get("results", [])
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    if isinstance(results, list):
        for item in results:
            if isinstance(item, Mapping):
                out[(str(item.get("task", "")), str(item.get("state_id", "")))] = item
    return out


def load_action_identity(path: Path) -> dict[int, Mapping[str, str]]:
    if not path.is_file():
        return {}
    rows = read_csv(path)
    out: dict[int, Mapping[str, str]] = {}
    for row in rows:
        try:
            out[int(row.get("step", ""))] = row
        except Exception:
            continue
    return out


def audit_parent_identity(
    handoff_rows: list[Mapping[str, str]],
    selected_rows: list[Mapping[str, str]],
    *,
    frame_manifest_path: Path,
    require_files: bool,
) -> list[dict[str, Any]]:
    manifest_by_parent = load_frame_manifest(frame_manifest_path)
    selected_by_parent: dict[str, list[int]] = {}
    task_state_by_parent: dict[str, tuple[str, str]] = {}
    for row in selected_rows:
        parent_id = str(row.get("parent_id", ""))
        try:
            step = int(row.get("frame_step", ""))
        except Exception:
            continue
        selected_by_parent.setdefault(parent_id, []).append(step)
        task_state_by_parent[parent_id] = (str(row.get("task", "")), str(row.get("state_id", "")))

    handoff_by_task_state = {(str(r.get("task", "")), str(r.get("state_id", ""))): r for r in handoff_rows}
    out: list[dict[str, Any]] = []
    for parent_id in sorted(EXPECTED_PARENTS):
        task, state_id = task_state_by_parent.get(parent_id, ("", ""))
        handoff = handoff_by_task_state.get((task, state_id), {})
        trace_path_text = str(handoff.get("trace_path", ""))
        trace_path = Path(trace_path_text) if trace_path_text else Path("")
        action_identity_path = trace_path.with_name("action_identity.csv") if trace_path_text else Path("")
        timing_identity = load_action_identity(action_identity_path)
        frame_result = manifest_by_parent.get((task, state_id), {})
        capture_identity = {}
        for item in frame_result.get("action_identity", []) if isinstance(frame_result, Mapping) else []:
            if isinstance(item, Mapping):
                try:
                    capture_identity[int(item.get("step", ""))] = item
                except Exception:
                    pass
        failures: list[str] = []
        compared = 0
        action_matches = 0
        env_matches = 0
        obs_matches = 0
        for step in sorted(selected_by_parent.get(parent_id, [])):
            cap = capture_identity.get(step)
            timing = timing_identity.get(step)
            if not cap:
                failures.append(f"step{step}:capture_identity_missing")
                continue
            if not timing:
                failures.append(f"step{step}:timing_identity_missing")
                continue
            compared += 1
            if str(cap.get("action_hash", "")) == str(timing.get("action_hash_pre", "")):
                action_matches += 1
            else:
                failures.append(f"step{step}:action_hash_mismatch")
            if str(cap.get("env_action_hash", "")) == str(timing.get("env_action_hash", "")):
                env_matches += 1
            else:
                failures.append(f"step{step}:env_action_hash_mismatch")
            # capture manifest does not currently contain obs_hash; report as not proven.
            if str(cap.get("obs_hash", "")) and str(cap.get("obs_hash", "")) == str(timing.get("obs_hash", "")):
                obs_matches += 1
        if not frame_result:
            failures.append("frame_manifest_parent_missing")
        if not action_identity_path.is_file():
            failures.append("timing_action_identity_missing")
        if require_files and not trace_path.is_file():
            failures.append("timing_trace_missing")
        expected_count = len(selected_by_parent.get(parent_id, []))
        if compared != expected_count:
            failures.append(f"identity_compared_{compared}_of_{expected_count}")
        exact_bound_status = "EXACT_BOUND" if not failures and compared == expected_count else "NOT_EXACT_BOUND"
        out.append(
            {
                "parent_id": parent_id,
                "task": task,
                "state_id": state_id,
                "expected_selected_steps": ",".join(str(s) for s in sorted(selected_by_parent.get(parent_id, []))),
                "trace_path": str(trace_path),
                "action_identity_path": str(action_identity_path),
                "frame_manifest_path": str(frame_manifest_path),
                "identity_compared_steps": compared,
                "action_hash_matches": action_matches,
                "env_action_hash_matches": env_matches,
                "obs_hash_matches": obs_matches,
                "obs_hash_contract": "NOT_CAPTURED_IN_FRAME_MANIFEST",
                "exact_bound_status": exact_bound_status,
                "failures": ";".join(failures),
            }
        )
    return out


def build_full_inventory(frame_manifest_path: Path) -> list[dict[str, Any]]:
    data = load_json_if_exists(frame_manifest_path)
    rows: list[dict[str, Any]] = []
    for item in data.get("results", []) if isinstance(data.get("results", []), list) else []:
        if not isinstance(item, Mapping):
            continue
        task = str(item.get("task", ""))
        state_id = str(item.get("state_id", ""))
        parent_id = f"{task}_s{state_id}"
        for frame in item.get("frames", []) if isinstance(item.get("frames", []), list) else []:
            if not isinstance(frame, Mapping):
                continue
            step = frame.get("step", "")
            try:
                key = (parent_id, int(step))
            except Exception:
                key = (parent_id, -1)
            rows.append(
                {
                    "parent_id": parent_id,
                    "task": task,
                    "state_id": state_id,
                    "step": step,
                    "role": frame.get("role", ""),
                    "raw_frame_sha256": frame.get("raw_frame_sha256", ""),
                    "processor_tensor_sha256": frame.get("processor_tensor_sha256", ""),
                    "prompt_token_sha256": frame.get("prompt_token_sha256", ""),
                    "selected_frame": key in EXPECTED_BY_KEY,
                    "selected_denominator": "PRIMARY"
                    if key in PRIMARY_FRAME_KEYS
                    else ("DIAGNOSTIC" if key in EXPECTED_BY_KEY else ""),
                }
            )
    return rows


def recursive_sha_manifest(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root in paths:
        if not root:
            continue
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = [p for p in root.rglob("*") if p.is_file()]
        else:
            continue
        for path in files:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return sorted(rows, key=lambda r: r["path"])


def frame_config(exp: ExpectedFrame, *, source_row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "stage": "L3_VIS_H0_HANDOFF_FIXED_FRAME_V4",
        "parent_id": exp.parent_id,
        "input": {
            "suite": "libero_object",
            "task": exp.task,
            "state_id": exp.state_id,
            "absolute_step": exp.step,
            "frame_role": exp.role,
            "frame_denominator": "PRIMARY" if exp.primary else "DIAGNOSTIC",
            "raw_frame_path": source_row.get("raw_frame_path", ""),
            "processor_tensor_path": source_row.get("processor_tensor_path", ""),
            "expected_clean_gripper_token": 31872 if exp.primary else None,
        },
        "model": {
            "path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object",
            "unnorm_key": source_row.get("unnorm_key", "libero_object"),
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
        },
        "attack_optimizer": {
            "strict_route": True,
            "allow_fallback": False,
            "method": "token_prefix_pgd",
            "objective": "autoregressive_prefix_gripper_target_token_cw_v1",
            "target_token_id": TARGET_TOKEN,
            "target_execution_class": "CLIP_MEDIATED_OPEN",
            "epsilon": 6 / 255,
            "num_steps": 20,
            "step_size": 1.5 * (6 / 255) / 20,
            "random_start": True,
            "prefix_refresh_interval": 1,
            "gripper_margin": 5.0,
            "arm_gate_min_match_count": 5,
            "selection_rule": "hard_feasible_official_decode_v4",
        },
        "controls": {
            "rand21_count": 21,
            "rand21_selection_metric": "official_target31744_margin_among_hard_feasible",
            "shuffled_grad_control": "trajectory21_selective",
        },
        "conditions": list(V4_CONDITIONS),
        "gates": {
            "exact_new_tokens": 7,
            "score_tie_tolerance": 1e-6,
            "arm_prefix_min_match_count": 5,
            "linf_max": 6 / 255,
        },
    }


def emit_configs(selected_rows: list[Mapping[str, str]], config_dir: Path) -> list[dict[str, Any]]:
    config_dir.mkdir(parents=True, exist_ok=True)
    rows_by_key = {(str(r.get("parent_id", "")), int(r.get("frame_step", -1))): r for r in selected_rows}
    out: list[dict[str, Any]] = []
    for exp in EXPECTED_FRAMES:
        cfg = frame_config(exp, source_row=rows_by_key[(exp.parent_id, exp.step)])
        path = config_dir / f"{exp.parent_id}_step{exp.step:04d}_v4.json"
        write_json(path, cfg)
        out.append(
            {
                "parent_id": exp.parent_id,
                "frame_step": exp.step,
                "config_path": str(path),
                "config_sha256": sha256_file(path),
                "conditions": "|".join(V4_CONDITIONS),
            }
        )
    return out


def summarize_gate(
    frame_rows: list[Mapping[str, Any]],
    parent_rows: list[Mapping[str, Any]],
    *,
    selected_frame_set_ok: bool,
    job_plan_ok: bool,
    selected_frame_set_failures: list[str],
    job_plan_failures: list[str],
) -> dict[str, Any]:
    frame_pass = sum(1 for r in frame_rows if r.get("frame_package_status") == "PASS")
    parent_pass = sum(1 for r in parent_rows if r.get("exact_bound_status") == "EXACT_BOUND")
    primary_close_pass = sum(
        1
        for r in frame_rows
        if r.get("frame_denominator") == "PRIMARY"
        and r.get("frame_package_status") == "PASS"
        and str(r.get("clean_gripper_token", "")) == "31872"
    )
    failures = []
    if not selected_frame_set_ok:
        failures.append("selected_frame_set_invalid")
    if not job_plan_ok:
        failures.append("job_plan_invalid")
    if frame_pass != len(EXPECTED_FRAMES):
        failures.append(f"selected_frame_packages_pass_{frame_pass}_of_{len(EXPECTED_FRAMES)}")
    if parent_pass != len(EXPECTED_PARENTS):
        failures.append(f"exact_bound_parents_{parent_pass}_of_{len(EXPECTED_PARENTS)}")
    if primary_close_pass != len(PRIMARY_FRAME_KEYS):
        failures.append(f"primary_clean_close_{primary_close_pass}_of_{len(PRIMARY_FRAME_KEYS)}")
    return {
        "stage": "L3_VIS_H0_HANDOFF_CONTRACT",
        "status": "PASS" if not failures else "BLOCKED",
        "git_head": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_dirty_status": git_value(["status", "--porcelain"]),
        "selected_frame_set_ok": selected_frame_set_ok,
        "job_plan_ok": job_plan_ok,
        "selected_frame_set_failures": selected_frame_set_failures,
        "job_plan_failures": job_plan_failures,
        "selected_frame_packages_pass": frame_pass,
        "selected_frame_packages_expected": len(EXPECTED_FRAMES),
        "exact_bound_parents_pass": parent_pass,
        "exact_bound_parents_expected": len(EXPECTED_PARENTS),
        "primary_clean_close_pass": primary_close_pass,
        "primary_clean_close_expected": len(PRIMARY_FRAME_KEYS),
        "failures": failures,
        "gpu_authorized_for_h1": not failures,
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# L3 VIS H0 Handoff Contract Audit",
        "",
        "This is a CPU-only handoff contract audit. It does not run GPU inference, PGD, RAND controls, or LIBERO rollout.",
        "",
        "## Result",
        "",
        f"STATUS: {summary['status']}",
        f"GPU_AUTHORIZED_FOR_H1: {summary['gpu_authorized_for_h1']}",
        "",
        "## Gate Counts",
        "",
        f"- selected frame packages: {summary['selected_frame_packages_pass']} / {summary['selected_frame_packages_expected']}",
        f"- EXACT_BOUND parents: {summary['exact_bound_parents_pass']} / {summary['exact_bound_parents_expected']}",
        f"- primary clean CLOSE frames: {summary['primary_clean_close_pass']} / {summary['primary_clean_close_expected']}",
        "",
        "## Failures",
        "",
    ]
    failures = summary.get("failures", [])
    if failures:
        lines.extend(f"- {x}" for x in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Allowed: handoff metadata and identity-contract status only.",
            "Forbidden: VIS > random, official-token attack effect, closed-loop Layer3 success.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected_frames", type=Path, default=REPO_ROOT / "tables" / "l3_vis_selected_frames_v1.csv")
    ap.add_argument("--selected_parents", type=Path, default=REPO_ROOT / "tables" / "l3_vis_selected_parents_v1.csv")
    ap.add_argument("--job_plan", type=Path, default=REPO_ROOT / "tables" / "l3_vis_job_plan_v1.csv")
    ap.add_argument("--timing_handoff", type=Path, default=REPO_ROOT / "tables" / "l12_to_l3_timing_handoff_v2.csv")
    ap.add_argument("--frame_manifest", type=Path, default=Path("/data/liuyu/outputs/l12_frame_handoff_v2_r1/frame_manifest.json"))
    ap.add_argument("--package_root", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--config_output_dir", type=Path, default=REPO_ROOT / "configs" / "l3_vis_handoff_v1")
    ap.add_argument("--path_policy", choices=["require", "metadata"], default="require")
    args = ap.parse_args()

    selected_rows = read_csv(args.selected_frames)
    parent_rows = read_csv(args.selected_parents)
    job_rows = read_csv(args.job_plan)
    handoff_rows = read_csv(args.timing_handoff)
    require_files = args.path_policy == "require"

    selected_ok, selected_failures = validate_selected_frame_set(selected_rows)
    job_ok, job_failures = validate_job_plan(job_rows)
    frame_audit_rows = audit_frame_rows(selected_rows, package_root=args.package_root, require_files=require_files)
    parent_identity_rows = audit_parent_identity(
        handoff_rows,
        selected_rows,
        frame_manifest_path=args.frame_manifest,
        require_files=require_files,
    )
    inventory_rows = build_full_inventory(args.frame_manifest)
    config_rows = emit_configs(selected_rows, args.config_output_dir)

    sha_roots = [args.selected_frames, args.selected_parents, args.job_plan, args.timing_handoff, args.frame_manifest]
    if args.package_root:
        sha_roots.append(args.package_root)
    recursive_rows = recursive_sha_manifest(sha_roots)

    summary = summarize_gate(
        frame_audit_rows,
        parent_identity_rows,
        selected_frame_set_ok=selected_ok,
        job_plan_ok=job_ok,
        selected_frame_set_failures=selected_failures,
        job_plan_failures=job_failures,
    )

    out = args.output_dir
    write_csv(out / "l3_vis_h0_selected_frame_audit.csv", frame_audit_rows)
    write_csv(out / "l3_vis_h0_parent_identity_audit.csv", parent_identity_rows)
    write_csv(out / "l3_vis_h0_full_inventory.csv", inventory_rows)
    write_csv(out / "l3_vis_h0_job_config_manifest.csv", config_rows)
    write_csv(out / "l3_vis_h0_recursive_sha256_manifest.csv", recursive_rows)
    write_json(out / "l3_vis_h0_gate_summary.json", summary)
    write_report(out / "L3_VIS_H0_HANDOFF_CONTRACT_AUDIT_20260617.md", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
