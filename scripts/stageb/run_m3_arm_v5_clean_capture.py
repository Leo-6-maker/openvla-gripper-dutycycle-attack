#!/usr/bin/env python3
"""M3 arm-v5 clean-only capture and event-selection runner.

This entrypoint is intentionally separated from the M3 attack runners. It does
not import PGD, RAND, shuffled-gradient, or attack-adapter code. GPU execution
for actual clean capture is gated by review; the offline selection path can be
tested on already captured per-state JSON records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gripper_attack.m3_event_panel import (  # noqa: E402
    V5_ATTACK_SEED_HASH,
    V5_FROZEN_ATTACK_SEED,
    V5_PANEL_SIZE,
    V5CleanCloseEvent,
    V5EventSelectionResult,
    V5StateCandidate,
    derive_state_pool_from_ledger,
    find_first_clean_close_onset_with_status,
    load_prior_layer3_state_ledger,
    select_first_eligible_events_by_hash,
    validate_state_pool_against_ledger,
    v5_state_hash,
)


V5_EXACT_INPUT_REQUIRED_FIELDS = (
    "raw_image_path",
    "raw_image_sha256",
    "processed_tensor_path",
    "processed_tensor_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "model_fingerprint",
    "model_checkpoint_sha256",
    "processor_config_sha256",
    "preprocess_config_sha256",
    "task_state_init_sha256",
    "clean_record_source_path",
    "clean_record_source_sha256",
    "runner_sha256",
    "config_sha256",
    "commit",
    "gpu_query",
    "worktree_status",
    "official_score_argmax_token_id",
    "previous_official_score_argmax_token_id",
)


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


def dirty_status_value() -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "GIT_STATUS_UNAVAILABLE"
    return "CLEAN" if not status else "DIRTY:" + status.replace("\n", "\\n")


def require_clean_worktree() -> None:
    dirty = dirty_status_value()
    if dirty != "CLEAN":
        raise RuntimeError(f"dirty worktree is not allowed for V5 clean capture: {dirty}")


def gpu_query_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "NVIDIA_SMI_EMPTY"
    except Exception:
        return "NVIDIA_SMI_UNAVAILABLE"


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def load_state_pool_csv(path: Path) -> list[V5StateCandidate]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        V5StateCandidate(
            task=str(row["task"]),
            state_id=int(row["state_id"]),
            task_rank=int(row["task_rank"]),
            state_hash=str(row["state_hash"]),
        )
        for row in rows
    ]


def assert_same_pool(left: Iterable[V5StateCandidate], right: Iterable[V5StateCandidate], *, label: str) -> None:
    left_rows = [(r.task, r.state_id, r.task_rank, r.state_hash) for r in left]
    right_rows = [(r.task, r.state_id, r.task_rank, r.state_hash) for r in right]
    if left_rows != right_rows:
        raise ValueError(f"state pool mismatch: {label}")


def state_pool_from_config(cfg: Mapping[str, Any]) -> list[V5StateCandidate]:
    rows = cfg.get("task_state_pool", [])
    if not isinstance(rows, list):
        raise ValueError("task_state_pool must be a list")
    pool: list[V5StateCandidate] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        task = str(row["task"])
        state_id = int(row["state_id"])
        pair = (task, state_id)
        if pair in seen:
            raise ValueError(f"duplicate frozen state in task_state_pool: {task}_s{state_id}")
        seen.add(pair)
        pool.append(
            V5StateCandidate(
                task=task,
                state_id=state_id,
                task_rank=int(row["task_rank"]),
                state_hash=str(row["state_hash"]),
            )
        )
    if len(pool) != 20:
        raise ValueError(f"task_state_pool must contain exactly 20 rows, got {len(pool)}")
    tasks = {row.task for row in pool}
    if len(tasks) != 10:
        raise ValueError(f"task_state_pool must contain exactly 10 tasks, got {len(tasks)}")
    for task in tasks:
        ranks = sorted(row.task_rank for row in pool if row.task == task)
        if ranks != [1, 2]:
            raise ValueError(f"task_state_pool must contain task ranks [1,2] for {task}, got {ranks}")
    for row in pool:
        if row.state_hash != v5_state_hash(row.task, row.state_id):
            raise ValueError(f"state hash mismatch for {row.task}_s{row.state_id}")
    return pool


def validate_frozen_pool_sources(cfg: Mapping[str, Any], *, config_path: Path) -> list[V5StateCandidate]:
    pool = state_pool_from_config(cfg)
    ledger_path = Path(str(cfg["selection"]["prior_layer3_state_ledger"]))
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    ledger = load_prior_layer3_state_ledger(ledger_path)
    expected = derive_state_pool_from_ledger(ledger)
    validate_state_pool_against_ledger(pool, ledger)
    assert_same_pool(pool, expected, label="config_vs_ledger_derived")
    csv_path = REPO_ROOT / "tables" / "m3_arm_v5_preregistered_state_pool.csv"
    assert_same_pool(pool, load_state_pool_csv(csv_path), label="config_vs_csv")
    if "unknown" in ledger_path.read_text(encoding="utf-8").lower():
        raise ValueError("prior Layer3 ledger contains unknown provenance")
    return pool


def validate_attempt_ledger_policy(
    rows: Iterable[Mapping[str, Any]],
    *,
    pool: Iterable[V5StateCandidate] | None = None,
    clean_records_dir: Path | None = None,
) -> None:
    rows = list(rows)
    pool_pairs = None if pool is None else {(row.task, row.state_id) for row in pool}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["state_id"]))
        if pool_pairs is not None and key not in pool_pairs:
            raise ValueError(f"attempt ledger contains pool-outside state: {key[0]}_s{key[1]}")
        grouped.setdefault(key, []).append(row)
    if pool_pairs is not None and set(grouped) != pool_pairs:
        missing = sorted(pool_pairs - set(grouped))
        extra = sorted(set(grouped) - pool_pairs)
        raise ValueError(f"attempt ledger coverage mismatch: missing={missing} extra={extra}")
    for (task, state_id), attempts in grouped.items():
        attempts = sorted(attempts, key=lambda row: int(row.get("attempt_index", 0) or 0))
        indices = [int(row.get("attempt_index", -1)) for row in attempts]
        if indices not in ([0], [0, 1]):
            raise ValueError(f"attempt indices must be [0] or [0,1] for {task}_s{state_id}")
        if len(attempts) > 2:
            raise ValueError(f"too many capture attempts for {task}_s{state_id}")
        captured = [row for row in attempts if str(row.get("attempt_status", "")) == "CAPTURED"]
        if len(captured) > 1:
            raise ValueError(f"multiple CAPTURED attempts for {task}_s{state_id}")
        final_status = str(attempts[-1].get("attempt_status", ""))
        if final_status == "":
            raise ValueError(f"missing terminal attempt status for {task}_s{state_id}")
        if len(attempts) == 2:
            first = attempts[0]
            status = str(first.get("attempt_status", ""))
            action_taken = str(first.get("first_action_taken", "")).strip().lower() in {"1", "true", "yes"}
            if status != "FIRST_ACTION_BEFORE_INFRA_FAILURE" or action_taken:
                raise ValueError(f"retry not allowed for {task}_s{state_id}")
        if clean_records_dir is not None:
            for row in captured:
                rel = str(row.get("clean_records_path", ""))
                expected_sha = str(row.get("clean_records_sha256", ""))
                if not rel or not expected_sha:
                    raise ValueError(f"CAPTURED attempt missing clean records binding for {task}_s{state_id}")
                path = Path(rel)
                if not path.is_absolute():
                    path = clean_records_dir / path
                if not path.exists():
                    raise ValueError(f"clean records file missing for {task}_s{state_id}: {path}")
                if sha256_file(path) != expected_sha:
                    raise ValueError(f"clean records sha mismatch for {task}_s{state_id}")


def load_attempt_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"attempt ledger missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_output_dir_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"--output_dir must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_provenance_manifest(output_dir: Path, *, config_path: Path, model_fingerprint: str = "") -> None:
    row = {
        "stage": "M3_ARM_V5_CLEAN_CAPTURE",
        "commit": git_value(["rev-parse", "HEAD"]),
        "dirty_status": dirty_status_value(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_fingerprint": model_fingerprint,
        "gpu_query": gpu_query_snapshot(),
        "hostname": socket.gethostname(),
        "python": sys.version.replace("\n", " "),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_csv(output_dir / "m3_arm_v5_clean_capture_manifest.csv", [row], list(row.keys()))


def model_fingerprint(model: Any) -> str:
    cfg = getattr(model, "config", None)
    payload = {
        "model_type": str(getattr(cfg, "model_type", "")),
        "vocab_size": int(getattr(getattr(cfg, "text_config", cfg), "vocab_size", 0) or 0),
        "pad_to_multiple_of": int(getattr(cfg, "pad_to_multiple_of", 0) or 0),
        "action_bins": int(getattr(getattr(model, "bin_centers", []), "shape", [0])[0] or 0),
        "norm_stats_keys": sorted(list(getattr(model, "norm_stats", {}).keys())),
    }
    return json.dumps(payload, sort_keys=True)


def load_model(model_path: str, model_gpu_device_id: int = -1):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip() or "10000MiB"
    if int(model_gpu_device_id) < 0:
        visible = torch.cuda.device_count()
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
    else:
        extra_kw = {"device_map": {"": int(model_gpu_device_id)}, "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"}}
    model = AutoModelCls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager"),
        **extra_kw,
    )
    device = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for value in model.hf_device_map.values():
            if isinstance(value, str) and value.startswith("cuda"):
                device = value
                break
            if isinstance(value, int):
                device = f"cuda:{value}"
                break
    return model, processor, device


def write_artifact_hash_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "m3_arm_v5_artifact_hash_manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(
        output_dir / "m3_arm_v5_artifact_hash_manifest.csv",
        rows,
        ["relative_path", "size_bytes", "sha256"],
    )


def load_clean_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError(f"clean records must be a list or contain records list: {path}")
    return [dict(row) for row in data]


def event_to_row(event: V5CleanCloseEvent, state_hash: str, task_rank: int) -> dict[str, Any]:
    artifacts = dict(event.artifacts)
    return {
        "task": event.task,
        "state_id": event.state_id,
        "task_rank": task_rank,
        "state_hash": state_hash,
        "selected_step": event.step,
        "previous_step": event.step - 1,
        "gripper_token": event.gripper_token,
        "previous_gripper_token": event.previous_gripper_token,
        "exact_7_tokens": json.dumps(list(event.exact_7_tokens)),
        "previous_exact_7_tokens": json.dumps(list(event.previous_exact_7_tokens)),
        "selection_status": "V5_CLEAN_EVENT_FOUND",
        "selection_reason": "",
        "raw_image_path": artifacts.get("raw_image_path", ""),
        "raw_image_sha256": artifacts.get("raw_image_sha256", ""),
        "processed_tensor_path": artifacts.get("processed_tensor_path", ""),
        "processed_tensor_sha256": artifacts.get("processed_tensor_sha256", ""),
        "prompt_token_ids": artifacts.get("prompt_token_ids", ""),
        "prompt_token_ids_sha256": artifacts.get("prompt_token_ids_sha256", ""),
        "score_invariant_status": "PASS",
        "official_score_argmax_token_id": artifacts.get("official_score_argmax_token_id", ""),
        "previous_official_score_argmax_token_id": artifacts.get("previous_official_score_argmax_token_id", ""),
        "model_fingerprint": artifacts.get("model_fingerprint", ""),
        "model_checkpoint_sha256": artifacts.get("model_checkpoint_sha256", ""),
        "processor_config_sha256": artifacts.get("processor_config_sha256", ""),
        "preprocess_config_sha256": artifacts.get("preprocess_config_sha256", ""),
        "task_state_init_sha256": artifacts.get("task_state_init_sha256", ""),
        "clean_record_source_path": artifacts.get("clean_record_source_path", ""),
        "clean_record_source_sha256": artifacts.get("clean_record_source_sha256", ""),
        "runner_sha256": artifacts.get("runner_sha256", ""),
        "config_sha256": artifacts.get("config_sha256", ""),
        "commit": artifacts.get("commit", ""),
        "gpu_query": artifacts.get("gpu_query", ""),
        "worktree_status": artifacts.get("worktree_status", dirty_status_value()),
    }


def result_to_row(candidate: V5StateCandidate, result: V5EventSelectionResult) -> dict[str, Any]:
    if result.event is not None:
        return event_to_row(result.event, candidate.state_hash, candidate.task_rank)
    return {
        "task": candidate.task,
        "state_id": candidate.state_id,
        "task_rank": candidate.task_rank,
        "state_hash": candidate.state_hash,
        "selected_step": "",
        "previous_step": "",
        "gripper_token": "",
        "previous_gripper_token": "",
        "exact_7_tokens": "",
        "previous_exact_7_tokens": "",
        "selection_status": result.status,
        "selection_reason": result.reason,
        "raw_image_path": "",
        "raw_image_sha256": "",
        "processed_tensor_path": "",
        "processed_tensor_sha256": "",
        "prompt_token_ids": "",
        "prompt_token_ids_sha256": "",
        "score_invariant_status": "",
        "official_score_argmax_token_id": "",
        "previous_official_score_argmax_token_id": "",
        "model_fingerprint": "",
        "model_checkpoint_sha256": "",
        "processor_config_sha256": "",
        "preprocess_config_sha256": "",
        "task_state_init_sha256": "",
        "clean_record_source_path": "",
        "clean_record_source_sha256": "",
        "runner_sha256": "",
        "config_sha256": "",
        "commit": "",
        "gpu_query": "",
        "worktree_status": dirty_status_value(),
    }


def selected_rows_have_exact_binding(rows: Iterable[Mapping[str, Any]]) -> tuple[bool, str]:
    for row in rows:
        for key in V5_EXACT_INPUT_REQUIRED_FIELDS:
            if row.get(key, "") in ("", None):
                return False, f"missing_exact_input_field:{key}:{row.get('task')}_s{row.get('state_id')}_step{row.get('selected_step')}"
    return True, ""


def select_events_from_clean_record_dir(
    *,
    cfg: Mapping[str, Any],
    clean_records_dir: Path,
) -> tuple[list[dict[str, Any]], list[V5CleanCloseEvent], str]:
    pool = validate_frozen_pool_sources(cfg, config_path=REPO_ROOT / "configs" / "m3_arm_v5_clean_close_event_panel.yaml")

    selection = cfg.get("selection", {})
    min_step = int(selection.get("min_step", 0))
    max_step = int(selection.get("max_step", 279))
    results_by_state: dict[tuple[str, int], V5CleanCloseEvent | None] = {}
    rows: list[dict[str, Any]] = []
    for candidate in pool:
        path = clean_records_dir / f"{candidate.task}_s{candidate.state_id}_clean_records.json"
        if not path.exists():
            result = V5EventSelectionResult("V5_CLEAN_EVENT_INFRA_INVALID", reason="missing_clean_records_file")
        else:
            result = find_first_clean_close_onset_with_status(
                load_clean_records(path),
                task=candidate.task,
                state_id=candidate.state_id,
                min_step=min_step,
                max_step=max_step,
            )
        rows.append(result_to_row(candidate, result))
        results_by_state[(candidate.task, candidate.state_id)] = result.event

    selected, status = select_first_eligible_events_by_hash(
        results_by_state,
        pool,
        panel_size=int(selection.get("panel_size", V5_PANEL_SIZE)),
    )
    return rows, selected, status


def run_offline_select(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    validate_output_dir_new(output_dir)
    config_path = Path(args.config)
    cfg = load_config(config_path)
    pool = validate_frozen_pool_sources(cfg, config_path=config_path)
    if not getattr(args, "attempt_ledger", ""):
        raise SystemExit("--attempt_ledger is required for offline_select")
    validate_attempt_ledger_policy(
        load_attempt_ledger(Path(args.attempt_ledger)),
        pool=pool,
        clean_records_dir=Path(args.clean_records_dir),
    )
    write_provenance_manifest(output_dir, config_path=config_path)
    rows, selected, status = select_events_from_clean_record_dir(
        cfg=cfg,
        clean_records_dir=Path(args.clean_records_dir),
    )
    fieldnames = [
        "task",
        "state_id",
        "task_rank",
        "state_hash",
        "selected_step",
        "previous_step",
        "gripper_token",
        "previous_gripper_token",
        "exact_7_tokens",
        "previous_exact_7_tokens",
        "selection_status",
        "selection_reason",
        "raw_image_sha256",
        "raw_image_path",
        "processed_tensor_sha256",
        "processed_tensor_path",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "score_invariant_status",
        "official_score_argmax_token_id",
        "previous_official_score_argmax_token_id",
        "model_fingerprint",
        "model_checkpoint_sha256",
        "processor_config_sha256",
        "preprocess_config_sha256",
        "task_state_init_sha256",
        "clean_record_source_path",
        "clean_record_source_sha256",
        "runner_sha256",
        "config_sha256",
        "commit",
        "gpu_query",
        "worktree_status",
    ]
    write_csv(output_dir / "m3_arm_v5_clean_event_selection_all_states.csv", rows, fieldnames)
    selected_rows = [
        row for row in rows if (row["task"], int(row["state_id"]), int(row["selected_step"] or -1)) in {
            (event.task, event.state_id, event.step) for event in selected
        }
    ]
    if status == "V5_EVENT_PANEL_INPUTS_FROZEN":
        ok, reason = selected_rows_have_exact_binding(selected_rows)
        if not ok:
            status = "V5_EXACT_INPUT_BINDING_INCOMPLETE"
    write_csv(output_dir / "m3_arm_v5_frozen_event_panel.csv", selected_rows, fieldnames)
    summary = {
        "status": status,
        "selected_count": len(selected),
        "panel_size": int(cfg.get("selection", {}).get("panel_size", V5_PANEL_SIZE)),
        "first_attack_seed": V5_FROZEN_ATTACK_SEED,
        "first_attack_seed_hash": V5_ATTACK_SEED_HASH,
    }
    if status == "V5_EXACT_INPUT_BINDING_INCOMPLETE":
        summary["failure_reason"] = reason
    write_json(output_dir / "m3_arm_v5_clean_capture_summary.json", summary)
    write_artifact_hash_manifest(output_dir)
    if status != "V5_EVENT_PANEL_INPUTS_FROZEN":
        raise SystemExit(status)


def hash_path_if_exists(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else ""


def save_clean_step_artifacts(
    *,
    state_dir: Path,
    task: str,
    state_id: int,
    step: int,
    raw: np.ndarray,
    action: np.ndarray,
    gen: Any,
    model: Any,
    processor: Any,
    instruction: str,
    cfg: Mapping[str, Any],
    device: str,
    model_dtype: torch.dtype,
    model_fp: str,
    init_state_sha: str,
    source_json_path: Path,
) -> dict[str, Any]:
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    from gripper_attack.v3_generation_parity import extract_exact_new_tokens
    from v4_run_eval_openvla import prompt

    step_dir = state_dir / f"step_{int(step):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    raw_npy = step_dir / "raw_agentview.npy"
    raw_png = step_dir / "raw_agentview.png"
    np.save(raw_npy, raw)
    Image.fromarray(np.asarray(raw).astype(np.uint8)).save(raw_png)

    prep = dict(cfg.get("preprocess", {}))
    image = prepare_openvla_image(raw, **prep)
    inputs = processor(prompt(str(instruction).lower()), image, return_tensors="pt")
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(dtype=model_dtype)
    tensor_path = step_dir / "processor_inputs.pt"
    torch.save({"input_ids": input_ids.cpu(), "pixel_values": pixel_values.cpu()}, tensor_path)
    prompt_token_ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
    prompt_ids_text = json.dumps(prompt_token_ids, separators=(",", ":"))

    tokens = extract_exact_new_tokens(gen.sequences, prompt_len=int(input_ids.shape[1]), expected_new_tokens=int(model.get_action_dim(cfg["model"]["unnorm_key"])))
    score_row = gen.scores[-1][0].detach().float().cpu()
    argmax = int(score_row.argmax().item())
    gripper_token = int(tokens[-1])
    source_payload = {
        "task": task,
        "state_id": int(state_id),
        "step": int(step),
        "tokens": [int(x) for x in tokens],
        "gripper_token": gripper_token,
        "official_score_argmax_token_id": argmax,
        "score_invariant": {"tie_aware_pass": argmax == gripper_token, "pass": argmax == gripper_token},
        "clean_action_raw": [float(x) for x in np.asarray(action, dtype=np.float32).tolist()],
        "raw_image_path": str(raw_npy),
        "raw_image_sha256": sha256_file(raw_npy),
        "raw_image_png_path": str(raw_png),
        "raw_image_png_sha256": sha256_file(raw_png),
        "processed_tensor_path": str(tensor_path),
        "processed_tensor_sha256": sha256_file(tensor_path),
        "prompt_token_ids": prompt_ids_text,
        "prompt_token_ids_sha256": sha256_text(prompt_ids_text),
        "model_fingerprint": model_fp,
        "model_checkpoint_sha256": hash_path_if_exists(Path(str(cfg["model"]["path"])) / "config.json"),
        "processor_config_sha256": hash_path_if_exists(Path(str(cfg["model"]["path"])) / "preprocessor_config.json"),
        "preprocess_config_sha256": sha256_text(json.dumps(prep, sort_keys=True)),
        "task_state_init_sha256": init_state_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(Path(str(cfg["_config_path"]))),
        "commit": git_value(["rev-parse", "HEAD"]),
        "gpu_query": gpu_query_snapshot(),
        "worktree_status": dirty_status_value(),
    }
    source_path = step_dir / "clean_generation_source.json"
    write_json(source_path, source_payload)
    record = dict(source_payload)
    record["clean_record_source_path"] = str(source_path)
    record["clean_record_source_sha256"] = sha256_file(source_path)
    write_json(step_dir / "clean_generation.json", record)
    return record


def run_clean_capture_for_state(
    *,
    cfg: Mapping[str, Any],
    candidate: V5StateCandidate,
    output_dir: Path,
    model: Any,
    processor: Any,
    device: str,
    model_dtype: torch.dtype,
    model_fp: str,
    max_steps: int,
    render_gpu_device_id: int,
    num_steps_wait: int,
) -> tuple[str, str]:
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env
    from libero.libero import benchmark, get_libero_path
    from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero

    task_idx = {
        "alphabet_soup": 0,
        "cream_cheese": 1,
        "salad_dressing": 2,
        "bbq_sauce": 3,
        "ketchup": 4,
        "tomato_sauce": 5,
        "butter": 6,
        "milk": 7,
        "chocolate_pudding": 8,
        "orange_juice": 9,
    }[candidate.task]
    state_dir = output_dir / "states" / f"{candidate.task}_s{candidate.state_id}"
    state_dir.mkdir(parents=True, exist_ok=True)
    source_json_path = state_dir / f"{candidate.task}_s{candidate.state_id}_clean_records.json"

    bm = benchmark.get_benchmark_dict()
    task_suite = bm["libero_object"]()
    task_obj = task_suite.get_task(task_idx)
    init_states = task_suite.get_task_init_states(task_idx)
    init_state = init_states[int(candidate.state_id)]
    init_state_sha = sha256_text(np.asarray(init_state).tobytes().hex())
    bddl_file = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env, obs = build_v4_exact_env(bddl_file, int(render_gpu_device_id), int(max_steps), int(num_steps_wait))
    try:
        obs = env.set_init_state(init_state)
        env, obs = apply_dummy_wait(env, obs, int(num_steps_wait))
        instruction = task_obj.language
        records: list[dict[str, Any]] = []
        for step in range(int(max_steps)):
            raw = np.asarray(obs["agentview_image"]).copy()
            action, _scores, _dt, gen = decode_with_scores(
                model,
                processor,
                device,
                raw,
                instruction,
                cfg["model"]["unnorm_key"],
                8,
                libero_official_preprocess=bool(cfg["preprocess"]["libero_official_preprocess"]),
                center_crop=bool(cfg["preprocess"]["center_crop"]),
                resize_size=int(cfg["preprocess"]["resize_size"]),
                libero_preprocess_backend=str(cfg["preprocess"]["libero_preprocess_backend"]),
            )
            records.append(
                save_clean_step_artifacts(
                    state_dir=state_dir,
                    task=candidate.task,
                    state_id=candidate.state_id,
                    step=step,
                    raw=raw,
                    action=np.asarray(action, dtype=np.float32),
                    gen=gen,
                    model=model,
                    processor=processor,
                    instruction=instruction,
                    cfg=cfg,
                    device=device,
                    model_dtype=model_dtype,
                    model_fp=model_fp,
                    init_state_sha=init_state_sha,
                    source_json_path=source_json_path,
                )
            )
            obs, _reward, done, _info = env.step(postprocess_openvla_action_for_libero(action, enabled=True))
            if done:
                break
        write_json(source_json_path, {"records": records})
        return str(source_json_path.relative_to(output_dir)), sha256_file(source_json_path)
    finally:
        env.close()


def run_capture_clean_pool(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    validate_output_dir_new(output_dir)
    require_clean_worktree()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    cfg["_config_path"] = str(config_path)
    pool = validate_frozen_pool_sources(cfg, config_path=config_path)
    write_provenance_manifest(output_dir, config_path=config_path, model_fingerprint="PENDING_MODEL_LOAD")
    model, processor, device = load_model(cfg["model"]["path"], int(args.model_gpu_device_id))
    model_fp = model_fingerprint(model)
    write_provenance_manifest(output_dir, config_path=config_path, model_fingerprint=model_fp)
    model_dtype = next(model.parameters()).dtype
    attempts: list[dict[str, Any]] = []
    for candidate in pool:
        attempt = {
            "task": candidate.task,
            "state_id": candidate.state_id,
            "attempt_index": 0,
            "attempt_status": "ATTEMPT_STARTED",
            "first_action_taken": "false",
            "clean_records_path": "",
            "clean_records_sha256": "",
        }
        try:
            rel_path, source_sha = run_clean_capture_for_state(
                cfg=cfg,
                candidate=candidate,
                output_dir=output_dir,
                model=model,
                processor=processor,
                device=device,
                model_dtype=model_dtype,
                model_fp=model_fp,
                max_steps=int(args.max_steps),
                render_gpu_device_id=int(args.render_gpu_device_id),
                num_steps_wait=int(args.num_steps_wait),
            )
            attempt.update(
                {
                    "attempt_status": "CAPTURED",
                    "first_action_taken": "true",
                    "clean_records_path": rel_path,
                    "clean_records_sha256": source_sha,
                }
            )
        except Exception as exc:
            attempt.update(
                {
                    "attempt_status": "CAPTURE_FAILED",
                    "failure_reason": repr(exc),
                }
            )
            attempts.append(attempt)
            write_csv(
                output_dir / "m3_arm_v5_capture_attempt_ledger.csv",
                attempts,
                ["task", "state_id", "attempt_index", "attempt_status", "first_action_taken", "clean_records_path", "clean_records_sha256", "failure_reason"],
            )
            write_artifact_hash_manifest(output_dir)
            raise
        attempts.append(attempt)
    write_csv(
        output_dir / "m3_arm_v5_capture_attempt_ledger.csv",
        attempts,
        ["task", "state_id", "attempt_index", "attempt_status", "first_action_taken", "clean_records_path", "clean_records_sha256", "failure_reason"],
    )
    validate_attempt_ledger_policy(attempts, pool=pool, clean_records_dir=output_dir)
    write_artifact_hash_manifest(output_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_arm_v5_clean_close_event_panel.yaml"))
    ap.add_argument("--mode", choices=["capture_clean_pool", "offline_select"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clean_records_dir", default="")
    ap.add_argument("--attempt_ledger", default="")
    ap.add_argument("--model_gpu_device_id", type=int, default=-1)
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    args = ap.parse_args()
    if args.mode == "capture_clean_pool":
        run_capture_clean_pool(args)
    elif args.mode == "offline_select":
        if not args.clean_records_dir:
            raise SystemExit("--clean_records_dir is required for offline_select")
        run_offline_select(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
