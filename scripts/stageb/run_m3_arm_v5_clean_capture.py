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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.m3_event_panel import (  # noqa: E402
    V5_ATTACK_SEED_HASH,
    V5_FROZEN_ATTACK_SEED,
    V5_PANEL_SIZE,
    V5CleanCloseEvent,
    V5EventSelectionResult,
    V5StateCandidate,
    find_first_clean_close_onset_with_status,
    load_prior_layer3_state_ledger,
    select_first_eligible_events_by_hash,
    validate_state_pool_against_ledger,
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
    return pool


def validate_attempt_ledger_policy(rows: Iterable[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["state_id"]))
        grouped.setdefault(key, []).append(row)
    for (task, state_id), attempts in grouped.items():
        attempts = sorted(attempts, key=lambda row: int(row.get("attempt_index", 0) or 0))
        if len(attempts) > 2:
            raise ValueError(f"too many capture attempts for {task}_s{state_id}")
        if len(attempts) == 2:
            first = attempts[0]
            status = str(first.get("attempt_status", ""))
            action_taken = str(first.get("first_action_taken", "")).strip().lower() in {"1", "true", "yes"}
            if status != "FIRST_ACTION_BEFORE_INFRA_FAILURE" or action_taken:
                raise ValueError(f"retry not allowed for {task}_s{state_id}")


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
        "raw_image_sha256": "",
        "processed_tensor_sha256": "",
        "prompt_token_ids_sha256": "",
        "score_invariant_status": "PASS",
        "model_fingerprint": "",
        "gpu_query": "",
        "worktree_status": dirty_status_value(),
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
        "raw_image_sha256": "",
        "processed_tensor_sha256": "",
        "prompt_token_ids_sha256": "",
        "score_invariant_status": "",
        "model_fingerprint": "",
        "gpu_query": "",
        "worktree_status": dirty_status_value(),
    }


def select_events_from_clean_record_dir(
    *,
    cfg: Mapping[str, Any],
    clean_records_dir: Path,
) -> tuple[list[dict[str, Any]], list[V5CleanCloseEvent], str]:
    pool = state_pool_from_config(cfg)
    ledger_path = Path(str(cfg["selection"]["prior_layer3_state_ledger"]))
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    ledger = load_prior_layer3_state_ledger(ledger_path)
    validate_state_pool_against_ledger(pool, ledger)

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
    if getattr(args, "attempt_ledger", ""):
        validate_attempt_ledger_policy(load_attempt_ledger(Path(args.attempt_ledger)))
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
        "processed_tensor_sha256",
        "prompt_token_ids_sha256",
        "score_invariant_status",
        "model_fingerprint",
        "gpu_query",
        "worktree_status",
    ]
    write_csv(output_dir / "m3_arm_v5_clean_event_selection_all_states.csv", rows, fieldnames)
    selected_rows = [
        row for row in rows if (row["task"], int(row["state_id"]), int(row["selected_step"] or -1)) in {
            (event.task, event.state_id, event.step) for event in selected
        }
    ]
    write_csv(output_dir / "m3_arm_v5_frozen_event_panel.csv", selected_rows, fieldnames)
    summary = {
        "status": status,
        "selected_count": len(selected),
        "panel_size": int(cfg.get("selection", {}).get("panel_size", V5_PANEL_SIZE)),
        "first_attack_seed": V5_FROZEN_ATTACK_SEED,
        "first_attack_seed_hash": V5_ATTACK_SEED_HASH,
    }
    write_json(output_dir / "m3_arm_v5_clean_capture_summary.json", summary)
    write_artifact_hash_manifest(output_dir)
    if status != "V5_EVENT_PANEL_INPUTS_FROZEN":
        raise SystemExit(status)


def run_capture_placeholder(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    validate_output_dir_new(output_dir)
    cfg = load_config(Path(args.config))
    if getattr(args, "attempt_ledger", ""):
        validate_attempt_ledger_policy(load_attempt_ledger(Path(args.attempt_ledger)))
    pool = state_pool_from_config(cfg)
    ledger_path = Path(str(cfg["selection"]["prior_layer3_state_ledger"]))
    if not ledger_path.is_absolute():
        ledger_path = REPO_ROOT / ledger_path
    validate_state_pool_against_ledger(pool, load_prior_layer3_state_ledger(ledger_path))
    write_provenance_manifest(output_dir, config_path=Path(args.config), model_fingerprint="CAPTURE_NOT_RUN_CPU_PLACEHOLDER")
    write_json(
        output_dir / "m3_arm_v5_capture_placeholder.json",
        {
            "status": "CAPTURE_ONLY_GPU_NOT_AUTHORIZED_IN_THIS_COMMIT",
            "states": [{"task": row.task, "state_id": row.state_id} for row in pool],
            "note": "This entrypoint is clean-only and does not import attack code.",
        },
    )
    write_artifact_hash_manifest(output_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_arm_v5_clean_close_event_panel.yaml"))
    ap.add_argument("--mode", choices=["capture_clean_pool", "offline_select"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clean_records_dir", default="")
    ap.add_argument("--attempt_ledger", default="")
    args = ap.parse_args()
    if args.mode == "capture_clean_pool":
        run_capture_placeholder(args)
    elif args.mode == "offline_select":
        if not args.clean_records_dir:
            raise SystemExit("--clean_records_dir is required for offline_select")
        run_offline_select(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
