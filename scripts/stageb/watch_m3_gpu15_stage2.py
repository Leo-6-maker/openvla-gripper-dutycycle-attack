#!/usr/bin/env python3
"""CPU-only Stage-2 planner for the GPU(1,5) Layer3 campaign.

This module intentionally does not launch GPU work.  It validates the completed
Tomato S3 gate, reads the Layer1/2 timing handoff, and writes a sealed command
ledger for the next watcher.  The generated commands are evidence for review;
they are not executed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_TOKEN = 31744
STAGE2_SEEDS = (81, 82)
STAGE2_CONDITIONS = ("CLEAN", "PGD_DELTA0", "TRUE_PGD", "RAND21", "SHUFFLED")
GPU_ENV = "1,5"
RENDER_PHYSICAL_GPU = 1
MODEL_GPU_DEVICE_ID = -1


class Stage2PlanError(RuntimeError):
    """Raised when the CPU-only Stage-2 plan cannot be safely generated."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _as_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise Stage2PlanError(f"{field} is not numeric: {value!r}") from exc


def _as_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Stage2PlanError(f"{field} is not an integer: {value!r}") from exc


@dataclass(frozen=True)
class SelectedLambda:
    value: float
    true_margin: float
    rand_margin: float
    shuffled_margin: float
    true_arm: int
    true_token: int
    output_dir: str


def load_selected_lambda(gate_path: Path, results_csv: Path | None = None) -> SelectedLambda:
    gate = read_json(gate_path)
    if gate.get("status") != "PASS":
        raise Stage2PlanError(f"S3 Tomato gate is not PASS: {gate.get('status')!r}")
    selected = gate.get("selected")
    if not isinstance(selected, Mapping):
        raise Stage2PlanError("S3 Tomato gate missing selected row")
    lam = _as_float(selected.get("lambda"), field="selected.lambda")
    true_margin = _as_float(selected.get("true_margin"), field="selected.true_margin")
    rand_margin = _as_float(selected.get("rand_margin"), field="selected.rand_margin")
    shuffled_margin = _as_float(selected.get("shuffled_margin"), field="selected.shuffled_margin")
    true_arm = _as_int(selected.get("true_arm"), field="selected.true_arm")
    true_token = _as_int(selected.get("true_token"), field="selected.true_token")
    output_dir = str(selected.get("output_dir") or "")
    if true_token != TARGET_TOKEN:
        raise Stage2PlanError(f"selected Tomato token is not {TARGET_TOKEN}: {true_token}")
    if true_arm < 5:
        raise Stage2PlanError(f"selected Tomato arm gate failed: {true_arm}")
    if not (true_margin > rand_margin and true_margin > shuffled_margin):
        raise Stage2PlanError("selected Tomato row does not beat both controls")
    if results_csv is not None:
        rows = read_csv(results_csv)
        matches = [r for r in rows if str(r.get("lambda")) == str(selected.get("lambda"))]
        if not matches:
            raise Stage2PlanError("selected lambda not found in Tomato results CSV")
        row = matches[0]
        if str(row.get("passed")).lower() not in {"true", "1", "yes"}:
            raise Stage2PlanError("selected lambda row is not marked passed in Tomato results CSV")
        for key in ("true_margin", "rand_margin", "shuffled_margin", "true_arm", "true_token"):
            if str(row.get(key)) != str(selected.get(key)):
                raise Stage2PlanError(f"selected lambda mismatch between gate and CSV for {key}")
    return SelectedLambda(
        value=lam,
        true_margin=true_margin,
        rand_margin=rand_margin,
        shuffled_margin=shuffled_margin,
        true_arm=true_arm,
        true_token=true_token,
        output_dir=output_dir,
    )


def _first_present(row: Mapping[str, str], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def parent_id(row: Mapping[str, str], index: int) -> str:
    explicit = _first_present(row, ("parent_id", "parent", "window_id", "candidate_id"))
    if explicit:
        return explicit
    task = _first_present(row, ("task", "task_key", "actual_task_key", "parent_task"), f"parent{index}")
    state = _first_present(row, ("state", "state_id", "parent_state", "state_idx"), "")
    step = _first_present(row, ("step", "selected_step", "absolute_step", "event_step", "window_start"), "")
    return "_".join(part for part in (task, f"s{state}" if state else "", f"t{step}" if step else "") if part)


def parent_task(row: Mapping[str, str], index: int) -> str:
    return _first_present(row, ("task", "task_key", "actual_task_key", "parent_task"), f"parent{index}")


def parent_state(row: Mapping[str, str]) -> str:
    return _first_present(row, ("state", "state_id", "parent_state", "state_idx"), "0")


def parent_step(row: Mapping[str, str]) -> str:
    return _first_present(row, ("step", "selected_step", "absolute_step", "event_step", "window_start"), "")


def is_handoff_row_eligible(row: Mapping[str, str]) -> bool:
    joined = " ".join(str(v) for v in row.values()).upper()
    if any(token in joined for token in ("FAIL", "ABSTAIN", "INELIGIBLE", "INVALID")):
        return False
    return True


def select_multi_parent_rows(handoff_csv: Path, *, max_parents: int = 3) -> list[dict[str, str]]:
    rows = [r for r in read_csv(handoff_csv) if is_handoff_row_eligible(r)]
    if not rows:
        raise Stage2PlanError("handoff contains no eligible rows")
    selected: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for index, row in enumerate(rows):
        task = parent_task(row, index)
        if task in seen_tasks:
            continue
        selected.append(dict(row))
        seen_tasks.add(task)
        if len(selected) == max_parents:
            return selected
    for row in rows:
        if len(selected) == max_parents:
            break
        pid = parent_id(row, rows.index(row))
        if any(parent_id(existing, i) == pid for i, existing in enumerate(selected)):
            continue
        selected.append(dict(row))
    if len(selected) < max_parents:
        raise Stage2PlanError(f"only {len(selected)} eligible parents available; need {max_parents}")
    return selected


def build_parent_plan(
    row: Mapping[str, str],
    *,
    index: int,
    selected_lambda: SelectedLambda,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    task = parent_task(row, index)
    state = parent_state(row)
    step = parent_step(row)
    if not step:
        raise Stage2PlanError(f"parent {task} missing step/absolute_step field")
    pid = parent_id(row, index)
    parent_root = f"{cfg['stage2_output_root']}/S5_MULTI_PARENT/{pid}"
    jobs = []
    for seed in STAGE2_SEEDS:
        for condition in STAGE2_CONDITIONS:
            out_dir = f"{parent_root}/seed{seed}/{condition}"
            jobs.append(
                {
                    "stage": "S5_MULTI_PARENT",
                    "parent_id": pid,
                    "task": task,
                    "state_id": state,
                    "step": step,
                    "seed": seed,
                    "condition": condition,
                    "cuda_visible_devices": GPU_ENV,
                    "selected_lambda": selected_lambda.value,
                    "command": [
                        str(cfg["python"]),
                        str(cfg["stage2_runner"]),
                        "--config",
                        str(cfg["stage2_config"]),
                        "--mode",
                        str(cfg["stage2_mode"]),
                        "--task",
                        task,
                        "--state_id",
                        str(state),
                        "--absolute_step",
                        str(step),
                        "--condition",
                        condition,
                        "--attack_seed",
                        str(seed),
                        "--selected_lambda",
                        str(selected_lambda.value),
                        "--output_dir",
                        out_dir,
                        "--model_gpu_device_id",
                        str(MODEL_GPU_DEVICE_ID),
                        "--render_gpu_device_id",
                        str(RENDER_PHYSICAL_GPU),
                    ],
                }
            )
    return {"parent_id": pid, "task": task, "state_id": state, "step": step, "source_row": dict(row), "jobs": jobs}


def build_stage2_plan(cfg: Mapping[str, Any], selected: SelectedLambda, parents: list[dict[str, str]]) -> dict[str, Any]:
    parent_plans = [
        build_parent_plan(row, index=index, selected_lambda=selected, cfg=cfg)
        for index, row in enumerate(parents)
    ]
    return {
        "created_at": now_iso(),
        "status": "CPU_ONLY_PLAN_READY",
        "no_gpu_execution": True,
        "tomato_selected_lambda": selected.value,
        "tomato_selected_output_dir": selected.output_dir,
        "target_token": TARGET_TOKEN,
        "seeds": list(STAGE2_SEEDS),
        "conditions": list(STAGE2_CONDITIONS),
        "parents": parent_plans,
        "s5_gate": {
            "required_parent_passes": 2,
            "parent_count": 3,
            "per_parent_required_seed_passes": 2,
            "frame_pass_criteria": [
                "token_31744",
                "arm_match_ge_5_of_6",
                "TRUE_margin_gt_RAND21",
                "TRUE_margin_gt_SHUFFLED",
                "linf_pass",
                "strict_route",
                "no_fallback",
            ],
        },
        "s6_gate": {
            "enabled_only_if": "S5 parent passes >= 2/3",
            "oracle_mode": "critical_close_gripper_only_probe",
            "no_execution_in_this_script": True,
        },
    }


def flatten_command_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent in plan.get("parents", []):
        for job in parent.get("jobs", []):
            rows.append(
                {
                    "stage": job["stage"],
                    "parent_id": job["parent_id"],
                    "task": job["task"],
                    "state_id": job["state_id"],
                    "step": job["step"],
                    "seed": job["seed"],
                    "condition": job["condition"],
                    "selected_lambda": job["selected_lambda"],
                    "cuda_visible_devices": job["cuda_visible_devices"],
                    "command": " ".join(job["command"]),
                }
            )
    return rows


def evaluate_s5_gate(frame_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a future S5 result table without producer-side shortcuts."""
    by_parent: dict[str, dict[int, bool]] = {}
    for row in frame_rows:
        parent = str(row.get("parent_id", ""))
        seed = _as_int(row.get("seed"), field="seed")
        passed = str(row.get("frame_status", "")).upper() in {"FRAME_FULL_SELECTIVE_PASS", "PASS"}
        by_parent.setdefault(parent, {})[seed] = passed
    parent_results = {
        parent: all(seeds.get(seed, False) for seed in STAGE2_SEEDS)
        for parent, seeds in by_parent.items()
    }
    parent_pass_count = sum(1 for value in parent_results.values() if value)
    status = "S5_MULTI_PARENT_PASS" if parent_pass_count >= 2 else "S5_MULTI_PARENT_FAIL"
    return {"status": status, "parent_pass_count": parent_pass_count, "parent_results": parent_results}


def s6_enabled_from_s5_gate(gate: Mapping[str, Any]) -> bool:
    return str(gate.get("status")) == "S5_MULTI_PARENT_PASS" and int(gate.get("parent_pass_count", 0)) >= 2


def run_prepare(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise Stage2PlanError("config must be a mapping")
    gate_path = Path(args.s3_gate or cfg["s3_gate_result_path"])
    results_csv = Path(args.s3_results_csv or cfg["s3_results_csv"])
    handoff_csv = Path(args.handoff_csv or cfg["handoff_csv"])
    output_dir = Path(args.output_dir or cfg["prepare_output_dir"])
    selected = load_selected_lambda(gate_path, results_csv)
    parents = select_multi_parent_rows(handoff_csv, max_parents=int(cfg.get("max_parents", 3)))
    plan = build_stage2_plan(cfg, selected, parents)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "m3_gpu15_stage2_plan.json", plan)
    command_rows = flatten_command_rows(plan)
    write_csv(
        output_dir / "m3_gpu15_stage2_command_ledger.csv",
        command_rows,
        ["stage", "parent_id", "task", "state_id", "step", "seed", "condition", "selected_lambda", "cuda_visible_devices", "command"],
    )
    manifest = {
        "created_at": now_iso(),
        "config_path": str(Path(args.config)),
        "config_sha256": sha256_file(Path(args.config)),
        "s3_gate_result_path": str(gate_path),
        "s3_gate_result_sha256": sha256_file(gate_path),
        "s3_results_csv": str(results_csv),
        "s3_results_csv_sha256": sha256_file(results_csv),
        "handoff_csv": str(handoff_csv),
        "handoff_csv_sha256": sha256_file(handoff_csv),
        "no_gpu_execution": True,
        "command_count": len(command_rows),
    }
    write_json(output_dir / "m3_gpu15_stage2_prepare_manifest.json", manifest)
    print(json.dumps({"status": "CPU_ONLY_STAGE2_PLAN_READY", "output_dir": str(output_dir), "command_count": len(command_rows)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU-only Stage-2 planner for GPU15 Layer3 campaign")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_gpu15_stage2_prepare.yaml"))
    ap.add_argument("--s3_gate", default="")
    ap.add_argument("--s3_results_csv", default="")
    ap.add_argument("--handoff_csv", default="")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--prepare", action="store_true", help="write a command ledger and plan; never executes GPU jobs")
    args = ap.parse_args()
    if not args.prepare:
        raise SystemExit("--prepare is required; this script does not execute GPU stages")
    run_prepare(args)


if __name__ == "__main__":
    main()
