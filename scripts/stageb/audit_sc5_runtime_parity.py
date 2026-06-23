#!/usr/bin/env python3
"""Audit parity between frozen Layer2 evaluator predictions and SC5 runtime.

This is a CPU-only, metadata/model audit. It does not launch LIBERO, use GPU, or
run attacks. It compares a frozen checkpoint's recorded evaluator predictions
against the online runtime state machine on the exact same dataset rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES, SC5_PHASES  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finite_feature(row: dict[str, str], feature: str) -> float:
    value = float(row[feature])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {feature} for {row.get('episode_key')} step {row.get('step')}")
    return value


def row_id(row: dict[str, str]) -> tuple[str, int]:
    return str(row["episode_key"]), int(float(row["step"]))


def episode_emit_from_prediction_rows(rows: list[dict[str, Any]], tau_c: float, tau_r: float, guard: int = 5) -> int:
    state = "IDLE"
    arm_step = -1
    for row in sorted(rows, key=lambda r: int(r["step"])):
        step = int(row["step"])
        if state == "IDLE":
            if row["pred_phase"] == "stable_carry" and float(row["corridor_p"]) > tau_c:
                state = "ARMED"
                arm_step = step
        elif state == "ARMED":
            if step >= arm_step + guard and float(row["corridor_p"]) > tau_c and float(row["release_p"]) < tau_r:
                return step
    return -1


def select_dataset_rows(dataset_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    wanted = {row_id(row) for row in prediction_rows}
    by_id = {row_id(row): row for row in dataset_rows}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"dataset missing {len(missing)} prediction rows; first={missing[:3]}")
    return [by_id[row_id(row)] for row in prediction_rows]


def build_runtime_rows(
    *,
    runtime: SC5DetectorRuntime,
    dataset_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows_by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in dataset_rows:
        rows_by_episode[str(row["episode_key"])].append(row)
    out: list[dict[str, Any]] = []
    for episode_key in sorted(rows_by_episode):
        runtime.reset()
        for row in sorted(rows_by_episode[episode_key], key=lambda r: int(float(r["step"]))):
            step = int(float(row["step"]))
            features = {feature: finite_feature(row, feature) for feature in SC5_FEATURES}
            pred = predict_runtime_features(runtime, features)
            decision = runtime.update(features, step)
            out.append(
                {
                    "episode_key": episode_key,
                    "suite": row.get("suite", ""),
                    "task_idx": row.get("task_idx", ""),
                    "state_id": row.get("state_id", ""),
                    "dataset_split": row.get("dataset_split", ""),
                    "step": step,
                    "pred_phase": pred["pred_phase"],
                    "corridor_p": pred["corridor_p"],
                    "release_p": pred["release_p"],
                    "state": decision.get("state"),
                    "arm_step": decision.get("arm_step"),
                    "emit_step": decision.get("emit_step"),
                    "emitted": decision.get("emitted"),
                }
            )
    return out


def predict_runtime_features(runtime: SC5DetectorRuntime, features_25d: dict[str, float]) -> dict[str, Any]:
    x = np.array([[features_25d[fn] for fn in SC5_FEATURES]], dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("NaN/Inf in input features")
    x = (x - runtime.mean) / runtime.std
    with torch.no_grad():
        out = runtime.model(torch.tensor(x, dtype=torch.float32))
    return {
        "pred_phase": SC5_PHASES[int(out["phase_logits"][0].argmax().item())],
        "corridor_p": float(torch.sigmoid(out["corridor_logit"]).item()),
        "release_p": float(torch.sigmoid(out["release_logit"]).item()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)
    checkpoint_path = Path(args.checkpoint)
    dataset_rows_all = read_csv(dataset_path)
    prediction_rows = read_csv(predictions_path)
    if not prediction_rows:
        raise ValueError("predictions file is empty")
    selected_dataset_rows = select_dataset_rows(dataset_rows_all, prediction_rows)
    if args.suite:
        bad_suites = sorted({row["suite"] for row in selected_dataset_rows if row.get("suite") != args.suite})
        if bad_suites:
            raise ValueError(f"selected prediction rows include suites outside {args.suite}: {bad_suites}")

    runtime = SC5DetectorRuntime(str(checkpoint_path), guard=args.guard)
    runtime_rows = build_runtime_rows(runtime=runtime, dataset_rows=selected_dataset_rows)
    runtime_by_id = {row_id({"episode_key": r["episode_key"], "step": str(r["step"])}): r for r in runtime_rows}

    row_diffs: list[dict[str, Any]] = []
    max_corridor_abs_diff = 0.0
    max_release_abs_diff = 0.0
    phase_mismatch_count = 0
    prob_mismatch_count = 0
    tolerance = args.tolerance
    for pred in prediction_rows:
        rid = row_id(pred)
        run = runtime_by_id[rid]
        corridor_diff = abs(float(pred["corridor_p"]) - float(run["corridor_p"]))
        release_diff = abs(float(pred["release_p"]) - float(run["release_p"]))
        phase_match = pred["pred_phase"] == run["pred_phase"]
        prob_match = corridor_diff <= tolerance and release_diff <= tolerance
        max_corridor_abs_diff = max(max_corridor_abs_diff, corridor_diff)
        max_release_abs_diff = max(max_release_abs_diff, release_diff)
        phase_mismatch_count += 0 if phase_match else 1
        prob_mismatch_count += 0 if prob_match else 1
        if (not phase_match) or (not prob_match) or args.write_all_rows:
            row_diffs.append(
                {
                    "episode_key": pred["episode_key"],
                    "suite": pred.get("suite", ""),
                    "task_idx": pred.get("task_idx", ""),
                    "state_id": pred.get("state_id", ""),
                    "step": int(float(pred["step"])),
                    "evaluator_pred_phase": pred["pred_phase"],
                    "runtime_pred_phase": run["pred_phase"],
                    "phase_match": phase_match,
                    "evaluator_corridor_p": pred["corridor_p"],
                    "runtime_corridor_p": run["corridor_p"],
                    "corridor_abs_diff": corridor_diff,
                    "evaluator_release_p": pred["release_p"],
                    "runtime_release_p": run["release_p"],
                    "release_abs_diff": release_diff,
                    "prob_match": prob_match,
                }
            )

    by_pred_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_runtime_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_pred_episode[str(row["episode_key"])].append(row)
    for row in runtime_rows:
        by_runtime_episode[str(row["episode_key"])].append(row)
    episode_rows: list[dict[str, Any]] = []
    emit_mismatch_count = 0
    evaluator_emit_positive_count = 0
    runtime_emit_positive_count = 0
    for episode_key in sorted(by_pred_episode):
        eval_emit = episode_emit_from_prediction_rows(
            by_pred_episode[episode_key], runtime.tau_c, runtime.tau_r, guard=args.guard
        )
        runtime_emit = int(by_runtime_episode[episode_key][-1]["emit_step"])
        evaluator_emit_positive_count += int(eval_emit >= 0)
        runtime_emit_positive_count += int(runtime_emit >= 0)
        emit_match = eval_emit == runtime_emit
        emit_mismatch_count += 0 if emit_match else 1
        episode_rows.append(
            {
                "episode_key": episode_key,
                "suite": by_pred_episode[episode_key][0].get("suite", ""),
                "task_idx": by_pred_episode[episode_key][0].get("task_idx", ""),
                "state_id": by_pred_episode[episode_key][0].get("state_id", ""),
                "row_count": len(by_pred_episode[episode_key]),
                "evaluator_emit_step": eval_emit,
                "runtime_emit_step": runtime_emit,
                "emit_match": emit_match,
            }
        )

    write_csv(output_dir / "sc5_runtime_parity_row_diffs.csv", row_diffs)
    write_csv(output_dir / "sc5_runtime_parity_episode_emits.csv", episode_rows)
    summary = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "checkpoint_dataset_sha256": runtime.dataset_sha256,
        "checkpoint_tau_corridor": runtime.checkpoint_tau_corridor,
        "checkpoint_tau_release": runtime.checkpoint_tau_release,
        "threshold_source": runtime.threshold_source,
        "guard": args.guard,
        "suite": args.suite,
        "row_count": len(prediction_rows),
        "episode_count": len(episode_rows),
        "phase_mismatch_count": phase_mismatch_count,
        "prob_mismatch_count": prob_mismatch_count,
        "emit_mismatch_count": emit_mismatch_count,
        "evaluator_emit_positive_count": evaluator_emit_positive_count,
        "runtime_emit_positive_count": runtime_emit_positive_count,
        "max_corridor_abs_diff": max_corridor_abs_diff,
        "max_release_abs_diff": max_release_abs_diff,
        "tolerance": tolerance,
        "parity_pass": (
            phase_mismatch_count == 0
            and prob_mismatch_count == 0
            and emit_mismatch_count == 0
            and runtime.threshold_source == "checkpoint"
        ),
    }
    write_json(output_dir / "sc5_runtime_parity_summary.json", summary)
    report = [
        "# SC5 Runtime Parity Audit",
        "",
        f"- Dataset SHA256: `{summary['dataset_sha256']}`",
        f"- Predictions SHA256: `{summary['predictions_sha256']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Thresholds: corridor `{runtime.tau_c}`, release `{runtime.tau_r}` from `{runtime.threshold_source}`",
        f"- Rows: {len(prediction_rows)}",
        f"- Episodes: {len(episode_rows)}",
        f"- Phase mismatches: {phase_mismatch_count}",
        f"- Probability mismatches > {tolerance}: {prob_mismatch_count}",
        f"- Emit mismatches: {emit_mismatch_count}",
        f"- Evaluator emit-positive episodes: {evaluator_emit_positive_count}",
        f"- Runtime emit-positive episodes: {runtime_emit_positive_count}",
        f"- Result: `{'PASS' if summary['parity_pass'] else 'FAIL'}`",
        "",
        "No LIBERO rollout, GPU, VIS, RAND, shuffled, oracle, or attack path was executed.",
    ]
    (output_dir / "SC5_RUNTIME_PARITY_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="")
    p.add_argument("--guard", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--write-all-rows", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    summary = audit(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
