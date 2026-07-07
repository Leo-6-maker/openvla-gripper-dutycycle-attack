#!/usr/bin/env python3
"""Recompute C5 safety false-trigger rates from the frozen detector.

This is a detector-only diagnostic. It reads frozen clean feature rows and a
frozen detector checkpoint, then reports whether DETECTOR_SAFETY episodes emit
at least once. It does not train, simulate, run policies, roll out, intervene,
or mutate any artifact.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.detector_dataset_closure_v1 import SC5_FEATURES, load_dataset_manifest, sha256_file  # noqa: E402

PRIMARY_SUITES = {"libero_goal", "libero_object", "libero_spatial"}
DIAGNOSTIC_SUITES = {"libero_10"}
ALL_SUITES = PRIMARY_SUITES | DIAGNOSTIC_SUITES


class C5SafetyRecomputeError(ValueError):
    pass


def fail(message: str) -> None:
    raise C5SafetyRecomputeError(message)


def read_json(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"{Path(path).name}: expected JSON object")
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str, field: str, episode: str) -> float:
    try:
        out = float(value)
    except ValueError:
        fail(f"{episode}: {field} must be float")
    if not math.isfinite(out):
        fail(f"{episode}: {field} must be finite")
    return out


def read_split_assignments(path: str | Path, fold_id: str) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = ["split_type", "fold_id", "group_id", "episode_key", "split"]
        if reader.fieldnames != expected:
            fail("split CSV header mismatch")
        out: dict[str, str] = {}
        for line_no, row in enumerate(reader, start=2):
            if row["fold_id"] != fold_id:
                continue
            ep = row["episode_key"]
            if ep in out:
                fail(f"split duplicate episode at line {line_no}: {ep}")
            out[ep] = row["split"]
    if not out:
        fail(f"fold_id not found in split CSV: {fold_id}")
    return out


def read_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    obj = read_json(path)
    names = obj.get("feature_names")
    if names != SC5_FEATURES:
        fail("normalization feature_names mismatch")
    mean = np.asarray(obj.get("mean"), dtype=np.float32)
    std = np.asarray(obj.get("std"), dtype=np.float32)
    if mean.shape != (len(SC5_FEATURES),) or std.shape != (len(SC5_FEATURES),):
        fail("normalization vector shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        fail("normalization contains invalid values")
    return mean, std


def load_torch_model(checkpoint: str | Path):
    try:
        import torch
        from src.gripper_attack.sc5mlp_v1 import SC5MLPV1
    except Exception as exc:  # pragma: no cover
        fail(f"torch/model import failed: {exc}")
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model = SC5MLPV1()
    model.load_state_dict(state)
    model.eval()
    return torch, model


def score_array(torch_mod: Any, model: Any, x: np.ndarray, mean: np.ndarray, std: np.ndarray, batch_size: int) -> np.ndarray:
    scores = []
    with torch_mod.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = (x[start:start + batch_size] - mean) / std
            tensor = torch_mod.tensor(xb, dtype=torch_mod.float32)
            out = model(tensor)["corridor_logit"]
            scores.append(torch_mod.sigmoid(out).detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def read_safety_features(feature_csv: str | Path, target_episodes: set[str]) -> dict[str, list[tuple[int, list[float]]]]:
    out: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    with Path(feature_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"episode_key", "step", *SC5_FEATURES}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            fail("feature CSV missing required columns")
        for row in reader:
            ep = row["episode_key"]
            if ep not in target_episodes:
                continue
            try:
                step = int(row["step"])
            except ValueError:
                fail(f"{ep}: non-integer step")
            out[ep].append((step, [parse_float(row[name], name, ep) for name in SC5_FEATURES]))
    missing = sorted(target_episodes - set(out))
    if missing:
        fail(f"missing safety feature episodes: {len(missing)}")
    for ep in out:
        out[ep].sort(key=lambda item: item[0])
    return dict(out)


def summarize_episode_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_suite_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row)
        by_suite_split[(row["suite"], row["split"])].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        ft = sum(1 for item in items if item["false_trigger"])
        return {
            "episode_count": n,
            "false_trigger_count": ft,
            "safety_false_trigger_rate": ft / n if n else 0.0,
            "mean_max_score": float(np.mean([item["max_score"] for item in items])) if n else 0.0,
            "median_max_score": float(np.median([item["max_score"] for item in items])) if n else 0.0,
        }

    suite_rows = []
    for suite, items in sorted(by_suite.items()):
        role = "primary_positive" if suite in PRIMARY_SUITES else "diagnostic_only" if suite in DIAGNOSTIC_SUITES else "other"
        suite_rows.append({"suite": suite, "role": role, **summarize(items)})
    split_rows = []
    for (suite, split), items in sorted(by_suite_split.items()):
        role = "primary_positive" if suite in PRIMARY_SUITES else "diagnostic_only" if suite in DIAGNOSTIC_SUITES else "other"
        split_rows.append({"suite": suite, "split": split, "role": role, **summarize(items)})
    return suite_rows, split_rows


def write_sha256sums(root: Path) -> tuple[str, str]:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = root / "SHA256SUMS.sha256"
    side.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(sums), sha256_file(side)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(Path(args.freeze_manifest)) != args.expected_freeze_sha256:
        fail("freeze manifest sha mismatch")
    if sha256_file(Path(args.checkpoint)) != args.expected_checkpoint_sha256:
        fail("checkpoint sha mismatch")
    dataset_rows = load_dataset_manifest(args.dataset_csv)
    if sha256_file(Path(args.dataset_csv)) != args.expected_dataset_csv_sha256:
        fail("dataset sha mismatch")
    if sha256_file(Path(args.split_csv)) != args.expected_split_csv_sha256:
        fail("split sha mismatch")
    split = read_split_assignments(args.split_csv, args.fold_id)
    mean, std = read_normalization(args.normalization_json)
    torch_mod, model = load_torch_model(args.checkpoint)
    safety_rows = [r for r in dataset_rows if r["population_id"] == "DETECTOR_SAFETY" and r["suite"] in ALL_SUITES]
    target_eps = {r["episode_key"] for r in safety_rows}
    features = read_safety_features(args.feature_csv, target_eps)
    by_ep = {r["episode_key"]: r for r in safety_rows}
    episode_rows = []
    for ep in sorted(target_eps):
        feats = np.asarray([vals for _, vals in features[ep]], dtype=np.float32)
        steps = [step for step, _ in features[ep]]
        scores = score_array(torch_mod, model, feats, mean, std, args.batch_size)
        emit_idx = np.nonzero(scores >= args.threshold)[0]
        false_trigger = bool(emit_idx.size)
        first_emit_step = int(steps[int(emit_idx[0])]) if false_trigger else "NO_EMIT"
        max_score = float(scores.max()) if scores.size else 0.0
        row = by_ep[ep]
        episode_rows.append({
            "episode_key": ep,
            "suite": row["suite"],
            "task_id": row["task_id"],
            "split": split.get(ep, "UNASSIGNED"),
            "trace_length": row["trace_length"],
            "false_trigger": false_trigger,
            "first_emit_step": first_emit_step,
            "max_score": max_score,
        })
    suite_rows, split_rows = summarize_episode_rows(episode_rows)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "safety_recomputed_by_episode.csv", ["episode_key", "suite", "task_id", "split", "trace_length", "false_trigger", "first_emit_step", "max_score"], episode_rows)
    write_csv(out / "safety_recomputed_by_suite.csv", ["suite", "role", "episode_count", "false_trigger_count", "safety_false_trigger_rate", "mean_max_score", "median_max_score"], suite_rows)
    write_csv(out / "safety_recomputed_by_suite_split.csv", ["suite", "split", "role", "episode_count", "false_trigger_count", "safety_false_trigger_rate", "mean_max_score", "median_max_score"], split_rows)
    primary_rates = [r["safety_false_trigger_rate"] for r in suite_rows if r["suite"] in PRIMARY_SUITES]
    diag_rates = [r["safety_false_trigger_rate"] for r in suite_rows if r["suite"] in DIAGNOSTIC_SUITES]
    max_primary = max(primary_rates) if primary_rates else None
    max_diag = max(diag_rates) if diag_rates else None
    if max_primary is None:
        status = "HOLD_PRIMARY_SAFETY"
    elif max_primary > args.max_primary_safety_false_trigger:
        status = "HOLD_PRIMARY_SAFETY"
    elif max_diag is not None and max_diag > args.max_diagnostic_safety_false_trigger:
        status = "PASS_PRIMARY_HOLD_DIAGNOSTIC"
    else:
        status = "PASS"
    report = {
        "status": status,
        "schema_version": "c5_safety_recompute_from_frozen_detector_v1",
        "threshold": args.threshold,
        "threshold_source": "validation",
        "normalization_source": "train_only",
        "freeze_manifest_sha256": args.expected_freeze_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "dataset_csv_sha256": args.expected_dataset_csv_sha256,
        "split_csv_sha256": args.expected_split_csv_sha256,
        "feature_csv_sha256": sha256_file(Path(args.feature_csv)),
        "primary_positive_suites": sorted(PRIMARY_SUITES),
        "diagnostic_only_suites": sorted(DIAGNOSTIC_SUITES),
        "max_primary_safety_false_trigger": max_primary,
        "max_diagnostic_safety_false_trigger": max_diag,
        "primary_gate_threshold": args.max_primary_safety_false_trigger,
        "diagnostic_gate_threshold": args.max_diagnostic_safety_false_trigger,
        "new_training": "NOT_PERFORMED",
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "simulator": "NOT_PERFORMED",
        "policy_run": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
        "artifact_mutation": "NOT_PERFORMED",
    }
    write_json(out / "safety_recompute_summary.json", report)
    write_json(out / "c6_release_recommendation.json", {
        "status": status,
        "recommendation": "RELEASE_PRIMARY_SUITES_ONLY" if status in {"PASS", "PASS_PRIMARY_HOLD_DIAGNOSTIC"} else "HOLD_C6",
        "primary_positive_suites": sorted(PRIMARY_SUITES),
        "diagnostic_only_suites": sorted(DIAGNOSTIC_SUITES),
    })
    sums_sha, side_sha = write_sha256sums(out)
    report["SHA256SUMS"] = sums_sha
    report["SHA256SUMS.sha256"] = side_sha
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization-json", required=True)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--fold-id", default="all_suite_stratified")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-dataset-csv-sha256", required=True)
    parser.add_argument("--expected-split-csv-sha256", required=True)
    parser.add_argument("--max-primary-safety-false-trigger", type=float, default=0.15)
    parser.add_argument("--max-diagnostic-safety-false-trigger", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, json.JSONDecodeError, csv.Error, C5SafetyRecomputeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
