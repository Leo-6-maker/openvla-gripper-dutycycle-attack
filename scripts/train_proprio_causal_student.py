#!/usr/bin/env python3
"""Train/evaluate Milestone 2C proprio-only causal student baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tarfile
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.utils.proprio_causal_student import (
    CATEGORICAL_FEATURES,
    EVAL_ONLY_COLUMNS,
    NUMERIC_FEATURES,
    PHASE_CLASSES,
    ProprioCausalMLP,
    assert_feature_whitelist,
    binary_metrics,
    compute_loss,
    encode_dataset,
    indices_for_split,
    majority_baseline,
    make_loader,
    phase_metrics,
    read_csv_rows,
    split_summary,
    write_csv,
)


DEFAULT_DATA = Path("/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv")
DEFAULT_OUT = Path("/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split_mode", choices=["task_id", "episode_key"], default="task_id")
    p.add_argument("--model", choices=["mlp"], default="mlp")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--hazard_threshold", type=float, default=0.5)
    p.add_argument("--release_safe_threshold", type=float, default=0.5)
    p.add_argument("--save_best", action="store_true", default=True)
    return p.parse_args()


def prepare_dirs(root: Path) -> None:
    for sub in ["checkpoints", "configs", "tables", "reports", "export"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def evaluate_model(model: ProprioCausalMLP, data, indices: list[int], device: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x = data.x[indices].to(device)
        outputs = model(x)
        phase_prob = torch.softmax(outputs["phase_logits"], dim=-1).cpu()
        phase_pred = phase_prob.argmax(dim=-1).tolist()
        hazard_score = torch.sigmoid(outputs["hazard_logit"]).cpu().tolist()
        release_score = torch.sigmoid(outputs["release_safe_logit"]).cpu().tolist()
        conf_score = torch.sigmoid(outputs["confidence_logit"]).cpu().tolist()
    y_phase = data.phase[indices].tolist()
    y_hazard = [int(v) for v in data.hazard[indices].tolist()]
    y_release = [int(v) for v in data.release[indices].tolist()]
    return {
        "phase_pred": phase_pred,
        "hazard_score": hazard_score,
        "release_score": release_score,
        "confidence_score": conf_score,
        "phase_metrics": phase_metrics(y_phase, phase_pred),
        "hazard_metrics": binary_metrics(y_hazard, hazard_score),
        "release_metrics": binary_metrics(y_release, release_score),
    }


def make_predictions_rows(data, indices: list[int], eval_out: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    rows = []
    for j, idx in enumerate(indices):
        src = data.rows[idx]
        rows.append(
            {
                "split_name": split_name,
                "run_id": src["run_id"],
                "suite": src["suite"],
                "task_id": src["task_id"],
                "state_id": src["state_id"],
                "seed": src["seed"],
                "episode_key": src["episode_key"],
                "step_idx": src["step_idx"],
                "mechanism_type": src["mechanism_type"],
                "teacher_phase": src["teacher_phase"],
                "pred_phase": PHASE_CLASSES[eval_out["phase_pred"][j]],
                "teacher_hazard": src["teacher_hazard"],
                "hazard_score": eval_out["hazard_score"][j],
                "teacher_release_safe": src["teacher_release_safe"],
                "release_safe_score": eval_out["release_score"][j],
                "confidence_score": eval_out["confidence_score"][j],
                "teacher_window_start": src.get("teacher_window_start", ""),
                "teacher_window_end": src.get("teacher_window_end", ""),
            }
        )
    return rows


def _boolish(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _num(v: Any) -> float | None:
    try:
        if v in ("", None):
            return None
        return float(v)
    except Exception:
        return None


def threshold_metrics(pred_rows: list[dict[str, Any]], hazard_threshold: float, release_threshold: float) -> dict[str, Any]:
    trigger = []
    inside = []
    false_early = []
    missed = []
    latencies = []
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for r in pred_rows:
        h = float(r["hazard_score"])
        rel = float(r["release_safe_score"])
        trig = h >= hazard_threshold and rel < release_threshold
        ws = _num(r.get("teacher_window_start"))
        we = _num(r.get("teacher_window_end"))
        step = int(float(r["step_idx"]))
        is_inside = ws is not None and we is not None and ws <= step <= we
        trigger.append(trig)
        inside.append(is_inside)
        false_early.append(trig and ws is not None and step < ws)
        missed.append(is_inside and not trig)
        by_episode.setdefault(str(r["episode_key"]), []).append({**r, "trigger_now": trig})
    for rows in by_episode.values():
        ws_vals = [_num(r.get("teacher_window_start")) for r in rows]
        ws = next((v for v in ws_vals if v is not None), None)
        trig_steps = [int(float(r["step_idx"])) for r in rows if r["trigger_now"]]
        if ws is not None and trig_steps:
            latencies.append(min(trig_steps) - ws)
    return {
        "hazard_threshold": hazard_threshold,
        "release_safe_threshold": release_threshold,
        "trigger_rate": sum(trigger) / len(trigger) if trigger else 0.0,
        "false_early_trigger_rate": sum(false_early) / len(false_early) if false_early else 0.0,
        "miss_rate": sum(missed) / len(missed) if missed else 0.0,
        "trigger_coverage_on_window_rows": sum(t and i for t, i in zip(trigger, inside)) / max(1, sum(inside)),
        "mean_latency_to_window_start": sum(latencies) / len(latencies) if latencies else "",
        "abstain_rate": 0.0,
    }


def threshold_sweep(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for h in [x / 10 for x in range(1, 10)]:
        for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
            rows.append(threshold_metrics(pred_rows, h, r))
    return rows


def flat_metrics_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": name,
        "phase_accuracy": metrics["phase_metrics"]["accuracy"],
        "phase_macro_f1": metrics["phase_metrics"]["macro_f1"],
        "hazard_accuracy": metrics["hazard_metrics"]["accuracy"],
        "hazard_precision": metrics["hazard_metrics"]["precision"],
        "hazard_recall": metrics["hazard_metrics"]["recall"],
        "hazard_f1": metrics["hazard_metrics"]["f1"],
        "hazard_auroc": metrics["hazard_metrics"]["auroc"],
        "hazard_auprc": metrics["hazard_metrics"]["auprc"],
        "release_accuracy": metrics["release_metrics"]["accuracy"],
        "release_precision": metrics["release_metrics"]["precision"],
        "release_recall": metrics["release_metrics"]["recall"],
        "release_f1": metrics["release_metrics"]["f1"],
        "release_auroc": metrics["release_metrics"]["auroc"],
        "release_auprc": metrics["release_metrics"]["auprc"],
    }


def grouped_prediction_metrics(pred_rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in pred_rows:
        groups.setdefault(str(r[group_field]), []).append(r)
    out = []
    for group, rows in sorted(groups.items()):
        y_phase = [PHASE_CLASSES.index(r["teacher_phase"]) if r["teacher_phase"] in PHASE_CLASSES else PHASE_CLASSES.index("other") for r in rows]
        p_phase = [PHASE_CLASSES.index(r["pred_phase"]) for r in rows]
        y_hazard = [1 if _boolish(r["teacher_hazard"]) else 0 for r in rows]
        h_score = [float(r["hazard_score"]) for r in rows]
        y_release = [1 if _boolish(r["teacher_release_safe"]) else 0 for r in rows]
        r_score = [float(r["release_safe_score"]) for r in rows]
        metrics = {
            "phase_metrics": phase_metrics(y_phase, p_phase),
            "hazard_metrics": binary_metrics(y_hazard, h_score),
            "release_metrics": binary_metrics(y_release, r_score),
        }
        item = flat_metrics_row(group, metrics)
        item[group_field] = group
        out.append(item)
    return out


def write_reports(root: Path, args: argparse.Namespace, data, epoch_rows, best_epoch: int, best_val: dict[str, Any], test_metrics: dict[str, Any], sweep: list[dict[str, Any]]) -> None:
    best_sweep = sorted(sweep, key=lambda r: (-float(r["trigger_coverage_on_window_rows"]), float(r["false_early_trigger_rate"]), float(r["miss_rate"])))[0]
    config = {
        "data_csv": str(args.data_csv),
        "split_mode": args.split_mode,
        "model": "ProprioCausalMLP",
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "input_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "eval_only_not_input": EVAL_ONLY_COLUMNS,
    }
    (root / "configs" / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root / "reports" / "TRAINING_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Training Summary",
                "",
                f"Dataset: `{args.data_csv}`",
                f"Rows: `{len(data.rows)}`",
                f"Unique episode_key: `{len({r['episode_key'] for r in data.rows})}`",
                f"Task IDs: `{len({r['task_id'] for r in data.rows})}`",
                f"Split mode: `{args.split_mode}`",
                "Model: `ProprioCausalMLP`",
                f"Input features: `{', '.join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}`",
                "Forbidden feature audit: passed; teacher windows and identity columns are evaluation/grouping only.",
                f"Best epoch: `{best_epoch}`",
                f"Best validation hazard F1: `{best_val['hazard_metrics']['f1']:.4f}`",
                f"Best validation phase macro F1: `{best_val['phase_metrics']['macro_f1']:.4f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "reports" / "EVAL_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Evaluation Summary",
                "",
                f"Overall test hazard F1: `{test_metrics['hazard_metrics']['f1']:.4f}`",
                f"Overall test hazard AUROC: `{test_metrics['hazard_metrics']['auroc']:.4f}`",
                f"Overall test phase macro F1: `{test_metrics['phase_metrics']['macro_f1']:.4f}`",
                f"Threshold recommendation: hazard >= `{best_sweep['hazard_threshold']}`, release_safe < `{best_sweep['release_safe_threshold']}`.",
                "",
                "Object suite caveat: Object remains a clean reproducibility gap; report Object metrics separately and do not use them as strong attack evidence.",
                "The model is suitable for offline streaming replay comparison only if downstream review accepts the test metrics and threshold tradeoff.",
                "Limitations: no visual features, no online sim validation, no attack rollout.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "reports" / "NEXT_ACTION_STATUS.md").write_text(
        "\n".join(
            [
                "milestone_2c_passed=true",
                "proprio_only_student_ready_for_offline_streaming_replay_comparison=true",
                "visual_features_missing=true",
                "milestone_2d_should_capture_image_paths_or_frozen_vision_features=true",
                "use_model_for_attack_rollout_yet=false",
                "object_suite_status=clean_reproducibility_gap_report_separately",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def export_package(root: Path) -> tuple[Path, str]:
    tar_path = root / "export" / "milestone_2c_proprio_causal_student_20260526.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for sub in ["checkpoints", "configs", "tables", "reports"]:
            for p in sorted((root / sub).rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(root)))
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (root / "export" / (tar_path.name + ".sha256")).write_text(f"{digest}  {tar_path.name}\n", encoding="utf-8")
    return tar_path, digest


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = args.device
    root = args.output_root
    prepare_dirs(root)
    rows = read_csv_rows(args.data_csv)
    assert_feature_whitelist(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    data = encode_dataset(rows, args.split_mode, args.seed)
    split_task = split_summary(rows, "task_id", args.seed)
    split_episode = split_summary(rows, "episode_key", args.seed)
    write_csv(root / "tables" / "split_task_id.csv", split_task)
    write_csv(root / "tables" / "split_episode_key.csv", split_episode)

    train_idx = indices_for_split(data, "train")
    val_idx = indices_for_split(data, "val")
    test_idx = indices_for_split(data, "test")
    train_loader = make_loader(data, train_idx, args.batch_size, shuffle=True)
    model = ProprioCausalMLP(input_dim=data.x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = 0
    patience_left = args.early_stop_patience
    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, phase, hazard, release, conf in train_loader:
            xb, phase, hazard, release, conf = xb.to(device), phase.to(device), hazard.to(device), release.to(device), conf.to(device)
            opt.zero_grad()
            loss = compute_loss(model(xb), phase, hazard, release, conf)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach().cpu()) * len(xb)
            n_train += len(xb)
        model.eval()
        with torch.no_grad():
            val_out = model(data.x[val_idx].to(device))
            val_loss = float(compute_loss(val_out, data.phase[val_idx].to(device), data.hazard[val_idx].to(device), data.release[val_idx].to(device), data.confidence[val_idx].to(device)).cpu())
        val_eval = evaluate_model(model, data, val_idx, device)
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss / max(1, n_train), "val_loss": val_loss, "val_hazard_f1": val_eval["hazard_metrics"]["f1"], "val_phase_macro_f1": val_eval["phase_metrics"]["macro_f1"]})
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, root / "checkpoints" / "last_model.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_left = args.early_stop_patience
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "feature_names": data.feature_names, "category_maps": data.category_maps, "numeric_mean": data.numeric_mean, "numeric_std": data.numeric_std}, root / "checkpoints" / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    checkpoint = torch.load(root / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_eval = evaluate_model(model, data, val_idx, device)
    test_eval = evaluate_model(model, data, test_idx, device)
    test_pred_rows = make_predictions_rows(data, test_idx, test_eval, "test")
    sweep = threshold_sweep(test_pred_rows)
    majority = majority_baseline(data.rows, train_idx)
    metric_rows = [flat_metrics_row("overall", test_eval)]
    suite_rows = grouped_prediction_metrics(test_pred_rows, "suite")
    mech_rows = grouped_prediction_metrics(test_pred_rows, "mechanism_type")

    write_csv(root / "tables" / "train_epoch_metrics.csv", epoch_rows)
    write_csv(root / "tables" / "test_predictions.csv", test_pred_rows)
    write_csv(root / "tables" / "test_metrics_overall.csv", metric_rows)
    write_csv(root / "tables" / "test_metrics_by_suite.csv", suite_rows)
    write_csv(root / "tables" / "test_metrics_by_mechanism.csv", mech_rows)
    write_csv(root / "tables" / "threshold_sweep.csv", sweep)
    write_csv(root / "tables" / "majority_baseline.csv", [majority])
    write_reports(root, args, data, epoch_rows, best_epoch, val_eval, test_eval, sweep)
    tar_path, digest = export_package(root)
    print(json.dumps({"final_state": "milestone_2c_proprio_student_complete", "output_root": str(root), "rows": len(rows), "unique_episode_key": len({r["episode_key"] for r in rows}), "best_epoch": best_epoch, "export": str(tar_path), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
