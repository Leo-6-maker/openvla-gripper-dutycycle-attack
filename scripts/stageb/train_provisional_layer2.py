#!/usr/bin/env python3
"""Train/evaluate provisional cross-suite Layer2 models.

Engineering-only. Consumes the provisional Layer2 frame dataset built from
frozen Layer1 labels. Does not launch LIBERO and does not inspect test data for
normalization or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5MLP, SC5_FEATURES, SC5_PHASES  # noqa: E402

SUITES = ["libero_spatial", "libero_goal", "libero_10"]
PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"nonfinite feature value: {value}")
    return out


def select_rows(rows: list[dict[str, str]], *, split: str, suites: set[str]) -> list[dict[str, str]]:
    out = [
        row
        for row in rows
        if row["dataset_split"] == split and row["suite"] in suites and str(row.get("ignore_for_loss", "0")) == "0"
    ]
    if not out:
        raise ValueError(f"no rows for split={split} suites={sorted(suites)}")
    return out


def count_selected_rows(rows: list[dict[str, str]], *, split: str, suites: set[str]) -> int:
    return sum(
        1
        for row in rows
        if row["dataset_split"] == split and row["suite"] in suites and str(row.get("ignore_for_loss", "0")) == "0"
    )


def skipped_run_summary(
    *,
    name: str,
    train_suites: set[str],
    val_suites: set[str],
    test_suites: set[str],
    dataset_rows: list[dict[str, str]],
    reason: str,
) -> dict[str, Any]:
    return {
        "run_name": name,
        "run_status": "SKIPPED_NO_SUPERVISED_ROWS",
        "skip_reason": reason,
        "train_suites": sorted(train_suites),
        "val_suites": sorted(val_suites),
        "test_suites": sorted(test_suites),
        "n_train_rows": count_selected_rows(dataset_rows, split="train", suites=train_suites),
        "n_val_rows": count_selected_rows(dataset_rows, split="val", suites=val_suites),
        "n_test_rows": count_selected_rows(dataset_rows, split="test", suites=test_suites),
        "checkpoint_path": "",
        "checkpoint_sha256": "",
        "selected_threshold": {},
        "train_metrics": {},
        "test_metrics": {},
        "training": {},
    }


def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.asarray([[finite_float(row[f]) for f in SC5_FEATURES] for row in rows], dtype=np.float32)
    phase = np.asarray([SC5_PHASES.index(row["teacher_phase"]) for row in rows], dtype=np.int64)
    corridor = np.asarray([float(row["teacher_corridor_active"]) for row in rows], dtype=np.float32)
    release = np.asarray([float(row["teacher_release_active"]) for row in rows], dtype=np.float32)
    return x, {"phase": phase, "corridor": corridor, "release": release}


def train_model(
    *,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    seed: int,
    device: str,
    epochs: int,
) -> tuple[SC5MLP, np.ndarray, np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_train_raw, y_train = arrays(train_rows)
    x_val_raw, y_val = arrays(val_rows)
    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0) + 1e-8
    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std
    model = SC5MLP(n_feat=len(SC5_FEATURES)).to(device)
    counts = Counter(y_train["phase"].tolist())
    total = sum(counts.values())
    weights = torch.tensor([total / max(counts.get(i, 1), 1) for i in range(len(SC5_PHASES))], dtype=torch.float32, device=device)
    phase_loss = torch.nn.CrossEntropyLoss(weight=weights)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        epoch_losses = []
        for start in range(0, len(xt), 256):
            idx = perm[start : start + 256]
            out = model(xt[idx])
            np_idx = idx.detach().cpu().numpy()
            yp = torch.tensor(y_train["phase"][np_idx], dtype=torch.long, device=device)
            yc = torch.tensor(y_train["corridor"][np_idx], dtype=torch.float32, device=device).unsqueeze(1)
            yr = torch.tensor(y_train["release"][np_idx], dtype=torch.float32, device=device).unsqueeze(1)
            loss = phase_loss(out["phase_logits"], yp) + 0.5 * bce(out["corridor_logit"], yc) + 0.3 * bce(out["release_logit"], yr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            ov = model(xv)
            ypv = torch.tensor(y_val["phase"], dtype=torch.long, device=device)
            val_loss = float(phase_loss(ov["phase_logits"], ypv).detach().cpu())
            val_acc = float((ov["phase_logits"].argmax(1) == ypv).float().mean().detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_phase_loss": val_loss, "val_phase_acc": val_acc})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    meta = {"history": history, "best_val_phase_loss": best_loss, "n_train_rows": len(train_rows), "n_val_rows": len(val_rows)}
    return model.cpu(), mean.astype(np.float32), std.astype(np.float32), meta


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_rows(model: SC5MLP, mean: np.ndarray, std: np.ndarray, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    x_raw, y = arrays(rows)
    x = (x_raw - mean) / std
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32))
    phase_idx = out["phase_logits"].argmax(1).cpu().numpy()
    cp = sigmoid(out["corridor_logit"].squeeze(1).cpu().numpy())
    rp = sigmoid(out["release_logit"].squeeze(1).cpu().numpy())
    preds: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        preds.append(
            {
                "episode_key": row["episode_key"],
                "suite": row["suite"],
                "task_idx": row["task_idx"],
                "state_id": row["state_id"],
                "dataset_split": row["dataset_split"],
                "step": int(float(row["step"])),
                "teacher_status": row["teacher_status"],
                "teacher_phase": row["teacher_phase"],
                "teacher_corridor_active": int(float(row["teacher_corridor_active"])),
                "teacher_release_active": int(float(row["teacher_release_active"])),
                "teacher_window_start": row.get("teacher_window_start", ""),
                "teacher_window_end": row.get("teacher_window_end", ""),
                "teacher_anchor_step": row.get("teacher_anchor_step", ""),
                "pred_phase": SC5_PHASES[int(phase_idx[i])],
                "corridor_p": float(cp[i]),
                "release_p": float(rp[i]),
            }
        )
    return preds


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(y.sum()), 1.0)
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def episode_emit(preds: list[dict[str, Any]], tau_c: float, tau_r: float, guard: int = 5) -> int:
    state = "IDLE"
    arm_step = -1
    for row in sorted(preds, key=lambda r: int(r["step"])):
        step = int(row["step"])
        if state == "IDLE":
            if row["pred_phase"] == "stable_carry" and float(row["corridor_p"]) > tau_c:
                state = "ARMED"
                arm_step = step
        elif state == "ARMED":
            if step >= arm_step + guard and float(row["corridor_p"]) > tau_c and float(row["release_p"]) < tau_r:
                return step
    return -1


def metrics(preds: list[dict[str, Any]], tau_c: float, tau_r: float) -> dict[str, Any]:
    y = np.asarray([int(row["teacher_corridor_active"]) for row in preds], dtype=np.int64)
    s = np.asarray([float(row["corridor_p"]) for row in preds], dtype=np.float64)
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        by_ep[str(row["episode_key"])].append(row)
    eligible = 0
    emits = 0
    hits = 0
    false_triggers = 0
    no_emit = 0
    latencies: list[int] = []
    for ep_rows in by_ep.values():
        emit = episode_emit(ep_rows, tau_c, tau_r)
        event_rows = [r for r in ep_rows if int(r["teacher_corridor_active"]) == 1]
        is_eligible = bool(event_rows)
        if is_eligible:
            eligible += 1
            start = min(int(float(r["teacher_window_start"])) for r in event_rows if str(r["teacher_window_start"]) != "")
            end = max(int(float(r["teacher_window_end"])) for r in event_rows if str(r["teacher_window_end"]) != "")
            anchors = [int(float(r["teacher_anchor_step"])) for r in event_rows if str(r["teacher_anchor_step"]) != ""]
            anchor = anchors[0] if anchors else start
            if emit >= 0:
                emits += 1
                if start <= emit <= end:
                    hits += 1
                    latencies.append(emit - anchor)
                else:
                    false_triggers += 1
            else:
                no_emit += 1
        elif emit >= 0:
            emits += 1
            false_triggers += 1
        else:
            no_emit += 1
    precision = hits / emits if emits else 0.0
    recall = hits / eligible if eligible else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "frame_auroc": auroc(y, s),
        "frame_auprc": auprc(y, s),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "eligible_episode_count": eligible,
        "emit_count": emits,
        "hit_count": hits,
        "false_trigger_episode_rate": false_triggers / max(len(by_ep), 1),
        "no_emit_rate": no_emit / max(len(by_ep), 1),
        "median_latency": float(np.median(latencies)) if latencies else float("inf"),
        "tau_corridor": tau_c,
        "tau_release": tau_r,
    }


def select_threshold(preds: list[dict[str, Any]], grid: list[float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for tau_c in grid:
        for tau_r in grid:
            row = metrics(preds, tau_c, tau_r)
            rows.append(row)
    rows.sort(key=lambda r: (-r["event_f1"], r["false_trigger_episode_rate"], r["median_latency"], r["tau_corridor"], r["tau_release"]))
    return rows[0], rows


def run_one(
    *,
    name: str,
    dataset_rows: list[dict[str, str]],
    train_suites: set[str],
    val_suites: set[str],
    test_suites: set[str],
    output_dir: Path,
    seed: int,
    device: str,
    epochs: int,
    dataset_sha: str,
) -> dict[str, Any]:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        train_rows = select_rows(dataset_rows, split="train", suites=train_suites)
        val_rows = select_rows(dataset_rows, split="val", suites=val_suites)
        test_rows = select_rows(dataset_rows, split="test", suites=test_suites)
    except ValueError as exc:
        summary = skipped_run_summary(
            name=name,
            train_suites=train_suites,
            val_suites=val_suites,
            test_suites=test_suites,
            dataset_rows=dataset_rows,
            reason=str(exc),
        )
        write_json(run_dir / "metrics.json", summary)
        return summary
    model, mean, std, train_meta = train_model(train_rows=train_rows, val_rows=val_rows, seed=seed, device=device, epochs=epochs)
    val_preds = predict_rows(model, mean, std, val_rows)
    grid = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
    selected, sweep = select_threshold(val_preds, grid)
    test_preds = predict_rows(model, mean, std, test_rows)
    train_preds = predict_rows(model, mean, std, train_rows)
    selected_test = metrics(test_preds, selected["tau_corridor"], selected["tau_release"])
    selected_train = metrics(train_preds, selected["tau_corridor"], selected["tau_release"])
    write_csv(run_dir / "threshold_sweep.csv", sweep)
    write_csv(run_dir / "predictions_val.csv", val_preds)
    write_csv(run_dir / "predictions_test.csv", test_preds)
    np.save(run_dir / "normalization_mean.npy", mean)
    np.save(run_dir / "normalization_std.npy", std)
    ckpt = {
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "feature_names": list(SC5_FEATURES),
        "phase_classes": list(SC5_PHASES),
        "dataset_sha256": dataset_sha,
        "split_mode": "provisional_cross_suite_frozen",
        "normalization_source": "train_only",
        "run_name": name,
        "train_suites": sorted(train_suites),
        "val_suites": sorted(val_suites),
        "test_suites": sorted(test_suites),
        "selected_tau_corridor": selected["tau_corridor"],
        "selected_tau_release": selected["tau_release"],
    }
    ckpt_path = run_dir / "model.pt"
    torch.save(ckpt, ckpt_path)
    summary = {
        "run_name": name,
        "run_status": "COMPLETED",
        "skip_reason": "",
        "train_suites": sorted(train_suites),
        "val_suites": sorted(val_suites),
        "test_suites": sorted(test_suites),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "n_test_rows": len(test_rows),
        "selected_threshold": selected,
        "train_metrics": selected_train,
        "test_metrics": selected_test,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "training": train_meta,
    }
    write_json(run_dir / "metrics.json", summary)
    return summary


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PROVISIONAL_SENTINEL).write_text("Provisional Layer2 training output. Not final paper evidence.\n", encoding="utf-8")
    dataset_path = Path(args.dataset)
    dataset_rows = read_csv(dataset_path)
    dataset_sha = sha256_file(dataset_path)
    matrix: list[tuple[str, set[str], set[str], set[str]]] = []
    for suite in SUITES:
        matrix.append((f"M1_in_domain_{suite}", {suite}, {suite}, {suite}))
    for heldout in SUITES:
        source = set(SUITES) - {heldout}
        matrix.append((f"M2_leave_one_suite_out_test_{heldout}", source, source, {heldout}))
    summaries = []
    for name, train_suites, val_suites, test_suites in matrix:
        summaries.append(
            run_one(
                name=name,
                dataset_rows=dataset_rows,
                train_suites=train_suites,
                val_suites=val_suites,
                test_suites=test_suites,
                output_dir=output_dir,
                seed=args.seed,
                device=args.device,
                epochs=args.epochs,
                dataset_sha=dataset_sha,
            )
        )
    report = {
        "provisional_engineering_only": True,
        "official_h2_status": "NOT_GRANTED",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "completed_runs": sum(1 for row in summaries if row.get("run_status") == "COMPLETED"),
        "skipped_runs": sum(1 for row in summaries if row.get("run_status") == "SKIPPED_NO_SUPERVISED_ROWS"),
        "runs": summaries,
    }
    write_json(output_dir / "provisional_layer2_training_summary.json", report)
    write_csv(
        output_dir / "provisional_layer2_metrics_summary.csv",
        [
            {
                "run_name": r["run_name"],
                "run_status": r.get("run_status", ""),
                "skip_reason": r.get("skip_reason", ""),
                "n_train_rows": r.get("n_train_rows", ""),
                "n_val_rows": r.get("n_val_rows", ""),
                "n_test_rows": r.get("n_test_rows", ""),
                "checkpoint_sha256": r.get("checkpoint_sha256", ""),
                "tau_corridor": r.get("selected_threshold", {}).get("tau_corridor", ""),
                "tau_release": r.get("selected_threshold", {}).get("tau_release", ""),
                "val_event_f1": r.get("selected_threshold", {}).get("event_f1", ""),
                "test_event_f1": r.get("test_metrics", {}).get("event_f1", ""),
                "test_event_precision": r.get("test_metrics", {}).get("event_precision", ""),
                "test_event_recall": r.get("test_metrics", {}).get("event_recall", ""),
                "test_false_trigger_episode_rate": r.get("test_metrics", {}).get("false_trigger_episode_rate", ""),
                "test_no_emit_rate": r.get("test_metrics", {}).get("no_emit_rate", ""),
                "test_frame_auroc": r.get("test_metrics", {}).get("frame_auroc", ""),
                "test_frame_auprc": r.get("test_metrics", {}).get("frame_auprc", ""),
            }
            for r in summaries
        ],
    )
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
