"""Train and evaluate the single authorized Stage VII S7-A candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_stage_vii_domain_shift_forensic import (
    DOSES,
    REPO,
    THRESHOLD,
    attach_windows,
    load_clean_streams,
    load_labels,
    metric_with_lift,
    read_json,
    summarize_scores,
)
from run_stage_vi_b2_candidates import ranking_loss


SEED = 20260816
COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


class MultiDoseCausalTCN(nn.Module):
    def __init__(self, channels: tuple[int, int] = (32, 32), kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(25, channels[0], kernel_size)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size)
        self.head = nn.Linear(channels[1], 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.relu(self.conv2(x))
        return self.head(x[:, :, -1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def standardization(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted({row["canonical_parent_key"] for row in rows})
    values = np.concatenate([streams[key]["features"] for key in keys], axis=0).astype(np.float32)
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) > 1e-8, values.std(axis=0), 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def dataset(rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if row["consumable"]]
    if not usable:
        raise ValueError("NO_CONSUMABLE_CANDIDATE_ROWS")
    x = np.asarray([row["window"] for row in usable], dtype=np.float32)
    x = (x - mean) / std
    y = np.asarray([row["y"] for row in usable], dtype=np.float32)
    dose = np.asarray([DOSES.index(row["dose"]) for row in usable], dtype=np.int64)
    groups = np.asarray([f"{row['stage']}::{row['canonical_parent_key']}::{row['dose']}" for row in usable])
    parent_counts = defaultdict(int)
    suite_parents: defaultdict[str, set[str]] = defaultdict(set)
    for row in usable:
        parent_counts[f"{row['stage']}::{row['canonical_parent_key']}"] += 1
        suite_parents[row["suite"]].add(f"{row['stage']}::{row['canonical_parent_key']}")
    weights = np.asarray([
        1.0 / (len(suite_parents[row["suite"]]) * parent_counts[f"{row['stage']}::{row['canonical_parent_key']}"])
        for row in usable
    ], dtype=np.float32)
    weights /= weights.mean()
    return x, y, dose, groups, weights


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_model(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], epochs: int = 120) -> tuple[MultiDoseCausalTCN, np.ndarray, np.ndarray]:
    seed_all()
    mean, std = standardization(rows, streams)
    x, y, dose, groups, sample_weights = dataset(rows, mean, std)
    model = MultiDoseCausalTCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    dose_tensor = torch.as_tensor(dose, dtype=torch.long)
    pos_weight = []
    for dose_index in range(3):
        mask = dose == dose_index
        positives = max(1, int(y[mask].sum()))
        negatives = max(1, int(mask.sum()) - positives)
        pos_weight.append(float(negatives / positives))
    class_weights = torch.as_tensor(pos_weight, dtype=torch.float32)
    sample_weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        selected = logits[torch.arange(len(logits)), dose_tensor]
        bce_terms = []
        for dose_index in range(3):
            mask = dose_tensor == dose_index
            if not bool(mask.any()):
                continue
            bce_terms.append(F.binary_cross_entropy_with_logits(
                selected[mask],
                y_tensor[mask],
                weight=sample_weight_tensor[mask],
                pos_weight=class_weights[dose_index],
            ))
        bce = torch.stack(bce_terms).mean()
        ranking = ranking_loss(selected, y_tensor, groups) if len(set(groups.tolist())) else selected.sum() * 0.0
        loss = bce + 0.25 * ranking
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


def predict_rows(model: MultiDoseCausalTCN, rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> list[dict[str, Any]]:
    if not rows:
        return []
    x = np.asarray([row["window"] for row in rows], dtype=np.float32)
    x = (x - mean) / std
    doses = np.asarray([DOSES.index(row["dose"]) for row in rows], dtype=np.int64)
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
        scores = torch.sigmoid(logits)[torch.arange(len(logits)), torch.as_tensor(doses)].numpy()
    return [{**row, "score": float(score)} for row, score in zip(rows, scores)]


def primary_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    t5 = [row for row in rows if row["dose"] == "T5"]
    summary = summarize_scores(t5, THRESHOLD)
    metric = summary["consumable_metrics"]
    overall = {
        "auroc": metric.get("auroc"),
        "auprc_lift": metric.get("auprc_lift"),
        "top_decile_lift": metric.get("top_decile_lift"),
        "ece_10bin": metric.get("ece_10bin"),
        "emission_rate": metric.get("emission_rate"),
        "pass_auroc": metric.get("auroc") is not None and metric["auroc"] >= 0.75,
        "pass_auprc_lift": metric.get("auprc_lift") is not None and metric["auprc_lift"] >= 1.25,
        "pass_top_decile_lift": metric.get("top_decile_lift") is not None and metric["top_decile_lift"] >= 1.5,
        "pass_ece": metric.get("ece_10bin") is not None and metric["ece_10bin"] <= 0.2,
        "pass_emission": metric.get("emission_rate") is not None and 0.05 <= metric["emission_rate"] <= 0.8,
    }
    per_suite = {}
    suites = sorted({row["suite"] for row in t5})
    for suite in suites:
        suite_rows = [row for row in t5 if row["suite"] == suite]
        suite_metric = summarize_scores(suite_rows, THRESHOLD)["consumable_metrics"]
        identifiable = suite_metric.get("auroc") is not None
        per_suite[suite] = {
            "identifiable": identifiable,
            "auroc": suite_metric.get("auroc"),
            "auprc_lift": suite_metric.get("auprc_lift"),
            "emission_rate": suite_metric.get("emission_rate"),
            "pass": (not identifiable) or (
                suite_metric.get("auroc", 0.0) >= 0.65
                and suite_metric.get("auprc_lift", 0.0) >= 1.05
                and 0.05 <= suite_metric.get("emission_rate", -1.0) <= 0.9
            ),
        }
    checks = list(overall.values())[5:] + [value["pass"] for value in per_suite.values()]
    return {
        "summary": summary,
        "overall": overall,
        "per_suite": per_suite,
        "pass": bool(all(checks)),
    }


def evaluate_splits(
    rows: list[dict[str, Any]],
    streams: dict[str, dict[str, Any]],
    split_map: dict[str, str],
) -> dict[str, Any]:
    split_result = {}
    for split in ("TRAIN", "VAL", "DEVTEST"):
        split_rows = [row for row in rows if split_map[row["canonical_parent_key"]] == split]
        model, mean, std = fit_model([row for row in rows if split_map[row["canonical_parent_key"]] == "TRAIN"], streams)
        split_result[split] = primary_gate(predict_rows(model, split_rows, mean, std))
    return split_result


def evaluate_loso(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]]) -> dict[str, Any]:
    suites = sorted({row["suite"] for row in rows})
    result = {}
    for suite in suites:
        train_keys = {row["canonical_parent_key"] for row in rows if row["suite"] != suite}
        test_rows = [row for row in rows if row["suite"] == suite]
        train_rows = [row for row in rows if row["canonical_parent_key"] in train_keys]
        model, mean, std = fit_model(train_rows, streams, epochs=80)
        result[suite] = primary_gate(predict_rows(model, test_rows, mean, std))
    values = [value["summary"]["consumable_metrics"].get("auroc") for value in result.values()]
    values = [value for value in values if value is not None]
    return {"per_suite": result, "mean_identifiable_auroc": float(np.mean(values)) if values else None}


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VII_S7A_CANDIDATE_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "candidate_training_performed": True,
        "formal_m4_executed": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-v-clean-root", required=True, type=Path)
    parser.add_argument("--stage-v-labels", required=True, type=Path)
    parser.add_argument("--stage-vi-clean-root", required=True, type=Path)
    parser.add_argument("--stage-vi-labels", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    split = read_json(args.split_root.resolve() / "STAGE_VII_DEVELOPMENT_PARENT_SPLIT_V1.json")
    if split.get("status") != "PASS_STAGE_VII_DEVELOPMENT_SPLIT" or split.get("selection_used_labels") or split.get("selection_used_outcomes"):
        raise SystemExit("SPLIT_NOT_FROZEN_OR_OUTCOME_BOUND")
    split_map = {row["canonical_parent_key"]: row["split"] for row in split["rows"]}
    if len(split_map) != int(split["parent_count"]):
        raise SystemExit("SPLIT_DUPLICATE_PARENT")
    stage_v_streams = load_clean_streams(args.stage_v_clean_root.resolve(), "stage_v")
    stage_vi_streams = load_clean_streams(args.stage_vi_clean_root.resolve(), "stage_vi")
    streams = {**stage_v_streams, **stage_vi_streams}
    if set(streams) != set(split_map):
        raise SystemExit("SPLIT_STREAM_POPULATION_MISMATCH")
    labels = attach_windows(
        load_labels(args.stage_v_labels.resolve(), "STAGE_V") + load_labels(args.stage_vi_labels.resolve(), "STAGE_VI_B2"),
        streams,
    )
    model, mean, std = fit_model([row for row in labels if split_map[row["canonical_parent_key"]] == "TRAIN"], streams)
    predictions = predict_rows(model, labels, mean, std)
    split_metrics = {}
    for split_name in ("TRAIN", "VAL", "DEVTEST"):
        split_metrics[split_name] = primary_gate([row for row in predictions if split_map[row["canonical_parent_key"]] == split_name])
    loso = evaluate_loso(labels, streams)
    status = "PASS_STAGE_VII_S7A_DEVELOPMENT" if split_metrics["DEVTEST"]["pass"] else "STAGE_VII_S7A_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR"
    summary = {
        "schema": "STAGE_VII_S7A_CANDIDATE_DEVELOPMENT_V1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "candidate": "S7-A",
        "candidate_training_performed": True,
        "input": "16x25D_causal_history",
        "model": "small_causal_tcn",
        "targets": list(DOSES),
        "primary_target": "T5",
        "loss": {
            "class_balanced_bce": True,
            "within_parent_pairwise_ranking_auxiliary": True,
            "ranking_weight": 0.25
        },
        "sampling": "suite_balanced_parent_balanced_inverse_row_weight",
        "threshold": THRESHOLD,
        "abstains_masked_never_negative": True,
        "split_root": str(args.split_root.resolve()),
        "split_root_sha256s_sha256": sha256_file(args.split_root.resolve() / "SHA256SUMS"),
        "metrics_by_split": split_metrics,
        "loso": loso,
        "protected_counters": COUNTERS,
        "formal_m4_executed": False,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    output.mkdir(parents=True)
    torch.save({"schema": "STAGE_VII_S7A_MULTIDOSE_16x25D_V1", "state_dict": model.state_dict(), "mean": mean, "std": std, "doses": list(DOSES)}, output / "S7_A_CHECKPOINT.pt")
    write_json(output / "STAGE_VII_S7A_CANDIDATE_DEVELOPMENT.json", summary)
    with (output / "S7_A_T5_PREDICTIONS.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps({key: row[key] for key in ("stage", "canonical_parent_key", "suite", "dose", "probe_id", "probe_step", "consumable", "abstain", "y", "label_class", "score")}, sort_keys=True) + "\n")
    seal(output, summary)
    print(json.dumps({"status": status, "output_root": str(output), "devtest_pass": split_metrics["DEVTEST"]["pass"], "protected_counters": COUNTERS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
