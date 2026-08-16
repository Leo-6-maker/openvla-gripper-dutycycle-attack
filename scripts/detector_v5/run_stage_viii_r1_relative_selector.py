#!/usr/bin/env python3
"""Run the frozen, CPU-only Stage VIII R1 relative selector study."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from run_stage_vii_domain_shift_forensic import (
    REPO,
    attach_windows,
    load_clean_streams,
    load_labels,
    read_json,
)
from run_stage_viii_r0_relative_timing_audit import verify_sealed_root as verify_r0_root
from run_stage_vii_s7bc_candidate import load_embeddings, load_policy, verify_sealed_root


SEED = 20260816
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ZERO_COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}


class R1Error(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R1Error(f"JSON_OBJECT_REQUIRED:{path}")
    return value


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


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise R1Error(f"INPUT_SHA256_MISMATCH:{path}")


def consumed_manifest(root: Path, filename: str, expected_count: int, expected_sha256: str) -> dict[str, Any]:
    paths = sorted(root.glob(f"parents/*/{filename}"))
    if len(paths) != expected_count:
        raise R1Error(f"CLEAN_INPUT_COUNT_MISMATCH:{root}:{len(paths)}:{expected_count}")
    lines = "".join(f"{sha256_file(path)}  {path.as_posix()}\n" for path in paths)
    digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        raise R1Error(f"CLEAN_INPUT_MANIFEST_MISMATCH:{root}:{digest}:{expected_sha256}")
    return {"root": str(root), "filename": filename, "file_count": len(paths), "manifest_sha256": digest}


def parent_id(row: dict[str, Any]) -> str:
    return f"{row['stage']}::{row['canonical_parent_key']}"


def context_id(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["stage"]), str(row["canonical_parent_key"]), int(row["probe_step"]))


def load_bound_population(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    population = protocol["population"]
    verify_file(Path(population["stage_v_labels"]), population["stage_v_labels_sha256"])
    verify_file(Path(population["stage_vi_labels"]), population["stage_vi_labels_sha256"])
    clean_bindings = {
        "STAGE_V": consumed_manifest(
            Path(population["stage_v_clean_root"]),
            "CLEAN_REPLAY_STUDENT_INPUTS_V1.json",
            int(population["stage_v_clean_consumed_file_count"]),
            population["stage_v_clean_consumed_manifest_sha256"],
        ),
        "STAGE_VI_B2": consumed_manifest(
            Path(population["stage_vi_clean_root"]),
            "RECONSTRUCTED_FIT670_EPISODE.json",
            int(population["stage_vi_clean_consumed_file_count"]),
            population["stage_vi_clean_consumed_manifest_sha256"],
        ),
    }
    split_root = Path(protocol["frozen_clean_context"]["split_root"])
    split_sums_sha = verify_sealed_root(split_root)
    expected_split_sha = protocol["frozen_clean_context"]["split_sha256s_sha256"]
    actual_split_sidecar_sha = sha256_file(split_root / "SHA256SUMS.sha256")
    if actual_split_sidecar_sha != expected_split_sha:
        raise R1Error(f"SPLIT_SHA256SUMS_SIDECAR_MISMATCH:{actual_split_sidecar_sha}:{expected_split_sha}")
    split = read_json(split_root / "STAGE_VII_DEVELOPMENT_PARENT_SPLIT_V1.json")
    if (
        split.get("status") != "PASS_STAGE_VII_DEVELOPMENT_SPLIT"
        or split.get("selection_used_labels")
        or split.get("selection_used_outcomes")
    ):
        raise R1Error("SPLIT_NOT_CLEAN_ONLY_OR_FROZEN")
    split_map = {str(row["canonical_parent_key"]): str(row["split"]) for row in split["rows"]}
    if len(split_map) != int(split["parent_count"]):
        raise R1Error("SPLIT_DUPLICATE_PARENT")
    stage_v_streams = load_clean_streams(Path(population["stage_v_clean_root"]), "stage_v")
    stage_vi_streams = load_clean_streams(Path(population["stage_vi_clean_root"]), "stage_vi")
    streams = {**stage_v_streams, **stage_vi_streams}
    if set(streams) != set(split_map):
        raise R1Error(f"SPLIT_STREAM_POPULATION_MISMATCH:{len(streams)}:{len(split_map)}")
    rows = attach_windows(
        load_labels(Path(population["stage_v_labels"]), "STAGE_V")
        + load_labels(Path(population["stage_vi_labels"]), "STAGE_VI_B2"),
        streams,
    )
    if not rows or not any(row["dose"] == "T5" and row["consumable"] for row in rows):
        raise R1Error("NO_CONSUMABLE_T5_ROWS")
    return rows, streams, split_map, {"clean": clean_bindings, "split": {"root": str(split_root), "sha256sums_sha256": split_sums_sha, "summary": split}}


def load_bound_context(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, int], dict[str, np.ndarray]], dict[str, Any]]:
    context = protocol["frozen_clean_context"]
    embedding_root = Path(context["frozen_embeddings_root"])
    embeddings, embedding_info = load_embeddings(embedding_root)
    if sha256_file(embedding_root / "SHA256SUMS.sha256") != context["frozen_embeddings_sha256s_sha256"]:
        raise R1Error("FROZEN_EMBEDDING_SHA256SUMS_SIDECAR_MISMATCH")
    policy_root = Path(context["clean_policy_root"])
    policies, policy_info = load_policy(policy_root)
    if sha256_file(policy_root / "SHA256SUMS.sha256") != context["clean_policy_sha256s_sha256"]:
        raise R1Error("CLEAN_POLICY_SHA256SUMS_SIDECAR_MISMATCH")
    usable = [row for row in rows if row["dose"] == "T5"]
    result: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    missing = []
    for row in usable:
        key = context_id(row)
        if key not in embeddings or key not in policies:
            missing.append(key)
            continue
        visual, language = embeddings[key]
        result[key] = {
            "language": np.asarray(language, dtype=np.float32),
            "visual": np.asarray(visual, dtype=np.float32),
            "policy": np.asarray(policies[key], dtype=np.float32),
        }
    if missing:
        raise R1Error(f"CONTEXT_JOIN_INCOMPLETE:{len(missing)}")
    return result, {"embeddings": embedding_info, "policy": policy_info, "joined_t5_rows": len(usable)}


def t5_rows(rows: list[dict[str, Any]], *, consumable_only: bool = False) -> list[dict[str, Any]]:
    result = [row for row in rows if row["dose"] == "T5"]
    return [row for row in result if row["consumable"]] if consumable_only else result


def feature_normalizer(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted({str(row["canonical_parent_key"]) for row in rows})
    values = np.concatenate([streams[key]["features"] for key in keys], axis=0).astype(np.float32)
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) > 1e-8, values.std(axis=0), 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_context_projection(rows: list[dict[str, Any]], context: dict[tuple[str, str, int], dict[str, np.ndarray]]) -> dict[str, Any]:
    values = [context[context_id(row)] for row in rows]
    language = np.asarray([item["language"] for item in values], dtype=np.float32)
    visual = np.asarray([item["visual"] for item in values], dtype=np.float32)
    policy = np.asarray([item["policy"] for item in values], dtype=np.float32)
    if language.shape[0] < 16 or visual.shape[0] < 16 or language.shape[1] < 16 or visual.shape[1] < 16:
        raise R1Error("CONTEXT_PCA_DIMENSION_OR_SAMPLE_FAILURE")

    def pca_block(data: np.ndarray) -> dict[str, Any]:
        mean = data.mean(axis=0)
        scale = np.where(data.std(axis=0) > 1e-8, data.std(axis=0), 1.0)
        scaled = (data - mean) / scale
        pca = PCA(n_components=16, svd_solver="full", random_state=SEED).fit(scaled)
        return {
            "mean": mean.astype(np.float32),
            "scale": scale.astype(np.float32),
            "pca_mean": pca.mean_.astype(np.float32),
            "components": pca.components_.astype(np.float32),
        }

    policy_mean = policy.mean(axis=0)
    policy_scale = np.where(policy.std(axis=0) > 1e-8, policy.std(axis=0), 1.0)
    return {
        "language": pca_block(language),
        "visual": pca_block(visual),
        "policy": {"mean": policy_mean.astype(np.float32), "scale": policy_scale.astype(np.float32)},
    }


def project_block(data: np.ndarray, spec: dict[str, Any], pca: bool) -> np.ndarray:
    scaled = (data - spec["mean"]) / spec["scale"]
    if pca:
        scaled = (scaled - spec["pca_mean"]) @ spec["components"].T
    return scaled.astype(np.float32)


def project_context(rows: list[dict[str, Any]], context: dict[tuple[str, str, int], dict[str, np.ndarray]], projection: dict[str, Any]) -> np.ndarray:
    values = [context[context_id(row)] for row in rows]
    language = project_block(np.asarray([item["language"] for item in values], dtype=np.float32), projection["language"], True)
    visual = project_block(np.asarray([item["visual"] for item in values], dtype=np.float32), projection["visual"], True)
    policy = project_block(np.asarray([item["policy"] for item in values], dtype=np.float32), projection["policy"], False)
    return np.concatenate([language, visual, policy], axis=1).astype(np.float32)


class PairwiseSelector(nn.Module):
    def __init__(self, context_dim: int = 0):
        super().__init__()
        self.context_dim = context_dim
        self.conv1 = nn.Conv1d(25, 32, 3)
        self.conv2 = nn.Conv1d(32, 32, 3)
        self.context = nn.Linear(context_dim, 32) if context_dim else None
        self.head = nn.Linear(64 if context_dim else 32, 1)

    def temporal(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.transpose(1, 2), (2, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv2(x))
        return x[:, :, -1]

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        temporal = self.temporal(x)
        if self.context is not None:
            if context is None:
                raise R1Error("CONTEXT_REQUIRED")
            context_features = F.relu(self.context(context))
            temporal = torch.cat([temporal, context_features], dim=1)
        return self.head(temporal).squeeze(-1)


def pairwise_loss(scores: torch.Tensor, y: torch.Tensor, groups: np.ndarray) -> torch.Tensor:
    losses = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        positive = scores[indices][y[indices] > 0.5]
        negative = scores[indices][y[indices] <= 0.5]
        if len(positive) and len(negative):
            losses.append(F.softplus(-(positive[:, None] - negative[None, :])).mean())
    if not losses:
        raise R1Error("NO_SAME_PARENT_TRAINING_PAIRS")
    return torch.stack(losses).mean()


def arrays(rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, context: dict[tuple[str, str, int], dict[str, np.ndarray]] | None, projection: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    x = (np.asarray([row["window"] for row in rows], dtype=np.float32) - mean) / std
    context_values = None if context is None else project_context(rows, context, projection or {})
    y = np.asarray([row["y"] for row in rows], dtype=np.float32)
    groups = np.asarray([parent_id(row) for row in rows])
    return x, context_values, y, groups


def fit_model(train_rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], candidate: str, context: dict[tuple[str, str, int], dict[str, np.ndarray]] | None, epochs: int) -> tuple[PairwiseSelector, np.ndarray, np.ndarray, dict[str, Any] | None]:
    seed_all()
    mean, std = feature_normalizer(train_rows, streams)
    projection = fit_context_projection(train_rows, context) if candidate == "R1-B" and context is not None else None
    x, context_values, y, groups = arrays(train_rows, mean, std, context if candidate == "R1-B" else None, projection)
    model = PairwiseSelector(0 if context_values is None else context_values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    context_tensor = None if context_values is None else torch.as_tensor(context_values, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        scores = model(x_tensor, context_tensor)
        loss = pairwise_loss(scores, y_tensor, groups)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std, projection


def predict(model: PairwiseSelector, rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, candidate: str, context: dict[tuple[str, str, int], dict[str, np.ndarray]] | None, projection: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not rows:
        return []
    x, context_values, _, _ = arrays(rows, mean, std, context if candidate == "R1-B" else None, projection)
    with torch.no_grad():
        scores = model(
            torch.as_tensor(x, dtype=torch.float32),
            None if context_values is None else torch.as_tensor(context_values, dtype=torch.float32),
        ).numpy()
    return [{**row, "score": float(score)} for row, score in zip(rows, scores)]


def auc(rows: list[dict[str, Any]]) -> float | None:
    positives = [row for row in rows if row["y"] == 1]
    negatives = [row for row in rows if row["y"] == 0]
    if not positives or not negatives:
        return None
    wins = ties = 0
    for positive in positives:
        for negative in negatives:
            if positive["score"] > negative["score"]:
                wins += 1
            elif positive["score"] == negative["score"]:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def parent_metric(parent: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    positives = [row for row in rows if row["y"] == 1]
    negatives = [row for row in rows if row["y"] == 0]
    if not positives or not negatives:
        return None
    wins = ties = 0
    for positive in positives:
        for negative in negatives:
            if positive["score"] > negative["score"]:
                wins += 1
            elif positive["score"] == negative["score"]:
                ties += 1
    ordered = sorted(rows, key=lambda row: (-row["score"], str(row["probe_id"]), int(row["probe_step"])))
    result: dict[str, Any] = {
        "parent": parent,
        "suite": rows[0]["suite"],
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "pair_count": len(positives) * len(negatives),
        "within_parent_auc": (wins + 0.5 * ties) / (len(positives) * len(negatives)),
        "random_prevalence": len(positives) / len(rows),
        "argmax_y": ordered[0]["y"],
        "argmax_probe_id": ordered[0]["probe_id"],
        "argmax_probe_step": ordered[0]["probe_step"],
    }
    for k in (1, 3, 5):
        result[f"top_{k}_positive_rate"] = sum(row["y"] for row in ordered[:k]) / k if len(ordered) >= k else None
    return result


def bootstrap(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    rng = random.Random(SEED)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000))
    return {"replicates": 2000, "seed": SEED, "ci_95": [samples[49], samples[1949]]}


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {"status": "NON_IDENTIFIABLE", "eligible_parent_count": 0}
    random_rate = float(np.mean([item["random_prevalence"] for item in metrics]))
    result: dict[str, Any] = {
        "status": "IDENTIFIABLE",
        "eligible_parent_count": len(metrics),
        "positive_probe_count": sum(item["positive_count"] for item in metrics),
        "negative_probe_count": sum(item["negative_count"] for item in metrics),
        "pair_count": sum(item["pair_count"] for item in metrics),
        "parent_macro_auc": float(np.mean([item["within_parent_auc"] for item in metrics])),
        "pooled_pair_auc": float(sum(item["within_parent_auc"] * item["pair_count"] for item in metrics) / sum(item["pair_count"] for item in metrics)),
        "median_parent_auc": float(np.median([item["within_parent_auc"] for item in metrics])),
        "random_baseline_prevalence": random_rate,
        "zero_regret": float(np.mean([item["argmax_y"] for item in metrics])),
        "bootstrap": bootstrap([item["within_parent_auc"] for item in metrics]),
    }
    result["zero_regret_margin_over_random"] = result["zero_regret"] - random_rate
    for k in (1, 3, 5):
        eligible = [item for item in metrics if item[f"top_{k}_positive_rate"] is not None]
        if not eligible:
            result[f"top_{k}"] = {"parent_count": 0, "selected_positive_rate": None, "lift": None}
            continue
        selected = float(np.mean([item[f"top_{k}_positive_rate"] for item in eligible]))
        baseline = float(np.mean([item["random_prevalence"] for item in eligible]))
        result[f"top_{k}"] = {
            "parent_count": len(eligible),
            "selected_positive_rate": selected,
            "random_expected_rate": baseline,
            "lift": selected / baseline if baseline else None,
        }
    result["top_1_lift"] = result["top_1"]["lift"]
    result["top_3_lift"] = result["top_3"]["lift"]
    result["top_5_lift"] = result["top_5"]["lift"]
    return result


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["consumable"]]
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[parent_id(row)].append(row)
    parent_metrics = [metric for parent, parent_rows in sorted(grouped.items()) if (metric := parent_metric(parent, parent_rows)) is not None]
    by_suite = {suite: aggregate([item for item in parent_metrics if item["suite"] == suite]) for suite in SUITES}
    return {
        "overall": aggregate(parent_metrics),
        "per_suite": by_suite,
        "global_auc_secondary": auc(usable),
        "parent_metrics": parent_metrics,
    }


def gate(metrics: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    gate_spec = protocol["promotion_gate"]
    suite_values = metrics["per_suite"]
    suite_pass = all(
        suite_values[suite]["status"] == "IDENTIFIABLE"
        and suite_values[suite]["parent_macro_auc"] >= gate_spec["every_suite_auc_min"]
        for suite in SUITES
    )
    loso = metrics["loso"]
    loso_values = [value["overall"]["parent_macro_auc"] for value in loso.values() if value["overall"]["status"] == "IDENTIFIABLE"]
    loso_pass = len(loso_values) == len(SUITES) and float(np.mean(loso_values)) >= gate_spec["loso_mean_auc_min"] and min(loso_values) >= gate_spec["loso_worst_suite_auc_min"]
    checks = {
        "parent_macro_auc": overall.get("parent_macro_auc", 0.0) >= gate_spec["parent_macro_auc_min"],
        "top_1_lift": (overall.get("top_1_lift") or 0.0) >= gate_spec["top_1_lift_min"],
        "top_3_lift": (overall.get("top_3_lift") or 0.0) >= gate_spec["top_3_lift_min"],
        "zero_regret": (overall.get("zero_regret") or 0.0) >= gate_spec["zero_regret_min"],
        "every_suite_auc": suite_pass,
        "loso_mean_auc": bool(loso_values) and float(np.mean(loso_values)) >= gate_spec["loso_mean_auc_min"],
        "loso_worst_suite_auc": bool(loso_values) and min(loso_values) >= gate_spec["loso_worst_suite_auc_min"],
        "loso_complete": loso_pass,
    }
    return {"checks": checks, "pass": all(checks.values()), "loso_mean_auc": float(np.mean(loso_values)) if loso_values else None, "loso_worst_suite_auc": min(loso_values) if loso_values else None}


def evaluate_candidate(candidate: str, rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], split_map: dict[str, str], context: dict[tuple[str, str, int], dict[str, np.ndarray]], protocol: dict[str, Any]) -> dict[str, Any]:
    usable = t5_rows(rows, consumable_only=True)
    train_rows = [row for row in usable if split_map[row["canonical_parent_key"]] == "TRAIN"]
    if not train_rows:
        raise R1Error(f"NO_TRAIN_ROWS:{candidate}")
    model, mean, std, projection = fit_model(train_rows, streams, candidate, context, int(protocol["training"]["epochs"]))
    predictions = predict(model, t5_rows(rows), mean, std, candidate, context, projection)
    metrics_by_split = {}
    for split in ("TRAIN", "VAL", "DEVTEST"):
        split_rows = [row for row in predictions if split_map[row["canonical_parent_key"]] == split]
        metrics_by_split[split] = evaluate(split_rows)
    loso = {}
    for suite in SUITES:
        loso_train = [row for row in usable if row["suite"] != suite]
        loso_test = [row for row in t5_rows(rows) if row["suite"] == suite]
        try:
            loso_model, loso_mean, loso_std, loso_projection = fit_model(loso_train, streams, candidate, context, int(protocol["training"]["epochs"]))
            loso_predictions = predict(loso_model, loso_test, loso_mean, loso_std, candidate, context, loso_projection)
            loso[suite] = evaluate(loso_predictions)
        except R1Error as exc:
            loso[suite] = {"status": "NON_IDENTIFIABLE", "error": str(exc), "overall": {"status": "NON_IDENTIFIABLE"}, "per_suite": {}}
    for split_metrics in metrics_by_split.values():
        split_metrics["loso"] = loso
    devtest_gate = gate(metrics_by_split["DEVTEST"], protocol)
    return {
        "candidate": candidate,
        "status": "PASS_R1_CANDIDATE" if devtest_gate["pass"] else "R1_CANDIDATE_FAIL",
        "train_parent_count": len({parent_id(row) for row in train_rows}),
        "train_row_count": len(train_rows),
        "predictions": predictions,
        "metrics_by_split": metrics_by_split,
        "devtest_gate": devtest_gate,
        "model": model,
        "mean": mean,
        "std": std,
        "projection": projection,
    }


def candidate_decision(results: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    a = results.get("R1-A")
    b = results.get("R1-B")
    if a is None or b is None:
        return {"selected_candidate": None, "status": "STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR", "reason": "CANDIDATE_RESULT_MISSING"}
    a_gate = a["devtest_gate"]
    b_gate = b["devtest_gate"]
    a_overall = a["metrics_by_split"]["DEVTEST"]["overall"]
    b_overall = b["metrics_by_split"]["DEVTEST"]["overall"]
    a_suites = a["metrics_by_split"]["DEVTEST"]["per_suite"]
    b_suites = b["metrics_by_split"]["DEVTEST"]["per_suite"]
    a_worst = min((value["parent_macro_auc"] for value in a_suites.values() if value["status"] == "IDENTIFIABLE"), default=None)
    b_worst = min((value["parent_macro_auc"] for value in b_suites.values() if value["status"] == "IDENTIFIABLE"), default=None)
    macro_gain = (b_overall.get("parent_macro_auc") - a_overall.get("parent_macro_auc")) if b_overall.get("parent_macro_auc") is not None and a_overall.get("parent_macro_auc") is not None else None
    worst_gain = b_worst - a_worst if b_worst is not None and a_worst is not None else None
    replacement = bool(
        b_gate["pass"]
        and ((macro_gain is not None and macro_gain >= protocol["candidate_decision"]["context_replacement_macro_gain_min"])
             or (worst_gain is not None and worst_gain >= protocol["candidate_decision"]["context_replacement_worst_suite_gain_min"]))
    )
    if replacement:
        selected = "R1-B"
    elif a_gate["pass"]:
        selected = "R1-A"
    else:
        selected = None
    return {
        "selected_candidate": selected,
        "status": "STAGE_VIII_R1_SELECTOR_ESTABLISHED" if selected else protocol["candidate_decision"]["no_pass_status"],
        "r1_a_pass": a_gate["pass"],
        "r1_b_pass": b_gate["pass"],
        "context_macro_gain_over_a": macro_gain,
        "context_worst_suite_gain_over_a": worst_gain,
        "context_replacement_rule_pass": replacement,
        "tie_rule": protocol["candidate_decision"]["tie_rule"],
    }


def seal(root: Path, summary: dict[str, Any]) -> None:
    summary_path = root / "STAGE_VIII_R1_RELATIVE_SELECTOR.json"
    write_json(summary_path, summary)
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in SEAL_FILES):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VIII_R1_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": sha256_file(summary_path),
        "sha256sums_sha256": sums_sha,
        "candidate_training_performed": True,
        "new_m4_authorized": False,
        "intervention_executed": False,
        "pgd_authorized": False,
        "protected_counters": ZERO_COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output = args.output_root.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("schema") != "STAGE_VIII_R1_RELATIVE_SELECTOR_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_R1_TRAINING":
        raise SystemExit("PROTOCOL_NOT_FROZEN_R1")
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    if git("status", "--porcelain"):
        raise SystemExit("WORKTREE_NOT_CLEAN")
    r0_root = Path(protocol["r0_binding"]["root"])
    r0_binding = verify_r0_root(r0_root)
    if sha256_file(r0_root / "SHA256SUMS.sha256") != protocol["r0_binding"]["sha256sums_sha256"]:
        raise SystemExit("R0_SHA256SUMS_SIDECAR_MISMATCH")
    r0_summary = read_json(r0_root / "STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY.json")
    if r0_summary.get("status") != protocol["r0_binding"]["status"] or not r0_summary.get("decision", {}).get("r1_authorized"):
        raise SystemExit("R0_NOT_AUTHORIZING_R1")
    rows, streams, split_map, input_bindings = load_bound_population(protocol)
    context, context_binding = load_bound_context(protocol, rows)
    results: dict[str, dict[str, Any]] = {}
    for candidate in ("R1-A", "R1-B"):
        result = evaluate_candidate(candidate, rows, streams, split_map, context, protocol)
        results[candidate] = result
    decision = candidate_decision(results, protocol)
    output.mkdir(parents=True)
    (output / "STAGE_VIII_R1_PROTOCOL_V1.json").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    parent_metric_lines = []
    for candidate, result in results.items():
        candidate_dir = output / candidate
        candidate_dir.mkdir()
        torch.save({
            "schema": "STAGE_VIII_R1_PAIRWISE_SELECTOR_CHECKPOINT_V1",
            "candidate": candidate,
            "state_dict": result["model"].state_dict(),
            "mean": result["mean"],
            "std": result["std"],
            "projection": result["projection"],
            "seed": SEED,
            "primary_dose": "T5",
        }, candidate_dir / "CHECKPOINT.pt")
        prediction_path = candidate_dir / "T5_PREDICTIONS.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in result["predictions"]:
                handle.write(json.dumps({key: row.get(key) for key in ("stage", "canonical_parent_key", "suite", "dose", "probe_id", "probe_step", "consumable", "abstain", "y", "label_class", "score", "split")}, sort_keys=True) + "\n")
        summary = {key: value for key, value in result.items() if key not in {"predictions", "model", "mean", "std", "projection"}}
        write_json(candidate_dir / "CANDIDATE_SUMMARY.json", summary)
        for split, split_metrics in result["metrics_by_split"].items():
            for metric in split_metrics["parent_metrics"]:
                parent_metric_lines.append(json.dumps({"candidate": candidate, "scope": split, **metric}, sort_keys=True))
    (output / "R1_PARENT_METRICS.jsonl").write_text("\n".join(parent_metric_lines) + "\n", encoding="utf-8")
    summary = {
        "schema": "STAGE_VIII_R1_RELATIVE_SELECTOR_V1",
        "status": decision["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "r0_binding": r0_binding,
        "input_bindings": input_bindings,
        "context_binding": context_binding,
        "row_counts": {
            "all_rows": len(rows),
            "t5_rows": len(t5_rows(rows)),
            "consumable_t5_rows": len(t5_rows(rows, consumable_only=True)),
            "parent_count": len(split_map),
            "split_counts": {split: sum(value == split for value in split_map.values()) for split in ("TRAIN", "VAL", "DEVTEST")},
        },
        "candidates": {
            candidate: {key: value for key, value in result.items() if key not in {"predictions", "model", "mean", "std", "projection"}}
            for candidate, result in results.items()
        },
        "decision": decision,
        "new_m4": False,
        "intervention": False,
        "pgd_rollout": False,
        "protected_counters": ZERO_COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(output / "PROVENANCE.json", {
        "schema": "STAGE_VIII_R1_PROVENANCE_V1",
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script": summary["source_script"],
        "source_script_sha256": summary["source_script_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "r0_binding": r0_binding,
        "input_bindings": input_bindings,
        "context_binding": context_binding,
        "protected_counters": ZERO_COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    seal(output, summary)
    print(json.dumps({"status": summary["status"], "output_root": str(output), "selected_candidate": decision["selected_candidate"], "protected_counters": ZERO_COUNTERS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
