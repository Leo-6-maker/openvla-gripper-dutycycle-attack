#!/usr/bin/env python3
"""Train one frozen Stage VII context candidate, S7-B or S7-C."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v5"))
from run_stage_vii_domain_shift_forensic import (  # noqa: E402
    DOSES,
    THRESHOLD,
    attach_windows,
    load_clean_streams,
    load_labels,
    read_json,
    read_jsonl,
)
from run_stage_vii_s7a_candidate import primary_gate  # noqa: E402
from run_stage_vi_b2_candidates import ranking_loss  # noqa: E402


SEED = 20260816
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SEAL_EXCLUDED = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}


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


def verify_sealed_root(root: Path) -> str:
    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"UNSEALED_ROOT:{root}")
    sums_sha = sha256_file(sums)
    if sidecar.read_text(encoding="utf-8").strip() != f"{sums_sha}  SHA256SUMS":
        raise ValueError(f"SEAL_SIDECAR_MISMATCH:{root}")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, relative = line.partition("  ")
        if not sep or relative in listed or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"SEAL_ROW_INVALID:{root}:{relative}")
        target = root / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"SEAL_FILE_MISMATCH:{target}")
        listed.add(relative)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != listed | {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}:
        raise ValueError(f"SEAL_FILE_SET_MISMATCH:{root}")
    return sums_sha


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_split(path: Path) -> tuple[dict[str, str], str]:
    seal_sha = verify_sealed_root(path)
    split = read_json(path / "STAGE_VII_DEVELOPMENT_PARENT_SPLIT_V1.json")
    if split.get("status") != "PASS_STAGE_VII_DEVELOPMENT_SPLIT" or split.get("selection_used_labels") or split.get("selection_used_outcomes"):
        raise ValueError("SPLIT_NOT_FROZEN_OR_OUTCOME_BOUND")
    mapping = {str(row["canonical_parent_key"]): str(row["split"]) for row in split["rows"]}
    if len(mapping) != int(split["parent_count"]):
        raise ValueError("SPLIT_DUPLICATE_PARENT")
    return mapping, seal_sha


def load_embeddings(root: Path) -> tuple[dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    seal_sha = verify_sealed_root(root)
    summary = read_json(root / "SUMMARY.json")
    if summary.get("status") != "PASS_STAGE_VII_FROZEN_CONTEXT_EMBEDDINGS" or summary.get("labels_or_outcomes_read") or summary.get("suite_or_task_id_input") or summary.get("formal_m4_executed"):
        raise ValueError("FROZEN_EMBEDDING_ROOT_NOT_CLEAN")
    if summary.get("protected_counters") != {"protected_reads": 0, "eval160_reads": 0}:
        raise ValueError("FROZEN_EMBEDDING_COUNTER_VIOLATION")
    arrays = np.load(root / "FROZEN_CONTEXT_EMBEDDINGS.npz")
    visual, language = np.asarray(arrays["visual"], dtype=np.float32), np.asarray(arrays["language"], dtype=np.float32)
    rows = read_jsonl(root / "ROWS.jsonl")
    if len(rows) != len(visual) or len(rows) != len(language) or not np.isfinite(visual).all() or not np.isfinite(language).all():
        raise ValueError("FROZEN_EMBEDDING_SHAPE_OR_FINITE_FAILURE")
    result = {}
    for index, row in enumerate(rows):
        key = (str(row["stage"]), str(row["canonical_parent_key"]), int(row["probe_step"]))
        if key in result:
            raise ValueError(f"FROZEN_EMBEDDING_DUPLICATE:{key}")
        result[key] = (visual[index], language[index])
    return result, {"root": str(root), "sha256sums_sha256": seal_sha, "row_count": len(rows), "summary": summary}


def load_policy(root: Path) -> tuple[dict[tuple[str, str, int], np.ndarray], dict[str, Any]]:
    seal_sha = verify_sealed_root(root)
    summary = read_json(root / "STAGE_VII_CLEAN_POLICY_INTENT_MATERIALIZATION.json")
    counters = summary.get("protected_counters")
    required = (
        summary.get("status") == "PASS_STAGE_VII_CLEAN_POLICY_INTENT_MATERIALIZATION",
        summary.get("labels_or_outcomes_read") is False,
        summary.get("privileged_state_consumed") is False,
        summary.get("formal_m4_executed") is False,
        summary.get("all_generation_passes_one") is True,
        summary.get("all_single_generation_parity") is True,
        summary.get("all_score_adapter_parity") is True,
        summary.get("all_reference_action_parity") is True,
        isinstance(counters, dict) and counters.get("protected_reads") == 0 and counters.get("eval160_reads") == 0 and all(value == 0 for value in counters.values()),
    )
    if not all(required):
        raise ValueError("POLICY_ROOT_NOT_CLEAN_OR_PARITY_COMPLETE")
    rows = read_jsonl(root / "POLICY_INTENT_ROWS.jsonl")
    result = {}
    for row in rows:
        key = (str(row["stage"]), str(row["canonical_parent_key"]), int(row["probe_step"]))
        values = np.asarray(row["clean_policy_intent_9d"], dtype=np.float32)
        if values.shape != (9,) or not np.isfinite(values).all() or key in result:
            raise ValueError(f"POLICY_ROW_INVALID_OR_DUPLICATE:{key}")
        result[key] = values
    if len(result) != int(summary["row_count"]):
        raise ValueError("POLICY_ROW_COUNT_MISMATCH")
    return result, {"root": str(root), "sha256sums_sha256": seal_sha, "row_count": len(rows), "summary": summary}


def fit_projection(rows: list[dict[str, Any]], candidate: str) -> dict[str, np.ndarray | int | str]:
    language = np.asarray([row["language"] for row in rows], dtype=np.float32)
    policy = np.asarray([row["policy"] for row in rows], dtype=np.float32)
    language_mean = language.mean(axis=0)
    language_scale = np.where(language.std(axis=0) > 1e-8, language.std(axis=0), 1.0)
    language_scaled = (language - language_mean) / language_scale
    language_pca = PCA(n_components=16, svd_solver="full", random_state=SEED).fit(language_scaled)
    policy_mean = policy.mean(axis=0)
    policy_scale = np.where(policy.std(axis=0) > 1e-8, policy.std(axis=0), 1.0)
    projection: dict[str, np.ndarray | int | str] = {
        "candidate": candidate,
        "language_mean": language_mean.astype(np.float32),
        "language_scale": language_scale.astype(np.float32),
        "language_components": language_pca.components_.astype(np.float32),
        "language_pca_mean": language_pca.mean_.astype(np.float32),
        "policy_mean": policy_mean.astype(np.float32),
        "policy_scale": policy_scale.astype(np.float32),
        "language_pca_components": 16,
    }
    if candidate == "S7-C":
        visual = np.asarray([row["visual"] for row in rows], dtype=np.float32)
        visual_mean = visual.mean(axis=0)
        visual_scale = np.where(visual.std(axis=0) > 1e-8, visual.std(axis=0), 1.0)
        visual_scaled = (visual - visual_mean) / visual_scale
        visual_pca = PCA(n_components=16, svd_solver="full", random_state=SEED).fit(visual_scaled)
        projection.update({
            "visual_mean": visual_mean.astype(np.float32),
            "visual_scale": visual_scale.astype(np.float32),
            "visual_components": visual_pca.components_.astype(np.float32),
            "visual_pca_mean": visual_pca.mean_.astype(np.float32),
            "visual_pca_components": 16,
        })
    return projection


def project_context(rows: list[dict[str, Any]], projection: dict[str, np.ndarray | int | str], candidate: str) -> np.ndarray:
    language = np.asarray([row["language"] for row in rows], dtype=np.float32)
    language = (language - projection["language_mean"]) / projection["language_scale"]
    language = (language - projection["language_pca_mean"]) @ projection["language_components"].T
    policy = np.asarray([row["policy"] for row in rows], dtype=np.float32)
    policy = (policy - projection["policy_mean"]) / projection["policy_scale"]
    values = [language, policy]
    if candidate == "S7-C":
        visual = np.asarray([row["visual"] for row in rows], dtype=np.float32)
        visual = (visual - projection["visual_mean"]) / projection["visual_scale"]
        visual = (visual - projection["visual_pca_mean"]) @ projection["visual_components"].T
        values.append(visual)
    return np.concatenate(values, axis=1).astype(np.float32)


class MultiDoseContextTCN(nn.Module):
    def __init__(self, context_dim: int):
        super().__init__()
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(25, 32, self.kernel_size)
        self.conv2 = nn.Conv1d(32, 32, self.kernel_size)
        self.context = nn.Linear(context_dim, 32)
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        temporal = x.transpose(1, 2)
        temporal = F.pad(temporal, (self.kernel_size - 1, 0))
        temporal = F.relu(self.conv1(temporal))
        temporal = F.pad(temporal, (self.kernel_size - 1, 0))
        temporal = F.relu(self.conv2(temporal))[:, :, -1]
        context = F.relu(self.context(context))
        return self.head(torch.cat([temporal, context], dim=1))


def standardization(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted({row["canonical_parent_key"] for row in rows})
    values = np.concatenate([streams[key]["features"] for key in keys], axis=0).astype(np.float32)
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) > 1e-8, values.std(axis=0), 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def arrays(rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray, projection: dict[str, Any], candidate: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if row["consumable"]]
    if not usable:
        raise ValueError("NO_CONSUMABLE_ROWS")
    x = (np.asarray([row["window"] for row in usable], dtype=np.float32) - mean) / std
    context = project_context(usable, projection, candidate)
    y = np.asarray([row["y"] for row in usable], dtype=np.float32)
    dose = np.asarray([DOSES.index(row["dose"]) for row in usable], dtype=np.int64)
    groups = np.asarray([f"{row['stage']}::{row['canonical_parent_key']}::{row['dose']}" for row in usable])
    parent_counts: defaultdict[str, int] = defaultdict(int)
    suite_parents: defaultdict[str, set[str]] = defaultdict(set)
    for row in usable:
        parent = f"{row['stage']}::{row['canonical_parent_key']}"
        parent_counts[parent] += 1
        suite_parents[row["suite"]].add(parent)
    weights = np.asarray([
        1.0 / (len(suite_parents[row["suite"]]) * parent_counts[f"{row['stage']}::{row['canonical_parent_key']}"])
        for row in usable
    ], dtype=np.float32)
    weights /= weights.mean()
    return x, context, y, dose, groups, weights


def fit_model(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], candidate: str, epochs: int = 120) -> tuple[MultiDoseContextTCN, dict[str, Any]]:
    seed_all()
    mean, std = standardization(rows, streams)
    projection = fit_projection(rows, candidate)
    x, context, y, dose, groups, weights = arrays(rows, mean, std, projection, candidate)
    model = MultiDoseContextTCN(context.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    context_tensor = torch.as_tensor(context, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    dose_tensor = torch.as_tensor(dose, dtype=torch.long)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    pos_weight = []
    for dose_index in range(3):
        mask = dose == dose_index
        positives = max(1, int(y[mask].sum()))
        negatives = max(1, int(mask.sum()) - positives)
        pos_weight.append(float(negatives / positives))
    class_weights = torch.as_tensor(pos_weight, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor, context_tensor)
        selected = logits[torch.arange(len(logits)), dose_tensor]
        bce_terms = []
        for dose_index in range(3):
            mask = dose_tensor == dose_index
            if bool(mask.any()):
                bce_terms.append(F.binary_cross_entropy_with_logits(
                    selected[mask], y_tensor[mask], weight=weight_tensor[mask], pos_weight=class_weights[dose_index]
                ))
        bce = torch.stack(bce_terms).mean()
        loss = bce + 0.25 * ranking_loss(selected, y_tensor, groups)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, {"mean": mean, "std": std, "projection": projection}


def predict_rows(model: MultiDoseContextTCN, rows: list[dict[str, Any]], prep: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    x = (np.asarray([row["window"] for row in rows], dtype=np.float32) - prep["mean"]) / prep["std"]
    context = project_context(rows, prep["projection"], candidate)
    doses = np.asarray([DOSES.index(row["dose"]) for row in rows], dtype=np.int64)
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(context, dtype=torch.float32))
        scores = torch.sigmoid(logits)[torch.arange(len(logits)), torch.as_tensor(doses)].numpy()
    return [{**row, "score": float(score), "uncertainty_margin_to_frozen_threshold": float(abs(score - THRESHOLD))} for row, score in zip(rows, scores)]


def evaluate_splits(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], split_map: dict[str, str], candidate: str) -> tuple[dict[str, Any], MultiDoseContextTCN, dict[str, Any], list[dict[str, Any]]]:
    train_rows = [row for row in rows if split_map[row["canonical_parent_key"]] == "TRAIN"]
    model, prep = fit_model(train_rows, streams, candidate)
    predictions = predict_rows(model, rows, prep, candidate)
    metrics = {
        split: primary_gate([row for row in predictions if split_map[row["canonical_parent_key"]] == split])
        for split in ("TRAIN", "VAL", "DEVTEST")
    }
    return metrics, model, prep, predictions


def evaluate_loso(rows: list[dict[str, Any]], streams: dict[str, dict[str, Any]], candidate: str) -> dict[str, Any]:
    result = {}
    for suite in sorted({row["suite"] for row in rows}):
        train_keys = {row["canonical_parent_key"] for row in rows if row["suite"] != suite}
        train_rows = [row for row in rows if row["canonical_parent_key"] in train_keys]
        test_rows = [row for row in rows if row["suite"] == suite]
        model, prep = fit_model(train_rows, streams, candidate, epochs=80)
        result[suite] = primary_gate(predict_rows(model, test_rows, prep, candidate))
    values = [item["summary"]["consumable_metrics"].get("auroc") for item in result.values()]
    values = [value for value in values if value is not None]
    return {"per_suite": result, "mean_identifiable_auroc": float(np.mean(values)) if values else None}


def seal(root: Path, summary: dict[str, Any]) -> None:
    entries = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in SEAL_EXCLUDED
    ]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    seal_value = {
        "schema": "STAGE_VII_S7BC_CANDIDATE_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "candidate": summary["candidate"],
        "candidate_training_performed": True,
        "formal_m4_executed": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(root / "ROOT_SEAL.json", seal_value)
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("S7-B", "S7-C"), required=True)
    parser.add_argument("--stage-v-clean-root", required=True, type=Path)
    parser.add_argument("--stage-v-labels", required=True, type=Path)
    parser.add_argument("--stage-vi-clean-root", required=True, type=Path)
    parser.add_argument("--stage-vi-labels", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--frozen-embeddings", required=True, type=Path)
    parser.add_argument("--policy-intent-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{args.output_root}")
    protocol = read_json(args.protocol.resolve())
    if protocol.get("schema") != "STAGE_VII_CONTEXT_CANDIDATE_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_S7_BC_TRAINING" or args.candidate not in protocol.get("candidates", []):
        raise SystemExit("CANDIDATE_PROTOCOL_NOT_FROZEN")
    if args.candidate == "S7-C" and float(protocol["visual_authorization"]["observed_loso_mean_auroc_gain_over_P0"]) < float(protocol["visual_authorization"]["required_loso_mean_auroc_gain_over_P0"]):
        raise SystemExit("S7_C_VISUAL_AUTHORIZATION_MISSING")
    split_map, split_seal_sha = load_split(args.split_root.resolve())
    embeddings, embedding_meta = load_embeddings(args.frozen_embeddings.resolve())
    policy, policy_meta = load_policy(args.policy_intent_root.resolve())
    stage_v_streams = load_clean_streams(args.stage_v_clean_root.resolve(), "stage_v")
    stage_vi_streams = load_clean_streams(args.stage_vi_clean_root.resolve(), "stage_vi")
    streams = {**stage_v_streams, **stage_vi_streams}
    if set(streams) != set(split_map):
        raise SystemExit("SPLIT_STREAM_POPULATION_MISMATCH")
    labels = attach_windows(
        load_labels(args.stage_v_labels.resolve(), "STAGE_V") + load_labels(args.stage_vi_labels.resolve(), "STAGE_VI_B2"), streams
    )
    for row in labels:
        key = (row["stage"], row["canonical_parent_key"], row["probe_step"])
        if key not in embeddings or key not in policy:
            raise SystemExit(f"CONTEXT_JOIN_MISSING:{key}")
        row["visual"], row["language"] = embeddings[key]
        row["policy"] = policy[key]
    train_rows = [row for row in labels if split_map[row["canonical_parent_key"]] == "TRAIN"]
    if len({row["canonical_parent_key"] for row in train_rows}) == 0:
        raise SystemExit("EMPTY_TRAIN")
    split_metrics, model, prep, predictions = evaluate_splits(labels, streams, split_map, args.candidate)
    loso = evaluate_loso(labels, streams, args.candidate)
    devtest_pass = bool(split_metrics["DEVTEST"]["pass"])
    status = f"PASS_STAGE_VII_{args.candidate.replace('-', '')}_DEVELOPMENT" if devtest_pass else f"STAGE_VII_{args.candidate.replace('-', '')}_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR"
    output = args.output_root.resolve()
    output.mkdir(parents=True)
    checkpoint = {
        "schema": "STAGE_VII_S7BC_CONTEXT_TCN_CHECKPOINT_V1",
        "candidate": args.candidate,
        "state_dict": model.state_dict(),
        "mean": prep["mean"],
        "std": prep["std"],
        "projection": prep["projection"],
        "doses": list(DOSES),
        "threshold": THRESHOLD,
    }
    torch.save(checkpoint, output / f"{args.candidate.replace('-', '_')}_CHECKPOINT.pt")
    with (output / f"{args.candidate.replace('-', '_')}_PREDICTIONS.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps({key: row[key] for key in ("stage", "canonical_parent_key", "suite", "dose", "probe_id", "probe_step", "consumable", "abstain", "y", "label_class", "score", "uncertainty_margin_to_frozen_threshold")}, sort_keys=True) + "\n")
    summary = {
        "schema": "STAGE_VII_S7BC_CANDIDATE_DEVELOPMENT_V1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "candidate": args.candidate,
        "candidate_training_performed": True,
        "inputs": list(protocol["input_order"].values()) if args.candidate == "S7-C" else ["16x25D_causal_history", "frozen_clean_language_embedding", "clean_policy_intent"],
        "projection": {
            "fit_scope": "TRAIN_PARENTS_ONLY",
            "language_components": 16,
            "visual_components": 16 if args.candidate == "S7-C" else None,
            "policy_dimensions": 9,
        },
        "model": protocol["model"],
        "threshold": THRESHOLD,
        "abstains_masked_never_negative": True,
        "uncertainty": "score distance to frozen threshold recorded; promotion requires a separately frozen emission/abstention rule",
        "split_root": str(args.split_root.resolve()),
        "split_root_sha256s_sha256": split_seal_sha,
        "frozen_embeddings": embedding_meta,
        "policy_intent_root": policy_meta,
        "metrics_by_split": split_metrics,
        "loso": loso,
        "protected_counters": COUNTERS,
        "formal_m4_executed": False,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(output / f"STAGE_VII_{args.candidate.replace('-', '')}_CANDIDATE_DEVELOPMENT.json", summary)
    seal(output, summary)
    print(json.dumps({"status": status, "candidate": args.candidate, "output_root": str(output), "devtest_pass": devtest_pass, "loso_mean_identifiable_auroc": loso["mean_identifiable_auroc"], "protected_counters": COUNTERS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
