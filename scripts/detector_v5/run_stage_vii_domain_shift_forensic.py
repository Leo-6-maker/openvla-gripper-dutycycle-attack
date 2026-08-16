"""Read-only Stage VII domain-shift forensic.

This is development evidence only.  It does not train S7-A/B/C, launch M4,
read protected data, or alter any frozen Stage V/VI artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v5"))

from gripper_attack.v5_r3_features import FEATURE_ORDER, materialize_fit670_features  # noqa: E402
from run_stage_vi_b2_candidates import CausalTCN, metric, predict_tcn, seed_all  # noqa: E402


SEED = 20260816
THRESHOLD = 0.69
ZERO_COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
DOSES = ("T3", "T5", "T10")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def safe_counters(value: Any) -> None:
    if value is None:
        return
    if value != ZERO_COUNTERS:
        raise ValueError(f"PROTECTED_COUNTER_VIOLATION:{value}")


def identity_parts(key: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"(libero_[^/]+)/task_(\d+)/state_(\d+)", key)
    if not match:
        raise ValueError(f"BAD_PARENT_KEY:{key}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def clean_episode_from_replay(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("replay_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("EMPTY_CLEAN_REPLAY")
    steps = []
    telemetry = []
    for expected, row in enumerate(rows):
        if int(row.get("step", -1)) != expected:
            raise ValueError(f"CLEAN_REPLAY_STEP_GAP:{data.get('canonical_parent_key')}:{expected}")
        steps.append({
            "step": expected,
            "raw_action_7d": row["raw_action_7d"],
            "action_env_7d": row["env_action_7d"],
        })
        telemetry.append({
            "step": expected,
            "robot0_gripper_qpos": row["robot0_gripper_qpos"],
            "robot0_eef_pos": row["robot0_eef_pos"],
        })
    return {"steps": steps, "telemetry": telemetry}


def load_clean_streams(root: Path, kind: str) -> dict[str, dict[str, Any]]:
    if kind == "stage_v":
        paths = sorted(root.glob("parents/*/CLEAN_REPLAY_STUDENT_INPUTS_V1.json"))
    elif kind == "stage_vi":
        paths = sorted(root.glob("parents/*/RECONSTRUCTED_FIT670_EPISODE.json"))
    else:
        raise ValueError(kind)
    if not paths:
        raise ValueError(f"NO_CLEAN_STREAMS:{root}")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path)
        key = str(data.get("canonical_parent_key") or data.get("episode_id"))
        if key in result:
            raise ValueError(f"DUPLICATE_CLEAN_STREAM:{key}")
        safe_counters(data.get("protected_counters"))
        if data.get("intervention_executed") is True or data.get("outcomes_read") is True or data.get("attack_enabled") is True:
            raise ValueError(f"CLEAN_STREAM_NOT_CLEAN:{key}")
        episode = clean_episode_from_replay(data) if kind == "stage_v" else data
        materialized = materialize_fit670_features(episode)
        vectors = np.asarray([row["features_25d"] for row in materialized], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != 25 or not np.isfinite(vectors).all():
            raise ValueError(f"BAD_25D_STREAM:{key}")
        suite, task, state = identity_parts(key)
        result[key] = {
            "canonical_parent_key": key,
            "suite": suite,
            "task": task,
            "state": state,
            "features": vectors,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "step_count": int(len(vectors)),
        }
    return result


def load_labels(path: Path, stage: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[tuple[str, str, int]] = set()
    result = []
    for row in rows:
        key = str(row["canonical_parent_key"])
        dose = str(row["dose"])
        step = int(row["probe_step"])
        if dose not in DOSES:
            raise ValueError(f"BAD_DOSE:{dose}")
        identity_parts(key)
        join_key = (key, dose, step)
        if join_key in seen:
            raise ValueError(f"DUPLICATE_LABEL:{stage}:{join_key}")
        seen.add(join_key)
        safe_counters(row.get("protected_counters"))
        consumable = bool(row.get("binary_label_consumable"))
        label_class = str(row.get("label_class", ""))
        if consumable and label_class not in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}:
            raise ValueError(f"BAD_CONSUMABLE_LABEL:{stage}:{join_key}:{label_class}")
        if not consumable and label_class in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}:
            raise ValueError(f"ABSTAIN_LABEL_CLASS_MISMATCH:{stage}:{join_key}")
        result.append({
            "stage": stage,
            "canonical_parent_key": key,
            "suite": identity_parts(key)[0],
            "dose": dose,
            "probe_id": str(row["probe_id"]),
            "probe_step": step,
            "consumable": consumable,
            "abstain": not consumable,
            "y": None if not consumable else int(label_class == "V_PHYS"),
            "label_class": label_class,
        })
    if not result:
        raise ValueError(f"EMPTY_LABELS:{path}")
    return result


def attach_windows(labels: list[dict[str, Any]], streams: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    zero = np.zeros(25, dtype=np.float32)
    result = []
    for row in labels:
        stream = streams.get(row["canonical_parent_key"])
        if stream is None:
            raise ValueError(f"LABEL_STREAM_JOIN_MISSING:{row}")
        step = row["probe_step"]
        if step < 0 or step >= stream["step_count"]:
            raise ValueError(f"LABEL_STEP_OUT_OF_RANGE:{row['canonical_parent_key']}:{step}:{stream['step_count']}")
        history = np.stack([
            stream["features"][past] if past >= 0 else zero
            for past in range(step - 15, step + 1)
        ]).astype(np.float32)
        result.append({**row, "window": history, "feature_25d": history[-1]})
    return result


def load_policy_intent(root: Path) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, Any]]:
    files = sorted(root.glob("**/policy_intent_records.jsonl"))
    if not files:
        return {}, {"status": "UNAVAILABLE_NO_POLICY_INTENT_RECORDS", "file_count": 0}
    candidates: dict[tuple[str, int], list[tuple[np.ndarray, str]]] = defaultdict(list)
    languages: dict[str, set[str]] = defaultdict(set)
    for path in files:
        parts = path.relative_to(root).parts
        identity_name = next((part for part in parts if "__task_" in part), None)
        if identity_name is None:
            raise ValueError(f"POLICY_IDENTITY_MISSING:{path}")
        match = re.fullmatch(r"(libero_[^_]+)__task_(\d+)__state_(\d+)", identity_name)
        if match is None:
            raise ValueError(f"POLICY_IDENTITY_BAD:{path}")
        key = f"{match.group(1)}/task_{int(match.group(2)):02d}/state_{int(match.group(3)):02d}"
        for row in read_jsonl(path):
            step = int(row["step"])
            values = np.asarray(row["clean_policy_intent_9d"], dtype=np.float32)
            if values.shape != (9,) or not np.isfinite(values).all():
                raise ValueError(f"POLICY_VECTOR_BAD:{key}:{step}")
            language = str(row.get("task_language", ""))
            candidates[(key, step)].append((values, language))
            languages[key].add(language)
    result: dict[tuple[str, int], np.ndarray] = {}
    conflicts = 0
    for join_key, values in candidates.items():
        first = values[0][0]
        if all(np.array_equal(first, value[0]) and values[0][1] == value[1] for value in values[1:]):
            result[join_key] = first
        else:
            conflicts += 1
    return result, {
        "status": "PASS_CLEAN_POLICY_INTENT_WITH_CONFLICTS" if conflicts else "PASS_CLEAN_POLICY_INTENT",
        "file_count": len(files),
        "identity_count": len(languages),
        "row_join_count": len(candidates),
        "usable_join_count": len(result),
        "conflict_join_count": conflicts,
        "language_identity_count": sum(bool(value) for value in languages.values()),
    }


def snapshot_inventory(root: Path) -> dict[str, Any]:
    paths = sorted(root.glob("**/CAUSAL_PROBE_SNAPSHOT_V2.json"))
    identities: set[str] = set()
    probes: set[tuple[str, str]] = set()
    fields = defaultdict(int)
    missing_arrays = []
    source_commits: set[str] = set()
    source_trees: set[str] = set()
    for path in paths:
        data = read_json(path)
        if data.get("status") != "SEALED_PROSPECTIVE_SNAPSHOT":
            raise ValueError(f"SNAPSHOT_STATUS:{path}")
        binding = data.get("binding") or {}
        key = str(binding.get("parent_key", ""))
        probe = str(binding.get("probe_id", ""))
        identity_parts(key)
        identities.add(key)
        probes.add((key, probe))
        source_commits.add(str(binding.get("source_commit", "")))
        source_trees.add(str(binding.get("source_tree", "")))
        descriptors = data.get("arrays") or []
        for field in ("canonical_policy_rgb_224", "pixel_values", "input_ids", "prompt"):
            if field not in (data.get("payload") or {}):
                continue
            fields[field] += 1
            reference = data["payload"][field]
            if isinstance(reference, dict) and "__causal_array__" in reference:
                index = int(reference["__causal_array__"])
                descriptor = descriptors[index]
                if not (path.parent / descriptor["binary_path"]).is_file():
                    missing_arrays.append(str(path.parent / descriptor["binary_path"]))
    return {
        "root": str(root),
        "snapshot_count": len(paths),
        "identity_count": len(identities),
        "probe_count": len(probes),
        "field_counts": dict(sorted(fields.items())),
        "missing_array_count": len(missing_arrays),
        "missing_array_examples": missing_arrays[:5],
        "source_commits": sorted(source_commits),
        "source_trees": sorted(source_trees),
        "raw_clean_rgb_available": fields["canonical_policy_rgb_224"] == len(paths),
        "raw_clean_pixel_values_available": fields["pixel_values"] == len(paths),
        "raw_clean_input_ids_available": fields["input_ids"] == len(paths),
        "frozen_visual_embedding_available": False,
        "frozen_language_embedding_available": False,
    }


def score_stats(scores: np.ndarray) -> dict[str, float] | None:
    if not len(scores):
        return None
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "q10": float(np.quantile(scores, 0.10)),
        "q25": float(np.quantile(scores, 0.25)),
        "median": float(np.median(scores)),
        "q75": float(np.quantile(scores, 0.75)),
        "q90": float(np.quantile(scores, 0.90)),
        "max": float(np.max(scores)),
    }


def metric_with_lift(y: np.ndarray, scores: np.ndarray, threshold: float = THRESHOLD) -> dict[str, Any]:
    value = metric(y, scores, threshold)
    prevalence = value.get("prevalence")
    value["auprc_lift"] = None if value.get("auprc") is None or not prevalence else float(value["auprc"] / prevalence)
    value["emission_rate"] = float(np.mean(scores >= threshold)) if len(scores) else None
    return value


def summarize_scores(rows: list[dict[str, Any]], threshold: float = THRESHOLD) -> dict[str, Any]:
    consumable = [row for row in rows if row["consumable"]]
    score_array = np.asarray([row["score"] for row in rows], dtype=np.float64)
    result = {
        "score_count": len(rows),
        "consumable_count": len(consumable),
        "abstain_count": len(rows) - len(consumable),
        "abstain_rate": float((len(rows) - len(consumable)) / len(rows)) if rows else None,
        "score_stats_all": score_stats(score_array),
        "emission_rate_all": float(np.mean(score_array >= threshold)) if rows else None,
        "threshold": threshold,
    }
    if consumable:
        y = np.asarray([row["y"] for row in consumable], dtype=np.int64)
        score = np.asarray([row["score"] for row in consumable], dtype=np.float64)
        result["consumable_metrics"] = metric_with_lift(y, score, threshold)
        result["score_stats_consumable"] = score_stats(score)
    else:
        result["consumable_metrics"] = metric_with_lift(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), threshold)
        result["score_stats_consumable"] = None
    return result


def grouped_score_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: summarize_scores(value) for key, value in sorted(groups.items())}


def load_b2c_model(path: Path) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("schema") != "16x25D":
        raise ValueError(f"B2C_SCHEMA:{data.get('schema')}")
    model = CausalTCN()
    model.load_state_dict(data["state_dict"])
    model.eval()
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("B2C_STANDARDIZATION_BAD")
    return model, mean, std


def score_b2c(rows: list[dict[str, Any]], model_path: Path) -> list[dict[str, Any]]:
    model, mean, std = load_b2c_model(model_path)
    selected = [row for row in rows if row["dose"] == "T5"]
    windows = np.asarray([row["window"] for row in selected], dtype=np.float32)
    normalized = (windows - mean) / np.where(std > 1e-8, std, 1.0)
    scores = predict_tcn(model, normalized)
    if len(scores) != len(selected):
        raise ValueError("B2C_SCORE_LENGTH")
    return [{**row, "score": float(score)} for row, score in zip(selected, scores)]


def feature_shift(reference: dict[str, np.ndarray], comparison: dict[str, np.ndarray]) -> dict[str, Any]:
    def flatten(values: dict[str, np.ndarray], suite: str | None = None) -> np.ndarray:
        chunks = [array for key, array in values.items() if suite is None or key.startswith(suite + "/")]
        if not chunks:
            return np.empty((0, 25), dtype=np.float64)
        return np.concatenate(chunks, axis=0).astype(np.float64)

    def one(ref: np.ndarray, comp: np.ndarray) -> dict[str, Any]:
        if not len(ref) or not len(comp):
            return {"status": "UNAVAILABLE_NO_SHARED_FEATURES", "reference_rows": len(ref), "comparison_rows": len(comp)}
        ref_mean = ref.mean(axis=0)
        comp_mean = comp.mean(axis=0)
        ref_var = ref.var(axis=0)
        comp_var = comp.var(axis=0)
        smd = np.abs(comp_mean - ref_mean) / np.sqrt(np.maximum((ref_var + comp_var) / 2.0, 1e-12))
        covariance = np.cov(ref, rowvar=False) + np.eye(ref.shape[1]) * 1e-6
        delta = comp_mean - ref_mean
        mahalanobis = float(np.sqrt(np.maximum(delta @ np.linalg.pinv(covariance) @ delta, 0.0)))
        pca = PCA(n_components=min(5, ref.shape[1], len(ref))).fit(ref)
        ref_pca = pca.transform(ref)
        comp_pca = pca.transform(comp)
        psi = []
        for column in range(ref.shape[1]):
            edges = np.unique(np.quantile(ref[:, column], np.linspace(0.0, 1.0, 11)))
            if len(edges) < 2:
                psi.append(0.0)
                continue
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_hist = np.histogram(ref[:, column], bins=edges)[0].astype(float)
            comp_hist = np.histogram(comp[:, column], bins=edges)[0].astype(float)
            ref_prob = (ref_hist + 1e-6) / (ref_hist.sum() + 1e-6 * len(ref_hist))
            comp_prob = (comp_hist + 1e-6) / (comp_hist.sum() + 1e-6 * len(comp_hist))
            psi.append(float(np.sum((comp_prob - ref_prob) * np.log(comp_prob / ref_prob))))
        return {
            "status": "PASS",
            "reference_rows": int(len(ref)),
            "comparison_rows": int(len(comp)),
            "reference_mean": ref_mean.tolist(),
            "comparison_mean": comp_mean.tolist(),
            "mean_delta": delta.tolist(),
            "smd": smd.tolist(),
            "mean_abs_smd": float(np.mean(smd)),
            "max_smd": float(np.max(smd)),
            "mahalanobis_mean_delta": mahalanobis,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_mean_delta": (comp_pca.mean(axis=0) - ref_pca.mean(axis=0)).tolist(),
            "psi": psi,
            "mean_psi": float(np.mean(psi)),
            "max_psi": float(np.max(psi)),
        }

    suites = sorted({key.split("/", 1)[0] for key in reference} | {key.split("/", 1)[0] for key in comparison})
    result = {"overall": one(flatten(reference), flatten(comparison))}
    result["per_suite"] = {suite: one(flatten(reference, suite), flatten(comparison, suite)) for suite in suites}
    result["feature_order"] = list(FEATURE_ORDER)
    return result


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, suites: np.ndarray) -> dict[str, Any]:
    def fit_predict(train: np.ndarray, test: np.ndarray) -> np.ndarray | None:
        if len(np.unique(y[train])) < 2:
            return None
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        )
        estimator.fit(X[train], y[train])
        return estimator.predict_proba(X[test])[:, 1]

    oof_scores = np.full(len(y), np.nan, dtype=np.float64)
    for group in sorted(set(groups.tolist())):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        prediction = fit_predict(train, test)
        if prediction is not None:
            oof_scores[test] = prediction
    valid = np.isfinite(oof_scores)
    oof = metric_with_lift(y[valid], oof_scores[valid]) if valid.any() else None

    loso = {}
    for suite in sorted(set(suites.tolist())):
        test = np.flatnonzero(suites == suite)
        train = np.flatnonzero(suites != suite)
        prediction = fit_predict(train, test)
        loso[suite] = None if prediction is None else metric_with_lift(y[test], prediction)
    identified = [value["auroc"] for value in loso.values() if value and value.get("auroc") is not None]
    return {
        "status": "PASS_DIAGNOSTIC_PROBE",
        "rows": int(len(y)),
        "parent_group_count": int(len(set(groups.tolist()))),
        "parent_grouped_oof": oof,
        "leave_one_suite_out": loso,
        "loso_identifiable_suite_mean_auroc": float(np.mean(identified)) if identified else None,
        "final_candidate_authorized": False,
    }


def context_probes(rows: list[dict[str, Any]], policy: dict[tuple[str, int], np.ndarray]) -> dict[str, Any]:
    usable = [row for row in rows if row["consumable"] and row["dose"] == "T5"]
    if not usable:
        raise ValueError("NO_CONSUMABLE_T5_FOR_CONTEXT_PROBES")
    groups = np.asarray([f"{row['stage']}::{row['canonical_parent_key']}" for row in usable])
    suites = np.asarray([row["suite"] for row in usable])
    y = np.asarray([row["y"] for row in usable], dtype=np.int64)
    p0 = np.asarray([row["window"].reshape(-1) for row in usable], dtype=np.float32)
    result: dict[str, Any] = {
        "candidate_training_performed": False,
        "target": "V_phys@T5",
        "abstains_used_as_negative": False,
        "P0_25D": fit_probe(p0, y, groups, suites),
        "P1_25D_plus_language": {"status": "UNAVAILABLE_NO_FROZEN_LANGUAGE_EMBEDDING"},
        "P2_25D_plus_policy_intent": None,
        "P3_P1_plus_policy_intent": {"status": "UNAVAILABLE_P1"},
        "P4_25D_plus_frozen_visual": {"status": "UNAVAILABLE_NO_FROZEN_VISUAL_EMBEDDING"},
        "P5_P3_plus_frozen_visual": {"status": "UNAVAILABLE_P3_OR_P4"},
    }
    policy_rows = [row for row in usable if (row["canonical_parent_key"], row["probe_step"]) in policy]
    if not policy_rows:
        result["P2_25D_plus_policy_intent"] = {"status": "UNAVAILABLE_NO_POLICY_JOIN"}
    else:
        policy_groups = np.asarray([f"{row['stage']}::{row['canonical_parent_key']}" for row in policy_rows])
        policy_suites = np.asarray([row["suite"] for row in policy_rows])
        policy_y = np.asarray([row["y"] for row in policy_rows], dtype=np.int64)
        policy_x = np.asarray([
            np.concatenate([row["window"].reshape(-1), policy[(row["canonical_parent_key"], row["probe_step"])]])
            for row in policy_rows
        ], dtype=np.float32)
        result["P2_25D_plus_policy_intent"] = {
            "status": "PASS_DIAGNOSTIC_PROBE",
            "policy_join_rows": len(policy_rows),
            "policy_join_coverage": float(len(policy_rows) / len(usable)),
            "probe": fit_probe(policy_x, policy_y, policy_groups, policy_suites),
        }
    oracle_suites = sorted(set(suites.tolist()))
    oracle_tasks = sorted({identity_parts(row["canonical_parent_key"])[1] for row in usable})
    oracle_x = []
    for row in usable:
        task = identity_parts(row["canonical_parent_key"])[1]
        oracle_x.append([
            float(row["suite"] == suite) for suite in oracle_suites
        ] + [float(task == value) for value in oracle_tasks])
    result["P_ORACLE_SUITE_TASK"] = {
        "status": "PASS_DIAGNOSTIC_UPPER_BOUND",
        "forbidden_final_input": True,
        "probe": fit_probe(np.asarray(oracle_x, dtype=np.float32), y, groups, suites),
    }
    return result


def root_binding(path: Path) -> dict[str, Any]:
    candidates = [path / "SHA256SUMS", path / "ROOT_SEAL.json", path / "ROOT_SEAL.sha256"] if path.is_dir() else [path]
    files = [item for item in candidates if item.is_file()]
    return {"path": str(path), "markers": [{"path": str(item), "sha256": sha256_file(item)} for item in files]}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    sums = "\n".join(entries) + "\n"
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    seal = {
        "schema": "STAGE_VII_DOMAIN_SHIFT_FORENSIC_ROOT_SEAL_V1",
        "status": "PASS_STAGE_VII_DOMAIN_SHIFT_FORENSIC",
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "candidate_training_performed": False,
        "formal_m4_executed": False,
        "protected_counters": ZERO_COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(root / "ROOT_SEAL.json", seal)
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-v-clean-root", required=True, type=Path)
    parser.add_argument("--stage-v-labels", required=True, type=Path)
    parser.add_argument("--stage-v-snapshot-root", required=True, type=Path)
    parser.add_argument("--stage-vi-clean-root", required=True, type=Path)
    parser.add_argument("--stage-vi-labels", required=True, type=Path)
    parser.add_argument("--stage-vi-snapshot-root", required=True, type=Path)
    parser.add_argument("--b2c-model", required=True, type=Path)
    parser.add_argument("--policy-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    for path in (
        args.stage_v_clean_root,
        args.stage_v_labels,
        args.stage_v_snapshot_root,
        args.stage_vi_clean_root,
        args.stage_vi_labels,
        args.stage_vi_snapshot_root,
        args.b2c_model,
        args.policy_root,
    ):
        if not path.exists():
            raise SystemExit(f"MISSING_INPUT:{path}")

    seed_all(SEED)
    stage_v_streams = load_clean_streams(args.stage_v_clean_root.resolve(), "stage_v")
    stage_vi_streams = load_clean_streams(args.stage_vi_clean_root.resolve(), "stage_vi")
    stage_v_labels = attach_windows(load_labels(args.stage_v_labels.resolve(), "STAGE_V"), stage_v_streams)
    stage_vi_labels = attach_windows(load_labels(args.stage_vi_labels.resolve(), "STAGE_VI_B2"), stage_vi_streams)
    all_labels = stage_v_labels + stage_vi_labels
    score_rows = score_b2c(all_labels, args.b2c_model.resolve())
    policy, policy_meta = load_policy_intent(args.policy_root.resolve())
    snapshots = {
        "stage_v": snapshot_inventory(args.stage_v_snapshot_root.resolve()),
        "stage_vi_b2": snapshot_inventory(args.stage_vi_snapshot_root.resolve()),
    }

    stage_v_features = {key: value["features"] for key, value in stage_v_streams.items()}
    stage_vi_features = {key: value["features"] for key, value in stage_vi_streams.items()}
    stage_v_score = [row for row in score_rows if row["stage"] == "STAGE_V"]
    stage_vi_score = [row for row in score_rows if row["stage"] == "STAGE_VI_B2"]
    context_rows = [row for row in all_labels if row["dose"] == "T5"]
    score_file_rows = []
    for row in score_rows:
        score_file_rows.append({
            key: row[key]
            for key in ("stage", "canonical_parent_key", "suite", "dose", "probe_id", "probe_step", "consumable", "abstain", "y", "label_class", "score")
        })

    summary = {
        "schema": "STAGE_VII_DOMAIN_SHIFT_FORENSIC_V1",
        "status": "PASS_STAGE_VII_DOMAIN_SHIFT_FORENSIC",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "STAGE_VII_CONTEXT_CONDITIONED_VULNERABILITY_DETECTOR",
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "candidate_training_performed": False,
        "diagnostic_probe_fits_performed": True,
        "formal_m4_executed": False,
        "teacher_student_modified": False,
        "threshold_retuned": False,
        "protected_counters": ZERO_COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "input_bindings": {
            "stage_v_clean": root_binding(args.stage_v_clean_root.resolve()),
            "stage_v_labels": root_binding(args.stage_v_labels.resolve()),
            "stage_v_snapshot": root_binding(args.stage_v_snapshot_root.resolve()),
            "stage_vi_clean": root_binding(args.stage_vi_clean_root.resolve()),
            "stage_vi_labels": root_binding(args.stage_vi_labels.resolve()),
            "stage_vi_snapshot": root_binding(args.stage_vi_snapshot_root.resolve()),
            "b2c_model": root_binding(args.b2c_model.resolve()),
            "policy_root": root_binding(args.policy_root.resolve()),
        },
        "population": {
            "stage_v_parent_count": len(stage_v_streams),
            "stage_v_label_count": len(stage_v_labels),
            "stage_vi_b2_parent_count": len(stage_vi_streams),
            "stage_vi_b2_label_count": len(stage_vi_labels),
            "stage_v_identity_overlap_stage_vi": sorted(set(stage_v_streams) & set(stage_vi_streams)),
            "t5_rows_scored": len(score_rows),
            "t5_consumable_rows_scored": sum(row["consumable"] for row in score_rows),
            "t5_abstain_rows_scored": sum(row["abstain"] for row in score_rows),
        },
        "b2_c_score_drift": {
            "threshold": THRESHOLD,
            "stage_v": {
                "overall": summarize_scores(stage_v_score),
                "per_suite": grouped_score_summary(stage_v_score, "suite"),
                "per_parent": grouped_score_summary(stage_v_score, "canonical_parent_key"),
            },
            "stage_vi_b2": {
                "overall": summarize_scores(stage_vi_score),
                "per_suite": grouped_score_summary(stage_vi_score, "suite"),
                "per_parent": grouped_score_summary(stage_vi_score, "canonical_parent_key"),
            },
        },
        "feature_distribution_shift": feature_shift(stage_v_features, stage_vi_features),
        "policy_intent_inventory": policy_meta,
        "snapshot_inventory": snapshots,
        "context_probes": context_probes(context_rows, policy),
        "interpretation_boundary": [
            "Stage V and Stage VI are development evidence for Stage VII, not Stage VII held-out evidence.",
            "Raw clean RGB and input_ids are not treated as frozen embeddings.",
            "Diagnostic probe metrics do not authorize candidate training, M4, timing, or protected evaluation.",
            "Abstains are masked and never converted to negatives.",
        ],
    }

    output.mkdir(parents=True)
    write_json(output / "STAGE_VII_DOMAIN_SHIFT_FORENSIC.json", summary)
    write_json(output / "INPUT_BINDINGS.json", summary["input_bindings"])
    write_json(output / "SNAPSHOT_INVENTORY.json", snapshots)
    write_json(output / "FEATURE_DISTRIBUTION_SHIFT.json", summary["feature_distribution_shift"])
    write_json(output / "CONTEXT_PROBES_DIAGNOSTIC.json", summary["context_probes"])
    with (output / "B2_C_T5_SCORE_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in score_file_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_seal(output, summary)
    print(json.dumps({
        "status": summary["status"],
        "output_root": str(output),
        "stage_v_parents": len(stage_v_streams),
        "stage_vi_b2_parents": len(stage_vi_streams),
        "t5_rows_scored": len(score_rows),
        "candidate_training_performed": False,
        "protected_counters": ZERO_COUNTERS,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
