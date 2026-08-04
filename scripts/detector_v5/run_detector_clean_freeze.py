"""Clean-only Detector-v3 Stage 2 evaluator and final-freeze entry point.

The script consumes only the sealed Cache A and sealed D8-3B OOF artifacts.
It never opens Eval160/protected/attack roots.  Final training is reachable
only after the clean event Gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_d8_3b_run import audit_run, verify_sha256_seal
from audit_r3_contact_input import sha256_file, verify_seal
from d8_event_consolidator import consolidate_physical_events, compute_consolidation_digest
from d8_train_core import apply_normalization, compute_loss, compute_normalization, create_model
from run_d8_formal_g_sensitivity import load_sidecar_correct, load_teacher_labels


DEPLOYMENT_SEED = 20260805
CONFIG = "B3"
EPOCHS = 100
THRESHOLD = 0.0
LEARNING_RATE = 1e-3
WEIGHT_NORMALIZATION = "mean_to_one"
SEEDS = tuple(range(20260720, 20260730))
FOLDS = tuple(range(5))
FEATURE_SCHEMA_PATH = ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"
H1_SOURCE_COMMIT = "9dd324ad70a9be17548f72437da8454356abfd28"
H1_SOURCE_TREE = "0333510e291f8ec0c5b8738136019f30c5de17aa"
SOURCE_SNAPSHOT_SHA256 = "99648bdee45cde6411159f6d6586b8b7e46b626ea000f07a6cff0b38251efdbd"
LINEAGE_DIGEST = "d42b1fd9a2e511facb71faaedb84c575ff5fa649e16071685626547c96a61833"
SIDECAR_SEAL = "633ae69da69916b85ebff70d3c38994b00081c7b5cbb47ef80d3356526d13be0"
TEACHER_SEAL = "16e8934fa564809adad68ed27a9324e895bdd7d4f32659fe2682218aa4709866"
EVENT_G = 3
EXPECTED_RAW_EVENT_SPANS = 734
EXPECTED_CONSOLIDATED_EVENTS = 675
EXPECTED_BRIDGED_GAPS = 59
EVENT_GATE = {
    "event_recall_min": 0.70,
    "false_trigger_episode_rate_max": 0.10,
    "safe_release_false_positive_rate_max": 0.05,
    "median_first_hit_delay_max": 2.0,
}
THRESHOLD_CANDIDATES = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0)
PERSISTENCE_CANDIDATES = (1, 2, 3, 5)
HYSTERESIS_CANDIDATES = (0.25, 0.5)
COOLDOWN_CANDIDATES = (0, 5, 10)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def python_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count(),
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
    }


def feature_names() -> list[str]:
    schema = json.loads(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    names = [row["name"] for row in schema["features"]]
    if schema.get("dimensions") != 25 or len(names) != 25 or not schema.get("causal_only"):
        raise RuntimeError("25D causal feature schema is not frozen as expected")
    if schema.get("future_fields") != 0 or schema.get("teacher_label_fields") != 0:
        raise RuntimeError("feature schema contains forbidden future/teacher fields")
    return names


def load_cache(cache_root: Path, expected_seal: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    seal = verify_seal(cache_root.resolve())
    actual = str(seal["sha256sums_sha256"]).lower()
    if actual != expected_seal.lower():
        raise RuntimeError(f"Cache A seal mismatch: {actual} != {expected_seal}")
    manifest = json.loads((cache_root / "CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "DETECTOR_V3_D8_25D_CACHE_V3":
        raise RuntimeError("unexpected Cache A schema")
    if manifest.get("feature_dim") != 25 or manifest.get("feature_names") != feature_names():
        raise RuntimeError("Cache A feature binding mismatch")
    if manifest.get("eval160_reads") != 0 or manifest.get("protected_reads") != 0:
        raise RuntimeError("Cache A boundary counters are not zero")
    if manifest.get("code_snapshot", {}).get("executable_source_commit") != H1_SOURCE_COMMIT:
        raise RuntimeError("Cache A H1 source commit mismatch")
    if manifest.get("code_snapshot", {}).get("executable_source_tree") != H1_SOURCE_TREE:
        raise RuntimeError("Cache A H1 source tree mismatch")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for path in sorted((cache_root / "per_episode").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise RuntimeError(f"Cache A episode is not a list: {path.name}")
        for row in value:
            key = (str(row["episode_id"]), int(row["step"]))
            if key in identities:
                raise RuntimeError(f"Cache A duplicate identity: {key}")
            identities.add(key)
            if len(row.get("features_25d_raw", [])) != 25:
                raise RuntimeError(f"Cache A feature length mismatch: {key}")
            if not np.isfinite(np.asarray(row["features_25d_raw"], dtype=np.float64)).all():
                raise RuntimeError(f"Cache A non-finite feature: {key}")
            if row.get("effective_mask"):
                if float(row["physical_target"]) not in (0.0, 1.0):
                    raise RuntimeError(f"Cache A effective target is not binary: {key}")
                if not math.isfinite(float(row["D8_weight"])) or float(row["D8_weight"]) <= 0:
                    raise RuntimeError(f"Cache A effective weight is invalid: {key}")
            rows.append(row)
    if len(rows) != int(manifest.get("total_steps", -1)):
        raise RuntimeError("Cache A total row count mismatch")
    return rows, manifest, {**seal, "sha256sums_sha256": actual}


def load_clean_event_groups(
    sidecar_root: Path,
    teacher_root: Path,
    cache_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Bind clean Teacher events to the Cache A effective identities.

    The cache contains step labels and weights, but not the formal event
    grouping.  Reconstructing G=3 from the sealed clean inputs keeps UNKNOWN
    gaps abstained while preserving the D8-1 event identity.
    """
    sidecar_root = sidecar_root.resolve(strict=True)
    teacher_root = teacher_root.resolve(strict=True)
    sidecar_seal = verify_seal(sidecar_root)
    teacher_seal = verify_seal(teacher_root)
    if sidecar_seal["sha256sums_sha256"].lower() != SIDECAR_SEAL:
        raise RuntimeError("D8 sidecar seal mismatch")
    if teacher_seal["sha256sums_sha256"].lower() != TEACHER_SEAL:
        raise RuntimeError("D8 Teacher seal mismatch")

    sidecar = load_sidecar_correct(sidecar_root)
    labels_by_episode, teacher_steps, teacher_identities = load_teacher_labels(teacher_root)
    if teacher_identities != 670 or teacher_steps != 196483:
        raise RuntimeError("D8 clean Teacher closure mismatch")
    if set(sidecar) != set(labels_by_episode):
        raise RuntimeError("D8 clean sidecar/Teacher identity mismatch")
    if sum(len(value) for value in sidecar.values()) != teacher_steps:
        raise RuntimeError("D8 clean sidecar/Teacher step mismatch")

    effective = cache_effective_rows(cache_rows)
    effective_keys = {(str(row["episode_id"]), int(row["step"])) for row in effective}
    effective_ids = {key[0] for key in effective_keys}
    cache_positive_keys = {
        (str(row["episode_id"]), int(row["step"]))
        for row in effective
        if float(row["physical_target"]) == 1.0
    }
    event_groups: dict[str, list[dict[str, Any]]] = {}
    digest_rows: list[dict[str, Any]] = []
    grouped_positive_keys: set[tuple[str, int]] = set()
    raw_spans = consolidated_events = bridged_gaps = 0

    for episode_id in sorted(labels_by_episode):
        result = consolidate_physical_events(
            episode_id,
            labels_by_episode[episode_id],
            relations=sidecar[episode_id],
            G=EVENT_G,
        )
        if result.get("articulated"):
            if episode_id in effective_ids:
                raise RuntimeError(f"articulated identity leaked into Cache A: {episode_id}")
            continue
        groups: list[dict[str, Any]] = []
        for group in result.get("event_groups", []):
            fragments = [
                (int(start), int(end))
                for start, end in group["fragment_ranges"]
            ]
            for start, end in fragments:
                for step in range(start, end + 1):
                    label = labels_by_episode[episode_id].get(step, {})
                    if label.get("value") != "TRUE" or not label.get("mask") or not label.get("valid_mask"):
                        raise RuntimeError(f"G=3 event contains non-effective TRUE step: {episode_id} {step}")
                    key = (episode_id, step)
                    if key not in effective_keys:
                        raise RuntimeError(f"G=3 event step missing from Cache A: {key}")
                    grouped_positive_keys.add(key)
            groups.append(
                {
                    "consolidated_event_id": int(group["consolidated_event_id"]),
                    "fragment_ranges": fragments,
                    "fragment_count": int(group["fragment_count"]),
                }
            )
        event_groups[episode_id] = groups
        raw_spans += int(result.get("raw_true_span_count", 0))
        consolidated_events += int(result.get("consolidated_event_count", 0))
        bridged_gaps += int(result.get("total_bridged_gaps", 0))
        digest_rows.append(
            {
                "episode_id": episode_id,
                "consolidation_digest": compute_consolidation_digest(result),
                "event_count": len(groups),
            }
        )

    if effective_ids != set(event_groups):
        raise RuntimeError("Cache A effective identity closure does not match clean event binding")
    if cache_positive_keys != grouped_positive_keys:
        raise RuntimeError("Cache A positive labels do not match formal G=3 event fragments")
    if (raw_spans, consolidated_events, bridged_gaps) != (
        EXPECTED_RAW_EVENT_SPANS,
        EXPECTED_CONSOLIDATED_EVENTS,
        EXPECTED_BRIDGED_GAPS,
    ):
        raise RuntimeError(
            "D8 G=3 closure mismatch: "
            f"raw={raw_spans} events={consolidated_events} bridges={bridged_gaps}"
        )

    binding = {
        "schema": "D8_CLEAN_TEACHER_EVENT_BINDING_V1",
        "G": EVENT_G,
        "sidecar_root": str(sidecar_root),
        "sidecar_seal": sidecar_seal["sha256sums_sha256"],
        "teacher_root": str(teacher_root),
        "teacher_seal": teacher_seal["sha256sums_sha256"],
        "teacher_identities": teacher_identities,
        "teacher_steps": teacher_steps,
        "effective_identities": len(effective_ids),
        "raw_true_spans": raw_spans,
        "consolidated_events": consolidated_events,
        "bridged_gaps": bridged_gaps,
        "event_group_digest": sha256_json(digest_rows),
        "unknown_gap_steps_remain_abstained": True,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    return event_groups, binding


def load_oof(
    formal_root: Path,
    cache_rows: list[dict[str, Any]],
    expected_source_commit: str,
    expected_source_tree: str,
) -> tuple[dict[int, dict[tuple[str, int], float]], dict[str, Any]]:
    audit = audit_run(formal_root.resolve(), write_artifacts=False)
    if audit.get("verdict") != "PASS":
        raise RuntimeError(f"formal D8 independent audit is not PASS: {audit.get('verdict')}")
    formal_seal = verify_sha256_seal(formal_root.resolve())
    labels = {
        (str(row["episode_id"]), int(row["step"])): float(row["physical_target"])
        for row in cache_rows
        if row.get("effective_mask")
    }
    seed_scores: dict[int, dict[tuple[str, int], float]] = {seed: {} for seed in SEEDS}
    expected_files = 0
    pattern = re.compile(r"^seed(\d+)$")
    for path in sorted(formal_root.glob("seed*/B3_fold*/predictions.json")):
        seed_match = pattern.match(path.parent.parent.name)
        fold_match = re.match(r"^B3_fold(\d+)$", path.parent.name)
        if not seed_match or not fold_match:
            raise RuntimeError(f"unexpected OOF path: {path}")
        seed, fold = int(seed_match.group(1)), int(fold_match.group(1))
        if seed not in SEEDS or fold not in FOLDS:
            raise RuntimeError(f"OOF matrix identity outside D8-3B: {path}")
        expected_files += 1
        rows = json.loads(path.read_text(encoding="utf-8"))
        target_scores = seed_scores[seed]
        for row in rows:
            key = (str(row["episode_id"]), int(row["step"]))
            if key not in labels:
                raise RuntimeError(f"OOF identity is not in effective Cache A: {key}")
            if key in target_scores:
                raise RuntimeError(f"OOF duplicate within seed: {seed} {key}")
            logit = float(row["logit"])
            if not math.isfinite(logit) or float(row["target"]) != labels[key]:
                raise RuntimeError(f"OOF target/logit binding mismatch: {path.name} {key}")
            expected_pred = 1.0 if logit > THRESHOLD else 0.0
            if float(row["pred"]) != expected_pred:
                raise RuntimeError(f"OOF threshold binding mismatch: {path.name} {key}")
            target_scores[key] = logit
    if expected_files != len(SEEDS) * len(FOLDS):
        raise RuntimeError(f"OOF file closure mismatch: {expected_files}")
    expected_identity_count = len(labels)
    if any(len(values) != expected_identity_count for values in seed_scores.values()):
        raise RuntimeError("OOF seed identity closure mismatch")
    return seed_scores, {
        "formal_audit": audit,
        "formal_seal": formal_seal,
        "prediction_file_count": expected_files,
        "effective_identity_count": expected_identity_count,
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
    }


def cache_effective_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("effective_mask")]


def metric_summary(labels: list[float], logits: list[float]) -> dict[str, Any]:
    if len(labels) != len(logits) or not labels:
        raise RuntimeError("metric input is empty or misaligned")
    y = np.asarray(labels, dtype=np.int64)
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all() or set(y.tolist()) - {0, 1}:
        raise RuntimeError("metric input contains non-finite/non-binary values")
    pred = z > THRESHOLD
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y, z))
    except ValueError:
        auroc = None
    return {
        "n": len(y),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "auroc": auroc,
    }


def positive_spans(sequence: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    spans: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_step: int | None = None
    for row in sequence:
        positive = float(row["target"]) == 1.0
        contiguous = previous_step is not None and int(row["step"]) == previous_step + 1
        if positive and (not current or contiguous):
            current.append(row)
        else:
            if current:
                spans.append(current)
            current = [row] if positive else []
        previous_step = int(row["step"])
    if current:
        spans.append(current)
    return spans


def scheduler_emissions(
    sequence: list[dict[str, Any]],
    threshold: float,
    persistence: int,
    hysteresis: float,
    cooldown: int,
) -> list[dict[str, Any]]:
    emissions: list[dict[str, Any]] = []
    consecutive = 0
    previous_step: int | None = None
    latched = False
    next_allowed = -10**9
    for row in sequence:
        step = int(row["step"])
        above = float(row["score"]) > threshold
        if above and previous_step is not None and step == previous_step + 1:
            consecutive += 1
        else:
            consecutive = 1 if above else 0
        previous_step = step
        if latched and float(row["score"]) < threshold - hysteresis:
            latched = False
        if not latched and consecutive >= persistence and step >= next_allowed:
            emissions.append({"step": step, "target": float(row["target"])})
            latched = True
            next_allowed = step + cooldown
    return emissions


def event_metrics(
    rows: list[dict[str, Any]],
    threshold: float,
    persistence: int,
    hysteresis: float,
    cooldown: int,
    event_groups: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
    for sequence in by_episode.values():
        sequence.sort(key=lambda row: int(row["step"]))
    event_count = hit_count = 0
    false_trigger_episodes: set[str] = set()
    safe_release_episodes: set[str] = set()
    delays: list[int] = []
    persistence_hits = 0
    positive_event_episode_count = 0
    suite_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "events": 0, "hits": 0})
    for episode_id, sequence in sorted(by_episode.items()):
        suite = episode_id.split("/", 1)[0]
        suite_totals[suite]["episodes"] += 1
        events = event_groups.get(episode_id, [])
        suite_totals[suite]["events"] += len(events)
        event_count += len(events)
        if events:
            positive_event_episode_count += 1
        emissions = scheduler_emissions(sequence, threshold, persistence, hysteresis, cooldown)
        false = [row for row in emissions if row["target"] == 0.0]
        if false:
            false_trigger_episodes.add(episode_id)
        last_event_end = max(
            (int(fragment[1]) for event in events for fragment in event["fragment_ranges"]),
            default=-1,
        )
        if events and any(int(row["step"]) > last_event_end for row in false):
            safe_release_episodes.add(episode_id)
        for event in events:
            fragments = [(int(start), int(end)) for start, end in event["fragment_ranges"]]
            start = fragments[0][0]
            hits = [
                row
                for row in emissions
                if any(fragment_start <= int(row["step"]) <= fragment_end for fragment_start, fragment_end in fragments)
            ]
            if hits:
                hit_count += 1
                suite_totals[suite]["hits"] += 1
                delays.append(int(hits[0]["step"]) - start)
                if any(
                    any(fragment_start <= int(row["step"]) <= fragment_end for fragment_start, fragment_end in fragments)
                    for row in emissions
                    if row["target"] == 1.0
                ):
                    persistence_hits += 1
    return {
        "threshold": threshold,
        "persistence": persistence,
        "hysteresis": hysteresis,
        "cooldown": cooldown,
        "event_count": event_count,
        "detected_event_count": hit_count,
        "event_recall": hit_count / max(event_count, 1),
        "event_persistence_rate": persistence_hits / max(event_count, 1),
        "median_first_hit_delay": median(delays) if delays else None,
        "first_hit_delay_samples": len(delays),
        "false_trigger_episode_count": len(false_trigger_episodes),
        "false_trigger_episode_rate": len(false_trigger_episodes) / max(len(by_episode), 1),
        "safe_release_false_positive_episode_count": len(safe_release_episodes),
        "safe_release_false_positive_rate": len(safe_release_episodes) / max(positive_event_episode_count, 1),
        "episode_count": len(by_episode),
        "positive_event_episode_count": positive_event_episode_count,
        "suite_breakdown": {key: dict(value) for key, value in sorted(suite_totals.items())},
        "event_binding": "D8-1 formal G=3 consolidated Teacher event groups",
    }


def b1_logit(row: Mapping[str, Any]) -> float:
    features = row["features_25d_raw"]
    # ponytail: one explicit causal rule is enough for the baseline.
    return 1.0 if float(features[13]) >= 3.0 and float(features[19]) > 0.02 else -1.0


def b1_report(
    rows: list[dict[str, Any]],
    event_groups: Mapping[str, list[dict[str, Any]]],
    event_binding: Mapping[str, Any],
) -> dict[str, Any]:
    effective = cache_effective_rows(rows)
    fold_metrics: dict[str, Any] = {}
    for fold in FOLDS:
        selected = [row for row in effective if int(row["fold_id"]) == fold]
        fold_metrics[str(fold)] = metric_summary(
            [float(row["physical_target"]) for row in selected], [b1_logit(row) for row in selected]
        )
    all_metrics = metric_summary(
        [float(row["physical_target"]) for row in effective], [b1_logit(row) for row in effective]
    )
    seed_metrics = {str(seed): all_metrics for seed in SEEDS}
    b1_rows = [dict(row, target=float(row["physical_target"]), score=b1_logit(row)) for row in effective]
    return {
        "schema": "D8_B1_CAUSAL_HEURISTIC_REPORT_V1",
        "rule": "recent_close_streak >= 3 AND eef_z_delta_since_close > 0.02",
        "feature_indices": {"recent_close_streak": 13, "eef_z_delta_since_close": 19},
        "causal_only": True,
        "future_fields": 0,
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "overall_metrics": all_metrics,
        "event_binding": dict(event_binding),
        "event_metrics": event_metrics(b1_rows, 0.0, 1, 0.5, 0, event_groups),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def oof_report(
    rows: list[dict[str, Any]],
    seed_scores: dict[int, dict[tuple[str, int], float]],
    oof_meta: dict[str, Any],
    event_groups: Mapping[str, list[dict[str, Any]]],
    event_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = cache_effective_rows(rows)
    by_key = {(str(row["episode_id"]), int(row["step"])): row for row in effective}
    aggregate_rows: list[dict[str, Any]] = []
    for key, row in sorted(by_key.items()):
        values = [seed_scores[seed][key] for seed in SEEDS]
        aggregate_rows.append(dict(row, target=float(row["physical_target"]), score=float(np.mean(values))))
    seed_metrics = {}
    for seed in SEEDS:
        logits = [seed_scores[seed][key] for key in by_key]
        seed_metrics[str(seed)] = metric_summary(
            [float(row["physical_target"]) for row in effective], logits
        )
    fold_metrics = {}
    for seed in SEEDS:
        fold_metrics[str(seed)] = {}
        for fold in FOLDS:
            selected = [row for row in effective if int(row["fold_id"]) == fold]
            logits = [seed_scores[seed][(str(row["episode_id"]), int(row["step"]))] for row in selected]
            fold_metrics[str(seed)][str(fold)] = metric_summary(
                [float(row["physical_target"]) for row in selected], logits
            )
    search = []
    for threshold in THRESHOLD_CANDIDATES:
        for persistence in PERSISTENCE_CANDIDATES:
            for hysteresis in HYSTERESIS_CANDIDATES:
                for cooldown in COOLDOWN_CANDIDATES:
                    metrics = event_metrics(
                        aggregate_rows, threshold, persistence, hysteresis, cooldown, event_groups
                    )
                    gate = {
                        "event_recall": metrics["event_recall"] >= EVENT_GATE["event_recall_min"],
                        "false_trigger_episode_rate": metrics["false_trigger_episode_rate"] <= EVENT_GATE["false_trigger_episode_rate_max"],
                        "safe_release_false_positive_rate": metrics["safe_release_false_positive_rate"] <= EVENT_GATE["safe_release_false_positive_rate_max"],
                        "median_first_hit_delay": metrics["median_first_hit_delay"] is not None and metrics["median_first_hit_delay"] <= EVENT_GATE["median_first_hit_delay_max"],
                    }
                    search.append({**metrics, "gate": gate, "pass": all(gate.values())})
    passed = [row for row in search if row["pass"]]
    passed.sort(key=lambda row: (-row["event_recall"], row["median_first_hit_delay"], row["false_trigger_episode_rate"], row["persistence"], row["threshold"]))
    selected = passed[0] if passed else None
    unknown_rows = [row for row in rows if not row.get("effective_mask")]
    suites = sorted({str(row["episode_id"]).split("/", 1)[0] for row in effective})
    event_gate = {
        "status": "PASS" if selected else "FAIL",
        "selected": selected,
        "criteria": EVENT_GATE,
        "all_suites_have_effective_rows": bool(suites) and all(
            any(str(row["episode_id"]).startswith(f"{suite}/") for row in effective) for suite in suites
        ),
        "search_candidate_count": len(search),
        "passing_candidate_count": len(passed),
        "failure_reasons": [] if selected else [
            "no_clean_OOF_scheduler_candidate_satisfies_all_event_gate_criteria",
            "formal_G3_event_candidate_did_not_meet_all_scheduler_criteria",
        ],
    }
    report = {
        "schema": "D8_B3_CLEAN_OOF_EVENT_REPORT_V1",
        "formal_oof": oof_meta,
        "aggregation": "mean_logit_over_10_seeds_at_each_effective_episode_step",
        "event_definition": "D8-1 formal G=3 consolidated Teacher physical-criticality events",
        "event_binding": dict(event_binding),
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "aggregate_step_metrics": metric_summary(
            [float(row["target"]) for row in aggregate_rows], [float(row["score"]) for row in aggregate_rows]
        ),
        "event_metrics_at_training_threshold": event_metrics(
            aggregate_rows, 0.0, 1, 0.5, 0, event_groups
        ),
        "scheduler_search": search,
        "scheduler_gate": event_gate,
        "clean_task_suite_breakdown": event_metrics(
            aggregate_rows, 0.0, 1, 0.5, 0, event_groups
        )["suite_breakdown"],
        "unknown_abstain": {
            "raw_cache_rows": len(rows),
            "excluded_rows": len(unknown_rows),
            "excluded_rate": len(unknown_rows) / max(len(rows), 1),
            "effective_unknown_rows": 0,
        },
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    return report, event_gate


def seal_directory(root: Path) -> dict[str, Any]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    lines = "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    atomic_bytes(root / "SHA256SUMS", lines.encode("utf-8"))
    sums_sha = sha256_file(root / "SHA256SUMS")
    atomic_bytes(root / "SHA256SUMS.sha256", f"{sums_sha}  SHA256SUMS\n".encode("utf-8"))
    return {"sha256sums_sha256": sums_sha, "file_count": len(files)}


def train_final_detector(rows: list[dict[str, Any]], output_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    effective = cache_effective_rows(rows)
    x_cpu = torch.tensor([row["features_25d_raw"] for row in effective], dtype=torch.float32)
    y_cpu = torch.tensor([float(row["physical_target"]) for row in effective], dtype=torch.float32)
    w_cpu = torch.tensor([float(row["D8_weight"]) for row in effective], dtype=torch.float32)
    w_cpu = w_cpu / w_cpu.mean()
    source_identity_digest = sha256_json(sorted(f"{row['episode_id']}::{row['step']}" for row in effective))
    norm = compute_normalization(x_cpu, source_identity_digest=source_identity_digest)
    norm["fit_on"] = "full_clean_cache_A"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(DEPLOYMENT_SEED).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x, y, w = x_cpu.to(device), y_cpu.to(device), w_cpu.to(device)
    losses: list[float] = []
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(apply_normalization(x, norm)), y, w)
        if not torch.isfinite(loss):
            raise RuntimeError("final detector loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        logits = model(apply_normalization(x, norm)).detach().cpu().numpy().astype(float).tolist()
    fit_metrics = metric_summary([float(value) for value in y_cpu.tolist()], logits)
    checkpoint = {
        "schema": "D8_3B_CHECKPOINT_V2",
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "normalization": norm,
        "config": CONFIG,
        "seed": DEPLOYMENT_SEED,
        "fold": "FULL_CLEAN_CACHE_A",
        "epochs": EPOCHS,
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_normalization": WEIGHT_NORMALIZATION,
        **provenance,
        "finished_utc": utc_now(),
    }
    checkpoint_path = output_root / "FINAL_DETECTOR_CHECKPOINT.pt"
    fd, name = tempfile.mkstemp(prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=str(output_root))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checkpoint_sha = sha256_file(checkpoint_path)
    metrics = {
        "schema": "D8_FINAL_DETECTOR_METRICS_V1",
        "fit_scope": "full_clean_cache_A",
        "deployment_seed": DEPLOYMENT_SEED,
        "config": CONFIG,
        "epochs": EPOCHS,
        "architecture": "25->32->16->1",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "normalization": norm,
        "effective_sample_count": len(effective),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "full_clean_fit_metrics": fit_metrics,
        "checkpoint_sha256": checkpoint_sha,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    atomic_json(output_root / "FINAL_DETECTOR_METRICS.json", metrics)
    atomic_json(
        output_root / "FINAL_DETECTOR_RECEIPT.json",
        {
            "schema": "D8_FINAL_DETECTOR_RECEIPT_V1",
            "status": "TRAINED_CLEAN_ONLY",
            "checkpoint_sha256": checkpoint_sha,
            "provenance": provenance,
            "metrics": "FINAL_DETECTOR_METRICS.json",
            "deployment_to_production": False,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
        },
    )
    return metrics


def clean_replay_report(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    scheduler: Mapping[str, Any],
    event_groups: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = create_model(DEPLOYMENT_SEED)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    effective = cache_effective_rows(rows)
    x = torch.tensor([row["features_25d_raw"] for row in effective], dtype=torch.float32)
    with torch.no_grad():
        first = model(apply_normalization(x, checkpoint["normalization"])).numpy()
        second = model(apply_normalization(x, checkpoint["normalization"])).numpy()
    deterministic_scores = bool(np.array_equal(first, second))
    replay_rows = [
        dict(row, target=float(row["physical_target"]), score=float(score))
        for row, score in zip(effective, first.tolist())
    ]
    scheduler_args = {
        "threshold": float(scheduler["threshold"]),
        "persistence": int(scheduler["persistence"]),
        "hysteresis": float(scheduler["hysteresis"]),
        "cooldown": int(scheduler["cooldown"]),
    }
    traces: dict[str, list[dict[str, Any]]] = {}
    for episode_id in sorted({str(row["episode_id"]) for row in replay_rows}):
        sequence = sorted(
            (row for row in replay_rows if str(row["episode_id"]) == episode_id),
            key=lambda row: int(row["step"]),
        )
        traces[episode_id] = scheduler_emissions(sequence, **scheduler_args)
    trace_digest = sha256_json(traces)
    traces_again = {}
    for episode_id in sorted(traces):
        sequence = sorted(
            (row for row in replay_rows if str(row["episode_id"]) == episode_id),
            key=lambda row: int(row["step"]),
        )
        traces_again[episode_id] = scheduler_emissions(sequence, **scheduler_args)
    replay_event_metrics = event_metrics(replay_rows, event_groups=event_groups, **scheduler_args)
    # Cache A deliberately has no task-success field, so success regression is
    # not silently replaced with Teacher physical_target accuracy.
    replay_gate = {
        "checkpoint_restore": True,
        "deterministic_scores": deterministic_scores,
        "deterministic_scheduler": trace_digest == sha256_json(traces_again),
        "clean_false_intervention_rate": replay_event_metrics["false_trigger_episode_rate"] <= 0.10,
        "clean_task_success_regression": False,
    }
    return {
        "schema": "D8_CLEAN_REPLAY_REPORT_V1",
        "mode": ["shadow", "offline_guard_counterfactual"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scheduler": dict(scheduler),
        "replay_event_metrics": replay_event_metrics,
        "score_count": len(first),
        "score_trace_sha256": hashlib.sha256(np.asarray(first, dtype=np.float32).tobytes()).hexdigest(),
        "scheduler_trace_sha256": trace_digest,
        "task_success": {
            "available": False,
            "reason": "Cache A contains no clean task success field; no success regression claim is made.",
        },
        "gate": replay_gate,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def self_test() -> None:
    rows = [
        {"episode_id": "s/task/state", "step": i, "target": float(i in {2, 3, 4}), "score": float(i in {2, 3, 4})}
        for i in range(7)
    ]
    assert [len(span) for span in positive_spans(rows)] == [3]
    assert [row["step"] for row in scheduler_emissions(rows, 0.0, 2, 0.5, 10)] == [3]
    event_groups = {"s/task/state": [{"fragment_ranges": [(2, 4)]}]}
    result = event_metrics(rows, 0.0, 2, 0.5, 10, event_groups)
    assert result["event_recall"] == 1.0
    assert result["median_first_hit_delay"] == 1
    bridged_rows = [
        {"episode_id": "s/task/state", "step": i, "target": float(i in {2, 4}), "score": 1.0}
        for i in (2, 4)
    ]
    bridged = event_metrics(
        bridged_rows,
        0.0,
        1,
        0.5,
        0,
        {"s/task/state": [{"fragment_ranges": [(2, 2), (4, 4)]}]},
    )
    assert bridged["event_count"] == 1
    assert bridged["event_recall"] == 1.0
    print("SELF_TEST_PASS")


def run(args: argparse.Namespace) -> int:
    if git_value("status", "--porcelain"):
        raise RuntimeError("clean worktree required")
    actual_commit = git_value("rev-parse", "HEAD")
    actual_tree = git_value("rev-parse", "HEAD^{tree}")
    if actual_commit != args.expected_source_commit or actual_tree != args.expected_source_tree:
        raise RuntimeError("source commit/tree binding mismatch")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(str(staging))
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        cache_rows, cache_manifest, cache_seal = load_cache(args.cache_root.resolve(), args.expected_cache_seal)
        event_groups, event_binding = load_clean_event_groups(
            args.sidecar_root,
            args.teacher_root,
            cache_rows,
        )
        seed_scores, oof_meta = load_oof(
            args.formal_root.resolve(), cache_rows, args.expected_source_commit, args.expected_source_tree
        )
        b1 = b1_report(cache_rows, event_groups, event_binding)
        b3, gate = oof_report(cache_rows, seed_scores, oof_meta, event_groups, event_binding)
        common_provenance = {
            "source_commit": actual_commit,
            "source_tree": actual_tree,
            "h1_source_commit": H1_SOURCE_COMMIT,
            "h1_source_tree": H1_SOURCE_TREE,
            "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
            "cache_root": str(args.cache_root.resolve()),
            "cache_seal": cache_seal["sha256sums_sha256"],
            "event_binding": event_binding,
            "lineage_digest": LINEAGE_DIGEST,
            "clean_script_sha256": sha256_file(Path(__file__).resolve()),
            "python_environment": python_environment(),
            "started_utc": started,
        }
        atomic_json(staging / "B1_CLEAN_HEURISTIC_REPORT.json", b1)
        atomic_json(staging / "B3_CLEAN_OOF_EVENT_REPORT.json", b3)
        atomic_json(
            staging / "DETECTOR_EVENT_GATE_RESULT.json",
            {
                "schema": "D8_DETECTOR_EVENT_GATE_RESULT_V1",
                "stage": "Stage 2 clean event Gate",
                "verdict": "PASS" if gate["status"] == "PASS" else "SCIENTIFIC_FAIL",
                "started_utc": started,
                "finished_utc": utc_now(),
                "provenance": common_provenance,
                "cache_manifest_schema": cache_manifest.get("schema"),
                "formal_root": str(args.formal_root.resolve()),
                "formal_audit_verdict": oof_meta["formal_audit"].get("verdict"),
                "b1_report": "B1_CLEAN_HEURISTIC_REPORT.json",
                "b3_report": "B3_CLEAN_OOF_EVENT_REPORT.json",
                "scheduler_freeze_created": False,
                "final_detector_created": False,
                "clean_replay_created": False,
                "detector_frozen": False,
                "stage3_authorized": False,
                "stage4_authorized": False,
                "event_gate": gate,
                "failure_boundary": (
                    "Event Gate passed; final detector and clean replay remain pending."
                    if gate["status"] == "PASS"
                    else "Detector clean/event scientific Gate FAIL; no final checkpoint or attack rollout is authorized."
                ),
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "attack_rollouts": 0,
            },
        )
        if gate["status"] != "PASS":
            atomic_json(
                staging / "STAGE2_STOP_REASON.json",
                {
                    "schema": "D8_STAGE2_STOP_REASON_V1",
                    "reason_type": "SCIENTIFIC_GATE_FAIL",
                    "reason": "No clean OOF scheduler candidate met all required event criteria.",
                    "next_stage": "HOLD",
                    "final_detector_training": False,
                    "attack_rollouts": 0,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                },
            )
        else:
            atomic_json(
                staging / "DETECTOR_SCHEDULER_FREEZE.json",
                {
                    "schema": "D8_DETECTOR_SCHEDULER_FREEZE_V1",
                    "status": "PASS_CLEAN_OOF_FROZEN",
                    "selected": gate["selected"],
                    "semantics": {
                        "score": "raw_ensemble_mean_logit",
                        "trigger": "persistence consecutive scores above threshold",
                        "clear": "score below threshold minus hysteresis",
                        "unknown_abstain": "excluded rows never emit",
                    },
                    "provenance": common_provenance,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "attack_rollouts": 0,
                },
            )
            final_metrics = train_final_detector(cache_rows, staging, common_provenance)
            replay = clean_replay_report(
                cache_rows,
                staging / "FINAL_DETECTOR_CHECKPOINT.pt",
                gate["selected"],
                event_groups,
            )
            atomic_json(staging / "CLEAN_REPLAY.json", replay)
            replay_pass = all(bool(value) for value in replay["gate"].values())
            atomic_json(
                staging / "DETECTOR_FREEZE_RECEIPT.json",
                {
                    "schema": "D8_DETECTOR_FREEZE_RECEIPT_V1",
                    "status": "PASS_CLEAN_ONLY_OFFLINE" if replay_pass else "SCIENTIFIC_FAIL_CLEAN_REPLAY",
                    "checkpoint_sha256": final_metrics["checkpoint_sha256"],
                    "deployment_to_production": False,
                    "clean_replay": "CLEAN_REPLAY.json",
                    "replay_gate": replay["gate"],
                    "provenance": common_provenance,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "attack_rollouts": 0,
                },
            )
            if not replay_pass:
                atomic_json(
                    staging / "STAGE2_STOP_REASON.json",
                    {
                        "schema": "D8_STAGE2_STOP_REASON_V1",
                        "reason_type": "SCIENTIFIC_GATE_FAIL",
                        "reason": "Clean replay cannot establish the required task-success regression bound from Cache A.",
                        "next_stage": "HOLD",
                        "final_detector_training": True,
                        "detector_frozen": False,
                        "attack_rollouts": 0,
                        "eval160_reads": 0,
                        "protected_eval_reads": 0,
                    },
                )
            atomic_json(
                staging / "DETECTOR_STAGE2_RESULT.json",
                {
                    "schema": "D8_DETECTOR_STAGE2_RESULT_V1",
                    "stage": "Stage 2 Detector build",
                    "verdict": "PASS" if replay_pass else "SCIENTIFIC_FAIL",
                    "provenance": common_provenance,
                    "event_gate": gate,
                    "replay_gate": replay["gate"],
                    "scheduler_freeze_created": True,
                    "final_detector_created": True,
                    "clean_replay_created": True,
                    "detector_frozen": replay_pass,
                    "stage3_authorized": replay_pass,
                    "stage4_authorized": False,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "attack_rollouts": 0,
                },
            )
        if not (staging / "DETECTOR_STAGE2_RESULT.json").is_file():
            atomic_json(
                staging / "DETECTOR_STAGE2_RESULT.json",
                {
                    "schema": "D8_DETECTOR_STAGE2_RESULT_V1",
                    "stage": "Stage 2 Detector build",
                    "verdict": "SCIENTIFIC_FAIL",
                    "provenance": common_provenance,
                    "event_gate": gate,
                    "scheduler_freeze_created": False,
                    "final_detector_created": False,
                    "clean_replay_created": False,
                    "detector_frozen": False,
                    "stage3_authorized": False,
                    "stage4_authorized": False,
                    "stop_reason": "Detector clean/event scientific Gate FAIL.",
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "attack_rollouts": 0,
                },
            )
        seal = seal_directory(staging)
        os.rename(staging, output_root)
        final_verdict = gate["status"]
        result_path = output_root / "DETECTOR_STAGE2_RESULT.json"
        if result_path.exists():
            final_verdict = json.loads(result_path.read_text(encoding="utf-8")).get("verdict", final_verdict)
        print(json.dumps({"output_root": str(output_root), "verdict": final_verdict, "seal": seal}, sort_keys=True))
        return 0 if final_verdict == "PASS" else 20
    except Exception:
        for path in staging.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        staging.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--formal-root", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--expected-cache-seal")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (
        args.cache_root,
        args.formal_root,
        args.sidecar_root,
        args.teacher_root,
        args.expected_cache_seal,
        args.expected_source_commit,
        args.expected_source_tree,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("all execution arguments are required unless --self-test is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
