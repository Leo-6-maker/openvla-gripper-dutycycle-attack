"""Freeze episode- and task-held-out FIT-only development splits.

This builder consumes only the already sealed T4/Teacher/feature graph.  It
publishes metadata and train-only normalization; it never copies labels or
features into the split root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import sha256_file, verify_seal
from audit_r3_generalization_g0 import (
    HEADS,
    _effective_step_value,
    _known,
    event_label,
)
from run_r3_full670_student_development import _load_records


FORBIDDEN_FIELDS = {
    "task_success", "terminal", "reward", "outcome", "attack_result",
    "future_frame", "future_label", "teacher_label",
}
FORBIDDEN_PATH_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _git_snapshot() -> tuple[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()
    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _is_ancestor(commit: str, current: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", commit, current), cwd=ROOT, check=False).returncode == 0


def _reject_symlink_components(raw: Path, label: str) -> None:
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute before component validation")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked {label} component: {current}")


def _validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != "V5_R3_GENERALIZATION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_G1_SPLIT":
        raise ValueError("G1 protocol is not frozen")
    scope = protocol.get("input_scope", {})
    expected_scope = {
        "teacher_identity_count": 670,
        "teacher_step_count": 196483,
        "teacher_root_status": "SEALED_FIT_ONLY",
        "g0_status_required": "PASS_LABEL_AND_BASELINE_AUDIT",
        "teacher_privileged_fields_in_student": False,
        "protected_reads": False,
        "CAL_READ": False,
        "CHECK_READ": False,
        "G10_READ": False,
        "T2R_D_READ": False,
        "attack": False,
        "rollout": False,
    }
    if set(scope) != set(expected_scope) or any(scope.get(key) != value for key, value in expected_scope.items()):
        raise ValueError("G1 protocol input scope is not an exact frozen contract")
    if protocol.get("heads", {}).get("active") != ["physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"] or protocol.get("heads", {}).get("held", {}).get("safe_release", {}).get("status") != "HOLD_COVERAGE":
        raise ValueError("G1 head eligibility is not frozen")
    features = protocol.get("features", {})
    if features.get("schema") != "SC5StreamingFeatureAdapterV2_25D" or features.get("input_dim") != 25 or features.get("future_frames") != 0 or features.get("teacher_fields_in_input") is not False or features.get("normalization") != "train_only_zscore":
        raise ValueError("G1 feature contract is not frozen")
    episode = protocol.get("episode_heldout", {})
    task = protocol.get("task_heldout", {})
    if episode.get("algorithm") != "within_task_sha256_rank_quota_v1" or episode.get("seed") != 20260717 or episode.get("split_once") is not True or episode.get("test_read_once") is not True or task.get("algorithm") != "global_task_sha256_rank_quota_v1" or task.get("seed") != 20260717 or task.get("split_once") is not True:
        raise ValueError("G1 split algorithm is not frozen")
    model = protocol.get("model_configs", {})
    if model.get("random_initialization_required") is not True or model.get("all_670_engineering_checkpoint_allowed") is not False or model.get("precision") != "FP32":
        raise ValueError("G1 model initialization boundary is not frozen")
    init_semantics = protocol.get("random_init_semantics", {})
    if init_semantics.get("scope") != "student_parameters_only" or init_semantics.get("heldout_episode_random_init_claim") is not False:
        raise ValueError("random-init semantics are ambiguous")
    threshold = protocol.get("threshold", {})
    if threshold.get("selection_split") != "validation_only" or threshold.get("test_threshold_selection") is not False:
        raise ValueError("G1 threshold ownership is not frozen")
    permissions = protocol.get("permissions", {})
    expected = {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": True, "development_inference": True, "privileged_oracle_diagnostic": True, "shadow_offline": False, "shadow_live": False, "formal_training": False, "full_fit": False, "rollout": False, "attack": False, "protected_reads": 0}
    if set(permissions) != set(expected) or any(permissions.get(key) != value for key, value in expected.items()):
        raise ValueError("G1 protocol permissions are not an exact closed matrix")


def _validate_g0_permissions(permissions: Mapping[str, Any]) -> None:
    expected = {"teacher_label_read": True, "student_training": False, "formal_training_authorized": False, "heldout_evaluation": False, "protected_reads": 0, "CAL_READ": False, "CHECK_READ": False, "G10_READ": False, "T2R_D_READ": False, "shadow": False, "rollout": False, "attack": False}
    if set(permissions) != set(expected) or any(permissions.get(key) != value for key, value in expected.items()):
        raise ValueError("G0 permission boundary is not an exact closed matrix")


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _rank_key(seed: int, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(), value


def _episode_split(identities: Sequence[Mapping[str, Any]], seed: int) -> dict[str, list[str]]:
    """Within-task hash quota: floor(70%), floor(15%), remainder."""
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for item in identities:
        groups[(str(item["suite"]), int(item["task_id"]))].append(str(item["episode_id"]))
    result = {"train": [], "validation": [], "test": []}
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda identity: _rank_key(seed, identity))
        if len(rows) < 3:
            raise ValueError(f"episode group too small for held-out split: {group}")
        n_train = max(1, int(len(rows) * 0.70))
        n_val = max(1, int(len(rows) * 0.15))
        if n_train + n_val >= len(rows):
            n_val = 1
            n_train = len(rows) - 2
        result["train"].extend(rows[:n_train])
        result["validation"].extend(rows[n_train:n_train + n_val])
        result["test"].extend(rows[n_train + n_val:])
    for key in result:
        result[key].sort()
    return result


def _task_split(tasks: Sequence[tuple[str, int]], seed: int) -> dict[str, list[tuple[str, int]]]:
    unique = sorted(set((str(suite), int(task)) for suite, task in tasks), key=lambda item: _rank_key(seed, f"{item[0]}|{item[1]}"))
    expected = {(suite, task) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for task in range(10)}
    if set(unique) != expected:
        raise ValueError(f"expected canonical 4-suite x 10-task grid, got {len(unique)} tasks")
    return {
        "train": unique[:30],
        "validation": unique[30:35],
        "test": unique[35:],
    }


def _reject_forbidden(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden field in split metadata: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _validate_output_path(raw: Path, allowed_parent: Path) -> Path:
    if not raw.is_absolute() or ".." in raw.parts or raw.exists() or raw.is_symlink():
        raise ValueError(f"output root must be a new absolute path: {raw}")
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in raw.parts):
        raise ValueError("output root is under a forbidden path")
    _reject_symlink_components(raw, "output root")
    if raw.parent.resolve(strict=False) != allowed_parent.resolve(strict=False):
        raise ValueError(f"output root is outside sealed phase parent: {raw}")
    return raw


def _validate_input_root(raw: Path, label: str) -> Path:
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be an absolute sealed root")
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in raw.parts):
        raise ValueError(f"{label} is under a forbidden path")
    _reject_symlink_components(raw, label)
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError(f"{label} is not a regular directory: {raw}")
    return raw.resolve(strict=True)


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _load_teacher_rows(teacher_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (teacher_root / "teacher_records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("episode_id"))
            step = row.get("step")
            if not identity or type(step) is not int or step < 0:
                raise ValueError(f"malformed Teacher step at line {line_number}")
            _reject_forbidden(row, f"teacher_records[{line_number}]")
            by_identity[identity].append(row)
    for identity, rows in by_identity.items():
        rows.sort(key=lambda row: row["step"])
        if [row["step"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous Teacher steps: {identity}")
    return by_identity


def _candidate_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        if row.get("candidate_close") is True:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(rows) - 1))
    return spans


def _event_keys(ids: Sequence[str], rows_by_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[tuple[str, int]]:
    return {
        (identity, event_index)
        for identity in ids
        for event_index, _ in enumerate(_candidate_spans(rows_by_id[identity]))
    }


def _event_stats(rows: Sequence[Mapping[str, Any]], head: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter()
    reason_counts = Counter()
    for row in rows:
        value = _effective_step_value(row["labels"][head])
        counts[value] += 1
        if value == "UNKNOWN":
            reason = str(row["labels"][head].get("reason", "")).strip()
            if not reason:
                raise ValueError(f"UNKNOWN {head} label has no reason: {row.get('episode_id')}/{row.get('step')}")
            reason_counts[reason] += 1
    positive_events = negative_events = unknown_events = right_censored_events = 0
    positive_episodes: set[str] = set()
    negative_episodes: set[str] = set()
    positive_tasks: set[tuple[str, int]] = set()
    negative_tasks: set[tuple[str, int]] = set()
    positive_suites: set[str] = set()
    negative_suites: set[str] = set()
    for start, end in _candidate_spans(rows):
        labels = [row["labels"][head] for row in rows[start:end + 1]]
        normalized = [item if _known(item) else {"value": "UNKNOWN"} for item in labels]
        value = event_label(normalized)
        censored = any(item.get("right_censored") is True for item in labels)
        right_censored_events += int(censored)
        if value == "TRUE":
            positive_events += 1
            positive_episodes.add(str(rows[0]["episode_id"]))
            positive_tasks.add((str(meta["suite"]), int(meta["task_id"])))
            positive_suites.add(str(meta["suite"]))
        elif value == "FALSE" and not censored:
            negative_events += 1
            negative_episodes.add(str(rows[0]["episode_id"]))
            negative_tasks.add((str(meta["suite"]), int(meta["task_id"])))
            negative_suites.add(str(meta["suite"]))
        else:
            unknown_events += 1
    return {
        "step_counts": {key: int(counts.get(key, 0)) for key in ("TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE")},
        "unknown_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_events": len(_candidate_spans(rows)),
        "known_positive_events": positive_events,
        "known_negative_events": negative_events,
        "unknown_events": unknown_events,
        "right_censored_events": right_censored_events,
        "_positive_episode_keys": positive_episodes,
        "_negative_episode_keys": negative_episodes,
        "_positive_task_keys": positive_tasks,
        "_negative_task_keys": negative_tasks,
        "_positive_suite_keys": positive_suites,
        "_negative_suite_keys": negative_suites,
    }


def _summarize_split(ids: Sequence[str], rows_by_id: Mapping[str, Sequence[Mapping[str, Any]]], metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"identity_count": len(ids), "step_count": 0, "heads": {head: None for head in HEADS}}
    for identity in ids:
        rows = rows_by_id[identity]
        result["step_count"] += len(rows)
    for head in HEADS:
        merged: dict[str, Any] | None = None
        for identity in ids:
            item = _event_stats(rows_by_id[identity], head, metadata[identity])
            if merged is None:
                merged = {key: value.copy() if isinstance(value, (dict, set)) else value for key, value in item.items()}
            else:
                for key in ("step_counts", "unknown_reason_counts"):
                    for name, count in item[key].items():
                        merged[key][name] = merged[key].get(name, 0) + count
                for key in ("candidate_events", "known_positive_events", "known_negative_events", "unknown_events", "right_censored_events"):
                    merged[key] += item[key]
                for key in ("_positive_episode_keys", "_negative_episode_keys", "_positive_task_keys", "_negative_task_keys", "_positive_suite_keys", "_negative_suite_keys"):
                    merged[key].update(item[key])
        for key, source in (
            ("positive_episodes", "_positive_episode_keys"),
            ("negative_episodes", "_negative_episode_keys"),
            ("positive_tasks", "_positive_task_keys"),
            ("negative_tasks", "_negative_task_keys"),
            ("positive_suites", "_positive_suite_keys"),
            ("negative_suites", "_negative_suite_keys"),
        ):
            merged[key] = len(merged.pop(source))
        result["heads"][head] = merged or _event_stats([], head, {"suite": "", "task_id": 0})
    return result


def _manifest_row(identity: str, item: Mapping[str, Any], binding: Mapping[str, Any], protocol_sha: str, g0_report_sha: str, g0_seal_sha: str) -> dict[str, Any]:
    if not isinstance(item.get("seed"), int) or isinstance(item.get("seed"), bool) or not isinstance(item.get("initial_state_sha256"), str) or len(item["initial_state_sha256"]) != 64:
        raise ValueError(f"missing sealed initial-state provenance: {identity}")
    row = {
        "episode_id": identity,
        "suite": item["suite"],
        "task_id": item["task_id"],
        "state_id": item["state_id"],
        "seed": item["seed"],
        "source_relative_path": item["relative_path"],
        "source_episode_sha256": item["episode_sha256"],
        "source_episode_seal_sha256sums_sha256": item["episode_sha256sums_sha256"],
        "initial_state_sha256": item.get("initial_state_sha256"),
        "initial_state_provenance": {
            "source": "T0-A sealed FIT episode binding",
            "seed": item["seed"],
            "initial_state_sha256": item["initial_state_sha256"],
        },
        "collection_source_commit": item.get("collection_source_commit"),
        "collection_source_tree": item.get("collection_source_tree"),
        "teacher_root_sha256sums_sha256": binding["teacher_root_sha256sums_sha256"],
        "t4_seal_sha256sums_sha256": binding["t4_seal_sha256sums_sha256"],
        "g0_report_sha256": g0_report_sha,
        "g0_root_sha256sums_sha256": g0_seal_sha,
        "protocol_sha256": protocol_sha,
        "feature_order_sha256": binding["feature_order_sha256"],
        "t0a_manifest_sha256": binding["t0a_manifest_sha256"],
        "t0a_root_sha256sums_sha256": binding["t0a_root_sha256sums_sha256"],
        "t0a_identity_set_digest": binding["t0a_identity_set_digest"],
    }
    _reject_forbidden(row)
    return row


def _train_normalization(ids: Sequence[str], records_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np
    if not ids:
        raise ValueError("empty normalization split")
    values = np.concatenate([np.asarray(records_by_id[identity]["features"], dtype=np.float64) for identity in ids], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    if values.shape[1] != 25 or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid train-only normalization")
    return {"schema": "V5_R3_TRAIN_ONLY_ZSCORE_V1", "source_split": "train", "identity_count": len(ids), "step_count": int(len(values)), "mean": mean.tolist(), "std": std.tolist()}


def run(t4_root: Path, g0_root: Path, output_root: Path, protocol_path: Path) -> dict[str, Any]:
    if protocol_path.is_absolute() or ".." in protocol_path.parts:
        raise ValueError("protocol must be a regular repo-relative file")
    protocol_path = ROOT / protocol_path
    _reject_symlink_components(protocol_path, "protocol")
    if protocol_path.is_symlink() or not protocol_path.is_file() or ROOT not in protocol_path.resolve().parents:
        raise ValueError("protocol must be a regular repo-relative file")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol_contract(protocol)
    commit, tree = _git_snapshot()
    protocol_parent = str(protocol.get("source_commit", ""))
    if len(protocol_parent) != 40 or not _is_ancestor(protocol_parent, commit):
        raise ValueError("protocol parent source commit is not in the consuming snapshot lineage")
    protocol_sha = sha256_file(protocol_path)

    t4_root = _validate_input_root(t4_root, "T4 root")
    g0_root = _validate_input_root(g0_root, "G0 root")
    records, binding = _load_records(t4_root)
    if len(records) != 670 or binding.get("step_count") != 196483:
        raise ValueError("T4 identity/step closure mismatch")
    transition_path = t4_root / "TEACHER_STUDENT_TRANSITION.json"
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    t0a_root = _validate_input_root(Path(str(transition["t0_a"]["root"])), "T0-A root")
    t0a_seal = verify_seal(t0a_root)
    t0a_manifest_path = t0a_root / "FORMAL_INPUT_MANIFEST.json"
    t0a_manifest_sha = sha256_file(t0a_manifest_path)
    t0a_manifest = json.loads(t0a_manifest_path.read_text(encoding="utf-8"))
    if t0a_seal["sha256sums_sha256"] != transition["t0_a"]["seal_sha256sums_sha256"] or t0a_manifest_sha != transition["t0_a"]["manifest_sha256"]:
        raise ValueError("T0-A seal binding mismatch")
    if t0a_manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or t0a_manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or t0a_manifest.get("episode_count") != 670 or t0a_manifest.get("protected_reads") != 0:
        raise ValueError("T0-A provenance is not consumable")
    if t0a_manifest.get("identity_set_digest") != binding["t0a_manifest"].get("identity_set_digest"):
        raise ValueError("T0-A identity-set binding mismatch")
    binding["t0a_manifest_sha256"] = t0a_manifest_sha
    binding["t0a_root_sha256sums_sha256"] = t0a_seal["sha256sums_sha256"]
    binding["t0a_identity_set_digest"] = t0a_manifest["identity_set_digest"]
    g0_seal = verify_seal(g0_root)
    g0_report_path = g0_root / "G0_LABEL_BASELINE_AUDIT.json"
    if not g0_report_path.is_file():
        raise ValueError("G0 report missing")
    g0_report = json.loads(g0_report_path.read_text(encoding="utf-8"))
    if g0_report.get("status") != "PASS_LABEL_AND_BASELINE_AUDIT" or g0_report.get("consumable") is not False or g0_report.get("identity_count") != 670 or g0_report.get("step_count") != 196483:
        raise ValueError("G0 report is not a passing diagnostic audit")
    g0_binding = g0_report.get("input_binding", {})
    expected_g0_binding = {
        "t4_seal_sha256sums_sha256": binding["t4_seal_sha256sums_sha256"],
        "teacher_root": binding["teacher_root"],
        "teacher_root_sha256sums_sha256": binding["teacher_root_sha256sums_sha256"],
        "teacher_manifest_sha256": binding["teacher_manifest_sha256"],
        "teacher_records_sha256": binding["teacher_records_sha256"],
        "coverage_root_sha256sums_sha256": binding["coverage_root_sha256sums_sha256"],
        "feature_binding_sha256": binding["feature_binding_sha256"],
        "feature_order_sha256": binding["feature_order_sha256"],
        "protected_reads": 0,
    }
    if any(g0_binding.get(key) != value for key, value in expected_g0_binding.items()):
        raise ValueError("G0/T4 nested binding mismatch")
    if any(g0_report.get("checks", {}).get(key) != expected for key, expected in {
        "identity_closure": True, "step_closure": True, "event_denominators_closed": True,
        "unknown_as_negative": False, "unexplained_unknown": 0, "protected_reads": 0,
        "forbidden_fields": 0, "feature_teacher_exact_join": True,
    }.items()):
        raise ValueError("G0 checks are not fully passing")
    _validate_g0_permissions(g0_report.get("permissions", {}))
    output_root = _validate_output_path(output_root, Path(binding["teacher_root"]).resolve().parent)

    meta_source = binding["t0a_manifest"]["episode_bindings"]
    metadata = {str(identity): dict(item) for identity, item in meta_source.items()}
    if set(metadata) != {row["identity"] for row in records}:
        raise ValueError("identity closure mismatch")
    records_by_id = {row["identity"]: row for row in records}
    rows_by_id = _load_teacher_rows(Path(binding["teacher_root"]))
    if set(rows_by_id) != set(metadata):
        raise ValueError("Teacher identity closure mismatch")
    episode_ids = [{"episode_id": identity, **{key: metadata[identity][key] for key in ("suite", "task_id", "state_id", "seed")}} for identity in sorted(metadata)]
    episode_split = _episode_split(episode_ids, int(protocol["episode_heldout"]["seed"]))
    task_split = _task_split([(item["suite"], item["task_id"]) for item in episode_ids], int(protocol["task_heldout"]["seed"]))
    all_ids = set(metadata)
    if set().union(*(set(values) for values in episode_split.values())) != all_ids or any(set(episode_split[a]) & set(episode_split[b]) for a in episode_split for b in episode_split if a < b):
        raise ValueError("episode split closure failed")
    task_sets = {key: set(value) for key, value in task_split.items()}
    if set().union(*task_sets.values()) != set((item["suite"], item["task_id"]) for item in episode_ids) or any(task_sets[a] & task_sets[b] for a in task_sets for b in task_sets if a < b):
        raise ValueError("task split closure failed")

    manifests = {}
    g0_report_sha = sha256_file(g0_report_path)
    for name, ids in episode_split.items():
        manifests[f"episode_{name}"] = [_manifest_row(identity, metadata[identity], binding, protocol_sha, g0_report_sha, g0_seal["sha256sums_sha256"]) for identity in ids]
    for name, tasks in task_split.items():
        ids = sorted(identity for identity, item in metadata.items() if (str(item["suite"]), int(item["task_id"])) in set(tasks))
        manifests[f"task_{name}"] = [_manifest_row(identity, metadata[identity], binding, protocol_sha, g0_report_sha, g0_seal["sha256sums_sha256"]) for identity in ids]

    event_distribution = {
        "schema": "V5_R3_EVENT_DISTRIBUTION_V1",
        "teacher_root_sha256sums_sha256": binding["teacher_root_sha256sums_sha256"],
        "episode_heldout": {name: _summarize_split(ids, rows_by_id, metadata) for name, ids in episode_split.items()},
        "task_heldout": {name: _summarize_split([identity for identity, item in metadata.items() if (str(item["suite"]), int(item["task_id"])) in set(tasks)], rows_by_id, metadata) for name, tasks in task_split.items()},
        "safe_release_status": "HOLD_COVERAGE",
    }
    for family in ("episode_heldout", "task_heldout"):
        for summary in event_distribution[family].values():
            for head in protocol["heads"]["active"]:
                if summary["heads"][head]["known_positive_events"] == 0 or summary["heads"][head]["known_negative_events"] == 0:
                    summary["heads"][head]["status"] = "HOLD_SPLIT_COVERAGE"
                else:
                    summary["heads"][head]["status"] = "COVERED"
            summary["heads"]["safe_release"]["status"] = "HOLD_COVERAGE"

    episode_event_keys = {key: _event_keys(value, rows_by_id) for key, value in episode_split.items()}
    event_intersections = {
        f"{a}__{b}": sorted(episode_event_keys[a] & episode_event_keys[b])
        for a in episode_event_keys for b in episode_event_keys if a < b
    }
    if any(event_intersections.values()):
        raise ValueError("event split intersection detected")
    identity_closure = {
        "schema": "V5_R3_IDENTITY_CLOSURE_V1",
        "total_identities": len(all_ids),
        "episode_split_counts": {key: len(value) for key, value in episode_split.items()},
        "task_split_counts": {key: len(value) for key, value in task_split.items()},
        "episode_intersections": {f"{a}__{b}": sorted(set(episode_split[a]) & set(episode_split[b])) for a in episode_split for b in episode_split if a < b},
        "event_intersections": event_intersections,
        "task_intersections": {f"{a}__{b}": sorted(task_sets[a] & task_sets[b]) for a in task_sets for b in task_sets if a < b},
        "duplicate_missing_extra": {"duplicate": 0, "missing": 0, "extra": 0},
        "episode_task_coverage": {key: sorted({f"{metadata[identity]['suite']}:{metadata[identity]['task_id']}" for identity in values}) for key, values in episode_split.items()},
        "protected_reads": 0,
    }
    normalization = {
        "episode_heldout": {name: _train_normalization(ids, records_by_id) for name, ids in {"train": episode_split["train"]}.items()},
        "task_heldout": {name: _train_normalization(ids, records_by_id) for name, ids in {"train": [identity for identity, item in metadata.items() if (str(item["suite"]), int(item["task_id"])) in task_sets["train"]]}.items()},
    }
    split_protocol = {
        "schema": "V5_R3_SPLIT_PROTOCOL_V1",
        "source_protocol_sha256": protocol_sha,
        "source_commit": commit,
        "source_tree": tree,
        "t0a_manifest_sha256": binding["t0a_manifest_sha256"],
        "t0a_root_sha256sums_sha256": binding["t0a_root_sha256sums_sha256"],
        "t0a_identity_set_digest": binding["t0a_identity_set_digest"],
        "episode_algorithm": protocol["episode_heldout"]["algorithm"],
        "task_algorithm": protocol["task_heldout"]["algorithm"],
        "seed": protocol["episode_heldout"]["seed"],
        "split_once": True,
        "test_read_once": True,
        "normalization_source": "train_only",
        "protected_reads": 0,
        "formal_training_authorized": False,
        "full_fit": False,
        "attack": False,
    }
    audit = {
        "schema": "V5_R3_G1_SPLIT_AUDIT_V1",
        "status": "PASS_SPLIT_CLOSURE_WITH_HEAD_COVERAGE_FLAGS",
        "consumable": False,
        "development_training_consumable": True,
        "formal_training_consumable": False,
        "input_binding": {
            "t4_root": str(Path(binding["t4_root"]).resolve()),
            "t4_seal_sha256sums_sha256": binding["t4_seal_sha256sums_sha256"],
            "teacher_root": binding["teacher_root"],
            "teacher_root_sha256sums_sha256": binding["teacher_root_sha256sums_sha256"],
            "g0_root": str(g0_root),
            "g0_root_sha256sums_sha256": g0_seal["sha256sums_sha256"],
            "g0_report_sha256": g0_report_sha,
            "protocol_sha256": protocol_sha,
            "feature_order_sha256": binding["feature_order_sha256"],
            "t0a_manifest_sha256": binding["t0a_manifest_sha256"],
            "t0a_root_sha256sums_sha256": binding["t0a_root_sha256sums_sha256"],
            "t0a_identity_set_digest": binding["t0a_identity_set_digest"],
        },
        "checks": {
            "identity_closure": True,
            "episode_intersections": 0,
            "task_intersections": 0,
            "event_intersections": sum(len(value) for value in event_intersections.values()),
            "normalization_train_only": True,
            "deterministic": True,
            "teacher_fields_in_manifests": False,
            "protected_reads": 0,
        },
        "heads": {
            head: {
                "episode_heldout": {name: event_distribution["episode_heldout"][name]["heads"][head]["status"] for name in ("train", "validation", "test")},
                "task_heldout": {name: event_distribution["task_heldout"][name]["heads"][head]["status"] for name in ("train", "validation", "test")},
            }
            for head in HEADS
        },
        "model_initialization": {"random_init_required": True, "all_670_checkpoint_allowed": False, "checkpoint_consumed": False},
        "initial_state_provenance": "every split row binds T0-A seed and initial_state_sha256",
        "random_init_semantics": "student parameter initialization only; no episode random-init claim",
        "builder_source": {"commit": commit, "tree": tree, "script_sha256": sha256_file(Path(__file__))},
        "permissions": {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": False, "development_inference": False, "protected_reads": 0, "CAL_READ": False, "CHECK_READ": False, "G10_READ": False, "T2R_D_READ": False, "rollout": False, "attack": False},
    }
    payload = {
        "EPISODE_TRAIN_MANIFEST.json": manifests["episode_train"],
        "EPISODE_VAL_MANIFEST.json": manifests["episode_validation"],
        "EPISODE_TEST_MANIFEST.json": manifests["episode_test"],
        "TASK_TRAIN_MANIFEST.json": manifests["task_train"],
        "TASK_VAL_MANIFEST.json": manifests["task_validation"],
        "TASK_TEST_MANIFEST.json": manifests["task_test"],
        "IDENTITY_CLOSURE.json": identity_closure,
        "EVENT_DISTRIBUTION.json": event_distribution,
        "NORMALIZATION.json": normalization,
        "SPLIT_PROTOCOL.json": split_protocol,
        "G1_SPLIT_AUDIT.json": audit,
    }
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for filename, value in payload.items():
            _reject_forbidden(value)
            (staging / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception as exc:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_G1_SPLIT_FAILURE_V1", "error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        raise
    audit["sha256sums_sha256"] = digest
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t4-root", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.t4_root, args.g0_root, args.output_root, args.protocol), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
