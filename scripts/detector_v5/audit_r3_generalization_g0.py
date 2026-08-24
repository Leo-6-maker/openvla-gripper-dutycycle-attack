"""G0: sealed FIT670 Teacher label and baseline audit.

This is a read-only audit.  It reuses the T4 loader, which verifies the
episode, Teacher, transition, feature and permission bindings before any
statistics are computed.  It writes summaries only; no labels or features
are copied into the output root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import sha256_file, verify_seal
from run_r3_full670_student_development import ACTIVE_HEADS, HEADS, _load_records


TRUTH = {"TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"}
FORBIDDEN_FIELDS = {
    "task_success",
    "terminal",
    "reward",
    "outcome",
    "attack_result",
    "future_frame",
    "future_label",
}
FORBIDDEN_PATH_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _write_seal(root: Path) -> str:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _reject_forbidden(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden field {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def event_label(labels: list[Mapping[str, Any]]) -> str:
    """Frozen tri-valued OR used for candidate-event diagnostics."""
    values = [str(label.get("value")) for label in labels]
    if any(value == "TRUE" for value in values):
        return "TRUE"
    if values and all(value == "FALSE" for value in values):
        return "FALSE"
    return "UNKNOWN"


def _known(label: Mapping[str, Any]) -> bool:
    return (
        label.get("value") in {"TRUE", "FALSE"}
        and label.get("valid_mask") is True
        and label.get("mask") is True
        and label.get("right_censored") is False
    )


def _effective_step_value(label: Mapping[str, Any]) -> str:
    """Apply mask semantics before step and event aggregation."""
    if _known(label):
        return str(label["value"])
    if label.get("value") == "NOT_APPLICABLE" or label.get("reason") == "GEOMETRY_NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    return "UNKNOWN"


def _bool_close(row: Mapping[str, Any]) -> bool:
    value = row.get("candidate_close")
    if type(value) is not bool:
        raise ValueError("candidate_close must be bool")
    return value


def _identity_metadata(binding: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source = binding.get("t0a_manifest", {}).get("episode_bindings")
    if not isinstance(source, Mapping) or len(source) != 670:
        raise ValueError("T0-A identity binding is not exact 670")
    result = {}
    for identity, item in source.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"malformed identity binding: {identity}")
        result[str(identity)] = {
            "suite": item.get("suite"),
            "task_id": item.get("task_id"),
            "state_id": item.get("state_id"),
            "seed": item.get("seed"),
        }
    return result


def _empty_head() -> dict[str, Any]:
    return {
        "step_counts": Counter(),
        "known_steps": 0,
        "positive_steps": 0,
        "negative_steps": 0,
        "unknown_steps": 0,
        "not_applicable_steps": 0,
        "right_censored_steps": 0,
        "unknown_reason_histogram": Counter(),
        "candidate_event_count": 0,
        "candidate_known_events": 0,
        "candidate_positive_events": 0,
        "candidate_negative_events": 0,
        "candidate_unknown_events": 0,
        "right_censored_events": 0,
        "positive_episodes": set(),
        "negative_episodes": set(),
        "positive_tasks": set(),
        "negative_tasks": set(),
        "positive_suites": set(),
        "negative_suites": set(),
        "teacher_true_intervals": 0,
        "teacher_true_intervals_touched_by_candidate": 0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(key): int(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _validate_output_path(raw_output: Path, allowed_parent: Path) -> Path:
    """Reject unsafe paths before any resolve can hide a symlink or escape."""
    if not raw_output.is_absolute():
        raise ValueError(f"output root must be a new absolute regular path: {raw_output}")
    if raw_output.is_symlink():
        raise ValueError(f"output root is a symlink: {raw_output}")
    if raw_output.exists():
        raise FileExistsError(raw_output)
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in raw_output.parts):
        raise ValueError("output root is under a forbidden path")
    current = Path(raw_output.anchor)
    for part in raw_output.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked output component: {current}")
    resolved_parent = raw_output.parent.resolve(strict=False)
    if resolved_parent != allowed_parent.resolve(strict=False):
        raise ValueError(f"output root is outside the sealed phase parent: {raw_output}")
    return raw_output


def _span_indices(rows: list[Mapping[str, Any]], predicate) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        if predicate(row):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(rows) - 1))
    return spans


def _event_metrics(rows: list[Mapping[str, Any]], head: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    state = _empty_head()
    candidate_spans = _span_indices(rows, _bool_close)
    true_spans = _span_indices(rows, lambda row: _known(row["labels"][head]) and row["labels"][head].get("value") == "TRUE")
    state["candidate_event_count"] = len(candidate_spans)
    state["teacher_true_intervals"] = len(true_spans)
    touched_true_intervals: set[int] = set()
    for start, end in candidate_spans:
        labels = [row["labels"][head] for row in rows[start : end + 1]]
        label = event_label([item if _known(item) else {"value": "UNKNOWN"} for item in labels])
        right_censored = any(item.get("right_censored") is True for item in labels)
        state["right_censored_events"] += int(right_censored)
        if label == "TRUE":
            state["candidate_positive_events"] += 1
            identity = str(rows[0]["episode_id"])
            state["positive_episodes"].add(identity)
            state["positive_tasks"].add((meta["suite"], meta["task_id"]))
            state["positive_suites"].add(meta["suite"])
            # A known TRUE is an observed positive even when another step is
            # right-censored; the censoring flag remains separately reported.
            state["candidate_known_events"] += 1
        elif label == "FALSE" and not right_censored:
            state["candidate_negative_events"] += 1
            state["candidate_known_events"] += 1
            identity = str(rows[0]["episode_id"])
            state["negative_episodes"].add(identity)
            state["negative_tasks"].add((meta["suite"], meta["task_id"]))
            state["negative_suites"].add(meta["suite"])
        else:
            state["candidate_unknown_events"] += 1
        for true_index, (true_start, true_end) in enumerate(true_spans):
            if true_start <= end and start <= true_end:
                touched_true_intervals.add(true_index)
    state["teacher_true_intervals_touched_by_candidate"] = len(touched_true_intervals)
    return state


def _audit_rows(teacher_root: Path, metadata: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_without_reason = 0
    total = 0
    with (teacher_root / "teacher_records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _reject_forbidden(row, f"teacher_records[{line_number}]")
            identity = str(row.get("episode_id"))
            step = row.get("step")
            if identity not in metadata or type(step) is not int or step < 0:
                raise ValueError(f"malformed Teacher identity/step at line {line_number}")
            if row.get("candidate_close") not in {True, False}:
                raise ValueError(f"candidate_close is not bool at line {line_number}")
            labels = row.get("labels")
            if not isinstance(labels, Mapping) or set(labels) != set(HEADS):
                raise ValueError(f"head closure failure at line {line_number}")
            for head in HEADS:
                label = labels[head]
                if not isinstance(label, Mapping) or label.get("value") not in TRUTH:
                    raise ValueError(f"invalid {head} label at line {line_number}")
                for field in ("valid_mask", "mask", "right_censored"):
                    if type(label.get(field)) is not bool:
                        raise ValueError(f"invalid {head}.{field} at line {line_number}")
                if not _known(label):
                    reason = str(label.get("reason", "")).strip()
                    if not reason:
                        unknown_without_reason += 1
                if label.get("value") == "NOT_APPLICABLE" and label.get("valid_mask") is True:
                    raise ValueError(f"NOT_APPLICABLE cannot be valid at line {line_number}")
            by_identity[identity].append(row)
            total += 1
    if unknown_without_reason:
        raise ValueError(f"unexplained UNKNOWN labels: {unknown_without_reason}")
    if set(by_identity) != set(metadata):
        raise ValueError("Teacher identity closure mismatch")
    head_stats: dict[str, dict[str, Any]] = {}
    task_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for identity in sorted(by_identity):
        rows = sorted(by_identity[identity], key=lambda row: row["step"])
        if [row["step"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous Teacher steps: {identity}")
        meta = metadata[identity]
        for head in HEADS:
            state = _event_metrics(rows, head, meta)
            head_stats.setdefault(head, _empty_head())
            target = head_stats[head]
            for key in ("step_counts", "unknown_reason_histogram"):
                target[key].update(state[key])
            for key in (
                "known_steps",
                "positive_steps",
                "negative_steps",
                "unknown_steps",
                "not_applicable_steps",
                "right_censored_steps",
                "candidate_event_count",
                "candidate_known_events",
                "candidate_positive_events",
                "candidate_negative_events",
                "candidate_unknown_events",
                "right_censored_events",
                "teacher_true_intervals",
                "teacher_true_intervals_touched_by_candidate",
            ):
                target[key] += state[key]
            target["positive_episodes"].update(state["positive_episodes"])
            target["negative_episodes"].update(state["negative_episodes"])
            target["positive_tasks"].update(state["positive_tasks"])
            target["negative_tasks"].update(state["negative_tasks"])
            target["positive_suites"].update(state["positive_suites"])
            target["negative_suites"].update(state["negative_suites"])
            for row in rows:
                label = row["labels"][head]
                value = _effective_step_value(label)
                target["step_counts"][value] += 1
                if value == "TRUE":
                    target["positive_steps"] += 1
                    target["known_steps"] += 1
                elif value == "FALSE":
                    target["negative_steps"] += 1
                    target["known_steps"] += 1
                elif value == "NOT_APPLICABLE":
                    target["not_applicable_steps"] += 1
                else:
                    target["unknown_steps"] += 1
                    reason = str(label.get("reason", "")).strip()
                    target["unknown_reason_histogram"][reason] += 1
                target["right_censored_steps"] += int(label["right_censored"])
                task_key = (meta["suite"], int(meta["task_id"]))
                task_rows.setdefault(task_key, {"suite": task_key[0], "task_id": task_key[1], "steps": 0})
                task_rows[task_key]["steps"] += 1
                task_rows[task_key].setdefault(head, Counter())[value] += 1
        
    return head_stats, task_rows, total


def _finalize_head(state: Mapping[str, Any]) -> dict[str, Any]:
    known = int(state["known_steps"])
    positives = int(state["positive_steps"])
    negatives = int(state["negative_steps"])
    candidate_known_events = int(state["candidate_known_events"])
    candidate_positive_events = int(state["candidate_positive_events"])
    candidate_negative_events = int(state["candidate_negative_events"])
    majority = 1.0 if positives >= negatives else 0.0
    majority_accuracy = _safe_ratio(max(positives, negatives), known)
    constant_true = _safe_ratio(positives, known)
    constant_false = _safe_ratio(negatives, known)
    event_majority = _safe_ratio(max(candidate_positive_events, candidate_negative_events), candidate_known_events)
    return {
        "step_counts": _jsonable(state["step_counts"]),
        "known_steps": known,
        "positive_steps": positives,
        "negative_steps": negatives,
        "unknown_steps": int(state["unknown_steps"]),
        "not_applicable_steps": int(state["not_applicable_steps"]),
        "right_censored_steps": int(state["right_censored_steps"]),
        "unknown_reason_histogram": _jsonable(state["unknown_reason_histogram"]),
        "candidate_event_count": int(state["candidate_event_count"]),
        "candidate_known_events": candidate_known_events,
        "candidate_positive_events": candidate_positive_events,
        "candidate_negative_events": candidate_negative_events,
        "candidate_unknown_events": int(state["candidate_unknown_events"]),
        "right_censored_events": int(state["right_censored_events"]),
        "positive_episodes": len(state["positive_episodes"]),
        "negative_episodes": len(state["negative_episodes"]),
        "positive_tasks": len(state["positive_tasks"]),
        "negative_tasks": len(state["negative_tasks"]),
        "positive_suites": len(state["positive_suites"]),
        "negative_suites": len(state["negative_suites"]),
        "teacher_true_intervals": int(state["teacher_true_intervals"]),
        "teacher_true_intervals_touched_by_candidate": int(state["teacher_true_intervals_touched_by_candidate"]),
        "baselines": {
            "majority_step_accuracy": majority_accuracy,
            "constant_true_step_accuracy": constant_true,
            "constant_false_step_accuracy": constant_false,
            "majority_class": "TRUE" if majority else "FALSE",
            "event_majority_accuracy": event_majority,
        },
    }


def run(t4_root: Path, output_root: Path) -> dict[str, Any]:
    records, binding = _load_records(t4_root.resolve())
    metadata = _identity_metadata(binding)
    allowed_parent = Path(binding["teacher_root"]).resolve().parent
    output_root = _validate_output_path(output_root, allowed_parent)
    head_stats, task_rows, total = _audit_rows(Path(binding["teacher_root"]), metadata)
    if total != 196483 or len(records) != 670:
        raise ValueError(f"G0 closure mismatch: {len(records)} identities / {total} rows")
    report = {
        "schema": "V5_R3_G0_LABEL_BASELINE_AUDIT_V1",
        "status": "PASS_LABEL_AND_BASELINE_AUDIT",
        "consumable": False,
        "identity_count": len(records),
        "step_count": total,
        "heads": {head: _finalize_head(state) for head, state in sorted(head_stats.items())},
        "active_heads": list(ACTIVE_HEADS),
        "held_heads": {"safe_release": "HOLD_COVERAGE"},
        "input_binding": {
            "t4_root": str(t4_root.resolve()),
            "t4_seal_sha256sums_sha256": binding["t4_seal_sha256sums_sha256"],
            "teacher_root": binding["teacher_root"],
            "teacher_root_sha256sums_sha256": binding["teacher_root_sha256sums_sha256"],
            "teacher_manifest_sha256": binding["teacher_manifest_sha256"],
            "teacher_records_sha256": binding["teacher_records_sha256"],
            "coverage_root": binding["coverage_root"],
            "coverage_root_sha256sums_sha256": binding["coverage_root_sha256sums_sha256"],
            "feature_binding_sha256": binding["feature_binding_sha256"],
            "feature_order_sha256": binding["feature_order_sha256"],
            "protected_reads": 0,
        },
        "checks": {
            "identity_closure": True,
            "step_closure": True,
            "event_denominators_closed": True,
            "unknown_as_negative": False,
            "unexplained_unknown": 0,
            "protected_reads": 0,
            "forbidden_fields": 0,
            "feature_teacher_exact_join": True,
        },
        "permissions": {
            "teacher_label_read": True,
            "student_training": False,
            "formal_training_authorized": False,
            "heldout_evaluation": False,
            "protected_reads": 0,
            "CAL_READ": False,
            "CHECK_READ": False,
            "G10_READ": False,
            "T2R_D_READ": False,
            "shadow": False,
            "rollout": False,
            "attack": False,
        },
        "heldout_evaluation": False,
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
    }
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "G0_LABEL_BASELINE_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "G0_TASK_PREVALENCE.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["suite", "task_id", "steps"] + [f"{head}_{value}" for head in HEADS for value in ("TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE")]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(task_rows):
                item = task_rows[key]
                row = {field: 0 for field in fieldnames}
                row.update({"suite": item["suite"], "task_id": item["task_id"], "steps": item["steps"]})
                for head in HEADS:
                    for value, count in item.get(head, {}).items():
                        if value in {"TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"}:
                            row[f"{head}_{value}"] = count
                writer.writerow(row)
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception as exc:
        (staging / "FAILURE.json").write_text(
            json.dumps({"schema": "V5_R3_G0_FAILURE_V1", "error": repr(exc)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_seal(staging)
        raise
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t4-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.t4_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
