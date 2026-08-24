"""Report causal five-head coverage for a non-consumable R3 Teacher root."""
from __future__ import annotations

import argparse
import re
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_teacher import FORBIDDEN_FIELDS, HEADS
from audit_r3_contact_input import load_consumable_episodes, sha256_file, verify_seal


def _event_label(rows: list[dict[str, Any]], head: str) -> str:
    values = [str(row["labels"][head]["value"]) for row in rows]
    if "TRUE" in values:
        return "TRUE"
    if "UNKNOWN" in values or any(bool(row.get("right_censored")) for row in rows):
        return "UNKNOWN"
    return "FALSE" if values and all(value == "FALSE" for value in values) else "UNKNOWN"


def _contiguous_intervals(rows: list[dict[str, Any]], predicate) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    previous = None
    for row in rows:
        step = int(row["step"])
        if predicate(row):
            if start is None or previous is None or step != previous + 1:
                if start is not None:
                    intervals.append((start, previous))
                start = step
            previous = step
        elif start is not None:
            intervals.append((start, previous))
            start = None
            previous = None
    if start is not None:
        intervals.append((start, previous))
    return intervals


def _overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    return sum(1 for a, b in left for c, d in right if max(a, c) <= min(b, d))


def _forbidden(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key) in FORBIDDEN_FIELDS:
                found.append(child_path)
            found.extend(_forbidden(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


_FORBIDDEN_ROOT_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
_T0_A_GATE_KEYS = {
    "bad_episode_seal", "bad_worker_seal", "duplicate", "empty_entity_records", "extra",
    "identity_binding_error", "missing", "nonfinite", "protected_reads", "schema_error",
    "source_binding_error", "staging_residue", "unallowlisted",
}


def _reject_symlink_components(raw: Path, label: str) -> None:
    for component in (raw, *raw.parents):
        if component.is_symlink():
            raise ValueError(f"symlink in {label} path")


def _safe_path(value: Any, label: str, *, directory: bool) -> Path:
    raw = Path(str(value))
    _reject_symlink_components(raw, label)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe {label} path")
    resolved = raw.resolve()
    if any(part.lower() in _FORBIDDEN_ROOT_PARTS for part in resolved.parts):
        raise ValueError(f"forbidden {label} path")
    if directory and not resolved.is_dir():
        raise ValueError(f"{label} directory missing")
    if not directory and not resolved.is_file():
        raise ValueError(f"{label} file missing")
    return resolved


def _safe_new_path(value: Any, label: str) -> Path:
    raw = Path(str(value))
    _reject_symlink_components(raw, label)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe {label} path")
    resolved = raw.resolve()
    if any(part.lower() in _FORBIDDEN_ROOT_PARTS for part in resolved.parts):
        raise ValueError(f"forbidden {label} path")
    _safe_path(raw.parent, f"{label} parent", directory=True)
    if raw.exists():
        raise FileExistsError(resolved)
    return resolved


def _exact_ids(value: Any, expected_count: int, label: str) -> set[str]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{label} identity count mismatch")
    if any(isinstance(identity, Mapping) and not identity.get("episode_id") for identity in value):
        raise ValueError(f"{label} identity field missing")
    ids = [str(identity.get("episode_id")) if isinstance(identity, Mapping) else str(identity) for identity in value]
    if len(set(ids)) != expected_count:
        raise ValueError(f"{label} contains duplicate identities")
    return set(ids)


def _require_transition_audit_binding(transition: Mapping[str, Any], audit_binding: Mapping[str, Any]) -> None:
    if transition.get("input_audit_manifest_sha256") != audit_binding.get("manifest_sha256"):
        raise ValueError("FIT-to-Teacher/T0-A audit manifest binding mismatch")
    if transition.get("input_audit_seal_sha256sums_sha256") != audit_binding.get("seal_sha256sums_sha256"):
        raise ValueError("FIT-to-Teacher/T0-A audit seal binding mismatch")


def _load_formal_audited_source(manifest: Mapping[str, Any], expected_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load the sealed T0-A source census used by the full-formal label root.

    Full-formal Teacher manifests intentionally keep provenance under the
    nested ``input_binding`` contract.  The older canary path remains handled
    by ``load_consumable_episodes`` below.
    """
    if expected_count != 670:
        raise ValueError("full-formal source must contain exactly 670 identities")
    binding = manifest.get("input_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("full-formal Teacher input_binding is missing")
    audit_binding = binding.get("input_audit")
    transition_binding = binding.get("transition")
    fit_binding = binding.get("fit_to_teacher_transition")
    if not all(isinstance(item, Mapping) for item in (audit_binding, transition_binding, fit_binding)):
        raise ValueError("full-formal Teacher input binding is incomplete")
    audit_root = _safe_path(audit_binding.get("root"), "T0-A audit root", directory=True)
    audit_seal = verify_seal(audit_root)
    if audit_seal["sha256sums_sha256"] != audit_binding.get("seal_sha256sums_sha256"):
        raise ValueError("T0-A audit seal binding mismatch")
    audit_path = audit_root / "FORMAL_INPUT_MANIFEST.json"
    if sha256_file(audit_path) != audit_binding.get("manifest_sha256"):
        raise ValueError("T0-A audit manifest binding mismatch")
    audited = json.loads(audit_path.read_text(encoding="utf-8"))
    if audited.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or audited.get("episode_count") != expected_count:
        raise ValueError("T0-A formal input is not consumable")
    if audited.get("protected_reads") != 0 or audited.get("teacher_labels_generated") is not False or audited.get("student_started") is not False or audited.get("labels_generated") is not False or audited.get("attack_authorized") is not False or audited.get("source_staging_residue") != []:
        raise ValueError("T0-A authorization boundary is not closed")
    gate = audited.get("gate")
    if not isinstance(gate, Mapping) or set(gate) != _T0_A_GATE_KEYS or any(type(value) is not int or value != 0 for value in gate.values()):
        raise ValueError("T0-A gate is not zero-failure")
    formal_root = _safe_path(binding.get("formal_root"), "full-formal source root", directory=True)
    if formal_root != Path(str(audited.get("formal_root"))).resolve():
        raise ValueError("full-formal source root binding mismatch")
    episode_bindings = audited.get("episode_bindings")
    if not isinstance(episode_bindings, Mapping) or len(episode_bindings) != expected_count:
        raise ValueError("T0-A episode binding closure is incomplete")
    episode_ids = set(str(identity) for identity in episode_bindings)
    if _exact_ids(transition_binding.get("identities"), expected_count, "parent transition") != episode_ids:
        raise ValueError("full-formal transition/source identity closure mismatch")
    selection = binding.get("selection")
    if not isinstance(selection, Mapping) or selection.get("schema") != "V5_R3_FULL_FORMAL_SELECTION_FROM_T0_A_V1" or selection.get("status") != "PASS_FULL_FORMAL_T2_SELECTION" or selection.get("source") != "T0-A_FORMAL_INPUT_MANIFEST":
        raise ValueError("full-formal selection contract mismatch")
    selection_path = _safe_path(selection.get("manifest_path"), "full-formal selection manifest", directory=False)
    if selection_path != audit_path or sha256_file(selection_path) != selection.get("manifest_sha256") or selection.get("seal_sha256sums_sha256") != audit_seal["sha256sums_sha256"]:
        raise ValueError("full-formal selection manifest binding mismatch")
    if selection.get("identity_count") != expected_count or _exact_ids(selection.get("identities"), expected_count, "full-formal selection") != episode_ids:
        raise ValueError("full-formal selection/source identity closure mismatch")
    transition_path = _safe_path(fit_binding.get("manifest_path"), "FIT-to-Teacher transition", directory=False)
    transition_root = transition_path.parent
    transition_seal = verify_seal(transition_root)
    if transition_seal["sha256sums_sha256"] != fit_binding.get("seal_sha256sums_sha256"):
        raise ValueError("FIT-to-Teacher transition seal binding mismatch")
    if sha256_file(transition_path) != fit_binding.get("manifest_sha256"):
        raise ValueError("FIT-to-Teacher transition manifest binding mismatch")
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    _require_transition_audit_binding(transition, audit_binding)
    if transition.get("status") != "PASS_FIT_TO_TEACHER_AUTHORIZATION" or transition.get("protected_reads") != 0:
        raise ValueError("FIT-to-Teacher transition is not consumable")
    if any(transition.get(key) is not False for key in ("labels_generated", "student_training_authorized", "attack_authorized", "rollout_authorized", "training_authorized")):
        raise ValueError("FIT-to-Teacher transition grants a forbidden permission")
    permissions = transition.get("permissions")
    required_permissions = {
        "CAL_READ": False, "CHECK_READ": False, "G10_READ": False, "T2R_D_READ": False,
        "attack": False, "detector_load": False, "fit_episode_read": True,
        "protected_payload_read": False, "rollout": False, "shadow": False,
        "student_dataset_generation": False, "student_training": False,
        "teacher_label_generation": True,
    }
    if not isinstance(permissions, Mapping) or set(permissions) != set(required_permissions) or any(permissions.get(key) is not expected for key, expected in required_permissions.items()):
        raise ValueError("FIT-to-Teacher transition permission contract mismatch")
    parent_manifest = transition_binding.get("manifest")
    if not isinstance(parent_manifest, Mapping) or parent_manifest.get("protected_payload_read") is not False or parent_manifest.get("protected_overlap_verified") != 0:
        raise ValueError("parent transition protected boundary is not closed")
    if transition.get("protected_payload_read") is not None and transition.get("protected_payload_read") is not False:
        raise ValueError("FIT-to-Teacher protected payload boundary is not closed")
    if transition.get("protected_overlap_verified") is not None and transition.get("protected_overlap_verified") != 0:
        raise ValueError("FIT-to-Teacher protected overlap boundary is not closed")
    if transition.get("identity_count") != expected_count or transition.get("identity_set_digest") != audited.get("identity_set_digest"):
        raise ValueError("full-formal transition identity closure mismatch")
    if transition.get("identity_set_digest") != audited.get("identity_set_digest"):
        raise ValueError("full-formal transition/source identity mismatch")
    source_manifest = {
        "source_root": str(formal_root),
        "identity_allowlist_sha256": transition_binding.get("allowlist_sha256"),
        "transition_manifest_sha256": transition_binding.get("manifest_sha256"),
        "transition_sha256sums_sha256": transition_binding.get("seal_sha256sums_sha256"),
        "protected_reads": False,
    }
    source_episodes = []
    for identity, row in sorted(episode_bindings.items()):
        if not isinstance(row, Mapping) or row.get("episode_id") != identity:
            raise ValueError(f"T0-A episode binding identity mismatch: {identity}")
        step_count = row.get("worker_result_steps")
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
            raise ValueError(f"T0-A episode step binding is not a positive integer: {identity}")
        source_episodes.append({"manifest": {"episode_id": str(identity), "step_count": step_count}, "rows": []})
    if len(source_episodes) != expected_count:
        raise ValueError("full-formal source step binding is incomplete")
    return source_manifest, source_episodes, {"sha256sums_sha256": audit_seal["sha256sums_sha256"], "file_count": audit_seal["file_count"], "seal_kind": "T0-A_FORMAL_INPUT_MANIFEST"}


def audit(input_root: Path, protocol_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    root = _safe_path(input_root, "Teacher input root", directory=True)
    seal = verify_seal(root)
    manifest_path = root / "teacher_manifest.json"
    records_path = root / "teacher_records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DEVELOPMENT_NONCONSUMABLE" or manifest.get("protected_reads") != 0 or manifest.get("attack_authorized") is not False:
        raise ValueError("Teacher root is not non-consumable FIT-only output")
    formal_binding = manifest.get("input_binding")
    if isinstance(formal_binding, Mapping):
        source_root_value = formal_binding.get("formal_root")
        transition_binding = formal_binding.get("transition")
        if not isinstance(transition_binding, Mapping):
            raise ValueError("full-formal Teacher transition binding is missing")
        provenance_values = {
            "identity_allowlist_sha256": transition_binding.get("allowlist_sha256"),
            "transition_manifest_sha256": transition_binding.get("manifest_sha256"),
            "transition_sha256sums_sha256": transition_binding.get("seal_sha256sums_sha256"),
        }
    else:
        source_root_value = manifest.get("source_root")
        provenance_values = {key: manifest.get(key) for key in ("identity_allowlist_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256")}
    if not source_root_value:
        raise ValueError("Teacher provenance field missing: source_root")
    if any(not value for value in provenance_values.values()):
        raise ValueError("Teacher provenance field missing")
    if any(part.lower() in _FORBIDDEN_ROOT_PARTS for part in Path(str(source_root_value)).parts):
        raise ValueError("Teacher source root is protected-looking")
    if not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in provenance_values.values()):
        raise ValueError("Teacher provenance SHA field invalid")
    if isinstance(manifest.get("input_binding"), Mapping):
        source_manifest, source_episodes, source_seal = _load_formal_audited_source(manifest, int(manifest["identity_count"]))
        expected_source_seal = manifest["input_binding"]["input_audit"]["seal_sha256sums_sha256"]
        expected_source = provenance_values
    else:
        source_manifest, source_episodes, source_seal = load_consumable_episodes(
            Path(str(manifest["source_root"])),
            expected_count=int(manifest["identity_count"]),
            transition_manifest_path=Path(str(manifest["transition_manifest_path"])),
        )
        expected_source_seal = manifest["input_sha256sums_sha256"]
        expected_source = provenance_values
    if source_seal["sha256sums_sha256"] != expected_source_seal:
        raise ValueError("Teacher input seal does not match source review seal")
    for key in ("identity_allowlist_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256"):
        if source_manifest.get(key) != expected_source[key]:
            raise ValueError(f"Teacher source binding mismatch: {key}")
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if _forbidden(row):
            raise ValueError("forbidden Teacher/input field in label records")
        if set(row.get("labels", {})) != set(HEADS) or not isinstance(row.get("step"), int) or not isinstance(row.get("episode_id"), str):
            raise ValueError("malformed Teacher label row")
        for head in HEADS:
            label = row["labels"][head]
            if label.get("value") not in {"TRUE", "FALSE", "UNKNOWN"}:
                raise ValueError("invalid Teacher truth value")
            if bool(label.get("mask")) != (label["value"] != "UNKNOWN") or bool(label.get("valid_mask")) != bool(label.get("mask")):
                raise ValueError("UNKNOWN mask/value mismatch")
        by_episode[str(row["episode_id"])].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["step"]))
        if [row["step"] for row in rows] != list(range(len(rows))):
            raise ValueError("Teacher step closure or duplicate/missing record")
    source_ids = {str(item["manifest"]["episode_id"]) for item in source_episodes}
    if set(by_episode) != source_ids:
        raise ValueError("Teacher label/source identity closure mismatch")
    source_steps = {str(item["manifest"]["episode_id"]): int(item["manifest"]["step_count"]) for item in source_episodes}
    if any(len(by_episode[identity]) != source_steps[identity] for identity in source_ids):
        raise ValueError("Teacher label/source step-count closure mismatch")
    head_report: dict[str, Any] = {}
    for head in HEADS:
        counts = {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0, "NOT_APPLICABLE": 0}
        candidate_events = {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0}
        event_episodes = {label: set() for label in candidate_events}
        event_tasks = {label: set() for label in candidate_events}
        event_suites = {label: set() for label in candidate_events}
        unknown_reasons: defaultdict[str, int] = defaultdict(int)
        not_applicable_steps = 0
        right_censored_events = 0
        teacher_true_intervals = []
        candidate_intervals = []
        for identity, rows in sorted(by_episode.items()):
            for row in rows:
                label_data = row["labels"][head]
                value = str(label_data["value"])
                counts[value] = counts.get(value, 0) + 1
                if value == "UNKNOWN":
                    reason = label_data.get("reason")
                    if not isinstance(reason, str) or not reason:
                        raise ValueError(f"UNKNOWN label has no reason: {identity} step={row['step']} head={head}")
                    unknown_reasons[reason] += 1
                    if reason == "GEOMETRY_NOT_APPLICABLE":
                        not_applicable_steps += 1
            teacher_true_intervals.extend((identity, *interval) for interval in _contiguous_intervals(rows, lambda row: row["labels"][head]["value"] == "TRUE"))
            candidate = _contiguous_intervals(rows, lambda row: bool(row.get("candidate_close")))
            candidate_intervals.extend((identity, *interval) for interval in candidate)
            for start, end in candidate:
                event_rows = [row for row in rows if start <= int(row["step"]) <= end]
                label = _event_label(event_rows, head)
                candidate_events[label] += 1
                if label == "UNKNOWN" and any(bool(row.get("right_censored")) for row in event_rows):
                    right_censored_events += 1
                event_episodes[label].add(identity)
                event_tasks[label].add((str(event_rows[0].get("suite")), int(event_rows[0].get("task_id"))))
                event_suites[label].add(str(event_rows[0].get("suite")))
        known_steps = counts["TRUE"] + counts["FALSE"]
        head_report[head] = {
            "step_counts": counts,
            "known_step_count": known_steps,
            "positive_events": candidate_events["TRUE"],
            "negative_events": candidate_events["FALSE"],
            "unknown_events": candidate_events["UNKNOWN"],
            "positive_episode_count": len(event_episodes["TRUE"]),
            "negative_episode_count": len(event_episodes["FALSE"]),
            "unknown_episode_count": len(event_episodes["UNKNOWN"]),
            "positive_task_count": len(event_tasks["TRUE"]),
            "negative_task_count": len(event_tasks["FALSE"]),
            "unknown_task_count": len(event_tasks["UNKNOWN"]),
            "positive_suite_count": len(event_suites["TRUE"]),
            "negative_suite_count": len(event_suites["FALSE"]),
            "unknown_suite_count": len(event_suites["UNKNOWN"]),
            "task_count": len({(row.get("suite"), row.get("task_id")) for rows in by_episode.values() for row in rows}),
            "suite_count": len({str(rows[0].get("suite")) for rows in by_episode.values() if rows}),
            "candidate_event_count": sum(candidate_events.values()),
            "right_censored_steps": sum(bool(row.get("right_censored")) for row in records),
            "right_censored_event_count": right_censored_events,
            "unknown_non_censored_event_count": candidate_events["UNKNOWN"] - right_censored_events,
            "not_applicable_step_count": not_applicable_steps,
            "unknown_reason_histogram": dict(sorted(unknown_reasons.items())),
            "teacher_true_intervals": len(teacher_true_intervals),
            "candidate_intervals": len(candidate_intervals),
            "teacher_true_intervals_touched_by_candidate": sum(
                _overlap(
                    [(start, end)],
                    [(c_start, c_end) for c_identity, c_start, c_end in candidate_intervals if c_identity == identity],
                )
                for identity, start, end in teacher_true_intervals
            ),
        }
    minima = json.loads(protocol_path.read_text(encoding="utf-8"))["minimum_coverage_for_student"]
    coverage = {
        head: {
            "pass": values["positive_events"] >= int(minima["per_head_positive_events"])
            and values["negative_events"] >= int(minima["per_head_negative_events"])
            and values["positive_episode_count"] >= 5
            and values["negative_episode_count"] >= 5
            and values["positive_task_count"] >= 2
            and values["negative_task_count"] >= 2
            and values["positive_suite_count"] >= 2
            and values["negative_suite_count"] >= 2,
            "required_positive_events": int(minima["per_head_positive_events"]),
            "required_negative_events": int(minima["per_head_negative_events"]),
            "required_positive_episodes": 5,
            "required_negative_episodes": 5,
            "required_positive_tasks": 2,
            "required_negative_tasks": 2,
            "required_positive_suites": 2,
            "required_negative_suites": 2,
        }
        | {key: values[key] for key in ("positive_events", "negative_events", "positive_episode_count", "negative_episode_count", "positive_task_count", "negative_task_count", "positive_suite_count", "negative_suite_count")}
        for head, values in head_report.items()
    }
    path_parts = {part.lower() for part in root.parts}
    forbidden_root_parts = sorted(path_parts & {"cal", "check", "g10", "t2r-d", "protected", "attack"})
    if forbidden_root_parts:
        raise ValueError(f"forbidden label root: {forbidden_root_parts}")
    report = {
        "schema": "V5_R3_TEACHER_COVERAGE_AUDIT_V1",
        "status": "PASS_COVERAGE" if all(item["pass"] for item in coverage.values()) else "HOLD_COVERAGE",
        "input_root": str(root),
        "input_sha256sums_sha256": seal["sha256sums_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "identity_count": len(by_episode),
        "source_identity_count": len(source_ids),
        "step_count": len(records),
        "head_metrics": head_report,
        "coverage": coverage,
        "teacher_event_source": "full_causal_timeline_before_candidate_gating",
        "candidate_event_source": "candidate_close_contiguous_segments",
        "unknown_as_negative": False,
        "right_censored_as_negative": False,
        "protected_read_audit": {"status": "PASS", "forbidden_root_parts": forbidden_root_parts},
        "protected_reads": 0 if source_manifest.get("protected_reads") is False else 1,
        "attack_authorized": False,
        "formal_training_authorized": False,
    }
    if output_root is not None:
        output = _safe_new_path(output_root, "coverage output root")
        staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
        if staging.exists():
            raise FileExistsError(staging)
        staging.mkdir(parents=True)
        (staging / "coverage_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in files), encoding="utf-8")
        digest = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
        rename_noreplace(staging, output)
        report["output_root"] = str(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.input_root, args.protocol, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
