"""Report causal five-head coverage for a non-consumable R3 Teacher root."""
from __future__ import annotations

import argparse
import re
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def audit(input_root: Path, protocol_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    root = input_root.resolve()
    seal = verify_seal(root)
    manifest_path = root / "teacher_manifest.json"
    records_path = root / "teacher_records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DEVELOPMENT_NONCONSUMABLE" or manifest.get("protected_reads") != 0 or manifest.get("attack_authorized") is not False:
        raise ValueError("Teacher root is not non-consumable FIT-only output")
    for key in ("source_root", "identity_allowlist_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256"):
        if not manifest.get(key):
            raise ValueError(f"Teacher provenance field missing: {key}")
    if any(part.lower() in {"cal", "check", "g10", "t2r-d", "protected", "attack"} for part in Path(str(manifest["source_root"])).parts):
        raise ValueError("Teacher source root is protected-looking")
    if not all(re.fullmatch(r"[0-9a-f]{64}", str(manifest[key])) for key in ("identity_allowlist_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256")):
        raise ValueError("Teacher provenance SHA field invalid")
    source_manifest, source_episodes, source_seal = load_consumable_episodes(
        Path(str(manifest["source_root"])),
        expected_count=int(manifest["identity_count"]),
        transition_manifest_path=Path(str(manifest["transition_manifest_path"])),
    )
    if source_seal["sha256sums_sha256"] != manifest["input_sha256sums_sha256"]:
        raise ValueError("Teacher input seal does not match source review seal")
    for key in ("identity_allowlist_sha256", "transition_manifest_sha256", "transition_sha256sums_sha256"):
        if source_manifest.get(key) != manifest[key]:
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
        positive_episodes = set()
        teacher_true_intervals = []
        candidate_intervals = []
        for identity, rows in sorted(by_episode.items()):
            for row in rows:
                value = str(row["labels"][head]["value"])
                counts[value] = counts.get(value, 0) + 1
            teacher_true_intervals.extend((identity, *interval) for interval in _contiguous_intervals(rows, lambda row: row["labels"][head]["value"] == "TRUE"))
            candidate = _contiguous_intervals(rows, lambda row: bool(row.get("candidate_close")))
            candidate_intervals.extend((identity, *interval) for interval in candidate)
            for start, end in candidate:
                event_rows = [row for row in rows if start <= int(row["step"]) <= end]
                label = _event_label(event_rows, head)
                candidate_events[label] += 1
                if label == "TRUE":
                    positive_episodes.add(identity)
        known_steps = counts["TRUE"] + counts["FALSE"]
        head_report[head] = {
            "step_counts": counts,
            "known_step_count": known_steps,
            "positive_events": candidate_events["TRUE"],
            "negative_events": candidate_events["FALSE"],
            "unknown_events": candidate_events["UNKNOWN"],
            "positive_episode_count": len(positive_episodes),
            "task_count": len({(row.get("suite"), row.get("task_id")) for rows in by_episode.values() for row in rows}),
            "candidate_event_count": sum(candidate_events.values()),
            "right_censored_steps": sum(bool(row.get("right_censored")) for rows in records),
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
            "pass": values["positive_events"] >= int(minima["per_head_positive_events"]) and values["negative_events"] >= int(minima["per_head_negative_events"]) and values["positive_episode_count"] >= 5 and values["task_count"] >= 2,
            "required_positive_events": int(minima["per_head_positive_events"]),
            "required_negative_events": int(minima["per_head_negative_events"]),
        }
        | {key: values[key] for key in ("positive_events", "negative_events", "positive_episode_count", "task_count")}
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
        output = output_root.resolve()
        if output.exists():
            raise FileExistsError(output)
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
