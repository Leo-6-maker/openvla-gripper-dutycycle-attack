#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

GATE = "D0B_CLEAN2000_LIBERO10_SEGMENT_RESOLUTION_DEBUG"
PASS_DEBUG_QUEUE = "PASS_LIBERO10_SEGMENT_DEBUG_QUEUE_BUILT"
PASS_ALREADY_SEGMENTED = "PASS_LIBERO10_SEGMENT_FIELDS_ALREADY_PRESENT"
OUT_FILES = [
    "clean2000_libero10_segment_debug_report.json",
    "clean2000_segment_skeleton.csv",
    "clean2000_libero10_resolution_queue.csv",
    "clean2000_artifact_availability.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
LIBERO10_SEGMENT_FIELDS = [
    "segment_id",
    "segment_index",
    "segment_start_step",
    "segment_end_step",
    "segment_role",
    "subtask_index",
    "event_id",
    "event_role",
]
ROOT_HINT_FIELDS = [
    "episode_root",
    "output_root",
    "run_root",
    "artifact_root",
    "episode_dir",
    "run_dir",
    "path",
    "directory",
    "root",
]
SENTINELS = [
    "step_records.jsonl",
    "step_telemetry.csv",
    "detector_telemetry.csv",
    "episode_manifest.json",
    "episode_summary.json",
    "results.json",
    "run_manifest.json",
    "summary.json",
    "phase_cues.csv",
    "video_manifest.json",
]
TEMPORAL_SEGMENT_SOURCES = [
    "step_records.jsonl",
    "step_telemetry.csv",
    "phase_cues.csv",
    "episode_manifest.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            obj = dict(obj)
            obj["__source_file"] = str(path)
            obj["__source_line"] = line_no
            yield obj


def read_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            yield row


def read_records(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(str(path))
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".jl"):
            rows.extend(read_jsonl(path))
        elif suffix == ".csv":
            rows.extend(read_csv_rows(path))
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            data = obj.get("records") if isinstance(obj, dict) else obj
            if not isinstance(data, list):
                raise TypeError(f"{path} must be a JSON list or mapping with records")
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(f"{path}[{i}] is not an object")
                item = dict(item)
                item["__source_file"] = str(path)
                item["__source_line"] = i
                rows.append(item)
        else:
            raise ValueError(f"unsupported input suffix: {path}")
    return rows


def first_value(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def record_id(row: Dict[str, Any]) -> str:
    return str(first_value(row, ["parent_id", "episode_key", "run_id", "record_id", "id"], f"{row.get('__source_file')}:{row.get('__source_line')}"))


def infer_suite(row: Dict[str, Any]) -> str:
    direct = str(first_value(row, ["suite", "suite_name", "benchmark", "libero_suite"], "")).strip()
    if direct in SUITES:
        return direct
    text = " ".join(str(first_value(row, [k], "")) for k in ["parent_id", "episode_key", "run_id", "task_id", "task_name", "path", "output_root"])
    low = text.lower()
    if "libero_10" in low or "libero-10" in low or "libero10" in low or "moka" in low:
        return "libero_10"
    if "libero_spatial" in low or "spatial" in low or "black_bowl" in low:
        return "libero_spatial"
    if "libero_goal" in low or "goal" in low or "drawer" in low:
        return "libero_goal"
    if "libero_object" in low or "object" in low or "alphabet_soup" in low:
        return "libero_object"
    return "UNKNOWN"


def clean_success_value(row: Dict[str, Any]) -> str:
    value = first_value(row, ["clean_success", "success", "task_success", "episode_success"], "")
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def task_name_value(row: Dict[str, Any]) -> str:
    return str(first_value(row, ["task_name", "libero_task_name", "task", "language_instruction", "instruction"], ""))


def has_segment_fields(row: Dict[str, Any]) -> bool:
    return any(field in row and row[field] not in (None, "") for field in LIBERO10_SEGMENT_FIELDS)


def root_candidates_from_row(row: Dict[str, Any], extra_roots: List[str]) -> List[Path]:
    out: List[Path] = []
    for field in ROOT_HINT_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if value:
            out.append(Path(os.path.expandvars(os.path.expanduser(value))))
    rid = record_id(row)
    for root in extra_roots:
        base = Path(os.path.expandvars(os.path.expanduser(root)))
        if not base.exists():
            continue
        # Cheap exact candidate paths only.  No recursive scanning by default.
        out.append(base / rid)
        out.append(base / rid.replace("/", "_"))
    seen = set()
    uniq = []
    for p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def sentinel_availability(row: Dict[str, Any], extra_roots: List[str], recursive_search: bool, max_recursive_hits: int) -> Tuple[Dict[str, str], List[str]]:
    availability = {name: "" for name in SENTINELS}
    candidates = root_candidates_from_row(row, extra_roots)
    checked_roots: List[str] = []
    for root in candidates:
        if not root.exists():
            continue
        checked_roots.append(str(root))
        if root.is_file():
            if root.name in availability:
                availability[root.name] = str(root)
            continue
        for name in SENTINELS:
            p = root / name
            if p.exists() and not availability[name]:
                availability[name] = str(p)
        if recursive_search:
            hits = 0
            for p in root.rglob("*"):
                if p.name in availability and not availability[p.name]:
                    availability[p.name] = str(p)
                    hits += 1
                    if hits >= max_recursive_hits:
                        break
    return availability, checked_roots


def segment_status_for(row: Dict[str, Any], suite: str, availability: Dict[str, str]) -> str:
    if suite != "libero_10":
        return "SEGMENT_SINGLE_EVENT_COMPATIBLE"
    if has_segment_fields(row):
        return "SEGMENT_FIELDS_PRESENT"
    if any(availability.get(name) for name in TEMPORAL_SEGMENT_SOURCES):
        return "NEEDS_LIBERO10_EVENT_RESOLVER_WITH_TEMPORAL_ARTIFACTS"
    return "NEEDS_LIBERO10_EVENT_RESOLVER_MISSING_TEMPORAL_ARTIFACTS"


def build_debug(records: List[Dict[str, Any]], extra_roots: List[str], recursive_search: bool, max_recursive_hits: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    segment_rows: List[Dict[str, Any]] = []
    queue_rows: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    counters = Counter()
    suite_counters: Dict[str, Counter] = defaultdict(Counter)
    for row in records:
        rid = record_id(row)
        suite = infer_suite(row)
        availability, checked_roots = sentinel_availability(row, extra_roots, recursive_search, max_recursive_hits)
        status = segment_status_for(row, suite, availability)
        task_name = task_name_value(row)
        clean_success = clean_success_value(row)
        present_segment_fields = ";".join([f for f in LIBERO10_SEGMENT_FIELDS if row.get(f) not in (None, "")])
        temporal_sources = ";".join([name for name in TEMPORAL_SEGMENT_SOURCES if availability.get(name)])
        segment_row = {
            "record_id": rid,
            "suite": suite,
            "task_name": task_name,
            "clean_success": clean_success,
            "segment_debug_status": status,
            "has_existing_segment_fields": has_segment_fields(row),
            "present_segment_fields": present_segment_fields,
            "temporal_segment_sources_present": temporal_sources,
            "recommended_next_action": recommended_action(status),
            "source_file": row.get("__source_file", ""),
            "source_line": row.get("__source_line", ""),
        }
        segment_rows.append(segment_row)
        artifact_row = {
            "record_id": rid,
            "suite": suite,
            "checked_roots": ";".join(checked_roots),
            **{f"has_{name}": bool(path) for name, path in availability.items()},
            **{f"path_{name}": path for name, path in availability.items()},
        }
        artifact_rows.append(artifact_row)
        if suite == "libero_10" and status != "SEGMENT_FIELDS_PRESENT":
            queue_rows.append({
                "record_id": rid,
                "suite": suite,
                "task_name": task_name,
                "clean_success": clean_success,
                "segment_debug_status": status,
                "temporal_segment_sources_present": temporal_sources,
                "checked_roots": ";".join(checked_roots),
                "recommended_next_action": recommended_action(status),
                "source_file": row.get("__source_file", ""),
                "source_line": row.get("__source_line", ""),
            })
        counters[status] += 1
        suite_counters[suite][status] += 1
        suite_counters[suite]["records"] += 1
    summary = {
        "status_counts": dict(sorted(counters.items())),
        "suite_status_counts": {suite: dict(counter) for suite, counter in sorted(suite_counters.items())},
        "libero10_resolution_queue_count": len(queue_rows),
    }
    return segment_rows, queue_rows, artifact_rows, summary


def recommended_action(status: str) -> str:
    if status == "SEGMENT_SINGLE_EVENT_COMPATIBLE":
        return "use_single_primary_event_teacher_label_v2"
    if status == "SEGMENT_FIELDS_PRESENT":
        return "validate_existing_libero10_segment_fields_then_label_v2"
    if status == "NEEDS_LIBERO10_EVENT_RESOLVER_WITH_TEMPORAL_ARTIFACTS":
        return "run_libero10_event_resolver_from_step_records_or_phase_cues"
    if status == "NEEDS_LIBERO10_EVENT_RESOLVER_MISSING_TEMPORAL_ARTIFACTS":
        return "recover_temporal_artifact_paths_before_segment_resolver"
    return "manual_debug_required"


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    config = load_yaml(Path(args.protocol_config))
    expected_total = int(args.expected_total or config.get("data_source", {}).get("expected_episode_count", 2000))
    records = read_records(args.clean2000_records)
    segment_rows, queue_rows, artifact_rows, summary = build_debug(records, args.artifact_root, args.recursive_artifact_search, args.max_recursive_hits)
    ids = Counter(record_id(r) for r in records)
    duplicate_ids = [rid for rid, n in ids.items() if n > 1]
    unknown_count = sum(1 for row in segment_rows if row["suite"] == "UNKNOWN")
    lib10_total = sum(1 for row in segment_rows if row["suite"] == "libero_10")
    lib10_unresolved = len(queue_rows)
    if len(records) != expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={expected_total} observed_total={len(records)}"
    elif duplicate_ids:
        status = "HOLD_CLEAN2000_DUPLICATE_RECORD_IDS"
        reason = json.dumps(duplicate_ids[:50])
    elif unknown_count:
        status = "HOLD_CLEAN2000_UNKNOWN_SUITE_ROWS"
        reason = f"unknown_suite_rows={unknown_count}"
    elif lib10_total == 0:
        status = "HOLD_NO_LIBERO10_RECORDS_FOUND"
        reason = "no libero_10 records found in CLEAN2000 source records"
    elif lib10_unresolved == 0:
        status = PASS_ALREADY_SEGMENTED
        reason = ""
    else:
        status = PASS_DEBUG_QUEUE
        reason = "LIBERO-10 segment resolver queue built; detector training still blocked until segment/event fields are materialized"

    write_csv(out / "clean2000_segment_skeleton.csv", segment_rows, [
        "record_id", "suite", "task_name", "clean_success", "segment_debug_status",
        "has_existing_segment_fields", "present_segment_fields", "temporal_segment_sources_present",
        "recommended_next_action", "source_file", "source_line",
    ])
    write_csv(out / "clean2000_libero10_resolution_queue.csv", queue_rows, [
        "record_id", "suite", "task_name", "clean_success", "segment_debug_status",
        "temporal_segment_sources_present", "checked_roots", "recommended_next_action", "source_file", "source_line",
    ])
    artifact_fields = ["record_id", "suite", "checked_roots"]
    for name in SENTINELS:
        artifact_fields.append(f"has_{name}")
        artifact_fields.append(f"path_{name}")
    write_csv(out / "clean2000_artifact_availability.csv", artifact_rows, artifact_fields)
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "protocol_config": str(args.protocol_config),
        "input_files": [str(p) for p in args.clean2000_records],
        "input_file_sha256": {str(p): sha256_file(Path(p)) for p in args.clean2000_records},
        "expected_total": expected_total,
        "observed_total": len(records),
        "artifact_roots": args.artifact_root,
        "recursive_artifact_search": bool(args.recursive_artifact_search),
        "summary": summary,
        "libero10_record_count": lib10_total,
        "libero10_unresolved_count": lib10_unresolved,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only debug stage. PASS_DEBUG_QUEUE means the exact LIBERO-10 records needing segment/event resolution have been identified; it does not authorize detector training or rollout.",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_reset": "NOT_PERFORMED",
            "env_set_init_state": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_libero10_segment_debug_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol-config", default="configs/clean2000_multisuite_detector_v2.yaml")
    p.add_argument("--clean2000-records", action="append", required=True)
    p.add_argument("--expected-total", type=int, default=0)
    p.add_argument("--artifact-root", action="append", default=[], help="Optional root used to check exact record-id artifact directories; no recursive search unless requested")
    p.add_argument("--recursive-artifact-search", action="store_true")
    p.add_argument("--max-recursive-hits", type=int, default=25)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
