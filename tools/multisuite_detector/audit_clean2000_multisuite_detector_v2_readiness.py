#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

GATE = "D0_CLEAN2000_MULTISUITE_DETECTOR_V2_READINESS_AUDIT"
PASS = "PASS_CLEAN2000_MULTISUITE_DETECTOR_V2_READY_FOR_LABEL_REBUILD"
OUT_FILES = [
    "clean2000_multisuite_detector_v2_readiness.json",
    "clean2000_suite_summary.csv",
    "clean2000_libero10_segment_policy_audit.csv",
    "clean2000_forbidden_input_audit.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
FORBIDDEN_HINTS = [
    "normalized_step",
    "timestep",
    "task_id",
    "state_id",
    "episode_key",
    "run_id",
    "parent_id",
    "object_pose",
    "target_pose",
    "object_to_target",
    "teacher_window",
    "teacher_anchor",
    "attack_outcome",
    "rand_outcome",
    "manual_anchor",
    "oracle_window",
]
LIBERO10_SEGMENT_HINTS = [
    "segment_id",
    "segment_index",
    "segment_start_step",
    "segment_end_step",
    "segment_role",
    "subtask_index",
    "event_id",
    "event_role",
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
        obj = yaml.safe_load(f)
    return obj or {}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            obj["__source_file"] = str(path)
            obj["__source_line"] = line_no
            yield obj


def read_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
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
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    if not isinstance(item, dict):
                        raise TypeError(f"{path}[{i}] is not a JSON object")
                    item = dict(item)
                    item["__source_file"] = str(path)
                    item["__source_line"] = i
                    rows.append(item)
            elif isinstance(obj, dict) and isinstance(obj.get("records"), list):
                for i, item in enumerate(obj["records"]):
                    item = dict(item)
                    item["__source_file"] = str(path)
                    item["__source_line"] = i
                    rows.append(item)
            else:
                raise TypeError(f"{path} must be a list or mapping with records")
        elif suffix == ".csv":
            rows.extend(read_csv_rows(path))
        else:
            raise ValueError(f"unsupported record file suffix for {path}")
    return rows


def first_value(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


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


def truthy(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if x in (None, ""):
        return None
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "pass", "success", "succeeded", "complete", "valid"):
        return True
    if s in ("0", "false", "no", "n", "fail", "failed", "invalid"):
        return False
    return None


def record_id(row: Dict[str, Any]) -> str:
    return str(first_value(row, ["parent_id", "episode_key", "run_id", "record_id", "id"], f"{row.get('__source_file')}:{row.get('__source_line')}"))


def has_any_field(row: Dict[str, Any], fields: Iterable[str]) -> bool:
    return any(field in row and row[field] not in (None, "") for field in fields)


def forbidden_columns(row: Dict[str, Any]) -> List[str]:
    cols = []
    for key in row.keys():
        low = str(key).lower()
        if low.startswith("__"):
            continue
        for hint in FORBIDDEN_HINTS:
            if hint in low:
                cols.append(str(key))
                break
    return sorted(set(cols))


def summarize(records: List[Dict[str, Any]], expected_total: int, require_libero10_segments: bool) -> tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    suite_counts: Dict[str, Counter] = defaultdict(Counter)
    ids = Counter(record_id(r) for r in records)
    duplicate_ids = [rid for rid, n in ids.items() if n > 1]
    lib10_rows: List[Dict[str, Any]] = []
    forbidden_rows: List[Dict[str, Any]] = []
    for row in records:
        suite = infer_suite(row)
        suite_counts[suite]["records"] += 1
        clean_success = truthy(first_value(row, ["clean_success", "success", "task_success", "episode_success"], ""))
        if clean_success is True:
            suite_counts[suite]["clean_success_true"] += 1
        elif clean_success is False:
            suite_counts[suite]["clean_success_false"] += 1
        else:
            suite_counts[suite]["clean_success_unknown"] += 1
        source_found = truthy(first_value(row, ["source_record_found", "source_found", "record_found", "source_valid"], ""))
        if source_found is True:
            suite_counts[suite]["source_found_true"] += 1
        elif source_found is False:
            suite_counts[suite]["source_found_false"] += 1
        else:
            suite_counts[suite]["source_found_unknown"] += 1
        if suite == "libero_10":
            has_segment = has_any_field(row, LIBERO10_SEGMENT_HINTS)
            lib10_rows.append({
                "record_id": record_id(row),
                "suite": suite,
                "has_segment_fields": has_segment,
                "present_segment_fields": ";".join([f for f in LIBERO10_SEGMENT_HINTS if f in row and row[f] not in (None, "")]),
                "source_file": row.get("__source_file", ""),
                "source_line": row.get("__source_line", ""),
            })
        bad_cols = forbidden_columns(row)
        if bad_cols:
            forbidden_rows.append({
                "record_id": record_id(row),
                "suite": suite,
                "forbidden_columns_present": ";".join(bad_cols),
                "source_file": row.get("__source_file", ""),
                "source_line": row.get("__source_line", ""),
            })
    suite_rows: List[Dict[str, Any]] = []
    for suite in SUITES + ["UNKNOWN"]:
        c = suite_counts.get(suite, Counter())
        if c or suite != "UNKNOWN":
            suite_rows.append({
                "suite": suite,
                "records": c.get("records", 0),
                "clean_success_true": c.get("clean_success_true", 0),
                "clean_success_false": c.get("clean_success_false", 0),
                "clean_success_unknown": c.get("clean_success_unknown", 0),
                "source_found_true": c.get("source_found_true", 0),
                "source_found_false": c.get("source_found_false", 0),
                "source_found_unknown": c.get("source_found_unknown", 0),
            })
    status = PASS
    reason = ""
    if len(records) != expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={expected_total} observed_total={len(records)}"
    elif duplicate_ids:
        status = "HOLD_CLEAN2000_DUPLICATE_RECORD_IDS"
        reason = json.dumps(duplicate_ids[:50])
    elif suite_counts.get("UNKNOWN", Counter()).get("records", 0) > 0:
        status = "HOLD_CLEAN2000_UNKNOWN_SUITE_ROWS"
        reason = json.dumps([r for r in suite_rows if r["suite"] == "UNKNOWN"], sort_keys=True)
    elif require_libero10_segments and any(not r["has_segment_fields"] for r in lib10_rows):
        status = "HOLD_LIBERO10_SEGMENT_FIELDS_MISSING"
        reason = "LIBERO-10 records require segment/event fields before detector-v2 training"
    elif forbidden_rows:
        status = "HOLD_FORBIDDEN_INPUT_COLUMNS_PRESENT"
        reason = "recovered rows contain fields that must not become model inputs; move them to label/eval-only schema before dataset freeze"
    return status, reason, suite_rows, lib10_rows, forbidden_rows


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
    require_lib10 = bool(config.get("libero10_multisegment_policy", {}).get("require_segment_fields_for_libero10", True))
    if args.allow_libero10_unsegmented_diagnostic:
        require_lib10 = False
    records = read_records(args.clean2000_records)
    status, reason, suite_rows, lib10_rows, forbidden_rows = summarize(records, expected_total, require_lib10)
    write_csv(out / "clean2000_suite_summary.csv", suite_rows, ["suite", "records", "clean_success_true", "clean_success_false", "clean_success_unknown", "source_found_true", "source_found_false", "source_found_unknown"])
    write_csv(out / "clean2000_libero10_segment_policy_audit.csv", lib10_rows, ["record_id", "suite", "has_segment_fields", "present_segment_fields", "source_file", "source_line"])
    write_csv(out / "clean2000_forbidden_input_audit.csv", forbidden_rows, ["record_id", "suite", "forbidden_columns_present", "source_file", "source_line"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "protocol_config": str(args.protocol_config),
        "input_files": [str(p) for p in args.clean2000_records],
        "input_file_sha256": {str(p): sha256_file(Path(p)) for p in args.clean2000_records},
        "expected_total": expected_total,
        "observed_total": len(records),
        "require_libero10_segments": require_lib10,
        "suite_summary": suite_rows,
        "libero10_record_count": len(lib10_rows),
        "libero10_missing_segment_count": sum(1 for r in lib10_rows if not r["has_segment_fields"]),
        "forbidden_input_row_count": len(forbidden_rows),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only readiness audit for CLEAN2000 detector-v2 recovery. PASS means source records are count-consistent, suite-resolved, and LIBERO-10 has segment/event fields ready for label-v2 rebuild.",
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
    write_json(out / "clean2000_multisuite_detector_v2_readiness.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol-config", default="configs/clean2000_multisuite_detector_v2.yaml")
    p.add_argument("--clean2000-records", action="append", required=True, help="Recovered CLEAN2000 source records as CSV/JSON/JSONL; repeatable")
    p.add_argument("--expected-total", type=int, default=0)
    p.add_argument("--allow-libero10-unsegmented-diagnostic", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
