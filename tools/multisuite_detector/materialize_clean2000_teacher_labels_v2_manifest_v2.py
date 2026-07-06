#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D1C2_CLEAN2000_TEACHER_LABEL_V2_MANIFEST_MATERIALIZATION"
PASS = "PASS_CLEAN2000_TEACHER_LABEL_V2_MANIFEST_MATERIALIZED"
PASS_LIBERO10_ONLY = "PASS_CLEAN2000_TEACHER_LABEL_V2_LIBERO10_ONLY_DEBUG"
ACCEPTED_QUALITY_STATUSES = {
    "PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_AUDITED",
    "PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_PADDING_AWARE_AUDITED",
}
OUT_FILES = [
    "clean2000_teacher_labels_v2_manifest_report.json",
    "clean2000_teacher_labels_v2_manifest.csv",
    "clean2000_teacher_labels_v2_missing_single_event_sources.csv",
    "clean2000_teacher_labels_v2_source_column_audit.csv",
    "clean2000_teacher_labels_v2_summary_by_suite.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
SINGLE_EVENT_SUITES = ["libero_spatial", "libero_goal", "libero_object"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction", "path", "output_root", "state_id"]
ANCHOR_FIELDS = [
    "teacher_anchor_step", "positive_anchor_step", "source_positive_anchor_step", "source_positive_anchor",
    "anchor_step", "event_step", "trigger_step", "target_step", "selected_preplace_step", "preplace_step",
    "release_intent_step", "source_release_intent_step", "grasp_anchor_step",
]
WINDOW_START_FIELDS = ["teacher_window_start", "window_start", "positive_window_start", "source_positive_window_start", "event_window_start"]
WINDOW_END_FIELDS = ["teacher_window_end", "window_end", "positive_window_end", "source_positive_window_end", "event_window_end"]
STATUS_FIELDS = ["teacher_label_status", "label_status", "source_label_status", "source_event_status", "event_status"]
NO_EVENT_HINTS = {"NO_EVENT", "SOURCE_NO_EVENT", "CLEAN_FAILURE_NO_POSITIVE", "UNSUPPORTED_MECHANISM", "AMBIGUOUS", "NONE", "0", "FALSE"}
VALID_HINTS = {"VALID", "VALID_PRIMARY", "SOURCE_POSITIVE", "POSITIVE", "1", "TRUE"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = []
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            rows.append(row)
        return rows


def first_value(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def record_id(row: Dict[str, Any]) -> str:
    return str(first_value(row, ID_FIELDS, f"{row.get('__source_file')}:{row.get('__source_line')}"))


def infer_suite(row: Dict[str, Any]) -> str:
    direct = str(first_value(row, ["suite", "suite_name", "benchmark", "libero_suite"], "")).strip()
    if direct in SUITES:
        return direct
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
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


def to_int_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_jsonish_anchor(row: Dict[str, Any]) -> Tuple[int | None, str]:
    for key, value in row.items():
        if value in (None, ""):
            continue
        low_key = key.lower()
        if not any(tok in low_key for tok in ["json", "record", "metadata", "source", "extra"]):
            continue
        text = str(value).strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for field in ANCHOR_FIELDS:
            anchor = to_int_value(obj.get(field))
            if anchor is not None:
                return anchor, f"json_field:{key}.{field}"
    return None, ""


def source_status(row: Dict[str, Any]) -> str:
    status = str(first_value(row, STATUS_FIELDS, "")).strip()
    if status:
        return status
    # Some recovered source ledgers store booleans rather than status strings.
    for key in ["source_positive_anchor_valid", "positive_anchor_valid", "has_positive_anchor", "source_event_valid"]:
        if key in row and str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}:
            return "SOURCE_POSITIVE"
    for key in ["source_no_event", "no_event", "clean_failed"]:
        if key in row and str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}:
            return "NO_EVENT"
    return "UNKNOWN"


def source_single_event_label(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    rid = record_id(row)
    suite = infer_suite(row)
    status = source_status(row)
    anchor_source = "direct_field"
    anchor = to_int_value(first_value(row, ANCHOR_FIELDS, ""))
    if anchor is None:
        anchor, anchor_source = parse_jsonish_anchor(row)
    if anchor is None:
        return None, {
            "record_id": rid,
            "suite": suite,
            "source_status": status,
            "missing_reason": "NO_SINGLE_EVENT_ANCHOR_FIELD",
            "available_anchor_like_columns": ";".join(sorted(k for k in row.keys() if re.search(r"anchor|window|event|trigger|release|preplace", k, re.I))),
            "available_columns": ";".join(sorted(k for k in row.keys() if not k.startswith("__"))),
            "source_file": row.get("__source_file", ""),
            "source_line": row.get("__source_line", ""),
        }
    wstart = to_int_value(first_value(row, WINDOW_START_FIELDS, ""))
    wend = to_int_value(first_value(row, WINDOW_END_FIELDS, ""))
    if wstart is None:
        wstart = anchor - args.single_event_window_pre
    if wend is None:
        wend = anchor + args.single_event_window_post
    if wstart > anchor or wend < anchor:
        return None, {
            "record_id": rid,
            "suite": suite,
            "source_status": status,
            "missing_reason": "SINGLE_EVENT_WINDOW_DOES_NOT_CONTAIN_ANCHOR",
            "available_anchor_like_columns": ";".join(sorted(k for k in row.keys() if re.search(r"anchor|window|event|trigger|release|preplace", k, re.I))),
            "available_columns": ";".join(sorted(k for k in row.keys() if not k.startswith("__"))),
            "source_file": row.get("__source_file", ""),
            "source_line": row.get("__source_line", ""),
        }
    return {
        "record_id": rid,
        "suite": suite,
        "task_id": str(first_value(row, ["task_id"], "")),
        "task_name": str(first_value(row, ["task_name", "instruction", "language_instruction"], "")),
        "event_id": f"{rid}::event_00",
        "segment_id": f"{rid}::segment_00",
        "segment_index": 0,
        "segment_start_step": wstart,
        "segment_end_step": wend,
        "teacher_anchor_step": anchor,
        "teacher_window_start": wstart,
        "teacher_window_end": wend,
        "segment_role": "primary_single_event",
        "subtask_index": 0,
        "event_role": "primary_attackable",
        "event_status": "VALID_PRIMARY_CANDIDATE",
        "phase_label": "stable_carry",
        "corridor_label": 1,
        "release_safe_label": 0,
        "label_confidence": args.single_event_label_confidence,
        "label_source": f"source_single_event_anchor_fields:{anchor_source}",
        "source_status": status,
        "online_input_allowed": "NO_LABEL_ONLY",
    }, None


def libero10_label_rows(segment_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in segment_rows:
        out.append({
            "record_id": s.get("record_id", ""),
            "suite": s.get("suite", "libero_10"),
            "task_id": s.get("task_id", ""),
            "task_name": s.get("task_name", ""),
            "event_id": s.get("event_id", ""),
            "segment_id": s.get("segment_id", ""),
            "segment_index": s.get("segment_index", ""),
            "segment_start_step": s.get("segment_start_step", ""),
            "segment_end_step": s.get("segment_end_step", ""),
            "teacher_anchor_step": s.get("teacher_anchor_step", ""),
            "teacher_window_start": s.get("teacher_window_start", ""),
            "teacher_window_end": s.get("teacher_window_end", ""),
            "segment_role": s.get("segment_role", ""),
            "subtask_index": s.get("subtask_index", ""),
            "event_role": s.get("event_role", ""),
            "event_status": s.get("event_status", ""),
            "phase_label": s.get("phase_label", ""),
            "corridor_label": s.get("corridor_label", ""),
            "release_safe_label": s.get("release_safe_label", ""),
            "label_confidence": s.get("label_confidence", ""),
            "label_source": "d1a_libero10_segment_candidate_resolver_d1b2_quality_audited",
            "source_status": "SEGMENT_RESOLVED",
            "online_input_allowed": "NO_LABEL_ONLY",
        })
    return out


def column_audit(source_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for row in source_rows:
        suite = infer_suite(row)
        for col, value in row.items():
            if col.startswith("__"):
                continue
            if re.search(r"anchor|window|event|trigger|release|preplace|positive|no_event|status", col, re.I):
                counts[(suite, col)]["present"] += int(value not in (None, ""))
                counts[(suite, col)]["rows"] += 1
    out = []
    for (suite, col), c in sorted(counts.items()):
        out.append({"suite": suite, "column": col, "nonempty_count": c.get("present", 0), "row_count": c.get("rows", 0)})
    return out


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
    source_rows = read_csv(Path(args.clean2000_records))
    segment_rows = read_csv(Path(args.libero10_segments))
    quality = json.loads(Path(args.segment_quality_report).read_text(encoding="utf-8")) if args.segment_quality_report else {}
    quality_status = str(quality.get("status", "NOT_PROVIDED"))
    if quality_status not in ACCEPTED_QUALITY_STATUSES and not args.allow_unverified_libero10_segments:
        raise SystemExit(f"segment quality report not accepted: {quality_status}")

    labels: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    labels.extend(libero10_label_rows(segment_rows))
    for row in source_rows:
        suite = infer_suite(row)
        if suite == "libero_10":
            continue
        if suite in SINGLE_EVENT_SUITES:
            label, miss = source_single_event_label(row, args)
            if label:
                labels.append(label)
            if miss:
                missing.append(miss)
        else:
            missing.append({"record_id": record_id(row), "suite": suite, "source_status": source_status(row), "missing_reason": "UNKNOWN_SUITE", "available_anchor_like_columns": "", "available_columns": "", "source_file": row.get("__source_file", ""), "source_line": row.get("__source_line", "")})

    by_suite = defaultdict(Counter)
    for row in labels:
        by_suite[row["suite"]]["label_rows"] += 1
        if row.get("event_role") == "primary_attackable":
            by_suite[row["suite"]]["primary_rows"] += 1
        if row.get("event_role") == "auxiliary_manipulation":
            by_suite[row["suite"]]["auxiliary_rows"] += 1
    for row in missing:
        by_suite[row["suite"]]["missing_records"] += 1
    summary_rows = []
    for suite in SUITES + ["UNKNOWN"]:
        c = by_suite.get(suite, Counter())
        if c or suite != "UNKNOWN":
            summary_rows.append({
                "suite": suite,
                "label_rows": c.get("label_rows", 0),
                "primary_rows": c.get("primary_rows", 0),
                "auxiliary_rows": c.get("auxiliary_rows", 0),
                "missing_records": c.get("missing_records", 0),
            })
    single_missing = sum(1 for m in missing if m.get("suite") in SINGLE_EVENT_SUITES)
    if len(source_rows) != args.expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={args.expected_total} observed={len(source_rows)}"
    elif single_missing and not args.allow_single_event_missing_debug:
        status = "HOLD_SINGLE_EVENT_LABEL_SOURCES_MISSING"
        reason = f"missing_single_event_records={single_missing}"
    elif single_missing and args.allow_single_event_missing_debug:
        status = PASS_LIBERO10_ONLY
        reason = "LIBERO-10 label manifest emitted; single-event suites still need source anchor labels"
    else:
        status = PASS
        reason = ""

    fields = ["record_id", "suite", "task_id", "task_name", "event_id", "segment_id", "segment_index", "segment_start_step", "segment_end_step", "teacher_anchor_step", "teacher_window_start", "teacher_window_end", "segment_role", "subtask_index", "event_role", "event_status", "phase_label", "corridor_label", "release_safe_label", "label_confidence", "label_source", "source_status", "online_input_allowed"]
    write_csv(out / "clean2000_teacher_labels_v2_manifest.csv", labels, fields)
    write_csv(out / "clean2000_teacher_labels_v2_missing_single_event_sources.csv", missing, ["record_id", "suite", "source_status", "missing_reason", "available_anchor_like_columns", "available_columns", "source_file", "source_line"])
    write_csv(out / "clean2000_teacher_labels_v2_source_column_audit.csv", column_audit(source_rows), ["suite", "column", "nonempty_count", "row_count"])
    write_csv(out / "clean2000_teacher_labels_v2_summary_by_suite.csv", summary_rows, ["suite", "label_rows", "primary_rows", "auxiliary_rows", "missing_records"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "clean2000_records": args.clean2000_records,
        "clean2000_records_sha256": sha256_file(Path(args.clean2000_records)),
        "libero10_segments": args.libero10_segments,
        "libero10_segments_sha256": sha256_file(Path(args.libero10_segments)),
        "segment_quality_report": args.segment_quality_report,
        "segment_quality_report_sha256": sha256_file(Path(args.segment_quality_report)) if args.segment_quality_report else "",
        "segment_quality_status": quality_status,
        "accepted_quality_statuses": sorted(ACCEPTED_QUALITY_STATUSES),
        "expected_total": args.expected_total,
        "source_record_count": len(source_rows),
        "label_row_count": len(labels),
        "missing_record_count": len(missing),
        "missing_single_event_record_count": single_missing,
        "summary_by_suite": summary_rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only teacher-label-v2 manifest materialization. PASS means four-suite label manifest is complete. LIBERO10_ONLY is debug-only and not detector-dataset-ready.",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
            "detector_training": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_teacher_labels_v2_manifest_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", required=True)
    p.add_argument("--libero10-segments", required=True)
    p.add_argument("--segment-quality-report", required=True)
    p.add_argument("--expected-total", type=int, default=2000)
    p.add_argument("--single-event-window-pre", type=int, default=3)
    p.add_argument("--single-event-window-post", type=int, default=12)
    p.add_argument("--single-event-label-confidence", type=float, default=0.60)
    p.add_argument("--allow-unverified-libero10-segments", action="store_true")
    p.add_argument("--allow-single-event-missing-debug", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
