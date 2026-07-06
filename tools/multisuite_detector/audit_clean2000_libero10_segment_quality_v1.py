#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D1B_CLEAN2000_LIBERO10_SEGMENT_QUALITY_AUDIT"
PASS = "PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_AUDITED"
OUT_FILES = [
    "libero10_segment_quality_report.json",
    "libero10_segment_quality_by_record.csv",
    "libero10_segment_quality_violations.csv",
    "libero10_segment_quality_by_task.csv",
    "checksum_report.json",
]
PRIMARY_ROLE = "primary_attackable"
AUX_ROLE = "auxiliary_manipulation"


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
        return [dict(row) for row in csv.DictReader(f)]


def to_int(row: Dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def to_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def record_id(row: Dict[str, Any]) -> str:
    return str(row.get("record_id", ""))


def add_violation(rows: List[Dict[str, Any]], record: str, code: str, detail: str, severity: str = "HOLD") -> None:
    rows.append({"record_id": record, "violation_code": code, "severity": severity, "detail": detail})


def segments_by_record(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in segments:
        out[record_id(row)].append(row)
    for rid in out:
        out[rid].sort(key=lambda r: (to_int(r, "segment_index"), to_int(r, "segment_start_step"), to_int(r, "segment_end_step")))
    return out


def audit_records(registry_rows: List[Dict[str, Any]], segment_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_record = segments_by_record(segment_rows)
    registry_ids = [record_id(r) for r in registry_rows]
    registry_set = set(registry_ids)
    segment_ids = [str(r.get("segment_id", "")) for r in segment_rows]
    event_ids = [str(r.get("event_id", "")) for r in segment_rows]
    violations: List[Dict[str, Any]] = []
    record_rows: List[Dict[str, Any]] = []
    task_counter: Dict[str, Counter] = defaultdict(Counter)

    if len(registry_ids) != len(registry_set):
        dup = [rid for rid, n in Counter(registry_ids).items() if n > 1]
        for rid in dup[:100]:
            add_violation(violations, rid, "DUPLICATE_REGISTRY_RECORD_ID", "record_id duplicated in registry")
    for sid, n in Counter(segment_ids).items():
        if sid and n > 1:
            add_violation(violations, sid, "DUPLICATE_SEGMENT_ID", f"segment_id count={n}")
    for eid, n in Counter(event_ids).items():
        if eid and n > 1:
            add_violation(violations, eid, "DUPLICATE_EVENT_ID", f"event_id count={n}")
    extra_records = sorted(set(by_record) - registry_set)
    for rid in extra_records[:200]:
        add_violation(violations, rid, "SEGMENT_RECORD_NOT_IN_REGISTRY", "segment candidate references record_id absent from frozen registry")

    for reg in registry_rows:
        rid = record_id(reg)
        task_key = str(reg.get("structured_key", reg.get("task_id", "UNKNOWN")) or "UNKNOWN")
        segs = by_record.get(rid, [])
        task_counter[task_key]["records"] += 1
        if not segs:
            add_violation(violations, rid, "NO_SEGMENTS_FOR_RECORD", "registry record has no segment candidates")
            record_rows.append({"record_id": rid, "structured_key": task_key, "segment_count": 0, "primary_count": 0, "status": "FAIL"})
            continue
        primary = [s for s in segs if s.get("event_role") == PRIMARY_ROLE]
        primary_count = len(primary)
        if primary_count != 1:
            add_violation(violations, rid, "PRIMARY_COUNT_NOT_ONE", f"primary_count={primary_count}")
        if len(segs) > args.max_segments_per_record:
            add_violation(violations, rid, "TOO_MANY_SEGMENTS", f"segment_count={len(segs)} max={args.max_segments_per_record}")
        if len(segs) < args.min_segments_per_record:
            add_violation(violations, rid, "TOO_FEW_SEGMENTS", f"segment_count={len(segs)} min={args.min_segments_per_record}")
        starts: List[int] = []
        ends: List[int] = []
        primary_index = None
        for s in segs:
            try:
                start = to_int(s, "segment_start_step")
                end = to_int(s, "segment_end_step")
                anchor = to_int(s, "teacher_anchor_step")
                wstart = to_int(s, "teacher_window_start")
                wend = to_int(s, "teacher_window_end")
                idx = to_int(s, "segment_index")
            except Exception as exc:
                add_violation(violations, rid, "SEGMENT_TIME_PARSE_ERROR", f"segment_id={s.get('segment_id')} error={type(exc).__name__}: {exc}")
                continue
            starts.append(start)
            ends.append(end)
            length = end - start + 1
            if end < start:
                add_violation(violations, rid, "NEGATIVE_SEGMENT_LENGTH", f"segment_id={s.get('segment_id')} start={start} end={end}")
            if length < args.min_segment_len:
                add_violation(violations, rid, "SEGMENT_TOO_SHORT", f"segment_id={s.get('segment_id')} length={length}", severity="WARN" if args.warn_short_segments else "HOLD")
            if length > args.max_segment_len:
                add_violation(violations, rid, "SEGMENT_TOO_LONG", f"segment_id={s.get('segment_id')} length={length}", severity="WARN" if args.warn_long_segments else "HOLD")
            if not (start <= anchor <= end):
                add_violation(violations, rid, "ANCHOR_NOT_INSIDE_SEGMENT", f"segment_id={s.get('segment_id')} start={start} anchor={anchor} end={end}")
            if not (wstart <= anchor <= wend):
                add_violation(violations, rid, "ANCHOR_NOT_INSIDE_TEACHER_WINDOW", f"segment_id={s.get('segment_id')} wstart={wstart} anchor={anchor} wend={wend}")
            role = str(s.get("event_role", ""))
            if role not in {PRIMARY_ROLE, AUX_ROLE, "distractor_or_setup", "unsupported_or_abstain"}:
                add_violation(violations, rid, "UNKNOWN_EVENT_ROLE", f"segment_id={s.get('segment_id')} role={role}")
            if role == PRIMARY_ROLE:
                primary_index = idx
        if args.require_primary_last and primary_index is not None:
            max_idx = max(to_int(s, "segment_index") for s in segs)
            if primary_index != max_idx:
                add_violation(violations, rid, "PRIMARY_NOT_LAST_SEGMENT", f"primary_index={primary_index} max_index={max_idx}")
        overlap_count = 0
        sorted_pairs = sorted(zip(starts, ends))
        for (s0, e0), (s1, e1) in zip(sorted_pairs, sorted_pairs[1:]):
            if s1 <= e0:
                overlap_count += 1
        if overlap_count and args.hold_on_overlap:
            add_violation(violations, rid, "OVERLAPPING_SEGMENTS", f"overlap_count={overlap_count}")
        task_counter[task_key]["segments"] += len(segs)
        task_counter[task_key]["primary_segments"] += primary_count
        task_counter[task_key]["records_with_primary"] += int(primary_count == 1)
        record_rows.append({
            "record_id": rid,
            "structured_key": task_key,
            "task_id": reg.get("task_id", ""),
            "state_index": reg.get("state_index", ""),
            "segment_count": len(segs),
            "primary_count": primary_count,
            "first_start": min(starts) if starts else "",
            "last_end": max(ends) if ends else "",
            "overlap_count": overlap_count,
            "primary_index": primary_index if primary_index is not None else "",
            "status": "PASS" if primary_count == 1 else "FAIL",
        })
    task_rows = []
    for key, c in sorted(task_counter.items()):
        task_rows.append({
            "structured_key": key,
            "records": c.get("records", 0),
            "segments": c.get("segments", 0),
            "primary_segments": c.get("primary_segments", 0),
            "records_with_primary": c.get("records_with_primary", 0),
            "mean_segments_per_record": c.get("segments", 0) / max(1, c.get("records", 0)),
        })
    hard_violations = [v for v in violations if v.get("severity") == "HOLD"]
    summary = {
        "registry_record_count": len(registry_rows),
        "segment_row_count": len(segment_rows),
        "record_count_with_segments": len(by_record),
        "extra_segment_record_count": len(extra_records),
        "hard_violation_count": len(hard_violations),
        "warning_count": len(violations) - len(hard_violations),
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "segments_per_record_min": min([r["segment_count"] for r in record_rows], default=0),
        "segments_per_record_max": max([r["segment_count"] for r in record_rows], default=0),
        "segments_per_record_mean": sum(r["segment_count"] for r in record_rows) / max(1, len(record_rows)),
        "primary_records": sum(1 for r in record_rows if r["primary_count"] == 1),
    }
    return record_rows, violations, task_rows, summary


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
    registry_rows = read_csv(Path(args.artifact_registry))
    segment_rows = read_csv(Path(args.segment_candidates))
    record_rows, violations, task_rows, summary = audit_records(registry_rows, segment_rows, args)
    if len(registry_rows) != args.expected_records:
        status = "HOLD_REGISTRY_RECORD_COUNT_MISMATCH"
        reason = f"expected_records={args.expected_records} observed={len(registry_rows)}"
    elif summary["primary_records"] != args.expected_records:
        status = "HOLD_PRIMARY_RECORD_COVERAGE_MISMATCH"
        reason = f"primary_records={summary['primary_records']} expected={args.expected_records}"
    elif summary["hard_violation_count"]:
        status = "HOLD_LIBERO10_SEGMENT_QUALITY_VIOLATIONS"
        reason = f"hard_violation_count={summary['hard_violation_count']}"
    else:
        status = PASS
        reason = ""
    write_csv(out / "libero10_segment_quality_by_record.csv", record_rows, [
        "record_id", "structured_key", "task_id", "state_index", "segment_count", "primary_count", "first_start", "last_end", "overlap_count", "primary_index", "status",
    ])
    write_csv(out / "libero10_segment_quality_violations.csv", violations, ["record_id", "violation_code", "severity", "detail"])
    write_csv(out / "libero10_segment_quality_by_task.csv", task_rows, ["structured_key", "records", "segments", "primary_segments", "records_with_primary", "mean_segments_per_record"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "artifact_registry": args.artifact_registry,
        "artifact_registry_sha256": sha256_file(Path(args.artifact_registry)),
        "segment_candidates": args.segment_candidates,
        "segment_candidates_sha256": sha256_file(Path(args.segment_candidates)),
        "expected_records": args.expected_records,
        **summary,
        "parameters": {
            "min_segments_per_record": args.min_segments_per_record,
            "max_segments_per_record": args.max_segments_per_record,
            "min_segment_len": args.min_segment_len,
            "max_segment_len": args.max_segment_len,
            "require_primary_last": args.require_primary_last,
            "hold_on_overlap": args.hold_on_overlap,
            "warn_short_segments": args.warn_short_segments,
            "warn_long_segments": args.warn_long_segments,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only segment candidate quality audit. PASS means D1A candidates are structurally consistent enough to feed teacher-label-v2 materialization; it does not train a detector.",
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
    write_json(out / "libero10_segment_quality_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-registry", required=True)
    p.add_argument("--segment-candidates", required=True)
    p.add_argument("--expected-records", type=int, default=500)
    p.add_argument("--min-segments-per-record", type=int, default=1)
    p.add_argument("--max-segments-per-record", type=int, default=12)
    p.add_argument("--min-segment-len", type=int, default=3)
    p.add_argument("--max-segment-len", type=int, default=200)
    p.add_argument("--require-primary-last", action="store_true")
    p.add_argument("--hold-on-overlap", action="store_true")
    p.add_argument("--warn-short-segments", action="store_true")
    p.add_argument("--warn-long-segments", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
