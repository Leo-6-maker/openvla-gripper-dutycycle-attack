#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

GATE = "D1B2_CLEAN2000_LIBERO10_SEGMENT_QUALITY_PADDING_AWARE_AUDIT"
PASS = "PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_PADDING_AWARE_AUDITED"
OUT_FILES = [
    "libero10_segment_quality_v2_report.json",
    "libero10_segment_quality_v2_by_record.csv",
    "libero10_segment_quality_v2_violations.csv",
    "libero10_segment_quality_v2_length_distribution.csv",
    "libero10_segment_quality_v2_by_task.csv",
    "checksum_report.json",
]
PRIMARY_ROLE = "primary_attackable"
VALID_ROLES = {"primary_attackable", "auxiliary_manipulation", "distractor_or_setup", "unsupported_or_abstain"}


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


def record_id(row: Dict[str, Any]) -> str:
    return str(row.get("record_id", ""))


def add_violation(rows: List[Dict[str, Any]], record: str, code: str, detail: str, severity: str = "HOLD") -> None:
    rows.append({"record_id": record, "violation_code": code, "severity": severity, "detail": detail})


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def segments_by_record(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in segments:
        out[record_id(row)].append(row)
    for rid in out:
        out[rid].sort(key=lambda r: (to_int(r, "segment_index"), to_int(r, "segment_start_step"), to_int(r, "segment_end_step")))
    return out


def audit(registry_rows: List[Dict[str, Any]], segment_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_record = segments_by_record(segment_rows)
    registry_ids = [record_id(r) for r in registry_rows]
    registry_set = set(registry_ids)
    violations: List[Dict[str, Any]] = []
    record_rows: List[Dict[str, Any]] = []
    task_counter: Dict[str, Counter] = defaultdict(Counter)
    all_lengths: List[int] = []
    primary_lengths: List[int] = []
    aux_lengths: List[int] = []

    if len(registry_ids) != len(registry_set):
        for rid, n in Counter(registry_ids).items():
            if n > 1:
                add_violation(violations, rid, "DUPLICATE_REGISTRY_RECORD_ID", f"count={n}")
    for sid, n in Counter(str(s.get("segment_id", "")) for s in segment_rows).items():
        if sid and n > 1:
            add_violation(violations, sid, "DUPLICATE_SEGMENT_ID", f"count={n}")
    for eid, n in Counter(str(s.get("event_id", "")) for s in segment_rows).items():
        if eid and n > 1:
            add_violation(violations, eid, "DUPLICATE_EVENT_ID", f"count={n}")
    for rid in sorted(set(by_record) - registry_set):
        add_violation(violations, rid, "SEGMENT_RECORD_NOT_IN_REGISTRY", "segment candidate references record absent from frozen registry")

    padded_expected_max = args.resolver_max_segment_len + args.resolver_pre_window + args.resolver_post_window + 1
    hold_len = args.max_padded_segment_len if args.max_padded_segment_len > 0 else padded_expected_max
    warn_len = args.warn_padded_segment_len if args.warn_padded_segment_len > 0 else args.resolver_max_segment_len

    for reg in registry_rows:
        rid = record_id(reg)
        task_key = str(reg.get("structured_key", reg.get("task_id", "UNKNOWN")) or "UNKNOWN")
        segs = by_record.get(rid, [])
        task_counter[task_key]["records"] += 1
        if not segs:
            add_violation(violations, rid, "NO_SEGMENTS_FOR_RECORD", "registry record has no segments")
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
        lengths: List[int] = []
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
            lengths.append(length)
            all_lengths.append(length)
            if s.get("event_role") == PRIMARY_ROLE:
                primary_lengths.append(length)
            else:
                aux_lengths.append(length)
            if end < start:
                add_violation(violations, rid, "NEGATIVE_SEGMENT_LENGTH", f"segment_id={s.get('segment_id')} start={start} end={end}")
            if length < args.min_segment_len:
                add_violation(violations, rid, "SEGMENT_TOO_SHORT", f"segment_id={s.get('segment_id')} length={length}", severity="WARN" if args.warn_short_segments else "HOLD")
            if length > hold_len:
                add_violation(violations, rid, "SEGMENT_PADDED_LENGTH_EXCEEDS_RESOLVER_BOUND", f"segment_id={s.get('segment_id')} length={length} hold_len={hold_len} resolver_core={args.resolver_max_segment_len} pre={args.resolver_pre_window} post={args.resolver_post_window}")
            elif length > warn_len:
                add_violation(violations, rid, "SEGMENT_LONG_DUE_TO_PADDING", f"segment_id={s.get('segment_id')} length={length} warn_len={warn_len} hold_len={hold_len}", severity="WARN")
            if not (start <= anchor <= end):
                add_violation(violations, rid, "ANCHOR_NOT_INSIDE_SEGMENT", f"segment_id={s.get('segment_id')} start={start} anchor={anchor} end={end}")
            if not (wstart <= anchor <= wend):
                add_violation(violations, rid, "ANCHOR_NOT_INSIDE_TEACHER_WINDOW", f"segment_id={s.get('segment_id')} wstart={wstart} anchor={anchor} wend={wend}")
            role = str(s.get("event_role", ""))
            if role not in VALID_ROLES:
                add_violation(violations, rid, "UNKNOWN_EVENT_ROLE", f"segment_id={s.get('segment_id')} role={role}")
            if role == PRIMARY_ROLE:
                primary_index = idx
        if args.require_primary_last and primary_index is not None:
            max_idx = max(to_int(s, "segment_index") for s in segs)
            if primary_index != max_idx:
                add_violation(violations, rid, "PRIMARY_NOT_LAST_SEGMENT", f"primary_index={primary_index} max_index={max_idx}")
        overlap_count = 0
        for (s0, e0), (s1, e1) in zip(sorted(zip(starts, ends)), sorted(zip(starts, ends))[1:]):
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
            "length_min": min(lengths) if lengths else "",
            "length_max": max(lengths) if lengths else "",
            "first_start": min(starts) if starts else "",
            "last_end": max(ends) if ends else "",
            "overlap_count": overlap_count,
            "primary_index": primary_index if primary_index is not None else "",
            "status": "PASS" if primary_count == 1 else "FAIL",
        })

    task_rows: List[Dict[str, Any]] = []
    for key, c in sorted(task_counter.items()):
        task_rows.append({
            "structured_key": key,
            "records": c.get("records", 0),
            "segments": c.get("segments", 0),
            "primary_segments": c.get("primary_segments", 0),
            "records_with_primary": c.get("records_with_primary", 0),
            "mean_segments_per_record": c.get("segments", 0) / max(1, c.get("records", 0)),
        })
    length_rows = []
    for name, values in [("all", all_lengths), ("primary", primary_lengths), ("auxiliary", aux_lengths)]:
        length_rows.append({
            "subset": name,
            "count": len(values),
            "min": min(values) if values else 0,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else 0,
            "warn_len": warn_len,
            "hold_len": hold_len,
        })
    hard = [v for v in violations if v.get("severity") == "HOLD"]
    summary = {
        "registry_record_count": len(registry_rows),
        "segment_row_count": len(segment_rows),
        "record_count_with_segments": len(by_record),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "hard_violations_by_code": dict(Counter(v["violation_code"] for v in hard)),
        "segments_per_record_min": min([r["segment_count"] for r in record_rows], default=0),
        "segments_per_record_max": max([r["segment_count"] for r in record_rows], default=0),
        "segments_per_record_mean": sum(r["segment_count"] for r in record_rows) / max(1, len(record_rows)),
        "primary_records": sum(1 for r in record_rows if r["primary_count"] == 1),
        "padded_expected_max": padded_expected_max,
        "warn_len": warn_len,
        "hold_len": hold_len,
    }
    return record_rows, violations, task_rows, length_rows, summary


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
    record_rows, violations, task_rows, length_rows, summary = audit(registry_rows, segment_rows, args)
    if len(registry_rows) != args.expected_records:
        status = "HOLD_REGISTRY_RECORD_COUNT_MISMATCH"
        reason = f"expected_records={args.expected_records} observed={len(registry_rows)}"
    elif summary["primary_records"] != args.expected_records:
        status = "HOLD_PRIMARY_RECORD_COVERAGE_MISMATCH"
        reason = f"primary_records={summary['primary_records']} expected={args.expected_records}"
    elif summary["hard_violation_count"]:
        status = "HOLD_LIBERO10_SEGMENT_QUALITY_V2_VIOLATIONS"
        reason = f"hard_violation_count={summary['hard_violation_count']}"
    else:
        status = PASS
        reason = ""
    write_csv(out / "libero10_segment_quality_v2_by_record.csv", record_rows, ["record_id", "structured_key", "task_id", "state_index", "segment_count", "primary_count", "length_min", "length_max", "first_start", "last_end", "overlap_count", "primary_index", "status"])
    write_csv(out / "libero10_segment_quality_v2_violations.csv", violations, ["record_id", "violation_code", "severity", "detail"])
    write_csv(out / "libero10_segment_quality_v2_by_task.csv", task_rows, ["structured_key", "records", "segments", "primary_segments", "records_with_primary", "mean_segments_per_record"])
    write_csv(out / "libero10_segment_quality_v2_length_distribution.csv", length_rows, ["subset", "count", "min", "p50", "p90", "p95", "p99", "max", "warn_len", "hold_len"])
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
            "resolver_max_segment_len": args.resolver_max_segment_len,
            "resolver_pre_window": args.resolver_pre_window,
            "resolver_post_window": args.resolver_post_window,
            "warn_padded_segment_len": args.warn_padded_segment_len,
            "max_padded_segment_len": args.max_padded_segment_len,
            "min_segment_len": args.min_segment_len,
            "max_segments_per_record": args.max_segments_per_record,
            "require_primary_last": args.require_primary_last,
            "hold_on_overlap": args.hold_on_overlap,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only padding-aware segment quality audit. Lengths greater than the resolver core bound can be warnings if they are explained by pre/post padding; HOLD is reserved for structural or padded-bound violations.",
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
    write_json(out / "libero10_segment_quality_v2_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-registry", required=True)
    p.add_argument("--segment-candidates", required=True)
    p.add_argument("--expected-records", type=int, default=500)
    p.add_argument("--resolver-max-segment-len", type=int, default=120)
    p.add_argument("--resolver-pre-window", type=int, default=5)
    p.add_argument("--resolver-post-window", type=int, default=10)
    p.add_argument("--warn-padded-segment-len", type=int, default=0)
    p.add_argument("--max-padded-segment-len", type=int, default=0)
    p.add_argument("--min-segments-per-record", type=int, default=1)
    p.add_argument("--max-segments-per-record", type=int, default=12)
    p.add_argument("--min-segment-len", type=int, default=3)
    p.add_argument("--require-primary-last", action="store_true")
    p.add_argument("--hold-on-overlap", action="store_true")
    p.add_argument("--warn-short-segments", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
