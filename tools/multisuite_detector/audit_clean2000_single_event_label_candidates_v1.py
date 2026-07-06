#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

GATE = "D1D2_CLEAN2000_SINGLE_EVENT_LABEL_CANDIDATE_AUDIT"
PASS = "PASS_CLEAN2000_SINGLE_EVENT_LABEL_CANDIDATES_AUDITED"
OUT_FILES = [
    "single_event_label_candidate_audit_report.json",
    "single_event_label_candidate_audit_by_suite.csv",
    "single_event_label_candidate_audit_violations.csv",
    "single_event_label_candidate_audit_by_record.csv",
    "checksum_report.json",
]
VALID_EVENT_STATUS = {"VALID_PRIMARY_CANDIDATE", "NO_EVENT"}
VALID_EVENT_ROLE = {"primary_attackable", "unsupported_or_abstain"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_int(value: Any) -> int:
    return int(float(value))


def add_violation(rows: List[Dict[str, Any]], rid: str, code: str, detail: str, severity: str = "HOLD") -> None:
    rows.append({"record_id": rid, "violation_code": code, "severity": severity, "detail": detail})


def status_summary(summary: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in summary or []:
        suite = str(row.get("suite", ""))
        out[suite] = {}
        for k, v in row.items():
            if k == "suite":
                continue
            try:
                out[suite][k] = int(float(v))
            except Exception:
                pass
    return out


def audit(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_csv(Path(args.single_event_labels))
    bindings = read_csv(Path(args.single_event_bindings)) if args.single_event_bindings else []
    d1d1_report = json.loads(Path(args.single_event_report).read_text(encoding="utf-8"))

    violations: List[Dict[str, Any]] = []
    record_rows: List[Dict[str, Any]] = []
    by_suite: Dict[str, Counter] = defaultdict(Counter)
    label_by_record: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    binding_by_record = {str(r.get("record_id", "")): r for r in bindings}

    for row in labels:
        rid = str(row.get("record_id", ""))
        label_by_record[rid].append(row)

    for rid, rows in sorted(label_by_record.items()):
        if len(rows) != 1:
            add_violation(violations, rid, "LABEL_ROWS_PER_RECORD_NOT_ONE", f"count={len(rows)}")
        row = rows[0]
        suite = str(row.get("suite", ""))
        event_status = str(row.get("event_status", ""))
        event_role = str(row.get("event_role", ""))
        source_status = str(row.get("source_status", ""))
        by_suite[suite]["label_rows"] += 1
        by_suite[suite][f"event_status:{event_status}"] += 1
        by_suite[suite][f"event_role:{event_role}"] += 1
        by_suite[suite][f"source_status:{source_status}"] += 1
        by_suite[suite][f"resolver_method:{row.get('resolver_method', '')}"] += 1
        by_suite[suite][f"resolver_signal_field:{row.get('resolver_signal_field', '')}"] += 1
        if event_status not in VALID_EVENT_STATUS:
            add_violation(violations, rid, "UNKNOWN_EVENT_STATUS", event_status)
        if event_role not in VALID_EVENT_ROLE:
            add_violation(violations, rid, "UNKNOWN_EVENT_ROLE", event_role)
        if event_status == "VALID_PRIMARY_CANDIDATE":
            if event_role != "primary_attackable":
                add_violation(violations, rid, "POSITIVE_ROLE_NOT_PRIMARY", event_role)
            try:
                anchor = to_int(row.get("teacher_anchor_step"))
                wstart = to_int(row.get("teacher_window_start"))
                wend = to_int(row.get("teacher_window_end"))
                sstart = to_int(row.get("segment_start_step"))
                send = to_int(row.get("segment_end_step"))
            except Exception as exc:
                add_violation(violations, rid, "POSITIVE_TIME_PARSE_ERROR", type(exc).__name__)
            else:
                if anchor < 0:
                    add_violation(violations, rid, "NEGATIVE_POSITIVE_ANCHOR", str(anchor))
                if not (wstart <= anchor <= wend):
                    add_violation(violations, rid, "ANCHOR_NOT_INSIDE_WINDOW", f"{wstart}<={anchor}<={wend}")
                if not (sstart <= anchor <= send):
                    add_violation(violations, rid, "ANCHOR_NOT_INSIDE_SEGMENT", f"{sstart}<={anchor}<={send}")
            b = binding_by_record.get(rid)
            if b and not str(b.get("temporal_path", "")):
                add_violation(violations, rid, "POSITIVE_BINDING_WITHOUT_TEMPORAL_PATH", "")
        if event_status == "NO_EVENT":
            if event_role != "unsupported_or_abstain":
                add_violation(violations, rid, "NO_EVENT_ROLE_NOT_ABSTAIN", event_role)
            try:
                anchor = to_int(row.get("teacher_anchor_step"))
            except Exception as exc:
                add_violation(violations, rid, "NO_EVENT_ANCHOR_PARSE_ERROR", type(exc).__name__)
            else:
                if anchor != -1:
                    add_violation(violations, rid, "NO_EVENT_ANCHOR_NOT_NEGATIVE_ONE", str(anchor))
        record_rows.append({
            "record_id": rid,
            "suite": suite,
            "event_status": event_status,
            "event_role": event_role,
            "source_status": source_status,
            "resolver_method": row.get("resolver_method", ""),
            "resolver_signal_field": row.get("resolver_signal_field", ""),
        })

    if bindings:
        missing_label = sorted(set(binding_by_record) - set(label_by_record))
        extra_label = sorted(set(label_by_record) - set(binding_by_record))
        for rid in missing_label[:5000]:
            add_violation(violations, rid, "BINDING_WITHOUT_LABEL", "")
        for rid in extra_label[:5000]:
            add_violation(violations, rid, "LABEL_WITHOUT_BINDING", "")
    suite_rows: List[Dict[str, Any]] = []
    for suite in sorted(by_suite):
        c = by_suite[suite]
        suite_rows.append({
            "suite": suite,
            "label_rows": c.get("label_rows", 0),
            "valid_primary": c.get("event_status:VALID_PRIMARY_CANDIDATE", 0),
            "no_event": c.get("event_status:NO_EVENT", 0),
            "primary_attackable": c.get("event_role:primary_attackable", 0),
            "unsupported_or_abstain": c.get("event_role:unsupported_or_abstain", 0),
            "source_positive": c.get("source_status:SOURCE_POSITIVE", 0),
            "source_no_event": c.get("source_status:NO_EVENT", 0),
            "heuristic_opening_delta_quantile": c.get("resolver_method:heuristic_opening_delta_quantile", 0),
            "no_event_passthrough": c.get("resolver_method:no_event_passthrough", 0),
            "gripper_qpos": c.get("resolver_signal_field:gripper_qpos", 0),
        })

    label_rows = len(labels)
    event_counts = Counter(str(r.get("event_status", "")) for r in labels)
    role_counts = Counter(str(r.get("event_role", "")) for r in labels)
    source_counts = Counter(str(r.get("source_status", "")) for r in labels)
    resolver_counts = Counter(str(r.get("resolver_method", "")) for r in labels)
    signal_counts = Counter(str(r.get("resolver_signal_field", "")) for r in labels)
    report_status = str(d1d1_report.get("status", ""))
    embedded_summary = status_summary(d1d1_report.get("summary_by_suite", []))
    embedded_summary_warning_count = 0
    for row in suite_rows:
        suite = row["suite"]
        embedded = embedded_summary.get(suite, {})
        for key in ["source_no_event", "no_event_labels", "no_event"]:
            if key in embedded and embedded[key] not in {row["source_no_event"], row["no_event"]}:
                embedded_summary_warning_count += 1
                add_violation(violations, suite, "EMBEDDED_REPORT_SUMMARY_MISMATCH", f"{key}: embedded={embedded[key]} audited_source_no_event={row['source_no_event']} audited_no_event={row['no_event']}", severity="WARN")

    hard = [v for v in violations if v.get("severity") == "HOLD"]
    if report_status != "PASS_CLEAN2000_SINGLE_EVENT_ANCHORS_RESOLVED_FROM_BINDINGS":
        status = "HOLD_SINGLE_EVENT_D1D1_REPORT_NOT_PASS"
        reason = report_status
    elif label_rows != args.expected_records:
        status = "HOLD_SINGLE_EVENT_LABEL_COUNT_MISMATCH"
        reason = f"labels={label_rows} expected={args.expected_records}"
    elif event_counts.get("VALID_PRIMARY_CANDIDATE", 0) != args.expected_positive:
        status = "HOLD_SINGLE_EVENT_POSITIVE_COUNT_MISMATCH"
        reason = f"valid_primary={event_counts.get('VALID_PRIMARY_CANDIDATE', 0)} expected={args.expected_positive}"
    elif event_counts.get("NO_EVENT", 0) != args.expected_no_event:
        status = "HOLD_SINGLE_EVENT_NO_EVENT_COUNT_MISMATCH"
        reason = f"no_event={event_counts.get('NO_EVENT', 0)} expected={args.expected_no_event}"
    elif hard:
        status = "HOLD_SINGLE_EVENT_LABEL_CANDIDATE_AUDIT_VIOLATIONS"
        reason = f"hard_violation_count={len(hard)}"
    else:
        status = PASS
        reason = ""

    write_csv(out / "single_event_label_candidate_audit_by_suite.csv", suite_rows, ["suite", "label_rows", "valid_primary", "no_event", "primary_attackable", "unsupported_or_abstain", "source_positive", "source_no_event", "heuristic_opening_delta_quantile", "no_event_passthrough", "gripper_qpos"])
    write_csv(out / "single_event_label_candidate_audit_violations.csv", violations, ["record_id", "violation_code", "severity", "detail"])
    write_csv(out / "single_event_label_candidate_audit_by_record.csv", record_rows, ["record_id", "suite", "event_status", "event_role", "source_status", "resolver_method", "resolver_signal_field"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "single_event_labels": args.single_event_labels,
        "single_event_labels_sha256": sha256_file(Path(args.single_event_labels)),
        "single_event_report": args.single_event_report,
        "single_event_report_sha256": sha256_file(Path(args.single_event_report)),
        "single_event_bindings": args.single_event_bindings,
        "single_event_bindings_sha256": sha256_file(Path(args.single_event_bindings)) if args.single_event_bindings else "",
        "d1d1_report_status": report_status,
        "expected_records": args.expected_records,
        "expected_positive": args.expected_positive,
        "expected_no_event": args.expected_no_event,
        "label_row_count": label_rows,
        "event_status_counts": dict(event_counts),
        "event_role_counts": dict(role_counts),
        "source_status_counts": dict(source_counts),
        "resolver_method_counts": dict(resolver_counts),
        "resolver_signal_field_counts": dict(signal_counts),
        "hard_violation_count": len(hard),
        "warning_count": len(violations) - len(hard),
        "violations_by_code": dict(Counter(v["violation_code"] for v in violations)),
        "audited_summary_by_suite": suite_rows,
        "embedded_summary_warning_count": embedded_summary_warning_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only audit of D1D1 single-event label candidates. The candidate CSV is treated as the source of truth; embedded D1D1 report per-suite summary mismatches are warnings when CSV-level invariants pass.",
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_dataset_build": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "single_event_label_candidate_audit_report.json", report)
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--single-event-labels", required=True)
    p.add_argument("--single-event-report", required=True)
    p.add_argument("--single-event-bindings", default="")
    p.add_argument("--expected-records", type=int, default=1500)
    p.add_argument("--expected-positive", type=int, default=803)
    p.add_argument("--expected-no-event", type=int, default=697)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return audit(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
