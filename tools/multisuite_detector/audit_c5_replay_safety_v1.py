#!/usr/bin/env python3
"""Audit C5 detector-only replay safety and emission rates.

This tool is metadata-only. It reads an existing C5 detector-only replay evidence
folder and classifies whether high safety-trigger rates are mainly from the
primary positive-window suites or from diagnostic-only suites such as
LIBERO-10. It does not train, score, simulate, or modify any frozen artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PRIMARY = {"libero_goal", "libero_object", "libero_spatial"}
DIAGNOSTIC_ONLY = {"libero_10"}
SAFETY_FIELD_CANDIDATES = ["safety_false_trigger_rate", "false_trigger_rate", "safety_rate"]


class C5SafetyAuditError(ValueError):
    pass


def fail(message: str) -> None:
    raise C5SafetyAuditError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b=""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"{path.name}: expected JSON object")
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"{path.name}: empty header")
        rows = list(reader)
    if not rows:
        fail(f"{path.name}: empty rows")
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "" or str(value).upper() in {"NA", "N/A", "NOT_APPLICABLE", "NONE"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_any(row: dict[str, Any], names: list[str]) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row:
            return row[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def suite_safety_fallback(report: dict[str, Any]) -> dict[str, float]:
    """Extract per-suite safety rates from flexible JSON schemas."""
    out: dict[str, float] = {}
    for node in iter_dicts(report):
        suite = get_any(node, ["suite", "suite_name", "name", "dataset"])
        if suite is None:
            continue
        suite = str(suite).strip()
        if not suite:
            continue
        rate = parse_float(get_any(node, SAFETY_FIELD_CANDIDATES))
        if rate is None:
            continue
        out[suite] = rate
    return out


def validate_sums(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    side = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not side.is_file():
        fail("SHA256SUMS files missing")
    entries = 0
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            fail(f"SHA256SUMS:{line_no}: malformed")
        digest, rel = parts
        target = root / rel
        if Path(rel).is_absolute() or ".." in Path(rel).parts or not target.is_file():
            fail(f"SHA256SUMS:{line_no}: unsafe or missing path")
        if sha256_file(target) != digest:
            fail(f"SHA256SUMS:{line_no}: digest mismatch")
        entries += 1
    side_parts = side.read_text(encoding="utf-8").strip().split()
    if len(side_parts) != 2 or side_parts[1] != "SHA256SUMS" or sha256_file(sums) != side_parts[0]:
        fail("SHA256SUMS.sha256 mismatch")
    return {"entry_count": entries, "SHA256SUMS": sha256_file(sums), "SHA256SUMS.sha256": sha256_file(side)}


def extract_suite_rows(rows: list[dict[str, str]], fallback_safety: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        suite = str(get_any(row, ["suite", "name", "dataset"])).strip()
        if not suite:
            continue
        role = str(get_any(row, ["role", "suite_role"]) or ("diagnostic_only" if suite in DIAGNOSTIC_ONLY else "primary_positive"))
        raw_safety = parse_float(get_any(row, SAFETY_FIELD_CANDIDATES))
        safety_source = "metrics_by_suite.csv"
        safety = raw_safety
        if safety is None and suite in fallback_safety:
            safety = fallback_safety[suite]
            safety_source = "safety_false_trigger_report.json"
        elif safety is None:
            safety_source = "missing"
        emission = parse_float(get_any(row, ["emission_rate", "emit_rate"]))
        hit = parse_float(get_any(row, ["hit_rate", "positive_hit_rate"]))
        positive = parse_float(get_any(row, ["positive_support", "positive_episode_count"]), default=0.0)
        out.append({
            "suite": suite,
            "role": role,
            "positive_support": positive,
            "hit_rate": hit,
            "emission_rate": emission,
            "safety_false_trigger_rate": safety,
            "safety_rate_source": safety_source,
            "is_primary_positive_suite": suite in PRIMARY,
            "is_diagnostic_only_suite": suite in DIAGNOSTIC_ONLY,
        })
    return out


def decision(suite_rows: list[dict[str, Any]], max_primary_safety: float, max_diagnostic_safety: float) -> tuple[str, list[str]]:
    reasons = []
    primary = [r for r in suite_rows if r["suite"] in PRIMARY]
    diag = [r for r in suite_rows if r["suite"] in DIAGNOSTIC_ONLY]
    if not primary:
        return "HOLD", ["no primary suite rows"]
    bad_primary = [r for r in primary if r["safety_false_trigger_rate"] is not None and r["safety_false_trigger_rate"] > max_primary_safety]
    missing_primary = [r for r in primary if r["safety_false_trigger_rate"] is None]
    if missing_primary:
        reasons.append("primary suite missing safety rate")
    if bad_primary:
        reasons.append("primary suite safety false-trigger exceeds threshold")
    bad_diag = [r for r in diag if r["safety_false_trigger_rate"] is not None and r["safety_false_trigger_rate"] > max_diagnostic_safety]
    if bad_diag:
        reasons.append("diagnostic-only suite safety false-trigger is high")
    if missing_primary or bad_primary:
        return "HOLD_PRIMARY_SAFETY", reasons
    if bad_diag:
        return "PASS_PRIMARY_HOLD_DIAGNOSTIC", reasons
    return "PASS", ["primary suite safety acceptable"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.c5_root)
    if not root.is_dir():
        fail("C5 root does not exist")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    sums = validate_sums(root)
    overall = read_json(root / "metrics_overall.json")
    safety_report = read_json(root / "safety_false_trigger_report.json") if (root / "safety_false_trigger_report.json").is_file() else {}
    fallback = suite_safety_fallback(safety_report)
    suite_rows = extract_suite_rows(read_csv(root / "metrics_by_suite.csv"), fallback)
    status, reasons = decision(suite_rows, args.max_primary_safety_false_trigger, args.max_diagnostic_safety_false_trigger)
    table_rows = []
    for row in suite_rows:
        table_rows.append({
            "suite": row["suite"],
            "role": "primary_positive" if row["suite"] in PRIMARY else "diagnostic_only" if row["suite"] in DIAGNOSTIC_ONLY else row["role"],
            "positive_support": row["positive_support"],
            "hit_rate": row["hit_rate"],
            "emission_rate": row["emission_rate"],
            "safety_false_trigger_rate": row["safety_false_trigger_rate"],
            "safety_rate_source": row["safety_rate_source"],
        })
    write_csv(out / "primary_suite_safety_table.csv", ["suite", "role", "positive_support", "hit_rate", "emission_rate", "safety_false_trigger_rate", "safety_rate_source"], table_rows)
    report = {
        "status": status,
        "schema_version": "c5_replay_safety_triage_v1",
        "c5_root": str(root),
        "c5_replay_manifest_sha256": sha256_file(root / "replay_manifest.json"),
        "overall_safety_false_trigger_rate": parse_float(overall.get("safety_false_trigger_rate")),
        "overall_emission_rate": parse_float(overall.get("emission_rate")),
        "overall_hit_rate": parse_float(overall.get("hit_rate")),
        "max_primary_safety_false_trigger": args.max_primary_safety_false_trigger,
        "max_diagnostic_safety_false_trigger": args.max_diagnostic_safety_false_trigger,
        "decision_reasons": reasons,
        "primary_positive_suites": sorted(PRIMARY),
        "diagnostic_only_suites": sorted(DIAGNOSTIC_ONLY),
        "fallback_safety_suites": sorted(fallback),
        "missing_primary_safety_suites": sorted(r["suite"] for r in suite_rows if r["suite"] in PRIMARY and r["safety_false_trigger_rate"] is None),
        "c5_sha256sums": sums,
        "new_training": "NOT_PERFORMED",
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "artifact_mutation": "NOT_PERFORMED",
    }
    write_json(out / "safety_triage_summary.json", report)
    write_json(out / "c6_release_recommendation.json", {
        "status": status,
        "recommendation": "RELEASE_PRIMARY_SUITES_ONLY" if status in {"PASS", "PASS_PRIMARY_HOLD_DIAGNOSTIC"} else "HOLD_C6",
        "primary_positive_suites": sorted(PRIMARY),
        "diagnostic_only_suites": sorted(DIAGNOSTIC_ONLY),
        "reasons": reasons,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c5-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-primary-safety-false-trigger", type=float, default=0.15)
    parser.add_argument("--max-diagnostic-safety-false-trigger", type=float, default=0.50)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, json.JSONDecodeError, csv.Error, C5SafetyAuditError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
