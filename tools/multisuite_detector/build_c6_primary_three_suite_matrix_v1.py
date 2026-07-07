#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

CONDITIONS = ["CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"]
PRIMARY = ["libero_goal", "libero_object", "libero_spatial"]
EXCLUDED = ["libero_10"]
BOUNDARY = {"label_mutation": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"}
REQ = ["parent_id", "episode_key", "suite", "task_id", "condition", "clean_success_parent", "condition_success", "contact_quality_failure", "contact_quality_success", "nad_g", "delta_open", "qpos_response", "width_response", "arm_dev", "latency", "command_open_duty", "sustained_open_duty", "exact_prefix_shared", "clean_success_parent_denominator"]
JSON_OUT = ["matrix_manifest.json", "detector_freeze_identity.json", "replay_identity.json", "run_config.json", "metrics_summary.json", "gripper_bridge_report.json", "command_duty_report.json", "control_integrity_report.json", "primary_suite_policy.json", "libero10_exclusion_policy.json"]
CSV_OUT = ["outcomes_overall.csv", "outcomes_by_suite.csv", "outcomes_by_task.csv"]

class C6BuildError(ValueError):
    pass

def fail(msg: str) -> None:
    raise C6BuildError(msg)

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b=""):
            h.update(chunk)
    return h.hexdigest()

def boolv(v: str, name: str) -> bool:
    if str(v) in {"1", "true", "TRUE", "yes", "YES"}:
        return True
    if str(v) in {"0", "false", "FALSE", "no", "NO", ""}:
        return False
    fail(f"{name}: expected boolean")

def flt(v: str) -> float:
    if v == "" or str(v).upper() in {"NA", "N/A", "NOT_APPLICABLE"}:
        return float("nan")
    x = float(v)
    if not math.isfinite(x):
        fail("non-finite float")
    return x

def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            fail("source CSV has no header")
        missing = [c for c in REQ if c not in r.fieldnames]
        if missing:
            fail("source CSV missing columns: " + ",".join(missing))
        for row in r:
            if row["suite"] not in set(PRIMARY):
                fail("source must contain only primary suites")
            if row["condition"] not in set(CONDITIONS):
                fail("unknown condition")
            rows.append({
                "parent_id": row["parent_id"],
                "episode_key": row["episode_key"],
                "suite": row["suite"],
                "task_id": row["task_id"],
                "condition": row["condition"],
                "clean_success_parent": boolv(row["clean_success_parent"], "clean_success_parent"),
                "condition_success": boolv(row["condition_success"], "condition_success"),
                "contact_quality_failure": boolv(row["contact_quality_failure"], "contact_quality_failure"),
                "contact_quality_success": boolv(row["contact_quality_success"], "contact_quality_success"),
                "nad_g": flt(row["nad_g"]),
                "delta_open": flt(row["delta_open"]),
                "qpos_response": flt(row["qpos_response"]),
                "width_response": flt(row["width_response"]),
                "arm_dev": flt(row["arm_dev"]),
                "latency": flt(row["latency"]),
                "command_open_duty": flt(row["command_open_duty"]),
                "sustained_open_duty": flt(row["sustained_open_duty"]),
                "exact_prefix_shared": boolv(row["exact_prefix_shared"], "exact_prefix_shared"),
                "clean_success_parent_denominator": boolv(row["clean_success_parent_denominator"], "clean_success_parent_denominator"),
            })
    if not rows:
        fail("source CSV has no rows")
    return rows

def check_rows(rows: list[dict[str, Any]]) -> None:
    suites = {x["suite"] for x in rows}
    if suites != set(PRIMARY):
        fail("source must include exactly the primary suites")
    pairs: dict[str, set[str]] = defaultdict(set)
    for x in rows:
        if not x["clean_success_parent"] or not x["exact_prefix_shared"] or not x["clean_success_parent_denominator"]:
            fail("source rows must use clean-success exact-prefix parent denominator")
        pairs[x["parent_id"]].add(x["condition"])
    bad = [p for p, c in pairs.items() if c != set(CONDITIONS)]
    if bad:
        fail("each parent must contain all required conditions")

def vals(items: list[dict[str, Any]], k: str) -> list[float]:
    return [x[k] for x in items if math.isfinite(x[k])]

def avg(items: list[dict[str, Any]], k: str) -> str:
    v = vals(items, k)
    return f"{mean(v):.10g}" if v else ""

def med(items: list[dict[str, Any]], k: str) -> str:
    v = vals(items, k)
    return f"{median(v):.10g}" if v else ""

def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    g: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for x in rows:
        g[tuple(str(x[k]) for k in keys)].append(x)
    out = []
    for key, items in sorted(g.items()):
        n = len(items)
        d = {k: key[i] for i, k in enumerate(keys)}
        d["parent_count"] = n
        d["FR"] = f"{sum(1 for x in items if not x['condition_success']) / n:.10g}"
        d["CQFR"] = f"{sum(1 for x in items if x['contact_quality_failure']) / n:.10g}"
        d["CQSR"] = f"{sum(1 for x in items if x['contact_quality_success']) / n:.10g}"
        for m in ["nad_g", "delta_open", "qpos_response", "width_response", "arm_dev", "latency", "command_open_duty", "sustained_open_duty"]:
            d[f"mean_{m}"] = avg(items, m)
            d[f"median_{m}"] = med(items, m)
        out.append(d)
    return out

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_sums(root: Path) -> tuple[str, str]:
    lines = [f"{sha256_file(root / name)}  {name}" for name in JSON_OUT + CSV_OUT]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(root / "SHA256SUMS"), sha256_file(root / "SHA256SUMS.sha256")

def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.freeze_manifest) != args.expected_freeze_sha256:
        fail("freeze sha mismatch")
    if sha256_file(args.replay_manifest) != args.expected_replay_sha256:
        fail("replay sha mismatch")
    rows = read_rows(args.source_csv)
    check_rows(rows)
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    overall = summarize(rows, ["condition"])
    by_suite = summarize(rows, ["suite", "condition"])
    by_task = summarize(rows, ["suite", "task_id", "condition"])
    write_csv(root / "outcomes_overall.csv", overall)
    write_csv(root / "outcomes_by_suite.csv", by_suite)
    write_csv(root / "outcomes_by_task.csv", by_task)
    common = {**BOUNDARY, "conditions": CONDITIONS, "primary_positive_suites": PRIMARY, "excluded_suites": EXCLUDED, "exact_prefix_shared": True, "clean_success_parent_denominator": True, "source_csv_sha256": sha256_file(args.source_csv)}
    write_json(root / "matrix_manifest.json", {"status": "PASS", "gate": "C6_PRIMARY_THREE_SUITE_CONDITION_MATRIX", "freeze_manifest_sha256": args.expected_freeze_sha256, "replay_manifest_sha256": args.expected_replay_sha256, **common})
    write_json(root / "detector_freeze_identity.json", {"status": "PASS", "freeze_manifest_sha256": args.expected_freeze_sha256, **BOUNDARY})
    write_json(root / "replay_identity.json", {"status": "PASS", "replay_manifest_sha256": args.expected_replay_sha256, **BOUNDARY})
    write_json(root / "run_config.json", {"status": "PASS", "detector_threshold": args.detector_threshold, "threshold_source": "validation", "normalization_source": "train_only", **common})
    write_json(root / "metrics_summary.json", {"status": "PASS", "clean_success_parent_count": len({x['parent_id'] for x in rows}), "overall": overall, **BOUNDARY})
    write_json(root / "gripper_bridge_report.json", {"status": "PASS", "by_condition": overall, **BOUNDARY})
    write_json(root / "command_duty_report.json", {"status": "PASS", "by_condition": overall, **BOUNDARY})
    write_json(root / "control_integrity_report.json", {"status": "PASS", "exact_prefix_shared": True, "clean_success_parent_denominator": True, "control_conditions_present": True, **BOUNDARY})
    write_json(root / "primary_suite_policy.json", {"status": "PASS", "primary_positive_suites": PRIMARY, **BOUNDARY})
    write_json(root / "libero10_exclusion_policy.json", {"status": "PASS", "libero_10_positive_denominator": "EXCLUDED", **BOUNDARY})
    a, b = write_sums(root)
    return {"status": "PASS", "output_root": str(root), "SHA256SUMS": a, "SHA256SUMS.sha256": b}

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-csv", required=True)
    p.add_argument("--freeze-manifest", required=True)
    p.add_argument("--replay-manifest", required=True)
    p.add_argument("--expected-freeze-sha256", required=True)
    p.add_argument("--expected-replay-sha256", required=True)
    p.add_argument("--detector-threshold", type=float, default=0.4)
    p.add_argument("--output-root", required=True)
    try:
        print(json.dumps(run(p.parse_args()), sort_keys=True))
        return 0
    except (OSError, csv.Error, json.JSONDecodeError, C6BuildError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
