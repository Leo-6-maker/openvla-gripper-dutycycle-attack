#!/usr/bin/env python3
"""Audit whether selected C6 parents can be bound to reset/exact-prefix execution.

This is a static, fail-closed audit. It does not run OpenVLA, LIBERO, rollout,
intervention, or attack code.
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
RESET_HINTS = ["initial_state", "reset", "state_id", "state_path", "bddl", "init", "seed"]
RUNNER_PARENT_TOKENS = ["parent-id", "parent_id", "episode-key", "episode_key", "initial-state", "initial_state", "state-id", "state_id"]
RUNNER_PREFIX_TOKENS = ["exact-prefix", "exact_prefix", "prefix", "replay-prefix", "restore"]
RUNNER_OUTPUT_TOKENS = ["output-json", "output_json", "result_json", "json"]


class AuditError(ValueError):
    pass


def fail(msg: str) -> None:
    raise AuditError(msg)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b=""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"{path}: missing header")
        return list(reader)


def load_parents(path: str | Path) -> list[dict[str, Any]]:
    obj = read_json(path)
    parents = obj.get("parents", obj) if isinstance(obj, dict) else obj
    if not isinstance(parents, list) or not parents:
        fail("selected parents must be nonempty list or object with parents")
    out = []
    for p in parents:
        if not isinstance(p, dict):
            fail("selected parent must be object")
        for key in ["parent_id", "episode_key", "suite", "task_id"]:
            if key not in p or not str(p[key]):
                fail(f"selected parent missing {key}")
        if p["suite"] not in PRIMARY:
            fail("selected parent suite must be primary")
        out.append({k: str(v) for k, v in p.items()})
    return out


def contains_any(text: str, tokens: list[str]) -> bool:
    low = text.lower()
    return any(tok.lower() in low for tok in tokens)


def row_matches(row: dict[str, str], parent: dict[str, Any]) -> bool:
    ep = row.get("episode_key") or row.get("episode") or row.get("trajectory_id") or ""
    suite = row.get("suite") or row.get("benchmark") or ""
    task = row.get("task_id") or row.get("task") or ""
    return (ep == parent["episode_key"] or parent["parent_id"] in ep or ep.startswith(parent["parent_id"])) and (not suite or suite == parent["suite"]) and (not task or task == parent["task_id"])


def reset_fields(row: dict[str, str]) -> dict[str, str]:
    out = {}
    for k, v in row.items():
        if v and contains_any(k, RESET_HINTS):
            out[k] = v
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    parents = load_parents(args.selected_parents_json)
    dataset = read_csv_rows(args.dataset_csv) if args.dataset_csv else []
    split = read_csv_rows(args.split_csv) if args.split_csv else []
    label_rows = read_csv_rows(args.label_csv) if args.label_csv else []
    runner_text = Path(args.legacy_runner).read_text(encoding="utf-8", errors="replace") if args.legacy_runner and Path(args.legacy_runner).is_file() else ""
    parent_reports = []
    for p in parents:
        dmatch = [r for r in dataset if row_matches(r, p)]
        smatch = [r for r in split if row_matches(r, p)]
        lmatch = [r for r in label_rows if row_matches(r, p)]
        fields = reset_fields(dmatch[0]) if dmatch else {}
        parent_reports.append({
            "parent_id": p["parent_id"],
            "episode_key": p["episode_key"],
            "suite": p["suite"],
            "task_id": p["task_id"],
            "dataset_match_count": len(dmatch),
            "split_match_count": len(smatch),
            "label_match_count": len(lmatch),
            "reset_candidate_fields": fields,
            "reset_binding_ready": bool(fields),
        })
    runner_report = {
        "path": args.legacy_runner,
        "exists": bool(args.legacy_runner and Path(args.legacy_runner).is_file()),
        "accepts_parent_or_state_args": contains_any(runner_text, RUNNER_PARENT_TOKENS),
        "mentions_exact_prefix_or_restore": contains_any(runner_text, RUNNER_PREFIX_TOKENS),
        "mentions_json_output": contains_any(runner_text, RUNNER_OUTPUT_TOKENS),
    }
    all_reset = all(r["reset_binding_ready"] for r in parent_reports)
    if not all_reset:
        status = "HOLD_PARENT_RESET_UNBOUND"
    elif not runner_report["accepts_parent_or_state_args"]:
        status = "HOLD_RUNNER_PARENT_CLI_UNBOUND"
    elif not runner_report["mentions_exact_prefix_or_restore"]:
        status = "HOLD_EXACT_PREFIX_UNBOUND"
    elif not runner_report["mentions_json_output"]:
        status = "HOLD_FIELD_OUTPUT_UNBOUND"
    else:
        status = "PASS_STATIC_RESET_BINDING_CANDIDATE"
    return {
        "status": status,
        "schema_version": "c6_parent_reset_binding_audit_v1",
        "selected_parents_sha256": sha256_file(args.selected_parents_json),
        "dataset_csv_sha256": sha256_file(args.dataset_csv) if args.dataset_csv else None,
        "split_csv_sha256": sha256_file(args.split_csv) if args.split_csv else None,
        "label_csv_sha256": sha256_file(args.label_csv) if args.label_csv else None,
        "parents": parent_reports,
        "legacy_runner": runner_report,
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "artifact_mutation": "NOT_PERFORMED",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-parents-json", required=True)
    p.add_argument("--legacy-runner", required=True)
    p.add_argument("--dataset-csv")
    p.add_argument("--split-csv")
    p.add_argument("--label-csv")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()
    try:
        report = audit(args)
    except (OSError, csv.Error, json.JSONDecodeError, AuditError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    write_json(args.output_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
