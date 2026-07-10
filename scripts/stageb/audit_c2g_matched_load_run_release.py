#!/usr/bin/env python3
"""Release runtime audit that preserves preregistered no-emit CLEAN denominators.

The base matched-load audit is intentionally closed-world over attacked jobs. The
release builder keeps detector no-emit and burst-infeasible parents in a separate
excluded ledger, while their detector-only CLEAN artifacts remain under the online
root. This wrapper allows exactly those ledger-bound CLEAN artifacts and no other
unexpected episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from scripts.stageb.audit_c2g_matched_load_run import audit as base_audit


def read_excluded(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"excluded denominator ledger missing: {path}")
    parents: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not str(row.get("parent_key", "")).strip():
            raise ValueError(f"{path}:{line_no} lacks parent_key")
        parent = str(row["parent_key"])
        if parent in parents:
            raise ValueError(f"duplicate excluded parent_key: {parent}")
        parents.add(parent)
    return parents


def normalize_job_key(value: Any) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    raise ValueError(f"unexpected job key representation: {value!r}")


def audit_release(
    *,
    jobs: Path,
    output_root: Path,
    excluded_ledger: Path,
    epsilon_tolerance: float = 1e-6,
) -> dict[str, Any]:
    report = base_audit(
        SimpleNamespace(
            jobs=jobs,
            output_root=output_root,
            epsilon_tolerance=epsilon_tolerance,
        )
    )
    excluded = read_excluded(excluded_ledger)
    unexpected = [normalize_job_key(value) for value in report.get("unexpected_jobs", [])]
    allowed = sorted(
        key for key in unexpected
        if key[1] == "CLEAN" and key[0] in excluded
    )
    remaining = sorted(set(unexpected) - set(allowed))
    report["unexpected_jobs"] = remaining
    report["allowed_excluded_clean_jobs"] = allowed
    report["excluded_denominator_parent_count"] = len(excluded)
    report["excluded_denominator_ledger"] = str(excluded_ledger.resolve())
    complete = (
        not report.get("missing_jobs")
        and not remaining
        and int(report.get("violation_count", 0)) == 0
        and len(allowed) == len(excluded)
    )
    if len(allowed) != len(excluded):
        missing_clean = sorted(excluded - {parent for parent, _ in allowed})
        report.setdefault("violations", []).append(
            {
                "reason": "EXCLUDED_DENOMINATOR_CLEAN_ARTIFACT_MISSING",
                "parents": missing_clean,
            }
        )
        report["violation_count"] = len(report["violations"])
        complete = False
    report["gate"] = "C2G_RELEASE_MATCHED_LOAD_RUN_AUDIT"
    report["status"] = (
        "PASS_C2G_MATCHED_LOAD_RUN_AUDIT"
        if complete else "HOLD_C2G_MATCHED_LOAD_RUN_AUDIT"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--excluded-ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epsilon-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    report = audit_release(
        jobs=args.jobs.resolve(),
        output_root=args.output_root.resolve(),
        excluded_ledger=args.excluded_ledger.resolve(),
        epsilon_tolerance=args.epsilon_tolerance,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
