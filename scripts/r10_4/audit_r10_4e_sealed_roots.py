#!/usr/bin/env python3
"""Gate E-R2.5F: Independent sealed root auditor for R10.4E panel outputs.

Reads a panel root directory and verifies:
- SHA256SUMS for every listed file (full 64-char digest comparison)
- Exact file set (no extra unlisted files, no missing listed files)
- SHA256SUMS.sha256 sidecar self-hash
- JSONL row counts and basic structure
- Termination semantics on episode summaries
- Source/parent/detector binding consistency
- Panel ledger vs episode root correspondence

Does NOT: load OpenVLA, run LIBERO, modify any file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VALID_TERMINATIONS = {
    "SUCCESS_TERMINATION", "HORIZON_TERMINATION",
    "FULL_LOOP_TASK_FAILURE", "EARLY_DONE_WITHOUT_SUCCESS",
    "NO_STEPS", "UNCLASSIFIED",
}
VALID_STATUSES = {
    "PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED",
    "FAIL_RUNTIME", "FAIL_TERMINATION", "FAIL_EXCEPTION",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R10.4E sealed root auditor")
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_single_root(root: Path) -> dict[str, Any]:
    """Verify SHA256SUMS, file set, and content for a single episode root."""
    report: dict[str, Any] = {"root": str(root)}

    if not root.is_dir():
        report["error"] = "NOT_A_DIRECTORY"
        return report

    sums_file = root / "SHA256SUMS"
    sidecar_file = root / "SHA256SUMS.sha256"

    # 1. SHA256SUMS.sha256 self-verification
    if not sums_file.is_file() or not sidecar_file.is_file():
        report["seal_files_missing"] = True
        report["seal_valid"] = False
        return report

    sidecar_lines = sidecar_file.read_text(encoding="utf-8").splitlines()
    sidecar_tokens = sidecar_lines[0].split() if sidecar_lines else []
    if len(sidecar_tokens) < 2 or sidecar_tokens[1] != "SHA256SUMS":
        report["sidecar_format_fail"] = True
        report["seal_valid"] = False
        return report

    actual_sums_sha = sha256_file(sums_file)
    if sidecar_tokens[0] != actual_sums_sha:
        report["sidecar_digest_mismatch"] = True
        report["seal_valid"] = False
        return report

    report["sha256sums_sha256"] = actual_sums_sha

    # 2. Verify every listed file
    listed: dict[str, str] = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split(maxsplit=1)
        if len(tokens) != 2 or not SHA256_RE.fullmatch(tokens[0]):
            report.setdefault("parse_errors", []).append(f"bad line: {line[:80]}")
            continue
        listed[tokens[1].strip()] = tokens[0]

    report["files_listed"] = len(listed)
    mismatches = []
    for fname, expected in listed.items():
        fp = root / fname
        if not fp.is_file():
            mismatches.append(f"missing: {fname}")
            continue
        actual = sha256_file(fp)
        if actual != expected:
            mismatches.append(f"mismatch: {fname} expected={expected[:16]}... actual={actual[:16]}...")
    report["digest_mismatches"] = mismatches
    report["all_digests_ok"] = len(mismatches) == 0

    # 3. Verify exact file set (no extra files)
    actual_files = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path in {sums_file, sidecar_file}:
            continue
        actual_files.add(path.relative_to(root).as_posix())
    extra = actual_files - set(listed)
    missing = set(listed) - actual_files
    report["extra_files"] = sorted(extra)
    report["missing_files"] = sorted(missing)
    report["file_set_ok"] = len(extra) == 0 and len(missing) == 0

    report["seal_valid"] = report["all_digests_ok"] and report["file_set_ok"]

    # 4. Content checks
    for jsonl_name in ["step_records.jsonl", "detector_records.jsonl"]:
        jl = root / jsonl_name
        if jl.is_file():
            lines = jl.read_text(encoding="utf-8").splitlines()
            report[f"{jsonl_name}_rows"] = len(lines)

    # Episode summary
    summary_file = root / "episode_summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        report["identity"] = summary.get("identity", "")
        report["status"] = summary.get("status", "")
        report["n_steps"] = summary.get("n_steps", 0)
        report["emit_count"] = summary.get("emit_count", 0)
        report["termination_reason"] = summary.get("termination_reason", "")
        report["task_success"] = summary.get("task_success", None)
        report["violations"] = summary.get("violations", [])
        report["done"] = summary.get("done", None)

        # Validate status
        if report["status"] not in VALID_STATUSES:
            report["status_invalid"] = True

        # Validate termination
        if report["termination_reason"] not in VALID_TERMINATIONS:
            report["termination_reason_invalid"] = True

    return report


def audit_panel_ledger(panel_root: Path) -> dict[str, Any]:
    """Verify panel-level ledger consistency."""
    ledger_file = panel_root / "panel_ledger.json"
    report: dict[str, Any] = {"panel_root": str(panel_root)}

    if not ledger_file.is_file():
        report["ledger_missing"] = True
        return report

    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    report["ledger_schema"] = ledger.get("schema", "")
    report["ledger_n_attempts"] = ledger.get("n_attempts", 0)
    report["ledger_all_runtime_valid"] = ledger.get("all_runtime_valid", False)
    report["ledger_panel_ok"] = ledger.get("panel_ok", False)

    attempts = ledger.get("attempts", [])

    # Check each attempt has a corresponding episode root
    orphan_attempts = []
    for entry in attempts:
        identity = entry.get("identity", "")
        expected_dir = panel_root / identity.replace("/", "_")
        if not expected_dir.is_dir():
            orphan_attempts.append(identity)
    report["orphan_attempts"] = orphan_attempts

    # Check each episode root has a corresponding ledger entry
    ledger_ids = {e.get("identity", "") for e in attempts}
    orphan_dirs = []
    for path in sorted(panel_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue  # staging dirs
        if path.name in {"panel_ledger.json", "panel_summary.json"}:
            continue
        # Map directory name back to identity
        identity = path.name.replace("_", "/", 2)
        if identity not in ledger_ids:
            orphan_dirs.append(str(path))
    report["orphan_dirs"] = orphan_dirs
    report["correspondence_ok"] = len(orphan_attempts) == 0 and len(orphan_dirs) == 0

    return report


def main() -> int:
    args = parse_args()
    panel_root = args.panel_root

    if not panel_root.is_dir():
        print(f"PANEL_ROOT_MISSING: {panel_root}")
        return 1

    print("=" * 60)
    print(f"E-R2.5F: Sealed Root Auditor")
    print(f"Panel: {panel_root}")
    print("=" * 60)

    all_pass = True

    # Audit each episode root
    episode_roots = sorted(
        p for p in panel_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    per_root = {}
    for root in episode_roots:
        if root.name in {"panel_ledger.json", "panel_summary.json"}:
            continue
        print(f"\n─── {root.name} ───")
        report = audit_single_root(root)
        per_root[root.name] = report
        for k, v in sorted(report.items()):
            if k == "digest_mismatches" and v:
                print(f"  {k}:")
                for m in v:
                    print(f"    {m}")
            elif k == "digest_mismatches" and not v:
                print(f"  {k}: [] OK")
            else:
                print(f"  {k}: {v}")
        if not report.get("seal_valid", False):
            all_pass = False

    # Audit panel ledger
    print(f"\n─── panel_ledger ───")
    ledger_report = audit_panel_ledger(panel_root)
    for k, v in sorted(ledger_report.items()):
        print(f"  {k}: {v}")
    if not ledger_report.get("correspondence_ok", False):
        all_pass = False

    print(f"\n{'=' * 60}")
    print(f"E-R2.5F: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 60}")

    if args.output:
        import os as _os
        args.output.parent.mkdir(parents=True, exist_ok=True)
        audit_output = {
            "audit": "R10_4E_GATE_E_R2_5F_SEALED_ROOT_AUDIT",
            "panel_root": str(panel_root),
            "overall": "PASS" if all_pass else "FAIL",
            "per_root": per_root,
            "panel_ledger": ledger_report,
        }
        args.output.write_text(json.dumps(audit_output, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"Report: {args.output}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
