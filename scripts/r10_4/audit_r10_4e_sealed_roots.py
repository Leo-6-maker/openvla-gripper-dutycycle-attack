#!/usr/bin/env python3
"""Gate E-R2.5F: Independent sealed root auditor for R10.4E panel outputs.

P0-10: Distinguishes reuse (REUSE_BINDING.json) from fresh episode roots.
P0-11: Reads identity from episode_summary.json, not directory name.
P0-12: Fails on content issues, not just seal — missing summaries, bad
JSONL, row-count mismatches, illegal statuses all cause audit FAIL.

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
    "CHECK_SUCCESS_FAILURE", "NO_STEPS", "UNCLASSIFIED",
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


def _is_reuse_dir(root: Path) -> bool:
    return (root / "REUSE_BINDING.json").is_file()


def audit_reuse_root(root: Path) -> dict[str, Any]:
    """Audit a REUSE directory (task00)."""
    report: dict[str, Any] = {"root": str(root), "type": "reuse", "checks": []}

    binding_file = root / "REUSE_BINDING.json"
    if not binding_file.is_file():
        report["checks"].append(("REUSE_BINDING_MISSING", False))
        report["valid"] = False
        return report

    try:
        binding = json.loads(binding_file.read_text(encoding="utf-8"))
        report["identity"] = binding.get("identity", "")
        report["external_root"] = binding.get("external_root", "")
        report["checks"].append(("BINDING_PARSE", True))
    except Exception as e:
        report["checks"].append(("BINDING_PARSE", False, str(e)[:200]))
        report["valid"] = False
        return report

    # Verify SHA256SUMS seal on reuse dir
    sums_file = root / "SHA256SUMS"
    sidecar_file = root / "SHA256SUMS.sha256"
    if sums_file.is_file() and sidecar_file.is_file():
        sidecar_lines = sidecar_file.read_text(encoding="utf-8").splitlines()
        sidecar_tokens = sidecar_lines[0].split() if sidecar_lines else []
        if len(sidecar_tokens) >= 2 and sidecar_tokens[1] == "SHA256SUMS":
            actual = sha256_file(sums_file)
            seal_ok = sidecar_tokens[0] == actual
            report["checks"].append(("SEAL_SELF_HASH", seal_ok))
            report["sha256sums_sha256"] = actual
            # Verify listed files
            listed = {}
            for line in sums_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                tokens = line.split(maxsplit=1)
                if len(tokens) != 2 or not SHA256_RE.fullmatch(tokens[0]):
                    continue
                listed[tokens[1].strip()] = tokens[0]
            mismatches = []
            for fname, expected in listed.items():
                fp = root / fname
                if not fp.is_file():
                    mismatches.append(f"missing:{fname}")
                elif sha256_file(fp) != expected:
                    mismatches.append(f"digest:{fname}")
            report["checks"].append(("SEAL_DIGESTS", len(mismatches) == 0, mismatches[:5]))
        else:
            report["checks"].append(("SEAL_SIDECAR_FORMAT", False))
    else:
        report["checks"].append(("SEAL_FILES_MISSING", False))

    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def audit_fresh_root(root: Path, identity: str) -> dict[str, Any]:
    """Full audit of a fresh episode root. P0-12: fails on content issues."""
    report: dict[str, Any] = {"root": str(root), "type": "fresh", "checks": []}

    if not root.is_dir():
        report["checks"].append(("DIR_EXISTS", False))
        report["valid"] = False
        return report

    # 1. SHA256SUMS seal
    sums_file = root / "SHA256SUMS"
    sidecar_file = root / "SHA256SUMS.sha256"
    if not sums_file.is_file() or not sidecar_file.is_file():
        report["checks"].append(("SEAL_FILES_MISSING", False))
        report["valid"] = False
        return report
    sidecar_lines = sidecar_file.read_text(encoding="utf-8").splitlines()
    sidecar_tokens = sidecar_lines[0].split() if sidecar_lines else []
    if len(sidecar_tokens) < 2 or sidecar_tokens[1] != "SHA256SUMS":
        report["checks"].append(("SEAL_SIDECAR_FORMAT", False))
        report["valid"] = False
        return report
    actual_sums_sha = sha256_file(sums_file)
    if sidecar_tokens[0] != actual_sums_sha:
        report["checks"].append(("SEAL_SELF_HASH", False))
        report["valid"] = False
        return report
    report["sha256sums_sha256"] = actual_sums_sha

    # Verify every listed file
    listed: dict[str, str] = {}
    parse_errors = []
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split(maxsplit=1)
        if len(tokens) != 2 or not SHA256_RE.fullmatch(tokens[0]):
            parse_errors.append(line[:80])
            continue
        listed[tokens[1].strip()] = tokens[0]
    report["checks"].append(("SHA256SUMS_PARSE", len(parse_errors) == 0, parse_errors[:3]))
    digest_mismatches = []
    for fname, expected in listed.items():
        fp = root / fname
        if not fp.is_file():
            digest_mismatches.append(f"missing:{fname}")
        else:
            actual = sha256_file(fp)
            if actual != expected:
                digest_mismatches.append(f"{fname}: expected={expected[:16]}... actual={actual[:16]}...")
    report["checks"].append(("DIGEST_VERIFY", len(digest_mismatches) == 0, digest_mismatches[:5]))

    # Exact file set
    actual_files = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path in {sums_file, sidecar_file}:
            continue
        actual_files.add(path.relative_to(root).as_posix())
    extra = sorted(actual_files - set(listed))
    missing = sorted(set(listed) - actual_files)
    report["checks"].append(("FILE_SET_EXACT", len(extra) == 0 and len(missing) == 0,
                             f"extra={extra[:3]} missing={missing[:3]}"))

    # 2. Episode summary — P0-12: content checks
    summary_file = root / "episode_summary.json"
    if not summary_file.is_file():
        report["checks"].append(("SUMMARY_MISSING", False))
        report["valid"] = False
        return report
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        report["summary_parse"] = True
    except Exception:
        report["checks"].append(("SUMMARY_PARSE", False))
        report["valid"] = False
        return report

    # Identity from summary (P0-11), not directory name
    summary_identity = summary.get("identity", "")
    report["identity"] = summary_identity
    if identity is not None and summary_identity != identity:
        report["checks"].append(("IDENTITY_MATCH", False, f"summary={summary_identity} ledger={identity}"))

    report["status"] = summary.get("status", "")
    report["n_steps"] = summary.get("n_steps", 0)
    report["emit_count"] = summary.get("emit_count", 0)
    report["termination_reason"] = summary.get("termination_reason", "")
    report["task_success"] = summary.get("task_success", None)
    report["violations"] = summary.get("violations", [])
    report["done"] = summary.get("done", None)

    # Validate status
    report["checks"].append(("STATUS_VALID", report["status"] in VALID_STATUSES,
                             f"got={report['status']}"))
    # Validate termination
    report["checks"].append(("TERMINATION_VALID",
                             report["termination_reason"] in VALID_TERMINATIONS,
                             f"got={report['termination_reason']}"))

    # P0-12: FAIL_RUNTIME, FAIL_TERMINATION, FAIL_EXCEPTION are content failures
    is_runtime_ok = report["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}
    report["checks"].append(("STATUS_RUNTIME_VALID", is_runtime_ok,
                             f"status={report['status']}"))

    # 3. Step records
    sr_file = root / "step_records.jsonl"
    if sr_file.is_file():
        try:
            srs = [json.loads(l) for l in sr_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            report["step_records_rows"] = len(srs)
            report["checks"].append(("STEP_RECORDS_PARSE", True))
            # Row count vs n_steps
            count_ok = len(srs) == report["n_steps"]
            report["checks"].append(("STEP_COUNT_MATCH", count_ok,
                                     f"records={len(srs)} n_steps={report['n_steps']}"))
            # Action error
            max_err = max((s.get("action_max_abs_error", -1) for s in srs), default=-1)
            report["checks"].append(("ACTION_ZERO_ERROR", max_err == 0.0, f"max={max_err}"))
            # Generation passes
            all_one = all(s.get("generation_passes_per_step") == 1 for s in srs)
            report["checks"].append(("GEN_PASSES_ALL_ONE", all_one))
            # Features
            all_25d = all(len(s.get("features_25d", [])) == 25 for s in srs)
            report["checks"].append(("FEATURES_25D", all_25d))
            # Info serializable
            for s in srs:
                if "info" in s:
                    try:
                        json.dumps(s["info"], sort_keys=True, default=str)
                    except Exception:
                        report["checks"].append(("INFO_SERIALIZABLE", False, f"step={s.get('step')}"))
                        break
            else:
                report["checks"].append(("INFO_SERIALIZABLE", True))
        except Exception:
            report["checks"].append(("STEP_RECORDS_PARSE", False))

    # 4. Detector records
    dr_file = root / "detector_records.jsonl"
    if dr_file.is_file():
        try:
            drs = [json.loads(l) for l in dr_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            report["detector_records_rows"] = len(drs)
            report["checks"].append(("DETECTOR_RECORDS_PARSE", True))
            count_ok = len(drs) == report["n_steps"]
            report["checks"].append(("DETECTOR_COUNT_MATCH", count_ok,
                                     f"records={len(drs)} n_steps={report['n_steps']}"))
            emit_from_records = sum(1 for d in drs if d.get("emit"))
            emit_ok = emit_from_records == report["emit_count"]
            report["checks"].append(("EMIT_COUNT_MATCH", emit_ok,
                                     f"records={emit_from_records} summary={report['emit_count']}"))
        except Exception:
            report["checks"].append(("DETECTOR_RECORDS_PARSE", False))

    # 5. Metadata
    meta_file = root / "episode_metadata.json"
    report["checks"].append(("METADATA_EXISTS", meta_file.is_file()))

    # 6. Runtime audit
    audit_file = root / "runtime_audit.json"
    report["checks"].append(("RUNTIME_AUDIT_EXISTS", audit_file.is_file()))

    # Aggregate
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def audit_panel_ledger(panel_root: Path, per_root: dict[str, Any]) -> dict[str, Any]:
    """Verify panel ledger vs episode roots. P0-10: handles reuse entries."""
    ledger_file = panel_root / "panel_ledger.json"
    report: dict[str, Any] = {"checks": []}

    if not ledger_file.is_file():
        report["checks"].append(("LEDGER_EXISTS", False))
        report["valid"] = False
        return report

    try:
        ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
        report["ledger_parse"] = True
    except Exception:
        report["checks"].append(("LEDGER_PARSE", False))
        report["valid"] = False
        return report

    report["n_attempts"] = ledger.get("n_attempts", 0)
    report["all_runtime_valid"] = ledger.get("all_runtime_valid", False)
    report["panel_ok"] = ledger.get("panel_ok", False)
    attempts = ledger.get("attempts", [])

    # Check each attempt has a corresponding root or reuse reference
    missing_roots = []
    for entry in attempts:
        identity = entry.get("identity", "")
        is_reuse = entry.get("reuse", False)
        if is_reuse:
            # Reuse entries should have an external reference
            found = False
            for dir_name, rpt in per_root.items():
                if rpt.get("identity") == identity:
                    found = True
                    break
            if not found:
                missing_roots.append(f"reuse:{identity}")
        else:
            # Fresh entries should have a directory
            ep_name = identity.replace("/", "_")
            if ep_name not in per_root:
                missing_roots.append(f"fresh:{identity}")
    report["checks"].append(("LEDGER_ROOT_CORRESPONDENCE", len(missing_roots) == 0, missing_roots[:5]))

    # Reverse: check every audited root is in ledger
    ledger_ids = {e.get("identity", "") for e in attempts}
    orphan_roots = []
    for dir_name, rpt in per_root.items():
        rid = rpt.get("identity", "")
        if rid and rid not in ledger_ids:
            orphan_roots.append(dir_name)
    report["checks"].append(("ROOT_LEDGER_CORRESPONDENCE", len(orphan_roots) == 0, orphan_roots[:5]))

    report["valid"] = all(ok for _, ok, *_ in report["checks"])
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

    # Collect all subdirectories
    subdirs = sorted(
        p for p in panel_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    per_root: dict[str, Any] = {}
    all_valid = True

    for subdir in subdirs:
        print(f"\n─── {subdir.name} ───")
        if _is_reuse_dir(subdir):
            report = audit_reuse_root(subdir)
        else:
            report = audit_fresh_root(subdir, identity=None)
        per_root[subdir.name] = report
        valid = report.get("valid", False)
        if not valid:
            all_valid = False
        status = "PASS" if valid else "FAIL"
        for k, v in sorted(report.items()):
            if k == "checks":
                for ck, cok, *details in v:
                    extra = f" — {details[0]}" if details else ""
                    print(f"  [{('PASS' if cok else 'FAIL')}] {ck}{extra}")
            elif k in ("valid", "type", "root"):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")

    # Audit ledger correspondence
    print(f"\n─── panel_ledger ───")
    ledger_report = audit_panel_ledger(panel_root, per_root)
    per_root["_panel_ledger"] = ledger_report
    for k, v in sorted(ledger_report.items()):
        if k == "checks":
            for ck, cok, *details in v:
                extra = f" — {details[0]}" if details else ""
                print(f"  [{('PASS' if cok else 'FAIL')}] {ck}{extra}")
        else:
            print(f"  {k}: {v}")
    if not ledger_report.get("valid", False):
        all_valid = False

    # Print summary
    runtime_valid = sum(
        1 for r in per_root.values()
        if r.get("status") in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}
    )
    hard_fail = sum(
        1 for r in per_root.values()
        if r.get("status") in {"FAIL_RUNTIME", "FAIL_TERMINATION", "FAIL_EXCEPTION"}
    )
    reuse_count = sum(1 for r in per_root.values() if r.get("type") == "reuse")
    fresh_count = sum(1 for r in per_root.values() if r.get("type") == "fresh")

    print(f"\n{'=' * 60}")
    print(f"Summary: {len(per_root)-1} roots ({reuse_count} reuse, {fresh_count} fresh)")
    print(f"  runtime-valid: {runtime_valid}  hard-fail: {hard_fail}")
    print(f"E-R2.5F: {'PASS' if all_valid else 'FAIL'}")
    print(f"{'=' * 60}")

    if args.output:
        audit_output = {
            "audit": "R10_4E_GATE_E_R2_5F_SEALED_ROOT_AUDIT",
            "panel_root": str(panel_root),
            "overall": "PASS" if all_valid else "FAIL",
            "per_root": {
                k: {kk: vv for kk, vv in v.items() if kk != "checks"}
                for k, v in per_root.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit_output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
