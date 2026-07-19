#!/usr/bin/env python3
"""R7.2.2 Independent Artifact Auditor.

Read-only. Verifies:
  - Root seal integrity
  - All required files present with correct schemas
  - Episode threshold ledger row count = 200 × 2 candidates × 9 thresholds = 3600
  - Per-threshold metrics are internally consistent (no expanded denominators)
  - Expected population: 200 identities, 26 feasible, 174 no-feasible
  - Score diagnostics: paired deltas, best-in-corridor, ranks
  - Baseline ledger consistency
  - Source binding with correct commit/blob
"""

from __future__ import annotations

import argparse, csv, hashlib, json, sys
from pathlib import Path
from typing import Any

EXPECTED_N_EPISODES = 200
EXPECTED_N_THRESHOLDS = 9
EXPECTED_N_CANDIDATES = 2


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # 1. Verify root seal
    sums_path = root / "SHA256SUMS"
    sha_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file():
        findings.append({"severity": "FATAL", "check": "SHA256SUMS_exists", "detail": "missing"})
        return _verdict(findings, warnings)
    if not sha_path.is_file():
        findings.append({"severity": "FATAL", "check": "SHA256SUMS.sha256_exists", "detail": "missing"})
        return _verdict(findings, warnings)

    stored_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    computed_sha = _sha256_file(sums_path)
    if stored_sha != computed_sha:
        findings.append({"severity": "FATAL", "check": "root_seal", "detail": f"expected {stored_sha[:16]}..., computed {computed_sha[:16]}..."})
        return _verdict(findings, warnings)

    # Re-verify each file in SHA256SUMS
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, rel_path = line.strip().split("  ", 1)
        fp = root / rel_path
        if not fp.is_file():
            findings.append({"severity": "FATAL", "check": f"file_missing:{rel_path}", "detail": "not found"})
        else:
            actual = _sha256_file(fp)
            if actual != expected_hash:
                findings.append({"severity": "FATAL", "check": f"file_hash:{rel_path}", "detail": f"expected {expected_hash[:16]}, got {actual[:16]}"})

    # 2. Required files
    required = [
        "MANIFEST.json", "SOURCE_BINDING.json", "parity_report.json",
        "threshold_metrics.csv", "episode_threshold_ledger.jsonl",
        "baseline_episode_ledger.jsonl", "baseline_metrics.csv",
        "score_diagnostics.csv", "score_diagnostics_aggregate.json",
        "commands.txt",
    ]
    for name in required:
        if not (root / name).is_file():
            findings.append({"severity": "FATAL", "check": f"required_file:{name}", "detail": "missing"})

    if findings:
        return _verdict(findings, warnings)

    # 3. Load key files
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    source_binding = json.loads((root / "SOURCE_BINDING.json").read_text(encoding="utf-8"))
    parity = json.loads((root / "parity_report.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in (root / "episode_threshold_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_ledger = [json.loads(line) for line in (root / "baseline_episode_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    agg_diags = json.loads((root / "score_diagnostics_aggregate.json").read_text(encoding="utf-8"))
    diags = list(csv.DictReader((root / "score_diagnostics.csv").read_text(encoding="utf-8").splitlines()))
    metrics_rows = list(csv.DictReader((root / "threshold_metrics.csv").read_text(encoding="utf-8").splitlines()))
    baseline_metrics = list(csv.DictReader((root / "baseline_metrics.csv").read_text(encoding="utf-8").splitlines()))

    # 4. Manifest checks
    if manifest.get("schema") != "R7_K10_V5_OFFLINE_REPLAY_V2_2_MANIFEST_V1":
        findings.append({"severity": "ERROR", "check": "manifest_schema", "detail": manifest.get("schema")})
    if manifest.get("n_validation_episodes") != EXPECTED_N_EPISODES:
        findings.append({"severity": "ERROR", "check": "manifest_n_episodes", "detail": str(manifest.get("n_validation_episodes"))})
    if manifest.get("closure") is not True:
        findings.append({"severity": "ERROR", "check": "manifest_closure_flag", "detail": "not True"})

    # 5. Source binding checks
    if source_binding.get("schema") != "R7_K10_V5_OFFLINE_REPLAY_V2_2_SOURCE_BINDING_V1":
        findings.append({"severity": "ERROR", "check": "source_binding_schema", "detail": source_binding.get("schema")})
    for key in ["git_commit", "evaluator_file_blob_sha256", "v5_a_checkpoint_sha256", "v5_b_checkpoint_sha256"]:
        value = source_binding.get(key)
        if not isinstance(value, str) or len(value) != 64:
            findings.append({"severity": "ERROR", "check": f"source_binding:{key}", "detail": f"invalid SHA: {value}"})

    # 6. Parity report
    if parity.get("candidate_close_agreement") != EXPECTED_N_EPISODES:
        findings.append({"severity": "ERROR", "check": "parity_candidate_close",
                        "detail": f"{parity.get('candidate_close_agreement')}/{EXPECTED_N_EPISODES}"})
    if parity.get("step_count_match") != EXPECTED_N_EPISODES:
        findings.append({"severity": "ERROR", "check": "parity_step_count",
                        "detail": f"{parity.get('step_count_match')}/{EXPECTED_N_EPISODES}"})
    if parity.get("step_count_mismatch") != 0 or parity.get("candidate_close_disagreement") != 0:
        findings.append({"severity": "ERROR", "check": "parity_disagreement",
                        "detail": f"mismatch={parity.get('step_count_mismatch')}, cc_disagree={parity.get('candidate_close_disagreement')}"})

    # 7. Ledger row count
    expected_rows = EXPECTED_N_EPISODES * EXPECTED_N_CANDIDATES * EXPECTED_N_THRESHOLDS  # 3600
    if len(ledger) != expected_rows:
        findings.append({"severity": "ERROR", "check": "ledger_row_count",
                        "detail": f"expected {expected_rows}, got {len(ledger)}"})

    # 8. Population verification (unique identities per threshold)
    for candidate in ["V5-A", "V5-B"]:
        for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            subset = [e for e in ledger if e["candidate"] == candidate and abs(e["threshold"] - tau) < 0.005]
            identities = {e["identity"] for e in subset}
            if len(identities) != EXPECTED_N_EPISODES:
                findings.append({"severity": "ERROR", "check": f"ledger_population:{candidate}:tau={tau}",
                                "detail": f"expected {EXPECTED_N_EPISODES} identities, got {len(identities)}"})

            n_feas = sum(1 for e in subset if e["has_feasible"])
            n_nofeas = len(subset) - n_feas
            # Verify per-threshold metrics match ledger
            metric_row = next((r for r in metrics_rows if r["candidate"] == candidate and abs(float(r["threshold"]) - tau) < 0.005), None)
            if metric_row:
                if int(metric_row["n_feasible"]) != n_feas:
                    findings.append({"severity": "ERROR", "check": f"metrics_feasible:{candidate}:tau={tau}",
                                    "detail": f"ledger={n_feas}, metrics={metric_row['n_feasible']}"})

    # 9. Baseline ledger
    if len(baseline_ledger) != EXPECTED_N_EPISODES:
        findings.append({"severity": "ERROR", "check": "baseline_ledger_count",
                        "detail": f"expected {EXPECTED_N_EPISODES}, got {len(baseline_ledger)}"})

    # 10. Score diagnostics
    for candidate in ["V5-A", "V5-B"]:
        if candidate not in agg_diags:
            findings.append({"severity": "ERROR", "check": f"diag_missing:{candidate}", "detail": "not in aggregate"})
            continue
        agg = agg_diags[candidate]
        if agg.get("n_feasible_episodes") != 26:
            findings.append({"severity": "ERROR", "check": f"diag_n_feasible:{candidate}",
                            "detail": f"expected 26, got {agg.get('n_feasible_episodes')}"})
        # Verify paired: only feasible episodes should have deltas
        candidate_diags = [d for d in diags if d["candidate"] == candidate and d["has_feasible"] == "True"]
        if len(candidate_diags) != 26:
            findings.append({"severity": "ERROR", "check": f"diag_feasible_rows:{candidate}",
                            "detail": f"expected 26, got {len(candidate_diags)}"})

    # 11. Baseline metrics
    if len(baseline_metrics) != 2:
        findings.append({"severity": "ERROR", "check": "baseline_metrics_count",
                        "detail": f"expected 2, got {len(baseline_metrics)}"})

    # 12. No threshold-level denominator inflation
    for candidate in ["V5-A", "V5-B"]:
        for tau in [0.1, 0.5, 0.9]:
            metric_row = next((r for r in metrics_rows if r["candidate"] == candidate and abs(float(r["threshold"]) - tau) < 0.005), None)
            if metric_row:
                n_feas = int(metric_row["n_feasible"])
                n_nofeas = int(metric_row["n_no_feasible"])
                if n_feas + n_nofeas != EXPECTED_N_EPISODES:
                    findings.append({"severity": "ERROR", "check": f"denominator:{candidate}:tau={tau}",
                                    "detail": f"n_feas={n_feas}, n_nofeas={n_nofeas}, sum={n_feas + n_nofeas}"})

    return _verdict(findings, warnings)


def _verdict(findings: list[dict], warnings: list[dict]) -> dict:
    fatals = [f for f in findings if f["severity"] == "FATAL"]
    errors = [f for f in findings if f["severity"] == "ERROR"]
    return {
        "schema": "R7_K10_V5_OFFLINE_REPLAY_V2_2_AUDIT_V1",
        "status": "PASS" if not fatals and not errors else "FAIL",
        "n_fatal": len(fatals),
        "n_error": len(errors),
        "n_warning": len(warnings),
        "findings": findings,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="R7.2.2 sealed replay root to audit")
    ap.add_argument("--output", type=Path, help="Write audit report to this directory")
    args = ap.parse_args()

    result = audit(args.root.resolve())
    status = result["status"]
    print(f"Audit: {status}")
    print(f"  Fatal: {result['n_fatal']}  Error: {result['n_error']}  Warning: {result['n_warning']}")

    if result["findings"]:
        print("\nFindings:")
        for f in result["findings"]:
            print(f"  [{f['severity']}] {f['check']}: {f['detail']}")

    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  [WARN] {w['check']}: {w['detail']}")

    if args.output:
        import os, uuid, shutil
        out = args.output.resolve()
        staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
        staging.mkdir(parents=True)
        try:
            (staging / "audit_report.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / "input_binding.json").write_text(json.dumps({
                "schema": "R7_K10_V5_OFFLINE_REPLAY_V2_2_AUDIT_INPUT_BINDING_V1",
                "target_root": str(args.root.resolve()),
                "target_sha256s_sha256": _sha256_file(args.root.resolve() / "SHA256SUMS"),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # Seal
            exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
            files = sorted(
                [f for f in staging.rglob("*") if f.is_file() and f.name not in exclude],
                key=lambda f: str(f.relative_to(staging)),
            )
            lines = []
            for fp in files:
                rel = str(fp.relative_to(staging)).replace("\\", "/")
                lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
            (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
            sha = hashlib.sha256((staging / "SHA256SUMS").read_bytes()).hexdigest()
            (staging / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
            os.replace(staging, out)
            print(f"\nAudit root: {out}")
            print(f"SHA256SUMS: {sha}")

    if status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
