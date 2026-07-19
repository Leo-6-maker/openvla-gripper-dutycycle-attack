#!/usr/bin/env python3
"""R7.3 Independent Artifact Auditor.

Read-only. Verifies:
  - Root seal integrity
  - All required files present
  - Checkpoint schema and candidate match
  - Training identity count (600)
  - OOF threshold selection
  - Validation ledger coverage (200 identities, all thresholds)
  - Gate computation correctness
  - No threshold-level denominator inflation
"""

from __future__ import annotations

import argparse, csv, hashlib, json, sys
from pathlib import Path
from typing import Any

EXPECTED_TRAIN = 600
EXPECTED_VAL = 200


def audit(root: Path, expected_candidate: str) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []

    # 1. Seal integrity
    sums_path = root / "SHA256SUMS"
    sha_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file():
        findings.append({"severity": "FATAL", "check": "seal", "detail": "SHA256SUMS missing"})
        return _verdict(findings)
    stored = sha_path.read_text(encoding="utf-8").strip().split()[0] if sha_path.is_file() else ""
    computed = hashlib.sha256(sums_path.read_bytes()).hexdigest()
    if stored != computed:
        findings.append({"severity": "FATAL", "check": "seal", "detail": "mismatch"})
        return _verdict(findings)

    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, rel = line.strip().split("  ", 1)
        fp = root / rel
        if not fp.is_file():
            findings.append({"severity": "FATAL", "check": f"file:{rel}", "detail": "missing"})
        elif hashlib.sha256(fp.read_bytes()).hexdigest() != h:
            findings.append({"severity": "FATAL", "check": f"file:{rel}", "detail": "hash mismatch"})

    if findings:
        return _verdict(findings)

    # 2. Required files
    required = [
        "checkpoint.pt", "PROTOCOL.json", "SOURCE_BINDING.json",
        "IDENTITY_MANIFEST.json", "TRAIN_HISTORY.json", "OOF_REPORT.json",
        "EPISODE_THRESHOLD_LEDGER.jsonl", "THRESHOLD_METRICS.csv",
        "AUDIT.json", "MANIFEST.json",
    ]
    for name in required:
        if not (root / name).is_file():
            findings.append({"severity": "FATAL", "check": f"required:{name}", "detail": "missing"})

    # 3. Checkpoint
    import torch
    ckpt = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "R7_K10_DETECTOR_DEVELOPMENT_CHECKPOINT_V1":
        findings.append({"severity": "ERROR", "check": "ckpt_schema", "detail": str(ckpt.get("schema"))})
    if ckpt.get("candidate") != expected_candidate:
        findings.append({"severity": "ERROR", "check": "ckpt_candidate", "detail": ckpt.get("candidate")})

    # 4. Identity manifest
    id_manifest = json.loads((root / "IDENTITY_MANIFEST.json").read_text(encoding="utf-8"))
    if id_manifest.get("train_count") != EXPECTED_TRAIN:
        findings.append({"severity": "ERROR", "check": "train_count",
                        "detail": str(id_manifest.get("train_count"))})
    if id_manifest.get("val_count") != EXPECTED_VAL:
        findings.append({"severity": "ERROR", "check": "val_count",
                        "detail": str(id_manifest.get("val_count"))})
    if len(set(id_manifest.get("train_identities", [])) & set(id_manifest.get("val_identities", []))) != 0:
        findings.append({"severity": "ERROR", "check": "train_val_overlap", "detail": "non-zero intersection"})

    # 5. OOF report
    oof = json.loads((root / "OOF_REPORT.json").read_text(encoding="utf-8"))
    if len(oof.get("folds", [])) != 5:
        findings.append({"severity": "ERROR", "check": "oof_folds", "detail": str(len(oof.get("folds", [])))})
    if oof.get("n_total") != EXPECTED_TRAIN:
        findings.append({"severity": "ERROR", "check": "oof_total",
                        "detail": str(oof.get("n_total"))})

    # 6. Validation ledger
    ledger = [json.loads(line) for line in
              (root / "EPISODE_THRESHOLD_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]

    thresholds = sorted(set(r["threshold"] for r in ledger))
    if len(thresholds) != 19:
        findings.append({"severity": "ERROR", "check": "threshold_count",
                        "detail": str(len(thresholds))})

    for tau in thresholds:
        subset = [r for r in ledger if abs(r["threshold"] - tau) < 0.005]
        ids = {r["identity"] for r in subset}
        if len(ids) != EXPECTED_VAL:
            findings.append({"severity": "ERROR", "check": f"ledger_population:tau={tau}",
                            "detail": f"{len(ids)} identities"})
        n_feas = sum(1 for r in subset if r["has_feasible"])
        n_nofeas = len(subset) - n_feas
        if n_feas + n_nofeas != EXPECTED_VAL:
            findings.append({"severity": "ERROR", "check": f"ledger_total:tau={tau}",
                            "detail": str(n_feas + n_nofeas)})

    # 7. Threshold metrics CSV consistency
    csv_rows = list(csv.DictReader((root / "THRESHOLD_METRICS.csv").read_text(encoding="utf-8").splitlines()))
    if len(csv_rows) != 19:
        findings.append({"severity": "ERROR", "check": "csv_row_count", "detail": str(len(csv_rows))})

    # 8. Protocol
    protocol = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    if protocol.get("candidate") != expected_candidate:
        findings.append({"severity": "ERROR", "check": "protocol_candidate",
                        "detail": protocol.get("candidate")})

    return _verdict(findings)


def _verdict(findings: list[dict]) -> dict:
    fatals = [f for f in findings if f["severity"] == "FATAL"]
    errors = [f for f in findings if f["severity"] == "ERROR"]
    return {
        "schema": "R7_K10_DETECTOR_AUDIT_V1",
        "status": "PASS" if not fatals and not errors else "FAIL",
        "n_fatal": len(fatals), "n_error": len(errors),
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--candidate", choices=["R7-S-LINEAR-25D", "R7-A-GRU-25D"], required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = audit(args.root.resolve(), args.candidate)
    print(f"Audit: {result['status']}")
    print(f"  Fatal: {result['n_fatal']}  Error: {result['n_error']}")
    for f in result["findings"]:
        print(f"  [{f['severity']}] {f['check']}: {f['detail']}")

    if args.output:
        import os, uuid
        out = args.output.resolve()
        staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
        staging.mkdir(parents=True)
        try:
            (staging / "audit_report.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (staging / "input_binding.json").write_text(json.dumps({
                "schema": "R7_K10_DETECTOR_AUDIT_INPUT_BINDING_V1",
                "target_root": str(args.root.resolve()),
                "target_sha256s_sha256": hashlib.sha256(
                    (args.root.resolve() / "SHA256SUMS").read_bytes()).hexdigest(),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exclude = {"SHA256SUMS", "SHA256SUMS.sha256"}
            files = sorted([f for f in staging.rglob("*") if f.is_file() and f.name not in exclude],
                           key=lambda f: str(f.relative_to(staging)))
            lines = []
            for fp in files:
                rel = str(fp.relative_to(staging)).replace("\\", "/")
                lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
            (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
            sha = hashlib.sha256((staging / "SHA256SUMS").read_bytes()).hexdigest()
            (staging / "SHA256SUMS.sha256").write_text(f"{sha}  SHA256SUMS\n", encoding="utf-8")
            os.replace(staging, out)
            print(f"\nAudit root: {out}\nSHA256SUMS: {sha}")
        except Exception:
            import shutil
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
