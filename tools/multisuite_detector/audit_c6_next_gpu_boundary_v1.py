#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

GATE = "C6_1O_NEXT_GPU_BOUNDARY_STATIC_AUDIT"
PASS = "PASS_NEXT_DYNAMIC_RESET_GATE_REQUIRES_GPU_OR_LIBERO_RUNTIME"
OUT_FILES = ["next_gpu_boundary_static_audit.json", "source_gpu_boundary_matches.csv", "checksum_report.json"]
TERMS = ["torch.cuda.is_available", "CUDA unavailable", "OffScreenRenderEnv", "load_model", "run_real", "run_offline_prompt_audit"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def scan(path):
    rows = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), start=1):
        hits = [t for t in TERMS if t in line]
        if hits:
            rows.append({"path": str(path), "line": i, "matched_terms": ";".join(hits), "text": line.strip()[:500]})
    return rows


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source_file)
    if not src.is_absolute():
        src = Path(args.repo_root) / src
    rows = scan(src) if src.exists() else []
    terms = set(";".join(r["matched_terms"] for r in rows).split(";")) if rows else set()
    needed = {"torch.cuda.is_available", "CUDA unavailable", "OffScreenRenderEnv", "load_model"}
    if not src.exists():
        status = "HOLD_SOURCE_FILE_NOT_FOUND"
    elif not needed.issubset(terms):
        status = "HOLD_GPU_BOUNDARY_EVIDENCE_INCOMPLETE"
    else:
        status = PASS
    report = {"gate": GATE, "status": status, "source_file": str(src), "matched_terms": sorted(t for t in terms if t), "interpretation": "The next non-dry-run reset/rollout boundary enters LIBERO/OpenVLA runtime paths guarded by CUDA/render/model code; stop CPU-only progression here unless explicitly authorizing runtime resources.", "boundaries": {"legacy_runner_execution": "STATIC_AUDIT_ONLY", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "next_gpu_boundary_static_audit.json", report)
    write_csv(out / "source_gpu_boundary_matches.csv", rows, ["path", "line", "matched_terms", "text"])
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-file", default="scripts/v4_run_eval_openvla.py")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
