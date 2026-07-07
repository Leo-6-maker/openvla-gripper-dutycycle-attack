#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

GATE = "C6_1N_LEGACY_DRY_RUN_ARTIFACT_VALIDATION"
INPUT_PASS = "PASS_LEGACY_RUNNER_STATE_IDS_DRY_RUN_RETURNS_ZERO"
PASS = "PASS_LEGACY_DRY_RUN_ARTIFACTS_VALIDATED"
OUT_FILES = ["legacy_dry_run_artifact_validation.json", "artifact_inventory.csv", "checksum_report.json"]
REQ = ["run_manifest.json", "progress.json", "summary.csv", "step_records.jsonl", "episode_records.jsonl"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def parse_ok_path(stdout):
    m = re.search(r"\[ok\]\s+v4 dry run ->\s+(.+)", stdout or "")
    return m.group(1).strip() if m else ""


def count_lines(path):
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines()) if path.exists() else 0


def inventory(root):
    rows = []
    for name in REQ:
        p = root / name
        rows.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": sha256_file(p) if p.exists() else "", "line_count": count_lines(p)})
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
    observed = sha256_file(args.input_c6_1m_json)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    status = PASS
    reason = ""
    inv = []
    manifest = {}
    progress = {}
    c6 = {}
    if observed != args.expected_c6_1m_sha256:
        status = "HOLD_C6_1M_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1m_json)
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1M_STATUS_NOT_PASS"
        else:
            if artifact_root is None:
                stdout_path = Path(args.input_c6_1m_json).parent / "stdout.txt"
                ok_path = parse_ok_path(stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else "")
                artifact_root = Path(ok_path) if ok_path else None
            if artifact_root is None or not artifact_root.exists():
                status = "HOLD_LEGACY_DRY_RUN_ARTIFACT_ROOT_NOT_FOUND"
            else:
                inv = inventory(artifact_root)
                missing = [r["path"] for r in inv if not r["exists"]]
                if missing:
                    status = "HOLD_LEGACY_DRY_RUN_ARTIFACTS_MISSING"
                    reason = ";".join(missing)
                else:
                    manifest = read_json(artifact_root / "run_manifest.json")
                    progress = read_json(artifact_root / "progress.json")
                    if progress.get("status") != "done":
                        status = "HOLD_LEGACY_DRY_RUN_PROGRESS_NOT_DONE"
                    elif str(progress.get("model_checkpoint_path")) != "dry_run":
                        status = "HOLD_LEGACY_DRY_RUN_MODEL_PATH_NOT_DRY_RUN"
                    elif count_lines(artifact_root / "step_records.jsonl") < 1 or count_lines(artifact_root / "episode_records.jsonl") < 1:
                        status = "HOLD_LEGACY_DRY_RUN_RECORDS_EMPTY"
    report = {"gate": GATE, "status": status, "reason": reason, "input_c6_1m_json_sha256": observed, "expected_c6_1m_json_sha256": args.expected_c6_1m_sha256, "artifact_root": str(artifact_root) if artifact_root is not None else "", "legacy_runner_state_id": c6.get("state_id"), "progress_status": progress.get("status"), "progress_model_checkpoint_path": progress.get("model_checkpoint_path"), "manifest_status": manifest.get("status"), "boundaries": {"legacy_runner_execution": "DRY_RUN_ONLY", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "legacy_dry_run_artifact_validation.json", report)
    write_csv(out / "artifact_inventory.csv", inv, ["path", "exists", "size_bytes", "sha256", "line_count"])
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1m-json", required=True)
    p.add_argument("--expected-c6-1m-sha256", required=True)
    p.add_argument("--artifact-root", default="")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
