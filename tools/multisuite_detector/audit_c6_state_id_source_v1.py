#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

GATE = "C6_1K_STATE_ID_SOURCE_STATIC_AUDIT"
INPUT_PASS = "PASS_SHIM_DRY_RUN_STATE_ID_BOUND"
PASS_PATCHABLE = "PASS_STATIC_STATE_ID_SOURCE_PATCHABLE"
PASS_DIRECT = "PASS_STATIC_STATE_ID_SOURCE_DIRECT"
STATE_FLAGS = {"--state-id", "--state_id", "--episode-idx", "--episode_idx"}
TERMS = ["set_init_state", "initial_states", "env.reset", "set_state"]
OUT_FILES = ["state_id_source_static_audit.json", "source_matches.csv", "checksum_report.json"]


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


def accepted_args(text):
    return set(re.findall(r"add_argument\(\s*[\"'](--[^\"']+)", text))


def scan(path, terms):
    rows = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), start=1):
        hits = [t for t in terms if t in line]
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
    observed = sha256_file(args.input_c6_1j_json)
    source = Path(args.source_file)
    if not source.is_absolute():
        source = Path(args.repo_root) / source
    parent = {}
    state_id = None
    flags = []
    matches = []
    if observed != args.expected_c6_1j_sha256:
        status = "HOLD_C6_1J_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1j_json)
        parent = dict(c6.get("selected_parent") or {})
        state_id = c6.get("state_id")
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1J_STATUS_NOT_PASS"
        elif state_id is None:
            status = "HOLD_STATE_ID_MISSING"
        elif not source.exists():
            status = "HOLD_SOURCE_FILE_NOT_FOUND"
        else:
            text = source.read_text(encoding="utf-8", errors="replace")
            flags = sorted(accepted_args(text) & STATE_FLAGS)
            matches = scan(source, list(STATE_FLAGS) + TERMS)
            status = PASS_DIRECT if flags else PASS_PATCHABLE if matches else "HOLD_NO_STATE_RESET_SOURCE_ANCHOR"
    report = {"gate": GATE, "status": status, "input_c6_1j_json_sha256": observed, "expected_c6_1j_json_sha256": args.expected_c6_1j_sha256, "selected_parent": parent, "state_id": state_id, "source_file": str(source), "accepted_state_flags": flags, "source_match_count": len(matches), "boundaries": {"runtime_execution": "NOT_PERFORMED", "env_execution": "NOT_PERFORMED", "rollout": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "state_id_source_static_audit.json", report)
    write_csv(out / "source_matches.csv", matches, ["path", "line", "matched_terms", "text"])
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1j-json", required=True)
    p.add_argument("--expected-c6-1j-sha256", required=True)
    p.add_argument("--source-file", default="scripts/v4_run_eval_openvla.py")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
