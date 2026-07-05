#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

GATE = "C6_PYTHON39_PLUS_INTERPRETER_FINDER"
PASS = "PASS_PYTHON39_PLUS_INTERPRETER_FOUND"
OUT_FILES = ["python39_plus_interpreter_finder.json", "checksum_report.json"]


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


def version_for(exe):
    try:
        proc = subprocess.run([exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"], text=True, capture_output=True, check=False)
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "", proc.stderr.strip() or proc.stdout.strip()
    return proc.stdout.strip(), ""


def vtuple(text):
    try:
        xs = [int(x) for x in str(text).split(".")[:2]]
    except Exception:
        xs = [0, 0]
    while len(xs) < 2:
        xs.append(0)
    return tuple(xs[:2])


def candidates(extra):
    names = []
    if os.environ.get("CONDA_PREFIX"):
        names.append(str(Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"))
    names += list(extra or [])
    names += ["python3.12", "python3.11", "python3.10", "python3.9", "python"]
    out = []
    seen = set()
    for name in names:
        resolved = shutil.which(name) or (name if Path(name).exists() else "")
        if resolved and resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    sums = out / "SHA256SUMS"
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = ""
    for exe in candidates(args.candidate):
        ver, err = version_for(exe)
        ok = (not err) and vtuple(ver) >= (3, 9)
        rows.append({"path": exe, "version": ver, "error": err, "is_python39_plus": ok})
        if ok and not selected:
            selected = exe
    status = PASS if selected else "HOLD_NO_PYTHON39_PLUS_INTERPRETER_FOUND"
    report = {"gate": GATE, "status": status, "selected_python": selected, "candidates": rows, "boundaries": {"legacy_runner_execution": "NOT_PERFORMED", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED"}, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "python39_plus_interpreter_finder.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
