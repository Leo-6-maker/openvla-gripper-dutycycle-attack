#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GATE = "C6_LIBERO_OFFICIAL_A800_ENV_VALIDATION"
PASS = "PASS_LIBERO_OFFICIAL_ENV_PYTHON_DRY_RUN_READY"
OUT_FILES = ["libero_official_env_validation.json", "checksum_report.json"]
MIN_PYTHON = (3, 9)
REQUIRED_IMPORTS = ["json", "yaml", "numpy", "PIL", "torch"]


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


def parse_version_tuple(text):
    try:
        vals = [int(x) for x in str(text).split(".")[:2]]
    except Exception:
        vals = [0, 0]
    while len(vals) < 2:
        vals.append(0)
    return tuple(vals[:2])


def probe(python_exe, repo_root):
    code = r'''
import importlib, json, os, sys
repo = os.environ.get("C6_REPO_ROOT", ".")
sys.path.insert(0, os.path.join(repo, "src"))
mods = {}
for name in ["json", "yaml", "numpy", "PIL", "torch"]:
    try:
        mod = importlib.import_module(name)
        mods[name] = {"ok": True, "version": str(getattr(mod, "__version__", ""))}
    except Exception as exc:
        mods[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
try:
    import gripper_attack.logging_schema as logging_schema
    repo_import = {"ok": True, "module": str(getattr(logging_schema, "__name__", ""))}
except Exception as exc:
    repo_import = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
torch_cuda_available = None
try:
    import torch
    torch_cuda_available = bool(torch.cuda.is_available())
except Exception:
    pass
print(json.dumps({
    "python_version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
    "executable": sys.executable,
    "imports": mods,
    "repo_logging_schema_import": repo_import,
    "torch_cuda_available": torch_cuda_available,
}, sort_keys=True))
'''
    env = {"C6_REPO_ROOT": str(repo_root)}
    try:
        proc = subprocess.run([python_exe, "-c", code], text=True, capture_output=True, check=False, env={**env})
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    try:
        obj = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "error": f"json parse failed: {type(exc).__name__}: {exc}", "stdout": proc.stdout, "stderr": proc.stderr}
    obj["ok"] = True
    return obj


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
    result = probe(args.python, args.repo_root)
    status = PASS
    if not result.get("ok"):
        status = "HOLD_ENV_PROBE_FAILED"
    elif parse_version_tuple(result.get("python_version", "")) < MIN_PYTHON:
        status = "HOLD_ENV_PYTHON_TOO_OLD"
    elif any(not (result.get("imports", {}).get(name, {}).get("ok")) for name in REQUIRED_IMPORTS):
        status = "HOLD_ENV_REQUIRED_IMPORT_MISSING"
    elif not result.get("repo_logging_schema_import", {}).get("ok"):
        status = "HOLD_ENV_REPO_IMPORT_FAILED"
    elif args.require_cuda and result.get("torch_cuda_available") is not True:
        status = "HOLD_ENV_CUDA_NOT_AVAILABLE"
    report = {"gate": GATE, "status": status, "python": args.python, "repo_root": str(args.repo_root), "probe": result, "require_cuda": bool(args.require_cuda), "boundaries": {"legacy_runner_execution": "NOT_PERFORMED", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED"}, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "libero_official_env_validation.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
