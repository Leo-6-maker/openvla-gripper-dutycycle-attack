#!/usr/bin/env python3
"""Run the four fresh-process M1 repeatability jobs for one exposed identity."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
RUNS = (("Q1", "CLEAN_QUALIFICATION", "rep_01"), ("C1", "COUNTERFACTUAL_CLEAN_PREFIX", "rep_01"), ("Q2", "CLEAN_QUALIFICATION", "rep_02"), ("C2", "COUNTERFACTUAL_CLEAN_PREFIX", "rep_02"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _run(command: list[str], cwd: Path, env: Mapping[str, str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        return int(subprocess.run(command, cwd=str(cwd), env=dict(env), stdout=stdout, stderr=stderr, check=False).returncode)


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest = _load(root / "M1_MANIFEST.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise RuntimeError("M1_ROOT_NOT_FRESH")
    if manifest.get("diagnostic_identity") in (None, "PENDING"):
        raise RuntimeError("M1_DIAGNOSTIC_IDENTITY_UNBOUND")
    if manifest.get("new_science_rollouts_authorized") is not False or manifest.get("formal_parent_promotion_authorized") is not False:
        raise RuntimeError("M1_BOUNDARY_INVALID")
    if sys.prefix != "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800":
        raise RuntimeError(f"M1_PYTHON_PREFIX_MISMATCH:{sys.prefix}")
    if _git("status", "--porcelain"):
        raise RuntimeError("M1_SOURCE_WORKTREE_DIRTY")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("M1_SOURCE_BINDING_MISMATCH")
    identity = str(manifest["diagnostic_identity"])
    suite, task, state = identity.split("/")
    task_index = int(task.split("_")[1])
    state_index = int(state.split("_")[1])
    candidate = root / "candidates" / f"{suite}__task_{task_index:02d}__state_{state_index:02d}.json"
    contract = root / "contracts" / f"{suite}.json"
    if not candidate.is_file() or not contract.is_file():
        raise RuntimeError("M1_INPUT_BINDING_MISSING")
    status_path = root / "M1_STATUS.json"
    status = _load(status_path)
    if args.run_set == "r2" and args.raw_capture_plan is None:
        raise RuntimeError("M1_R2_RAW_CAPTURE_PLAN_REQUIRED")
    run_base = root / ("runs" if args.run_set == "r1" else "raw_runs")
    status.update({"status": "RUNNING_R1_REPEATABILITY" if args.run_set == "r1" else "RUNNING_R2_RAW_CAPTURE", "phase": "M1-R1" if args.run_set == "r1" else "M1-R2", "run_set": args.run_set, "started_utc": _now(), "current_identity": identity, "python_executable": str(PYTHON), "python_prefix": sys.prefix, "completed_runs": [], "completed_audits": []})
    _write(status_path, status)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "7"})
    runner = REPO_ROOT / "scripts/detector_v5/run_stage_v_canonical_clean.py"
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_rb1_receipt.py"
    try:
        for label, mode, replicate in RUNS:
            output_dir = run_base / identity.replace("/", "__") / mode / replicate
            if output_dir.exists():
                raise RuntimeError(f"M1_OUTPUT_ALREADY_EXISTS:{output_dir}")
            command = [
                str(PYTHON), str(runner), "--candidate", str(candidate), "--contract", str(contract), "--output-dir", str(output_dir),
                "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root),
                "--model-path", str(manifest["models"][suite]["path"]), "--suite", suite, "--gpu", str(args.gpu), "--seed", "7", "--mode", mode,
                "--source-commit", str(manifest["source_commit"]), "--source-tree", str(manifest["source_tree"]), "--enable-runtime",
            ]
            if args.raw_capture_plan is not None:
                command.extend(["--raw-capture-plan", str(args.raw_capture_plan.resolve())])
            code = _run(command, REPO_ROOT, env, output_dir.parent / f"{label}.stdout.log", output_dir.parent / f"{label}.stderr.log")
            if code != 0:
                raise RuntimeError(f"M1_RUN_FAIL:{label}:{code}")
            status["completed_runs"].append(label)
            _write(status_path, status)
            producer = output_dir / "RB1_PRODUCER_RECEIPT.json"
            audited = output_dir / "RB1_INDEPENDENT_RECEIPT.json"
            audit_cmd = [str(PYTHON), str(auditor), "--protocol", str(REPO_ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json"), "--receipt", str(producer), "--artifact-root", str(output_dir / "trace"), "--core", str(REPO_ROOT / "src/gripper_attack/stage_v_canonical_execution_core.py"), "--output", str(audited), "--repo", str(REPO_ROOT)]
            audit_code = _run(audit_cmd, REPO_ROOT, env, output_dir.parent / f"{label}.audit.stdout.log", output_dir.parent / f"{label}.audit.stderr.log")
            if audit_code != 0 or not audited.is_file():
                raise RuntimeError(f"M1_AUDIT_FAIL:{label}:{audit_code}")
            status["completed_audits"].append(label)
            _write(status_path, status)
        status.update({"r1_runs_complete": True, "ended_utc": _now(), "current_identity": None})
        _write(status_path, status)
        return 0
    except Exception as exc:
        status.update({"status": "HOLD", "ended_utc": _now(), "failure": f"{type(exc).__name__}:{str(exc)[:1000]}"})
        _write(status_path, status)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--raw-capture-plan", type=Path)
    parser.add_argument("--run-set", choices=["r1", "r2"], default="r1")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".M1.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"verdict": "HOLD", "reason": "M1_DUPLICATE_LOCK"}, sort_keys=True))
            return 2
        try:
            return run(args)
        except (OSError, KeyError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
