#!/usr/bin/env python3
"""Detached-safe RB1A diagnostic runner for already-exposed identities."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable).resolve()
MODES = ("CLEAN_QUALIFICATION", "COUNTERFACTUAL_CLEAN_PREFIX")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _identity_dir(key: str) -> str:
    return key.replace("/", "__")


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(cwd), env=dict(env), stdout=stdout, stderr=stderr, check=False)
    return int(completed.returncode)


def _pair_audit(*, protocol: Path, left_receipt: Path, right_receipt: Path, left_root: Path, right_root: Path, output: Path, env: Mapping[str, str]) -> None:
    command = [
        str(PYTHON), str(REPO_ROOT / "scripts/detector_v5/stage_v_rb1_runtime_equivalence.py"),
        "--protocol", str(protocol), "--left-receipt", str(left_receipt), "--right-receipt", str(right_receipt),
        "--left-root", str(left_root), "--right-root", str(right_root), "--pair-kind", "RB1A_CLEAN_PATH",
    ]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=dict(env), capture_output=True, text=True, check=False)
    output.write_text(completed.stdout or "", encoding="utf-8")
    output.with_suffix(".stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"RB1A_PAIR_AUDIT_FAIL:{completed.returncode}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("RB1A_PAIR_AUDIT_JSON_INVALID") from exc
    if result.get("verdict") != "PASS":
        raise RuntimeError("RB1A_PAIR_AUDIT_NOT_PASS")


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest = _load(root / "RB1_DIAGNOSTIC_MANIFEST.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise RuntimeError("RB1A_ROOT_NOT_FRESH_OR_ALREADY_CONSUMED")
    if manifest.get("formal_parent_promotion_authorized") is True or manifest.get("new_science_rollouts_authorized") is True:
        raise RuntimeError("RB1A_DIAGNOSTIC_BOUNDARY_INVALID")
    identities = [str(item) for item in manifest.get("diagnostic_identities", [])]
    if len(identities) != 8 or len(set(identities)) != 8:
        raise RuntimeError("RB1A_EXPECTED_EIGHT_UNIQUE_EXCLUDED_IDENTITIES")
    protocol = Path(str(manifest["protocol"])).resolve()
    if not protocol.is_file():
        raise RuntimeError("RB1_PROTOCOL_MISSING")
    runner = REPO_ROOT / "scripts/detector_v5/run_stage_v_canonical_clean.py"
    core = REPO_ROOT / "src/gripper_attack/stage_v_canonical_execution_core.py"
    if _git("status", "--porcelain"):
        raise RuntimeError("RB1A_SOURCE_WORKTREE_DIRTY")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("RB1A_SOURCE_BINDING_MISMATCH")
    if manifest.get("core_sha256") != _sha256(core) or manifest.get("runner_sha256") != _sha256(runner):
        raise RuntimeError("RB1A_RUNNER_OR_CORE_SHA256_MISMATCH")
    if manifest.get("protocol_sha256") != _sha256(protocol):
        raise RuntimeError("RB1A_PROTOCOL_SHA256_MISMATCH")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    status_path = root / "RB1A_STATUS.json"
    status: dict[str, Any] = {
        "schema": "STAGE_V_RB1A_DIAGNOSTIC_STATUS_V1", "status": "RUNNING", "started_utc": _now(),
        "root": str(root), "source_commit": manifest.get("source_commit"), "source_tree": manifest.get("source_tree"),
        "gpu": int(args.gpu), "planned_identities": identities, "completed_identities": [],
        "planned_modes_per_identity": list(MODES), "completed_mode_count": 0,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    }
    _write(status_path, status)
    candidate_dir = root / "candidates"
    contract_dir = root / "contracts"
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_rb1_receipt.py"
    python_path = str(PYTHON)
    try:
        for identity in identities:
            suite, task, state = identity.split("/")
            task_index = int(task.split("_")[1])
            state_index = int(state.split("_")[1])
            candidate = candidate_dir / f"{suite}__task_{task_index:02d}__state_{state_index:02d}.json"
            contract = contract_dir / f"{suite}.json"
            if not candidate.is_file() or not contract.is_file():
                raise RuntimeError(f"RB1A_INPUT_MISSING:{identity}")
            candidate_root = root / "runs" / _identity_dir(identity)
            receipts: dict[str, Path] = {}
            for mode in MODES:
                output_dir = candidate_root / mode
                if output_dir.exists():
                    raise RuntimeError(f"RB1A_OUTPUT_ALREADY_EXISTS:{output_dir}")
                command = [
                    python_path, str(runner), "--candidate", str(candidate), "--contract", str(contract),
                    "--output-dir", str(output_dir), "--official-snapshot-root", str(args.official_snapshot_root),
                    "--upstream-root", str(args.upstream_root), "--model-path", str(manifest["models"][suite]["path"]),
                    "--suite", suite, "--gpu", str(args.gpu), "--seed", "7", "--mode", mode,
                    "--source-commit", str(manifest["source_commit"]), "--source-tree", str(manifest["source_tree"]),
                    "--enable-runtime",
                ]
                run_log = output_dir.parent / f"{mode}.stdout.log"
                err_log = output_dir.parent / f"{mode}.stderr.log"
                code = _run(command, cwd=REPO_ROOT, env=env, stdout_path=run_log, stderr_path=err_log)
                if code != 0:
                    raise RuntimeError(f"RB1A_RUNNER_FAIL:{identity}:{mode}:{code}")
                producer = output_dir / "RB1_PRODUCER_RECEIPT.json"
                audited = output_dir / "RB1_INDEPENDENT_RECEIPT.json"
                audit_command = [
                    python_path, str(auditor), "--protocol", str(protocol), "--receipt", str(producer),
                    "--artifact-root", str(output_dir / "trace"), "--core", str(core), "--output", str(audited),
                    "--repo", str(REPO_ROOT),
                ]
                audit_log = output_dir.parent / f"{mode}.audit.stdout.log"
                audit_err = output_dir.parent / f"{mode}.audit.stderr.log"
                audit_code = _run(audit_command, cwd=REPO_ROOT, env=env, stdout_path=audit_log, stderr_path=audit_err)
                if audit_code != 0 or not audited.is_file():
                    raise RuntimeError(f"RB1A_INDEPENDENT_AUDIT_FAIL:{identity}:{mode}:{audit_code}")
                receipts[mode] = audited
                status["completed_mode_count"] = int(status["completed_mode_count"]) + 1
                _write(status_path, status)
            pair_output = candidate_root / "RB1A_PAIR_AUDIT.json"
            _pair_audit(
                protocol=protocol, left_receipt=receipts["CLEAN_QUALIFICATION"], right_receipt=receipts["COUNTERFACTUAL_CLEAN_PREFIX"],
                left_root=candidate_root / "CLEAN_QUALIFICATION" / "trace", right_root=candidate_root / "COUNTERFACTUAL_CLEAN_PREFIX" / "trace",
                output=pair_output, env=env,
            )
            status["completed_identities"].append(identity)
            _write(status_path, status)
        status.update({"status": "PASS", "ended_utc": _now(), "pair_audit_count": len(identities)})
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
    args = parser.parse_args(argv)
    root = args.root.resolve()
    lock_path = root / ".RB1A.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"verdict": "HOLD", "reason": "RB1A_DUPLICATE_LOCK"}, sort_keys=True))
            return 2
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
