#!/usr/bin/env python3
"""Global fail-closed dispatcher for the preregistered Stage 3A matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STAGE2_COMMIT = "f14415b104501df2a5cf7b35e8965966866eea9e"
STAGE2_TREE = "6a982a474b9fe788e8466f2a7f144b45d6a1cd89"
CHECKPOINT_SHA = "ce7f03088d84a796d38fbdc107cea7f21bdb4808e35f7dc754e1b52e48bce1d4"
PARENT_MANIFEST_SHA = "a82715cf2ca111de9f7fe24e9812a8c4166658026ab846fc6a61cb57f92e70fd"
FORMAL_SEEDS = (2026080501, 2026080502, 2026080503)
DIAGNOSTIC_SEED = 2026080499
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CONDITIONS = (
    "CLEAN_SHADOW",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW",
    "PGD_E003_S20_SHADOW",
    "PGD_E006_S20_SHADOW",
    "ORACLE_OPEN_SHADOW",
)
DIAGNOSTIC_CONDITIONS = (
    "CLEAN_SHADOW",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW",
    "PGD_E006_S20_SHADOW",
    "ORACLE_OPEN_SHADOW",
)
CONFIG_BY_CONDITION = {
    "CLEAN_SHADOW": "configs/sweep_v5_e0.06_s20.yaml",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW": "configs/sweep_v5_e0.06_s20.yaml",
    "PGD_E003_S20_SHADOW": "configs/sweep_v5_e0.03_s20.yaml",
    "PGD_E006_S20_SHADOW": "configs/sweep_v5_e0.06_s20.yaml",
    "ORACLE_OPEN_SHADOW": "configs/sweep_v5_e0.06_s20.yaml",
}
ARM_BY_CONDITION = {
    "CLEAN_SHADOW": "CLEAN",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW": "RAND_T10",
    "PGD_E003_S20_SHADOW": "TRUE_T10",
    "PGD_E006_S20_SHADOW": "TRUE_T10",
    "ORACLE_OPEN_SHADOW": "COMMAND_OPEN_ORACLE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def fail(message: str) -> None:
    raise RuntimeError(message)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def assert_safe_path(path: Path) -> Path:
    resolved = path.resolve()
    forbidden = {"eval160", "protected_eval", "protected"}
    if any(part.lower() in forbidden for part in resolved.parts):
        fail(f"forbidden evaluation path: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True, help="comma-separated physical GPU ids")
    parser.add_argument("--n4-norm-data", type=Path, required=True)
    parser.add_argument("--branch", default="codex/stage3a-shadow-only-20260805")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--r3-ensemble-root", type=Path, default=None)
    parser.add_argument("--r3-ensemble-manifest", type=Path, default=None)
    parser.add_argument("--r3-transfer-audit", type=Path, default=None)
    parser.add_argument("--r3-transfer-receipt", type=Path, default=None)
    parser.add_argument("--r3-ensemble-source-commit", default=None)
    parser.add_argument("--r3-ensemble-source-tree", default=None)
    return parser.parse_args()


def _valid_parent(row: dict[str, Any]) -> bool:
    return all(
        bool(row.get(key))
        for key in ("eligible", "clean_success", "k10_executable", "runtime_valid")
    ) and int(row.get("remaining_horizon") or 0) >= 10


def choose_tasks(parent: dict[str, Any], parent_path: Path) -> list[dict[str, Any]]:
    census = parent.get("all_fec_census")
    if not isinstance(census, dict):
        fail("parent manifest all_fec_census is not a suite mapping")
    selected: list[dict[str, Any]] = []
    for suite in SUITE_ORDER:
        rows = [row for row in census.get(suite, []) if isinstance(row, dict) and _valid_parent(row)]
        rows.sort(key=lambda row: str(row.get("canonical_parent_key", "")))
        if rows:
            row = dict(rows[0])
            artifact_root = Path(str(row["artifact_root"])).resolve(strict=True)
            if "openvla_attack_evidence" not in str(artifact_root) or "/clean/" not in str(artifact_root).replace("\\", "/"):
                fail(f"task artifact is not a frozen clean root: {artifact_root}")
            metadata_path = artifact_root / "episode_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("condition") != "CLEAN" or metadata.get("success") is not True or metadata.get("runtime_valid") is not True:
                fail(f"task metadata is not clean-success/runtime-valid: {artifact_root}")
            init_sha = str(metadata.get("initial_state_sha256", "")).lower()
            if len(init_sha) != 64:
                fail(f"task missing exact initial state SHA: {artifact_root}")
            row.update(
                {
                    "suite": suite,
                    "task_index": int(row["task_idx"]),
                    "state_index": int(row["state_id"]),
                    "canonical_parent_key": str(row["canonical_parent_key"]),
                    "initial_state_sha256": init_sha,
                    "model_path": str(metadata["model_path"]),
                    "official_horizon": int(metadata.get("official_horizon", row.get("artifact_steps", 0))),
                    "task_language": str(metadata.get("task_language", row.get("task_language", ""))),
                    "parent_manifest": str(parent_path),
                }
            )
            selected.append(row)
    if len(selected) != 4 or len({row["canonical_parent_key"] for row in selected}) != 4:
        fail(f"deterministic task selection did not produce four unique suites: {selected}")
    return selected


def config_index(repo: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import yaml

    paths = sorted(
        [*repo.glob("configs/sweep_v5_e*.yaml"), repo / "configs/fec_attack_v5_open.yaml"],
        key=lambda path: path.as_posix(),
    )
    records: list[dict[str, Any]] = []
    by_rel: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        opt = value.get("attack_optimizer", {})
        runtime = value.get("runtime", {})
        rand = value.get("arms", {}).get("RAND_T10", {})
        record = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": sha256_file(path),
            "epsilon": opt.get("epsilon"),
            "step_size": opt.get("step_size"),
            "pgd_steps": opt.get("num_steps"),
            "objective": opt.get("objective"),
            "target_token": opt.get("target_token_id"),
            "target_execution_class": opt.get("target_execution_class"),
            "burst_frames": runtime.get("attack_burst_frames"),
            "rand_matching_fields": rand.get("matched_to_TRUE", []),
            "status": value.get("status"),
            "version": value.get("version"),
        }
        records.append(record)
        by_rel[record["path"]] = record
    required = {
        "configs/sweep_v5_e0.03_s20.yaml",
        "configs/sweep_v5_e0.06_s20.yaml",
        "configs/fec_attack_v5_open.yaml",
    }
    if not required.issubset(by_rel):
        fail(f"formal config inventory missing required files: {sorted(required - set(by_rel))}")
    for rel, epsilon, step in (
        ("configs/sweep_v5_e0.03_s20.yaml", 0.03, 0.0015),
        ("configs/sweep_v5_e0.06_s20.yaml", 0.06, 0.003),
    ):
        rec = by_rel[rel]
        if (float(rec["epsilon"]), float(rec["step_size"]), int(rec["pgd_steps"])) != (epsilon, step, 20):
            fail(f"frozen config fields mismatch: {rel}")
        if rec["objective"] != "autoregressive_prefix_gripper_target_token_logratio_arm_v3" or rec["target_token"] != 31745 or rec["target_execution_class"] != "NATIVE_OPEN" or rec["burst_frames"] != 10:
            fail(f"frozen attack route binding mismatch: {rel}")
        if sorted(rec["rand_matching_fields"]) != sorted(["epsilon", "num_steps", "step_size", "random_start", "burst_frames", "surrogate_score_path", "prefix_refresh_interval", "temporal_init", "temporal_smooth_lambda"]):
            fail(f"RAND matching contract mismatch: {rel}")
    if by_rel["configs/fec_attack_v5_open.yaml"]["status"] != "EXPERIMENTAL_CANARY":
        fail("experimental canary config status changed unexpectedly")
    index = {
        "schema": "D8_STAGE3A_ATTACK_CONFIG_INDEX_V1",
        "formal_conditions": {
            condition: {
                "config_path": CONFIG_BY_CONDITION[condition],
                "arm": ARM_BY_CONDITION[condition],
                "role": "CLEAN_BASE_CONFIG" if condition == "CLEAN_SHADOW" else "EXISTING_FROZEN_SWEEP_CONFIG",
            }
            for condition in CONDITIONS
        },
        "files": records,
        "experimental_canary_excluded": "configs/fec_attack_v5_open.yaml",
    }
    return index, by_rel


def build_jobs(tasks: list[dict[str, Any]], *, diagnostic: bool, repo: Path, root: Path, checkpoint: Path, receipt: Path, n4_norm: Path, attacker_sha: str) -> list[dict[str, Any]]:
    conditions = DIAGNOSTIC_CONDITIONS if diagnostic else CONDITIONS
    seeds = (DIAGNOSTIC_SEED,) if diagnostic else FORMAL_SEEDS
    jobs: list[dict[str, Any]] = []
    index = 0
    task_rows = tasks[:1] if diagnostic else tasks
    for condition in conditions:
        for task in task_rows:
            for seed in seeds:
                episode_id = f"{task['canonical_parent_key']}|{condition}|seed{seed}"
                job_name = f"{index:03d}_{condition}__{task['suite']}_task{task['task_index']:02d}_state{task['state_index']:02d}__seed{seed}"
                job_root = root / "jobs" / job_name
                jobs.append(
                    {
                        "job_index": index,
                        "job_id": job_name,
                        "condition": condition,
                        "arm": ARM_BY_CONDITION[condition],
                        "suite": task["suite"],
                        "task_index": task["task_index"],
                        "state_index": task["state_index"],
                        "canonical_parent_key": task["canonical_parent_key"],
                        "seed": seed,
                        "episode_id": episode_id,
                        "initial_state_sha256": task["initial_state_sha256"],
                        "model_path": task["model_path"],
                        "config_path": CONFIG_BY_CONDITION[condition],
                        "job_root": str(job_root),
                        "status": "PENDING",
                        "returncode": None,
                    }
                )
                index += 1
    expected = (4 * 3 * 5) if not diagnostic else 4
    if len(jobs) != expected:
        fail(f"exact Stage3A matrix mismatch: planned={len(jobs)} expected={expected}")
    return jobs


def validate_job_artifact(job: dict[str, Any]) -> None:
    root = Path(job["job_root"])
    run_manifest = root / "run_manifest.json"
    summary = root / "smoke_summary.json"
    arm_root = root / job["arm"]
    result_path = arm_root / "result.json"
    trace_path = arm_root / "evaluation_detector_trace.json"
    complete = arm_root / "COMPLETE.json"
    if not all(path.is_file() for path in (run_manifest, summary, result_path, trace_path, complete)):
        fail(f"job artifact closure missing: {job['job_id']}")
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("scientific_role") != "STAGE3A_SHADOW_ONLY" or result.get("status") != "PASS":
        fail(f"job artifact status invalid: {job['job_id']}")
    for key, expected in (("condition", job["condition"]), ("suite", job["suite"]), ("task_index", job["task_index"]), ("seed", job["seed"])):
        if manifest.get(key) != expected and not (key == "condition" and manifest.get("stage3a_condition") == expected):
            fail(f"job identity mismatch {job['job_id']}: {key}")
    state = manifest.get("state_identity", {})
    if state.get("initial_state_sha256") != job["initial_state_sha256"]:
        fail(f"job initial-state binding mismatch: {job['job_id']}")
    if result.get("guard_intervention_count") != 0 or result.get("eval160_reads") != 0 or result.get("protected_eval_reads") != 0:
        fail(f"job crossed shadow-only boundary: {job['job_id']}")


def kill_own_processes(active: dict[int, tuple[subprocess.Popen[Any], dict[str, Any], Any]]) -> None:
    for process, _job, _log in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 5
    while time.time() < deadline and any(process.poll() is None for process, _job, _log in active.values()):
        time.sleep(0.2)
    for process, _job, _log in active.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def dispatch(jobs: list[dict[str, Any]], gpu_ids: list[int], repo: Path, output_root: Path, checkpoint: Path, receipt: Path, n4_norm: Path, attacker_sha: str, stage2_source_commit: str, stage2_source_tree: str, r3_binding: dict[str, str] | None = None) -> None:
    runner = repo / "scripts" / "fec" / "run_gpu_smoke_v5_open.py"
    active: dict[int, tuple[subprocess.Popen[Any], dict[str, Any], Any]] = {}
    pending = list(jobs)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    def write_snapshot() -> None:
        path = output_root / "STAGE3A_SHADOW_RUN_MANIFEST.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        current["jobs"] = jobs
        current["active_gpu_count"] = len(active)
        current["updated_unix"] = time.time()
        atomic_json(path, current)
    try:
        while pending or active:
            for gpu in gpu_ids:
                if not pending or gpu in active:
                    continue
                job = pending.pop(0)
                job["status"] = "RUNNING"
                job["gpu_id"] = gpu
                job["started_unix"] = time.time()
                log_path = Path(job["job_root"]).with_suffix(".dispatcher.log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("w", encoding="utf-8")
                cmd = [
                    sys.executable, str(runner), "--gpu-id", str(gpu),
                    "--suite", job["suite"], "--task-index", str(job["task_index"]),
                    "--state-index", str(job["state_index"]), "--output-root", job["job_root"],
                    "--model-path", job["model_path"], "--config", job["config_path"],
                    "--repo-root", str(repo), "--n4-module", str(repo / "scripts/fec/n4_detector_adapter_v4.py"),
                    "--n4-norm-data", str(n4_norm), "--expected-attacker-sha256", attacker_sha,
                    "--seed", str(job["seed"]), "--rand-direction-seed", str(job["seed"]),
                    "--random-time-seed", str(job["seed"] + 1),
                    "--stage3a-condition", job["condition"], "--stage3a-episode-id", job["episode_id"],
                    "--stage3a-checkpoint", str(checkpoint), "--stage3a-freeze-receipt", str(receipt),
                    "--stage3a-expected-checkpoint-sha256", CHECKPOINT_SHA,
                    "--stage3a-expected-scheduler-sha256", sha256_file(receipt),
                    "--stage3a-source-commit", stage2_source_commit, "--stage3a-source-tree", stage2_source_tree,
                    "--stage3a-expected-init-state-sha256", job["initial_state_sha256"],
                ]
                if r3_binding is not None:
                    cmd.extend([
                        "--stage3a-ensemble-root", r3_binding["ensemble_root"],
                        "--stage3a-ensemble-manifest", r3_binding["ensemble_manifest"],
                        "--stage3a-transfer-audit", r3_binding["transfer_audit"],
                        "--stage3a-transfer-receipt", r3_binding["transfer_receipt"],
                        "--stage3a-expected-ensemble-root-seal", r3_binding["ensemble_root_seal"],
                        "--stage3a-expected-ensemble-manifest-sha256", r3_binding["ensemble_manifest_sha256"],
                        "--stage3a-expected-transfer-audit-sha256", r3_binding["transfer_audit_sha256"],
                        "--stage3a-expected-transfer-receipt-sha256", r3_binding["transfer_receipt_sha256"],
                        "--stage3a-ensemble-source-commit", r3_binding["ensemble_source_commit"],
                        "--stage3a-ensemble-source-tree", r3_binding["ensemble_source_tree"],
                    ])
                process = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (process, job, log)
            write_snapshot()
            if not active:
                break
            time.sleep(2)
            for gpu, (process, job, log) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log.close()
                job["returncode"] = returncode
                job["ended_unix"] = time.time()
                if returncode != 0:
                    job["status"] = "FAILED"
                    write_snapshot()
                    fail(f"first Stage3A artifact/worker failure: {job['job_id']} rc={returncode}")
                validate_job_artifact(job)
                job["status"] = "COMPLETED"
                del active[gpu]
                write_snapshot()
    except Exception:
        for job in pending:
            job["status"] = "ABORTED_INCOMPLETE"
        for _gpu, (_process, job, _log) in active.items():
            job["status"] = "ABORTED_INCOMPLETE"
        kill_own_processes(active)
        atomic_json(output_root / "STAGE3A_ABORTED_INCOMPLETE.json", {"jobs": jobs, "reason": "fail_fast", "created_unix": time.time()})
        write_snapshot()
        raise


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve(strict=True)
    output_root = assert_safe_path(args.output_root)
    stage2_root = assert_safe_path(args.stage2_root.resolve(strict=True))
    checkpoint = assert_safe_path(args.checkpoint.resolve(strict=True))
    receipt = assert_safe_path(args.freeze_receipt.resolve(strict=True))
    parent_path = assert_safe_path(args.parent_manifest.resolve(strict=True))
    n4_norm = assert_safe_path(args.n4_norm_data.resolve(strict=True))
    r3_mode = args.r3_transfer_audit is not None
    r3_binding: dict[str, str] | None = None
    if r3_mode:
        r3_root = assert_safe_path(args.r3_ensemble_root.resolve(strict=True))
        r3_manifest = assert_safe_path(args.r3_ensemble_manifest.resolve(strict=True))
        r3_audit = assert_safe_path(args.r3_transfer_audit.resolve(strict=True))
        r3_receipt = assert_safe_path(args.r3_transfer_receipt.resolve(strict=True))
        if not all(path.is_file() for path in (r3_manifest, r3_audit, r3_receipt, r3_root / "SHA256SUMS", r3_root / "SHA256SUMS.sha256")):
            fail("R3-A detector artifacts are incomplete")
        root_seal = sha256_file(r3_root / "SHA256SUMS")
        if (r3_root / "SHA256SUMS.sha256").read_text(encoding="utf-8").strip() != f"{root_seal}  SHA256SUMS":
            fail("R3-A ensemble root sidecar mismatch")
        transfer = json.loads(r3_audit.read_text(encoding="utf-8"))
        receipt_value = json.loads(r3_receipt.read_text(encoding="utf-8"))
        if transfer.get("status") != "R3A_MATCHED_ENSEMBLE_TRANSFER_PASS" or receipt_value.get("status") != "PASS":
            fail("R3-A clean transfer is not PASS")
        if not args.r3_ensemble_source_commit or not args.r3_ensemble_source_tree:
            fail("R3-A ensemble producer source binding is missing")
        r3_binding = {
            "ensemble_root": str(r3_root), "ensemble_manifest": str(r3_manifest),
            "transfer_audit": str(r3_audit), "transfer_receipt": str(r3_receipt),
            "ensemble_root_seal": root_seal, "ensemble_manifest_sha256": sha256_file(r3_manifest),
            "transfer_audit_sha256": sha256_file(r3_audit), "transfer_receipt_sha256": sha256_file(r3_receipt),
            "ensemble_source_commit": args.r3_ensemble_source_commit, "ensemble_source_tree": args.r3_ensemble_source_tree,
        }
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        fail("inherited CUDA_VISIBLE_DEVICES must be unset/empty")
    gpu_ids = sorted({int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()})
    if not gpu_ids:
        fail("no physical GPU ids supplied")
    if output_root.exists() and any(output_root.iterdir()):
        fail(f"output root must be new and empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    head = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    # The server checkout is intentionally detached; bind the explicit remote
    # branch passed by the launch plan instead of relying on branch --show-current.
    branch = args.branch
    if git(repo, "status", "--porcelain"):
        fail("Stage3A server worktree is not clean")
    remote_ref = git(repo, "ls-remote", "origin", f"refs/heads/{branch}").split()
    if not remote_ref or remote_ref[0] != head:
        fail(f"GitHub remote branch does not contain exact Stage3A HEAD: {branch} {remote_ref}")
    if not r3_mode and (not receipt.is_file() or sha256_file(checkpoint) != CHECKPOINT_SHA):
        fail("Stage 2 R2 checkpoint binding failed")
    parent_sha = sha256_file(parent_path)
    if parent_sha != PARENT_MANIFEST_SHA:
        fail(f"frozen parent manifest SHA mismatch: {parent_sha}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    tasks = choose_tasks(parent, parent_path)
    index, config_records = config_index(repo)
    atomic_json(output_root / "STAGE3A_TASK_MANIFEST.json", {
        "schema": "D8_STAGE3A_TASK_MANIFEST_V1",
        "status": "FROZEN",
        "source_manifest": str(parent_path),
        "source_manifest_sha256": parent_sha,
        "selection_rule": "suite priority, canonical_parent_key dictionary order, clean_success+eligible+k10+runtime_valid",
        "tasks": tasks[:1] if args.diagnostic else tasks,
    })
    atomic_json(output_root / "STAGE3A_ATTACK_CONFIG_INDEX.json", index)
    attacker_path = repo / "src" / "gripper_attack" / "attack_adapter.py"
    attacker_sha = sha256_file(attacker_path)
    jobs = build_jobs(tasks, diagnostic=args.diagnostic, repo=repo, root=output_root, checkpoint=checkpoint, receipt=receipt, n4_norm=n4_norm, attacker_sha=attacker_sha)
    run_manifest = {
        "schema": "D8_STAGE3A_SHADOW_RUN_MANIFEST_V1",
        "status": "PENDING",
        "diagnostic": bool(args.diagnostic),
        "planned_jobs": len(jobs),
        "conditions": list(DIAGNOSTIC_CONDITIONS if args.diagnostic else CONDITIONS),
        "seeds": [DIAGNOSTIC_SEED] if args.diagnostic else list(FORMAL_SEEDS),
        "gpu_ids": gpu_ids,
        "max_workers_per_physical_gpu": 1,
        "runner_source_commit": head,
        "runner_source_tree": tree,
        "runner_branch": branch,
        "github_remote_ref": f"refs/heads/{branch}",
        "stage2_source_commit": STAGE2_COMMIT,
        "stage2_source_tree": STAGE2_TREE,
        "stage2_checkpoint_sha256": CHECKPOINT_SHA,
        "stage2_scheduler_freeze_sha256": sha256_file(receipt),
        "stage2_scheduler_freeze_path": str(receipt),
        "stage3a_detector_mode": "R3A_MATCHED_ENSEMBLE" if r3_mode else "R2_SINGLE_CHECKPOINT",
        "stage3a_ensemble_root_seal": None if r3_binding is None else r3_binding["ensemble_root_seal"],
        "stage3a_ensemble_manifest_sha256": None if r3_binding is None else r3_binding["ensemble_manifest_sha256"],
        "stage3a_transfer_audit_sha256": None if r3_binding is None else r3_binding["transfer_audit_sha256"],
        "stage3a_transfer_receipt_sha256": None if r3_binding is None else r3_binding["transfer_receipt_sha256"],
        "stage3a_ensemble_source_commit": None if r3_binding is None else r3_binding["ensemble_source_commit"],
        "stage3a_ensemble_source_tree": None if r3_binding is None else r3_binding["ensemble_source_tree"],
        "stage2_root": str(stage2_root),
        "n4_trigger_source": "N4 first emit",
        "evaluation_detector_role": "shadow_logging_only",
        "guard_intervention_count": 0,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "other_user_processes_terminated": False,
        "jobs": jobs,
    }
    atomic_json(output_root / "STAGE3A_SHADOW_RUN_MANIFEST.json", run_manifest)
    if r3_mode:
        shutil.copyfile(args.r3_transfer_audit.resolve(strict=True), output_root / "R3A_TRANSFER_AUDIT.json")
        transfer_report = json.loads(args.r3_transfer_audit.resolve(strict=True).read_text(encoding="utf-8"))
    else:
        transfer_cmd = [
            sys.executable, str(repo / "scripts/fec/audit_stage2_r2_transfer.py"),
            "--stage2-root", str(stage2_root), "--checkpoint", str(checkpoint),
            "--freeze-receipt", str(receipt), "--output-root", str(output_root),
            "--expected-source-commit", STAGE2_COMMIT, "--expected-source-tree", STAGE2_TREE,
            "--expected-checkpoint-sha256", CHECKPOINT_SHA,
            "--expected-cache-seal", "929a0a666a867c93094b13752f4c2f848640bbedb2dadc9a20d834f3ee8b6814",
        ]
        transfer = subprocess.run(transfer_cmd, cwd=str(repo), check=False)
        if transfer.returncode != 0:
            fail("FINAL_CHECKPOINT_TRANSFER_FAIL")
        transfer_report = json.loads((output_root / "FINAL_CHECKPOINT_TRANSFER_AUDIT.json").read_text(encoding="utf-8"))
    if (transfer_report.get("status") != "R3A_MATCHED_ENSEMBLE_TRANSFER_PASS" if r3_mode else transfer_report.get("status") != "PASS"):
        fail("R3A_MATCHED_ENSEMBLE_TRANSFER_FAIL" if r3_mode else "FINAL_CHECKPOINT_TRANSFER_FAIL")
    if args.preflight_only:
        run_manifest["status"] = "PREFLIGHT_PASS"
        run_manifest["attack_rollouts"] = 0
        atomic_json(output_root / "STAGE3A_PREFLIGHT_PASS.json", {"status": "PASS", "planned_jobs": len(jobs), "eval160_reads": 0, "attack_rollouts": 0})
        atomic_json(output_root / "STAGE3A_SHADOW_RUN_MANIFEST.json", run_manifest)
        return 0
    run_manifest["status"] = "RUNNING"
    run_manifest["attack_rollouts"] = len(jobs)
    atomic_json(output_root / "STAGE3A_SHADOW_RUN_MANIFEST.json", run_manifest)
    dispatch(jobs, gpu_ids, repo, output_root, checkpoint, receipt, n4_norm, attacker_sha, head, tree, r3_binding)
    run_manifest["status"] = "CLOSED"
    run_manifest["completed_jobs"] = sum(job["status"] == "COMPLETED" for job in jobs)
    atomic_json(output_root / "STAGE3A_SHADOW_RUN_MANIFEST.json", run_manifest)
    audit_cmd = [sys.executable, str(repo / "scripts/fec/audit_stage3a_shadow_matrix.py"), "--root", str(output_root)]
    audit = subprocess.run(audit_cmd, cwd=str(repo), check=False)
    if audit.returncode != 0:
        return audit.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"STAGE3A HOLD: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
