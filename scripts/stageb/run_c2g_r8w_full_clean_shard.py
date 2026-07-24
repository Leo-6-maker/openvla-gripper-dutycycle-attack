#!/usr/bin/env python3
"""Preview or run one hash-bound R8W full-clean worker shard."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
    AUTHORIZATION_TOKEN,
    CANARY_PASS_STATUS,
    CANARY_PURPOSE,
    PASS_STATUS,
    PURPOSE,
    SCHEMA,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
)

COLLECTOR = REPO / "scripts" / "stageb" / "collect_c2g_r8w_teacher_v2_clean.py"
PREVIEW_STATUS = "PASS_C2G_R8W_FULL_CLEAN_SHARD_PREVIEW"
RUN_STATUS = "PASS_C2G_R8W_FULL_CLEAN_SHARD_RUN"
RECEIPT_SCHEMA = "c2g.r8w.full_clean_worker_receipt.2026-07-12.v1"
EPISODE_RECEIPT_SCHEMA = "c2g.r8w.episode_receipt.2026-07-12.v1"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require_file(path: Path, name: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_within(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def validate_episode_receipt(
    episode_dir: Path,
    *,
    expected_parent_key: str,
    expected_worker_id: str,
    expected_shard_id: str,
    expected_git_head: str,
    expected_manifest_sha: str,
) -> tuple[bool, str]:
    try:
        receipt_path = episode_dir / "episode_receipt.json"
        metadata_path = episode_dir / "episode_metadata.json"
        steps_path = episode_dir / "step_records.jsonl"
        rgb_manifest_path = episode_dir / "rgb_manifest.jsonl"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema": EPISODE_RECEIPT_SCHEMA,
            "parent_key": expected_parent_key,
            "worker_id": expected_worker_id,
            "shard_id": expected_shard_id,
            "git_head": expected_git_head,
            "manifest_sha256": expected_manifest_sha,
            "runtime_valid": True,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                return False, f"receipt {key} mismatch"
        if metadata.get("runtime_valid") is not True or type(metadata.get("clean_success_observed")) is not bool:
            return False, "metadata runtime/success mismatch"
        if not steps_path.is_file() or steps_path.stat().st_size == 0:
            return False, "step records missing or empty"
        if sha256_file(metadata_path) != receipt.get("metadata_sha256"):
            return False, "metadata SHA mismatch"
        if sha256_file(steps_path) != receipt.get("step_records_sha256"):
            return False, "step records SHA mismatch"
        if sha256_file(rgb_manifest_path) != receipt.get("rgb_manifest_sha256"):
            return False, "RGB manifest SHA mismatch"
        entries = read_jsonl(rgb_manifest_path)
        if not entries:
            return False, "RGB manifest empty"
        for row in entries:
            frame = episode_dir / "rgb" / str(row["path"])
            if not frame.is_file() or frame.stat().st_size != int(row["bytes"]):
                return False, "RGB frame missing or wrong size"
            if sha256_file(frame) != row["sha256"]:
                return False, "RGB frame SHA mismatch"
        return True, "PASS"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = require_file(path, "R8W plan report")
    if sha256_file(path) != expected_sha256:
        raise ValueError("R8W plan report SHA mismatch")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA or plan.get("status") not in {PASS_STATUS, CANARY_PASS_STATUS}:
        raise ValueError("R8W plan is not an accepted materialized plan")
    if plan.get("authorization_token") != AUTHORIZATION_TOKEN:
        raise ValueError("R8W plan authorization token mismatch")
    expected_cardinality = {
        PURPOSE: (2000, 16),
        CANARY_PURPOSE: (8, 4),
    }.get(plan.get("plan_kind"))
    if expected_cardinality is None:
        raise ValueError("R8W plan kind mismatch")
    if (plan.get("episode_count"), plan.get("worker_count")) != expected_cardinality:
        raise ValueError("R8W plan cardinality mismatch")
    return plan


def select_shard(plan: Mapping[str, Any], worker_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = [dict(row) for row in plan.get("shards", []) if row.get("worker_id") == worker_id]
    if len(matches) != 1:
        raise ValueError(f"expected one worker {worker_id!r}, found {len(matches)}")
    shard = matches[0]
    manifest = require_file(Path(str(shard["manifest"])), "R8W shard manifest")
    if sha256_file(manifest) != str(shard["manifest_sha256"]):
        raise ValueError("R8W shard manifest SHA mismatch")
    rows = read_jsonl(manifest)
    expected_count = int(shard.get("episode_count", -1))
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError("R8W shard episode cardinality mismatch")
    expected_suite = str(shard["suite"])
    expected_gpu = int(shard["physical_gpu"])
    expected_shard = str(shard["shard_id"])
    if any(
        row.get("assigned_worker_id") != worker_id
        or row.get("assigned_shard_id") != expected_shard
        or row.get("assigned_physical_gpu") != expected_gpu
        or row.get("suite") != expected_suite
        for row in rows
    ):
        raise ValueError("R8W shard row assignment mismatch")
    ids = [identity(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("R8W shard contains duplicate identity")
    if len({row.get("max_steps") for row in rows}) != 1:
        raise ValueError("R8W shard contains mixed max_steps")
    return shard, rows


def worker_environment(physical_gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "C2G_PHYSICAL_GPU": str(physical_gpu),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def collector_command(
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    collection_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    model_load_lock_file: Path,
    worker_status_file: Path,
    dummy_wait: int,
    base_seed: int,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(COLLECTOR),
        "--manifest", str(Path(str(shard["manifest"])).resolve()),
        "--manifest-sha256", str(shard["manifest_sha256"]),
        "--output-root", str(collection_root.resolve()),
        "--expected-git-commit", str(plan["expected_git_commit"]),
        "--suite-model-map", str(suite_model_map.resolve()),
        "--suite-model-report", str(suite_model_report.resolve()),
        "--goal-model-manifest", str(goal_model_manifest.resolve()),
        "--model-verification-report", str(model_verification_report.resolve()),
        "--worker-id", str(shard["worker_id"]),
        "--shard-id", str(shard["shard_id"]),
        "--physical-gpu", str(shard["physical_gpu"]),
        "--model-load-lock-file", str(model_load_lock_file.resolve()),
        "--worker-status-file", str(worker_status_file.resolve()),
        "--device", "cuda:0",
        "--dummy-wait", str(dummy_wait),
        "--base-seed", str(base_seed),
    ]
    if resume:
        command.append("--resume")
    return command


def verify_completed_worker(
    collection_root: Path,
    rows: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
    expected_head: str,
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for row in rows:
        episode_dir = collection_root / "episodes" / str(shard["suite"]) / str(row["parent_key"])
        valid, reason = validate_episode_receipt(
            episode_dir,
            expected_parent_key=str(row["parent_key"]),
            expected_worker_id=str(shard["worker_id"]),
            expected_shard_id=str(shard["shard_id"]),
            expected_git_head=expected_head,
            expected_manifest_sha=str(shard["manifest_sha256"]),
        )
        if not valid:
            raise RuntimeError(f"worker episode receipt failed: {row['parent_key']}: {reason}")
        verified.append({"parent_key": row["parent_key"], "receipt_status": "PASS"})
    return verified


def write_worker_checksums(output_root: Path) -> tuple[Path, str]:
    paths = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    checksums = output_root / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n" for path in paths),
        encoding="ascii",
    )
    digest = output_root / "SHA256SUMS.sha256"
    digest.write_text(f"{sha256_file(checksums)}  SHA256SUMS\n", encoding="ascii")
    return checksums, sha256_file(checksums)


def run_shard(
    *,
    mode: str,
    plan_report: Path,
    expected_plan_report_sha256: str,
    worker_id: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    model_verification_report: Path,
    model_load_lock_file: Path,
    dummy_wait: int,
    base_seed: int,
    authorization: str,
    resume: bool,
) -> dict[str, Any]:
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")
    plan = load_plan(plan_report, expected_plan_report_sha256)
    head = git_output("rev-parse", "HEAD")
    if head != plan["expected_git_commit"]:
        raise RuntimeError("current HEAD differs from R8W plan HEAD")
    if git_output("status", "--porcelain"):
        raise RuntimeError("R8W runner requires a clean worktree")
    shard, rows = select_shard(plan, worker_id)
    for path, name in (
        (suite_model_map, "suite model map"),
        (suite_model_report, "suite model report"),
        (goal_model_manifest, "Goal model manifest"),
        (model_verification_report, "model verification report"),
    ):
        require_file(path, name)

    output_root = output_root.resolve()
    if is_within(output_root, REPO):
        raise ValueError("R8W worker output must be outside repository")
    if output_root.exists() and not resume:
        raise FileExistsError(output_root)
    collection_root = output_root / "collection"
    worker_status = output_root / "worker_status.json"
    command = collector_command(
        plan=plan,
        shard=shard,
        collection_root=collection_root,
        suite_model_map=suite_model_map,
        suite_model_report=suite_model_report,
        goal_model_manifest=goal_model_manifest,
        model_verification_report=model_verification_report,
        model_load_lock_file=model_load_lock_file,
        worker_status_file=worker_status,
        dummy_wait=dummy_wait,
        base_seed=base_seed,
        resume=resume,
    )
    preview = {
        "status": PREVIEW_STATUS,
        "worker_id": worker_id,
        "suite": shard["suite"],
        "physical_gpu": shard["physical_gpu"],
        "cuda_visible_devices": str(shard["physical_gpu"]),
        "model_device": "cuda:0",
        "render_gpu_device_id": shard["physical_gpu"],
        "plan_kind": plan["plan_kind"],
        "episode_count": len(rows),
        "shard_manifest": shard["manifest"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "output_root": str(output_root),
        "command": command,
    }
    if mode == "preview":
        return preview
    if authorization != AUTHORIZATION_TOKEN:
        raise PermissionError("R8W shard run authorization mismatch")
    output_root.mkdir(parents=True, exist_ok=resume)
    completed = subprocess.run(command, cwd=REPO, env=worker_environment(int(shard["physical_gpu"])))
    if completed.returncode != 0:
        raise RuntimeError(f"R8W collector exited {completed.returncode}")
    verified = verify_completed_worker(collection_root, rows, shard, str(plan["expected_git_commit"]))
    report_path = collection_root / "c2g_r8w_collection_report.json"
    report = read_json(require_file(report_path, "R8W collector report"))
    if report.get("status") != "PASS_C2G_R8W_TEACHER_V2_CLEAN_SHARD_COLLECTION":
        raise RuntimeError("R8W collector report did not PASS")
    if report.get("runtime_valid_episode_count") != len(rows) or report.get("failed_episode_count") != 0:
        raise RuntimeError("R8W collector report cardinality failed")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RUN_STATUS,
        "worker_id": worker_id,
        "suite": shard["suite"],
        "physical_gpu": shard["physical_gpu"],
        "shard_id": shard["shard_id"],
        "git_head": head,
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": expected_plan_report_sha256,
        "shard_manifest": shard["manifest"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "suite_model_map_sha256": sha256_file(suite_model_map.resolve()),
        "suite_model_report_sha256": sha256_file(suite_model_report.resolve()),
        "goal_model_manifest_sha256": sha256_file(goal_model_manifest.resolve()),
        "model_verification_report_sha256": sha256_file(model_verification_report.resolve()),
        "episode_count": len(verified),
        "runtime_valid_episode_count": len(verified),
        "collector_report": str(report_path),
        "collector_report_sha256": sha256_file(report_path),
        "resumed_episode_count": int(report.get("resumed_episode_count", 0)),
        "failed_episode_count": 0,
        "attacks": 0,
        "training_epochs": 0,
        "materialization_runs": 0,
    }
    receipt_path = output_root / "worker_receipt.json"
    write_json(receipt_path, receipt)
    checksums, checksums_sha = write_worker_checksums(output_root)
    return {
        **receipt,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checksums": str(checksums),
        "checksums_sha256": checksums_sha,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run"))
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    parser.add_argument("--model-load-lock-file", type=Path, default=Path("/tmp/c2g_r8w_global_model_load.lock"))
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260711)
    parser.add_argument("--authorization", default=os.environ.get("R8W_COLLECTION_AUTHORIZATION", ""))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_shard(
        mode=args.mode,
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        worker_id=args.worker_id,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        model_verification_report=args.model_verification_report,
        model_load_lock_file=args.model_load_lock_file,
        dummy_wait=args.dummy_wait,
        base_seed=args.base_seed,
        authorization=args.authorization,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
