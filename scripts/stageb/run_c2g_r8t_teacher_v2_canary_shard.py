#!/usr/bin/env python3
"""Preview or execute one hash-bound R8T Teacher-v2 canary shard."""
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

from tools.multisuite_detector.build_c2g_r8t_teacher_v2_canary import (
    AUTHORIZATION_TOKEN,
    PASS_STATUS,
    SCHEMA,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
)

COLLECTOR = REPO / "scripts" / "stageb" / "collect_c2g_r8t_teacher_v2_canary.py"
PREVIEW_STATUS = "PASS_C2G_R8T_CANARY_SHARD_PREVIEW"
RUN_STATUS = "PASS_C2G_R8T_CANARY_SHARD_RUN"
RECEIPT_SCHEMA = "c2g.r8t.teacher_v2_canary_shard_receipt.2026-07-11.v1"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require_file(path: Path, name: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def is_within(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def load_plan(path: Path, expected_sha: str) -> dict[str, Any]:
    path = require_file(path, "R8T plan report")
    if sha256_file(path) != expected_sha:
        raise ValueError("R8T plan report SHA mismatch")
    report = read_json(path)
    if report.get("schema") != SCHEMA or report.get("status") != PASS_STATUS:
        raise ValueError("R8T plan report is not accepted")
    if report.get("collection_authorization") != "AUTHORIZED_BY_USER_FOR_R8T_24EP_TRAIN_ONLY_CANARY":
        raise ValueError("R8T plan lacks explicit user authorization")
    if not all(report.get("invariants", {}).values()):
        raise ValueError("R8T plan invariant failure")
    return report


def select_shard(plan: Mapping[str, Any], shard_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = [dict(row) for row in plan.get("shards", []) if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected one shard {shard_id!r}, found {len(matches)}")
    shard = matches[0]
    manifest = require_file(Path(str(shard["manifest"])), "shard manifest")
    if sha256_file(manifest) != str(shard["manifest_sha256"]):
        raise ValueError("shard manifest SHA mismatch")
    rows = read_jsonl(manifest)
    if len(rows) != int(shard["episode_count"]):
        raise ValueError("shard episode count mismatch")
    if any(str(row.get("suite")) != str(shard.get("suite")) for row in rows):
        raise ValueError("shard contains mixed/wrong suite")
    if any(row.get("cohort") != "DETECTOR_TRAIN" or row.get("split") != "train" for row in rows):
        raise ValueError("R8T shard contains non-train identity")
    ids = [identity(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate shard identity")
    return shard, rows


def run_shard(
    *,
    mode: str,
    plan_report: Path,
    expected_plan_report_sha256: str,
    shard_id: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    device: str,
    max_steps: int,
    dummy_wait: int,
    base_seed: int,
    authorization: str,
) -> dict[str, Any]:
    if mode not in {"preview", "run"}:
        raise ValueError("mode must be preview or run")
    plan = load_plan(plan_report, expected_plan_report_sha256)
    expected_head = str(plan["expected_git_commit"])
    head = git_output("rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(f"current head {head} differs from R8T plan head {expected_head}")
    if git_output("status", "--porcelain"):
        raise RuntimeError("R8T runner requires a clean worktree")
    shard, rows = select_shard(plan, shard_id)

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if is_within(output_root, REPO):
        raise ValueError("R8T output root must be outside repository")
    for path, name in (
        (suite_model_map, "suite model map"),
        (suite_model_report, "suite model report"),
        (goal_model_manifest, "Goal model manifest"),
    ):
        require_file(path, name)

    collection_root = output_root / "clean_collection"
    verification_report = output_root / "config" / "c2g_suite_model_verification_report.json"
    command = [
        sys.executable,
        str(COLLECTOR),
        "--manifest", str(Path(str(shard["manifest"])).resolve()),
        "--output-root", str(collection_root),
        "--expected-git-commit", expected_head,
        "--suite-model-map", str(suite_model_map.resolve()),
        "--suite-model-report", str(suite_model_report.resolve()),
        "--goal-model-manifest", str(goal_model_manifest.resolve()),
        "--model-verification-report", str(verification_report),
        "--device", device,
        "--max-steps", str(max_steps),
        "--dummy-wait", str(dummy_wait),
        "--base-seed", str(base_seed),
    ]
    preview = {
        "status": PREVIEW_STATUS,
        "mode": mode,
        "git_commit": head,
        "shard_id": shard_id,
        "suite": shard["suite"],
        "episode_count": shard["episode_count"],
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": sha256_file(plan_report.resolve()),
        "shard_manifest": shard["manifest"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "output_root": str(output_root),
        "command": command,
        "boundaries": {
            "output_created": False,
            "non_train_parent_count": 0,
            "attacks_launched": 0,
            "training_epochs": 0,
        },
    }
    if mode == "preview":
        return preview
    if authorization != AUTHORIZATION_TOKEN:
        raise PermissionError(f"exact authorization token required: {AUTHORIZATION_TOKEN}")

    completed = subprocess.run(command, cwd=REPO)
    if completed.returncode != 0:
        raise RuntimeError(
            f"R8T shard failed with return code {completed.returncode}; retain {output_root}"
        )
    collection_report_path = collection_root / "c2g_r8t_collection_report.json"
    collection_report = read_json(require_file(collection_report_path, "collection report"))
    if collection_report.get("status") != "PASS_C2G_R8T_TEACHER_V2_CANARY_COLLECTION":
        raise ValueError("collector did not emit PASS")
    if int(collection_report.get("episode_count", -1)) != len(rows):
        raise ValueError("collection report episode count mismatch")
    expected_ids = {identity(row) for row in rows}
    actual_ids = {
        (str(row["suite"]), int(row["task_index"]), int(row["state_id"]))
        for row in collection_report.get("results", [])
    }
    if actual_ids != expected_ids:
        raise ValueError("collected identities differ from shard manifest")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RUN_STATUS,
        "git_commit": head,
        "shard_id": shard_id,
        "suite": shard["suite"],
        "episode_count": len(rows),
        "plan_report": str(plan_report.resolve()),
        "plan_report_sha256": sha256_file(plan_report.resolve()),
        "shard_manifest": shard["manifest"],
        "shard_manifest_sha256": shard["manifest_sha256"],
        "output_root": str(output_root),
        "collection_root": str(collection_root),
        "collection_report": str(collection_report_path),
        "collection_report_sha256": sha256_file(collection_report_path),
        "model_verification_report": str(verification_report),
        "model_verification_report_sha256": sha256_file(verification_report),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": device,
        "command": command,
        "boundaries": {
            "clean_only": True,
            "train_only": True,
            "existing_output_overwritten": False,
            "attacks_launched": 0,
            "training_epochs": 0,
            "storage_deletions": 0,
        },
    }
    receipt_path = output_root / "c2g_r8t_canary_shard_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run"))
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dummy-wait", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260711)
    parser.add_argument("--authorization", default=os.environ.get("R8T_COLLECTION_AUTHORIZATION", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_shard(
        mode=args.mode,
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        shard_id=args.shard_id,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        device=args.device,
        max_steps=args.max_steps,
        dummy_wait=args.dummy_wait,
        base_seed=args.base_seed,
        authorization=args.authorization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
