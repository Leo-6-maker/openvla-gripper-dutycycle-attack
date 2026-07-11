#!/usr/bin/env python3
"""Preview or execute one provenance-bound R8 clean-collection shard.

Every shard writes to a new isolated output root.  Existing roots are never
removed or overwritten.  A failed shard therefore leaves inspectable evidence
and can be retried only under a different attempt root.
"""
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

from tools.multisuite_detector.audit_c2g_r8_collection_wave import load_wave_plan  # noqa: E402
from tools.multisuite_detector.build_c2g_r8_collection_waves import (  # noqa: E402
    ATTACK_EVAL_WAVE,
    DETECTOR_CANARY,
    DETECTOR_FULL,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
)

RELEASE_COLLECTOR = REPO / "scripts" / "stageb" / "collect_c2g_clean_window_rollouts_release.py"
AUTHORIZATION_TOKENS = {
    DETECTOR_CANARY: "R8_CANARY_COLLECTION_AUTHORIZED",
    DETECTOR_FULL: "R8_DETECTOR_FULL_COLLECTION_AUTHORIZED",
    ATTACK_EVAL_WAVE: "R8_ATTACK_EVAL_COLLECTION_AUTHORIZED",
}
PREVIEW_STATUS = "PASS_C2G_R8_COLLECTION_SHARD_PREVIEW"
RUN_STATUS = "PASS_C2G_R8_COLLECTION_SHARD_RUN"
RECEIPT_SCHEMA = "c2g.r8.collection_shard_receipt.2026-07-11.v1"


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _require_file(path: Path, name: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _assert_clean_current_head(expected: str) -> str:
    head = _git_output("rev-parse", "HEAD")
    if head != expected:
        raise RuntimeError(f"current head {head} differs from wave-plan head {expected}")
    if _git_output("status", "--porcelain"):
        raise RuntimeError("R8 collection runner requires a clean worktree")
    return head


def _select_shard(wave_info: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in wave_info.get("shards", []) if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one shard {shard_id!r}, found {len(matches)}")
    shard = matches[0]
    manifest = _require_file(Path(str(shard["manifest"])), "shard manifest")
    if sha256_file(manifest) != str(shard.get("manifest_sha256", "")):
        raise ValueError("shard manifest hash differs from wave plan")
    rows = read_jsonl(manifest)
    if len(rows) != int(shard.get("episode_count", -1)):
        raise ValueError("shard episode count differs from wave plan")
    if any(str(row.get("suite")) != str(shard.get("suite")) for row in rows):
        raise ValueError("shard contains more than one suite or wrong suite")
    wave_ids = {identity(row) for row in wave_info["rows"]}
    shard_ids = [identity(row) for row in rows]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("shard contains duplicate identity")
    if not set(shard_ids).issubset(wave_ids):
        raise ValueError("shard contains identity outside selected wave")
    shard["rows"] = rows
    return shard


def _command(
    *,
    manifest: Path,
    collection_root: Path,
    model_verification_report: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    expected_git_commit: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(RELEASE_COLLECTOR),
        "--suite-model-map",
        str(suite_model_map),
        "--suite-model-report",
        str(suite_model_report),
        "--goal-model-manifest",
        str(goal_model_manifest),
        "--model-verification-report",
        str(model_verification_report),
        "--manifest",
        str(manifest),
        "--output-root",
        str(collection_root),
        "--expected-git-commit",
        expected_git_commit,
        "--device",
        device,
        "--max-episodes",
        "0",
    ]


def run_shard(
    *,
    mode: str,
    wave_plan_report: Path,
    expected_wave_plan_report_sha256: str,
    wave: str,
    shard_id: str,
    output_root: Path,
    suite_model_map: Path,
    suite_model_report: Path,
    goal_model_manifest: Path,
    device: str,
    authorization: str,
) -> dict[str, Any]:
    if mode not in ("preview", "run"):
        raise ValueError("mode must be preview or run")
    plan, _, _, wave_info = load_wave_plan(
        wave_plan_report,
        expected_wave_plan_report_sha256,
        wave,
    )
    expected_head = str(plan.get("expected_git_commit", ""))
    head = _assert_clean_current_head(expected_head)
    shard = _select_shard(wave_info, shard_id)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if _is_within(output_root, REPO):
        raise ValueError("R8 collection output root must be outside the repository")
    if _is_within(output_root, Path(str(plan["r7_source_audit_report"])).resolve().parent):
        raise ValueError("R8 collection output must not be inside the frozen R7 evidence root")

    suite_model_map = _require_file(suite_model_map, "suite model map")
    suite_model_report = _require_file(suite_model_report, "suite model report")
    goal_model_manifest = _require_file(goal_model_manifest, "Goal model manifest")
    manifest = Path(str(shard["manifest"])).resolve()
    collection_root = output_root / "clean_collection"
    verification_report = output_root / "config" / "c2g_suite_model_verification_report.json"
    command = _command(
        manifest=manifest,
        collection_root=collection_root,
        model_verification_report=verification_report,
        suite_model_map=suite_model_map,
        suite_model_report=suite_model_report,
        goal_model_manifest=goal_model_manifest,
        expected_git_commit=head,
        device=device,
    )
    preview = {
        "status": PREVIEW_STATUS,
        "mode": mode,
        "wave": wave,
        "shard_id": shard_id,
        "suite": shard["suite"],
        "episode_count": shard["episode_count"],
        "wave_plan_report": str(wave_plan_report.resolve()),
        "wave_plan_report_sha256": sha256_file(wave_plan_report.resolve()),
        "shard_manifest": str(manifest),
        "shard_manifest_sha256": sha256_file(manifest),
        "output_root": str(output_root),
        "command": command,
        "boundaries": {
            "output_root_created": False,
            "existing_output_overwritten": False,
            "attacks_launched": 0,
            "training_epochs": 0,
        },
    }
    if mode == "preview":
        return preview

    required_token = AUTHORIZATION_TOKENS[wave]
    if authorization != required_token:
        raise PermissionError(
            f"wave {wave} requires exact authorization token {required_token!r}"
        )
    completed = subprocess.run(command, cwd=REPO)
    if completed.returncode != 0:
        raise RuntimeError(
            f"R8 collection shard failed with return code {completed.returncode}; "
            f"retain output root for inspection: {output_root}"
        )
    collection_report_path = collection_root / "c2g_clean_collection_report.json"
    collection_report = read_json(_require_file(collection_report_path, "collection report"))
    if collection_report.get("status") != "PASS_CLEAN_COLLECTION":
        raise ValueError("release collector did not emit PASS_CLEAN_COLLECTION")
    if int(collection_report.get("episode_count", -1)) != int(shard["episode_count"]):
        raise ValueError("collection report episode count differs from shard plan")
    expected_ids = {identity(row) for row in shard["rows"]}
    actual_ids = {
        (str(row["suite"]), int(row["task_index"]), int(row["state_id"]))
        for row in collection_report.get("results", [])
    }
    if actual_ids != expected_ids:
        raise ValueError("collection report identities differ from shard manifest")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RUN_STATUS,
        "git_commit": head,
        "wave": wave,
        "shard_id": shard_id,
        "suite": shard["suite"],
        "episode_count": shard["episode_count"],
        "wave_plan_report": str(wave_plan_report.resolve()),
        "wave_plan_report_sha256": sha256_file(wave_plan_report.resolve()),
        "shard_manifest": str(manifest),
        "shard_manifest_sha256": sha256_file(manifest),
        "collection_root": str(collection_root),
        "collection_report": str(collection_report_path),
        "collection_report_sha256": sha256_file(collection_report_path),
        "model_verification_report": str(verification_report),
        "model_verification_report_sha256": sha256_file(verification_report),
        "suite_model_map_sha256": sha256_file(suite_model_map),
        "suite_model_report_sha256": sha256_file(suite_model_report),
        "goal_model_manifest_sha256": sha256_file(goal_model_manifest),
        "command": command,
        "boundaries": {
            "clean_only": True,
            "existing_output_overwritten": False,
            "attacks_launched": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
            "storage_deletions": 0,
        },
    }
    receipt_path = output_root / "c2g_r8_collection_shard_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run"))
    parser.add_argument("--wave-plan-report", type=Path, required=True)
    parser.add_argument("--expected-wave-plan-report-sha256", required=True)
    parser.add_argument("--wave", choices=tuple(AUTHORIZATION_TOKENS), required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--authorization", default=os.environ.get("R8_COLLECTION_AUTHORIZATION", ""))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_shard(
        mode=args.mode,
        wave_plan_report=args.wave_plan_report,
        expected_wave_plan_report_sha256=args.expected_wave_plan_report_sha256,
        wave=args.wave,
        shard_id=args.shard_id,
        output_root=args.output_root,
        suite_model_map=args.suite_model_map,
        suite_model_report=args.suite_model_report,
        goal_model_manifest=args.goal_model_manifest,
        device=args.device,
        authorization=args.authorization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
