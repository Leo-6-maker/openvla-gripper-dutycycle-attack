#!/usr/bin/env python3
"""Build the one-use R10.4E ten-task passive panel authorization receipt.

Read-only hashing and contract validation. Never imports OpenVLA, torch,
LIBERO, or a detector checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256, sha256_file, verify_checksum_manifest

TASK_MANIFEST = [
    {"identity": "libero_10/task_00/state_20", "reuse": True},
    {"identity": "libero_10/task_01/state_20", "reuse": False},
    {"identity": "libero_10/task_02/state_20", "reuse": False},
    {"identity": "libero_10/task_03/state_20", "reuse": False},
    {"identity": "libero_10/task_04/state_20", "reuse": False},
    {"identity": "libero_10/task_05/state_20", "reuse": False},
    {"identity": "libero_10/task_06/state_20", "reuse": False},
    {"identity": "libero_10/task_07/state_20", "reuse": False},
    {"identity": "libero_10/task_08/state_20", "reuse": False},
    {"identity": "libero_10/task_09/state_20", "reuse": False},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--task00-root", required=True, type=Path)
    parser.add_argument("--panel-protocol", required=True, type=Path)
    parser.add_argument("--authorization-comment-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def checkpoint_tree_fingerprint(path: Path) -> tuple[str, int, int]:
    rows = []
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda v: v.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        rows.append({"path": item.relative_to(path).as_posix(), "size": size, "sha256": sha256_file(item)})
    if not rows:
        raise SystemExit("MODEL_TREE_EMPTY")
    return canonical_json_sha(rows), len(rows), total_bytes


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def git_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"RECEIPT_OUTPUT_EXISTS:{args.output}")
    if not args.model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{args.model_path}")
    if not args.task00_root.is_dir():
        raise SystemExit(f"TASK00_ROOT_MISSING:{args.task00_root}")
    if not args.panel_protocol.is_file():
        raise SystemExit("PROTOCOL_MISSING")

    repo_root = Path(__file__).resolve().parents[2]
    head = git_head(repo_root)
    if not git_clean(repo_root):
        raise SystemExit("RECEIPT_WORKTREE_DIRTY")

    protocol = json.loads(args.panel_protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "R10_4E_TEN_TASK_PASSIVE_PANEL_PROTOCOL_V1":
        raise SystemExit("PROTOCOL_SCHEMA_FAIL")
    for key in ("command_open_authorized", "visual_attack_authorized", "random_attack_authorized",
                "formal_training_authorized", "formal_attack_authorized"):
        if protocol.get(key) is not False:
            raise SystemExit(f"PROTOCOL_FORBIDDEN_FAIL:{key}")

    # Verify task00 seal
    task00_seal = verify_checksum_manifest(args.task00_root)

    bundle_seal = verify_checksum_manifest(args.detector_bundle)
    checkpoint_path = args.detector_bundle / "full_fit_deploy.pt"
    if not checkpoint_path.is_file():
        raise SystemExit("DETECTOR_CHECKPOINT_MISSING")
    checkpoint_sha = sha256_file(checkpoint_path)
    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)

    receipt = {
        "schema": "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1",
        "scope": "R10_4E_TEN_TASK_PASSIVE_PANEL",
        "authorization_comment_id": args.authorization_comment_id,
        "source_commit": head,
        "episodes_authorized": 10,
        "fresh_executions_authorized": 9,
        "reuse_authorized": 1,
        "task_manifest": TASK_MANIFEST,
        "task_manifest_sha256": canonical_json_sha(TASK_MANIFEST),
        "task00_root": str(args.task00_root.resolve()),
        "task00_root_sha256s": task00_seal["sha256sums_sha256"],
        "passive_only": True,
        "model_load_authorized": True,
        "detector_execution_authorized": True,
        "action_mutation_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "command_open_authorized": False,
        "visual_attack_authorized": False,
        "random_attack_authorized": False,
        "retry_authorized": False,
        "parent_substitution_authorized": False,
        "threshold_or_fsm_change_authorized": False,
        "output_overwrite_authorized": False,
        "protocol_sha256": sha256_file(args.panel_protocol),
        "detector_checkpoint_sha256": checkpoint_sha,
        "bundle_sha256s_sha256": bundle_seal["sha256sums_sha256"],
        "model_tree_sha256": model_tree_sha,
        "model_file_count": model_file_count,
        "model_bytes": model_bytes,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "gpu": 0,
        "render_gpu": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{receipt_digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "receipt": str(args.output),
        "receipt_sha256": receipt_digest,
        "source_commit": head,
        "detector_checkpoint_sha256": checkpoint_sha,
        "bundle_sha256s_sha256": bundle_seal["sha256sums_sha256"],
        "model_tree_sha256": model_tree_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
