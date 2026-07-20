#!/usr/bin/env python3
"""Build the one-use R10.4E passive panel authorization receipt.

P0-4: Phase-gated — supports E-R3a (task01 canary), E-R3b (task02-03),
E-R3c (task04-09), and full-panel modes.

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


# ── Phase definitions ─────────────────────────────────────────────────────────

PHASES = {
    "E_R3A_TASK01_CANARY": {
        "description": "E-R3a: task00 reuse + task01 single fresh canary",
        "reuse_identities": ["libero_10/task_00/state_20"],
        "fresh_identities": ["libero_10/task_01/state_20"],
    },
    "E_R3B_TASK02_03": {
        "description": "E-R3b: task00 reuse + task01 reuse + task02-03 fresh",
        "reuse_identities": [
            "libero_10/task_00/state_20",
            "libero_10/task_01/state_20",
        ],
        "fresh_identities": [
            "libero_10/task_02/state_20",
            "libero_10/task_03/state_20",
        ],
    },
    "E_R3C_TASK04_09": {
        "description": "E-R3c: task00-03 reuse + task04-09 fresh",
        "reuse_identities": [
            f"libero_10/task_{i:02d}/state_20" for i in range(4)
        ],
        "fresh_identities": [
            f"libero_10/task_{i:02d}/state_20" for i in range(4, 10)
        ],
    },
    "E_R3_FULL_PANEL": {
        "description": "E-R3 full: task00 reuse + task01-09 fresh (10 tasks)",
        "reuse_identities": ["libero_10/task_00/state_20"],
        "fresh_identities": [
            f"libero_10/task_{i:02d}/state_20" for i in range(1, 10)
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--task00-root", required=True, type=Path)
    parser.add_argument("--panel-protocol", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
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

    # Phase definition
    phase_def = PHASES[args.phase]
    reuse_ids = phase_def["reuse_identities"]
    fresh_ids = phase_def["fresh_identities"]

    # Verify task00 is in reuse set
    if "libero_10/task_00/state_20" not in reuse_ids:
        raise SystemExit("TASK00_NOT_IN_REUSE_SET")

    # Verify task00 seal
    task00_seal = verify_checksum_manifest(args.task00_root)

    # Build ordered task manifest (all 10, with reuse flags)
    task_manifest = []
    for i in range(10):
        identity = f"libero_10/task_{i:02d}/state_20"
        task_manifest.append({
            "identity": identity,
            "reuse": identity in reuse_ids,
        })

    bundle_seal = verify_checksum_manifest(args.detector_bundle)
    checkpoint_path = args.detector_bundle / "full_fit_deploy.pt"
    if not checkpoint_path.is_file():
        raise SystemExit("DETECTOR_CHECKPOINT_MISSING")
    checkpoint_sha = sha256_file(checkpoint_path)
    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)

    receipt = {
        "schema": "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1",
        "scope": "R10_4E_TEN_TASK_PASSIVE_PANEL",
        "phase": args.phase,
        "phase_description": phase_def["description"],
        "authorization_comment_id": args.authorization_comment_id,
        "source_commit": head,
        "episodes_authorized": len(task_manifest),
        "fresh_executions_authorized": len(fresh_ids),
        "reuse_authorized": len(reuse_ids),
        "task_manifest": task_manifest,
        "task_manifest_sha256": canonical_json_sha(task_manifest),
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
        "phase": args.phase,
        "source_commit": head,
        "reuse_identities": reuse_ids,
        "fresh_identities": fresh_ids,
        "detector_checkpoint_sha256": checkpoint_sha,
        "bundle_sha256s_sha256": bundle_seal["sha256sums_sha256"],
        "model_tree_sha256": model_tree_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
