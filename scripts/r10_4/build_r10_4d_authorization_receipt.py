#!/usr/bin/env python3
"""Build the one-use R10.4D passive-smoke authorization receipt.

This script performs only read-only hashing and contract validation.  It never
imports OpenVLA, torch, LIBERO, or a detector checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256, sha256_file, verify_checksum_manifest


SUPPORTED_PARENT = "libero_10/task_00/state_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--r4c-classification", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
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
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        rows.append({
            "path": item.relative_to(path).as_posix(),
            "size": size,
            "sha256": sha256_file(item),
        })
    if not rows:
        raise SystemExit("MODEL_TREE_EMPTY")
    return canonical_json_sha(rows), len(rows), total_bytes


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def selected_parent(manifest: dict[str, Any]) -> str:
    selected = manifest.get("selected_parent")
    if isinstance(selected, dict):
        selected = selected.get("identity") or selected.get("canonical_parent_key")
    if not isinstance(selected, str):
        raise SystemExit("PARENT_MANIFEST_SELECTED_PARENT_MISSING")
    return selected


def validate_r4c(payload: dict[str, Any]) -> None:
    classification = payload.get("classification") or payload.get("r4c_classification")
    if classification != "CONTACT_DYNAMICS_REPLAY_DIVERGENCE":
        raise SystemExit(f"R4C_CLASSIFICATION_FAIL:{classification}")
    if payload.get("clean_s1_exact_parity") is not True:
        raise SystemExit("R4C_CLEAN_S1_PARITY_FAIL")
    if float(payload.get("clean_s1_max_abs_error", -1.0)) != 0.0:
        raise SystemExit("R4C_CLEAN_S1_ERROR_FAIL")
    if payload.get("action_mutated") is not False:
        raise SystemExit("R4C_ACTION_MUTATION_FAIL")
    if payload.get("first_divergence_layer") != "DIRECT_13D":
        raise SystemExit("R4C_DIVERGENCE_LAYER_FAIL")
    if payload.get("feature_adapter_bug") is not False:
        raise SystemExit("R4C_ADAPTER_BUG_NOT_EXCLUDED")
    if payload.get("training_source_binding_failure") is not False:
        raise SystemExit("R4C_TRAINING_BINDING_NOT_EXCLUDED")


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"AUTH_RECEIPT_OUTPUT_EXISTS:{args.output}")
    if not args.model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{args.model_path}")
    if not args.parent_manifest.is_file() or not args.r4c_classification.is_file() or not args.protocol.is_file():
        raise SystemExit("AUTH_SOURCE_FILE_MISSING")

    repo_root = Path(__file__).resolve().parents[2]
    head = git_head(repo_root)
    if not git_clean(repo_root):
        raise SystemExit("AUTH_WORKTREE_DIRTY")

    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    parent = selected_parent(parent_manifest)
    if parent != SUPPORTED_PARENT:
        raise SystemExit(f"AUTH_PARENT_FAIL:{parent}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_PROTOCOL_V1":
        raise SystemExit("AUTH_PROTOCOL_SCHEMA_FAIL")
    if protocol.get("selected_parent") != parent or int(protocol.get("episodes_authorized", 0)) != 1:
        raise SystemExit("AUTH_PROTOCOL_PARENT_OR_COUNT_FAIL")
    if any(protocol.get(key) is not False for key in (
        "formal_training_authorized",
        "formal_attack_authorized",
        "command_open_authorized",
        "visual_attack_authorized",
        "random_attack_authorized",
    )):
        raise SystemExit("AUTH_PROTOCOL_FORBIDDEN_SCOPE")

    r4c = json.loads(args.r4c_classification.read_text(encoding="utf-8"))
    validate_r4c(r4c)

    bundle_seal = verify_checksum_manifest(args.detector_bundle)
    checkpoint_path = args.detector_bundle / "full_fit_deploy.pt"
    if not checkpoint_path.is_file():
        raise SystemExit("AUTH_DETECTOR_CHECKPOINT_MISSING")
    checkpoint_sha = sha256_file(checkpoint_path)
    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)

    receipt = {
        "schema": "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_AUTH_V1",
        "scope": "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE",
        "authorization_comment_id": args.authorization_comment_id,
        "source_commit": head,
        "selected_parent": parent,
        "episodes_authorized": 1,
        "passive_only": True,
        "model_load_authorized": True,
        "detector_execution_authorized": True,
        "action_mutation_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "command_open_authorized": False,
        "visual_attack_authorized": False,
        "random_attack_authorized": False,
        "r4c_classification": "CONTACT_DYNAMICS_REPLAY_DIVERGENCE",
        "r4c_classification_sha256": sha256_file(args.r4c_classification),
        "protocol_sha256": sha256_file(args.protocol),
        "parent_manifest_sha256": sha256_file(args.parent_manifest),
        "detector_checkpoint_sha256": checkpoint_sha,
        "bundle_sha256s_sha256": bundle_seal["sha256sums_sha256"],
        "model_tree_sha256": model_tree_sha,
        "model_file_count": model_file_count,
        "model_bytes": model_bytes,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "second_episode_authorized": False,
        "parent_substitution_authorized": False,
        "threshold_or_fsm_change_authorized": False,
        "output_overwrite_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_sha = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{receipt_sha}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "receipt": str(args.output),
        "receipt_sha256": receipt_sha,
        "source_commit": head,
        "selected_parent": parent,
        "detector_checkpoint_sha256": checkpoint_sha,
        "bundle_sha256s_sha256": bundle_seal["sha256sums_sha256"],
        "model_tree_sha256": model_tree_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
