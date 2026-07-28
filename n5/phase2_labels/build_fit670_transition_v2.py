"""Build the fail-closed FIT670_INFERENCE_TRANSITION_V2 receipt."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import fit670_strict_contract as strict
from fit_transition import (
    FROZEN_R5E,
    compute_model_tree_fingerprint,
    verify_r5e_comparison_root,
)


def seal_root(root: Path) -> str:
    payload = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [
        f"{strict.sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in payload
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = strict.sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--r5e-comparison-root", type=Path, required=True)
    parser.add_argument("--allowed-output-root", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True, help="e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    allowlist_path = args.identity_allowlist.resolve()
    shard_plan_path = args.shard_plan.resolve()
    output_root = args.allowed_output_root.resolve()
    transition_root = args.out.resolve()
    if transition_root.exists():
        raise SystemExit(f"transition output already exists: {transition_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"collection output is not empty: {output_root}")

    allowlist, _ = strict.validate_allowlist(allowlist_path)
    physical_gpus = [int(value) for value in args.physical_gpus.split(",")]
    if len(set(physical_gpus)) != len(physical_gpus) or not physical_gpus:
        raise SystemExit("physical GPU list must be non-empty and unique")
    plan, _ = strict.validate_shard_plan(
        shard_plan_path, allowlist_path, expected_n_shards=len(physical_gpus)
    )

    comp_root = args.r5e_comparison_root.resolve()
    ok, _, issues = verify_r5e_comparison_root(comp_root)
    if not ok:
        raise SystemExit(f"R5-E comparison verification failed: {issues}")
    comparison_sha = strict.sha256_file(comp_root / "SHA256SUMS")
    if comparison_sha != FROZEN_R5E["r5e_comparison_sha256"]:
        raise SystemExit("R5-E comparison SHA differs from frozen evidence")

    source_files = {
        "fit670_strict_contract.py": here / "fit670_strict_contract.py",
        "run_fit670_atomic_worker_v2.py": here / "run_fit670_atomic_worker_v2.py",
        "run_fit670_atomic_worker.py": here / "run_fit670_atomic_worker.py",
        "fit_collection_core.py": here / "fit_collection_core.py",
        "run_fit670_supervisor_v2.py": here / "run_fit670_supervisor_v2.py",
        "finalize_fit670_collection_v2.py": here / "finalize_fit670_collection_v2.py",
        "run_fit670_v2.sh": here / "run_fit670_v2.sh",
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing V2 source files: {missing}")

    source_commit, source_tree = strict.git_identity(repo_root)
    upstream_commit, upstream_tree = strict.git_identity(args.upstream_root)
    libero_commit, libero_tree = strict.git_identity(args.libero_root)
    shard_to_gpu = {
        str(shard_id): physical_gpus[shard_id]
        for shard_id in range(plan["n_shards"])
    }

    manifest = {
        "gate": strict.TRANSITION_GATE,
        "schema": strict.TRANSITION_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_BEFORE_EXECUTION",
        "consumer_eligible": False,
        **FROZEN_R5E,
        "identity_pool": "D0-R2_DEV_POOL_670",
        "protected_overlap_verified": 0,
        "identity_allowlist_path": str(allowlist_path),
        "identity_allowlist_file_sha256": strict.sha256_file(allowlist_path),
        "identity_set_digest": allowlist["identity_set_digest"],
        "authorized_identities": 670,
        "max_episodes": 670,
        "identity_set_frozen": True,
        "shard_plan_path": str(shard_plan_path),
        "shard_plan_sha256": strict.sha256_file(shard_plan_path),
        "n_shards": plan["n_shards"],
        "allowed_physical_gpus": physical_gpus,
        "shard_to_physical_gpu": shard_to_gpu,
        "allowed_gpus": [0],
        "physical_to_logical_gpu": {str(gpu): 0 for gpu in physical_gpus},
        "allowed_output_roots": [str(output_root)],
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": compute_model_tree_fingerprint(args.model_path),
        "processor_sha256": strict.sha256_file(
            args.model_path / "preprocessor_config.json"
        ),
        "official_worker_path": str(args.official_worker.resolve()),
        "official_worker_sha256": strict.sha256_file(args.official_worker),
        "registry_summary_sha256": strict.sha256_file(args.registry_summary),
        "alias_ledger_sha256": strict.sha256_file(args.alias_ledger),
        "collection_source_commit": source_commit,
        "collection_source_tree": source_tree,
        "collection_source_files": {
            name: strict.sha256_file(path) for name, path in source_files.items()
        },
        "upstream_commit": upstream_commit,
        "upstream_tree": upstream_tree,
        "libero_commit": libero_commit,
        "libero_tree": libero_tree,
        "cross_gpu_trajectory_parity_guaranteed": False,
        "sdpa_bf16_non_determinism_documented": True,
        "openvla_inference_authorized": True,
        "clean_action_only": True,
        "forward_before_capture": True,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "detector_load_authorized": False,
        "attack_authorized": False,
        "protected_payload_read": False,
    }

    staging = transition_root.parent / (
        f".{transition_root.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    published = False
    try:
        (staging / "TRANSITION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        seal_root(staging)
        staging.rename(transition_root)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(f"FIT670 transition V2 sealed: {transition_root}")
    print(f"source: {source_commit}")
    print(f"identity_set_digest: {allowlist['identity_set_digest']}")
    print(f"shard_plan_sha256: {strict.sha256_file(shard_plan_path)}")
    print(f"shard_to_physical_gpu: {shard_to_gpu}")


if __name__ == "__main__":
    main()
