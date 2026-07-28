"""Validate and seal the 8-episode FIT670 V2 canary."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import fit670_strict_contract as strict


def seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "\n".join(
            f"{strict.sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in files
        ) + "\n",
        encoding="utf-8",
    )
    digest = strict.sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    review_root = output / "CANARY_REVIEW_V2"
    if review_root.exists():
        raise SystemExit(f"canary review already exists: {review_root}")
    transition = strict.load_json(
        args.transition_receipt / "TRANSITION_MANIFEST.json"
    )
    strict.full_seal_check(args.transition_receipt)
    if (
        transition.get("schema") != strict.TRANSITION_SCHEMA
        or transition.get("collection_mode") != "canary"
    ):
        raise SystemExit("canary validator requires a V2 canary transition")
    allowlist, identities = strict.validate_allowlist(args.identity_allowlist)
    plan, _ = strict.validate_shard_plan(
        args.shard_plan, args.identity_allowlist, transition["n_shards"]
    )
    expected = {
        shard["identities"][0]["episode_id"]: shard["shard_id"]
        for shard in plan["shards"]
    }
    if len(expected) != plan["n_shards"]:
        raise SystemExit("canary first-identity set is not unique")

    episode_seals = {}
    worker_seals = {}
    for episode_id, shard_id in expected.items():
        identity = identities[episode_id]
        episode_root = output / "episodes" / episode_id
        strict.validate_episode_v2(episode_root, identity, transition)
        episode_seals[episode_id] = strict.sha256_file(episode_root / "SHA256SUMS")
        gpu = transition["shard_to_physical_gpu"][str(shard_id)]
        worker_root = output / f"gpu_{gpu}"
        strict.full_seal_check(worker_root)
        worker = strict.load_json(worker_root / "WORKER_MANIFEST.json")
        if (
            worker.get("schema") != "FIT670_ATOMIC_WORKER_V2"
            or worker.get("shard_id") != shard_id
            or worker.get("gpu") != gpu
            or worker.get("n_assigned") != 1
            or worker.get("n_fail") != 0
            or worker.get("n_success", 0) + worker.get("n_skipped", 0) != 1
        ):
            raise SystemExit(f"canary worker closure failed: shard {shard_id}")
        worker_seals[str(shard_id)] = strict.sha256_file(
            worker_root / "SHA256SUMS"
        )

    discovered = {
        json.loads(path.read_text(encoding="utf-8")).get("episode_id")
        for path in (output / "episodes").rglob("episode.json")
    }
    if discovered != set(expected):
        raise SystemExit(
            f"canary episode closure failed: expected={len(expected)} "
            f"observed={len(discovered)}"
        )

    staging = output / (
        f".CANARY_REVIEW_V2.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    published = False
    try:
        report = {
            "gate": "FIT670_V2_CANARY",
            "schema": "FIT670_CANARY_REVIEW_V2",
            "status": "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE",
            "n_shards": plan["n_shards"],
            "n_episodes": len(expected),
            "identity_set_digest": allowlist["identity_set_digest"],
            "shard_plan_sha256": strict.sha256_file(args.shard_plan),
            "transition_sha256sums_sha256": strict.sha256_file(
                args.transition_receipt / "SHA256SUMS"
            ),
            "collection_source_commit": transition["collection_source_commit"],
            "collection_source_tree": transition["collection_source_tree"],
            "episode_seals": episode_seals,
            "worker_seals": worker_seals,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staging / "CANARY_REVIEW.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        seal(staging)
        staging.rename(review_root)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    strict.full_seal_check(review_root)
    print(f"FIT670 V2 canary PASS: {review_root}")


if __name__ == "__main__":
    main()
