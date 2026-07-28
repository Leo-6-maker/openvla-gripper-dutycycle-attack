"""Strict FIT670 V2 finalizer.

It never mutates episode roots.  It verifies every recursive episode seal and
publishes a small immutable FINALIZATION_V2 receipt bound to all 670 episode
seal digests.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import fit670_strict_contract as strict


def seal_root(root: Path) -> str:
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text(
        "\n".join(
            f"{strict.sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in paths
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

    output_root = args.output_root.resolve()
    final_root = output_root / "FINALIZATION_V2"
    if final_root.exists():
        raise SystemExit(f"finalization already exists: {final_root}")
    strict.full_seal_check(args.transition_receipt)
    transition = strict.load_json(
        args.transition_receipt / "TRANSITION_MANIFEST.json"
    )
    if transition.get("schema") != strict.TRANSITION_SCHEMA:
        raise SystemExit("finalizer rejects legacy transition")
    allowlist, expected = strict.validate_allowlist(args.identity_allowlist)
    plan, membership = strict.validate_shard_plan(
        args.shard_plan, args.identity_allowlist, transition["n_shards"]
    )
    if transition.get("identity_set_digest") != allowlist["identity_set_digest"]:
        raise SystemExit("transition/allowlist identity digest mismatch")
    if transition.get("shard_plan_sha256") != strict.sha256_file(args.shard_plan):
        raise SystemExit("transition/shard-plan SHA mismatch")
    if str(output_root) not in [
        str(Path(value).resolve())
        for value in transition.get("allowed_output_roots", [])
    ]:
        raise SystemExit("finalizer output root is not transition-authorized")

    episode_root = output_root / "episodes"
    if not episode_root.is_dir():
        raise SystemExit(f"missing episodes root: {episode_root}")
    found_paths = [
        path.parent for path in episode_root.rglob("episode.json")
        if path.is_file()
    ]
    if len(found_paths) != 670:
        raise SystemExit(f"expected exactly 670 episode directories, got {len(found_paths)}")

    observed = {}
    episode_seals = {}
    for path in sorted(found_paths):
        data = json.loads((path / "episode.json").read_text(encoding="utf-8"))
        episode_id = data.get("episode_id")
        if episode_id in observed:
            raise SystemExit(f"duplicate episode_id: {episode_id}")
        if episode_id not in expected:
            raise SystemExit(f"unallowlisted episode: {episode_id}")
        expected_path = episode_root / episode_id
        if path.resolve() != expected_path.resolve():
            raise SystemExit(f"episode path/identity mismatch: {path} vs {episode_id}")
        strict.validate_episode_v2(path, expected[episode_id], transition)
        observed[episode_id] = str(path)
        episode_seals[episode_id] = strict.sha256_file(path / "SHA256SUMS")
    if set(observed) != set(expected):
        raise SystemExit(
            f"identity closure failed: missing={len(set(expected)-set(observed))} "
            f"extra={len(set(observed)-set(expected))}"
        )

    mapping = transition["shard_to_physical_gpu"]
    worker_results = {}
    for shard_id in range(plan["n_shards"]):
        gpu = mapping[str(shard_id)]
        root = output_root / f"gpu_{gpu}"
        strict.full_seal_check(root)
        manifest = strict.load_json(root / "WORKER_MANIFEST.json")
        assigned = sum(1 for value in membership.values() if value == shard_id)
        if (
            manifest.get("shard_id") != shard_id
            or manifest.get("gpu") != gpu
            or manifest.get("n_assigned") != assigned
            or manifest.get("n_fail") != 0
            or manifest.get("n_success", 0) + manifest.get("n_skipped", 0) != assigned
        ):
            raise SystemExit(f"worker manifest closure failed: shard {shard_id}")
        worker_results[str(shard_id)] = {
            "physical_gpu": gpu,
            "n_assigned": assigned,
            "n_success": manifest.get("n_success"),
            "n_skipped": manifest.get("n_skipped"),
            "worker_sha256sums_sha256": strict.sha256_file(root / "SHA256SUMS"),
        }

    residues = [
        str(path)
        for path in output_root.parent.iterdir()
        if path.name.startswith(".") and "staging" in path.name
    ]
    if residues:
        raise SystemExit(f"staging residue present: {residues[:10]}")

    staging = output_root / (
        f".FINALIZATION_V2.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    published = False
    try:
        manifest = {
            "gate": "FIT670_ATOMIC_COLLECTION_V2",
            "schema": "FIT670_GLOBAL_FINALIZATION_V2",
            "status": "PASS_CONSUMABLE",
            "n_identities_expected": 670,
            "n_identities_found": 670,
            "identity_set_digest": allowlist["identity_set_digest"],
            "shard_plan_sha256": strict.sha256_file(args.shard_plan),
            "transition_sha256sums_sha256": strict.sha256_file(
                args.transition_receipt / "SHA256SUMS"
            ),
            "collection_source_commit": transition["collection_source_commit"],
            "collection_source_tree": transition["collection_source_tree"],
            "worker_results": worker_results,
            "episode_seal_digest": strict.canonical_json_sha(episode_seals),
            "episode_seals": episode_seals,
            "staging_residue": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staging / "GLOBAL_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        seal_root(staging)
        staging.rename(final_root)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    strict.full_seal_check(final_root)
    print(f"FIT670 V2 PASS_CONSUMABLE: {final_root}")
    print(f"episodes: 670")
    print(f"episode_seal_digest: {manifest['episode_seal_digest']}")


if __name__ == "__main__":
    main()
