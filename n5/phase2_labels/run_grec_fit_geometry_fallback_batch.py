"""Collect one suite of FIT-only clean telemetry with one model load."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import socket
import sys
from pathlib import Path

try:
    from n5.phase2_labels.run_v23_recorded_geometry_grec import publish_noreplace
    from n5.phase2_labels.run_grec_fit_geometry_fallback_canary import (
        HORIZONS, capture_episode, git_value, load_official_worker,
        reject_path, sha256_bytes, sha256_file, verify_source_record,
    )
except ModuleNotFoundError:
    from run_v23_recorded_geometry_grec import publish_noreplace
    from run_grec_fit_geometry_fallback_canary import (
        HORIZONS, capture_episode, git_value, load_official_worker,
        reject_path, sha256_bytes, sha256_file, verify_source_record,
    )


class BatchHold(RuntimeError):
    pass


def seal(root: Path) -> dict[str, str]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "sidecar_sha256": sha256_file(root / "SHA256SUMS.sha256"), "file_count": len(files)}


def child_manifest(args: argparse.Namespace, collection: dict, record: dict, source_meta: dict, registry_sha: str, model_tree_sha: str, worker_sha: str, collector_commit: str, collector_tree: str, libero_root: Path, pilot_sha: str, alias_sha: str) -> dict:
    return {
        "schema": "V23_G_REC_DATA_FALLBACK_CANARY_V1",
        "status": "DERIVED_FIT_ONLY_CLEAN_TELEMETRY",
        "source_parent_identity": record["episode_id"],
        "collection_identity_is_original_payload": False,
        "pilot_manifest_sha256": pilot_sha,
        "registry_task_sha256": registry_sha,
        "alias_ledger_sha256": alias_sha,
        "official_worker_sha256": worker_sha,
        "collector_source_commit": collector_commit,
        "collector_source_tree": collector_tree,
        "upstream_root": str(args.upstream_root.resolve()),
        "upstream_commit": git_value(args.upstream_root, "rev-parse", "HEAD"),
        "libero_root": str(libero_root),
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": model_tree_sha,
        "step_count": collection["step_count"],
        "relation_count": len(collection.get("relations", [])),
        "official_horizon": HORIZONS[args.suite],
        "environment": {"python": sys.executable, "python_version": platform.python_version(), "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "no_detector": True,
        "model_inference": True,
        "attack_enabled": False,
        "teacher_labels_generated": False,
        "protected_payload_read": False,
        "source_mode": "NEW_FIT_ONLY_CLEAN_RUNTIME_TELEMETRY",
        "original_payload_target_pose_available": False,
    }


def run(args: argparse.Namespace) -> dict:
    for path in (args.model_path, args.upstream_root, args.official_worker, args.pilot_manifest, args.registry_root, args.alias_ledger):
        reject_path(path)
    if args.output.exists() or args.output.is_symlink():
        raise BatchHold(f"output exists: {args.output}")
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("protected_payload_read") is not False or pilot.get("no_attack") is not True:
        raise BatchHold("pilot boundary failed")
    records = sorted((row for row in pilot.get("records", []) if row.get("suite") == args.suite), key=lambda row: row["episode_id"])
    if args.episode_id:
        wanted = set(args.episode_id)
        records = [row for row in records if row.get("episode_id") in wanted]
        if len(records) != len(wanted):
            raise BatchHold(f"requested episode mismatch: {sorted(wanted - {row['episode_id'] for row in records})}")
    if not records:
        raise BatchHold(f"no records for {args.suite}")

    source_rows = []
    for record in records:
        source_meta = verify_source_record(record)
        declared = source_meta["metadata"].get("checkpoint_path_verified") or source_meta["metadata"].get("checkpoint_path_declared")
        if not isinstance(declared, str) or Path(declared).resolve() != args.model_path.resolve():
            raise BatchHold(f"checkpoint path binding failed: {record['episode_id']}")
        source_rows.append((record, source_meta))
    alias = json.loads(args.alias_ledger.read_text(encoding="utf-8"))
    if not isinstance(alias.get("entries"), list):
        raise BatchHold("alias ledger missing")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    module = load_official_worker(args.official_worker, args, args.pilot_manifest)
    module.set_official_seed(args.seed)
    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(model, processor, device, unnorm_key, center_crop=True, base_vla_name=str(args.model_path))
    from libero.libero import get_libero_path
    from libero.libero import benchmark
    suite_instance = benchmark.get_benchmark_dict()[args.suite]()
    libero_root = Path(get_libero_path("bddl_files")).resolve().parents[2]
    worker_sha = sha256_file(args.official_worker)
    collector_commit = git_value(args.official_worker.parent.parent, "rev-parse", "HEAD")
    collector_tree = git_value(args.official_worker.parent.parent, "rev-parse", "HEAD^{tree}")
    model_tree_sha = module.checkpoint_tree_fingerprint(args.model_path)[0]
    pilot_sha = sha256_file(args.pilot_manifest)
    alias_sha = sha256_file(args.alias_ledger)
    child_records = []
    final_parent = args.output.parent
    final_parent.mkdir(parents=True, exist_ok=True)
    staging = final_parent / f".{args.output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise BatchHold(f"staging exists: {staging}")
    (staging / "episodes").mkdir(parents=True)
    try:
        for record, source_meta in source_rows:
            task_id = int(record["task_id"]); state_id = int(record["state_id"])
            task = suite_instance.get_task(task_id)
            state = suite_instance.get_task_init_states(task_id)[state_id]
            if sha256_bytes(pickle.dumps(state, protocol=4)) != source_meta["metadata"].get("initial_state_sha256"):
                raise BatchHold(f"initial state binding failed: {record['episode_id']}")
            collection = capture_episode(module, args, record, args.registry_root, state, task, args.seed, adapter)
            slug = record["episode_id"].replace("/", "__")
            child = staging / "episodes" / slug
            child.mkdir()
            manifest = child_manifest(args, collection, record, source_meta, collection["registry_task_sha256"], model_tree_sha, worker_sha, collector_commit, collector_tree, libero_root, pilot_sha, alias_sha)
            (child / "FALLBACK_CANARY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (child / "episode.json").write_text(json.dumps(collection, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            (child / "SEAL_RECEIPT.json").write_text(json.dumps({"schema": "V23_G_REC_DATA_FALLBACK_CANARY_SEAL_V1", "status": "SEALED_AFTER_PAYLOAD", "episode_id": record["episode_id"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            child_seal = seal(child)
            child_records.append({"episode_id": record["episode_id"], "relative_root": str(child.relative_to(staging).as_posix()), "step_count": collection["step_count"], "relation_count": len(collection.get("relations", [])), "sha256sums_sha256": child_seal["sha256sums_sha256"]})
    finally:
        del adapter, model, processor
    top = {"schema": "V23_G_REC_DATA_FALLBACK_BATCH_V1", "status": "DERIVED_FIT_ONLY_CLEAN_TELEMETRY_BATCH", "suite": args.suite, "records": child_records, "record_count": len(child_records), "protected_payload_read": False, "no_detector": True, "attack_enabled": False, "teacher_labels_generated": False, "model_inference": True, "consumer_eligible": False, "source_mode": "NEW_FIT_ONLY_CLEAN_RUNTIME_TELEMETRY"}
    (staging / "BATCH_MANIFEST.json").write_text(json.dumps(top, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "SEAL_RECEIPT.json").write_text(json.dumps({"schema": "V23_G_REC_DATA_FALLBACK_BATCH_SEAL_V1", "status": "SEALED_AFTER_PAYLOAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    top_seal = seal(staging)
    publish_noreplace(staging, args.output)
    return {"status": "PASS_FIT_ONLY_BATCH", "suite": args.suite, "record_count": len(child_records), "output": str(args.output), "seal": top_seal}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=sorted(HORIZONS)); p.add_argument("--gpu", type=int, required=True); p.add_argument("--worker-id", required=True)
    p.add_argument("--model-path", type=Path, required=True); p.add_argument("--upstream-root", type=Path, required=True); p.add_argument("--official-worker", type=Path, required=True)
    p.add_argument("--pilot-manifest", type=Path, required=True); p.add_argument("--registry-root", type=Path, required=True); p.add_argument("--alias-ledger", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--episode-id", action="append")
    try:
        print(json.dumps(run(p.parse_args()), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc), "attack_enabled": False, "detector_loaded": False}, sort_keys=True)); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
