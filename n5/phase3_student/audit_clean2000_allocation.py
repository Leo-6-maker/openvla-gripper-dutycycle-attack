"""Metadata-only Clean2000/V22 allocation census.

This script reads only campaign and worker manifest JSON plus file names.  It
never opens episode payloads, labels, logs, or shards, and it fails closed when
the V22 identity root is not explicitly supplied and sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def walk_identity_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        suite = value.get("suite")
        task = value.get("task_idx", value.get("task_index"))
        state = value.get("state_id")
        if isinstance(suite, str) and isinstance(task, int) and isinstance(state, int):
            yield value
        for child in value.values():
            yield from walk_identity_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_identity_objects(child)


def manifest_files(root: Path) -> List[Path]:
    return sorted(root.glob("shards/*/worker_*/worker_manifest.json"))


def build_rows(c2f_root: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    worker_manifests = manifest_files(c2f_root)
    for worker_manifest in worker_manifests:
        data = json.loads(worker_manifest.read_text(encoding="utf-8"))
        worker = worker_manifest.parent.name
        suite_dir = worker_manifest.parent.parent.name
        for obj in walk_identity_objects(data):
            key = (str(obj["suite"]), int(obj["task_idx"] if "task_idx" in obj else obj["task_index"]), int(obj["state_id"]))
            rows.setdefault(key, {
                "episode_id": f"{key[0]}/task_{key[1]:02d}/state_{key[2]}",
                "suite": key[0],
                "task_id": key[1],
                "state_id": key[2],
                "seed": "UNKNOWN_METADATA_ONLY",
                "source_root": str(c2f_root),
                "worker_manifest": str(worker_manifest),
                "worker_manifest_sha256": sha256_file(worker_manifest),
                "worker_shard": f"{suite_dir}/{worker}",
                "historical_split": "UNKNOWN_METADATA_ONLY",
                "protected_status": "UNVERIFIED_NO_FIT_BINDING",
                "in_v22_800": "UNKNOWN_ROOT_NOT_MOUNTED",
                "has_v1_1_label": "UNKNOWN_PAYLOAD_NOT_READ",
                "has_v22_label": "UNKNOWN_ROOT_NOT_MOUNTED",
                "has_object_pose": "UNKNOWN_PAYLOAD_NOT_READ",
                "has_target_geometry": "UNKNOWN_PAYLOAD_NOT_READ",
                "has_init_state": "UNKNOWN_PAYLOAD_NOT_READ",
                "has_action_trace": "UNKNOWN_PAYLOAD_NOT_READ",
                "replayable": "UNKNOWN_PAYLOAD_NOT_READ",
                "v23_labelable": "UNKNOWN_PROVENANCE",
            })
    return list(rows.values()), {
        "worker_manifest_count": len(worker_manifests),
        "observed_identity_count": len(rows),
        "identity_extraction": "manifest_metadata_only",
        "payloads_read": False,
    }


def seal(staging: Path, final: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    if final.exists():
        raise FileExistsError(final)
    write_json(staging / "MANIFEST.json", manifest)
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    sums = staging / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256_file(staging / name)}  {name}" for name in names) + "\n", encoding="utf-8")
    sums_sha = sha256_file(sums)
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.replace(staging, final)
    return {"root": str(final), "sha256sums_sha256": sums_sha}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    c2f_root = Path(args.c2f_root).resolve()
    if not c2f_root.is_dir():
        raise ValueError(f"C2F root missing: {c2f_root}")
    rows, observed = build_rows(c2f_root)
    top_level_seal = all((c2f_root / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256"))
    v22_status = "ROOT_NOT_MOUNTED" if not args.v22_manifest else "NOT_EXECUTED_METADATA_ONLY"
    decision = "HOLD_PROVENANCE"
    summary = {
        "schema": "C3_S3_CLEAN2000_ALLOCATION_LEDGER_V1",
        "status": "HOLD_PROVENANCE",
        "decision": decision,
        "c2f_root": str(c2f_root),
        "c2f_top_level_seal_present": top_level_seal,
        "c2f_worker_shard_count": len(sorted(c2f_root.glob("shards/*/worker_*"))),
        "v22_manifest": args.v22_manifest,
        "v22_status": v22_status,
        "clean2000_claimed_total": 2000,
        "clean2000_claim_source": "user_statement_not_audited",
        "v22_claimed_total": 800,
        "v22_claim_source": "user_statement_not_audited",
        "observed": observed,
        "allocation_counts": {
            "clean2000_total": "UNVERIFIED",
            "v22_overlap": "UNVERIFIED",
            "unupgraded": "UNVERIFIED",
            "direct_v23_relabel": "UNVERIFIED",
            "deterministic_replay": "UNVERIFIED",
            "minimal_collection": "UNVERIFIED",
            "protected": "UNVERIFIED",
            "train": "UNVERIFIED",
            "validation": "UNVERIFIED",
            "test": "UNVERIFIED",
        },
        "protected_reads": [],
        "payloads_read": False,
        "model_inference": False,
        "training": False,
        "rollout": False,
        "attack": False,
    }
    parent = Path(args.out_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (args.output_name or f"clean2000_allocation_{uuid.uuid4().hex[:8]}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        with (staging / "allocation_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0]) if rows else ["episode_id", "status"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(staging / "summary.json", summary)
        seal(staging, final, {"schema": summary["schema"], "status": summary["status"], "payloads_read": False, "decision": decision})
        return summary | {"output_root": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2f-root", required=True)
    parser.add_argument("--v22-manifest")
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
