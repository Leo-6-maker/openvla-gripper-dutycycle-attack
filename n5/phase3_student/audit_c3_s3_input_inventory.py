"""Seal the C3-S3 input inventory without opening episode semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

from c3_s3_input_contract import load_allowlist, require_allowed_path, sha256_file, verify_manifest_binding


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def seal(staging: Path, final: Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    if final.exists():
        raise FileExistsError(final)
    manifest["payload_files"] = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    write_json(staging / "MANIFEST.json", manifest)
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    rows = [f"{sha256_file(staging / name)}  {name}" for name in files if name not in {"SHA256SUMS", "SHA256SUMS.sha256"}]
    (staging / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    os.replace(staging, final)
    return {"root": str(final), "sha256sums_sha256": sums_sha}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    allowlist_path = Path(args.allowlist).resolve()
    allowlist = load_allowlist(allowlist_path)
    audit_events = []
    r6_root, r6_entry = require_allowed_path(Path(args.r6_root).resolve(), allowlist, regular=False, audit_events=audit_events)
    r6_binding = verify_manifest_binding(r6_root, r6_entry)
    candidates = []
    candidate = Path(args.observed_candidate)
    candidate_record = {"path": str(candidate), "metadata_only": True, "accepted_as_c3_s3_input": False}
    try:
        candidate_resolved, _ = require_allowed_path(candidate, allowlist, regular=False, audit_events=audit_events)
    except (ValueError, FileNotFoundError) as exc:
        candidate_record.update({"top_level_entries": [], "top_level_seal_present": False, "reason": f"UNVERIFIED_PATH: {exc}"})
    else:
        names = sorted(item.name for item in candidate_resolved.iterdir()) if candidate_resolved.is_dir() else []
        candidate_record.update({
            "top_level_entries": names,
            "top_level_seal_present": all((candidate_resolved / name).is_file() for name in ("SHA256SUMS", "SHA256SUMS.sha256")),
            "reason": "candidate metadata was allowlisted but is not accepted without an explicit C3-S3 geometry contract",
        })
    candidates.append(candidate_record)
    allowed_events = [event for event in audit_events if event.get("event") == "allowed_root_checked"]
    purposes = sorted({str(event["purpose"]) for event in allowed_events if event.get("purpose")})
    status = "PASS_INPUT_INVENTORY_NO_EPISODE_SOURCE" if allowlist.get("allowed_episode_geometry_roots") else "HOLD_INPUTS_MISSING"
    inventory = {
        "schema": "C3_S3_INPUT_INVENTORY_V1",
        "status": status,
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": sha256_file(allowlist_path),
        "r6_binding": r6_binding,
        "allowed_episode_geometry_root_count": len(allowlist.get("allowed_episode_geometry_roots", [])),
        "observed_candidates": candidates,
        "protected_reads": [str(event["path"]) for event in audit_events if event.get("read") is True],
        "validated_roots": sorted({str(event["root"]) for event in allowed_events}),
        "purpose": purposes[0] if len(purposes) == 1 else purposes,
        "verification_events": audit_events,
        "model_inference": False,
        "rollout_steps": 0,
        "attack_steps": 0,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "decision": "C3-S3 numerical replay remains HOLD until an explicit FIT-only telemetry/reference root is sealed and allowlisted",
    }
    parent = Path(args.out_parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (args.output_name or f"c3_s3_input_inventory_{args.code_commit[:8]}_{uuid.uuid4().hex[:8]}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        write_json(staging / "inventory.json", inventory)
        result = seal(staging, final, {
            "schema": "C3_S3_INPUT_INVENTORY_V1",
            "source_commit": args.source_commit,
            "code_commit": args.code_commit,
            "allowlist_sha256": inventory["allowlist_sha256"],
            "protected_reads": inventory["protected_reads"],
            "validated_roots": inventory["validated_roots"],
            "purpose": inventory["purpose"],
        })
        print(json.dumps({**result, "status": status}, sort_keys=True))
        return inventory
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--r6-root", required=True)
    parser.add_argument("--observed-candidate", required=True)
    parser.add_argument("--out-parent", required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
