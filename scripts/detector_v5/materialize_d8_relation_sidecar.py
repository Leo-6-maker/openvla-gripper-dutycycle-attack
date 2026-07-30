"""D8-1-R2: Per-step relation sidecar materializer.

Reads Teacher records (which already contain per-relation labels),
extracts per-relation physical_criticality evidence, and builds
a structured sidecar for the D8-1 consolidator identity checks.

Does NOT modify Teacher labels. Does NOT read G7 test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

FORBIDDEN = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8",
    )
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def _extract_relation_signature_fields(binding: dict | None, identity_entry: dict | None) -> dict:
    """Extract identity fields from relation binding + identity."""
    fields = {
        "logical_object": "", "logical_target": "",
        "selected_relation": "", "binding_identity": "",
        "entity_role": "", "entity_type": "",
        "object_entity_id": -1, "target_entity_id": -1,
        "object_name": "", "target_name": "",
    }
    if isinstance(binding, dict):
        obj = binding.get("object", {})
        tgt = binding.get("target", {})
        fields["logical_object"] = str(obj.get("logical_name", ""))
        fields["logical_target"] = str(tgt.get("logical_name", ""))
        fields["selected_relation"] = str(binding.get("predicate", ""))
        fields["entity_role"] = str(obj.get("role", ""))
        fields["object_name"] = str(obj.get("alias_to") or obj.get("logical_name", ""))
        fields["target_name"] = str(tgt.get("alias_to") or tgt.get("logical_name", ""))
    if isinstance(identity_entry, dict):
        fields["binding_identity"] = str(identity_entry.get("entity_id", ""))
        fields["object_entity_id"] = int(identity_entry.get("entity_id", -1))
        fields["entity_type"] = str(identity_entry.get("entity_type", ""))
    return fields


def build(
    teacher_root: Path,
    output_root: Path,
    allowlist_path: Path | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("clean checkout required")

    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")

    # Validate Teacher root
    teacher_root = teacher_root.resolve(strict=True)
    teacher_seal = verify_seal(teacher_root)
    teacher_manifest = json.loads((teacher_root / "teacher_manifest.json").read_text(encoding="utf-8"))

    expected_steps = teacher_manifest.get("step_count", 196483)
    expected_identities = teacher_manifest.get("identity_count", 670)

    # Read all Teacher records and extract per-relation data
    print(f"Reading Teacher records (expecting {expected_steps} steps)...")
    records_path = teacher_root / "teacher_records.jsonl"

    sidecar: dict[str, dict[int, dict]] = defaultdict(dict)  # eid -> step -> per_relation
    total_steps = 0
    identities_seen = set()
    raw_label_digest_parts = []

    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            eid = str(row["episode_id"])
            step = row["step"]
            identities_seen.add(eid)
            total_steps += 1

            # Collect raw label for digest
            pc = row.get("labels", {}).get("physical_criticality", {})
            raw_label_digest_parts.append(
                f"{eid}|{step}|{pc.get('value')}|{pc.get('reason')}|{pc.get('mask')}"
            )

            # Extract per-relation data
            relation_labels = row.get("relation_labels", [])
            relation_bindings = row.get("relation_bindings", [])
            relation_indices = row.get("relation_indices", [])
            relation_identity = row.get("relation_identity", [])
            candidate_close = row.get("candidate_close", False)

            per_rel = []
            for idx in relation_indices:
                if idx >= len(relation_labels) or idx >= len(relation_bindings):
                    continue
                rl = relation_labels[idx]
                rb = relation_bindings[idx]

                # Get identity entries for object and target
                obj_ident = None
                tgt_ident = None
                if isinstance(relation_identity, list):
                    for ident in relation_identity:
                        if isinstance(ident, dict) and ident.get("side") == "object":
                            if obj_ident is None:
                                obj_ident = ident
                        elif isinstance(ident, dict) and ident.get("side") == "target":
                            if tgt_ident is None:
                                tgt_ident = ident

                sig = _extract_relation_signature_fields(rb, obj_ident)
                # Also extract target identity fields
                if isinstance(tgt_ident, dict):
                    sig["target_entity_id"] = int(tgt_ident.get("entity_id", -1))
                    sig["target_name"] = str(tgt_ident.get("logical_name", ""))

                pc_rel = rl.get("labels", {}).get("physical_criticality", {})
                if isinstance(pc_rel, dict):
                    verdict = pc_rel.get("value", "UNKNOWN")
                    reason = pc_rel.get("reason", "")
                    evidence = pc_rel.get("evidence_fields", [])
                    mask = pc_rel.get("mask", False)
                    valid_mask = pc_rel.get("valid_mask", False)
                else:
                    verdict = "UNKNOWN"
                    reason = "RELATION_LABEL_MISSING"
                    evidence = []
                    mask = False
                    valid_mask = False

                per_rel.append({
                    "relation_index": idx,
                    "predicate": sig["selected_relation"],
                    "verdict": verdict,
                    "reason": reason,
                    "mask": mask,
                    "valid_mask": valid_mask,
                    "evidence_fields": evidence,
                    **sig,
                })

            # Determine selection status
            supporting = [r for r in per_rel if r["verdict"] == "TRUE"]
            if len(supporting) == 1:
                selected_id = supporting[0]["relation_index"]
                selection_status = "UNIQUE_SUPPORT"
            elif len(supporting) > 1:
                selected_id = None
                selection_status = "MULTI_SUPPORT"
            elif any(r["verdict"] == "FALSE" for r in per_rel):
                selected_id = None
                selection_status = "NO_SUPPORT"
            elif any(r["reason"] == "GEOMETRY_NOT_APPLICABLE" for r in per_rel):
                selected_id = None
                selection_status = "GEOMETRY_NOT_APPLICABLE"
            else:
                selected_id = None
                selection_status = "RELATION_AMBIGUOUS"

            candidate_ids = [r["relation_index"] for r in per_rel]

            sidecar_entry = {
                "episode_id": eid,
                "step": step,
                "suite": row.get("suite", ""),
                "task_id": row.get("task_id", -1),
                "state_id": row.get("state_id", -1),
                "seed": row.get("seed"),
                "candidate_close": candidate_close,
                "aggregate_physical_label": pc.get("value"),
                "aggregate_mask": pc.get("mask", False) and pc.get("valid_mask", False),
                "aggregate_reason": pc.get("reason", ""),
                "per_relation": per_rel,
                "selection_status": selection_status,
                "selected_relation_index": selected_id,
                "candidate_relation_indices": candidate_ids,
                "supporting_relation_indices": [r["relation_index"] for r in supporting],
            }
            sidecar[eid][step] = sidecar_entry

    # Verify closure
    if total_steps != expected_steps:
        raise ValueError(f"step count mismatch: {total_steps} != {expected_steps}")
    if len(identities_seen) != expected_identities:
        raise ValueError(f"identity count mismatch: {len(identities_seen)} != {expected_identities}")

    # Compute raw label digest
    raw_label_digest = hashlib.sha256(
        "\n".join(raw_label_digest_parts).encode()
    ).hexdigest()

    print(f"Processed {total_steps} steps across {len(identities_seen)} identities")
    print(f"Raw label digest: {raw_label_digest}")

    # Build output per episode
    if output_root.exists():
        raise FileExistsError(str(output_root))

    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)

    # Statistics
    stats = defaultdict(lambda: {
        "steps": 0, "unique_support": 0, "multi_support": 0, "no_support": 0,
        "geom_na": 0, "relation_ambiguous": 0,
    })

    for eid in sorted(sidecar):
        ep_data = sidecar[eid]
        suite = eid.split("/")[0] if "/" in eid else "?"
        for step in sorted(ep_data):
            entry = ep_data[step]
            stats[suite]["steps"] += 1
            ss = entry["selection_status"]
            if ss == "UNIQUE_SUPPORT":
                stats[suite]["unique_support"] += 1
            elif ss == "MULTI_SUPPORT":
                stats[suite]["multi_support"] += 1
            elif ss == "NO_SUPPORT":
                stats[suite]["no_support"] += 1
            elif ss == "GEOMETRY_NOT_APPLICABLE":
                stats[suite]["geom_na"] += 1
            elif ss == "RELATION_AMBIGUOUS":
                stats[suite]["relation_ambiguous"] += 1

        # Write per-episode sidecar
        ep_dir = staging / "per_episode"
        ep_dir.mkdir(parents=True, exist_ok=True)
        safe_name = eid.replace("/", "_")
        (ep_dir / f"{safe_name}.json").write_text(
            json.dumps(ep_data, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    # Build manifest
    payload = {
        "schema": "DETECTOR_V3_D8_RELATION_SIDECAR_V1",
        "status": "PASS_MATERIALIZED",
        "code_snapshot": {"commit": commit, "tree": tree},
        "teacher_root": str(teacher_root),
        "teacher_seal": teacher_seal["sha256sums_sha256"],
        "teacher_manifest_sha256": sha256_file(teacher_root / "teacher_manifest.json"),
        "teacher_schema_sha256": teacher_manifest.get("schema_sha256", ""),
        "expected_identities": expected_identities,
        "identities_found": len(identities_seen),
        "expected_steps": expected_steps,
        "steps_found": total_steps,
        "raw_label_digest": raw_label_digest,
        "selection_statistics": {k: dict(v) for k, v in stats.items()},
        "builder_sha256": sha256_file(Path(__file__)),
        "d8_consolidator_sha256": sha256_file(ROOT / "scripts" / "detector_v5" / "d8_event_consolidator.py"),
        "d8_protocol_sha256": sha256_file(ROOT / "configs" / "DETECTOR_V3_D8_EVENT_CONSOLIDATION_PROTOCOL.json"),
        "protected_reads": 0,
        "test_payload_read": 0,
    }

    (staging / "SIDECAR_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    payload["sha256sums_sha256"] = digest
    print(f"Sidecar sealed: {digest}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    result = build(args.teacher_root, args.output_root, num_workers=args.num_workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
