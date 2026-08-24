"""Audit the FIT-only DEV payload for recorded geometry telemetry.

This is a metadata/schema audit.  It does not load a model, replay actions,
read protected data, or generate Teacher labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class AuditHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from walk(child, child_path)
    elif isinstance(value, list):
        yield path, "list", len(value), all(isinstance(x, (int, float)) for x in value)
        for index, child in enumerate(value[:2]):
            yield from walk(child, f"{path}[{index}]")
    else:
        yield path, type(value).__name__, None, False


def field_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths: dict[str, Counter[str]] = defaultdict(Counter)
    dimensions: dict[str, Counter[str]] = defaultdict(Counter)
    finite = Counter()
    nonfinite = Counter()
    for row in rows:
        for path, kind, length, numeric_list in walk(row):
            paths[path][kind] += 1
            if length is not None:
                dimensions[path][str(length)] += 1
            if isinstance(row, dict) and path in row and isinstance(row[path], list):
                values = row[path]
                if numeric_list:
                    for value in values:
                        if math.isfinite(float(value)):
                            finite[path] += 1
                        else:
                            nonfinite[path] += 1
    return {
        "paths": {k: dict(v) for k, v in sorted(paths.items())},
        "dimensions": {k: dict(v) for k, v in sorted(dimensions.items())},
        "finite_numeric_values": dict(sorted(finite.items())),
        "nonfinite_numeric_values": dict(sorted(nonfinite.items())),
    }


def contains_pose_path(path: str) -> bool:
    lowered = path.lower()
    pose_token = any(token in lowered for token in ("pose", "position", "quat", "xpos", "xquat", "body_pos", "site_pos"))
    target_token = any(token in lowered for token in ("target", "site", "fixture", "receptacle"))
    return pose_token and target_token


def audit(output: Path, pilot_manifest: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise AuditHold(f"output exists: {output}")
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("protected_payload_read") is not False:
        raise AuditHold("pilot does not prove protected exclusion")
    records = pilot.get("records", [])
    if len(records) != 40 or len({r.get("episode_id") for r in records}) != 40:
        raise AuditHold("pilot identity closure is not exactly 40 unique episodes")

    episode_rows = []
    collector_counts = Counter()
    collector_component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    worker_counts = Counter()
    schema_counts = Counter()
    official_seed_count = 0
    initial_state_digest_count = 0
    step_total = sidecar_total = 0
    metadata_field_counts: Counter[str] = Counter()
    step_field_counts: Counter[str] = Counter()
    sidecar_field_counts: Counter[str] = Counter()
    target_pose_paths: Counter[str] = Counter()
    initial_state_payload_paths: Counter[str] = Counter()
    source_roots = set()
    for record in records:
        root = Path(record["source_episode_root"]).resolve()
        source_roots.add(str(root.parent.parent.parent.parent))
        files = {item["name"]: item for item in record.get("source_files", [])}
        required = {"episode_metadata.json", "step_records.jsonl", "privileged_teacher_sidecar.jsonl"}
        if set(files) != required:
            raise AuditHold(f"source file closure mismatch: {record['episode_id']}")
        payload = {}
        for name in required:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise AuditHold(f"source file is not regular: {path}")
            if sha256_file(path) != files[name]["sha256"]:
                raise AuditHold(f"source SHA mismatch: {path}")
            payload[name] = path
        metadata = json.loads(payload["episode_metadata.json"].read_text(encoding="utf-8"))
        steps = [json.loads(line) for line in payload["step_records.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
        sidecar = [json.loads(line) for line in payload["privileged_teacher_sidecar.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = int(record["observed_step_count"])
        if len(steps) != expected or len(sidecar) != expected:
            raise AuditHold(f"step coverage mismatch: {record['episode_id']}")
        if [int(x.get("step", -1)) for x in steps] != list(range(expected)):
            raise AuditHold(f"step identity mismatch: {record['episode_id']}")
        if [int(x.get("step", -1)) for x in sidecar] != list(range(expected)):
            raise AuditHold(f"sidecar identity mismatch: {record['episode_id']}")
        collector_map = metadata.get("collector_source_sha256")
        if isinstance(collector_map, dict):
            collector = json.dumps({str(k): str(v) for k, v in sorted(collector_map.items())}, sort_keys=True, separators=(",", ":"))
            for component, digest in collector_map.items():
                collector_component_counts[str(component)][str(digest)] += 1
        else:
            collector = str(collector_map or "MISSING")
        collector_counts[collector] += 1
        worker_counts[str(metadata.get("collector_worker_id") or "MISSING")] += 1
        official_seed_count += int("official_seed" in metadata)
        initial_state_digest_count += int(bool(metadata.get("initial_state_sha256")))
        schema_counts[str(metadata.get("schema_version") or record.get("schema_version") or "MISSING")] += 1
        metadata_keys = sorted(metadata)
        step_keys = sorted({key for row in steps for key in row})
        sidecar_keys = sorted({key for row in sidecar for key in row})
        metadata_field_counts.update(metadata_keys)
        step_field_counts.update(step_keys)
        sidecar_field_counts.update(sidecar_keys)
        metadata_paths = field_summary([metadata])["paths"]
        step_paths = field_summary(steps)["paths"]
        sidecar_paths = field_summary(sidecar)["paths"]
        pose_paths = sorted({path for path in set(metadata_paths) | set(step_paths) | set(sidecar_paths) if contains_pose_path(path)})
        for path in pose_paths:
            target_pose_paths[path] += 1
        initial_paths = sorted(path for path in metadata_paths if "initial" in path.lower() and ("state" in path.lower() or "pose" in path.lower()))
        for path in initial_paths:
            initial_state_payload_paths[path] += 1
        object_state_dims = sorted({len(row["object_state"]) for row in sidecar if isinstance(row.get("object_state"), list)})
        step_total += len(steps)
        sidecar_total += len(sidecar)
        episode_rows.append({
            "episode_id": record["episode_id"],
            "suite": record["suite"],
            "task_id": record["task_id"],
            "state_id": record["state_id"],
            "step_count": expected,
            "collector_source_sha256": collector,
            "schema_version": metadata.get("schema_version"),
            "official_seed_present": "official_seed" in metadata,
            "initial_state_sha256_present": bool(metadata.get("initial_state_sha256")),
            "initial_state_payload_paths": initial_paths,
            "metadata_keys": metadata_keys,
            "step_keys": step_keys,
            "sidecar_keys": sidecar_keys,
            "object_state_dimensions": object_state_dims,
            "target_pose_paths": pose_paths,
            "source_root": str(root),
        })
    status = "PASS_SCHEMA_METADATA_ONLY"
    result = {
        "schema": "V23_G_REC_RECORDED_TELEMETRY_AUDIT_V1",
        "status": status,
        "purpose": "FIT-only payload schema/provenance audit; no model, replay, labeling, protected read, or attack",
        "pilot_manifest": str(pilot_manifest.resolve()),
        "pilot_manifest_sha256": sha256_file(pilot_manifest),
        "protected_payload_read": False,
        "model_inference": False,
        "action_replay": False,
        "episode_count": len(episode_rows),
        "step_count": step_total,
        "sidecar_step_count": sidecar_total,
        "source_roots": sorted(source_roots),
        "collector_source_sha256_counts": dict(sorted(collector_counts.items())),
        "collector_component_sha256_counts": {component: dict(sorted(values.items())) for component, values in sorted(collector_component_counts.items())},
        "collector_worker_id_counts": dict(sorted(worker_counts.items())),
        "official_seed_present_count": official_seed_count,
        "initial_state_sha256_present_count": initial_state_digest_count,
        "schema_version_counts": dict(sorted(schema_counts.items())),
        "metadata_field_episode_counts": dict(sorted(metadata_field_counts.items())),
        "step_field_episode_counts": dict(sorted(step_field_counts.items())),
        "sidecar_field_episode_counts": dict(sorted(sidecar_field_counts.items())),
        "target_pose_path_episode_counts": dict(sorted(target_pose_paths.items())),
        "initial_state_payload_path_episode_counts": dict(sorted(initial_state_payload_paths.items())),
        "episodes": episode_rows,
    }
    staging = output.parent / f".{output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise AuditHold(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "TELEMETRY_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "TELEMETRY_AUDIT_EPISODES.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_rows), encoding="utf-8")
    manifest = {
        "schema": "V23_G_REC_RECORDED_TELEMETRY_AUDIT_BUNDLE_V1",
        "status": status,
        "pilot_manifest": str(pilot_manifest.resolve()),
        "pilot_manifest_sha256": sha256_file(pilot_manifest),
        "episode_count": len(episode_rows),
        "step_count": step_total,
        "protected_payload_read": False,
        "model_inference": False,
        "action_replay": False,
    }
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in payload), encoding="utf-8")
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    os.rename(staging, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.output.resolve(), args.pilot_manifest.resolve())
    except Exception as exc:
        print(json.dumps({"schema": "V23_G_REC_RECORDED_TELEMETRY_AUDIT_V1", "status": "HOLD", "error_type": type(exc).__name__, "error": str(exc), "protected_payload_read": False}, sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "episode_count": result["episode_count"], "step_count": result["step_count"], "collector_source_sha256_counts": result["collector_source_sha256_counts"], "collector_component_sha256_counts": result["collector_component_sha256_counts"], "target_pose_path_episode_counts": result["target_pose_path_episode_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
