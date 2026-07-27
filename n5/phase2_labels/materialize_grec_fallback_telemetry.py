"""Materialize geometry cases from sealed direct FIT telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from n5.phase2_labels.run_v23_recorded_geometry_grec import publish_noreplace
except ModuleNotFoundError:  # direct script execution from the repository
    from run_v23_recorded_geometry_grec import publish_noreplace


class MaterializeHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(x) for x in value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    return True


def sealed(root: Path) -> str:
    sums = root / "SHA256SUMS"; side = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not side.is_file():
        raise MaterializeHold(f"input root not sealed: {root}")
    parts = side.read_text(encoding="utf-8").split()
    if parts != [sha256_file(sums), "SHA256SUMS"]:
        raise MaterializeHold(f"input sidecar mismatch: {root}")
    expected = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise MaterializeHold(f"unsafe input seal path: {name}")
        target = root / rel
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise MaterializeHold(f"input checksum mismatch: {target}")
        expected.add(rel.as_posix())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}}
    if actual != expected:
        raise MaterializeHold(f"input file closure mismatch: {root}")
    return sha256_file(sums)


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    data = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows).encode()
    return hashlib.sha256(data).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise MaterializeHold(f"output exists: {args.output}")
    cases = []
    bindings = []
    for input_root in args.input_root:
        input_root = input_root.resolve()
        input_sha = sealed(input_root)
        manifest = json.loads((input_root / "FALLBACK_CANARY_MANIFEST.json").read_text(encoding="utf-8"))
        episode = json.loads((input_root / "episode.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "V23_G_REC_DATA_FALLBACK_CANARY_V1" or manifest.get("attack_enabled") is not False:
            raise MaterializeHold(f"input boundary mismatch: {input_root}")
        telemetry_rows = episode.get("telemetry", [])
        if [int(row.get("step", -1)) for row in telemetry_rows] != list(range(len(telemetry_rows))):
            raise MaterializeHold(f"telemetry step closure failed: {input_root}")
        entities_by_step = {int(row["step"]): {(str(entity["entity_type"]), int(entity["entity_id"])): entity for entity in row.get("entities", [])} for row in telemetry_rows}
        relations = episode.get("relations", [])
        if not relations:
            raise MaterializeHold(f"empty relation set: {input_root}")
        for step, entity_map in sorted(entities_by_step.items()):
            for relation_index, relation in enumerate(relations):
                resolved = []
                for side, role in (("object_resolution", "MANIPULATED_OBJECT"), ("target_resolution", "REGION_TARGET" if relation.get("target_is_region") else "OBJECT_TARGET")):
                    resolution = relation[side]
                    key = (str(resolution.get("entity_type")), int(resolution.get("entity_id", -1)))
                    entity = entity_map.get(key)
                    if entity is None:
                        raise MaterializeHold(f"missing entity telemetry: {episode['episode_id']}:{step}:{relation_index}:{side}")
                    if not finite(entity.get("world_pose")):
                        raise MaterializeHold(f"nonfinite entity telemetry: {episode['episode_id']}:{step}:{relation_index}:{side}")
                    resolved.append({"side": side, "role": role, "resolution": resolution, "entity": entity})
                cases.append({
                    "episode_id": episode["episode_id"], "suite": episode["suite"], "task_id": episode["task_id"], "state_id": episode["state_id"],
                    "step": step, "relation_index": relation_index, "predicate": relation.get("predicate"),
                    "object": {"role": resolved[0]["role"], "resolution": resolved[0]["resolution"], "pose": resolved[0]["entity"]["world_pose"], "source": "RECORDED_MUJOCO_WORLD_POSE"},
                    "target": {"role": resolved[1]["role"], "resolution": resolved[1]["resolution"], "pose": resolved[1]["entity"]["world_pose"], "source": "RECORDED_MUJOCO_WORLD_POSE"},
                    "reference_mode": "DIRECT_RECORDED_MUJOCO_WORLD_POSE",
                    "teacher_fields_present": False,
                    "attack_fields_present": False,
                })
        bindings.append({"root": str(input_root), "input_sha256sums_sha256": input_sha, "episode_id": episode["episode_id"], "step_count": len(entities_by_step), "relation_count": len(relations), "source_parent_identity": manifest.get("source_parent_identity")})
    cases.sort(key=lambda row: (row["episode_id"], row["step"], row["relation_index"]))
    final_parent = args.output.parent; final_parent.mkdir(parents=True, exist_ok=True)
    staging = final_parent / f".{args.output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise MaterializeHold(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    manifest = {"schema": "V23_G_REC_FALLBACK_GEOMETRY_CANARY_V1", "status": "DERIVED_FIT_ONLY_CANARY_NONCONSUMABLE", "run_label": args.run_label, "input_bindings": bindings, "case_count": len(cases), "canonical_cases_sha256": canonical_sha(cases), "protected_payload_read": False, "model_inference": False, "action_replay": False, "teacher_labels_generated": False, "consumer_eligible": False}
    (staging / "GEOMETRY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "geometry_cases.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}\n" for p in payload), encoding="utf-8")
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    if args.output.exists():
        raise MaterializeHold(f"output appeared during materialization: {args.output}")
    publish_noreplace(staging, args.output)
    return {"status": "PASS_NONCONSUMABLE_CANARY", "output": str(args.output), "case_count": len(cases), "canonical_cases_sha256": manifest["canonical_cases_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-label", required=True, choices=("run_A", "run_B"))
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
