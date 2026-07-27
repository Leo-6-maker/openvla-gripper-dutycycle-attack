"""Independent CPU audit for a sealed G-REC fallback canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(finite_tree(x) for x in value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    return True


def verify_root(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"; sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise AuditHold(f"seal files missing: {root}")
    parts = sidecar.read_text(encoding="utf-8").split()
    if parts != [sha256_file(sums), "SHA256SUMS"]:
        raise AuditHold(f"seal sidecar mismatch: {root}")
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise AuditHold(f"unsafe seal path: {name}")
        target = root / rel
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise AuditHold(f"sealed payload mismatch: {target}")
        expected[rel.as_posix()] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}}
    if set(expected) != actual:
        raise AuditHold(f"seal file closure mismatch: {root}")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": len(expected)}


def audit_one(root: Path) -> dict[str, Any]:
    seal = verify_root(root)
    manifest = json.loads((root / "FALLBACK_CANARY_MANIFEST.json").read_text(encoding="utf-8"))
    episode = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "V23_G_REC_DATA_FALLBACK_CANARY_V1" or manifest.get("status") != "DERIVED_FIT_ONLY_CLEAN_TELEMETRY":
        raise AuditHold(f"manifest contract failed: {root}")
    if manifest.get("attack_enabled") is not False or manifest.get("no_detector") is not True or manifest.get("teacher_labels_generated") is not False:
        raise AuditHold(f"execution boundary failed: {root}")
    steps = episode.get("steps", []); telemetry = episode.get("telemetry", [])
    if not steps or len(steps) != len(telemetry):
        raise AuditHold(f"step/telemetry count mismatch: {root}")
    if [int(x.get("step", -1)) for x in steps] != list(range(len(steps))):
        raise AuditHold(f"step sequence mismatch: {root}")
    if [int(x.get("step", -1)) for x in telemetry] != list(range(len(telemetry))):
        raise AuditHold(f"telemetry sequence mismatch: {root}")
    action_parity = 0.0
    detector_count = 0
    attack_mutations = 0
    bad_generation = 0
    nonfinite = 0
    contact_invalid = 0
    entity_count = 0
    for step, record in zip(steps, telemetry):
        if int(step.get("generation_passes_per_step", -1)) != 1 or step.get("single_generation_parity_pass") is not True:
            bad_generation += 1
        raw = step.get("action_raw_7d", []); score = step.get("score_action_7d", []); env = step.get("action_env_7d", [])
        if len(raw) != 7 or len(score) != 7 or len(env) != 7:
            raise AuditHold(f"action shape mismatch: {root}:{step.get('step')}")
        action_parity = max(action_parity, max(abs(float(x) - float(y)) for x, y in zip(raw, score)))
        detector_count += int(step.get("detector_loaded", False) is True)
        attack_mutations += int(step.get("action_mutation_by_detector", True) is not False)
        if not finite_tree(record):
            nonfinite += 1
        if record.get("contact_capture_valid") is not True:
            contact_invalid += 1
        entity_count += len(record.get("entities", []))
    if bad_generation or detector_count or attack_mutations or nonfinite:
        raise AuditHold(f"runtime boundary failed: generation={bad_generation}, detector={detector_count}, action_mutation={attack_mutations}, nonfinite={nonfinite}")
    return {
        "root": str(root.resolve()), "episode_id": episode.get("episode_id"), "suite": episode.get("suite"),
        "step_count": len(steps), "relation_count": len(episode.get("relations", [])),
        "entity_pose_rows": entity_count, "contact_invalid_steps": contact_invalid,
        "generation_bad_steps": bad_generation, "detector_steps": detector_count,
        "action_mutation_steps": attack_mutations, "nonfinite_telemetry_steps": nonfinite,
        "max_action_parity_error": action_parity, "seal": seal,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise AuditHold(f"output exists: {args.output}")
        rows = [audit_one(root.resolve()) for root in args.root]
        result = {"schema": "V23_G_REC_DATA_FALLBACK_CANARY_AUDIT_V1", "status": "PASS", "canaries": rows, "protected_payload_read": False, "model_inference": False, "attack": False}
        staging = args.output.parent / f".{args.output.name}.staging.{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise AuditHold(f"staging exists: {staging}")
        staging.mkdir(parents=True)
        (staging / "AUDIT_REPORT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload = sorted(p for p in staging.rglob("*") if p.is_file())
        (staging / "MANIFEST.json").write_text(json.dumps({"schema": "V23_G_REC_DATA_FALLBACK_CANARY_AUDIT_BUNDLE_V1", "status": "PASS", "canary_count": len(rows), "protected_payload_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload = sorted(p for p in staging.rglob("*") if p.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}\n" for p in payload), encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, args.output)
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
