"""Independent verifier for direct-telemetry geometry canary runs."""

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


class VerifyHold(RuntimeError):
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


def read_run(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise VerifyHold(f"run root missing or symlink: {root}")
    sums = root / "SHA256SUMS"; side = root / "SHA256SUMS.sha256"
    if side.read_text(encoding="utf-8").split() != [sha256_file(sums), "SHA256SUMS"]:
        raise VerifyHold(f"seal sidecar mismatch: {root}")
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1); target = root / name
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise VerifyHold(f"seal mismatch: {target}")
        expected[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}}
    if set(expected) != actual:
        raise VerifyHold(f"file closure mismatch: {root}")
    manifest = json.loads((root / "GEOMETRY_MANIFEST.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (root / "geometry_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest.get("status") != "DERIVED_FIT_ONLY_CANARY_NONCONSUMABLE" or manifest.get("protected_payload_read") is not False or manifest.get("consumer_eligible") is not False:
        raise VerifyHold(f"manifest boundary mismatch: {root}")
    if int(manifest.get("case_count", -1)) != len(cases) or not cases:
        raise VerifyHold(f"case count mismatch: {root}")
    seen = set()
    for case in cases:
        key = (case.get("episode_id"), int(case.get("step", -1)), int(case.get("relation_index", -1)))
        if key in seen:
            raise VerifyHold(f"duplicate geometry case: {key}")
        seen.add(key)
        if case.get("reference_mode") != "DIRECT_RECORDED_MUJOCO_WORLD_POSE" or case.get("teacher_fields_present") is not False or case.get("attack_fields_present") is not False:
            raise VerifyHold(f"case provenance mismatch: {key}")
        if case.get("object", {}).get("source") != "RECORDED_MUJOCO_WORLD_POSE" or case.get("target", {}).get("source") != "RECORDED_MUJOCO_WORLD_POSE":
            raise VerifyHold(f"case source mismatch: {key}")
        if not finite(case.get("object")) or not finite(case.get("target")):
            raise VerifyHold(f"nonfinite geometry case: {key}")
    canonical = "".join(json.dumps(case, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for case in cases)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != manifest.get("canonical_cases_sha256"):
        raise VerifyHold(f"canonical digest mismatch: {root}")
    return {"root": str(root.resolve()), "run_label": manifest.get("run_label"), "case_count": len(cases), "canonical_cases_sha256": digest, "input_count": len(manifest.get("input_bindings", [])), "protected_payload_read": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        a = read_run(args.run_a.resolve()); b = read_run(args.run_b.resolve())
        if a["canonical_cases_sha256"] != b["canonical_cases_sha256"] or a["case_count"] != b["case_count"]:
            raise VerifyHold("A/B canonical geometry mismatch")
        result = {"schema": "V23_G_REC_FALLBACK_GEOMETRY_INDEPENDENT_REVIEW_V1", "status": "PASS_NONCONSUMABLE_CANARY", "run_A": a, "run_B": b, "canonical_equal": True, "protected_payload_read": False, "model_inference": False, "action_replay": False, "teacher_labels_generated": False}
        if args.output.exists() or args.output.is_symlink():
            raise VerifyHold(f"output exists: {args.output}")
        staging = args.output.parent / f".{args.output.name}.staging.{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise VerifyHold(f"staging exists: {staging}")
        staging.mkdir(parents=True)
        (staging / "INDEPENDENT_REVIEW.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "MANIFEST.json").write_text(json.dumps({"schema": "V23_G_REC_FALLBACK_GEOMETRY_INDEPENDENT_REVIEW_BUNDLE_V1", "status": result["status"], "protected_payload_read": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload = sorted(p for p in staging.rglob("*") if p.is_file())
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}\n" for p in payload), encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        publish_noreplace(staging, args.output)
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
