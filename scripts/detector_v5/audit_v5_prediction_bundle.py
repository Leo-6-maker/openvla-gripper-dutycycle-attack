"""Fail-closed audit for a causal V5 prediction/scheduler bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import verify_sealed_directory


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    verify_sealed_directory(root)
    summary = json.loads((root / "evaluation_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    current_fields = {
        "canonical_parent_key", "step", "utility_probability", "release_probability",
        "regrasp_probability", "candidate_close", "student_valid", "raw_quality_emit",
        "candidate_gated_emit", "scheduler_emit",
    }
    legacy_fields = {"canonical_parent_key", "step", "utility_probability", "release_probability", "regrasp_probability", "candidate_close", "student_valid", "scheduler_emit"}
    schema = str(summary.get("schema", ""))
    prediction_fields = current_fields if schema == "DETECTOR_V5_CAUSAL_ONLINE_EVALUATION_V3" else legacy_fields
    emitted: dict[str, int] = {}
    steps_by_identity: dict[str, list[int]] = {}
    for line in (root / "prediction_records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if set(row) != prediction_fields:
            raise ValueError("prediction row field whitelist mismatch")
        identity = str(row["canonical_parent_key"])
        step = int(row["step"])
        steps_by_identity.setdefault(identity, []).append(step)
        if schema == "DETECTOR_V5_CAUSAL_ONLINE_EVALUATION_V3":
            expected_gate = bool(row["student_valid"] and row["candidate_close"] and row["raw_quality_emit"])
            if bool(row["candidate_gated_emit"]) != expected_gate:
                raise ValueError("candidate-close/valid gate mismatch")
        if row["scheduler_emit"]:
            emitted[identity] = emitted.get(identity, 0) + 1
    if any(count > 1 for count in emitted.values()) or summary.get("one_shot_compliance") is not True:
        raise ValueError("one-shot scheduler contract failed")
    if not steps_by_identity or any(steps != list(range(len(steps))) for steps in steps_by_identity.values()):
        raise ValueError("prediction step coverage is not contiguous")
    if summary.get("protected_splits_read") not in (None, []):
        raise ValueError("prediction bundle records protected split access")
    if schema == "DETECTOR_V5_CAUSAL_ONLINE_EVALUATION_V3":
        if manifest.get("schema") != "DETECTOR_V5_CAUSAL_ONLINE_BUNDLE_V3":
            raise ValueError("prediction manifest schema mismatch")
        if summary.get("working_point_status") not in {"PASS", "HOLD"}:
            raise ValueError("missing working-point status")
        if summary.get("selected_one_shot_compliance") is not True:
            raise ValueError("selected working point is not one-shot compliant")
    if summary.get("formal_training_authorized") is not False or summary.get("formal_attack_authorized") is not False:
        raise ValueError("prediction bundle carries an authorization flag")
    return {"schema": "DETECTOR_V5_PREDICTION_BUNDLE_AUDIT_V2", "status": "PASS", "emitted_episode_count": len(emitted), "formal_training_authorized": False, "formal_attack_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
