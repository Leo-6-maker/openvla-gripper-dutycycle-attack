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
    prediction_fields = {"canonical_parent_key", "step", "utility_probability", "release_probability", "regrasp_probability", "candidate_close", "student_valid", "scheduler_emit"}
    emitted: dict[str, int] = {}
    for line in (root / "prediction_records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if set(row) != prediction_fields:
            raise ValueError("prediction row field whitelist mismatch")
        if row["scheduler_emit"]:
            emitted[row["canonical_parent_key"]] = emitted.get(row["canonical_parent_key"], 0) + 1
    if any(count > 1 for count in emitted.values()) or summary.get("one_shot_compliance") is not True:
        raise ValueError("one-shot scheduler contract failed")
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
