#!/usr/bin/env python3
"""Static feasibility audit for group-cross-fitted Factorized calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(plan_path: Path) -> dict[str, Any]:
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = value.get("jobs", value.get("checkpoints"))
    if not isinstance(jobs, list) or len(jobs) != 12:
        return {"status": "HOLD_SOURCE_INCOMPLETE", "reason": "12 checkpoint/manifest jobs are required", "production_inference": False}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for job in jobs:
        split = job.get("split", f"o{job.get('outer_fold')}_i{job.get('inner_fold')}")
        train = set(job.get("inner_train_identities", []))
        heldout = set(job.get("heldout_identities", []))
        if not train or not heldout or train & heldout:
            return {"status": "HOLD_SOURCE_INCOMPLETE", "reason": f"invalid train/heldout closure for {split}", "production_inference": False}
        if split in seen or split not in EXPECTED:
            return {"status": "HOLD_SOURCE_INCOMPLETE", "reason": f"invalid split {split}", "production_inference": False}
        seen.add(split)
        for field in ("checkpoint_path", "identity_manifest_path", "feature_root"):
            path = Path(str(job.get(field, "")))
            if not path.exists():
                missing.append(f"{split}:{field}:{path}")
        rows.append({"split": split, "train_count": len(train), "heldout_count": len(heldout), "heldout_identities": sorted(heldout)})
    if seen != set(EXPECTED) or missing:
        return {"status": "HOLD_SOURCE_INCOMPLETE", "reason": "required sealed roots are not mounted", "missing": missing, "rows": rows, "production_inference": False}
    coverage: dict[str, list[str]] = {}
    for row, job in zip(rows, jobs):
        for identity in row["heldout_identities"]:
            coverage.setdefault(identity, []).append(row["split"])
    eligible = {identity: splits for identity, splits in coverage.items() if len(splits) >= 1}
    return {
        "status": "STATIC_FEASIBILITY_PASS" if eligible else "HOLD_SOURCE_INCOMPLETE",
        "split_names": list(EXPECTED),
        "identity_prediction_coverage": {key: value for key, value in sorted(eligible.items())},
        "calibrator_fit_source": "inner-train predictions only; not executed",
        "policy_selection_source": "separate held-out group; not selected",
        "heldout_evaluation_source": "protected CAL/CHECK excluded",
        "production_inference": False,
        "source_plan_sha256": _sha(plan_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "STATIC_FEASIBILITY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
