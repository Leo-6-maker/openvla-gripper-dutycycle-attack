#!/usr/bin/env python3
"""Independent audit for the sealed 24-run FIT viability aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import verify_sealed_directory, sha256_file


def audit_matrix(root: Path) -> dict:
    verify_sealed_directory(root)
    path = root / "viability_aggregate.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_FIT_VIABILITY_AGGREGATE_V1" or value.get("run_count") != 24:
        raise ValueError("viability aggregate schema/count mismatch")
    if value.get("formal_training_authorized") is not False or value.get("formal_attack_authorized") is not False:
        raise ValueError("viability aggregate authorization boundary failed")
    coordinates = {(int(row["fold_id"]), row["variant"], int(row["seed"])) for row in value.get("runs", [])}
    expected = {(fold, variant, seed) for fold in range(4) for variant in ("B3_25D", "B3_25D9D") for seed in (20260717, 20260718, 20260719)}
    if coordinates != expected or any(int(row.get("validation_identity_count", 0)) != 200 for row in value.get("runs", [])):
        raise ValueError("viability matrix coordinate or held-out count closure failed")
    if sha256_file(path) != (root / "viability_aggregate.json.sha256").read_text(encoding="utf-8").split()[0]:
        raise ValueError("viability aggregate sidecar mismatch")
    return {"schema": "B3_OFFICIAL_V3_FIT_VIABILITY_MATRIX_AUDIT_V1", "status": "PASS_PREPARATION_ONLY", "run_count": 24, "formal_training_authorized": False, "formal_attack_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_matrix(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
