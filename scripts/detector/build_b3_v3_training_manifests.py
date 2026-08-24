#!/usr/bin/env python3
"""Build a non-authorizing, exact FIT training manifest.

This script only validates registry identity/quota and records input hashes. It
does not read Teacher rows, train a model, or authorize a formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gripper_attack.b3_v3_dataset import load_formal_registry_csv


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_training_manifest(registry_csv: Path, *, protocol_sha256: str = "") -> dict[str, Any]:
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    return {
        "schema": "B3_OFFICIAL_V3_TRAINING_MANIFEST_V1",
        "status": "PREPARATION_ONLY",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "registry_csv_sha256": sha256_file(registry_csv),
        "protocol_sha256": protocol_sha256,
        "identity_count": len(rows),
        "fit_train_count": len(rows),
        "task_success_included": True,
        "teacher_labels_read": False,
        "attack_outcomes_read": False,
        "canonical_parent_keys": [row["canonical_parent_key"] for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", default="")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    payload = build_training_manifest(args.registry_csv, protocol_sha256=args.protocol_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "identity_count": payload["identity_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
