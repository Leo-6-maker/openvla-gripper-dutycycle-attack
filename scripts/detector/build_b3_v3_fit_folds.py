#!/usr/bin/env python3
"""Build the sealed four-fold FIT viability manifest.

This is a registry-only preparation step.  It does not open S1 episodes and
does not run a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import build_fit_fold_manifest, sha256_file, write_fit_fold_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with args.registry_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fit_rows = [row for row in rows if row.get("split") == "FIT_TRAIN"]
    manifest = build_fit_fold_manifest(fit_rows, registry_sha256=sha256_file(args.registry_csv))
    write_fit_fold_bundle(args.output_root, manifest)
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "fold_count": 4, "train_per_fold": 600, "validation_per_fold": 200}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
