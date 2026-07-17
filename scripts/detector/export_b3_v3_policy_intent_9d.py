#!/usr/bin/env python3
"""Export the optional 9D policy-intent ablation outside the primary S1 root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_official_v3_s1 import export_policy_intent_9d, load_formal_fit_registry
from gripper_attack.official_v3_contract import load_external_manifest_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--canonical-parent-key", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--provenance-registry", type=Path)
    args = parser.parse_args()
    rows = load_formal_fit_registry(args.registry_csv, args.registry_summary)
    row = next((item for item in rows if item["canonical_parent_key"] == args.canonical_parent_key), None)
    if row is None:
        raise SystemExit(f"identity is not in formal FIT registry: {args.canonical_parent_key}")
    external_registry = None
    external_registry_sha256 = None
    if args.provenance_registry:
        external_registry, external_registry_sha256 = load_external_manifest_registry(args.provenance_registry)
    manifest = export_policy_intent_9d(
        row,
        args.contract,
        args.output_root,
        external_registry=external_registry,
        external_registry_sha256=external_registry_sha256,
    )
    print(json.dumps({"status": "PASS", "schema": manifest["schema"], "identity": args.canonical_parent_key}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
