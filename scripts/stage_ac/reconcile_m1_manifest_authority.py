#!/usr/bin/env python3
"""Seal the AC2R1 LF/CRLF M1 manifest reconciliation without model access."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_ac.m1_manifest_authority import reconcile, sha256_bytes


def write_exact(path: Path, value: dict) -> dict[str, object]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"AC2R1_RECONCILIATION_APPEND_ONLY_CONFLICT:{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return {"path": path.as_posix(), "bytes": len(data), "sha256": sha256_bytes(data)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json")
    parser.add_argument("--z1-config", type=Path, default=ROOT / "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/STAGE_AC_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_V1.json")
    args = parser.parse_args()
    value = reconcile(args.manifest, args.z1_config)
    print(json.dumps({"status": value["status"], "output": write_exact(args.output, value), "suite_authority": value["suite_authority"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
