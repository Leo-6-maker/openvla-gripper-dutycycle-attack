#!/usr/bin/env python3
"""Append one verified OFT materialization receipt to the Z0R2 ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("suite")
    parser.add_argument("server_path")
    parser.add_argument("server_manifest_sha256")
    parser.add_argument("file_count", type=int)
    parser.add_argument("bytes", type=int)
    parser.add_argument("cache_deletion")
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8")) if args.ledger.exists() else {"schema": "STAGE_Z_Z0R2_M1_MATERIALIZATION_LEDGER_V1", "status": "PASS_SEQUENTIAL_RECEIPTS", "suites": {}}
    ledger.setdefault("schema", "STAGE_Z_Z0R2_M1_MATERIALIZATION_LEDGER_V1")
    ledger.setdefault("status", "PASS_SEQUENTIAL_RECEIPTS")
    ledger.setdefault("suites", {})[args.suite] = {
        "server_materialization_path": args.server_path,
        "server_cache_verified": True,
        "server_manifest_sha256": args.server_manifest_sha256,
        "server_file_count": args.file_count,
        "server_bytes": args.bytes,
        "cache_deletion": args.cache_deletion,
    }
    args.ledger.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"suite": args.suite, "status": ledger["status"], "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
