#!/usr/bin/env python3
"""Validate the non-self-referential Factorized V2 handoff binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_handoff_sha(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("handoff_blob_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V2":
        raise ValueError("HANDOFF_SCHEMA")
    if "full_head" in value:
        raise ValueError("SELF_REFERENTIAL_FULL_HEAD_FORBIDDEN")
    for key in ("code_snapshot_commit", "metadata_parent_commit"):
        if not SHA40.fullmatch(str(value.get(key, ""))):
            raise ValueError(f"{key.upper()}_MUST_BE_FULL_SHA")
    handoff_sha = str(value.get("handoff_blob_sha256", ""))
    if not SHA64.fullmatch(handoff_sha) or handoff_sha != canonical_handoff_sha(value):
        raise ValueError("HANDOFF_BLOB_SHA_MISMATCH")
    if value.get("status") != "READY_FOR_DEEPSEEK_STATIC_INTEGRATION":
        raise ValueError("HANDOFF_NOT_STATIC_READY")
    return {
        "status": "PASS",
        "schema": value["schema"],
        "code_snapshot_commit": value["code_snapshot_commit"],
        "metadata_parent_commit": value["metadata_parent_commit"],
        "handoff_blob_sha256": handoff_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", type=Path)
    try:
        print(json.dumps(validate(parser.parse_args().handoff), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
