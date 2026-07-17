#!/usr/bin/env python3
"""Read-only audit for the preparation-only attack manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_v3_attack_protocol import audit_attack_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit_attack_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
