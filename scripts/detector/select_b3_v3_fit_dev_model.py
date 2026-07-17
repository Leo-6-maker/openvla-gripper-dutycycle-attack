#!/usr/bin/env python3
"""Preparation-only FIT-DEV model-selection contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("no FIT-DEV candidates")
    if any(item.get("split") not in (None, "FIT_DEV") for item in candidates):
        raise ValueError("FIT-DEV selector received CAL/CHECK/attack data")
    ordered = sorted(candidates, key=lambda item: (
        -float(item.get("full_t10_event_hit_rate", 0.0)),
        float(item.get("negative_episode_any_emit_rate", 1.0)),
        float(item.get("release_overlap_rate", 1.0)),
        str(item.get("candidate_id", "")),
    ))
    chosen = dict(ordered[0])
    chosen.update({"schema": "B3_OFFICIAL_V3_FIT_DEV_SELECTION_V1", "status": "FIT_DEV_SELECTED", "formal_attack_authorized": False})
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    value = json.loads(args.metrics.read_text(encoding="utf-8"))
    candidates = value if isinstance(value, list) else value.get("candidates", [])
    args.output.write_text(json.dumps(select_candidate(candidates), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
