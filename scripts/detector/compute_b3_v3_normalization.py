#!/usr/bin/env python3
"""Compute FIT-only normalization for a sealed S1 root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_formal import B3Normalization
from gripper_attack.b3_v3_dataset import compute_fit_normalization, load_episode, load_formal_registry_csv


def compute_normalization_from_s1(
    registry_csv: Path,
    s1_root: Path,
    *,
    include_9d: bool = False,
    policy_intent_root: Path | None = None,
) -> B3Normalization:
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    episodes = []
    for row in rows:
        episode_root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d_root = None
        if include_9d:
            if policy_intent_root is None:
                raise ValueError("--policy-intent-root is required for 9D normalization")
            nine_d_root = policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(episode_root, row, include_9d_root=nine_d_root))
    return compute_fit_normalization(episodes, include_9d=include_9d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-9d", action="store_true")
    parser.add_argument("--policy-intent-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    norm = compute_normalization_from_s1(
        args.registry_csv, args.s1_root, include_9d=args.include_9d,
        policy_intent_root=args.policy_intent_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "B3_OFFICIAL_V3_NORMALIZATION_V1", "normalization": norm.to_dict(), "normalization_sha256": norm.sha256}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "normalization_sha256": norm.sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
