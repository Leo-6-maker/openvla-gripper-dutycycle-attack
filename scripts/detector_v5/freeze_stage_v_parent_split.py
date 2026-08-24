"""Freeze the parent-grouped Stage V TRAIN/VAL/TEST split before interventions."""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SPLIT_COUNTS = (("TRAIN", 6), ("VAL", 2), ("TEST", 2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}::{key}".encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze(parent_manifest: Path, output_dir: Path, *, split_salt: str) -> dict[str, Any]:
    parent_manifest = parent_manifest.resolve()
    output_dir = output_dir.resolve()
    if not split_salt:
        raise ValueError("split salt is required")
    if not parent_manifest.is_file():
        raise ValueError(f"parent manifest missing: {parent_manifest}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"split output must be new/empty: {output_dir}")

    value = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("parent manifest must be an object")
    if value.get("status") not in {"FROZEN", "PASS"}:
        raise ValueError("parent manifest is not closed")
    rows = value.get("selected_parents")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("parent manifest must contain exactly 40 selected parents")

    by_suite: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("parent row is not an object")
        key = str(raw.get("canonical_parent_key", ""))
        suite = str(raw.get("suite", ""))
        if not key or suite not in by_suite or key in seen:
            raise ValueError("parent identity is invalid or duplicated")
        if raw.get("old_artifacts_reused") is not False or raw.get("source_artifact_read") is not False:
            raise ValueError(f"parent is not clean-only: {key}")
        seen.add(key)
        by_suite[suite].append(dict(raw))
    if any(len(by_suite[suite]) != 10 for suite in SUITES):
        raise ValueError("each suite must contain exactly 10 parents")

    assignments: list[dict[str, Any]] = []
    for suite in SUITES:
        ranked = sorted(by_suite[suite], key=lambda row: (_hash(split_salt, str(row["canonical_parent_key"])), str(row["canonical_parent_key"])))
        cursor = 0
        for split, count in SPLIT_COUNTS:
            for row in ranked[cursor:cursor + count]:
                assignments.append({
                    **row,
                    "split": split,
                    "split_rank_sha256": _hash(split_salt, str(row["canonical_parent_key"])),
                })
            cursor += count

    split_rows = {split: [row for row in assignments if row["split"] == split] for split, _ in SPLIT_COUNTS}
    output = {
        "schema": "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1",
        "status": "FROZEN",
        "source_parent_manifest": str(parent_manifest),
        "source_parent_manifest_sha256": sha256_file(parent_manifest),
        "source_commit": value.get("source_commit"),
        "source_tree": value.get("source_tree"),
        "split_salt": split_salt,
        "assignment_rule": "per_suite sha256(split_salt::canonical_parent_key), first 6 TRAIN, next 2 VAL, final 2 TEST",
        "parent_grouped": True,
        "vulnerability_outcomes_read": False,
        "intervention_branches_read": False,
        "split_counts": {split: len(split_rows[split]) for split, _ in SPLIT_COUNTS},
        "split_counts_by_suite": {
            suite: {split: sum(row["suite"] == suite for row in split_rows[split]) for split, _ in SPLIT_COUNTS}
            for suite in SUITES
        },
        "parents": assignments,
        "parents_by_split": split_rows,
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "vis_pgd_attack_rollouts": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1.json"
    _atomic_json(path, output)
    digest = sha256_file(path)
    (output_dir / "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-salt", required=True)
    args = parser.parse_args(argv)
    freeze(args.parent_manifest, args.output_dir, split_salt=args.split_salt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
