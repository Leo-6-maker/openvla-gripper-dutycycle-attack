#!/usr/bin/env python3
"""Build CLEAN2000 canonical episode index from explicit source roots.

This script scans allowlisted source directories, adapts each episode through
the correct source-format adapter, and produces a unified CLEAN2000 index.

Usage:
  python build_clean2000_index.py \
    --object500_root /path/to/sc5_object_privileged_loto_v1 \
    --clean1500_root /path/to/sc5_cross_suite_clean1500_v1 \
    --output_dir /path/to/output

Output:
  CLEAN2000_INDEX_DRAFT.jsonl    — canonical episode rows
  CLEAN2000_SOURCE_INVENTORY.json — per-source counts and SHAs
  CLEAN2000_ATTEMPT_LEDGER.jsonl  — every episode attempt, including skipped
"""

import argparse
import hashlib
import json
import os
import sys
import time

# Add parent to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canonical_schema import (
    CANONICAL_FIELDS, EXPECTED_PER_SUITE, EXPECTED_TOTAL,
    VALID_SUITES, REQUIRED_CONDITION,
    TASK_ID_RANGE, STATE_ID_RANGE,
)
from clean1500_adapter import build_canonical as adapt_clean1500
from clean1500_adapter import list_episode_dirs as list_clean1500_dirs
from object500_adapter import build_canonical as adapt_object500
from object500_adapter import list_episode_dirs as list_object500_dirs


def parse_args():
    p = argparse.ArgumentParser(description="Build CLEAN2000 canonical episode index")
    p.add_argument("--object500_root", required=True,
                   help="Root of sc5_object_privileged_loto_v1 (contains wave1_50/, wave2_*/)")
    p.add_argument("--clean1500_root", required=True,
                   help="Root of sc5_cross_suite_clean1500_v1")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output files")
    p.add_argument("--dry_run", action="store_true",
                   help="Scan and report counts without writing index")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_roots = {
        "object500": os.path.abspath(args.object500_root),
        "clean1500": os.path.abspath(args.clean1500_root),
    }

    # Validate source roots
    for name, root in source_roots.items():
        if not os.path.isdir(root):
            print("ERROR: {} source root not found: {}".format(name, root))
            sys.exit(1)

    print("=== CLEAN2000 Index Builder ===")
    print("Object500 root: {}".format(source_roots["object500"]))
    print("CLEAN1500 root: {}".format(source_roots["clean1500"]))
    print()

    # ── Phase 1: Discover episodes ──
    print("Phase 1: Discovering episodes...")

    object_dirs = list_object500_dirs(source_roots["object500"])
    clean1500_dirs = list_clean1500_dirs(source_roots["clean1500"])

    print("  Object500 CLEAN episodes found: {}".format(len(object_dirs)))
    print("  CLEAN1500 episodes found:       {}".format(len(clean1500_dirs)))

    # ── Phase 2: Adapt to canonical rows ──
    print("Phase 2: Adapting to canonical rows...")

    index_rows = []
    attempt_ledger = []
    skipped = []
    errors = []

    # Adapt Object500
    object_root_sha = _compute_root_inventory(source_roots["object500"])
    for ep_dir in object_dirs:
        try:
            row = adapt_object500(ep_dir, manifest_sha256=object_root_sha)
            if row is None:
                skipped.append({"ep_dir": ep_dir, "reason": "not_CLEAN"})
            else:
                index_rows.append(row)
                attempt_ledger.append({
                    "episode_key": row["episode_key"],
                    "source_root": ep_dir,
                    "source_format": "object500_v1",
                    "status": "INDEXED",
                })
        except Exception as e:
            errors.append({"ep_dir": ep_dir, "error": str(e)})
            attempt_ledger.append({
                "source_root": ep_dir,
                "source_format": "object500_v1",
                "status": "ERROR",
                "error": str(e),
            })

    print("  Object500 indexed: {} rows".format(
        sum(1 for r in index_rows if r["source_format"] == "object500_v1")))

    # Adapt CLEAN1500
    clean1500_root_sha = _compute_root_inventory(source_roots["clean1500"])
    for ep_dir in clean1500_dirs:
        try:
            row = adapt_clean1500(ep_dir, manifest_sha256=clean1500_root_sha)
            if row is None:
                skipped.append({"ep_dir": ep_dir, "reason": "not_CLEAN"})
            else:
                index_rows.append(row)
                attempt_ledger.append({
                    "episode_key": row["episode_key"],
                    "source_root": ep_dir,
                    "source_format": "clean1500_v1",
                    "status": "INDEXED",
                })
        except Exception as e:
            errors.append({"ep_dir": ep_dir, "error": str(e)})
            attempt_ledger.append({
                "source_root": ep_dir,
                "source_format": "clean1500_v1",
                "status": "ERROR",
                "error": str(e),
            })

    print("  CLEAN1500 indexed: {} rows".format(
        sum(1 for r in index_rows if r["source_format"] == "clean1500_v1")))
    print()

    # ── Phase 3: Source inventory ──
    print("Phase 3: Building source inventory...")
    inventory = _build_source_inventory(index_rows, source_roots)
    for suite, count in sorted(inventory["per_suite"].items()):
        expected = EXPECTED_PER_SUITE.get(suite, "?")
        print("  {}: {} (expected {})".format(suite, count, expected))
    print("  Total: {} (expected {})".format(inventory["total"], EXPECTED_TOTAL))

    # Check suite coverage
    for suite, expected in EXPECTED_PER_SUITE.items():
        actual = inventory["per_suite"].get(suite, 0)
        if actual < expected:
            print("  WARNING: {} has {} episodes, expected {}".format(suite, actual, expected))

    # ── Quick integrity checks ──
    print()
    print("Phase 4: Quick integrity checks...")
    issues = _quick_checks(index_rows)
    for issue in issues:
        print("  {}".format(issue))
    if not issues:
        print("  PASS")

    # ── Phase 5: Write outputs ──
    if not args.dry_run:
        print()
        print("Phase 5: Writing outputs...")

        # Episode index
        index_path = os.path.join(args.output_dir, "CLEAN2000_INDEX_DRAFT.jsonl")
        with open(index_path, "w") as f:
            for row in index_rows:
                f.write(json.dumps(row) + "\n")
        print("  {}".format(index_path))

        # Source inventory
        inv_path = os.path.join(args.output_dir, "CLEAN2000_SOURCE_INVENTORY.json")
        with open(inv_path, "w") as f:
            json.dump(inventory, f, indent=2)
        print("  {}".format(inv_path))

        # Attempt ledger
        ledger_path = os.path.join(args.output_dir, "CLEAN2000_ATTEMPT_LEDGER.jsonl")
        with open(ledger_path, "w") as f:
            for entry in attempt_ledger:
                f.write(json.dumps(entry) + "\n")
        print("  {}".format(ledger_path))

        # Errors & skipped
        if errors or skipped:
            diag = {"errors": errors, "skipped": skipped}
            diag_path = os.path.join(args.output_dir, "CLEAN2000_DIAGNOSTICS.json")
            with open(diag_path, "w") as f:
                json.dump(diag, f, indent=2)
            print("  {} ({} errors, {} skipped)".format(diag_path, len(errors), len(skipped)))

        # Manifest of the build
        build_manifest = {
            "gate": "CLEAN2000_INDEX_DRAFT_V1",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "object500_root": source_roots["object500"],
            "clean1500_root": source_roots["clean1500"],
            "total_indexed": len(index_rows),
            "total_errors": len(errors),
            "total_skipped": len(skipped),
            "per_suite": inventory["per_suite"],
        }
        bm_path = os.path.join(args.output_dir, "CLEAN2000_BUILD_MANIFEST.json")
        with open(bm_path, "w") as f:
            json.dump(build_manifest, f, indent=2)
        print("  {}".format(bm_path))

    print()
    print("DONE.")
    print("  Indexed: {} episodes".format(len(index_rows)))
    print("  Errors:  {}".format(len(errors)))
    print("  Skipped: {}".format(len(skipped)))


def _build_source_inventory(index_rows, source_roots):
    per_suite = {}
    per_format = {}
    for row in index_rows:
        s = row["suite"]
        per_suite[s] = per_suite.get(s, 0) + 1
        fmt = row["source_format"]
        per_format[fmt] = per_format.get(fmt, 0) + 1

    return {
        "gate": "CLEAN2000_SOURCE_INVENTORY_V1",
        "total": len(index_rows),
        "expected_total": EXPECTED_TOTAL,
        "per_suite": per_suite,
        "per_format": per_format,
        "source_roots": {
            "object500": source_roots["object500"],
            "clean1500": source_roots["clean1500"],
        },
    }


def _quick_checks(index_rows):
    issues = []

    # Check condition is always CLEAN
    non_clean = [r for r in index_rows if r["condition"] != "CLEAN"]
    if non_clean:
        issues.append("FATAL: {} rows have condition != CLEAN".format(len(non_clean)))

    # Check episode_key uniqueness
    keys = [r["episode_key"] for r in index_rows]
    dup_keys = len(keys) - len(set(keys))
    if dup_keys > 0:
        issues.append("FATAL: {} duplicate episode_keys".format(dup_keys))

    # Check parent_key uniqueness (within current clean collection)
    parents = [r["parent_key"] for r in index_rows]
    dup_parents = len(parents) - len(set(parents))
    if dup_parents > 0:
        issues.append("WARNING: {} duplicate parent_keys".format(dup_parents))

    # Check suite validity
    bad_suites = [r for r in index_rows if r["suite"] not in VALID_SUITES]
    if bad_suites:
        issues.append("FATAL: {} rows with invalid suite".format(len(bad_suites)))

    # Check task_id range
    bad_tasks = [r for r in index_rows if not (0 <= r["task_id"] <= 9)]
    if bad_tasks:
        issues.append("FATAL: {} rows with task_id out of range".format(len(bad_tasks)))

    # Check state_id range
    bad_states = [r for r in index_rows if not (0 <= r["state_id"] <= 49)]
    if bad_states:
        issues.append("FATAL: {} rows with state_id out of range".format(len(bad_states)))

    # Check n_steps > 0
    zero_steps = [r for r in index_rows if r["n_steps"] <= 0]
    if zero_steps:
        issues.append("FATAL: {} rows with n_steps <= 0".format(len(zero_steps)))

    return issues


def _compute_root_inventory(root):
    """Compute a quick inventory SHA of the source root."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, fn), root))
    files.sort()
    return hashlib.sha256("\n".join(files).encode()).hexdigest()


if __name__ == "__main__":
    main()
