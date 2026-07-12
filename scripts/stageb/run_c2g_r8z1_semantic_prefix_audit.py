#!/usr/bin/env python3
"""Run R8Z1 semantic prefix audit on server with minimal L10 impact.

Uses nice -n 19 and single-threaded access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
for c in (REPO, REPO / "src"):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

from tools.multisuite_detector.audit_c2g_r8z1_semantic_prefix_closure import (
    run_audit, audit_exact_prefix, audit_checksum_completeness,
    compute_train_density, analyze_teacher_semantics, verify_provenance,
    sha256_file, read_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--spatial-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full-prefix-audit", action="store_true",
                        help="Run full 1500-episode prefix comparison (slow)")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Episodes per suite for quick prefix audit")
    args = parser.parse_args(argv)

    report = run_audit(
        repo=REPO,
        source_run_root=args.source_run_root,
        spatial_root=args.spatial_root,
        object_root=args.object_root,
        goal_root=args.goal_root,
        composite_root=args.composite_root,
        canary_root=args.canary_root,
        output_root=args.output_root,
    )

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if str(report["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
