#!/usr/bin/env python3
"""Run the bounded R8Z canary and, only after PASS, the full OGS-1500 derivation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.multisuite_detector.audit_c2g_r8z_ogs_canary import (
    PASS_STATUS as CANARY_PASS_STATUS,
    run_audit as audit_canary,
)
from tools.multisuite_detector.audit_c2g_r8z_ogs_full1500 import (
    PASS_STATUS as FULL_PASS_STATUS,
    run_audit as audit_full,
)
from tools.multisuite_detector.build_c2g_r8z_ogs_official_views import (
    CANARY_BUILD_PASS,
    SUITE_BUILD_PASS,
    add_source_arguments,
    build_canary,
    build_suite,
    source_context_from_args,
)
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    TARGET_SUITES,
    verify_git_head,
)


def is_within(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def validate_output_roots(args: argparse.Namespace) -> None:
    roots = (
        args.canary_root,
        args.spatial220_root,
        args.object280_root,
        args.goal300_root,
        args.composite_root,
    )
    resolved = [path.resolve() for path in roots]
    if len(set(resolved)) != len(resolved):
        raise ValueError("R8Z output roots must be distinct")
    for path in resolved:
        if path.exists():
            raise FileExistsError(path)
        if is_within(path, REPO) or is_within(REPO, path):
            raise ValueError("R8Z outputs must remain outside the repository")
        if is_within(path, args.source_r8w_run_root) or is_within(
            args.source_r8w_run_root, path
        ):
            raise ValueError("R8Z outputs and immutable R8W source root must be disjoint")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--spatial220-root", type=Path, required=True)
    parser.add_argument("--object280-root", type=Path, required=True)
    parser.add_argument("--goal300-root", type=Path, required=True)
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--expected-r8z-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verify_git_head(args.expected_r8z_head, REPO)
    validate_output_roots(args)
    context = source_context_from_args(args)
    canary_build = build_canary(
        context,
        output_root=args.canary_root,
        r8z_head=args.expected_r8z_head,
    )
    if canary_build["status"] != CANARY_BUILD_PASS:
        print(json.dumps({"phase": "CANARY_BUILD", **canary_build}, indent=2, sort_keys=True))
        return 1
    canary_audit = audit_canary(
        context,
        canary_root=args.canary_root,
        r8z_head=args.expected_r8z_head,
    )
    if canary_audit["status"] != CANARY_PASS_STATUS:
        print(json.dumps({"phase": "CANARY_AUDIT", **canary_audit}, indent=2, sort_keys=True))
        return 1

    suite_roots = {
        "libero_spatial": args.spatial220_root,
        "libero_object": args.object280_root,
        "libero_goal": args.goal300_root,
    }
    suite_reports = {}
    for suite in TARGET_SUITES:
        report = build_suite(
            context,
            suite=suite,
            output_root=suite_roots[suite],
            r8z_head=args.expected_r8z_head,
        )
        suite_reports[suite] = report
        if report["status"] != SUITE_BUILD_PASS:
            print(
                json.dumps(
                    {
                        "phase": "FULL_SUITE_BUILD",
                        "failed_suite": suite,
                        "canary": canary_audit,
                        "suite_reports": suite_reports,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
    full_audit = audit_full(
        context,
        suite_roots=suite_roots,
        composite_root=args.composite_root,
        r8z_head=args.expected_r8z_head,
    )
    result = {
        "phase": "COMPLETE",
        "canary": canary_audit,
        "suite_reports": suite_reports,
        "composite": full_audit,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if full_audit["status"] == FULL_PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())

