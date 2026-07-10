#!/usr/bin/env python3
"""Release clean collector with canonical 25D and full model provenance.

The wrapper verifies the strict suite model map and Goal integrity manifest before
calling the mature clean collector. It also enforces the canonical 25D feature order.
Extra collector arguments are forwarded unchanged after the release-only arguments
are removed from ``sys.argv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.collect_c2g_clean_window_rollouts_strict import install_canonical_order_patch
from scripts.stageb.verify_c2g_suite_model_map_strict import verify


def parse_release_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    return parser.parse_known_args(list(argv))


def verify_internal_suite_paths(model_map_path: Path) -> None:
    from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS

    value = json.loads(model_map_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("suite model map must be a JSON object")
    mismatches = {}
    for suite, internal in SUITE_MODELS.items():
        if suite not in value:
            continue
        frozen = Path(str(value[suite])).resolve()
        actual = Path(str(internal)).resolve()
        if frozen != actual:
            mismatches[suite] = {"frozen": str(frozen), "collector": str(actual)}
    if mismatches:
        raise ValueError(f"collector SUITE_MODELS differs from frozen map: {mismatches}")


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = parse_release_args(raw)
    result = verify(
        args.suite_model_map.resolve(),
        args.suite_model_report.resolve(),
        args.goal_model_manifest.resolve(),
    )
    verify_internal_suite_paths(args.suite_model_map.resolve())
    args.model_verification_report.parent.mkdir(parents=True, exist_ok=True)
    args.model_verification_report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    install_canonical_order_patch()
    from scripts.stageb.collect_c2g_clean_window_rollouts import main as collector_main

    original_argv = sys.argv
    try:
        sys.argv = [str(original_argv[0]), *forwarded]
        return int(collector_main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
