#!/usr/bin/env python3
"""Dispatch one M3.5 parent to its suite-bound model under the atomic worker."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MODEL_RELATIVE = {
    "libero_10": Path("libero-10/openvla-7b-finetuned-libero-10"),
    "libero_goal": Path("libero-goal"),
    "libero_object": Path("openvla-7b-finetuned-libero-object"),
    "libero_spatial": Path("libero-spatial/spatial_c8f03f4_20260620"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)

    suite = str(args.parent_key).split("/", 1)[0]
    relative = MODEL_RELATIVE.get(suite)
    if relative is None:
        raise SystemExit(f"UNKNOWN_SUITE:{suite}")
    model_path = (args.model_root / relative).resolve()
    if not model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{model_path}")
    command = [
        sys.executable, str(args.runner),
        "--protocol", str(args.protocol),
        "--parent-key", args.parent_key,
        "--output-dir", str(args.output_dir),
        "--official-snapshot-root", str(args.official_snapshot_root),
        "--upstream-root", str(args.upstream_root),
        "--model-path", str(model_path),
        "--gpu", str(args.gpu),
        "--source-commit", args.source_commit,
        "--source-tree", args.source_tree,
        "--authorization-receipt", str(args.authorization_receipt),
        "--enable-runtime",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
