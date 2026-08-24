#!/usr/bin/env python3
"""Download one immutable HF file with byte-range reconnects."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("expected_sha256")
    parser.add_argument("--range-size", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() and args.output.stat().st_size == args.expected_size and sha256_file(args.output) == args.expected_sha256:
        print(json.dumps({"status": "ALREADY_EXACT", "path": str(args.output), "sha256": args.expected_sha256}, sort_keys=True))
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > args.expected_size:
        partial.unlink()
        offset = 0
    attempts = 0
    while offset < args.expected_size:
        end = min(args.expected_size - 1, offset + args.range_size - 1)
        request = urllib.request.Request(args.url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("ab") as stream:
                if offset and response.status != 206:
                    raise RuntimeError(f"range response status {response.status} at offset {offset}")
                remaining = end - offset + 1
                while remaining:
                    data = response.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ConnectionError(f"short range at offset {offset}; {remaining} bytes remain")
                    stream.write(data)
                    offset += len(data)
                    remaining -= len(data)
            attempts = 0
            print(json.dumps({"status": "PROGRESS", "offset": offset, "size": args.expected_size}, sort_keys=True), flush=True)
        except Exception as exc:  # network failures are expected; the offset is durable on disk
            current = partial.stat().st_size if partial.exists() else 0
            if current > offset:
                offset = current
            attempts += 1
            if attempts > 100:
                raise RuntimeError(f"reconnect budget exhausted at {offset}/{args.expected_size}: {exc}") from exc
            time.sleep(min(30, attempts))
    if partial.stat().st_size != args.expected_size:
        raise RuntimeError("partial size differs after download")
    actual = sha256_file(partial)
    if actual != args.expected_sha256:
        raise RuntimeError(f"sha256 mismatch: {actual} != {args.expected_sha256}")
    partial.replace(args.output)
    print(json.dumps({"status": "PASS_EXACT", "path": str(args.output), "size": args.expected_size, "sha256": actual}, sort_keys=True))


if __name__ == "__main__":
    main()
