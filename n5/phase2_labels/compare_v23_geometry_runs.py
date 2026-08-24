"""Independent byte/canonical comparison for the two real-geometry replays."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any


class ComparisonHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_root(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise ComparisonHold(f"unsealed root: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if side != [sha256_file(sums), "SHA256SUMS"]:
        raise ComparisonHold(f"sidecar mismatch: {root}")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        if name in {"SHA256SUMS", "SHA256SUMS.sha256"} or (root / name).is_symlink():
            raise ComparisonHold(f"unsafe checksum entry: {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ComparisonHold(f"file mismatch: {path}")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not path.is_symlink()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}}
    if listed != actual:
        raise ComparisonHold(f"file closure mismatch: {root}")
    return sha256_file(sums)


def episode_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for path in sorted((root / "episodes").glob("*/geometry_cases.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ComparisonHold(f"empty geometry stream: {path}")
        episode_ids = {row.get("episode_id") for row in rows}
        if len(episode_ids) != 1:
            raise ComparisonHold(f"mixed episode stream: {path}")
        episode_id = next(iter(episode_ids))
        if not isinstance(episode_id, str):
            raise ComparisonHold(f"missing episode identity: {path}")
        steps = [row.get("step") for row in rows]
        if steps != list(range(len(rows))):
            raise ComparisonHold(f"non-contiguous steps: {path}")
        result[episode_id] = rows
    if len(result) != 40:
        raise ComparisonHold(f"expected 40 episodes, got {len(result)}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_a = Path(args.run_a).resolve(); run_b = Path(args.run_b).resolve()
    seal_a = verify_root(run_a); seal_b = verify_root(run_b)
    rows_a = episode_rows(run_a); rows_b = episode_rows(run_b)
    if set(rows_a) != set(rows_b):
        raise ComparisonHold("episode identity sets differ")
    differing = []
    relation_counts = set()
    for episode_id in sorted(rows_a):
        if len(rows_a[episode_id]) != len(rows_b[episode_id]):
            raise ComparisonHold(f"step counts differ: {episode_id}")
        for left, right in zip(rows_a[episode_id], rows_b[episode_id]):
            if left.get("step") != right.get("step"):
                raise ComparisonHold(f"step identity differs: {episode_id}")
            relation_counts.add(len(left.get("relations", [])))
            if canonical(left) != canonical(right):
                differing.append({"episode_id": episode_id, "step": left.get("step")})
    if differing:
        raise ComparisonHold(f"canonical geometry differs at {len(differing)} rows")
    payload = {
        "schema": "C3_T1_REAL_GEOMETRY_COMPARISON_V1",
        "status": "PASS",
        "run_A": {"root": str(run_a), "sha256sums_sha256": seal_a},
        "run_B": {"root": str(run_b), "sha256sums_sha256": seal_b},
        "episode_count": len(rows_a),
        "step_count": sum(len(rows) for rows in rows_a.values()),
        "relation_count_values": sorted(relation_counts),
        "canonical_differences": 0,
        "protected_payload_read": False,
        "model_inference": False,
        "teacher_labeling": False,
        "attack": False,
    }
    parent = Path(args.output_parent).resolve(); parent.mkdir(parents=True, exist_ok=True)
    final = parent / args.output_name
    if final.exists() or final.is_symlink():
        raise ComparisonHold(f"output exists: {final}")
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex}"; staging.mkdir()
    try:
        (staging / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = ["comparison.json"]
        (staging / "SHA256SUMS").write_text(f"{sha256_file(staging / files[0])}  {files[0]}\n", encoding="utf-8")
        sums_sha = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "output_root": str(final), "sha256sums_sha256": sums_sha, **payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--output-name", required=True)
    try:
        print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
