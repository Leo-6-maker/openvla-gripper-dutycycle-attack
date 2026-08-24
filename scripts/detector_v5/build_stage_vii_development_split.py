"""Freeze the outcome-blind Stage VII development parent split."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SEED = "STAGE_VII_SPLIT_V1_20260816"
COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def identity_suite(key: str) -> str:
    parts = key.split("/")
    if len(parts) != 3 or not parts[0].startswith("libero_"):
        raise ValueError(f"BAD_PARENT_KEY:{key}")
    return parts[0]


def load_keys(root: Path, filename: str, field: str) -> set[str]:
    paths = sorted(root.glob(f"parents/*/{filename}"))
    if not paths:
        raise ValueError(f"NO_PARENT_FILES:{root}:{filename}")
    keys = set()
    for path in paths:
        data = read_json(path)
        key = str(data.get(field, ""))
        identity_suite(key)
        if key in keys:
            raise ValueError(f"DUPLICATE_PARENT:{key}")
        keys.add(key)
    return keys


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    root_seal = {
        "schema": "STAGE_VII_DEVELOPMENT_SPLIT_ROOT_SEAL_V1",
        "status": "PASS_STAGE_VII_DEVELOPMENT_SPLIT",
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "candidate_training_performed": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(root / "ROOT_SEAL.json", root_seal)
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-v-clean-root", required=True, type=Path)
    parser.add_argument("--stage-vi-clean-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    stage_v = load_keys(args.stage_v_clean_root.resolve(), "CLEAN_REPLAY_STUDENT_INPUTS_V1.json", "canonical_parent_key")
    stage_vi = load_keys(args.stage_vi_clean_root.resolve(), "RECONSTRUCTED_FIT670_EPISODE.json", "episode_id")
    overlap = sorted(stage_v & stage_vi)
    if overlap:
        raise ValueError(f"DEVELOPMENT_IDENTITY_OVERLAP:{overlap}")
    identities = sorted(stage_v | stage_vi)
    by_suite: dict[str, list[str]] = defaultdict(list)
    for key in identities:
        by_suite[identity_suite(key)].append(key)

    rows = []
    for suite, keys in sorted(by_suite.items()):
        ordered = sorted(keys, key=lambda key: hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest())
        n = len(ordered)
        if n < 3:
            raise ValueError(f"SUITE_TOO_SMALL_FOR_THREE_SPLITS:{suite}:{n}")
        train_end = max(1, min(n - 2, (3 * n) // 5))
        val_end = max(train_end + 1, min(n - 1, (4 * n) // 5))
        for rank, key in enumerate(ordered):
            split = "TRAIN" if rank < train_end else "VAL" if rank < val_end else "DEVTEST"
            rows.append({
                "canonical_parent_key": key,
                "suite": suite,
                "split": split,
                "suite_rank": rank,
                "order_sha256": hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest(),
                "source_population": "STAGE_V" if key in stage_v else "STAGE_VI_B2",
            })
    rows.sort(key=lambda row: row["canonical_parent_key"])
    suites = sorted(by_suite)
    loso = {
        suite: {
            "test_suites": [suite],
            "train_suites": [other for other in suites if other != suite],
            "test_parent_keys": sorted(row["canonical_parent_key"] for row in rows if row["suite"] == suite),
            "train_parent_keys": sorted(row["canonical_parent_key"] for row in rows if row["suite"] != suite),
        }
        for suite in suites
    }
    summary = {
        "schema": "STAGE_VII_DEVELOPMENT_PARENT_SPLIT_V1",
        "status": "PASS_STAGE_VII_DEVELOPMENT_SPLIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "seed": SEED,
        "parent_grouped": True,
        "suite_stratified": True,
        "selection_used_outcomes": False,
        "selection_used_labels": False,
        "stage_v_parent_count": len(stage_v),
        "stage_vi_b2_parent_count": len(stage_vi),
        "parent_count": len(rows),
        "counts_by_suite": {suite: dict(sorted(Counter(row["split"] for row in rows if row["suite"] == suite).items())) for suite in suites},
        "counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    output.mkdir(parents=True)
    write_json(output / "STAGE_VII_DEVELOPMENT_PARENT_SPLIT_V1.json", {**summary, "rows": rows})
    write_json(output / "STAGE_VII_LOSO_SPLITS_V1.json", {"schema": "STAGE_VII_LOSO_SPLITS_V1", "status": "FROZEN", "splits": loso})
    seal(output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
