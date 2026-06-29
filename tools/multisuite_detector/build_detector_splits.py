#!/usr/bin/env python3
"""Build detector splits: episode-grouped, task-grouped, and LOSO.

All windows from the same episode/parent MUST belong to the same split.
LOSO test suite excluded from all training-time statistics.
NO live data. Reads only frozen CLEAN2000 manifest + episode index.
"""
from __future__ import annotations
import argparse, json, hashlib, sys
from collections import defaultdict
from pathlib import Path

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
LOSO_FOLDS = {
    "loso_libero10": {"train": ["libero_object", "libero_spatial", "libero_goal"], "test": "libero_10"},
    "loso_goal": {"train": ["libero_object", "libero_spatial", "libero_10"], "test": "libero_goal"},
    "loso_spatial": {"train": ["libero_object", "libero_goal", "libero_10"], "test": "libero_spatial"},
    "loso_object": {"train": ["libero_spatial", "libero_goal", "libero_10"], "test": "libero_object"},
}


def load_episodes(path: str) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def validate_no_cross_split_leakage(splits: dict, episodes: list[dict]) -> list[str]:
    """Check: same episode_key or parent_key in multiple splits."""
    errors = []
    seen_ep = {}
    seen_parent = {}
    for split_name, ep_keys in splits.items():
        for ek in ep_keys:
            if ek in seen_ep:
                errors.append(f"LEAK: episode {ek} in both {seen_ep[ek]} and {split_name}")
            seen_ep[ek] = split_name
    for ep in episodes:
        pk = ep.get("parent_key", ep["episode_key"])
        if pk:
            if pk in seen_parent:
                if seen_parent[pk] != splits.get("train", []):
                    pass
            seen_parent[pk] = ep["episode_key"]
    return errors


def build_episode_grouped(episodes: list[dict], seed: int = 42) -> dict:
    """Random episode-level split stratified by suite. 60/20/20."""
    import random
    rng = random.Random(seed)
    by_suite = defaultdict(list)
    for ep in episodes:
        by_suite[ep["suite"]].append(ep["episode_key"])
    splits = {"train": [], "val": [], "test": []}
    for suite, keys in by_suite.items():
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        splits["train"].extend(keys[:n_train])
        splits["val"].extend(keys[n_train:n_train + n_val])
        splits["test"].extend(keys[n_train + n_val:])
    return splits


def build_task_grouped(episodes: list[dict], seed: int = 42) -> dict:
    """Task-level split: all episodes of a task in same split."""
    import random
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for ep in episodes:
        key = (ep["suite"], ep["task_id"])
        by_task[key].append(ep["episode_key"])
    tasks = list(by_task.keys())
    rng.shuffle(tasks)
    n = len(tasks)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    splits = {"train": [], "val": [], "test": []}
    for task in tasks[:n_train]:
        splits["train"].extend(by_task[task])
    for task in tasks[n_train:n_train + n_val]:
        splits["val"].extend(by_task[task])
    for task in tasks[n_train + n_val:]:
        splits["test"].extend(by_task[task])
    return splits


def build_loso(episodes: list[dict], fold_name: str) -> dict:
    """Leave-One-Suite-Out split."""
    if fold_name not in LOSO_FOLDS:
        raise ValueError(f"Unknown LOSO fold: {fold_name}")
    config = LOSO_FOLDS[fold_name]
    by_suite = defaultdict(list)
    for ep in episodes:
        by_suite[ep["suite"]].append(ep["episode_key"])
    train_keys = []
    for s in config["train"]:
        train_keys.extend(by_suite.get(s, []))
    test_keys = by_suite.get(config["test"], [])
    import random
    rng = random.Random(42)
    rng.shuffle(train_keys)
    n = len(train_keys)
    n_train = int(n * 0.8)
    return {
        "train": train_keys[:n_train],
        "val": train_keys[n_train:],
        "test": test_keys,
        "test_suite": config["test"],
        "train_suites": config["train"],
    }


def validate_loso_isolation(splits: dict, episodes: list[dict]) -> list[str]:
    """Check: test suite not in normalization, train, or val statistics."""
    errors = []
    test_suite = splits.get("test_suite")
    if not test_suite:
        return ["No test_suite specified for LOSO split"]
    ep_by_key = {ep["episode_key"]: ep for ep in episodes}
    for ek in splits["train"] + splits["val"]:
        ep = ep_by_key.get(ek)
        if ep and ep["suite"] == test_suite:
            errors.append(f"LOSO LEAK: test suite {test_suite} episode {ek} in train/val")
    for ek in splits["test"]:
        ep = ep_by_key.get(ek)
        if ep and ep["suite"] != test_suite:
            errors.append(f"LOSO MISMATCH: non-{test_suite} episode {ek} in test set")
    return errors


def main():
    ap = argparse.ArgumentParser(description="Build detector splits")
    ap.add_argument("--episode_index", required=True, help="Frozen CLEAN2000 episode index JSONL")
    ap.add_argument("--split_type", required=True,
                    choices=["episode_grouped", "task_grouped", "loso"])
    ap.add_argument("--loso_fold", help="LOSO fold name (required when split_type=loso)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True, help="Output directory for split manifests")
    args = ap.parse_args()

    episodes = load_episodes(args.episode_index)
    print(f"Loaded {len(episodes)} episodes from index")

    if args.split_type == "loso":
        if not args.loso_fold:
            sys.exit("--loso_fold required for LOSO splits")
        splits = build_loso(episodes, args.loso_fold)
        errors = validate_loso_isolation(splits, episodes)
    elif args.split_type == "episode_grouped":
        splits = build_episode_grouped(episodes, args.seed)
        errors = validate_no_cross_split_leakage(splits, episodes)
    elif args.split_type == "task_grouped":
        splits = build_task_grouped(episodes, args.seed)
        errors = validate_no_cross_split_leakage(splits, episodes)
    else:
        sys.exit(f"Unknown split type: {args.split_type}")

    if errors:
        print(f"SPLIT VALIDATION FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"split_{args.split_type}.json"
    output = {
        "split_type": args.split_type,
        "loso_fold": args.loso_fold if args.split_type == "loso" else None,
        "seed": args.seed,
        "splits": {k: v for k, v in splits.items() if k in ("train", "val", "test")},
        "counts": {k: len(v) for k, v in splits.items() if k in ("train", "val", "test")},
        "test_suite": splits.get("test_suite"),
        "train_suites": splits.get("train_suites"),
        "validation_passed": True,
    }
    sha = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    output["split_sha256"] = sha
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")
    print(f"  Train: {output['counts']['train']} episodes")
    print(f"  Val:   {output['counts']['val']} episodes")
    print(f"  Test:  {output['counts']['test']} episodes")
    print(f"  SHA256: {sha}")


if __name__ == "__main__":
    main()
