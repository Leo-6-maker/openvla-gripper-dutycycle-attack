#!/usr/bin/env python3
"""Build detector splits with strict parent/episode/suite isolation.

ALL episodes sharing a parent_key belong to the same split.
Split at parent level to prevent leakage.
LOSO test suite excluded from all training statistics.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
LOSO_FOLDS = {
    "loso_libero10": {"train": ["libero_object","libero_spatial","libero_goal"], "test": "libero_10"},
    "loso_goal": {"train": ["libero_object","libero_spatial","libero_10"], "test": "libero_goal"},
    "loso_spatial": {"train": ["libero_object","libero_goal","libero_10"], "test": "libero_spatial"},
    "loso_object": {"train": ["libero_spatial","libero_goal","libero_10"], "test": "libero_object"},
}


def load_episodes(path: str) -> list[dict]:
    eps = []
    with open(path) as f:
        for line in f:
            eps.append(json.loads(line))
    for ep in eps:
        if "episode_key" not in ep:
            raise ValueError(f"Episode missing episode_key: {ep}")
        if "suite" not in ep:
            raise ValueError(f"Episode missing suite: {ep}")
    return eps


def validate_no_parent_leakage(splits: dict, episodes: list[dict]) -> list[str]:
    """Enforce: same parent_key must be in exactly one split."""
    errors = []
    ep_to_split = {}
    for sn in ["train", "val", "test"]:
        for ek in splits.get(sn, []):
            if ek in ep_to_split:
                errors.append(f"DUPLICATE_EPISODE: {ek} in both {ep_to_split[ek]} and {sn}")
            ep_to_split[ek] = sn

    parent_to_split = {}
    for ep in episodes:
        ek = ep["episode_key"]
        actual = ep_to_split.get(ek)
        if actual is None:
            errors.append(f"MISSING_FROM_SPLITS: {ek}")
            continue
        pk = ep.get("parent_key") or ek
        if pk in parent_to_split:
            expected = parent_to_split[pk]
            if actual != expected:
                errors.append(f"PARENT_LEAK: parent {pk} in {expected}, child {ek} in {actual}")
        else:
            parent_to_split[pk] = actual

    all_manifest = {ep["episode_key"] for ep in episodes}
    all_split = set(ep_to_split.keys())
    missing = all_manifest - all_split
    unknown = all_split - all_manifest
    if missing:
        errors.append(f"MISSING_EPISODES: {len(missing)} in index but not in any split: {sorted(list(missing))[:5]}...")
    if unknown:
        errors.append(f"UNKNOWN_EPISODES: {len(unknown)} in splits but not in index: {sorted(list(unknown))[:5]}...")

    return errors


def build_episode_grouped(episodes: list[dict], seed: int = 42) -> dict:
    import random
    rng = random.Random(seed)
    parent_groups = defaultdict(set)
    ek_to_parent = {}
    for ep in episodes:
        ek = ep["episode_key"]
        pk = ep.get("parent_key") or ek
        ek_to_parent[ek] = pk
        parent_groups[ep["suite"]].add(pk)

    splits = {"train": [], "val": [], "test": []}
    for suite in SUITES:
        parents = sorted(parent_groups.get(suite, set()))
        if not parents:
            continue
        rng.shuffle(parents)
        n = len(parents)
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))
        train_p = set(parents[:n_train])
        val_p = set(parents[n_train:n_train + n_val])
        test_p = set(parents[n_train + n_val:])
        for ep in episodes:
            if ep["suite"] != suite:
                continue
            pk = ek_to_parent[ep["episode_key"]]
            if pk in train_p:
                splits["train"].append(ep["episode_key"])
            elif pk in val_p:
                splits["val"].append(ep["episode_key"])
            elif pk in test_p:
                splits["test"].append(ep["episode_key"])
    return splits


def build_task_grouped(episodes: list[dict], seed: int = 42) -> dict:
    import random
    rng = random.Random(seed)
    tasks = defaultdict(list)
    for ep in episodes:
        tasks[(ep["suite"], ep["task_id"])].append(ep["episode_key"])
    task_keys = sorted(tasks.keys())
    rng.shuffle(task_keys)
    n = len(task_keys)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    splits = {"train": [], "val": [], "test": []}
    for tk in task_keys[:n_train]:
        splits["train"].extend(tasks[tk])
    for tk in task_keys[n_train:n_train + n_val]:
        splits["val"].extend(tasks[tk])
    for tk in task_keys[n_train + n_val:]:
        splits["test"].extend(tasks[tk])
    return splits


def build_loso(episodes: list[dict], fold_name: str, seed: int = 42) -> dict:
    if fold_name not in LOSO_FOLDS:
        raise ValueError(f"Unknown LOSO fold: {fold_name}")
    config = LOSO_FOLDS[fold_name]
    by_suite = defaultdict(list)
    ek_to_parent = {}
    for ep in episodes:
        ek = ep["episode_key"]
        pk = ep.get("parent_key") or ek
        ek_to_parent[ek] = pk
        by_suite[ep["suite"]].append(ek)

    import random
    rng = random.Random(seed)

    # Per-suite stratification: split train/val within each training suite
    train_final, val_final = [], []
    for s in config["train"]:
        suite_eks = by_suite.get(s, [])
        suite_parents = sorted({ek_to_parent[ek] for ek in suite_eks})
        rng.shuffle(suite_parents)
        n = len(suite_parents)
        n_train = max(1, int(n * 0.8))
        train_p = set(suite_parents[:n_train])
        val_p = set(suite_parents[n_train:])
        if not train_p:
            raise ValueError("LOSO fold {} suite {} has no training parents".format(fold_name, s))
        if not val_p:
            raise ValueError("LOSO fold {} suite {} has no validation parents".format(fold_name, s))
        for ep in episodes:
            ek = ep["episode_key"]
            if ek not in suite_eks:
                continue
            pk = ek_to_parent[ek]
            if pk in train_p:
                train_final.append(ek)
            elif pk in val_p:
                val_final.append(ek)

    test_final = by_suite.get(config["test"], [])

    return {
        "train": train_final, "val": val_final, "test": test_final,
        "test_suite": config["test"], "train_suites": config["train"],
    }


def main():
    ap = argparse.ArgumentParser(description="Build detector splits")
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--split_type", required=True, choices=["episode_grouped","task_grouped","loso"])
    ap.add_argument("--loso_fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    episodes = load_episodes(args.episode_index)
    print(f"Loaded {len(episodes)} episodes")

    if args.split_type == "loso":
        if not args.loso_fold:
            sys.exit("--loso_fold required")
        splits = build_loso(episodes, args.loso_fold, args.seed)
    elif args.split_type == "episode_grouped":
        splits = build_episode_grouped(episodes, args.seed)
    elif args.split_type == "task_grouped":
        splits = build_task_grouped(episodes, args.seed)
    else:
        sys.exit(f"Unknown: {args.split_type}")

    errors = validate_no_parent_leakage(splits, episodes)
    if errors:
        print(f"SPLIT VALIDATION FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"  FAIL: {e}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"split_{args.split_type}.json"
    output = {
        "split_type": args.split_type,
        "loso_fold": args.loso_fold if args.split_type == "loso" else None,
        "seed": args.seed,
        "splits": {k: v for k, v in splits.items() if k in ("train","val","test")},
        "counts": {k: len(v) for k, v in splits.items() if k in ("train","val","test")},
        "test_suite": splits.get("test_suite"),
        "train_suites": splits.get("train_suites"),
        "validation_passed": True,
        "parent_leakage_checked": True,
    }
    sha = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    output["split_sha256"] = sha
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")
    for k in ["train","val","test"]:
        print(f"  {k}: {output['counts'][k]} episodes")
    print(f"  SHA256: {sha[:16]}")


if __name__ == "__main__":
    main()
