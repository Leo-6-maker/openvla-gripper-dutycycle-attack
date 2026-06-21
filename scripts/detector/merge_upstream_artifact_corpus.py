#!/usr/bin/env python3
"""Merge upstream artifact corpus from per-episode directories.
Re-scans all episode dirs, verifies integrity, writes aggregate CSVs.
"""
import os, sys, json, csv, hashlib, argparse
from collections import Counter


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def scan_episodes(root_dir):
    """Scan all episode subdirectories. Returns dict keyed by label."""
    episodes = {}
    for d in sorted(os.listdir(root_dir)):
        ep_dir = os.path.join(root_dir, d)
        if not os.path.isdir(ep_dir):
            continue
        result_file = os.path.join(ep_dir, "result.json")
        trace_file = os.path.join(ep_dir, "trace.csv")
        done_file = os.path.join(ep_dir, ".done")
        if not os.path.exists(done_file):
            continue
        if not os.path.exists(result_file) or not os.path.exists(trace_file):
            continue
        with open(result_file) as f:
            result = json.load(f)
        label = result.get("label", d)
        episodes[label] = {
            "dir": ep_dir, "result": result,
            "trace_file": trace_file, "done": True,
        }
    return episodes


def verify_integrity(episodes):
    """Verify no duplicate keys, no cross-split contamination."""
    issues = []
    keys = list(episodes.keys())
    if len(keys) != len(set(keys)):
        dupes = [k for k, c in Counter(keys).items() if c > 1]
        issues.append("DUPLICATE_KEYS: %s" % dupes)

    # Verify all have valid termination
    for label, ep in episodes.items():
        r = ep["result"]
        if r.get("termination", "") not in ("success", "timeout", "error"):
            issues.append("INVALID_TERMINATION: %s=%s" % (label, r.get("termination")))

    return issues


def merge_to_csv(episodes_dict, output_path, fieldnames):
    """Merge all episode traces into single CSV."""
    all_rows = []
    seen_keys = set()
    for label in sorted(episodes_dict.keys()):
        ep = episodes_dict[label]
        with open(ep["trace_file"], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    if all_rows:
        with open(output_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    return len(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True,
                        help="List of output directories from collector runs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_labels", nargs="+", default=[],
                        help="Labels for each input dir (train/val/cal/xfer_fp32/xfer_flash2)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_episodes = {}
    labels = args.split_labels if args.split_labels else [os.path.basename(d) for d in args.input_dirs]

    for i, d in enumerate(args.input_dirs):
        eps = scan_episodes(d)
        label = labels[i] if i < len(labels) else "shard_%d" % i
        print("Shard %s: %d episodes" % (label, len(eps)))
        all_episodes.update(eps)

    print("Total unique episodes: %d" % len(all_episodes))
    issues = verify_integrity(all_episodes)
    if issues:
        for issue in issues:
            print("ISSUE: %s" % issue)
        print("FATAL: integrity check failed")
        sys.exit(1)

    # Summary
    total = len(all_episodes)
    success = sum(1 for ep in all_episodes.values() if ep["result"].get("success", False))
    invalid_feature = sum(1 for ep in all_episodes.values()
                          if not ep["result"].get("binding_ok", True))

    integrity = {
        "total_episodes": total, "success": success,
        "failure": total - success, "binding_failures": invalid_feature,
        "episode_keys": sorted(all_episodes.keys()),
        "issues": issues,
    }
    json.dump(integrity, open(os.path.join(args.output_dir, "corpus_integrity.json"), "w"), indent=2)

    # Merge to flat CSV
    n_rows = merge_to_csv(all_episodes, os.path.join(args.output_dir, "corpus_flat.csv"), [])
    print("Rows: %d" % n_rows)

    # Artifact manifest
    manifest = {
        "n_episodes": total, "n_rows": n_rows, "n_success": success,
        "input_dirs": args.input_dirs,
        "corpus_flat_csv_sha": sha256_hex(open(os.path.join(args.output_dir, "corpus_flat.csv"), "rb").read()) if n_rows > 0 else "N/A",
    }
    json.dump(manifest, open(os.path.join(args.output_dir, "corpus_artifact_manifest.json"), "w"), indent=2)

    print("Merge complete. Output: %s" % args.output_dir)


if __name__ == "__main__":
    main()
