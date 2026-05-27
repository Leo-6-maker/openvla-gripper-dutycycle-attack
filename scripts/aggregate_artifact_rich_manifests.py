#!/usr/bin/env python3
"""Aggregate parallel worker manifest shards into final artifact-rich manifest."""
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    shard_dir = root / "tables" / "shards"
    tables_dir = root / "tables"
    reports_dir = root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Read all shards
    shards = sorted(shard_dir.glob("manifest_worker_*.csv"))
    if not shards:
        print("ERROR: No worker shards found in", shard_dir)
        return 1

    all_rows = []
    episode_keys = set()
    duplicate_keys = []
    incomplete_episodes = []

    for shard in shards:
        with open(shard) as f:
            for row in csv.DictReader(f):
                ep_key = f"{row['suite']}::{row['task_id']}::{row['task_name']}::{row['state_id']}::{row['seed']}::{row['run_id']}"
                if ep_key in episode_keys:
                    duplicate_keys.append(ep_key)
                episode_keys.add(ep_key)
                all_rows.append(row)

    if duplicate_keys:
        print(f"ERROR: {len(duplicate_keys)} duplicate episode keys found!")
        for k in duplicate_keys:
            print(f"  {k}")
        return 1

    print(f"Collected {len(all_rows)} episodes from {len(shards)} shards")
    print(f"Unique episode keys: {len(episode_keys)}")

    # Validate artifacts
    missing_manifests = []
    missing_steps = []
    missing_images = []

    for row in all_rows:
        run_dir = root / "runs" / row["suite"] / f"{row['task_name']}_state{row['state_id']}"
        manifest_path = run_dir / "run_manifest.json"
        step_path = run_dir / "step_records.jsonl"
        rgb_dir = run_dir / "frames"

        if not manifest_path.exists():
            missing_manifests.append(row["run_id"])

        # Check step_records
        if not step_path.exists() or step_path.stat().st_size == 0:
            missing_steps.append(row["run_id"])

        # Check image frames
        if rgb_dir.exists():
            frames = list(rgb_dir.glob("step_*.png")) + list(rgb_dir.glob("step_*.jpg"))
            if not frames:
                missing_images.append(row["run_id"])
        else:
            missing_images.append(row["run_id"])

    # Write aggregate manifest
    aggregate_path = tables_dir / "official_clean_artifact_rich_manifest.csv"
    if aggregate_path.exists() and not args.overwrite:
        print(f"ERROR: {aggregate_path} exists. Use --overwrite to replace.")
        return 1

    fields = list(all_rows[0].keys())
    with open(aggregate_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Completeness summary
    total = len(all_rows)
    success_count = sum(1 for r in all_rows if str(r.get("success", "")).lower() == "true")
    runtime_errors = sum(1 for r in all_rows if str(r.get("runtime_error", "")).lower() == "true")

    summary = [{
        "total_episodes": total,
        "n_success": success_count,
        "n_runtime_error": runtime_errors,
        "n_missing_manifest": len(missing_manifests),
        "n_missing_step_records": len(missing_steps),
        "n_missing_images": len(missing_images),
        "n_duplicate_keys": len(duplicate_keys),
    }]
    with open(tables_dir / "artifact_completeness_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Audit report
    report = [
        "# Artifact Completeness Audit",
        "",
        f"Total episodes: {total}",
        f"Successes: {success_count}",
        f"Runtime errors: {runtime_errors}",
        f"Missing manifests: {len(missing_manifests)}",
        f"Missing step_records: {len(missing_steps)}",
        f"Missing images: {len(missing_images)}",
        f"Duplicate keys: {len(duplicate_keys)}",
        "",
        "## Verdict",
        "",
    ]
    if missing_manifests or missing_steps or duplicate_keys:
        report.append("**FAILED** — see issues above.")
    else:
        report.append("**PASSED** — all artifacts complete.")

    with open(reports_dir / "ARTIFACT_COMPLETENESS_AUDIT.md", "w") as f:
        f.write("\n".join(report))

    print(f"Aggregate manifest: {aggregate_path}")
    print(f"Completeness: missing_manifests={len(missing_manifests)} missing_steps={len(missing_steps)} missing_images={len(missing_images)}")
    return 0 if not (missing_manifests or missing_steps or duplicate_keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
