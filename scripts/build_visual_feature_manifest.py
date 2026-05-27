#!/usr/bin/env python3
"""Build visual feature manifest from artifact-rich clean data.

Collects image_path references. Does NOT run visual encoders.
Frozen CLIP/SigLIP/DINO extraction is deferred to Milestone 2E.
"""
import argparse, csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_root", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.artifact_root)
    runs_dir = root / "runs"
    tables_dir = root / "tables"
    reports_dir = root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    suites = {}

    # Walk all run directories
    for suite_dir in sorted(runs_dir.glob("*")):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        suites.setdefault(suite, {"n_runs": 0, "n_images": 0, "n_missing": 0})

        for run_dir in sorted(suite_dir.glob("*_state*")):
            if not run_dir.is_dir():
                continue
            suites[suite]["n_runs"] += 1

            # Collect step_records if available
            step_file = run_dir / "step_records.jsonl"
            if step_file.exists():
                import json
                with open(step_file) as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            img_path = rec.get("image_path", "")
                            img_avail = rec.get("image_path_available", False)
                            if img_avail and img_path:
                                suites[suite]["n_images"] += 1
                            else:
                                suites[suite]["n_missing"] += 1
                            manifest_rows.append({
                                "run_id": rec.get("run_id", ""),
                                "episode_key": f"{rec.get('suite','')}::{rec.get('task_id','')}::{rec.get('task_name','')}::{rec.get('state_id','')}::{rec.get('seed','')}::{rec.get('run_id','')}",
                                "suite": rec.get("suite", suite),
                                "task_id": rec.get("task_id", ""),
                                "task_name": rec.get("task_name", ""),
                                "state_id": rec.get("state_id", ""),
                                "step_idx": rec.get("step_idx", ""),
                                "image_path": img_path,
                                "image_path_available": img_avail,
                                "visual_feature_path": "",
                                "visual_feature_available": False,
                                "visual_encoder_name": "",
                                "visual_feature_dim": "",
                                "visual_feature_status": "pending",
                                "visual_feature_reason": "Frozen encoder extraction deferred to Milestone 2E",
                            })
            else:
                # Fallback: scan frames directory
                frames_dir = run_dir / "frames"
                if frames_dir.exists():
                    for frame_file in sorted(frames_dir.glob("step_*.*")):
                        suites[suite]["n_images"] += 1
                        frame_name = frame_file.stem
                        try:
                            step_idx = int(frame_name.split("_")[1])
                        except (IndexError, ValueError):
                            step_idx = 0
                        manifest_rows.append({
                            "run_id": run_dir.name,
                            "episode_key": f"{suite}::unknown::unknown::unknown::0::{run_dir.name}",
                            "suite": suite,
                            "task_id": "",
                            "task_name": "",
                            "state_id": "",
                            "step_idx": step_idx,
                            "image_path": str(frame_file),
                            "image_path_available": True,
                            "visual_feature_path": "",
                            "visual_feature_available": False,
                            "visual_encoder_name": "",
                            "visual_feature_dim": "",
                            "visual_feature_status": "pending",
                            "visual_feature_reason": "Frozen encoder extraction deferred to Milestone 2E",
                        })

    # Write manifest
    if manifest_rows:
        fields = list(manifest_rows[0].keys())
        with open(tables_dir / "visual_feature_manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"Visual manifest: {len(manifest_rows)} rows")
    else:
        print("No images found. Visual manifest empty.")

    # Report
    report = [
        "# Visual Feature Manifest Status",
        "",
        "| Suite | Runs | Images | Missing |",
        "|-------|------|--------|---------|",
    ]
    for suite, stats in sorted(suites.items()):
        report.append(f"| {suite} | {stats['n_runs']} | {stats['n_images']} | {stats['n_missing']} |")

    report.extend([
        "",
        "## Status",
        "",
        f"Total images found: {sum(s['n_images'] for s in suites.values())}",
        f"Visual features extracted: 0",
        "",
        "## Next Steps",
        "",
        "- Frozen CLIP/SigLIP/DINO feature extraction deferred to Milestone 2E",
        "- visual_feature_available=false for all entries",
    ])

    with open(reports_dir / "VISUAL_FEATURE_MANIFEST_STATUS.md", "w") as f:
        f.write("\n".join(report))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
