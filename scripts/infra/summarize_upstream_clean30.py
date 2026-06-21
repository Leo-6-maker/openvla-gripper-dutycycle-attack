#!/usr/bin/env python3
"""Summarize upstream clean30 results: paired CSV, common success, manifest."""
import os, json, csv, hashlib, argparse, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def load_results(output_dir):
    results = {}
    for d in sorted(os.listdir(output_dir)):
        ep_dir = os.path.join(output_dir, d)
        if not os.path.isdir(ep_dir):
            continue
        rf = os.path.join(ep_dir, "result.json")
        if os.path.exists(rf):
            with open(rf) as f:
                r = json.load(f)
            key = r.get("label", d)
            results[key] = r
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_dir", required=True)
    parser.add_argument("--flash2_dir", required=True)
    parser.add_argument("--bf16_eager_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fp32 = load_results(args.fp32_dir)
    flash2 = load_results(args.flash2_dir)
    bf16_eager = load_results(args.bf16_eager_dir) if args.bf16_eager_dir else {}

    all_keys = sorted(set(list(fp32.keys()) + list(flash2.keys()) + list(bf16_eager.keys())))

    # Paired CSV
    paired_rows = []
    summary = {"total_episodes": len(all_keys), "fp32_success": 0, "flash2_success": 0,
               "bf16_eager_success": 0,
               "both_success": 0, "fp32_only": 0, "flash2_only": 0, "neither": 0, "invalid": 0,
               "common_success_keys": []}

    for key in all_keys:
        f = fp32.get(key, {})
        b = flash2.get(key, {})
        e = bf16_eager.get(key, {})

        row = {
            "episode_key": key,
            "task_idx": f.get("task_idx", b.get("task_idx", "")),
            "init_idx": f.get("init_idx", b.get("init_idx", "")),
            "init_state_sha": f.get("init_state_sha", ""),
            "fp32_success": f.get("success", False),
            "fp32_steps": f.get("steps", -1),
            "flash2_success": b.get("success", False),
            "flash2_steps": b.get("steps", -1),
        }
        if e:
            row["bf16_eager_success"] = e.get("success", False)
            row["bf16_eager_steps"] = e.get("steps", -1)

        # Pair class
        fs = row["fp32_success"]
        f2s = row["flash2_success"]
        inv = f.get("invalid", False) or b.get("invalid", False)

        if inv:
            row["pair_class"] = "invalid"
            summary["invalid"] += 1
        elif fs and f2s:
            row["pair_class"] = "both_success"
            summary["both_success"] += 1
            summary["common_success_keys"].append(key)
        elif fs:
            row["pair_class"] = "fp32_only"
            summary["fp32_only"] += 1
        elif f2s:
            row["pair_class"] = "flash2_only"
            summary["flash2_only"] += 1
        else:
            row["pair_class"] = "neither"
            summary["neither"] += 1

        if fs:
            summary["fp32_success"] += 1
        if f2s:
            summary["flash2_success"] += 1
        if e and e.get("success", False):
            summary["bf16_eager_success"] += 1

        paired_rows.append(row)

    # Write paired CSV
    csv_path = os.path.join(args.output_dir, "paired_fp32_flash2.csv")
    if paired_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=paired_rows[0].keys())
            w.writeheader()
            w.writerows(paired_rows)

    # Write per-profile CSVs
    for name, data in [("fp32_eager", fp32), ("bf16_flash2", flash2)]:
        if not data:
            continue
        csv_path_p = os.path.join(args.output_dir, "%s_results.csv" % name)
        keys = ["label", "task_idx", "init_idx", "steps", "success", "invalid",
                "termination", "gripper_flips", "duration_s"]
        with open(csv_path_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in sorted(data.values(), key=lambda r: r.get("label", "")):
                w.writerow(row)

    if bf16_eager:
        csv_path_e = os.path.join(args.output_dir, "bf16_eager_results.csv")
        keys = ["label", "task_idx", "init_idx", "steps", "success", "invalid",
                "termination", "gripper_flips", "duration_s"]
        with open(csv_path_e, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in sorted(bf16_eager.values(), key=lambda r: r.get("label", "")):
                w.writerow(row)

    # Task coverage of common success
    task_coverage = {}
    for key in summary["common_success_keys"]:
        ti = None
        for r in [fp32.get(key, {}), flash2.get(key, {})]:
            if "task_idx" in r:
                ti = r["task_idx"]
                break
        if ti is not None:
            task_coverage.setdefault(ti, []).append(key)

    # Common success manifest
    common_manifest = {
        "count": summary["both_success"],
        "keys": sorted(summary["common_success_keys"]),
        "task_coverage": {str(k): v for k, v in sorted(task_coverage.items())},
    }
    cm_path = os.path.join(args.output_dir, "common_success_manifest.json")
    json.dump(common_manifest, open(cm_path, "w"), indent=2)
    common_manifest["sha"] = sha256_hex(json.dumps(common_manifest, sort_keys=True).encode())

    # Summary JSON
    summary_json = {
        "fp32_success": summary["fp32_success"],
        "fp32_total": len(fp32),
        "flash2_success": summary["flash2_success"],
        "flash2_total": len(flash2),
        "bf16_eager_success": summary["bf16_eager_success"],
        "bf16_eager_total": len(bf16_eager) if bf16_eager else 0,
        "both_success": summary["both_success"],
        "fp32_only": summary["fp32_only"],
        "flash2_only": summary["flash2_only"],
        "neither": summary["neither"],
        "invalid": summary["invalid"],
        "common_success_keys": sorted(summary["common_success_keys"]),
        "task_coverage": {str(k): len(v) for k, v in sorted(task_coverage.items())},
    }
    json.dump(summary_json, open(os.path.join(args.output_dir, "upstream_clean30_summary.json"), "w"), indent=2)

    # Artifact manifest
    artifact_manifest = {
        "summary_sha": sha256_hex(json.dumps(summary_json, sort_keys=True).encode()),
        "common_success_sha": common_manifest["sha"],
        "paired_csv_sha": sha256_hex(open(csv_path, "rb").read()) if paired_rows else "N/A",
    }
    json.dump(artifact_manifest, open(os.path.join(args.output_dir, "upstream_clean30_artifact_manifest.json"), "w"), indent=2)

    print("FP32: %d/%d  Flash2: %d/%d" % (summary["fp32_success"], len(fp32),
                                          summary["flash2_success"], len(flash2)))
    print("Both: %d  FP32-only: %d  Flash2-only: %d  Neither: %d  Invalid: %d" % (
        summary["both_success"], summary["fp32_only"], summary["flash2_only"],
        summary["neither"], summary["invalid"]))
    print("Common success: %d episodes across %d tasks" % (summary["both_success"], len(task_coverage)))
    print("Output: %s" % args.output_dir)


if __name__ == "__main__":
    main()
