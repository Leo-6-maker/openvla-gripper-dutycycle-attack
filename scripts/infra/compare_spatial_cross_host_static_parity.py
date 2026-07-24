#!/usr/bin/env python3
"""Compare A800 vs 2080Ti spatial static parity results (MIG2B).
Uses csv.DictReader with schema assertions. No hardcoded column indices."""
import csv, hashlib, json, math, os, sys, argparse
import numpy as np

A800_REQUIRED = [
    "episode", "step", "frame_file", "lane", "run",
    "final_action", "gripper_class",
    "raw_file_sha256", "decoded_rgb_sha256",
]
TI2080_REQUIRED = [
    "episode", "step", "frame_file",
    "generated_token_ids", "final_action", "gripper_class",
]

def assert_schema(reader, required, label):
    missing = set(required) - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    return True

def load_a800_lane_o(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert_schema(reader, A800_REQUIRED, "A800 CSV")
        results = {}
        for row in reader:
            if row["lane"] != "Lane_O" or row["run"] != "1":
                continue
            key = row["episode"] + "_step" + row["step"]
            action = [float(x) for x in row["final_action"].split()]
            assert len(action) == 7, f"A800 {key}: action dim != 7"
            assert all(math.isfinite(x) for x in action), f"A800 {key}: non-finite action"
            results[key] = {
                "action": action,
                "gripper_class": row["gripper_class"],
                "frame_file": row["frame_file"],
                "raw_file_sha256": row.get("raw_file_sha256", ""),
                "decoded_rgb_sha256": row.get("decoded_rgb_sha256", ""),
            }
    print(f"A800-O Lane O frames: {len(results)}")
    assert len(results) == 10, f"A800: expected 10 frames, got {len(results)}"
    assert len(set(results)) == 10, "A800: duplicate keys found"
    return results

def load_2080ti(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert_schema(reader, TI2080_REQUIRED, "2080Ti CSV")
        results = {}
        for row in reader:
            key = row["episode"] + "_step" + row["step"]
            action = [float(x) for x in row["final_action"].split()]
            assert len(action) == 7, f"2080Ti {key}: action dim != 7"
            assert all(math.isfinite(x) for x in action), f"2080Ti {key}: non-finite action"
            results[key] = {
                "action": action,
                "gripper_class": row["gripper_class"],
                "frame_file": row["frame_file"],
            }
    print(f"2080Ti-M frames: {len(results)}")
    assert len(results) == 10, f"2080Ti: expected 10 frames, got {len(results)}"
    assert len(set(results)) == 10, "2080Ti: duplicate keys found"
    return results

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def compare(a800_csv, ti_csv, output_dir, expected_a800_sha, expected_ti_sha):
    # Verify frozen SHAs
    a800_sha = sha256_file(a800_csv)
    ti_sha = sha256_file(ti_csv)
    assert a800_sha == expected_a800_sha, f"A800 CSV SHA mismatch: {a800_sha} != {expected_a800_sha}"
    assert ti_sha == expected_ti_sha, f"2080Ti CSV SHA mismatch: {ti_sha} != {expected_ti_sha}"
    print(f"A800 CSV SHA verified: {a800_sha[:16]}...")
    print(f"2080Ti CSV SHA verified: {ti_sha[:16]}...")

    a800 = load_a800_lane_o(a800_csv)
    ti = load_2080ti(ti_csv)

    common = sorted(set(a800) & set(ti))
    only_a800 = sorted(set(a800) - set(ti))
    only_ti = sorted(set(ti) - set(a800))
    assert len(only_a800) == 0, f"Frames only in A800: {only_a800}"
    assert len(only_ti) == 0, f"Frames only in 2080Ti: {only_ti}"
    assert len(common) == 10, f"Expected 10 common frames, got {len(common)}"

    max_action_diff = 0.0
    gripper_matches = 0
    summary_rows = []
    per_dim_rows = []

    for key in common:
        av = a800[key]["action"]
        tv = ti[key]["action"]
        dim_diffs = [abs(av[i] - tv[i]) for i in range(7)]
        max_dim = max(dim_diffs)
        l2 = math.sqrt(sum(d * d for d in dim_diffs))
        max_action_diff = max(max_action_diff, max_dim)
        g_match = a800[key]["gripper_class"] == ti[key]["gripper_class"]
        if g_match:
            gripper_matches += 1

        summary_rows.append({
            "frame_key": key,
            "a800_action": av,
            "ti2080_action": tv,
            "max_abs_diff": max_dim,
            "l2_diff": l2,
            "gripper_match": g_match,
        })
        per_dim_rows.append([key] + dim_diffs)

        marker = "MATCH" if max_dim < 1e-4 else "DIFF"
        print(f"{marker} {key}: max_diff={max_dim:.2e} L2={l2:.2e} "
              f"gripper={'OK' if g_match else 'MISMATCH'}")

    # Write summary CSV
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/spatial_cross_host_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_key", "a800_action", "ti2080_action",
                     "max_abs_diff", "l2_diff", "gripper_match"])
        for r in summary_rows:
            w.writerow([
                r["frame_key"],
                " ".join(f"{x:.12f}" for x in r["a800_action"]),
                " ".join(f"{x:.12f}" for x in r["ti2080_action"]),
                f"{r['max_abs_diff']:.6e}",
                f"{r['l2_diff']:.6e}",
                r["gripper_match"],
            ])

    # Write per-dim diff CSV
    dim_names = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]
    with open(f"{output_dir}/spatial_cross_host_per_dim_diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_key"] + [f"diff_{d}" for d in dim_names])
        for row in per_dim_rows:
            w.writerow(row)

    # Evidence manifest
    manifest = {
        "a800_csv_sha256": a800_sha,
        "ti2080_csv_sha256": ti_sha,
        "input_bundle_sha256": "13d31e707c0b5c3895c61472dee4cc97546b6cdcd0859d1a70dce1ed785ff1ab",
        "comparison_script": "scripts/infra/compare_spatial_cross_host_static_parity.py",
        "common_frames": len(common),
        "gripper_matches": gripper_matches,
        "max_action_abs_diff": max_action_diff,
        "classification": "CROSS_HOST_RUNTIME_PARITY_WITH_CONFOUNDERS",
        "confounders": [
            "2080Ti: FP32, 4-GPU device_map=auto, 10GB per-GPU cap",
            "A800: BF16, single GPU, no memory cap",
        ],
        "token_parity": "NOT_MEASURED",
    }
    with open(f"{output_dir}/spatial_cross_host_evidence_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== MIG2B SUMMARY ===")
    print(f"Common frames: {len(common)}")
    print(f"Gripper match: {gripper_matches}/{len(common)}")
    print(f"Max action abs diff: {max_action_diff:.2e}")
    print(f"Classification: {manifest['classification']}")
    for c in manifest["confounders"]:
        print(f"  {c}")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800_csv", required=True)
    parser.add_argument("--ti2080_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_a800_sha", default="3138dbf7799f5c6899840b084a698cbfa191226c9b2ae2ec95a1bf94430eff35")
    parser.add_argument("--expected_ti_sha", default="286b462fb795782c70bb63a919911a01571f100a6a486a62301a5e82b870349b")
    args = parser.parse_args()
    compare(args.a800_csv, args.ti2080_csv, args.output_dir,
            args.expected_a800_sha, args.expected_ti_sha)
