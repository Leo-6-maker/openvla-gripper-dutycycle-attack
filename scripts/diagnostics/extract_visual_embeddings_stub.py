#!/usr/bin/env python3
"""Dummy visual embedding extractor for pipeline shape tests.

This does not load DINO/CLIP/SigLIP/OpenVLA and does not use GPU. Dummy features
are pipeline smoke only, not visual evidence.
"""

import argparse
import csv
import os
import random


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv", default="tables/visual_transfer_dataset_v0.csv")
    ap.add_argument("--output-dir", default="outputs/visual_transfer_features_stub_v0")
    ap.add_argument("--encoder", default="dummy", choices=["dummy"])
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--output-manifest", default="tables/visual_transfer_feature_manifest_stub_v0.csv")
    ap.add_argument("--output-report", default="reports/VISUAL_TRANSFER_STUB_FEATURE_SUMMARY.md")
    return ap.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_vector(path, dim, seed):
    rng = random.Random(seed)
    values = ["%.6f" % 0.0 for _ in range(dim)]
    # Keep deterministic zero-like vectors; one tiny checksum value helps catch file swaps.
    if dim:
        values[0] = "%.6f" % (rng.random() * 0.0)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(values) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_rows(args.dataset_csv)
    manifest = []
    for r in rows:
        if str(r.get("visual_available", "")).lower() != "true":
            continue
        sample_id = r["sample_id"]
        out = os.path.join(args.output_dir, sample_id + ".csv")
        write_vector(out, args.feature_dim, sample_id)
        manifest.append({
            "sample_id": sample_id,
            "encoder": args.encoder,
            "feature_dim": args.feature_dim,
            "feature_path": out,
            "source_image_path": r.get("image_trigger_path", ""),
            "scientific_use": "pipeline_smoke_only",
        })
    os.makedirs(os.path.dirname(args.output_manifest) or ".", exist_ok=True)
    with open(args.output_manifest, "w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "encoder", "feature_dim", "feature_path", "source_image_path", "scientific_use"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# Visual Transfer Stub Feature Summary",
            "",
            f"**Dataset rows**: {len(rows)}",
            f"**Feature rows written**: {len(manifest)}",
            f"**Encoder**: `{args.encoder}`",
            f"**Feature dim**: {args.feature_dim}",
            "",
            "Dummy features are pipeline smoke only, not visual evidence.",
            "",
        ]))
    print("Feature manifest: %s" % args.output_manifest)
    print("Report: %s" % args.output_report)


if __name__ == "__main__":
    main()
