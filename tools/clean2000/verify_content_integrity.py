#!/usr/bin/env python3
"""P0-6: Verify CLEAN2000 content integrity by recomputing artifact SHAs.

For every episode in the canonical index, recompute SHA256 of:
  episode_summary.json, step_telemetry.csv, privileged_step_records.jsonl, COMPLETE.json

Compare against declared values in the canonical index.
Report all mismatches. Fail-closed: any mismatch = exit 1.

Usage:
  python verify_content_integrity.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --output_dir /path/to/output

Output:
  CONTENT_INTEGRITY_REPORT.json
  SOURCE_ARTIFACT_SHA256SUMS.txt
  EPISODE_INTEGRITY_LEDGER.jsonl
  INTEGRITY_ENVELOPE.json
"""

import argparse
import hashlib
import json
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="CLEAN2000 content integrity verification")
    p.add_argument("--index", required=True,
                   help="Path to CLEAN2000_INDEX_DRAFT.jsonl")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def compute_sha(path):
    """Compute SHA256 of a file. Returns empty string if file missing."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load index
    print("Loading index: {}".format(args.index))
    rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print("  {} episodes".format(len(rows)))

    # Verify each episode
    print("Recomputing artifact SHAs...")
    ledger = []
    mismatches = []
    verified = 0
    missing_artifacts = 0
    modified_after_freeze = 0
    mtime_anomalies = 0

    freeze_ts = time.time()  # current time as reference

    source_sha_lines = []
    per_suite = {}
    artifact_types = ["episode_summary.json", "step_telemetry.csv",
                      "privileged_step_records.jsonl", "COMPLETE.json"]
    field_map = {
        "episode_summary.json": "episode_summary_sha256",
        "step_telemetry.csv": "step_telemetry_sha256",
        "privileged_step_records.jsonl": "privileged_records_sha256",
        "COMPLETE.json": "complete_marker_sha256",
    }

    for i, row in enumerate(rows):
        ek = row["episode_key"]
        ep_dir = row["source_root"]
        suite = row["suite"]

        entry = {"episode_key": ek, "source_root": ep_dir, "suite": suite,
                 "artifacts": {}, "mismatches": []}
        all_ok = True

        for art_name in artifact_types:
            art_path = os.path.join(ep_dir, art_name)
            fresh_sha = compute_sha(art_path)
            declared_field = field_map[art_name]
            declared_sha = row.get(declared_field, "")

            entry["artifacts"][art_name] = {
                "path": art_path,
                "exists": os.path.exists(art_path),
                "fresh_sha": fresh_sha,
                "declared_sha": declared_sha,
            }

            # COMPLETE.json may be absent (Object500 episodes don't have it)
            if art_name == "COMPLETE.json" and not os.path.exists(art_path):
                continue

            if not os.path.exists(art_path):
                entry["mismatches"].append({
                    "artifact": art_name,
                    "error": "FILE_MISSING",
                })
                missing_artifacts += 1
                all_ok = False
                continue

            if fresh_sha != declared_sha:
                entry["mismatches"].append({
                    "artifact": art_name,
                    "fresh_sha": fresh_sha,
                    "declared_sha": declared_sha,
                    "error": "SHA_MISMATCH",
                })
                all_ok = False

            # Check mtime
            try:
                mtime = os.path.getmtime(art_path)
                if mtime > freeze_ts:
                    entry["mismatches"].append({
                        "artifact": art_name,
                        "error": "MTIME_AFTER_FREEZE",
                        "mtime": mtime,
                        "freeze_ts": freeze_ts,
                    })
                    mtime_anomalies += 1
            except OSError:
                pass

            # Add to source SHA manifest
            if fresh_sha:
                rel_path = os.path.relpath(art_path, os.path.dirname(ep_dir))
                source_sha_lines.append("{}  {}".format(fresh_sha, rel_path))

        if not all_ok:
            mismatches.append(entry)
        else:
            verified += 1

        ledger.append(entry)

        if (i + 1) % 500 == 0:
            print("  {} / {} ...".format(i + 1, len(rows)))

        # Per-suite stats
        per_suite[suite] = per_suite.get(suite, {"verified": 0, "mismatches": 0})
        if all_ok:
            per_suite[suite]["verified"] += 1
        else:
            per_suite[suite]["mismatches"] += 1

    # ── Results ──
    print()
    total_episodes = len(rows)
    passed = len(mismatches) == 0

    print("=== Content Integrity Summary ===")
    print("  Total episodes:     {}".format(total_episodes))
    print("  Verified (all OK):  {}".format(verified))
    print("  Mismatches:         {}".format(len(mismatches)))
    print("  Missing artifacts:  {}".format(missing_artifacts))
    print("  Mtime anomalies:    {}".format(mtime_anomalies))
    for suite in sorted(per_suite):
        s = per_suite[suite]
        print("  {}: verified={}, mismatches={}".format(suite, s["verified"], s["mismatches"]))
    print("  RESULT: {}".format("PASS" if passed else "FAIL"))

    # ── Write outputs ──
    # Integrity ledger
    ledger_path = os.path.join(args.output_dir, "EPISODE_INTEGRITY_LEDGER.jsonl")
    with open(ledger_path, "w") as f:
        for entry in ledger:
            f.write(json.dumps(entry) + "\n")
    print("  {}".format(ledger_path))

    # Source artifact SHAs
    sums_path = os.path.join(args.output_dir, "SOURCE_ARTIFACT_SHA256SUMS.txt")
    with open(sums_path, "w") as f:
        f.write("\n".join(sorted(source_sha_lines)) + "\n")
    print("  {} ({} entries)".format(sums_path, len(source_sha_lines)))

    # Integrity report
    report = {
        "gate": "CLEAN2000_CONTENT_INTEGRITY_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "total_episodes": total_episodes,
        "verified": verified,
        "mismatches": len(mismatches),
        "missing_artifacts": missing_artifacts,
        "mtime_anomalies": mtime_anomalies,
        "per_suite": per_suite,
        "canonical_index_sha256": hashlib.sha256(
            open(args.index, "rb").read()).hexdigest(),
    }
    report_path = os.path.join(args.output_dir, "CONTENT_INTEGRITY_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  {}".format(report_path))

    # Integrity envelope
    envelope = {
        "gate": "CLEAN2000_CONTENT_INTEGRITY_ENVELOPE_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "binds_to": "CLEAN2000_CANONICAL_V1",
        "requires_p0_6": "VERIFIED" if passed else "FAILED",
    }
    env_path = os.path.join(args.output_dir, "INTEGRITY_ENVELOPE.json")
    with open(env_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print("  {}".format(env_path))

    if not passed:
        # Show first 5 mismatches
        print()
        print("First 5 mismatches:")
        for m in mismatches[:5]:
            print("  {}: {}".format(m["episode_key"], m["mismatches"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
