#!/usr/bin/env python3
"""G0.5: Recalculate upstream clean30 integrity from raw episode directories.
Does NOT read summary files — only reads per-episode result.json.
Outputs integrity_audit.json and integrity_audit.csv."""
import os, json, csv, hashlib, argparse, sys
from collections import Counter


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def load_episodes(output_dir):
    """Load all per-episode result.json files. Returns dict keyed by label."""
    episodes = {}
    for d in sorted(os.listdir(output_dir)):
        ep_dir = os.path.join(output_dir, d)
        if not os.path.isdir(ep_dir):
            continue
        rf = os.path.join(ep_dir, "result.json")
        if os.path.exists(rf):
            with open(rf) as f:
                r = json.load(f)
            key = r.get("label", d)
            episodes[key] = r
    return episodes


def verify_profile(name, episodes, expected_total=30, expected_invalid=0):
    """Verify a single profile's integrity."""
    keys = sorted(episodes.keys())
    unique = len(set(keys))
    completed = len(episodes)
    success = sum(1 for v in episodes.values() if v.get("success", False))
    invalid = sum(1 for v in episodes.values() if v.get("invalid", False))
    has_result = sum(1 for v in episodes.values())

    issues = []
    if unique != expected_total:
        issues.append("expected %d unique keys, got %d" % (expected_total, unique))
    if completed != expected_total:
        issues.append("expected %d results, got %d" % (expected_total, completed))

    return {
        "profile": name,
        "unique_keys": unique,
        "completed": completed,
        "success": success,
        "invalid": invalid,
        "success_rate": round(100 * success / max(1, completed), 1),
        "issues": issues,
        "keys": keys,
    }


def compute_paired(fp32_eps, flash2_eps):
    """Compute paired analysis from raw episode dicts."""
    all_keys = sorted(set(list(fp32_eps.keys()) + list(flash2_eps.keys())))

    both_success, fp32_only, flash2_only, neither, invalid = [], [], [], [], []
    rows = []

    for key in all_keys:
        f = fp32_eps.get(key, {})
        b = flash2_eps.get(key, {})
        fs = f.get("success", False)
        f2s = b.get("success", False)
        inv = f.get("invalid", False) or b.get("invalid", False)

        row = {
            "episode_key": key,
            "task_idx": f.get("task_idx", b.get("task_idx", "")),
            "init_idx": f.get("init_idx", b.get("init_idx", "")),
            "fp32_success": fs,
            "fp32_steps": f.get("steps", -1),
            "flash2_success": f2s,
            "flash2_steps": b.get("steps", -1),
        }

        if inv:
            pair_class = "invalid"
            invalid.append(key)
        elif fs and f2s:
            pair_class = "both_success"
            both_success.append(key)
        elif fs:
            pair_class = "fp32_only"
            fp32_only.append(key)
        elif f2s:
            pair_class = "flash2_only"
            flash2_only.append(key)
        else:
            pair_class = "neither"
            neither.append(key)

        row["pair_class"] = pair_class
        rows.append(row)

    return {
        "both_success": both_success,
        "fp32_only": fp32_only,
        "flash2_only": flash2_only,
        "neither": neither,
        "invalid": invalid,
        "counts": {
            "both": len(both_success), "fp32": len(fp32_only),
            "flash2": len(flash2_only), "neither": len(neither),
            "invalid": len(invalid),
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_dir", required=True)
    parser.add_argument("--flash2_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fp32 = load_episodes(args.fp32_dir)
    flash2 = load_episodes(args.flash2_dir)

    result_fp32 = verify_profile("fp32_eager_upstream", fp32)
    result_flash2 = verify_profile("bf16_flash2_upstream", flash2)
    paired = compute_paired(fp32, flash2)

    # Common success verification
    common = paired["both_success"]
    for key in common:
        assert fp32.get(key, {}).get("success", False), "%s not success in FP32" % key
        assert flash2.get(key, {}).get("success", False), "%s not success in Flash2" % key

    # Task coverage
    task_coverage = {}
    for key in common:
        ti = fp32.get(key, {}).get("task_idx", flash2.get(key, {}).get("task_idx"))
        if ti is not None:
            task_coverage.setdefault(ti, []).append(key)

    integrity = {
        "fp32": result_fp32,
        "flash2": result_flash2,
        "paired": paired["counts"],
        "common_success_count": len(common),
        "common_success_keys": sorted(common),
        "task_coverage": {str(k): len(v) for k, v in sorted(task_coverage.items())},
        "common_success_verified": True,
        "self_consistency": {
            "fp32_success_eq_24": result_fp32["success"] == 24,
            "flash2_success_eq_22": result_flash2["success"] == 22,
            "both_eq_19": paired["counts"]["both"] == 19,
            "fp32_only_eq_5": paired["counts"]["fp32"] == 5,
            "flash2_only_eq_3": paired["counts"]["flash2"] == 3,
            "neither_eq_3": paired["counts"]["neither"] == 3,
            "invalid_eq_0": paired["counts"]["invalid"] == 0,
        },
    }

    all_ok = all(integrity["self_consistency"].values())
    integrity["all_checks_pass"] = all_ok

    # Output
    json.dump(integrity, open(os.path.join(args.output_dir, "integrity_audit.json"), "w"), indent=2)

    with open(os.path.join(args.output_dir, "integrity_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=paired["rows"][0].keys())
        w.writeheader()
        w.writerows(paired["rows"])

    print("FP32: %d unique / %d completed / %d success / %d invalid" % (
        result_fp32["unique_keys"], result_fp32["completed"],
        result_fp32["success"], result_fp32["invalid"]))
    print("Flash2: %d unique / %d completed / %d success / %d invalid" % (
        result_flash2["unique_keys"], result_flash2["completed"],
        result_flash2["success"], result_flash2["invalid"]))
    print("Paired: both=%d fp32=%d flash2=%d neither=%d invalid=%d" % (
        paired["counts"]["both"], paired["counts"]["fp32"],
        paired["counts"]["flash2"], paired["counts"]["neither"],
        paired["counts"]["invalid"]))
    print("Common success: %d keys, %d tasks" % (len(common), len(task_coverage)))
    print("All checks: %s" % ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
