#!/usr/bin/env python3
"""Compare A800 vs 2080Ti spatial static parity results (MIG2B)."""
import csv, sys, json

def load_a800_lane_o(path):
    """Load A800-O Lane O results."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 15:
                continue
            if row[10] != "Lane_O" or row[11] != "1":
                continue
            key = row[0] + "_step" + row[2]
            results[key] = {
                "action": [float(x) for x in row[13].split()],
                "gripper_class": row[16],
                "file": row[4],
            }
    return results

def load_2080ti(path):
    """Load 2080Ti-M results."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 7 or "episode" in row[0]:
                continue
            key = row[0] + "_step" + row[2]
            results[key] = {
                "action": [float(x) for x in row[5].split()],
                "gripper_class": row[6],
                "file": row[3],
            }
    return results

def compare(a800_csv, ti_csv):
    a800 = load_a800_lane_o(a800_csv)
    ti = load_2080ti(ti_csv)

    print(f"A800-O frames: {len(a800)}")
    print(f"2080Ti-M frames: {len(ti)}")
    print()

    common = sorted(set(a800) & set(ti))
    only_a800 = sorted(set(a800) - set(ti))
    only_ti = sorted(set(ti) - set(a800))

    if only_a800:
        print(f"Only in A800: {only_a800}")
    if only_ti:
        print(f"Only in 2080Ti: {only_ti}")

    gripper_matches = 0
    action_matches_1e4 = 0
    max_overall_diff = 0.0
    rows = []

    for key in common:
        av, tv = a800[key]["action"], ti[key]["action"]
        dim_diffs = [abs(av[i] - tv[i]) for i in range(7)]
        max_dim_diff = max(dim_diffs)
        max_overall_diff = max(max_overall_diff, max_dim_diff)

        g_match = a800[key]["gripper_class"] == ti[key]["gripper_class"]
        if g_match:
            gripper_matches += 1

        a_match = max_dim_diff < 1e-4
        if a_match:
            action_matches_1e4 += 1

        rows.append({
            "key": key, "file": av[4] if len(av) > 4 else key,
            "max_dim_diff": max_dim_diff,
            "dim_diffs": dim_diffs,
            "gripper_match": g_match,
            "action_match_1e4": a_match,
        })
        print(f"{'MATCH' if a_match else 'DIFF'} {key}: max_diff={max_dim_diff:.2e} "
              f"gripper={'OK' if g_match else 'MISMATCH'}")

    print(f"\n=== SUMMARY ===")
    print(f"Common frames: {len(common)}")
    print(f"Action match (1e-4): {action_matches_1e4}/{len(common)}")
    print(f"Gripper match: {gripper_matches}/{len(common)}")
    print(f"Max overall action diff: {max_overall_diff:.2e}")
    print(f"Token parity: NOT_MEASURED (2080Ti used predict_action API)")
    print(f"Classification: CROSS_HOST_RUNTIME_PARITY_WITH_CONFOUNDERS")
    print(f"  Confounders: 2080Ti=FP32+4GPU+10GBcap vs A800=BF16+singleGPU")

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800_csv", required=True)
    parser.add_argument("--ti2080_csv", required=True)
    args = parser.parse_args()
    compare(args.a800_csv, args.ti2080_csv)
