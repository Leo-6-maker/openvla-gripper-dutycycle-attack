#!/usr/bin/env python3
"""D7 Table1 paired statistics — McNemar test for TRUE vs RAND, TRUE vs CLEAN."""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def paired_success_matrix(audit_rows: List[Dict], cond_a: str, cond_b: str) -> Dict[str, int]:
    """Count pairs: (a_success, b_success) across 4 combos."""
    # Group by (suite, parent_key)
    by_parent: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    for r in audit_rows:
        if r.get("completed", "").lower() != "true":
            continue
        pk = (r["suite"], r["parent_key"])
        cond = r["condition"]
        success = r.get("task_success", "").lower() in ("true", "1")
        by_parent[pk][cond] = success

    matrix = {"both_ok": 0, "a_ok_b_fail": 0, "a_fail_b_ok": 0, "both_fail": 0}
    for pk, conds in by_parent.items():
        if cond_a not in conds or cond_b not in conds:
            continue
        a_ok = conds[cond_a]; b_ok = conds[cond_b]
        if a_ok and b_ok: matrix["both_ok"] += 1
        elif a_ok and not b_ok: matrix["a_ok_b_fail"] += 1
        elif not a_ok and b_ok: matrix["a_fail_b_ok"] += 1
        else: matrix["both_fail"] += 1
    return matrix


def mcnemar_p_value(b: int, c: int) -> float:
    """McNemar exact p-value (two-sided binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    p = 0.0
    for k in range(min(b, c) + 1):
        p += comb(n, k) * (0.5 ** n)
    for k in range(max(b, c), n + 1):
        p += comb(n, k) * (0.5 ** n)
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--postrun-audit-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = read_csv(args.postrun_audit_csv)

    clean_suites = ["libero_object", "libero_goal", "libero_spatial"]
    all_suites = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]

    comparisons = [
        ("TRUE_T10", "RAND_T10"),
        ("TRUE_T10", "CLEAN"),
        ("COMMAND_OPEN_ORACLE", "CLEAN"),
    ]

    results = []
    for suite_filter_name, suite_list in [("O_G_S_pooled", clean_suites), ("all_suites", all_suites)]:
        for cond_a, cond_b in comparisons:
            filtered = [r for r in audit if r["suite"] in suite_list]
            m = paired_success_matrix(filtered, cond_a, cond_b)
            b, c = m["a_ok_b_fail"], m["a_fail_b_ok"]
            p = mcnemar_p_value(b, c)
            total = sum(m.values())
            results.append({
                "scope": suite_filter_name, "cond_a": cond_a, "cond_b": cond_b,
                "n_pairs": total, "both_ok": m["both_ok"], "a_ok_b_fail": b,
                "a_fail_b_ok": c, "both_fail": m["both_fail"],
                "mcnemar_p": f"{p:.6f}",
                "significant_p05": p < 0.05,
            })

    for r in results:
        sig = "**" if r["significant_p05"] else ""
        print(f"  {r['scope']} {r['cond_a']} vs {r['cond_b']}: "
              f"{r['a_ok_b_fail']}/{r['a_fail_b_ok']} discordant, p={r['mcnemar_p']} {sig}")

    with open(out / "d7_paired_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    report = {
        "gate": "D7_PAIRED_STATISTICS",
        "created_at_unix": time.time(),
        "git_commit": args.git_commit,
        "results": results,
    }
    (out / "d7_paired_stats_report.json").write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
