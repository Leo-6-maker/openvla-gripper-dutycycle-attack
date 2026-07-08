#!/usr/bin/env python3
"""D7 Table1 four-suite aggregator.

Reads postrun audit CSV + episode summaries. Computes:
  - Success Rate (SR)
  - Failure Rate (FR = 1 - SR)
  - Wilson 95% Confidence Interval
  - Attack Frames, Trigger Rate, Duty Cycle
  - Per-condition, per-suite Panel A main table

CPU-only. No env.step, no rollout, no attack.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_panel_a(
    episode_rows: List[Dict[str, str]],
    conditions: List[str],
    suites: List[str],
) -> List[Dict[str, Any]]:
    """Build Panel A: formal main results and mechanistic oracle."""
    # Group by suite, condition
    groups: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in episode_rows:
        if r.get("completed", "").lower() != "true":
            continue
        groups[r["suite"]][r["condition"]].append(r)

    panel_rows = []
    for suite in suites:
        for condition in conditions:
            eps = groups[suite].get(condition, [])
            n = len(eps)
            if n == 0:
                panel_rows.append({
                    "Suite": suite, "Condition": condition,
                    "Intervention": "", "Timing": "", "Eval": "ITT",
                    "Success": 0, "N": 0, "SR": "", "FR": "",
                    "CI_95_low": "", "CI_95_high": "", "Attack_Frames": "",
                    "Status": "NO_DATA",
                })
                continue

            successes = sum(1 for e in eps if e.get("task_success", "").lower() in ("true", "1"))
            sr = successes / n
            fr = 1.0 - sr
            ci_low, ci_high = wilson_ci(successes, n)
            attack_frames = sum(int(e.get("attack_frames", 0) or 0) for e in eps)
            trigger_rate = sum(1 for e in eps if e.get("detector_emitted", "").lower() in ("true", "1")) / n

            # Determine protocol description
            if condition == "CLEAN":
                intervention = "None (clean baseline)"
                timing = "n/a"
                eval_type = "ITT"
            elif condition == "TRUE_T10":
                intervention = "Force-Gripper-Open (Token-CE, ε=6/255, K=10)"
                timing = f"Detector trigger (C2e3 GRU)"
                eval_type = "ITT"
            elif condition == "RAND_T10":
                intervention = "Random-direction payload (K=10)"
                timing = "Same trigger as TRUE_T10"
                eval_type = "ITT"
            elif condition == "COMMAND_OPEN_ORACLE":
                intervention = "Command-Open Oracle (K=10)"
                timing = "Same trigger as TRUE_T10"
                eval_type = "ITT"

            status = "PASS" if condition == "CLEAN" and sr >= 0.80 else (
                "ORACLE" if condition == "COMMAND_OPEN_ORACLE" else "MAIN")

            panel_rows.append({
                "Suite": suite,
                "Condition": condition,
                "Intervention": intervention,
                "Timing": timing,
                "Eval": eval_type,
                "Success": successes,
                "N": n,
                "SR": f"{sr:.4f}",
                "FR": f"{fr:.4f}",
                "CI_95_low": f"{ci_low:.4f}",
                "CI_95_high": f"{ci_high:.4f}",
                "Attack_Frames": attack_frames,
                "Trigger_Rate": f"{trigger_rate:.4f}",
                "Status": status,
            })

    return panel_rows


def main():
    ap = argparse.ArgumentParser(description="D7 Table1 four-suite aggregator")
    ap.add_argument("--postrun-audit-csv", required=True)
    ap.add_argument("--postrun-audit-report", default="",
                    help="JSON audit report (for runtime_contract_status guard)")
    ap.add_argument("--queue-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Force aggregation even if audit contract FAIL (emergency only)")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv(args.postrun_audit_csv)
    queue_rows = read_csv(args.queue_manifest)

    # ── D7D contract guard: only aggregate if runtime contract PASS ──
    contract_pass = True
    if args.postrun_audit_report:
        audit_report = read_json(Path(args.postrun_audit_report))
        contract_status = audit_report.get("runtime_contract_status", "MISSING")
        d7d_blocked = audit_report.get("d7d_aggregation_blocked", False)
        if d7d_blocked or contract_status != "PASS":
            contract_pass = False
            if not args.force:
                print(f"FATAL: D7D aggregation blocked by audit contract status={contract_status}")
                print(f"  Reason: {audit_report.get('d7d_block_reason', 'runtime contract mismatch')}")
                print(f"  Use --force to override (emergency only, results will be marked QUARANTINED)")
                return 1
            else:
                print(f"WARNING: aggregation forced despite contract status={contract_status}")

    # Determine suites and conditions from manifest
    suites = sorted(set(r["suite"] for r in queue_rows))
    conditions = sorted(set(r["condition"] for r in queue_rows))

    print(f"D7 Aggregate: {len(audit_rows)} episodes, {len(suites)} suites, {len(conditions)} conditions")

    # Build Panel A
    panel_a = build_panel_a(audit_rows, conditions, suites)
    write_csv(out / "d7_table1_panel_a_formal_main_results.csv", panel_a,
              ["Suite", "Condition", "Intervention", "Timing", "Eval",
               "Success", "N", "SR", "FR", "CI_95_low", "CI_95_high",
               "Attack_Frames", "Trigger_Rate", "Status"])

    # Suite summary
    suite_summary = []
    for suite in suites:
        suite_eps = [r for r in audit_rows if r["suite"] == suite and r.get("completed","").lower()=="true"]
        clean_successes = sum(1 for e in suite_eps if e["condition"]=="CLEAN" and e.get("task_success","").lower() in ("true","1"))
        clean_n = sum(1 for e in suite_eps if e["condition"]=="CLEAN")
        suite_summary.append({
            "Suite": suite,
            "Total_Episodes": len(suite_eps),
            "Clean_N": clean_n,
            "Clean_SR": f"{clean_successes/max(1,clean_n):.4f}" if clean_n > 0 else "",
        })
    write_csv(out / "d7_table1_suite_summary.csv", suite_summary,
              ["Suite", "Total_Episodes", "Clean_N", "Clean_SR"])

    # Report
    report = {
        "gate": "D7_TABLE1_AGGREGATION",
        "status": "PASS_D7_AGGREGATION_BUILT" if contract_pass else "QUARANTINED_D7_AGGREGATION_CONTRACT_MISMATCH",
        "runtime_contract_pass": contract_pass,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "suites": suites,
        "conditions": conditions,
        "total_episodes": len(audit_rows),
        "recommendation": "render_panel_a_markdown" if contract_pass else "DO_NOT_RENDER_FIX_CONTRACT_FIRST",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    write_json(out / "d7_table1_aggregation_report.json", report)

    csums = {}
    for fn in sorted(out.glob("*")):
        if fn.is_file() and fn.name != "checksum_report.json":
            csums[fn.name] = sha256_file(fn)
    write_json(out / "checksum_report.json", csums)

    print(f"D7 Aggregate: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
