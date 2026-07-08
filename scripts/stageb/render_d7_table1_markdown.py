#!/usr/bin/env python3
"""D7 Table1 markdown renderer.

Reads Panel A aggregation CSV and renders a publication-style markdown table.
Layout follows the Object Table 1 PDF format:
  Panel A: Formal main results and mechanistic oracle
  Suite → Condition → Intervention/Timing/Eval → Success/N → SR → FR → CI → Attack Frames
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


SUITE_NAMES = {
    "libero_10": "LIBERO-10",
    "libero_goal": "LIBERO-Goal",
    "libero_object": "LIBERO-Object",
    "libero_spatial": "LIBERO-Spatial",
}

CONDITION_NAMES = {
    "CLEAN": "Clean",
    "TRUE_T10": "TRUE-T10 (C2e3 GRU)",
    "RAND_T10": "RAND-T10",
    "COMMAND_OPEN_ORACLE": "Command-Open Oracle",
}


def render_markdown(panel_a: List[Dict[str, str]]) -> str:
    """Render Panel A as a grouped markdown table."""
    lines = []
    lines.append("# Table 1. OpenVLA-7B on Four LIBERO Suites: Gripper Duty-Cycle Attack Results")
    lines.append("")
    lines.append("## Panel A — Formal Main Results and Mechanistic Oracle")
    lines.append("")

    # Group by suite
    by_suite: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in panel_a:
        by_suite[r["Suite"]].append(r)

    for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        rows = by_suite.get(suite, [])
        if not rows:
            continue

        suite_name = SUITE_NAMES.get(suite, suite)
        lines.append(f"### {suite_name} (N={sum(int(r.get('N',0) or 0) for r in rows)} episodes)")
        lines.append("")
        lines.append("| Condition | Intervention | Timing | Eval | Success/N | SR | FR | 95% CI | Attack Frames | Trigger Rate | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for r in rows:
            cond = CONDITION_NAMES.get(r["Condition"], r["Condition"])
            sr = r.get("SR", "")
            fr = r.get("FR", "")
            ci_low = r.get("CI_95_low", "")
            ci_high = r.get("CI_95_high", "")
            ci = f"[{ci_low}, {ci_high}]" if ci_low and ci_high else ""
            lines.append(
                f"| {cond} | {r.get('Intervention','')} | {r.get('Timing','')} | {r.get('Eval','')} | "
                f"{r.get('Success','')}/{r.get('N','')} | {sr} | {fr} | {ci} | "
                f"{r.get('Attack_Frames','')} | {r.get('Trigger_Rate','')} | {r.get('Status','')} |"
            )
        lines.append("")

    # Advisor interpretation
    lines.append("## Advisor-Ready Interpretation")
    lines.append("")
    lines.append("1. **Direction Specificity**: RAND-T10 with random-direction payload controls for perturbation direction. "
                 "If TRUE-T10 SR << RAND-T10 SR, the attack is direction-specific (gripper-open).")
    lines.append("2. **Timing Specificity**: COMMAND_OPEN_ORACLE uses same timing as TRUE-T10. "
                 "If ORACLE SR << TRUE-T10 SR, the attack's weakness is detection timing, not the intervention itself.")
    lines.append("3. **Mechanism Oracle**: COMMAND_OPEN_ORACLE provides the upper-bound failure rate if the detector "
                 "were perfect at identifying attackable moments.")
    lines.append("4. **LIBERO-10 Limitation**: C2e3 GRU detector has L10 recall ~46%, which limits attack effectiveness "
                 "on LIBERO-10. This is a known detector limitation documented in C2e3 package.")
    lines.append("5. **C2f Blocked**: Observation-enhanced detectors (C2f) cannot be evaluated because current clean "
                 "rollout artifacts lack RGB frames and task language. This is documented as a limitation.")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="D7 Table1 markdown renderer")
    ap.add_argument("--panel-a-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel_a = read_csv(args.panel_a_csv)
    print(f"D7 Render: {len(panel_a)} rows")

    md = render_markdown(panel_a)
    md_path = out / "d7_table1_main_results.md"
    md_path.write_text(md, encoding="utf-8")

    # Also save as plain text for easy viewing
    txt_path = out / "d7_table1_main_results.txt"
    txt_path.write_text(md, encoding="utf-8")

    report = {
        "gate": "D7_TABLE1_RENDERER",
        "status": "PASS_D7_RENDERER_BUILT",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "output_md": str(md_path),
        "output_txt": str(txt_path),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    write_json(out / "d7_table1_renderer_report.json", report)

    print(f"D7 Render: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
