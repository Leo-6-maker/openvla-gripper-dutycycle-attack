#!/usr/bin/env python3
"""Create a non-destructive GitHub branch/PR hygiene audit from local git refs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

LOW_RISK_DELETE_CANDIDATES = {
    "denominator/full4-clean-20260519",
    "fix/protocol-schema-and-condition-config-20260523",
    "fix/table1-generic-autowindow-baseline-20260524",
    "eval/official-libero-clean-20260525",
    "merge/sc5-mainline-20260618",
}
KEEP_BRANCHES = {
    "main",
    "feature/sc5-cross-suite-generalization-20260619",
    "feature/sc5-video-export-20260618",
    "exp/l12-production-streaming-adapter-20260615",
}
OLD_DRAFT_PR_BRANCHES = {
    "exp/codex-autonomous-vis-crosssuite-20260531",
    "audit/m3-v2-seed81-trajectory-feasibility-20260615",
    "exp/m3-arm-constrained-logratio-v3-20260615",
    "exp/m3-arm-v3-fresh-seed82-canary-20260615",
    "exp/m3-arm-v4-hard-feasible-selection-20260615",
    "exp/m3-arm-v4-fixed-frame-panel-prereg-20260615",
}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT).decode("utf-8", errors="replace")


def remote_refs() -> list[str]:
    rows = []
    for line in run_git(["for-each-ref", "--format=%(refname:short),%(objectname),%(committerdate:iso8601)", "refs/remotes/origin"]).splitlines():
        parts = line.split(",", 2)
        if len(parts) == 3:
            ref, sha, date = parts
            if ref == "origin/HEAD":
                continue
            rows.append((ref.replace("origin/", "", 1), sha, date))
    return rows


def branch_rows() -> list[dict[str, Any]]:
    merged = {x.strip().replace("origin/", "", 1) for x in run_git(["branch", "-r", "--merged", "origin/main"]).splitlines()}
    no_merged = {x.strip().replace("origin/", "", 1) for x in run_git(["branch", "-r", "--no-merged", "origin/main"]).splitlines()}
    out = []
    for name, sha, date in remote_refs():
        if name in KEEP_BRANCHES:
            action = "KEEP"
            reason = "active_mainline_or_current_work"
        elif name in LOW_RISK_DELETE_CANDIDATES:
            action = "DELETE_CANDIDATE_AFTER_WORKTREE_CHECK"
            reason = "documented_merged_branch_low_risk"
        elif name in OLD_DRAFT_PR_BRANCHES:
            action = "CLOSE_OR_ARCHIVE_PR_BEFORE_DELETE"
            reason = "historical_draft_or_superseded_experiment"
        elif name in merged:
            action = "REVIEW_DELETE_CANDIDATE"
            reason = "merged_into_origin_main"
        elif name in no_merged:
            action = "KEEP_OR_ARCHIVE_REVIEW_REQUIRED"
            reason = "not_merged_into_origin_main"
        else:
            action = "REVIEW_REQUIRED"
            reason = "merge_status_unknown"
        out.append({
            "branch": name,
            "sha": sha,
            "committer_date": date,
            "merged_origin_main": name in merged,
            "no_merged_origin_main": name in no_merged,
            "recommended_action": action,
            "reason": reason,
        })
    return sorted(out, key=lambda r: (r["recommended_action"], r["branch"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["recommended_action"]] = counts.get(row["recommended_action"], 0) + 1
    low_risk = [r["branch"] for r in rows if r["recommended_action"] == "DELETE_CANDIDATE_AFTER_WORKTREE_CHECK"]
    old_drafts = [r["branch"] for r in rows if r["recommended_action"] == "CLOSE_OR_ARCHIVE_PR_BEFORE_DELETE"]
    lines = [
        "# GitHub Branch and PR Hygiene Audit",
        "",
        "This is non-destructive. It does not close PRs or delete remote branches.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(counts, indent=2, sort_keys=True),
        "```",
        "",
        "## First Low-Risk Delete Candidates",
        "",
        *[f"- `{b}`" for b in low_risk],
        "",
        "Connector spot-check note: if a historically suggested branch is absent from the current `origin` refs, do not include it in a deletion command unless a fresh remote ref check shows it exists.",
        "",
        "## Old Draft / Historical PR Branches To Archive Then Close",
        "",
        *[f"- `{b}`" for b in old_drafts],
        "",
        "## Guardrails",
        "",
        "- Check server worktrees before deleting any branch.",
        "- Use archive tags for historical M3 evidence before deleting old experiment heads.",
        "- Keep `feature/sc5-cross-suite-generalization-20260619` until CLEAN300 is fully audited.",
        "- Keep `feature/sc5-video-export-20260618` until video evidence is accepted.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = branch_rows()
    out = Path(args.output_dir)
    write_csv(out / "tables" / "github_branch_hygiene_20260619.csv", rows)
    write_report(out / "reports" / "GITHUB_BRANCH_PR_HYGIENE_20260619.md", rows)
    print(json.dumps({"result": "GITHUB_HYGIENE_AUDIT_DONE", "branches": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
