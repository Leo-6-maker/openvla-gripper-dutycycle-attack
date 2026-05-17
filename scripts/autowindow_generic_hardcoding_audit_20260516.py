#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any


FORBIDDEN_TEXT = [
    "BLACK_BOWL_FIXED_WINDOWS",
    "reference_windows_blackbowl_eval_only",
    "evaluate_blackbowl_reference_window_recovery",
    "detect_black_bowl_window",
    "blackbowl_calibrated_clean_proxy",
    "libero_spatial_black_bowl",
    "78, 87",
    "75, 84",
    "oracle_gripper_only",
    "vis_arm_clean",
    "random_gripper_clean",
    "manual",
    "failure_phase",
    "official_failure",
]
FORBIDDEN_SUBSTRINGS = ["black", "bowl", "moka", "drawer", "spatial", "libero10"]
METADATA_NAMES = {"task_id", "state", "state_id", "suite", "task_name", "seed"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "pattern", "line", "evidence", "finding"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def constants_in(node: ast.AST) -> list[str]:
    return [str(child.value) for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def line_for(lines: list[str], node: ast.AST) -> str:
    lineno = getattr(node, "lineno", 0)
    if lineno and 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def audit_detector(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []

    for pattern in FORBIDDEN_TEXT:
        for line_no, line in enumerate(lines, 1):
            if pattern in line:
                category = "attack_outcome_leakage" if pattern in {"oracle_gripper_only", "vis_arm_clean", "random_gripper_clean", "manual", "failure_phase", "official_failure"} else "state_or_task_hardcoding"
                findings.append(
                    {
                        "category": category,
                        "pattern": pattern,
                        "line": line_no,
                        "evidence": line.strip(),
                        "finding": "forbidden literal/reference in generic detector",
                    }
                )

    for line_no, line in enumerate(lines, 1):
        lowered = line.lower()
        for token in FORBIDDEN_SUBSTRINGS:
            if token in lowered:
                findings.append(
                    {
                        "category": "task_specific_substring",
                        "pattern": token,
                        "line": line_no,
                        "evidence": line.strip(),
                        "finding": "task/suite-specific substring in generic detector",
                    }
                )

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.comprehension)):
            test = node.test if not isinstance(node, ast.comprehension) else node.ifs[0] if node.ifs else None
            if test is None:
                continue
            used = names_in(test)
            literals = constants_in(test)
            if used & METADATA_NAMES:
                findings.append(
                    {
                        "category": "metadata_branch",
                        "pattern": ",".join(sorted(used & METADATA_NAMES)),
                        "line": getattr(node, "lineno", ""),
                        "evidence": line_for(lines, node),
                        "finding": "generic detector branches on metadata",
                    }
                )
            for literal in literals:
                lowered = literal.lower()
                if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
                    findings.append(
                        {
                            "category": "task_specific_branch_literal",
                            "pattern": literal,
                            "line": getattr(node, "lineno", ""),
                            "evidence": line_for(lines, node),
                            "finding": "branch contains task/suite-specific literal",
                        }
                    )

    leakage = any(row["category"] == "attack_outcome_leakage" for row in findings)
    hardcoded = any(row["category"] in {"state_or_task_hardcoding", "task_specific_substring", "metadata_branch", "task_specific_branch_literal"} for row in findings)
    if leakage:
        decision = "D. attack_outcome_leakage_detected"
    elif hardcoded:
        decision = "C. state_specific_hardcoding_detected"
    else:
        decision = "A. no_state_or_task_hardcoding_detected"
    return findings, decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the generic detector only.")
    parser.add_argument("--detector", default="scripts/detect_contact_window_from_clean.py")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    detector = Path(args.detector)
    findings, decision = audit_detector(detector)
    write_csv(output_root / "tables/autowindow_generic_hardcoding_audit_20260516.csv", findings)
    state_task = sum(1 for row in findings if row["category"] != "attack_outcome_leakage")
    leakage = sum(1 for row in findings if row["category"] == "attack_outcome_leakage")
    lines = [
        "# Generic Auto-Window Hardcoding Audit 20260516",
        "",
        f"Decision: {decision}",
        "",
        "Scope: generic detector only. Reference evaluator/config are intentionally excluded.",
        "",
        f"- detector: `{detector}`",
        f"- state/task/reference findings: {state_task}",
        f"- attack-outcome leakage findings: {leakage}",
        "",
        "## Findings",
    ]
    if findings:
        for row in findings[:80]:
            lines.append(f"- line {row['line']}: `{row['evidence']}` ({row['finding']})")
    else:
        lines.append("- none")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "autowindow_generic_hardcoding_audit_summary_20260516.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
