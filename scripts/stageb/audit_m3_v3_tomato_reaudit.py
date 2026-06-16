#!/usr/bin/env python3
"""Independent forensic re-audit for the GPU15 Tomato S3 artifacts.

This script is read-only with respect to the original output root. It does not
rewrite the original watcher gate. It writes separate audit tables/reports whose
status is explicitly named S3_INDEPENDENT_REAUDIT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
    PANEL_FROZEN_OBJECTIVE,
    audit_candidate_artifacts,
    audit_route_artifacts,
)
from gripper_attack.m3_telemetry_schema import read_required_int


TARGET_TOKEN = 31744
CONDITIONS = (
    "TRUE_PGD_TRAJECTORY21_SELECTIVE",
    "RAND21_SELECTIVE",
    "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def rows_by_condition(rows: list[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("condition", "")): dict(row) for row in rows}


def artifact_hash_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def audit_lambda_dir(lambda_value: str, canary_dir: Path, *, expected_seed: int, expected_commit: str, epsilon: float) -> dict[str, Any]:
    selected_path = canary_dir / "m3_v4_selected_results.csv"
    candidate_path = canary_dir / "m3_v4_candidate_audit.csv"
    route_path = canary_dir / "m3_v4_route_audit.csv"
    artifact_manifest = canary_dir / "m3_artifact_hash_manifest.csv"
    debug_path = canary_dir / "m3_v4_debug.json"
    missing = [str(p.name) for p in [selected_path, candidate_path, route_path, artifact_manifest, debug_path] if not p.exists()]
    if missing:
        return {"lambda": lambda_value, "reaudit_status": "INFRA_INVALID", "reason": "missing:" + ",".join(missing)}

    selected_rows = rows_by_condition(read_csv(selected_path))
    candidate_rows = read_csv(candidate_path)
    route_ok, route_reason = audit_route_artifacts(
        route_path,
        expected_seed=expected_seed,
        expected_commit=expected_commit,
        expected_objective=PANEL_FROZEN_OBJECTIVE,
    )
    reasons: list[str] = []
    if not route_ok:
        reasons.append(route_reason)

    condition_details: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = selected_rows.get(condition)
        ok, reason = audit_candidate_artifacts(
            candidate_rows=candidate_rows,
            selected_row=selected,
            condition=condition,
            expected_seed=expected_seed,
            expected_commit=expected_commit,
            epsilon=epsilon,
        )
        if not ok:
            reasons.append(reason)
        if selected is None:
            reasons.append(f"{condition}:selected_row_missing")
            continue
        arm = read_required_int(
            selected,
            canonical="arm_prefix_match_count",
            legacy_aliases=["official_arm_match_count", "arm_match_count"],
        )
        token = int(selected.get("official_gripper_token", -1) or -1)
        margin = float(selected.get("official_target31744_margin", "-inf"))
        linf = float(selected.get("processor_linf", "inf"))
        invariant = str(selected.get("score_invariant_status", ""))
        condition_details[condition] = {
            "arm": arm,
            "token": token,
            "margin": margin,
            "linf": linf,
            "score_invariant": invariant,
            "selected_candidate_id": selected.get("selected_candidate_id", ""),
        }
        if token != TARGET_TOKEN:
            reasons.append(f"{condition}:token_not_31744")
        if arm < 5:
            reasons.append(f"{condition}:arm_below_gate")
        if invariant != "PASS":
            reasons.append(f"{condition}:score_invariant_not_pass")
        if linf > epsilon + 1e-9:
            reasons.append(f"{condition}:linf_over_budget")

    true = condition_details.get("TRUE_PGD_TRAJECTORY21_SELECTIVE", {})
    rand = condition_details.get("RAND21_SELECTIVE", {})
    shuffled = condition_details.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", {})
    if true and rand and not (float(true["margin"]) > float(rand["margin"])):
        reasons.append("TRUE_not_greater_than_RAND21")
    if true and shuffled and not (float(true["margin"]) > float(shuffled["margin"])):
        reasons.append("TRUE_not_greater_than_SHUFFLED")

    status = "PASS" if not reasons else "INFRA_INVALID"
    return {
        "lambda": lambda_value,
        "reaudit_status": status,
        "reason": ";".join(reasons),
        "true_token": true.get("token", ""),
        "true_arm_match": true.get("arm", ""),
        "true_margin": true.get("margin", ""),
        "rand_margin": rand.get("margin", ""),
        "shuffled_margin": shuffled.get("margin", ""),
        "true_minus_rand": "" if not (true and rand) else float(true["margin"]) - float(rand["margin"]),
        "true_minus_shuffled": "" if not (true and shuffled) else float(true["margin"]) - float(shuffled["margin"]),
        "true_linf": true.get("linf", ""),
        "strict_route": route_ok,
        "fallback": False if route_ok else "",
        "canary_dir": str(canary_dir),
    }


def run_reaudit(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    out_dir = Path(args.audit_output_dir)
    s3 = root / "S3_TOMATO_SCREEN"
    original_gate_path = s3 / "gate_result.json"
    original_gate = json.loads(original_gate_path.read_text(encoding="utf-8"))
    lambda_dirs = sorted(p for p in s3.glob("lambda_*") if (p / "canary").exists())
    rows = [
        audit_lambda_dir(
            p.name.replace("lambda_", ""),
            p / "canary",
            expected_seed=int(args.expected_seed),
            expected_commit=str(args.expected_commit),
            epsilon=float(args.epsilon),
        )
        for p in lambda_dirs
    ]
    pass_rows = [row for row in rows if row["reaudit_status"] == "PASS"]
    overall = "PASS" if pass_rows else "FAIL"
    if pass_rows:
        selected = sorted(
            pass_rows,
            key=lambda row: (-int(row["true_arm_match"]), -float(row["true_minus_rand"]), float(row["lambda"])),
        )[0]
    else:
        selected = {}
    write_csv(
        out_dir / "tables" / "m3_v3_tomato_independent_reaudit.csv",
        rows,
        [
            "lambda",
            "reaudit_status",
            "reason",
            "true_token",
            "true_arm_match",
            "true_margin",
            "rand_margin",
            "shuffled_margin",
            "true_minus_rand",
            "true_minus_shuffled",
            "true_linf",
            "strict_route",
            "fallback",
            "canary_dir",
        ],
    )
    hash_rows = artifact_hash_rows(root)
    write_csv(out_dir / "tables" / "artifact_hashes_r3.csv", hash_rows, ["path", "relative_path", "size_bytes", "sha256"])
    summary = {
        "S3_ORIGINAL_GATE": original_gate.get("status"),
        "S3_ORIGINAL_FAILURE_CLASS": original_gate.get("failure_class", ""),
        "S3_INDEPENDENT_REAUDIT": overall,
        "selected_lambda": selected.get("lambda", ""),
        "selected": selected,
        "audit_rows": rows,
    }
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "M3_V3_TOMATO_TELEMETRY_REAUDIT.md").write_text(
        "# M3 V3 Tomato Telemetry Re-Audit\n\n"
        f"- S3_ORIGINAL_GATE: {summary['S3_ORIGINAL_GATE']}\n"
        f"- S3_ORIGINAL_FAILURE_CLASS: {summary['S3_ORIGINAL_FAILURE_CLASS']}\n"
        f"- S3_INDEPENDENT_REAUDIT: {overall}\n"
        f"- selected_lambda: {summary['selected_lambda']}\n\n"
        "The original `gate_result.json` was not modified. This audit only reads\n"
        "the immutable r3 artifacts and writes separate forensic outputs.\n",
        encoding="utf-8",
    )
    (out_dir / "S3_independent_reaudit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--audit_output_dir", required=True)
    ap.add_argument("--expected_seed", type=int, default=81)
    ap.add_argument("--expected_commit", default="")
    ap.add_argument("--epsilon", type=float, default=6.0 / 255.0)
    args = ap.parse_args()
    run_reaudit(args)


if __name__ == "__main__":
    main()
