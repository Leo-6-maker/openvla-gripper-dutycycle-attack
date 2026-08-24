"""Aggregate the immutable T1-A clean-only token/score-path receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"head": run("rev-parse", "HEAD"), "tree": run("rev-parse", "HEAD^{tree}"), "status": run("status", "--porcelain")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--score-root", dest="score_root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_pre_pgd_20260818_v2"))
    parser.add_argument("--native-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_detector_authority_20260818T034500Z"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1_PROSPECTIVE_DETECTOR_PGD_PROTOCOL_V1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    source = git()
    if source["status"]:
        errors.append("WORKTREE_NOT_CLEAN")
    protocol = load(args.protocol)
    if protocol.get("iterate_selection", {}).get("rule") != "final_iterate_only":
        errors.append("ITERATE_RULE_NOT_FROZEN")

    natives = sorted(args.native_root.glob("NATIVE_ACTION_TOKEN_AUTHORITY_AUDIT_V1.json"))
    if len(natives) != 1:
        errors.append("NATIVE_RECEIPT_COUNT_NOT_ONE")
        native = None
    else:
        native = load(natives[0])
        if native.get("status") != "PASS_V2_NATIVE_PARITY_LEGACY_HELPER_DIAGNOSTIC_MISMATCH":
            errors.append("NATIVE_RECEIPT_NOT_PASS")
        if any(int(v) != 0 for v in native.get("protected_counters", {}).values()) or native.get("eval160") != "UNREAD" or native.get("protected_evaluation") != "UNREAD":
            errors.append("NATIVE_PROTECTED_BOUNDARY_FAIL")
        for suite in SUITES:
            item = native.get("suites", {}).get(suite, {})
            if item.get("v2_native_mismatch_count") != 0:
                errors.append(f"NATIVE_MISMATCH:{suite}")
            if item.get("binding", {}).get("legacy_helper_status") != "HISTORICAL_COMPATIBILITY_ONLY":
                errors.append(f"LEGACY_HELPER_NOT_DIAGNOSTIC_ONLY:{suite}")

    reports = []
    for path in sorted(args.score_root.glob("SCORE_PATH_*_V2.json")):
        report = load(path)
        if report.get("suite") in SUITES:
            reports.append((path, report))
    counts = Counter(report.get("suite") for _, report in reports)
    if len(reports) != 8 or any(counts[suite] != 2 for suite in SUITES):
        errors.append("SCORE_REPORT_FLEET_NOT_4X2")

    report_receipts: list[dict[str, Any]] = []
    for path, report in reports:
        suite = report["suite"]
        if report.get("status") != "DIAGNOSTIC_CLEAN_ONLY":
            errors.append(f"SCORE_STATUS:{path.name}")
        counters = report.get("counters", {})
        if any(int(v) != 0 for v in counters.values()) or report.get("eval160") != "UNREAD" or report.get("protected_evaluation") != "UNREAD":
            errors.append(f"SCORE_PROTECTED_BOUNDARY:{path.name}")
        if report.get("processor_parity") != {"attention_mask_exact": True, "input_ids_exact": True, "pixel_values_exact": True}:
            errors.append(f"PROCESSOR_PARITY:{path.name}")
        authority = report.get("authority", {})
        if authority.get("authority_version") != "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_V2" or authority.get("legacy_helper_status") != "HISTORICAL_COMPATIBILITY_ONLY":
            errors.append(f"TOKEN_AUTHORITY:{path.name}")
        cached_failures = []
        for row in report.get("row_comparisons", []):
            cached = row.get("official_vs_cached", {})
            if cached.get("max_abs_logit_diff") != 0.0 or cached.get("top1_exact") is not True or cached.get("top2_exact") is not True:
                cached_failures.append(row.get("dim"))
        if cached_failures:
            errors.append(f"CACHED_ROUTE_NOT_EXACT:{path.name}:{cached_failures}")
        target_grad = report.get("target_token", {}).get("gradient", {})
        arm_grad = report.get("arm_preservation", {}).get("gradient", {})
        for name, grad in (("target", target_grad), ("arm", arm_grad)):
            if not isinstance(grad.get("cosine"), (int, float)) or not isinstance(grad.get("sign_agreement_fraction"), (int, float)):
                errors.append(f"{name.upper()}_GRADIENT_MISSING:{path.name}")
        no_cache_disagreements = sum(
            row.get("official_vs_nocache", {}).get("top1_exact") is not True
            or row.get("official_vs_nocache", {}).get("top2_exact") is not True
            for row in report.get("row_comparisons", [])
        )
        report_receipts.append({
            "path": str(path), "sha256": sha256(path), "suite": suite,
            "source": report.get("source"), "runtime": report.get("runtime"),
            "cached_route_exact": not cached_failures,
            "no_cache_top1_or_top2_disagreement_rows": no_cache_disagreements,
            "target_gradient": target_grad, "arm_gradient": arm_grad,
        })

    receipt = {
        "schema": "STAGE_X_X1R_T1_SCORE_PATH_AUTHORITY_RECEIPT_V1",
        "status": "PASS_T1_A_CLEAN_ONLY_SCORE_AUTHORITY" if not errors else "HOLD_T1_A_SCORE_PATH_AUTHORITY",
        "source": source,
        "score_root": str(args.score_root), "native_root": str(args.native_root),
        "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol), "iterate_rule": protocol.get("iterate_selection", {}).get("rule")},
        "native_receipt": {"path": str(natives[0]) if natives else None, "sha256": sha256(natives[0]) if native else None, "status": native.get("status") if native else None},
        "score_reports": report_receipts,
        "route_decision": "manual_cached_autoregressive_path_bound; historical_no_cache_teacher_forced_path_diagnostic_only",
        "scientific_boundary": "clean-only; no PGD, attacked image/action, env.step, physical intervention, V_phys, Eval160, or protected read",
        "protected_counters": {"pgd_calls": 0, "env_step_calls": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "vphys_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "eval160": "UNREAD", "protected_evaluation": "UNREAD", "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
