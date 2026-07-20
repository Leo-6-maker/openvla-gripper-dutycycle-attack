#!/usr/bin/env python3
"""Gate E-R2: Read-only audit of existing task00-03 episode roots.

Does NOT: rerun any episode, load OpenVLA, load LIBERO, modify files.
Reads existing rooted output and reports per-task telemetry.

Run on server: python3 audit_r10_4e_roots.py
"""

import json, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# ── Known output roots ─────────────────────────────────────────────────────────

R10_4D_SMOKE = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_passive_smoke_output_20260720")
R10_4E_PANEL = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_passive_panel_20260720")
BUNDLE_ROOT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_3_full_fit_deployment_bundle_1353e3b4_20260720")
WORKTREE = Path("/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847")
RECEIPT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_auth_receipt_20260720")

TASKS = {
    "libero_10/task_00/state_20": {
        "root": R10_4D_SMOKE,
        "source": "R10_4D single-episode passive smoke",
        "note": "Original R10.4D smoke — only task with preserved per-episode root",
    },
    "libero_10/task_01/state_20": {
        "root": None,
        "source": "R10_4E panel (first run)",
        "note": "Failed at authorization gate — no episode data generated",
    },
    "libero_10/task_02/state_20": {
        "root": None,
        "source": "R10_4E panel — SKIPPED_EARLY_TERMINATION",
        "note": "Skipped in panel — no episode data",
    },
    "libero_10/task_03/state_20": {
        "root": None,
        "source": "R10_4E panel — SKIPPED_EARLY_TERMINATION",
        "note": "Skipped in panel — no episode data",
    },
}


def compute_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_task_00() -> dict:
    """Full read-only audit of task_00 from R10.4D smoke output."""
    root = R10_4D_SMOKE
    report = {"identity": "libero_10/task_00/state_20", "source": "R10_4D"}

    # 1. SHA256SUMS integrity
    sums_file = root / "SHA256SUMS"
    sums_sha_file = root / "SHA256SUMS.sha256"
    if sums_file.is_file() and sums_sha_file.is_file():
        expected_digest = sums_sha_file.read_text().strip().split()[0]
        actual_digest = compute_sha256(sums_file)
        report["sha256sums_verified"] = actual_digest == expected_digest
        report["sha256sums_digest"] = actual_digest[:16] + "..."
    else:
        report["sha256sums_verified"] = False
        report["sha256sums_digest"] = "MISSING"

    # 2. Episode summary
    summary_file = root / "episode_summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text())
        report["status"] = summary.get("status", "?")
        report["n_steps"] = summary.get("n_steps", -1)
        report["emit_count"] = summary.get("emit_count", -1)
        report["task_success"] = summary.get("task_success", None)
        report["violations"] = summary.get("violations", [])
        report["termination_reason"] = summary.get("termination_reason", "NOT_PRESENT_IN_ORIGINAL")
    else:
        report["summary_missing"] = True

    # 3. Episode metadata
    meta_file = root / "episode_metadata.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        report["identity_bound"] = meta.get("identity", "?")
        report["parent_bound"] = meta.get("parent", "?")
    else:
        report["metadata_missing"] = True

    # 4. Step records
    steps_file = root / "step_records.jsonl"
    if steps_file.is_file():
        steps = [json.loads(l) for l in steps_file.read_text().splitlines() if l.strip()]
        report["policy_steps"] = len(steps)
        if steps:
            last = steps[-1]
            report["done_step"] = last["step"] if last.get("done") else "not_done"
            report["last_done"] = last.get("done", None)
            report["last_reward"] = last.get("reward", None)
            report["last_info"] = last.get("info", "NOT_PRESENT_IN_ORIGINAL")
        # Generation passes
        gen_passes = [s.get("generation_passes_per_step") for s in steps]
        report["gen_passes_all_one"] = all(g == 1 for g in gen_passes)
        report["gen_passes_unique"] = list(set(gen_passes))
        # Action errors
        max_err = max(s.get("action_max_abs_error", -1) for s in steps)
        report["action_max_error"] = max_err
        # Feature validity
        n_valid_features = sum(1 for s in steps if "features_25d" in s and len(s.get("features_25d", [])) == 25)
        report["feature_valid_steps"] = n_valid_features
    else:
        report["step_records_missing"] = True
        report["policy_steps"] = 0

    # 5. Detector records
    det_file = root / "detector_records.jsonl"
    if det_file.is_file():
        dets = [json.loads(l) for l in det_file.read_text().splitlines() if l.strip()]
        report["detector_record_count"] = len(dets)
        report["fsm_states"] = list(set(d.get("fsm_state", "?") for d in dets))
        report["detector_emits"] = sum(1 for d in dets if d.get("emit"))
        report["n_events"] = max((d.get("event_id", 0) for d in dets), default=0)
    else:
        report["detector_records_missing"] = True

    # 6. Runtime audit
    audit_file = root / "runtime_audit.json"
    if audit_file.is_file():
        audit = json.loads(audit_file.read_text())
        report["runtime_audit_pass"] = audit.get("overall", "?")
    else:
        report["runtime_audit_missing"] = True

    # 7. Root seal
    seal_file = root / "ROOT_SEAL_RECEIPT.json"
    if seal_file.is_file():
        seal = json.loads(seal_file.read_text())
        report["root_seal_present"] = True
    else:
        report["root_seal_present"] = False

    # 8. Receipt binding
    if RECEIPT.is_file():
        receipt = json.loads(RECEIPT.read_text())
        report["receipt_parent"] = receipt.get("selected_parent", "?")
        report["receipt_scope"] = receipt.get("scope", "?")
        report["receipt_passive_only"] = receipt.get("passive_only", None)
        report["receipt_action_mutation"] = receipt.get("action_mutation_authorized", None)
    else:
        report["receipt_missing"] = True

    # 9. Worktree
    import subprocess
    try:
        head = subprocess.check_output(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True).strip()
        report["worktree_head"] = head[:16] + "..."
        status = subprocess.check_output(["git", "-C", str(WORKTREE), "status", "--porcelain"], text=True)
        report["worktree_clean"] = len(status.strip()) == 0
    except Exception:
        report["worktree_head"] = "ERROR"

    # 10. Bundle binding
    ckpt = BUNDLE_ROOT / "full_fit_deploy.pt"
    if ckpt.is_file():
        report["bundle_checkpoint_sha"] = compute_sha256(ckpt)[:16] + "..."

    # 11. Classification under E-R1 rules
    report["er1_termination_classification"] = classify_from_existing(report)

    return report


def classify_from_existing(report: dict) -> str:
    """Apply E-R1 classification rules to existing recorded data."""
    n_steps = report.get("policy_steps", 0)
    last_done = report.get("last_done", None)
    task_success = report.get("task_success", None)

    if n_steps == 0:
        return "NO_STEPS - cannot classify"
    if last_done and task_success is True:
        return "SUCCESS_TERMINATION"
    elif last_done and n_steps >= 520 and task_success is not True:
        return "HORIZON_TERMINATION"
    elif not last_done and n_steps >= 520:
        return "FULL_LOOP_TASK_FAILURE"
    elif last_done and n_steps < 520 and task_success is not True:
        return "EARLY_DONE_WITHOUT_SUCCESS (HARD FAILURE)"
    else:
        return f"UNCLASSIFIED: done={last_done} steps={n_steps} success={task_success}"


def audit_task_missing(identity: str, note: str) -> dict:
    return {
        "identity": identity,
        "evidence_status": "EVIDENCE_NOT_FOUND",
        "note": note,
        "policy_steps": 0,
        "n_steps_available": 0,
        "termination_classifiable": False,
    }


def main():
    print("=" * 60)
    print("Gate E-R2: Read-Only Root Audit (task_00-03)")
    print("=" * 60)
    print()

    # ── Task 00: Full audit ──
    print("─── task_00/state_20 ───")
    r00 = audit_task_00()
    for k, v in sorted(r00.items()):
        print(f"  {k}: {v}")
    print()

    # ── Task 01-03: Missing evidence ──
    for identity, info in TASKS.items():
        if identity == "libero_10/task_00/state_20":
            continue
        print(f"─── {identity.split('/')[1]}/{identity.split('/')[2]} ───")
        r = audit_task_missing(identity, info["note"])
        for k, v in sorted(r.items()):
            print(f"  {k}: {v}")
        print()

    # ── Aggregated gate status ──
    print("=" * 60)
    print("Gate E-R2 Summary")

    findings = []
    if r00.get("policy_steps", 0) > 0:
        findings.append(f"task_00: {r00['er1_termination_classification']}")
    else:
        findings.append("task_00: NO DATA")

    findings.append("task_01: EVIDENCE_NOT_FOUND — no episode data (auth gate failure)")
    findings.append("task_02: EVIDENCE_NOT_FOUND — skipped in panel")
    findings.append("task_03: EVIDENCE_NOT_FOUND — skipped in panel")

    for f in findings:
        print(f"  {f}")

    print()
    print("Conclusion: Only task_00 has preserved per-episode evidence.")
    print("task_01-03 require remediation re-runs (Gate E-R3) to produce auditable roots.")
    print("Original failures must be preserved as ORIGINAL_PROTOCOL_FAILURE, not overwritten.")

    # Write report
    report = {
        "audit": "R10_4E_GATE_E_R2_ROOT_AUDIT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_00": {k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                    for k, v in r00.items()},
        "task_01": {"evidence_status": "EVIDENCE_NOT_FOUND"},
        "task_02": {"evidence_status": "EVIDENCE_NOT_FOUND"},
        "task_03": {"evidence_status": "EVIDENCE_NOT_FOUND"},
        "conclusion": "Only task_00 has auditable per-episode root. task_01-03 require remediation.",
    }
    out_path = Path("/tmp/r10_4e_gate_e_r2_root_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
