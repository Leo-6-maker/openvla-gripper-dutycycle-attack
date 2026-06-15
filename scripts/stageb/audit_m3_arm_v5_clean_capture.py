#!/usr/bin/env python3
"""Independent auditor for M3 arm-v5 clean capture artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.stageb.run_m3_arm_v5_clean_capture import (  # noqa: E402
    V5_PANEL_SIZE,
    git_value,
    load_attempt_ledger,
    load_config,
    select_events_from_clean_record_dir,
    validate_attempt_ledger_policy,
    validate_frozen_pool_sources,
    verify_model_bundle_manifest,
    verify_selected_rows_exact_bindings,
    write_json,
)


def audit_capture_root(*, capture_root: Path, config_path: Path, expected_commit: str = "") -> dict[str, object]:
    cfg = load_config(config_path)
    pool = validate_frozen_pool_sources(cfg, config_path=config_path)
    attempt_path = capture_root / "m3_arm_v5_capture_attempt_ledger.csv"
    attempt_rows = load_attempt_ledger(attempt_path)
    validate_attempt_ledger_policy(attempt_rows, pool=pool, clean_records_dir=capture_root)
    model_bundle_sha = verify_model_bundle_manifest(
        capture_root / "m3_arm_v5_model_bundle_manifest.csv",
        str(cfg["model"]["path"]),
    )
    rows, selected, status = select_events_from_clean_record_dir(
        cfg=cfg,
        clean_records_dir=capture_root,
        attempt_rows=attempt_rows,
    )
    selected_keys = {(event.task, event.state_id, event.step) for event in selected}
    selected_rows = [
        row
        for row in rows
        if (row["task"], int(row["state_id"]), int(row["selected_step"] or -1)) in selected_keys
    ]
    binding_ok, binding_reason = verify_selected_rows_exact_bindings(
        selected_rows,
        capture_root=capture_root,
        expected_commit=expected_commit or git_value(["rev-parse", "HEAD"]),
        expected_model_bundle_sha=model_bundle_sha,
    )
    audit_status = "PASS"
    failure_reason = ""
    if status != "V5_EVENT_PANEL_INPUTS_FROZEN":
        audit_status = "FAIL"
        failure_reason = status
    elif len(selected_rows) != V5_PANEL_SIZE:
        audit_status = "FAIL"
        failure_reason = f"selected row count mismatch: {len(selected_rows)}"
    elif not binding_ok:
        audit_status = "FAIL"
        failure_reason = binding_reason
    return {
        "audit_status": audit_status,
        "failure_reason": failure_reason,
        "capture_root": str(capture_root),
        "config_path": str(config_path),
        "attempt_rows": len(attempt_rows),
        "pool_size": len(pool),
        "selection_status": status,
        "selected_count": len(selected_rows),
        "model_bundle_sha256": model_bundle_sha,
        "expected_commit": expected_commit or git_value(["rev-parse", "HEAD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_root", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_arm_v5_clean_close_event_panel.yaml"))
    ap.add_argument("--expected_commit", default="")
    ap.add_argument("--audit_output", default="")
    args = ap.parse_args()
    out = audit_capture_root(
        capture_root=Path(args.capture_root),
        config_path=Path(args.config),
        expected_commit=str(args.expected_commit or ""),
    )
    output_path = Path(args.audit_output) if args.audit_output else Path(args.capture_root) / "m3_arm_v5_clean_capture_audit.json"
    write_json(output_path, out)
    if out["audit_status"] != "PASS":
        print(json.dumps(out, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
