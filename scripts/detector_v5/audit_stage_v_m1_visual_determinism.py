#!/usr/bin/env python3
"""Independent CPU audit for M1 repeatability and raw sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from analyze_stage_v_m1_visual_divergence import _load, _sha256, write_r1

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PREFIX = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800"


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _verify_raw_manifest(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    bad: list[str] = []
    entries = list(manifest.get("entries", []))
    for entry in entries:
        binary = run_root / "trace" / "raw_capture" / str(entry["binary_path"]).replace("\\", "/")
        descriptor = run_root / "trace" / "raw_capture" / str(entry["descriptor_path"]).replace("\\", "/")
        if not binary.is_file() or not descriptor.is_file():
            bad.append(str(entry.get("binary_path")))
            continue
        actual = hashlib.sha256(binary.read_bytes()).hexdigest()
        if actual != entry.get("raw_sha256"):
            bad.append(str(entry.get("binary_path")))
        descriptor_value = _load(descriptor)
        if descriptor_value.get("raw_sha256") != entry.get("raw_sha256") or descriptor_value.get("byte_length") != binary.stat().st_size:
            bad.append(str(entry.get("descriptor_path")))
    return {"entry_count": len(entries), "bad_entries": bad, "verdict": "PASS" if not bad else "FAIL"}


def _audit_receipts(root: Path, identity: str) -> dict[str, Any]:
    identity_root = root / "runs" / identity.replace("/", "__")
    runs = {
        "Q1": identity_root / "CLEAN_QUALIFICATION" / "rep_01",
        "Q2": identity_root / "CLEAN_QUALIFICATION" / "rep_02",
        "C1": identity_root / "COUNTERFACTUAL_CLEAN_PREFIX" / "rep_01",
        "C2": identity_root / "COUNTERFACTUAL_CLEAN_PREFIX" / "rep_02",
    }
    receipts: dict[str, dict[str, Any]] = {}
    for label, run_root in runs.items():
        receipt_path = run_root / "RB1_INDEPENDENT_RECEIPT.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"M1_RECEIPT_MISSING:{label}")
        receipt = _load(receipt_path)
        if receipt.get("canonical_parent_key") != identity:
            raise RuntimeError(f"M1_IDENTITY_MISMATCH:{label}")
        for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts", "intervention_applied_steps", "counterfactual_open_steps"):
            if receipt.get(field) != 0:
                raise RuntimeError(f"M1_PROTECTED_BOUNDARY_NONZERO:{label}:{field}")
        receipts[label] = receipt
    return {"runs": runs, "receipts": receipts}


def audit(root: Path, *, final: bool) -> dict[str, Any]:
    manifest = _load(root / "M1_MANIFEST.json")
    protocol = _load(root / "M1_PROTOCOL.json")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise RuntimeError("M1_MANIFEST_STATUS_CHANGED")
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("M1_SOURCE_BINDING_MISMATCH")
    if _git("status", "--porcelain"):
        raise RuntimeError("M1_AUDITOR_WORKTREE_DIRTY")
    if protocol.get("rb1_v1_protocol_sha256") != "18d2421b172cef881b8de70c4bccf0c65174dd66a8402b75f0696d1878f96a69":
        raise RuntimeError("M1_V1_PROTOCOL_BINDING_MISMATCH")
    if manifest.get("python_prefix") != PYTHON_PREFIX:
        raise RuntimeError("M1_MANIFEST_PYTHON_PREFIX_MISMATCH")
    identity = str(manifest["diagnostic_identity"])
    receipt_data = _audit_receipts(root, identity)
    matrix = write_r1(root, identity)
    raw_audit: dict[str, Any] = {}
    if final:
        for label, run_root in receipt_data["runs"].items():
            raw_path = run_root / "M1_RAW_CAPTURE_MANIFEST.json"
            if not raw_path.is_file():
                raise RuntimeError(f"M1_RAW_CAPTURE_MANIFEST_MISSING:{label}")
            raw_audit[label] = _verify_raw_manifest(run_root, _load(raw_path))
            if raw_audit[label]["verdict"] != "PASS":
                raise RuntimeError(f"M1_RAW_CAPTURE_AUDIT_FAIL:{label}")
    classification = matrix["classification"]
    classification_receipt = {
        "schema": "STAGE_V_M1_VISUAL_DETERMINISM_CLASSIFICATION_V1",
        "status": "PASS_CLASSIFIED" if final else "PENDING_R2_RAW_CAPTURE",
        "rb1a_status": "HOLD",
        "classification": classification,
        "identity": identity,
        "same_mode_q_exact": matrix["pairs"]["SAME_MODE_Q"]["initial_state_exact"] and all(item["equal"] for item in matrix["pairs"]["SAME_MODE_Q"]["traces"].values()),
        "same_mode_c_exact": matrix["pairs"]["SAME_MODE_C"]["initial_state_exact"] and all(item["equal"] for item in matrix["pairs"]["SAME_MODE_C"]["traces"].values()),
        "cross_mode_r1_exact": matrix["pairs"]["CROSS_MODE_R1"]["initial_state_exact"] and all(item["equal"] for item in matrix["pairs"]["CROSS_MODE_R1"]["traces"].values()),
        "cross_mode_r2_exact": matrix["pairs"]["CROSS_MODE_R2"]["initial_state_exact"] and all(item["equal"] for item in matrix["pairs"]["CROSS_MODE_R2"]["traces"].values()),
        "initial_state_exact": all(pair["initial_state_exact"] for pair in matrix["pairs"].values()),
        "full_sim_state_exact": all(pair["traces"]["full_sim_state"]["equal"] for pair in matrix["pairs"].values()),
        "physical_state_exact": all(pair["traces"]["physical_state"]["equal"] for pair in matrix["pairs"].values()),
        "raw_observation_exact": all(pair["traces"]["raw_observation"]["equal"] for pair in matrix["pairs"].values()),
        "policy_rgb_exact": all(pair["traces"]["policy_rgb"]["equal"] for pair in matrix["pairs"].values()),
        "pixel_values_exact": all(pair["traces"]["model_input"]["equal"] for pair in matrix["pairs"].values()),
        "input_ids_exact": all(pair["first_mismatch_by_component"]["input_ids"] is None for pair in matrix["pairs"].values()),
        "attention_mask_exact": all(pair["first_mismatch_by_component"]["attention_mask"] is None for pair in matrix["pairs"].values()),
        "token_trace_exact": all(pair["traces"]["token"]["equal"] for pair in matrix["pairs"].values()),
        "action_trace_exact": all(pair["traces"]["postprocessed_action"]["equal"] for pair in matrix["pairs"].values()),
        "first_mismatch": {name: pair["first_mismatch_by_component"] for name, pair in matrix["pairs"].items()},
        "numeric_forensic_available": bool(final),
        "current_rb1_v1_modified": False,
        "current_rb1_v1_relaxed": False,
        "new_science_rollouts_authorized": False,
        "formal_map_authorized": False,
        "qualification_authorized": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "raw_capture_audit": raw_audit,
    }
    _write(root / "M1_CLASSIFICATION_RECEIPT.json", classification_receipt)
    status = _load(root / "M1_STATUS.json")
    status.update({"classification": classification, "classification_status": classification_receipt["status"], "independent_audit": "PASS", "raw_capture_audit": raw_audit})
    _write(root / "M1_STATUS.json", status)
    return classification_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit(args.root.resolve(), final=args.final)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": "PASS", "classification": result["classification"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
