#!/usr/bin/env python3
"""Validate Codex Factorized V3.1 handoff. V2/old-V3 rejected."""
from __future__ import annotations

import argparse, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V3_1_SCHEMA = "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1"
EXPECTED_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def check_ref(obj, label, errors, required=True):
    if not isinstance(obj, dict):
        if required: errors.append(f"FAIL: {label} not object"); return False
        return False
    rel = obj.get("path", ""); expected = obj.get("sha256", "")
    if not rel or not expected:
        errors.append(f"FAIL: {label} missing path/sha256"); return False
    fpath = ROOT / rel
    if not fpath.is_file(): errors.append(f"FAIL: {label} not found: {fpath}"); return False
    if ".." in rel or rel.startswith("/"): errors.append(f"FAIL: {label} unsafe: {rel}"); return False
    actual = sha256_file(fpath)
    if actual != expected:
        errors.append(f"FAIL: {label} SHA mismatch exp={expected[:16]} act={actual[:16]}"); return False
    return True


def detect_duplicate_keys(filepath):
    dups = set()
    def hook(pairs):
        seen = set()
        for k, v in pairs:
            if k in seen: dups.add(k)
            seen.add(k)
        return dict(pairs)
    with open(filepath) as f:
        json.load(f, object_pairs_hook=hook)
    return dups


def static_validation(handoff, errors):
    schema = handoff.get("schema", "")
    if schema != V3_1_SCHEMA:
        if "HANDOFF_V2" in schema: errors.append("FAIL: V2 STATIC_REJECTED")
        else: errors.append(f"FAIL: schema must be {V3_1_SCHEMA}")
        return False
    if handoff.get("status") != "READY_FOR_DEEPSEEK_STATIC_INTEGRATION":
        errors.append("FAIL: status")
    if handoff.get("interface_revision") != "V3.1":
        errors.append("FAIL: interface_revision != V3.1")
    sha = handoff.get("code_snapshot_commit", "")
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)):
        errors.append("FAIL: code_snapshot_commit")

    # Real V3.1 nested refs
    sa = handoff.get("scheduler_api", {})
    check_ref(sa.get("contract"), "scheduler_api.contract", errors)
    check_ref(sa.get("config"), "scheduler_api.config", errors)
    check_ref(sa.get("fixture"), "scheduler_api.fixture", errors)

    ra = handoff.get("runtime_adapter", {})
    check_ref(ra.get("source"), "runtime_adapter.source", errors)

    rb = handoff.get("runtime_bundle", {})
    # Corrected V3.1: schema_name + schema_file (not duplicate schema)
    if "schema" in rb and isinstance(rb["schema"], dict) and "schema_file" not in rb:
        errors.append("FAIL: runtime_bundle uses duplicate 'schema' key — Codex must fix to schema_name+schema_file")
    else:
        check_ref(rb.get("schema_file"), "runtime_bundle.schema_file", errors)
        if not rb.get("schema_name"): errors.append("FAIL: runtime_bundle.schema_name missing")
    check_ref(rb.get("rematerializer"), "runtime_bundle.rematerializer", errors)

    ob = handoff.get("offline_bundles", {})
    ocal = ob.get("calibration", {})
    check_ref(ocal.get("schema"), "offline.calibration.schema", errors)
    check_ref(ocal.get("label_schema"), "offline.calibration.label_schema", errors)
    oev = ob.get("evaluation", {})
    check_ref(oev.get("schema"), "offline.evaluation.schema", errors)
    check_ref(oev.get("label_schema"), "offline.evaluation.label_schema", errors)
    check_ref(oev.get("materializer"), "offline.evaluation.materializer", errors)

    cc = handoff.get("calibration_contract", {})
    check_ref(cc.get("schema"), "calibration_contract.schema", errors)

    check_ref(handoff.get("structural_config"), "structural_config", errors)

    hv = handoff.get("handoff_validator", {})
    check_ref(hv.get("source"), "handoff_validator.source", errors)

    ia = handoff.get("identity_manifest_audit", {})
    check_ref(ia.get("json"), "identity_manifest_audit.json", errors)
    check_ref(ia.get("csv"), "identity_manifest_audit.csv", errors)

    pr = handoff.get("production_receipt_requirements", {})
    check_ref(pr.get("handoff_receipt"), "production_receipt.handoff", errors)
    check_ref(pr.get("read_only_receipt_root"), "production_receipt.read_only_root", errors)
    check_ref(pr.get("receipt_seal"), "production_receipt.seal", errors)

    # Execution boundary — static must verify forbidden flags are false
    eb = handoff.get("execution_boundary", {})
    if not isinstance(eb, dict):
        errors.append("FAIL: execution_boundary missing"); return False
    for k in ["model_inference", "training", "full_fit", "cal_check", "rollout", "shadow", "attack"]:
        if eb.get(k) is not False:
            errors.append(f"FAIL: execution_boundary.{k} must be false, got {eb.get(k)}")
    if eb.get("static_interface") is not True:
        errors.append("FAIL: execution_boundary.static_interface must be true")

    splits = set(handoff.get("expected_split_keys", []))
    if splits != EXPECTED_SPLITS:
        miss = EXPECTED_SPLITS - splits; extra = splits - EXPECTED_SPLITS
        if miss: errors.append(f"FAIL: missing splits {sorted(miss)}")
        if extra: errors.append(f"FAIL: extra splits {sorted(extra)}")

    fb = set(handoff.get("forbidden_substitutions", []))
    required = {"grasp_probability -> utility_probability",
                "manipulation_probability -> regrasp_probability",
                "Teacher/event_id -> candidate_close"}
    missing_fb = required - fb
    if missing_fb: errors.append(f"FAIL: missing forbidden: {sorted(missing_fb)}")

    eb = handoff.get("execution_boundary", {})
    if not isinstance(eb, dict): errors.append("FAIL: execution_boundary missing")

    return len(errors) == 0


def execution_validation(handoff, errors):
    if not static_validation(handoff, errors): return False
    eb = handoff.get("execution_boundary", {})
    if not eb.get("runtime_rematerialization"):
        errors.append("FAIL: runtime rematerialization not done")
    if not eb.get("offline_evaluation_bundle"):
        errors.append("FAIL: offline eval bundle not done")
    if not eb.get("sealed_artifact_audit"):
        errors.append("FAIL: sealed artifact audit not done")
    return len(errors) == 0


def validate_handoff_static(handoff):
    errors = []
    ok = static_validation(handoff, errors)
    return ok, errors


def validate_handoff_execution(handoff):
    errors = []
    ok = execution_validation(handoff, errors)
    return ok, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff-json", type=Path, required=True)
    ap.add_argument("--mode", choices=["static","execution"], default="static")
    args = ap.parse_args()

    from load_factorized_handoff import load_handoff_file
    handoff = load_handoff_file(args.handoff_json.resolve(), ROOT)

    if args.mode == "execution":
        ok, errors = validate_handoff_execution(handoff)
    else:
        ok, errors = validate_handoff_static(handoff)
    if not ok:
        for e in errors: print(f"  {e}")
        print(f"CODEX_V3_1_{args.mode.upper()} = REJECTED"); sys.exit(1)
    print(f"CODEX_V3_1_{args.mode.upper()} = PASS")


if __name__ == "__main__":
    main()
