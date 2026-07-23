#!/usr/bin/env python3
"""Phase B identity-disjointness validator — V3.

Authoritative mode reads raw C/P/H Teacher bundles and recomputes coverage
from label rows.  Manifest summaries are cross-checked but never trusted
as the sole source.

Three gates:
  Gate 1 — Identity Disjointness (hard): contamination → NESTED_RETRAIN_REQUIRED
  Gate 2 — Statistical Coverage (soft): computed from raw Teacher labels
  Gate 3 — Heldout Teacher Closure (hard): K10, step closure, contract parity

K10 contract parity: authoritative mode REJECTS INTERNAL_SIMPLIFIED_V1.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, math, os, sys, uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIVE_ROLES = ["checkpoint_training", "calibrator_fit", "policy_selection", "heldout_l3", "attack_eval"]
ROLE_LABELS = {"checkpoint_training": "T", "calibrator_fit": "C", "policy_selection": "P",
               "heldout_l3": "H", "attack_eval": "A"}
ROLE_TO_COHORT = {"checkpoint_training": "DETECTOR_TRAIN", "calibrator_fit": "DETECTOR_VAL",
                  "policy_selection": "DETECTOR_VAL", "heldout_l3": "DETECTOR_TEST",
                  "attack_eval": "ATTACK_EVAL"}
COHORT_TO_ROLE = {"DETECTOR_TRAIN": "checkpoint_training",
                  "DETECTOR_VAL": ["calibrator_fit", "policy_selection"],
                  "DETECTOR_TEST": "heldout_l3", "ATTACK_EVAL": "attack_eval"}
PAIRWISE_CONSTRAINTS = [
    ("checkpoint_training", "calibrator_fit"), ("checkpoint_training", "policy_selection"),
    ("checkpoint_training", "heldout_l3"), ("checkpoint_training", "attack_eval"),
    ("calibrator_fit", "policy_selection"), ("calibrator_fit", "heldout_l3"),
    ("policy_selection", "heldout_l3"), ("attack_eval", "checkpoint_training"),
    ("attack_eval", "calibrator_fit"), ("attack_eval", "policy_selection"),
    ("attack_eval", "heldout_l3"),
]
ACCEPTED_PROVENANCE = {"TRAINING_DATALOADER_LOG", "CANONICAL_TRAINING_LEDGER", "CHECKPOINT_SAMPLER_STATE"}
CALIBRATION_HEADS = ["grasp", "manipulation", "release"]
HEAD_TARGET_MAP = {"grasp": "grasp_established", "manipulation": "manipulation_active", "release": "release_or_instability"}
HEAD_KNOWN_MAP = {"grasp": "grasp_established_known_mask", "manipulation": "manipulation_active_known_mask", "release": "release_or_instability_known_mask"}
FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
REJECTED_K10_SCHEMAS = {"INTERNAL_SIMPLIFIED_V1"}
REQUIRED_K10_FIELDS = ("strict_k10_feasible", "strict_k10_known_mask")
REQUIRED_IDENTITY_FIELDS = ("canonical_parent_key", "step", "source_artifact_recursive_sha256")
SELF_SHA = None


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

def load_manifest(path, label):
    if not path.is_file():
        raise SystemExit(f"{label}_MANIFEST_NOT_FOUND: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_MANIFEST_PARSE_ERROR: {e}")

def load_strict_json(path, label):
    """Duplicate-key-aware JSON loader."""
    dups = []
    def hook(pairs):
        seen = set(); result = {}
        for k, v in pairs:
            if k in seen: dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw, object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE_ERROR: {path} {e}")
    if dups:
        raise SystemExit(f"{label}_DUPLICATE_KEYS: {path} keys={dups[:5]}")
    return value

def extract_identities(manifest, role, split_key):
    if "identities" in manifest:
        return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    if split_key in splits:
        sd = splits[split_key]
        if isinstance(sd, list): return set(sd)
        if isinstance(sd, dict): return set(sd.get(role, []))
    if role in manifest:
        rd = manifest[role]
        if isinstance(rd, list): return set(rd)
    return set()

def load_teacher_labels(bundle_root, split_key):
    """Load and validate per-split Teacher label rows. Returns list[dict]."""
    if not bundle_root:
        return None
    bp = Path(bundle_root) / split_key / "factorized_teacher_v1.jsonl"
    if not bp.is_file():
        return None
    rows = []
    seen = set()
    for line_nr, line in enumerate(open(bp), 1):
        if not line.strip(): continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"TEACHER_LABEL_PARSE: {bp}:{line_nr} {e}")
        for fld in REQUIRED_IDENTITY_FIELDS:
            if fld not in r:
                raise SystemExit(f"TEACHER_LABEL_MISSING_FIELD: {bp}:{line_nr} {fld}")
        key = (r["canonical_parent_key"], int(r["step"]))
        if key in seen:
            raise SystemExit(f"TEACHER_LABEL_DUPLICATE: {bp}:{line_nr} key={key}")
        seen.add(key)
        rows.append(r)
    return rows

def verify_bundle_seal(bundle_root, label):
    """Verify SHA256SUMS + SHA256SUMS.sha256 seal of a bundle directory."""
    bp = Path(bundle_root)
    if not bp.is_dir(): return
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED: missing SHA256SUMS or .sha256 in {bp}")
    expected = sha256_file(sums)
    actual = sidecar.read_text().strip().split()[0]
    if actual != expected:
        raise SystemExit(f"{label}_SEAL_BROKEN: expected {expected[:16]} got {actual[:16]}")


# ══════════════════════════════════════════════════════
# Gate 1 — Identity Disjointness
# ══════════════════════════════════════════════════════

def check_pairwise_disjoint(sets_by_role, split_key, errors):
    ok = True
    for r1, r2 in PAIRWISE_CONSTRAINTS:
        s1, s2 = sets_by_role.get(r1, set()), sets_by_role.get(r2, set())
        overlap = s1 & s2
        if overlap:
            n = len(overlap); preview = sorted(overlap)[:5]
            errors.append(f"IDENTITY_LEAKAGE: {split_key} {ROLE_LABELS[r1]}∩{ROLE_LABELS[r2]}={n} examples={preview}{'...' if n>5 else ''}")
            ok = False
    return ok

def check_training_provenance(training_manifest, split_key, errors):
    method = training_manifest.get("provenance_method", "")
    if method == "SET_SUBTRACTION":
        errors.append(f"PROVENANCE_REJECTED: {split_key} SET_SUBTRACTION not accepted")
    if method not in ACCEPTED_PROVENANCE:
        errors.append(f"PROVENANCE_UNVERIFIED: {split_key} method='{method}' must be one of {sorted(ACCEPTED_PROVENANCE)}")

def check_cohort_membership(sets_by_role, cohort_membership, split_key, errors):
    if not cohort_membership:
        errors.append(f"COHORT_MEMBERSHIP_MISSING: {split_key}")
        return
    for role_name, ids in sets_by_role.items():
        expected_cohort = ROLE_TO_COHORT.get(role_name)
        if expected_cohort is None: continue
        for eid in ids:
            actual = cohort_membership.get(eid)
            if actual is None:
                errors.append(f"COHORT_UNKNOWN: {split_key} {ROLE_LABELS[role_name]} identity '{eid}' not in membership ledger")
            elif actual != expected_cohort:
                if expected_cohort == "DETECTOR_VAL" and actual == "DETECTOR_VAL": continue
                errors.append(f"COHORT_VIOLATION: {split_key} {ROLE_LABELS[role_name]} '{eid}' in '{actual}' expected '{expected_cohort}'")

def check_deterministic_allocation(allocation, sets_by_role, split_key, errors):
    da = allocation.get("deterministic_allocation", {})
    if not da:
        # If source_status is DETERMINISTIC_ALLOCATION, allocation receipt is required
        if allocation.get("allocation_method") == "DETERMINISTIC_SPLIT":
            errors.append(f"ALLOC_MISSING: {split_key} deterministic_allocation block required for DETERMINISTIC_SPLIT")
        return

    parent = da.get("parent_cohort", "")
    if parent and parent != "DETECTOR_VAL":
        errors.append(f"ALLOC_PARENT: {split_key} expected DETECTOR_VAL, got '{parent}'")

    # Required fields
    for field in ["parent_cohort_manifest_sha256", "fixed_salt", "canonical_sort_key",
                   "allocation_algorithm_sha256", "allocation_code_sha256"]:
        val = da.get(field, "")
        if not val or not isinstance(val, str) or len(val) < 8:
            errors.append(f"ALLOC_MISSING: {split_key} deterministic_allocation.{field}")
        elif field.endswith("_sha256") and len(val) != 64:
            errors.append(f"ALLOC_SHA_LEN: {split_key} {field} len={len(val)} expected 64")

    # C∪P closure
    c_ids, p_ids = sets_by_role.get("calibrator_fit", set()), sets_by_role.get("policy_selection", set())
    cp_union = c_ids | p_ids
    if c_ids & p_ids:
        errors.append(f"ALLOC_CP_OVERLAP: {split_key} C∩P={len(c_ids & p_ids)}")

    val_ids = da.get("parent_cohort_identities", {})
    if isinstance(val_ids, dict):
        val_split = set(val_ids.get(split_key, []))
    elif isinstance(val_ids, list):
        val_split = set(val_ids)
    else:
        val_split = set()

    if val_split:
        missing = val_split - cp_union
        extra = cp_union - val_split
        if missing: errors.append(f"ALLOC_CLOSURE: {split_key} {len(missing)} VAL ids not in C∪P")
        if extra: errors.append(f"ALLOC_EXTRA: {split_key} {len(extra)} ids in C∪P not in VAL")

    unassigned = da.get("unassigned_identities", {})
    if isinstance(unassigned, dict):
        un_count = len(unassigned.get(split_key, []))
    elif isinstance(unassigned, list):
        un_count = len(unassigned)
    else:
        un_count = 0


# ══════════════════════════════════════════════════════
# Gate 2 — Statistical Coverage (from raw Teacher labels)
# ══════════════════════════════════════════════════════

def compute_calibration_coverage_from_labels(teacher_rows, split_key, teacher_contract_sha, authoritative, cov_issues):
    """Recompute per-head known positive/negative from raw Teacher labels.

    In authoritative mode, every row's source_artifact_recursive_sha256 must
    match the Teacher contract SHA.
    """
    if teacher_rows is None:
        cov_issues.append(f"CALIBRATION_BUNDLE_MISSING: {split_key}")
        return
    if not teacher_rows:
        cov_issues.append(f"CALIBRATION_EMPTY: {split_key}")
        return

    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)

    for head in CALIBRATION_HEADS:
        target_key = HEAD_TARGET_MAP[head]
        known_key = HEAD_KNOWN_MAP[head]
        n_pos = sum(1 for r in teacher_rows if r.get(known_key) and r.get(target_key))
        n_neg = sum(1 for r in teacher_rows if r.get(known_key) and not r.get(target_key))
        n_unknown = sum(1 for r in teacher_rows if not r.get(known_key))

        if n_pos == 0:
            cov_issues.append(f"CALIBRATION_NO_POSITIVE: {split_key}/{head} 0 known positive")
        if n_neg == 0:
            cov_issues.append(f"CALIBRATION_NO_NEGATIVE: {split_key}/{head} 0 known negative")

        # K10 contract check: every row's k10 schema
        if authoritative and teacher_contract_sha:
            for r in teacher_rows:
                k10_schema = r.get("strict_k10_binding_schema", "")
                if k10_schema in REJECTED_K10_SCHEMAS:
                    cov_issues.append(f"CALIBRATION_K10_REJECTED: {split_key}/{head} step={r.get('step')} schema='{k10_schema}' — authoritative mode requires external K10")
                    break
            # Also check source SHA consistency across all rows
            for r in teacher_rows:
                src = r.get("source_artifact_recursive_sha256", "")
                if src and teacher_contract_sha and src != teacher_contract_sha:
                    cov_issues.append(f"CALIBRATION_SOURCE_SHA_MISMATCH: {split_key} step={r.get('step')} expected={teacher_contract_sha[:16]} got={src[:16]}")
                    break


def compute_policy_coverage_from_labels(teacher_rows, split_key, authoritative, cov_issues):
    """Recompute policy coverage (negative episodes, K10 opportunities) from raw labels."""
    if teacher_rows is None:
        cov_issues.append(f"POLICY_BUNDLE_MISSING: {split_key}")
        return
    if not teacher_rows:
        cov_issues.append(f"POLICY_EMPTY: {split_key}")
        return

    # K10 contract check
    if authoritative:
        for r in teacher_rows:
            k10_schema = r.get("strict_k10_binding_schema", "")
            if k10_schema in REJECTED_K10_SCHEMAS:
                cov_issues.append(f"POLICY_K10_REJECTED: {split_key} step={r.get('step')} schema='{k10_schema}' — authoritative mode requires external K10")
                break

    for fld in REQUIRED_K10_FIELDS:
        for r in teacher_rows:
            if not isinstance(r.get(fld), bool):
                cov_issues.append(f"POLICY_K10_FIELD_TYPE: {split_key} step={r.get('step')} {fld} is {type(r.get(fld)).__name__}, expected bool")
                break

    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)

    n_neg = n_pos_k10 = n_unknown = n_eligible = n_known_denom = 0
    for ep_id, ep_rows in by_ep.items():
        T = len(ep_rows)
        if T < 10:
            n_unknown += 1; continue
        n_eligible += 1
        last_eligible = T - 10
        eligible = ep_rows[:last_eligible + 1]
        known_all = all(r.get("strict_k10_known_mask", False) for r in eligible)
        has_pos = any(r.get("strict_k10_feasible", False) and r.get("strict_k10_known_mask", False) for r in eligible)
        if has_pos: n_pos_k10 += 1; n_known_denom += 1
        elif known_all: n_neg += 1; n_known_denom += 1
        else: n_unknown += 1

    if n_neg == 0:
        cov_issues.append(f"POLICY_NO_NEGATIVE: {split_key} 0 negative episodes — false-start rate undefined")
    if n_pos_k10 == 0:
        cov_issues.append(f"POLICY_NO_OPPORTUNITY: {split_key} 0 strict-K10 positive opportunities — recall undefined")
    if n_eligible == 0:
        cov_issues.append(f"POLICY_NO_ELIGIBLE: {split_key} 0 eligible episodes")
    if n_known_denom == 0:
        cov_issues.append(f"POLICY_NO_KNOWN_DENOM: {split_key} 0 episodes with complete known denominator")


# ══════════════════════════════════════════════════════
# Gate 3 — Heldout Teacher Closure
# ══════════════════════════════════════════════════════

def check_heldout_teacher_closure_v3(h_ids, teacher_rows, split_key, teacher_contract_sha, authoritative, htc_errors):
    """H Teacher closure with K10 enforcement, step closure, and contract parity."""
    if teacher_rows is None:
        htc_errors.append(f"H_BUNDLE_MISSING: {split_key}")
        return
    if not teacher_rows:
        htc_errors.append(f"H_EMPTY: {split_key}")
        return

    # Identity closure
    teacher_ids = set()
    for r in teacher_rows:
        teacher_ids.add(r["canonical_parent_key"])
    missing = h_ids - teacher_ids
    extra = teacher_ids - h_ids
    if missing:
        htc_errors.append(f"H_ID_MISSING: {split_key} {len(missing)}: {sorted(missing)[:5]}")
    if extra:
        htc_errors.append(f"H_ID_EXTRA: {split_key} {len(extra)}: {sorted(extra)[:5]}")
    if len(teacher_ids) != len(h_ids):
        htc_errors.append(f"H_ID_COUNT: {split_key} manifest={len(h_ids)} teacher={len(teacher_ids)}")

    # K10 contract parity
    if authoritative:
        for r in teacher_rows:
            k10_schema = r.get("strict_k10_binding_schema", "")
            if k10_schema in REJECTED_K10_SCHEMAS:
                htc_errors.append(f"H_K10_REJECTED: {split_key} step={r.get('step')} schema='{k10_schema}' — authoritative mode requires external K10")
                break

    # K10 fields must be bool
    for fld in REQUIRED_K10_FIELDS:
        for r in teacher_rows:
            if not isinstance(r.get(fld), bool):
                htc_errors.append(f"H_K10_TYPE: {split_key} step={r.get('step')} {fld} is {type(r.get(fld)).__name__}")
                break

    # Source SHA consistency (all rows, not just first)
    if teacher_contract_sha:
        for r in teacher_rows:
            src = r.get("source_artifact_recursive_sha256", "")
            if src and len(src) == 64 and src != teacher_contract_sha:
                htc_errors.append(f"H_SOURCE_SHA: {split_key} step={r.get('step')} expected={teacher_contract_sha[:16]} got={src[:16]}")
                break

    # Step closure per identity
    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)

    for ep_id in sorted(h_ids & teacher_ids):
        ep_rows = sorted(by_ep.get(ep_id, []), key=lambda r: r["step"])
        if not ep_rows:
            htc_errors.append(f"H_NO_ROWS: {split_key}/{ep_id}")
            continue
        steps = [r["step"] for r in ep_rows]
        if steps[0] != 0:
            htc_errors.append(f"H_STEP_START: {split_key}/{ep_id} first_step={steps[0]} expected 0")
        for i, s in enumerate(steps):
            if not isinstance(s, int) or isinstance(s, bool):
                htc_errors.append(f"H_STEP_TYPE: {split_key}/{ep_id} step[{i}]={s} type={type(s).__name__}")
                break
            if s != i:
                htc_errors.append(f"H_STEP_GAP: {split_key}/{ep_id} expected {i} got {s}")
                break

        # K10 denominator: each identity must have at least one evaluable position
        T = len(ep_rows)
        if T >= 10:
            eligible = ep_rows[:T - 9]
            has_known_k10 = any(r.get("strict_k10_known_mask", False) for r in eligible)
            if not has_known_k10:
                htc_errors.append(f"H_K10_DENOM_EMPTY: {split_key}/{ep_id} 0 K10-known positions in eligible domain")
        else:
            htc_errors.append(f"H_TOO_SHORT: {split_key}/{ep_id} T={T} < 10")


# ══════════════════════════════════════════════════════
# Input audit
# ══════════════════════════════════════════════════════

REQUIRED_INPUTS = ["identity_source_discovery", "checkpoint_training_ledger",
                   "calibrator_fit_manifest", "policy_selection_manifest",
                   "heldout_l3_manifest", "attack_eval_manifest"]
REQUIRED_INPUT_LABELS = {
    "identity_source_discovery": "FACTORIZED_IDENTITY_SOURCE_DISCOVERY_V1.json",
    "checkpoint_training_ledger": "FACTORIZED_CHECKPOINT_TRAINING_IDENTITY_LEDGER_V1.json",
    "calibrator_fit_manifest": "calibrator_fit identity manifest",
    "policy_selection_manifest": "policy_selection identity manifest",
    "heldout_l3_manifest": "heldout_l3 identity manifest",
    "attack_eval_manifest": "attack_eval identity manifest",
}

def audit_inputs(input_paths):
    present = {}; missing = []
    for key in REQUIRED_INPUTS:
        p = input_paths.get(key)
        if p and Path(p).is_file():
            present[key] = {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))}
        else:
            missing.append({"input_key": key, "expected": REQUIRED_INPUT_LABELS[key], "provided": str(p) if p else None})
    return present, missing


# ══════════════════════════════════════════════════════
# Verdict classification
# ══════════════════════════════════════════════════════

def classify_verdict(disjointness_pass, source_status, inputs_complete):
    if not inputs_complete: return "HOLD_INPUTS_MISSING"
    if not disjointness_pass: return "NESTED_RETRAIN_REQUIRED"
    if source_status == "RECOVERED_EXISTING_ROOTS": return "PASS_EXISTING_ROOTS"
    if source_status == "DETERMINISTIC_ALLOCATION": return "PASS_DETERMINISTIC_ALLOCATION"
    return "HOLD_INPUTS_MISSING"

def classify_coverage(coverage_issues, inputs_complete):
    if not inputs_complete: return "NOT_AUDITABLE"
    if not coverage_issues: return "PASS"
    return "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE"

def classify_k10_parity(cal_issues, pol_issues, htc_issues, authoritative):
    """If authoritative mode and K10 is rejected anywhere, flag mismatch."""
    if not authoritative: return "DIAGNOSTIC_ONLY"
    all_issues = cal_issues + pol_issues + htc_issues
    for iss in all_issues:
        if "K10_REJECTED" in iss: return "NOT_AUDITABLE_K10_CONTRACT_MISMATCH"
    return "PASS"

def phase_c_authorization(verdict, cal_coverage_pass, pol_coverage_pass, htc_pass, k10_pass, authoritative):
    identity_clean = verdict in ("PASS_EXISTING_ROOTS", "PASS_DETERMINISTIC_ALLOCATION")
    cp_inference = "AUTHORIZED" if (identity_clean and cal_coverage_pass and pol_coverage_pass and (not authoritative or k10_pass == "PASS")) else "HOLD"
    l3_data_ready = identity_clean and htc_pass and (not authoritative or k10_pass == "PASS")
    l3_inference = False
    return {
        "cp_inference_authorized": cp_inference == "AUTHORIZED",
        "cp_inference_status": cp_inference,
        "heldout_l3_data_ready": l3_data_ready,
        "heldout_l3_inference_authorized": l3_inference,
        "heldout_l3_blocker": "PENDING_EXTERNAL_FREEZE" if l3_data_ready else "HOLD_DATA_NOT_READY",
        "k10_contract_parity": k10_pass,
    }


# ══════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════

def write_csv(staging, filename, headers, rows):
    with open(staging / filename, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for row in rows: w.writerow(row)
    return sha256_str(open(staging / filename).read())

def seal_dir(root):
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    with open(root / "SHA256SUMS", "w") as f: f.write(sums)
    seal = sha256_file(root / "SHA256SUMS")
    with open(root / "SHA256SUMS.sha256", "w") as f: f.write(f"{seal}  SHA256SUMS\n")
    return seal


def main():
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-source-discovery", type=Path, required=True)
    ap.add_argument("--checkpoint-training-ledger", type=Path, required=True)
    ap.add_argument("--calibrator-fit-manifest", type=Path, default=None)
    ap.add_argument("--policy-selection-manifest", type=Path, default=None)
    ap.add_argument("--heldout-l3-manifest", type=Path, default=None)
    ap.add_argument("--attack-eval-manifest", type=Path, default=None)
    ap.add_argument("--calibration-teacher-bundle-root", type=Path, default=None)
    ap.add_argument("--policy-teacher-bundle-root", type=Path, default=None)
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, default=None)
    ap.add_argument("--teacher-contract-file", type=Path, default=None)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    ap.add_argument("--require-cp-authorization", action="store_true",
                    help="Exit non-zero if CP inference not authorized")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    authoritative = args.mode == "authoritative"
    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)

    # ── 12-split enforcement in authoritative mode ──
    if authoritative:
        if expected_set != FROZEN_SPLITS:
            raise SystemExit(f"AUTHORITATIVE_SPLIT_ENFORCEMENT: expected exactly {sorted(FROZEN_SPLITS)}, got {sorted(expected_set)}")

    # ── Input audit ──
    input_paths = {
        "identity_source_discovery": args.identity_source_discovery,
        "checkpoint_training_ledger": args.checkpoint_training_ledger,
        "calibrator_fit_manifest": args.calibrator_fit_manifest,
        "policy_selection_manifest": args.policy_selection_manifest,
        "heldout_l3_manifest": args.heldout_l3_manifest,
        "attack_eval_manifest": args.attack_eval_manifest,
    }
    present_inputs, missing_inputs = audit_inputs(input_paths)
    inputs_complete = len(missing_inputs) == 0

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # ── Teacher contract SHA — computed from file, never trusted string ──
    teacher_contract_sha = None
    if args.teacher_contract_file:
        teacher_contract_sha = sha256_file(args.teacher_contract_file)

    # ── HOLD if inputs incomplete ──
    if not inputs_complete:
        receipt = {"schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "validator_code_sha256": SELF_SHA,
                   "status": "HOLD_INPUTS_MISSING", "verdict": "HOLD_INPUTS_MISSING",
                   "present_inputs": present_inputs, "missing_inputs": missing_inputs,
                   "cp_inference_authorized": False, "heldout_l3_data_ready": False,
                   "heldout_l3_inference_authorized": False}
        (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2.json").write_text(json.dumps(receipt, indent=2) + "\n")
        seal_dir(staging); os.replace(staging, out_root)
        print(f"HOLD_INPUTS_MISSING: {len(missing_inputs)} missing")
        for m in missing_inputs: print(f"  MISSING: {m['input_key']} ({m['expected']})")
        return 0

    # ── Load manifests ──
    discovery = load_strict_json(args.identity_source_discovery, "IDENTITY_SOURCE_DISCOVERY")
    training_ledger = load_strict_json(args.checkpoint_training_ledger, "CHECKPOINT_TRAINING_LEDGER")
    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CALIBRATOR_FIT")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POLICY_SELECTION")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELDOUT_L3")
    atk_manifest = load_strict_json(args.attack_eval_manifest, "ATTACK_EVAL")

    # ── Verify Teacher bundle seals ──
    if authoritative:
        for bundle_root, label in [
            (args.calibration_teacher_bundle_root, "CALIBRATION_TEACHER"),
            (args.policy_teacher_bundle_root, "POLICY_TEACHER"),
            (args.heldout_teacher_bundle_root, "HELDOUT_TEACHER"),
        ]:
            if bundle_root:
                verify_bundle_seal(bundle_root, label)

    source_status = discovery.get("identity_source_status", "UNKNOWN")
    cohort_membership = discovery.get("cohort_membership", {})

    # ── Per-split audit ──
    all_disjoint_errors = []; all_cov_issues = []; all_htc_errors = []
    per_split = {}; pairwise_rows = []; cohort_rows = []

    for sk in expected:
        disjoint_errors = []; cov_issues = []; htc_local = []

        # Gather identity sets
        sets_by_role = {}
        sets_by_role["checkpoint_training"] = extract_identities(training_ledger, "checkpoint_training", sk)
        sets_by_role["calibrator_fit"] = extract_identities(cal_manifest, "calibrator_fit", sk)
        sets_by_role["policy_selection"] = extract_identities(pol_manifest, "policy_selection", sk)
        sets_by_role["heldout_l3"] = extract_identities(held_manifest, "heldout_l3", sk)
        sets_by_role["attack_eval"] = extract_identities(atk_manifest, "attack_eval", sk)

        counts = {role: len(ids) for role, ids in sets_by_role.items()}

        # ── Empty-set guards ──
        for role, ids in sets_by_role.items():
            if len(ids) == 0:
                disjoint_errors.append(f"EMPTY_ROLE: {sk} {role} has 0 identities")

        # ── Gate 1: Identity Disjointness ──
        check_pairwise_disjoint(sets_by_role, sk, disjoint_errors)
        check_training_provenance(training_ledger, sk, disjoint_errors)
        check_cohort_membership(sets_by_role, cohort_membership, sk, disjoint_errors)

        alloc_manifest = cal_manifest if cal_manifest.get("deterministic_allocation") else pol_manifest
        if pol_manifest.get("deterministic_allocation"): alloc_manifest = pol_manifest
        check_deterministic_allocation(alloc_manifest, sets_by_role, sk, disjoint_errors)

        # Total union check
        all_ids = set()
        for ids in sets_by_role.values(): all_ids |= ids
        total_unique = len(all_ids)
        if total_unique != sum(counts.values()):
            disjoint_errors.append(f"IDENTITY_DUPLICATION: {sk} {sum(counts.values()) - total_unique} duplicates")

        # ── Gate 2: Statistical Coverage (from raw Teacher labels) ──
        cal_teacher_rows = load_teacher_labels(args.calibration_teacher_bundle_root, sk) if args.calibration_teacher_bundle_root else None
        pol_teacher_rows = load_teacher_labels(args.policy_teacher_bundle_root, sk) if args.policy_teacher_bundle_root else None

        compute_calibration_coverage_from_labels(cal_teacher_rows, sk, teacher_contract_sha, authoritative, cov_issues)
        compute_policy_coverage_from_labels(pol_teacher_rows, sk, authoritative, cov_issues)

        # Cross-check: manifest summaries must match recomputed values when both exist
        if authoritative and cal_teacher_rows:
            manifest_summaries = cal_manifest.get("calibration_head_summaries", {}).get(sk, {})
            if manifest_summaries:
                for head in CALIBRATION_HEADS:
                    target_key = HEAD_TARGET_MAP[head]
                    known_key = HEAD_KNOWN_MAP[head]
                    actual_pos = sum(1 for r in cal_teacher_rows if r.get(known_key) and r.get(target_key))
                    summary_pos = manifest_summaries.get(head, {}).get("known_positive", -1)
                    if summary_pos != -1 and actual_pos != summary_pos:
                        cov_issues.append(f"CALIBRATION_SUMMARY_MISMATCH: {sk}/{head} manifest={summary_pos} computed={actual_pos}")

        # ── Gate 3: Heldout Teacher Closure ──
        h_teacher_rows = load_teacher_labels(args.heldout_teacher_bundle_root, sk) if args.heldout_teacher_bundle_root else None
        check_heldout_teacher_closure_v3(sets_by_role["heldout_l3"], h_teacher_rows, sk, teacher_contract_sha, authoritative, htc_local)

        disjoint_ok = len(disjoint_errors) == 0
        cov_ok = len(cov_issues) == 0
        htc_ok = len(htc_local) == 0

        per_split[sk] = {"identity_disjointness_pass": disjoint_ok, "statistical_coverage_pass": cov_ok,
                         "heldout_teacher_closure_pass": htc_ok,
                         "disjointness_errors": disjoint_errors, "coverage_issues": cov_issues,
                         "htc_errors": htc_local, "identity_counts": counts, "total_unique": total_unique}
        all_disjoint_errors.extend(disjoint_errors)
        all_cov_issues.extend(cov_issues)
        all_htc_errors.extend(htc_local)

        for r1, r2 in PAIRWISE_CONSTRAINTS:
            s1, s2 = sets_by_role.get(r1, set()), sets_by_role.get(r2, set())
            pairwise_rows.append([sk, ROLE_LABELS[r1], ROLE_LABELS[r2], len(s1 & s2)])

        for role_name, ids in sets_by_role.items():
            expected_cohort = ROLE_TO_COHORT.get(role_name, "UNKNOWN")
            for eid in sorted(ids):
                actual = cohort_membership.get(eid, "UNKNOWN") if cohort_membership else "NO_LEDGER"
                violation = "" if actual == expected_cohort else "VIOLATION"
                cohort_rows.append([sk, ROLE_LABELS.get(role_name, role_name), eid, actual, expected_cohort, violation])

    # ── Final classification ──
    disjointness_pass = len(all_disjoint_errors) == 0
    cal_cov_issues = [c for c in all_cov_issues if "CALIBRATION" in c]
    pol_cov_issues = [c for c in all_cov_issues if "POLICY" in c]
    cal_coverage_pass = len(cal_cov_issues) == 0
    pol_coverage_pass = len(pol_cov_issues) == 0
    htc_pass = len(all_htc_errors) == 0

    coverage_status = classify_coverage(all_cov_issues, inputs_complete)
    k10_pass = classify_k10_parity(all_cov_issues, all_cov_issues, all_htc_errors, authoritative)
    verdict = classify_verdict(disjointness_pass, source_status, inputs_complete)
    phase_c = phase_c_authorization(verdict, cal_coverage_pass, pol_coverage_pass, htc_pass, k10_pass, authoritative)

    overall_data_integrity = disjointness_pass and htc_pass
    overall_scientific = cal_coverage_pass and pol_coverage_pass and (k10_pass == "PASS" or not authoritative)
    phase_b_overall = "PASS" if (overall_data_integrity and overall_scientific) else "HOLD"

    # ── Receipt V2 ──
    receipt = {
        "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
        "validator_code_sha256": SELF_SHA, "status": "COMPLETE", "verdict": verdict,
        "identity_disjointness": "PASS" if disjointness_pass else "FAIL",
        "statistical_coverage": coverage_status,
        "heldout_teacher_closure": "PASS" if htc_pass else "HOLD",
        "k10_contract_parity": k10_pass,
        "phase_b_data_integrity": "PASS" if overall_data_integrity else "HOLD",
        "phase_b_scientific_coverage": "PASS" if overall_scientific else "HOLD",
        "phase_b_overall": phase_b_overall,
        "cp_inference_authorized": phase_c["cp_inference_authorized"],
        "cp_inference_status": phase_c["cp_inference_status"],
        "heldout_l3_data_ready": phase_c["heldout_l3_data_ready"],
        "heldout_l3_inference_authorized": phase_c["heldout_l3_inference_authorized"],
        "heldout_l3_blocker": phase_c["heldout_l3_blocker"],
        "calibration_coverage_pass": cal_coverage_pass, "policy_coverage_pass": pol_coverage_pass,
        "heldout_teacher_closure_pass": htc_pass,
        "identity_source_status": source_status, "mode": args.mode,
        "n_disjointness_errors": len(all_disjoint_errors), "n_coverage_issues": len(all_cov_issues),
        "n_htc_errors": len(all_htc_errors), "n_splits": len(expected),
        "input_manifests": {"identity_source_discovery_sha256": sha256_file(args.identity_source_discovery),
            "checkpoint_training_ledger_sha256": sha256_file(args.checkpoint_training_ledger),
            "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
            "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
            "heldout_l3_manifest_sha256": sha256_file(args.heldout_l3_manifest),
            "attack_eval_manifest_sha256": sha256_file(args.attack_eval_manifest)},
        "per_split": per_split,
    }
    if teacher_contract_sha:
        receipt["teacher_contract_sha256"] = teacher_contract_sha
    if args.calibration_teacher_bundle_root:
        receipt["calibration_teacher_bundle_sha256"] = sha256_file(Path(args.calibration_teacher_bundle_root) / "SHA256SUMS")
    if args.policy_teacher_bundle_root:
        receipt["policy_teacher_bundle_sha256"] = sha256_file(Path(args.policy_teacher_bundle_root) / "SHA256SUMS")
    if args.heldout_teacher_bundle_root:
        htb_path = Path(args.heldout_teacher_bundle_root)
        htb_seal = htb_path / "SHA256SUMS"
        receipt["heldout_teacher_bundle_sha256"] = sha256_file(htb_seal) if htb_seal.is_file() else None
    if all_disjoint_errors: receipt["disjointness_errors"] = all_disjoint_errors
    if all_cov_issues: receipt["coverage_issues"] = all_cov_issues
    if all_htc_errors: receipt["heldout_teacher_closure_errors"] = all_htc_errors

    (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2.json").write_text(json.dumps(receipt, indent=2) + "\n")

    # ── CSVs ──
    write_csv(staging, "DEEPSEEK_PHASE_B_PAIRWISE_INTERSECTIONS_V1.csv",
              ["split", "role_a", "role_b", "intersection_count"], pairwise_rows)
    write_csv(staging, "DEEPSEEK_PHASE_B_COHORT_MEMBERSHIP_V1.csv",
              ["split", "role", "identity", "actual_cohort", "expected_cohorts", "status"], cohort_rows)
    cov_rows = []
    for sk in expected:
        ps = per_split[sk]
        for role in ["calibrator_fit", "policy_selection", "heldout_l3"]:
            cov_rows.append([sk, role, ps["identity_counts"].get(role, 0),
                             "PASS" if (role == "calibrator_fit" and not any("CALIBRATION" in c for c in ps["coverage_issues"]))
                                    or (role == "policy_selection" and not any("POLICY" in c for c in ps["coverage_issues"]))
                                    or (role == "heldout_l3" and not any("H_" in c for c in ps["htc_errors"])) else "ISSUES"])
    write_csv(staging, "DEEPSEEK_PHASE_B_STATISTICAL_COVERAGE_V1.csv",
              ["split", "role", "identity_count", "coverage_status"], cov_rows)
    write_csv(staging, "DEEPSEEK_PHASE_B_HELDOUT_TEACHER_CLOSURE_V1.csv",
              ["split", "heldout_identity_count", "teacher_closure_status"],
              [[sk, per_split[sk]["identity_counts"].get("heldout_l3", 0),
                "PASS" if len(per_split[sk].get("htc_errors", [])) == 0 else "FAIL"] for sk in expected])

    seal_dir(staging)
    os.replace(staging, out_root)

    # ── Report ──
    print(f"Phase B V3 Validation Complete")
    print(f"  Mode:                           {args.mode}")
    print(f"  Verdict:                        {verdict}")
    print(f"  Identity Disjointness:          {'PASS' if disjointness_pass else 'FAIL'} ({len(all_disjoint_errors)} errors)")
    print(f"  Statistical Coverage:           {coverage_status} ({len(all_cov_issues)} issues)")
    print(f"  Calibration Coverage:           {'PASS' if cal_coverage_pass else 'HOLD'}")
    print(f"  Policy Coverage:                {'PASS' if pol_coverage_pass else 'HOLD'}")
    print(f"  Heldout Teacher Closure:        {'PASS' if htc_pass else 'HOLD'} ({len(all_htc_errors)} errors)")
    print(f"  K10 Contract Parity:            {k10_pass}")
    print(f"  Phase B Data Integrity:         {'PASS' if overall_data_integrity else 'HOLD'}")
    print(f"  Phase B Scientific Coverage:    {'PASS' if overall_scientific else 'HOLD'}")
    print(f"  Phase B Overall:                {phase_b_overall}")
    print(f"  CP Inference Authorized:        {phase_c['cp_inference_authorized']}")
    print(f"  Heldout L3 Data Ready:          {phase_c['heldout_l3_data_ready']}")
    print(f"  Heldout L3 Inference:           {phase_c['heldout_l3_inference_authorized']}")
    print(f"  Heldout L3 Blocker:             {phase_c['heldout_l3_blocker']}")
    print(f"  Output:                         {out_root}")

    if all_disjoint_errors:
        print(f"\nDisjointness errors:"); [print(f"  {e}") for e in all_disjoint_errors[:10]]
    if all_cov_issues:
        print(f"\nCoverage issues:"); [print(f"  {c}") for c in all_cov_issues[:10]]
    if all_htc_errors:
        print(f"\nHTC errors:"); [print(f"  {e}") for e in all_htc_errors[:10]]

    # ── Exit code ──
    if args.require_cp_authorization and not phase_c["cp_inference_authorized"]:
        return 2
    if not disjointness_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
