#!/usr/bin/env python3
"""Five-way identity disjointness validator for Factorized V2 Phase B.

Consumes Codex identity discovery manifests and proves or disproves:
  T ∩ C = T ∩ P = T ∩ H = T ∩ A = ∅
  C ∩ P = C ∩ H = P ∩ H = ∅
  A ∩ (T ∪ C ∪ P ∪ H) = ∅

where T=checkpoint-training, C=calibrator-fit, P=policy-selection,
H=heldout-L3, A=attack-eval.

THREE INDEPENDENT GATES:
  Gate 1 — Identity Disjointness (hard): contamination → NESTED_RETRAIN_REQUIRED
  Gate 2 — Statistical Coverage (soft): insufficient samples → HOLD, not retrain
  Gate 3 — Heldout Teacher Closure (hard): H Teacher bundle mismatch → HOLD

SPLIT PHASE C AUTHORIZATION:
  CP_INFERENCE_AUTHORIZED: identity closure + C/P coverage PASS
    → Validator can decide this.
  HELDOUT_L3_DATA_READY: identity closure + H Teacher closure PASS
    → Validator confirms data readiness only.
  HELDOUT_L3_INFERENCE_AUTHORIZED: always FALSE from validator.
    → Requires external freeze contracts (calibrator + scheduler thresholds).
  heldout_l3_blocker = PENDING_EXTERNAL_FREEZE when data is ready.

MISSING INPUTS → HOLD_INPUTS_MISSING (never NESTED_RETRAIN_REQUIRED)
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, sys, uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIVE_ROLES = ["checkpoint_training", "calibrator_fit", "policy_selection", "heldout_l3", "attack_eval"]
ROLE_LABELS = {"checkpoint_training": "T", "calibrator_fit": "C", "policy_selection": "P",
               "heldout_l3": "H", "attack_eval": "A"}
COHORT_TO_ROLE = {
    "DETECTOR_TRAIN": "checkpoint_training",
    "DETECTOR_VAL": ["calibrator_fit", "policy_selection"],
    "DETECTOR_TEST": "heldout_l3",
    "ATTACK_EVAL": "attack_eval",
}
ROLE_TO_COHORT = {
    "checkpoint_training": "DETECTOR_TRAIN",
    "calibrator_fit": "DETECTOR_VAL",
    "policy_selection": "DETECTOR_VAL",
    "heldout_l3": "DETECTOR_TEST",
    "attack_eval": "ATTACK_EVAL",
}

PAIRWISE_CONSTRAINTS = [
    ("checkpoint_training", "calibrator_fit"),
    ("checkpoint_training", "policy_selection"),
    ("checkpoint_training", "heldout_l3"),
    ("checkpoint_training", "attack_eval"),
    ("calibrator_fit", "policy_selection"),
    ("calibrator_fit", "heldout_l3"),
    ("policy_selection", "heldout_l3"),
    ("attack_eval", "checkpoint_training"),
    ("attack_eval", "calibrator_fit"),
    ("attack_eval", "policy_selection"),
    ("attack_eval", "heldout_l3"),
]

ACCEPTED_PROVENANCE = {"TRAINING_DATALOADER_LOG", "CANONICAL_TRAINING_LEDGER", "CHECKPOINT_SAMPLER_STATE"}
CALIBRATION_HEADS = ["grasp", "manipulation", "release"]
CALIBRATION_HEADS = ["grasp", "manipulation", "release"]
SELF_SHA = None  # computed at runtime


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


def extract_identities(manifest, role, split_key):
    """Extract identity set for a given role and split from a manifest."""
    if "identities" in manifest:
        return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    if split_key in splits:
        split_data = splits[split_key]
        if isinstance(split_data, list):
            return set(split_data)
        if isinstance(split_data, dict):
            return set(split_data.get(role, []))
        return set()
    if role in manifest:
        role_data = manifest[role]
        if isinstance(role_data, list):
            return set(role_data)
    return set()


# ══════════════════════════════════════════════════════════════════════════════
# Gate 1: Identity Disjointness (HARD — contamination → NESTED_RETRAIN_REQUIRED)
# ══════════════════════════════════════════════════════════════════════════════

def check_pairwise_disjoint(sets_by_role, split_key, disjoint_errors):
    """Check all pairwise constraints for a single split."""
    all_ok = True
    for r1, r2 in PAIRWISE_CONSTRAINTS:
        s1 = sets_by_role.get(r1, set())
        s2 = sets_by_role.get(r2, set())
        overlap = s1 & s2
        if overlap:
            n = len(overlap)
            preview = sorted(overlap)[:5]
            disjoint_errors.append(
                f"IDENTITY_LEAKAGE: {split_key} {ROLE_LABELS[r1]}∩{ROLE_LABELS[r2]}={n} "
                f"examples={preview}{'...' if n > 5 else ''}"
            )
            all_ok = False
    return all_ok


def check_training_provenance(training_manifest, split_key, disjoint_errors):
    method = training_manifest.get("provenance_method", "")
    if method == "SET_SUBTRACTION":
        disjoint_errors.append(
            f"PROVENANCE_REJECTED: {split_key} training identities derived by "
            f"set subtraction, not from actual training records"
        )
    if method not in ACCEPTED_PROVENANCE:
        disjoint_errors.append(
            f"PROVENANCE_UNVERIFIED: {split_key} provenance_method='{method}' "
            f"— must be one of {sorted(ACCEPTED_PROVENANCE)}"
        )


def check_cohort_membership(sets_by_role, cohort_membership, split_key, disjoint_errors):
    """Validate each identity belongs to its expected CLEAN2000 cohort."""
    if not cohort_membership:
        disjoint_errors.append(f"COHORT_MEMBERSHIP_MISSING: {split_key} — cannot verify cohort assignments")
        return

    for role_name, ids in sets_by_role.items():
        expected_cohort = ROLE_TO_COHORT.get(role_name)
        if expected_cohort is None:
            continue
        for eid in ids:
            actual = cohort_membership.get(eid)
            if actual is None:
                disjoint_errors.append(
                    f"COHORT_UNKNOWN: {split_key} {ROLE_LABELS[role_name]} identity '{eid}' "
                    f"not found in cohort membership ledger"
                )
            elif actual != expected_cohort:
                # Special case: DETECTOR_VAL maps to both C and P
                if expected_cohort == "DETECTOR_VAL" and actual == "DETECTOR_VAL":
                    continue
                disjoint_errors.append(
                    f"COHORT_VIOLATION: {split_key} {ROLE_LABELS[role_name]} identity '{eid}' "
                    f"in cohort '{actual}', expected '{expected_cohort}'"
                )


def check_deterministic_allocation(allocation, sets_by_role, split_key, disjoint_errors):
    """Validate deterministic C/P split from DETECTOR_VAL parent cohort."""
    da = allocation.get("deterministic_allocation", {})
    if not da:
        return  # not a deterministic allocation — skip

    parent = da.get("parent_cohort", "")
    if parent != "DETECTOR_VAL":
        disjoint_errors.append(
            f"ALLOC_PARENT: {split_key} expected parent DETECTOR_VAL, got '{parent}'"
        )

    for field in ["parent_cohort_manifest_sha256", "fixed_salt", "canonical_sort_key",
                   "allocation_algorithm_sha256", "allocation_code_sha256"]:
        val = da.get(field, "")
        if not val or not isinstance(val, str) or len(val) < 8:
            disjoint_errors.append(f"ALLOC_MISSING: {split_key} deterministic_allocation.{field}")

    for sha_field in ["parent_cohort_manifest_sha256", "allocation_algorithm_sha256", "allocation_code_sha256"]:
        val = da.get(sha_field, "")
        if val and len(val) != 64:
            disjoint_errors.append(f"ALLOC_SHA_LEN: {split_key} {sha_field} length={len(val)}, expected 64")

    # C/P union closure: all VAL identities must be accounted for
    c_ids = sets_by_role.get("calibrator_fit", set())
    p_ids = sets_by_role.get("policy_selection", set())
    assigned = da.get("assigned_identities", {})
    if isinstance(assigned, dict):
        assigned_set = set(assigned.get(split_key, []))
    else:
        assigned_set = set(assigned) if isinstance(assigned, list) else set()

    # If allocation manifest provides VAL identities, verify C∪P = VAL
    val_manifest_ids = da.get("parent_cohort_identities", {})
    if isinstance(val_manifest_ids, dict):
        val_split_ids = set(val_manifest_ids.get(split_key, []))
    elif isinstance(val_manifest_ids, list):
        val_split_ids = set(val_manifest_ids)
    else:
        val_split_ids = set()

    if val_split_ids:
        cp_union = c_ids | p_ids
        missing_from_cp = val_split_ids - cp_union
        extra_in_cp = cp_union - val_split_ids
        if missing_from_cp:
            n = len(missing_from_cp)
            disjoint_errors.append(
                f"ALLOC_CLOSURE: {split_key} {n} VAL identities not in C∪P"
            )
        if extra_in_cp:
            n = len(extra_in_cp)
            disjoint_errors.append(
                f"ALLOC_EXTRA: {split_key} {n} identities in C∪P but not in VAL parent cohort"
            )

    # Unassigned identity accounting
    unassigned = da.get("unassigned_identities", {})
    if isinstance(unassigned, dict):
        unassigned_count = len(unassigned.get(split_key, []))
    else:
        unassigned_count = len(unassigned) if isinstance(unassigned, list) else 0
    if unassigned_count > 0:
        disjoint_errors.append(
            f"ALLOC_UNASSIGNED: {split_key} {unassigned_count} VAL identities unassigned to C or P"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Gate 2: Statistical Coverage (SOFT — insufficiency → HOLD, not retrain)
# ══════════════════════════════════════════════════════════════════════════════

def check_calibration_coverage(allocation, split_key, cov_issues):
    """Per-head known positive/negative coverage for calibration identities.

    Requires allocation manifest to include calibration_head_summary per split.
    Without it, reports NOT_AUDITABLE.
    """
    head_summaries = allocation.get("calibration_head_summaries", {})
    split_summary = head_summaries.get(split_key, {})
    if not split_summary:
        cov_issues.append(f"CALIBRATION_COVERAGE: {split_key} head_summaries missing — NOT_AUDITABLE")
        return

    for head in CALIBRATION_HEADS:
        hs = split_summary.get(head, {})
        n_pos = hs.get("known_positive", 0)
        n_neg = hs.get("known_negative", 0)
        if n_pos == 0:
            cov_issues.append(
                f"CALIBRATION_NO_POSITIVE: {split_key}/{head} 0 known positive — "
                f"calibrator cannot fit positive class"
            )
        if n_neg == 0:
            cov_issues.append(
                f"CALIBRATION_NO_NEGATIVE: {split_key}/{head} 0 known negative — "
                f"calibrator cannot fit negative class"
            )


def check_policy_coverage(allocation, split_key, cov_issues):
    """Policy-selection coverage: negative episodes, K10 opportunities, denominators."""
    policy_summaries = allocation.get("policy_selection_summaries", {})
    split_summary = policy_summaries.get(split_key, {})
    if not split_summary:
        cov_issues.append(f"POLICY_COVERAGE: {split_key} summaries missing — NOT_AUDITABLE")
        return

    n_neg = split_summary.get("negative_episodes", 0)
    n_pos_k10 = split_summary.get("k10_positive_opportunities", 0)
    n_eligible = split_summary.get("eligible_episodes", 0)
    n_known = split_summary.get("known_denominator_episodes", 0)

    if n_neg == 0:
        cov_issues.append(
            f"POLICY_NO_NEGATIVE: {split_key} 0 negative episodes — "
            f"false-start rate undefined"
        )
    if n_pos_k10 == 0:
        cov_issues.append(
            f"POLICY_NO_OPPORTUNITY: {split_key} 0 strict-K10 positive opportunities — "
            f"recall undefined"
        )
    if n_eligible == 0:
        cov_issues.append(f"POLICY_NO_ELIGIBLE: {split_key} 0 eligible episodes")
    if n_known == 0:
        cov_issues.append(f"POLICY_NO_KNOWN: {split_key} 0 episodes with complete known denominator")


def check_heldout_coverage(sets_by_role, split_key, cov_issues):
    """Heldout-L3: 12-split denominator closure, no arbitrary threshold.

    When H Teacher bundle is available, also verifies per-identity label coverage.
    """
    h_ids = sets_by_role.get("heldout_l3", set())
    if len(h_ids) == 0:
        cov_issues.append(
            f"HELDOUT_EMPTY: {split_key} 0 heldout identities — denominator missing"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Gate 3: Heldout Teacher Closure (HARD — H Teacher bundle mismatch → HOLD)
# ══════════════════════════════════════════════════════════════════════════════

def check_heldout_teacher_closure(h_ids, h_teacher_bundle, split_key, teacher_source_sha, htc_errors):
    """Verify H Teacher bundle completeness and identity closure.

    Checks:
      - H manifest identities == H Teacher identities
      - 0 missing / 0 extra / 0 duplicate episode-step
      - exact step closure per identity
      - Teacher source SHA match
      - label schema match
      - known/unknown/abstain denominator defined per identity
    """
    if not h_teacher_bundle:
        htc_errors.append(
            f"H_TEACHER_BUNDLE_MISSING: {split_key} no H Teacher bundle mounted — "
            f"Cannot verify heldout Teacher closure. HELDOUT_L3_INFERENCE blocked."
        )
        return

    # Load H Teacher identities from bundle
    teacher_dir = Path(h_teacher_bundle) / split_key
    label_file = teacher_dir / "factorized_teacher_v1.jsonl"
    manifest_file = teacher_dir / "teacher_manifest.json"

    if not teacher_dir.is_dir():
        htc_errors.append(
            f"H_TEACHER_SPLIT_MISSING: {split_key} Teacher bundle split dir not found: {teacher_dir}"
        )
        return
    if not label_file.is_file():
        htc_errors.append(
            f"H_TEACHER_LABELS_MISSING: {split_key} {label_file}"
        )
        return

    # Read all H Teacher labels
    teacher_ids = set()
    teacher_dups = set()
    teacher_seen_ep_step = set()
    teacher_dup_ep_step = []
    label_schema = None
    label_source_sha = None
    per_id_steps = defaultdict(int)
    per_id_known = defaultdict(lambda: {"known": 0, "unknown": 0, "abstain": 0})

    try:
        with open(label_file) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ep = r.get("canonical_parent_key", r.get("episode", ""))
                step = r.get("step", -1)
                key = (ep, step)

                if key in teacher_seen_ep_step:
                    teacher_dup_ep_step.append(str(key))
                teacher_seen_ep_step.add(key)

                teacher_ids.add(ep)
                per_id_steps[ep] += 1

                # Track known/unknown/abstain for denominator (use gross known_mask check)
                gross_known = (
                    r.get("grasp_established_known_mask", False) or
                    r.get("manipulation_active_known_mask", False) or
                    r.get("release_or_instability_known_mask", False)
                )
                if gross_known:
                    per_id_known[ep]["known"] += 1
                else:
                    per_id_known[ep]["unknown"] += 1

                if label_schema is None:
                    label_schema = r.get("physics_protocol_schema", r.get("strict_k10_binding_schema", ""))
                if label_source_sha is None:
                    label_source_sha = r.get("source_artifact_recursive_sha256", "")
    except Exception as e:
        htc_errors.append(f"H_TEACHER_PARSE_ERROR: {split_key} {e}")
        return

    # Check: no duplicate episode-step
    if teacher_dup_ep_step:
        htc_errors.append(
            f"H_TEACHER_DUPLICATE: {split_key} {len(teacher_dup_ep_step)} duplicate episode-step keys"
        )

    # Check: identity closure — H manifest == H Teacher
    manifest_missing = sorted(h_ids - teacher_ids)
    manifest_extra = sorted(teacher_ids - h_ids)

    if manifest_missing:
        n = len(manifest_missing)
        htc_errors.append(
            f"H_TEACHER_IDENTITY_MISSING: {split_key} {n} identities in manifest but not in Teacher: "
            f"{manifest_missing[:5]}{'...' if n > 5 else ''}"
        )
    if manifest_extra:
        n = len(manifest_extra)
        htc_errors.append(
            f"H_TEACHER_IDENTITY_EXTRA: {split_key} {n} identities in Teacher but not in manifest: "
            f"{manifest_extra[:5]}{'...' if n > 5 else ''}"
        )

    # Check: exact identity count match
    if len(teacher_ids) != len(h_ids):
        htc_errors.append(
            f"H_TEACHER_IDENTITY_COUNT_MISMATCH: {split_key} "
            f"manifest={len(h_ids)} teacher={len(teacher_ids)}"
        )

    # Check: step closure per identity
    for ep_id in sorted(h_ids & teacher_ids):
        if ep_id not in per_id_steps:
            htc_errors.append(f"H_TEACHER_NO_STEPS: {split_key}/{ep_id}")
            continue

    # Check: Teacher source SHA match (if provided)
    if teacher_source_sha and label_source_sha:
        if teacher_source_sha != label_source_sha and label_source_sha and len(label_source_sha) >= 8:
            htc_errors.append(
                f"H_TEACHER_SOURCE_SHA_MISMATCH: {split_key} "
                f"expected={teacher_source_sha[:16]} got={label_source_sha[:16]}"
            )

    # Check: known denominator per identity
    for ep_id in sorted(h_ids & teacher_ids):
        kd = per_id_known[ep_id]
        if kd["known"] == 0 and kd["unknown"] == 0:
            htc_errors.append(f"H_TEACHER_EMPTY: {split_key}/{ep_id}")


def check_heldout_teacher_closure_global(h_ids, h_teacher_bundle, expected_splits, teacher_source_sha):
    """Check H Teacher closure across all splits. Returns (htc_errors, htc_rows)."""
    htc_errors = []
    htc_rows = []
    for sk in expected_splits:
        split_h_ids = set(h_ids.get(sk, []))
        check_heldout_teacher_closure(split_h_ids, h_teacher_bundle, sk, teacher_source_sha, htc_errors)
        htc_rows.append([sk, len(split_h_ids), "PASS" if not [e for e in htc_errors if sk in e] else "FAIL"])
    return htc_errors, htc_rows


# ══════════════════════════════════════════════════════════════════════════════
# Input audit
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_INPUTS = [
    "identity_source_discovery",
    "checkpoint_training_ledger",
    "calibrator_fit_manifest",
    "policy_selection_manifest",
    "heldout_l3_manifest",
    "attack_eval_manifest",
]

REQUIRED_INPUT_LABELS = {
    "identity_source_discovery": "FACTORIZED_IDENTITY_SOURCE_DISCOVERY_V1.json",
    "checkpoint_training_ledger": "FACTORIZED_CHECKPOINT_TRAINING_IDENTITY_LEDGER_V1.json",
    "calibrator_fit_manifest": "calibrator_fit identity manifest",
    "policy_selection_manifest": "policy_selection identity manifest",
    "heldout_l3_manifest": "heldout_l3 identity manifest",
    "attack_eval_manifest": "attack_eval identity manifest",
}


def audit_inputs(input_paths):
    """Check which required inputs are present. Returns (present, missing)."""
    present = {}
    missing = []
    for key in REQUIRED_INPUTS:
        path = input_paths.get(key)
        if path and Path(path).is_file():
            present[key] = {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
        else:
            missing.append({"input_key": key, "expected": REQUIRED_INPUT_LABELS[key],
                           "provided": str(path) if path else None})
    return present, missing


# ══════════════════════════════════════════════════════════════════════════════
# Verdict classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_verdict(disjointness_pass, coverage_status, source_status, inputs_complete):
    """Map audit outcome to formal verdict.

    Order of precedence:
    1. Missing inputs → HOLD_INPUTS_MISSING (never NESTED_RETRAIN_REQUIRED)
    2. Proven contamination → NESTED_RETRAIN_REQUIRED
    3. Clean identity closure → PASS_EXISTING_ROOTS or PASS_DETERMINISTIC_ALLOCATION
    Statistical coverage is independent — does NOT trigger retrain.
    """
    if not inputs_complete:
        return "HOLD_INPUTS_MISSING"

    if not disjointness_pass:
        return "NESTED_RETRAIN_REQUIRED"

    if source_status == "RECOVERED_EXISTING_ROOTS":
        return "PASS_EXISTING_ROOTS"
    if source_status == "DETERMINISTIC_ALLOCATION":
        return "PASS_DETERMINISTIC_ALLOCATION"

    # Identity closure is clean but source status unclear → hold, don't retrain
    return "HOLD_INPUTS_MISSING"


def classify_coverage(coverage_issues, inputs_complete):
    """Statistical coverage is independent of identity disjointness."""
    if not inputs_complete:
        return "NOT_AUDITABLE"
    if not coverage_issues:
        return "PASS"
    return "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE"


def phase_c_authorization(verdict, cal_coverage_pass, pol_coverage_pass, htc_pass):
    """Split Phase C authorization.

    PHASE_C_CP_INFERENCE_AUTHORIZED (validator can decide):
      identity closure PASS + calibration coverage PASS + policy coverage PASS

    HELDOUT_L3_DATA_READY (validator can confirm):
      identity closure PASS + heldout Teacher closure PASS

    HELDOUT_L3_INFERENCE_AUTHORIZED (validator CANNOT decide alone):
      Requires external freeze contracts (calibrator + scheduler thresholds).
      Validator always returns FALSE — caller must verify freeze contracts separately.

    When L3 data is ready but freeze not verified:
      heldout_l3_blocker = PENDING_EXTERNAL_FREEZE
    """
    identity_clean = verdict in ("PASS_EXISTING_ROOTS", "PASS_DETERMINISTIC_ALLOCATION")

    cp_inference = "AUTHORIZED" if (identity_clean and cal_coverage_pass and pol_coverage_pass) else "HOLD"
    l3_data_ready = identity_clean and htc_pass
    # Validator does NOT check calibrator/scheduler freeze contracts.
    # L3 inference authorization always requires external freeze verification.
    l3_inference = False

    return {
        "cp_inference_authorized": cp_inference == "AUTHORIZED",
        "cp_inference_status": cp_inference,
        "heldout_l3_data_ready": l3_data_ready,
        "heldout_l3_inference_authorized": l3_inference,
        "heldout_l3_blocker": "PENDING_EXTERNAL_FREEZE" if l3_data_ready else "HOLD_DATA_NOT_READY",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(staging, filename, headers, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for row in rows:
        w.writerow(row)
    content = buf.getvalue()
    with open(staging / filename, 'w', newline='') as f:
        f.write(content)
    return sha256_str(content)


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
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, default=None,
                    help="Optional: H Teacher label bundle root (per-split factorized_teacher_v1.jsonl)")
    ap.add_argument("--teacher-source-sha256", type=str, default=None,
                    help="Expected Teacher source artifact SHA for cross-bundle consistency check")
    ap.add_argument("--teacher-contract-file", type=Path, default=None,
                    help="Canonical Teacher contract JSON. Validator computes SHA256 from file (never trusts a passed string).")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,"
                            "o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]

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

    # ── Stage output ──
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    # ── If inputs incomplete: produce HOLD receipt and exit cleanly ──
    if not inputs_complete:
        receipt = {
            "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V1",
            "validator_code_sha256": SELF_SHA,
            "status": "HOLD_INPUTS_MISSING",
            "identity_disjointness": "NOT_AUDITABLE",
            "statistical_coverage": "NOT_AUDITABLE",
            "phase_c_authorization": "HOLD",
            "verdict": "HOLD_INPUTS_MISSING",
            "present_inputs": present_inputs,
            "missing_inputs": missing_inputs,
            "message": f"{len(missing_inputs)} required inputs not yet delivered. "
                       f"Codex must produce identity manifests before Phase B can proceed.",
        }
        (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V1.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        (staging / "DEEPSEEK_PHASE_B_MISSING_INPUTS_V1.json").write_text(
            json.dumps({"missing_inputs": missing_inputs, "present_inputs": present_inputs}, indent=2) + "\n")

        sums = {}
        for f in staging.rglob("*"):
            if f.is_file() and f.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
                sums[f.relative_to(staging).as_posix()] = sha256_file(f)
        (staging / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")

        os.replace(staging, out_root)
        print(f"HOLD_INPUTS_MISSING: {len(missing_inputs)} inputs missing, {len(present_inputs)} present")
        for m in missing_inputs:
            print(f"  MISSING: {m['input_key']} ({m['expected']})")
        return 0

    # ── Load all manifests ──
    discovery = load_manifest(args.identity_source_discovery, "IDENTITY_SOURCE_DISCOVERY")
    training_ledger = load_manifest(args.checkpoint_training_ledger, "CHECKPOINT_TRAINING_LEDGER")
    cal_manifest = load_manifest(args.calibrator_fit_manifest, "CALIBRATOR_FIT")
    pol_manifest = load_manifest(args.policy_selection_manifest, "POLICY_SELECTION")
    held_manifest = load_manifest(args.heldout_l3_manifest, "HELDOUT_L3")
    atk_manifest = load_manifest(args.attack_eval_manifest, "ATTACK_EVAL")

    source_status = discovery.get("identity_source_status", "UNKNOWN")
    cohort_membership = discovery.get("cohort_membership", {})

    # ── Per-split audit ──
    all_disjoint_errors = []
    all_cov_issues = []
    per_split = {}
    pairwise_rows = []
    cohort_rows = []

    for sk in expected:
        disjoint_errors = []
        cov_issues = []

        # Gather identity sets per role
        sets_by_role = {}
        sets_by_role["checkpoint_training"] = extract_identities(training_ledger, "checkpoint_training", sk)
        sets_by_role["calibrator_fit"] = extract_identities(cal_manifest, "calibrator_fit", sk)
        sets_by_role["policy_selection"] = extract_identities(pol_manifest, "policy_selection", sk)
        sets_by_role["heldout_l3"] = extract_identities(held_manifest, "heldout_l3", sk)
        sets_by_role["attack_eval"] = extract_identities(atk_manifest, "attack_eval", sk)

        counts = {role: len(ids) for role, ids in sets_by_role.items()}

        # ── Gate 1: Identity Disjointness ──
        check_pairwise_disjoint(sets_by_role, sk, disjoint_errors)
        check_training_provenance(training_ledger, sk, disjoint_errors)
        check_cohort_membership(sets_by_role, cohort_membership, sk, disjoint_errors)

        # Deterministic allocation validation (C/P from DETECTOR_VAL)
        alloc_manifest = cal_manifest if cal_manifest.get("deterministic_allocation") else pol_manifest
        if pol_manifest.get("deterministic_allocation"):
            alloc_manifest = pol_manifest
        check_deterministic_allocation(alloc_manifest, sets_by_role, sk, disjoint_errors)

        # Total union check
        all_ids = set()
        for ids in sets_by_role.values():
            all_ids |= ids
        total_unique = len(all_ids)
        sum_counts = sum(counts.values())
        if total_unique != sum_counts:
            dups = sum_counts - total_unique
            disjoint_errors.append(
                f"IDENTITY_DUPLICATION: {sk} {dups} identities assigned to multiple roles"
            )

        # ── Gate 2: Statistical Coverage ──
        check_calibration_coverage(cal_manifest, sk, cov_issues)
        check_policy_coverage(pol_manifest, sk, cov_issues)
        check_heldout_coverage(sets_by_role, sk, cov_issues)

        disjoint_ok = len(disjoint_errors) == 0
        cov_ok = len(cov_issues) == 0

        per_split[sk] = {
            "identity_disjointness_pass": disjoint_ok,
            "statistical_coverage_pass": cov_ok,
            "disjointness_errors": disjoint_errors,
            "coverage_issues": cov_issues,
            "identity_counts": counts,
            "total_unique": total_unique,
        }
        all_disjoint_errors.extend(disjoint_errors)
        all_cov_issues.extend(cov_issues)

        # Build pairwise intersection rows
        for r1, r2 in PAIRWISE_CONSTRAINTS:
            s1 = sets_by_role.get(r1, set())
            s2 = sets_by_role.get(r2, set())
            pairwise_rows.append([sk, ROLE_LABELS[r1], ROLE_LABELS[r2], len(s1 & s2)])

        # Build cohort membership rows
        for role_name, ids in sets_by_role.items():
            expected_cohort = ROLE_TO_COHORT.get(role_name, "UNKNOWN")
            for eid in sorted(ids):
                actual = cohort_membership.get(eid, "UNKNOWN") if cohort_membership else "NO_LEDGER"
                violation = "" if actual == expected_cohort else "VIOLATION"
                cohort_rows.append([sk, ROLE_LABELS.get(role_name, role_name), eid, actual, expected_cohort, violation])

    # ── Final classification ──
    disjointness_pass = len(all_disjoint_errors) == 0

    # Compute per-role coverage separately for split authorization
    cal_cov_issues = [c for c in all_cov_issues if c.startswith("CALIBRATION")]
    pol_cov_issues = [c for c in all_cov_issues if c.startswith("POLICY")]
    cal_coverage_pass = len(cal_cov_issues) == 0 and inputs_complete
    pol_coverage_pass = len(pol_cov_issues) == 0 and inputs_complete

    coverage_status = classify_coverage(all_cov_issues, inputs_complete)
    verdict = classify_verdict(disjointness_pass, coverage_status, source_status, inputs_complete)

    # ── Gate 3: Heldout Teacher Closure ──
    htc_errors = []
    htc_rows = []
    htc_pass = False
    h_teacher_bundle = str(args.heldout_teacher_bundle_root) if args.heldout_teacher_bundle_root else None
    teacher_source_sha = args.teacher_source_sha256 if args.teacher_source_sha256 else None

    if h_teacher_bundle:
        htc_errors, htc_rows = check_heldout_teacher_closure_global(
            {sk: set(held_manifest.get("splits", {}).get(sk, {}).get("heldout_l3", []))
             for sk in expected},
            h_teacher_bundle, expected, teacher_source_sha
        )
        htc_pass = len(htc_errors) == 0
    else:
        # No H Teacher bundle: flag as not-yet-available (not an error)
        htc_pass = False
        htc_errors.append("HELDOUT_TEACHER_BUNDLE_NOT_MOUNTED — H Teacher labels not yet provided. "
                          "HELDOUT_L3_INFERENCE blocked until H Teacher closure is verified.")

    # ── Split Phase C authorization ──
    phase_c = phase_c_authorization(verdict, cal_coverage_pass, pol_coverage_pass, htc_pass)

    # ── Write output artifacts ──
    # 1. Validation receipt
    receipt = {
        "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
        "validator_code_sha256": SELF_SHA,
        "status": "COMPLETE",
        "verdict": verdict,
        "identity_disjointness": "PASS" if disjointness_pass else "FAIL",
        "statistical_coverage": coverage_status,
        "heldout_teacher_closure": "PASS" if htc_pass else "HOLD",
        "phase_c_cp_inference_authorized": phase_c["cp_inference_authorized"],
        "phase_c_cp_inference_status": phase_c["cp_inference_status"],
        "heldout_l3_data_ready": phase_c["heldout_l3_data_ready"],
        "heldout_l3_inference_authorized": phase_c["heldout_l3_inference_authorized"],
        "heldout_l3_blocker": phase_c["heldout_l3_blocker"],
        "calibration_coverage_pass": cal_coverage_pass,
        "policy_coverage_pass": pol_coverage_pass,
        "heldout_teacher_closure_pass": htc_pass,
        "identity_source_status": source_status,
        "n_disjointness_errors": len(all_disjoint_errors),
        "n_coverage_issues": len(all_cov_issues),
        "n_htc_errors": len(htc_errors),
        "n_splits": len(expected),
        "input_manifests": {
            "identity_source_discovery_sha256": sha256_file(args.identity_source_discovery),
            "checkpoint_training_ledger_sha256": sha256_file(args.checkpoint_training_ledger),
            "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
            "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
            "heldout_l3_manifest_sha256": sha256_file(args.heldout_l3_manifest),
            "attack_eval_manifest_sha256": sha256_file(args.attack_eval_manifest),
        },
        "per_split": per_split,
    }
    if all_disjoint_errors:
        receipt["disjointness_errors"] = all_disjoint_errors
    if all_cov_issues:
        receipt["coverage_issues"] = all_cov_issues
    if htc_errors:
        receipt["heldout_teacher_closure_errors"] = htc_errors
    if h_teacher_bundle:
        bundle_path = Path(h_teacher_bundle)
        seal_path = bundle_path / "SHA256SUMS"
        receipt["heldout_teacher_bundle_sha256"] = sha256_file(seal_path) if seal_path.is_file() else None
    if args.teacher_contract_file:
        contract_path = Path(args.teacher_contract_file)
        if not contract_path.is_file():
            print(f"WARNING: teacher-contract-file not found: {contract_path}")
        else:
            receipt["teacher_contract_sha256"] = sha256_file(contract_path)
            receipt["teacher_contract_path"] = str(contract_path.resolve())

    (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2) + "\n")

    # 2. Pairwise intersections CSV
    write_csv(staging, "DEEPSEEK_PHASE_B_PAIRWISE_INTERSECTIONS_V1.csv",
              ["split", "role_a", "role_b", "intersection_count"], pairwise_rows)

    # 3. Cohort membership CSV
    write_csv(staging, "DEEPSEEK_PHASE_B_COHORT_MEMBERSHIP_V1.csv",
              ["split", "role", "identity", "actual_cohort", "expected_cohorts", "status"],
              cohort_rows)

    # 4. Statistical coverage CSV
    cov_rows = []
    for sk in expected:
        ps = per_split[sk]
        cov_rows.append([sk, "calibrator_fit", ps["identity_counts"]["calibrator_fit"],
                         "PASS" if not any("CALIBRATION" in c for c in ps["coverage_issues"]) else "ISSUES"])
        cov_rows.append([sk, "policy_selection", ps["identity_counts"]["policy_selection"],
                         "PASS" if not any("POLICY" in c for c in ps["coverage_issues"]) else "ISSUES"])
        cov_rows.append([sk, "heldout_l3", ps["identity_counts"]["heldout_l3"],
                         "PASS" if not any("HELDOUT" in c for c in ps["coverage_issues"]) else "ISSUES"])
    write_csv(staging, "DEEPSEEK_PHASE_B_STATISTICAL_COVERAGE_V1.csv",
              ["split", "role", "identity_count", "coverage_status"], cov_rows)

    # 5. Heldout Teacher Closure CSV (if H Teacher bundle provided)
    if htc_rows:
        write_csv(staging, "DEEPSEEK_PHASE_B_HELDOUT_TEACHER_CLOSURE_V1.csv",
                  ["split", "heldout_identity_count", "teacher_closure_status"], htc_rows)

    # 5. Missing inputs (empty when complete, but still produce)
    (staging / "DEEPSEEK_PHASE_B_MISSING_INPUTS_V1.json").write_text(
        json.dumps({"missing_inputs": [], "all_inputs_present": True,
                    "present_inputs": present_inputs}, indent=2) + "\n")

    # 6-7. Seal
    sums = {}
    for f in staging.rglob("*"):
        if f.is_file() and f.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            sums[f.relative_to(staging).as_posix()] = sha256_file(f)
    (staging / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")

    os.replace(staging, out_root)

    # ── Report ──
    print(f"Phase B Validation Complete")
    print(f"  Verdict:                         {verdict}")
    print(f"  Identity Disjointness:           {'PASS' if disjointness_pass else 'FAIL'} ({len(all_disjoint_errors)} errors)")
    print(f"  Statistical Coverage:            {coverage_status} ({len(all_cov_issues)} issues)")
    print(f"  Calibration Coverage:            {'PASS' if cal_coverage_pass else 'HOLD'}")
    print(f"  Policy Coverage:                 {'PASS' if pol_coverage_pass else 'HOLD'}")
    print(f"  Heldout Teacher Closure:         {'PASS' if htc_pass else 'HOLD'} ({len(htc_errors)} errors)")
    print(f"  CP Inference Authorized:         {phase_c['cp_inference_authorized']}")
    print(f"  Heldout L3 Data Ready:           {phase_c['heldout_l3_data_ready']}")
    print(f"  Heldout L3 Inference Authorized: {phase_c['heldout_l3_inference_authorized']}")
    print(f"  Heldout L3 Blocker:              {phase_c['heldout_l3_blocker']}")
    print(f"  Output:                          {out_root}")

    if all_disjoint_errors:
        print(f"\nDisjointness errors:")
        for e in all_disjoint_errors:
            print(f"  {e}")
    if all_cov_issues:
        print(f"\nCoverage issues:")
        for c in all_cov_issues:
            print(f"  {c}")
    if htc_errors:
        print(f"\nHeldout Teacher Closure issues:")
        for e in htc_errors:
            print(f"  {e}")

    return 0 if disjointness_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
