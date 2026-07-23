#!/usr/bin/env python3
"""Phase B identity-disjointness validator — V3.1.

Authoritative mode reads raw C/P/H Teacher bundles and recomputes coverage
from label rows.  Manifest summaries are cross-checked but never trusted.

Three gates:
  Gate 1 — Identity Disjointness (hard): contamination -> NESTED_RETRAIN_REQUIRED
  Gate 2 — Statistical Coverage (soft): computed from raw Teacher labels
  Gate 3 — Heldout Teacher Closure (hard): K10, step closure, contract parity

V3.1 fixes (from review):
  - teacher_contract_sha vs source_artifact_recursive_sha256 separated
  - C/P identity closure enforced (not just H)
  - K10 whitelist (--expected-k10-schema), not blacklist
  - Full SHA256SUMS file-level verification
  - Mandatory deterministic allocation closure in authoritative mode
  - Policy rows sorted by step; JSONL duplicate-key detection; strict step type
  - 12-split duplicate rejection
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
REQUIRED_K10_FIELDS = ("strict_k10_feasible", "strict_k10_known_mask")
REQUIRED_LABEL_FIELDS = ("canonical_parent_key", "step")
EXPECTED_K10_SCHEMA = "R7_K10_OPPORTUNITY_LABELER_V1_2_2_V21C_CANONICAL"
CONTAMINATION_PREFIXES = ("IDENTITY_LEAKAGE:", "IDENTITY_DUPLICATION:")
SELF_SHA = None


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

def is_64char_hex(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)

def load_manifest(path, label):
    if not path.is_file():
        raise SystemExit(f"{label}_MANIFEST_NOT_FOUND: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_MANIFEST_PARSE_ERROR: {e}")

def load_strict_json(path, label):
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

def load_strict_jsonl(path, label):
    """Duplicate-key-aware JSONL loader with strict step type check."""
    if not path.is_file():
        return None
    rows = []
    seen = set()
    with open(path) as f:
        for line_nr, line in enumerate(f, 1):
            if not line.strip(): continue
            dups = []
            def hook(pairs):
                s = set(); r = {}
                for k, v in pairs:
                    if k in s: dups.append(k)
                    s.add(k)
                    r[k] = v
                return r
            try:
                r = json.loads(line, object_pairs_hook=hook)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{label}_JSONL_PARSE: {path}:{line_nr} {e}")
            if dups:
                raise SystemExit(f"{label}_JSONL_DUP_KEY: {path}:{line_nr} keys={dups}")
            for fld in REQUIRED_LABEL_FIELDS:
                if fld not in r:
                    raise SystemExit(f"{label}_JSONL_MISSING: {path}:{line_nr} {fld}")
            ep = r["canonical_parent_key"]
            if not isinstance(ep, str) or not ep:
                raise SystemExit(f"{label}_JSONL_IDENTITY_TYPE: {path}:{line_nr} canonical_parent_key={ep!r}")
            step = r["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                raise SystemExit(f"{label}_JSONL_STEP_TYPE: {path}:{line_nr} step={step!r} type={type(step).__name__}")
            key = (ep, step)
            if key in seen:
                raise SystemExit(f"{label}_JSONL_DUP_KEY: {path}:{line_nr} key={key}")
            seen.add(key)
            r["step"] = step  # normalize
            rows.append(r)
    return rows

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

def load_teacher_labels(bundle_root, split_key, label="TEACHER"):
    """Load per-split Teacher labels with strict duplicate-key JSONL."""
    if not bundle_root:
        return None
    bp = Path(bundle_root) / split_key / "factorized_teacher_v1.jsonl"
    return load_strict_jsonl(bp, label)

def verify_bundle_seal(bundle_root, label):
    """Full seal verification: SHA256SUMS.sha256 + per-file SHA check."""
    bp = Path(bundle_root)
    if bp.is_symlink():
        raise SystemExit(f"{label}_ROOT_SYMLINK: {bp}")
    if not bp.is_dir():
        raise SystemExit(f"{label}_NOT_DIR: {bp}")
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if sums.is_symlink() or sidecar.is_symlink():
        raise SystemExit(f"{label}_SEAL_SYMLINK: seal files must not be symlinks")
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED: missing SHA256SUMS or .sha256")
    # Verify sidecar
    expected_seal = sha256_file(sums)
    actual_seal_line = sidecar.read_text().strip().split()
    if not actual_seal_line or actual_seal_line[0] != expected_seal:
        raise SystemExit(f"{label}_SIDECAR_BROKEN: expected {expected_seal[:16]} got {actual_seal_line[0][:16] if actual_seal_line else '?'}")
    # Parse and verify each file in SHA256SUMS
    seen = set()
    all_listed = set()
    with open(sums) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2:
                raise SystemExit(f"{label}_SEAL_PARSE: bad line '{line}'")
            file_sha, rel = parts[0], " ".join(parts[1:])
            if not is_64char_hex(file_sha):
                raise SystemExit(f"{label}_SEAL_SHA_LEN: {rel}")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise SystemExit(f"{label}_SEAL_PATH_ESCAPE: {rel}")
            target = bp / rel_path
            if target.is_symlink():
                raise SystemExit(f"{label}_SEAL_SYMLINK: {rel}")
            try:
                target.resolve().relative_to(bp.resolve())
            except ValueError:
                raise SystemExit(f"{label}_SEAL_PATH_ESCAPE: {rel}")
            if rel in seen:
                raise SystemExit(f"{label}_SEAL_DUP: {rel}")
            seen.add(rel)
            all_listed.add(rel)
            if not target.is_file():
                raise SystemExit(f"{label}_SEAL_FILE_MISSING: {rel}")
            actual = sha256_file(target)
            if actual != file_sha:
                raise SystemExit(f"{label}_SEAL_FILE_MISMATCH: {rel} expected {file_sha[:16]} got {actual[:16]}")
    # Check no extra files
    for p in bp.rglob("*"):
        if p.is_symlink():
            raise SystemExit(f"{label}_SEAL_SYMLINK: {p.relative_to(bp).as_posix()}")
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            rel = p.relative_to(bp).as_posix()
            if rel not in all_listed:
                raise SystemExit(f"{label}_SEAL_EXTRA_FILE: {rel} not in SHA256SUMS")


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
                errors.append(f"COHORT_UNKNOWN: {split_key} {ROLE_LABELS[role_name]} '{eid}' not in membership")
            elif actual != expected_cohort:
                if expected_cohort == "DETECTOR_VAL" and actual == "DETECTOR_VAL": continue
                errors.append(f"COHORT_VIOLATION: {split_key} {ROLE_LABELS[role_name]} '{eid}' in '{actual}' expected '{expected_cohort}'")


# ══════════════════════════════════════════════════════
# Gate 2 — Statistical Coverage (from raw Teacher labels)
# ══════════════════════════════════════════════════════

def verify_identity_closure(manifest_ids, teacher_rows, role_label, split_key, errors):
    """Verify Teacher bundle identities match manifest exactly."""
    if teacher_rows is None:
        errors.append(f"{role_label}_BUNDLE_MISSING: {split_key}")
        return
    teacher_ids = set(r["canonical_parent_key"] for r in teacher_rows)
    missing = manifest_ids - teacher_ids
    extra = teacher_ids - manifest_ids
    if missing:
        errors.append(f"{role_label}_ID_MISSING: {split_key} {len(missing)}: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"{role_label}_ID_EXTRA: {split_key} {len(extra)}: {sorted(extra)[:5]}")
    if len(teacher_ids) != len(manifest_ids):
        errors.append(f"{role_label}_ID_COUNT: {split_key} manifest={len(manifest_ids)} teacher={len(teacher_ids)}")

def verify_step_closure(teacher_rows, role_label, split_key, errors):
    """Per-identity step closure: start-at-0, contiguous, no gaps."""
    if teacher_rows is None:
        return
    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)
    for ep_id, ep_rows in by_ep.items():
        ep_rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in ep_rows]
        if steps[0] != 0:
            errors.append(f"{role_label}_STEP_START: {split_key}/{ep_id} first={steps[0]} expected 0")
        for i, s in enumerate(steps):
            if s != i:
                errors.append(f"{role_label}_STEP_GAP: {split_key}/{ep_id} expected {i} got {s}")
                break

def check_k10_parity(teacher_rows, expected_k10_schema, role_label, split_key, errors):
    """Whitelist K10 schema check.  authoritative mode requires exact match."""
    if teacher_rows is None or expected_k10_schema is None:
        return
    schemas_seen = set()
    for r in teacher_rows:
        k10_schema = r.get("strict_k10_binding_schema", None)
        if k10_schema is None or not isinstance(k10_schema, str) or k10_schema.strip() == "":
            errors.append(f"{role_label}_K10_MISSING: {split_key} step={r.get('step')} k10_schema={k10_schema!r}")
            return
        schemas_seen.add(k10_schema)
        if k10_schema != expected_k10_schema:
            errors.append(f"{role_label}_K10_MISMATCH: {split_key} step={r.get('step')} got='{k10_schema}' expected='{expected_k10_schema}'")
            return
    if len(schemas_seen) > 1:
        errors.append(f"{role_label}_K10_MULTIPLE: {split_key} schemas={sorted(schemas_seen)}")

def check_contract_sha_consistency(teacher_rows, teacher_contract_sha, role_label, split_key, errors):
    """Verify teacher_contract_sha256 field matches canonical contract file SHA."""
    if teacher_rows is None or teacher_contract_sha is None:
        return
    for r in teacher_rows:
        row_contract_sha = r.get("teacher_contract_sha256", None)
        if row_contract_sha is None or not is_64char_hex(str(row_contract_sha)):
            errors.append(f"{role_label}_CONTRACT_SHA_MISSING: {split_key} step={r.get('step')}")
            return
        if row_contract_sha != teacher_contract_sha:
            errors.append(f"{role_label}_CONTRACT_SHA_MISMATCH: {split_key} step={r.get('step')} got={str(row_contract_sha)[:16]} expected={teacher_contract_sha[:16]}")
            return

def check_source_sha_validity(teacher_rows, role_label, split_key, errors,
                              require_source_step_count=False):
    """Validate per-identity source binding and declared source step closure."""
    if teacher_rows is None:
        return
    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)
        src = r.get("source_artifact_recursive_sha256", "")
        if not is_64char_hex(str(src)):
            errors.append(
                f"{role_label}_SOURCE_SHA_INVALID: {split_key} step={r.get('step')} "
                f"sha={str(src)[:40]}"
            )
            return
    for ep_id, ep_rows in by_ep.items():
        source_shas = {r.get("source_artifact_recursive_sha256") for r in ep_rows}
        if len(source_shas) != 1:
            errors.append(
                f"{role_label}_SOURCE_SHA_MULTIPLE: {split_key}/{ep_id} "
                f"count={len(source_shas)}"
            )
            continue
        if require_source_step_count:
            actual_count = len(ep_rows)
            for r in ep_rows:
                declared_count = r.get("source_episode_step_count")
                if (
                    isinstance(declared_count, bool)
                    or not isinstance(declared_count, int)
                    or declared_count != actual_count
                ):
                    errors.append(
                        f"{role_label}_SOURCE_STEP_COUNT_MISMATCH: {split_key}/{ep_id} "
                        f"declared={declared_count!r} actual={actual_count}"
                    )
                    break


def validate_head_label_types(teacher_rows, role_label, split_key, errors):
    """Require calibration head targets and known masks to be strict booleans."""
    if teacher_rows is None:
        return
    for r in teacher_rows:
        for head in CALIBRATION_HEADS:
            for field in (HEAD_TARGET_MAP[head], HEAD_KNOWN_MAP[head]):
                if not isinstance(r.get(field), bool):
                    errors.append(
                        f"{role_label}_HEAD_FIELD_TYPE: {split_key} step={r.get('step')} "
                        f"{field} is {type(r.get(field)).__name__}, expected bool"
                    )
                    return


def validate_k10_field_types(teacher_rows, role_label, split_key, errors):
    """Require strict-K10 fields to be present booleans."""
    if teacher_rows is None:
        return
    for r in teacher_rows:
        for field in REQUIRED_K10_FIELDS:
            if not isinstance(r.get(field), bool):
                errors.append(
                    f"{role_label}_K10_TYPE: {split_key} step={r.get('step')} "
                    f"{field} is {type(r.get(field)).__name__}, expected bool"
                )
                return


def compute_calibration_coverage_from_labels(teacher_rows, split_key, cov_issues):
    if teacher_rows is None or not teacher_rows:
        cov_issues.append(f"CALIBRATION_BUNDLE_MISSING: {split_key}")
        return
    for head in CALIBRATION_HEADS:
        target_key = HEAD_TARGET_MAP[head]
        known_key = HEAD_KNOWN_MAP[head]
        n_pos = sum(1 for r in teacher_rows if r.get(known_key) and r.get(target_key))
        n_neg = sum(1 for r in teacher_rows if r.get(known_key) and not r.get(target_key))
        if n_pos == 0:
            cov_issues.append(f"CALIBRATION_NO_POSITIVE: {split_key}/{head} 0 known positive")
        if n_neg == 0:
            cov_issues.append(f"CALIBRATION_NO_NEGATIVE: {split_key}/{head} 0 known negative")

def compute_policy_coverage_from_labels(teacher_rows, split_key, cov_issues):
    if teacher_rows is None or not teacher_rows:
        cov_issues.append(f"POLICY_BUNDLE_MISSING: {split_key}")
        return
    # Validate K10 field types
    for fld in REQUIRED_K10_FIELDS:
        for r in teacher_rows:
            if not isinstance(r.get(fld), bool):
                cov_issues.append(f"POLICY_K10_TYPE: {split_key} step={r.get('step')} {fld} is {type(r.get(fld)).__name__}")
                return
    # Sort by step
    by_ep = defaultdict(list)
    for r in teacher_rows:
        by_ep[r["canonical_parent_key"]].append(r)
    n_neg = n_pos_k10 = n_unknown = n_eligible = n_known_denom = 0
    for ep_id, ep_rows in by_ep.items():
        ep_rows.sort(key=lambda r: r["step"])
        T = len(ep_rows)
        if T < 10: n_unknown += 1; continue
        n_eligible += 1
        last_eligible = T - 10
        eligible = ep_rows[:last_eligible + 1]
        known_all = all(r.get("strict_k10_known_mask", False) for r in eligible)
        has_pos = any(r.get("strict_k10_feasible", False) and r.get("strict_k10_known_mask", False) for r in eligible)
        if has_pos: n_pos_k10 += 1; n_known_denom += 1
        elif known_all: n_neg += 1; n_known_denom += 1
        else: n_unknown += 1
    if n_neg == 0:
        cov_issues.append(f"POLICY_NO_NEGATIVE: {split_key} 0 negative episodes")
    if n_pos_k10 == 0:
        cov_issues.append(f"POLICY_NO_OPPORTUNITY: {split_key} 0 K10 opportunities")
    if n_eligible == 0:
        cov_issues.append(f"POLICY_NO_ELIGIBLE: {split_key} 0 eligible episodes")
    if n_known_denom == 0:
        cov_issues.append(f"POLICY_NO_KNOWN_DENOM: {split_key} 0 episodes with known denominator")


# ══════════════════════════════════════════════════════
# Deterministic allocation (V3.1: mandatory in authoritative mode)
# ══════════════════════════════════════════════════════

def check_deterministic_allocation(allocation, sets_by_role, split_key, authoritative, errors,
                                   expected_parent_manifest_sha=None,
                                   expected_algorithm_sha=None,
                                   expected_code_sha=None):
    da = allocation.get("deterministic_allocation", {})
    has_da = bool(da)
    alloc_method = allocation.get("allocation_method", "")
    if authoritative and alloc_method == "DETERMINISTIC_SPLIT" and not has_da:
        errors.append(f"ALLOC_BLOCK_MISSING: {split_key} authoritative DETERMINISTIC_SPLIT requires deterministic_allocation block")
        return
    if not has_da:
        if authoritative and (alloc_method == "DETERMINISTIC_SPLIT" or not alloc_method):
            errors.append(f"ALLOC_BLOCK_MISSING: {split_key}")
        return
    parent = da.get("parent_cohort", "")
    if parent and parent != "DETECTOR_VAL":
        errors.append(f"ALLOC_PARENT: {split_key} expected DETECTOR_VAL, got '{parent}'")
    for field in ["parent_cohort_manifest_sha256", "fixed_salt", "canonical_sort_key",
                   "allocation_algorithm_sha256", "allocation_code_sha256"]:
        val = da.get(field, "")
        if not val or not isinstance(val, str) or len(val) < 8:
            errors.append(f"ALLOC_MISSING: {split_key} deterministic_allocation.{field}")
        elif field.endswith("_sha256") and not is_64char_hex(val):
            errors.append(f"ALLOC_SHA_INVALID: {split_key} {field}")

    declared_parent_sha = da.get("parent_cohort_manifest_sha256", "")
    if authoritative and expected_parent_manifest_sha and declared_parent_sha != expected_parent_manifest_sha:
        errors.append(
            f"ALLOC_PARENT_SHA_MISMATCH: {split_key} "
            f"declared={str(declared_parent_sha)[:16]} actual={expected_parent_manifest_sha[:16]}"
        )
    declared_algorithm_sha = da.get("allocation_algorithm_sha256", "")
    if authoritative and expected_algorithm_sha and declared_algorithm_sha != expected_algorithm_sha:
        errors.append(
            f"ALLOC_ALGORITHM_SHA_MISMATCH: {split_key} "
            f"declared={str(declared_algorithm_sha)[:16]} actual={expected_algorithm_sha[:16]}"
        )
    declared_code_sha = da.get("allocation_code_sha256", "")
    if authoritative and expected_code_sha and declared_code_sha != expected_code_sha:
        errors.append(
            f"ALLOC_CODE_SHA_MISMATCH: {split_key} "
            f"declared={str(declared_code_sha)[:16]} actual={expected_code_sha[:16]}"
        )

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

    if authoritative:
        if not val_split:
            errors.append(f"ALLOC_CLOSURE_MISSING: {split_key} parent_cohort_identities required in authoritative mode")
        else:
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
    if authoritative and un_count > 0:
        errors.append(f"ALLOC_UNASSIGNED: {split_key} {un_count} unassigned VAL identities")


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

def classify_verdict(disjointness_result, source_status, inputs_complete):
    """Only proven cross-role contamination mandates retraining.

    Missing/unverifiable provenance, allocation receipts, seals, or cohort metadata
    remain HOLD_MANIFEST_INCOMPLETE; they are not evidence that retraining is needed.
    """
    if not inputs_complete:
        return "HOLD_INPUTS_MISSING"
    if isinstance(disjointness_result, bool):
        errors = [] if disjointness_result else ["IDENTITY_LEAKAGE: legacy boolean failure"]
    else:
        errors = list(disjointness_result)
    if any(error.startswith(CONTAMINATION_PREFIXES) for error in errors):
        return "NESTED_RETRAIN_REQUIRED"
    if errors:
        return "HOLD_MANIFEST_INCOMPLETE"
    if source_status == "RECOVERED_EXISTING_ROOTS":
        return "PASS_EXISTING_ROOTS"
    if source_status == "DETERMINISTIC_ALLOCATION":
        return "PASS_DETERMINISTIC_ALLOCATION"
    return "HOLD_MANIFEST_INCOMPLETE"

def classify_coverage(coverage_issues, inputs_complete):
    if not inputs_complete: return "NOT_AUDITABLE"
    if not coverage_issues: return "PASS"
    return "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE"

def classify_k10_parity(issues_by_role, authoritative, expected_k10_schema):
    if not authoritative: return "DIAGNOSTIC_ONLY"
    all_issues = []
    for role_issues in issues_by_role.values():
        all_issues.extend(role_issues)
    for iss in all_issues:
        if "K10_MISSING" in iss or "K10_MISMATCH" in iss or "K10_MULTIPLE" in iss:
            return "NOT_AUDITABLE_K10_CONTRACT_MISMATCH"
    return "PASS"

def phase_c_authorization(verdict, cal_pass, pol_pass, htc_pass, k10_pass, authoritative,
                          cp_contract_integrity_pass=True):
    identity_clean = verdict in ("PASS_EXISTING_ROOTS", "PASS_DETERMINISTIC_ALLOCATION")
    cp_inf = "AUTHORIZED" if (
        identity_clean and cal_pass and pol_pass and cp_contract_integrity_pass
        and (not authoritative or k10_pass == "PASS")
    ) else "HOLD"
    l3_ready = identity_clean and htc_pass and (not authoritative or k10_pass == "PASS")
    return {"cp_inference_authorized": cp_inf == "AUTHORIZED", "cp_inference_status": cp_inf,
            "heldout_l3_data_ready": l3_ready, "heldout_l3_inference_authorized": False,
            "heldout_l3_blocker": "PENDING_EXTERNAL_FREEZE" if l3_ready else "HOLD_DATA_NOT_READY",
            "k10_contract_parity": k10_pass}


# ══════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════

def write_csv(staging, filename, headers, rows):
    with open(staging / filename, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for row in rows: w.writerow(row)

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
    ap.add_argument("--deterministic-allocation-receipt", type=Path, default=None)
    ap.add_argument("--parent-cohort-manifest", type=Path, default=None)
    ap.add_argument("--allocation-algorithm-file", type=Path, default=None)
    ap.add_argument("--allocation-code-file", type=Path, default=None)
    ap.add_argument("--expected-k10-schema", type=str, default=None,
                    help=f"Required K10 schema for authoritative mode (default: {EXPECTED_K10_SCHEMA})")
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    ap.add_argument("--require-cp-authorization", action="store_true")
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

    # 12-split enforcement: reject duplicates and wrong set
    if authoritative:
        if len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS:
            raise SystemExit(f"AUTHORITATIVE_SPLIT_ENFORCEMENT: requires exactly 12 unique frozen splits, got {len(expected)} items, {len(expected_set)} unique")
    if authoritative and not args.teacher_contract_file:
        raise SystemExit("AUTHORITATIVE_MODE requires --teacher-contract-file")

    # Teacher contract SHA — computed from file
    teacher_contract_sha = sha256_file(args.teacher_contract_file) if args.teacher_contract_file else None
    if authoritative and args.expected_k10_schema not in (None, EXPECTED_K10_SCHEMA):
        raise SystemExit(
            f"AUTHORITATIVE_K10_SCHEMA_FROZEN: expected {EXPECTED_K10_SCHEMA}, "
            f"got {args.expected_k10_schema}"
        )
    expected_k10 = EXPECTED_K10_SCHEMA if authoritative else args.expected_k10_schema

    teacher_contract = load_strict_json(args.teacher_contract_file, "TEACHER_CONTRACT") if args.teacher_contract_file else None
    if authoritative:
        declared_k10 = teacher_contract.get(
            "k10_schema", teacher_contract.get("strict_k10_binding_schema")
        )
        if declared_k10 != EXPECTED_K10_SCHEMA:
            raise SystemExit(
                f"TEACHER_CONTRACT_K10_MISMATCH: expected {EXPECTED_K10_SCHEMA}, got {declared_k10!r}"
            )

    # Input audit
    input_paths = {k: getattr(args, k.replace("-", "_") + ("_manifest" if k.endswith("_manifest") else ""), None)
                   for k in ["identity_source_discovery", "checkpoint_training_ledger",
                             "calibrator_fit_manifest", "policy_selection_manifest",
                             "heldout_l3_manifest", "attack_eval_manifest"]}
    # Fix mapping
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

    # HOLD on missing inputs
    if not inputs_complete:
        receipt = {"schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "validator_code_sha256": SELF_SHA,
                   "status": "HOLD_INPUTS_MISSING", "verdict": "HOLD_INPUTS_MISSING",
                   "present_inputs": present_inputs, "missing_inputs": missing_inputs,
                   "cp_inference_authorized": False, "heldout_l3_data_ready": False,
                   "heldout_l3_inference_authorized": False}
        (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2.json").write_text(json.dumps(receipt, indent=2) + "\n")
        seal_dir(staging); os.replace(staging, out_root)
        for m in missing_inputs: print(f"  MISSING: {m['input_key']} ({m['expected']})")
        return 0

    # Load manifests with strict JSON
    discovery = load_strict_json(args.identity_source_discovery, "SOURCE_DISCOVERY")
    training_ledger = load_strict_json(args.checkpoint_training_ledger, "TRAINING_LEDGER")
    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")
    atk_manifest = load_strict_json(args.attack_eval_manifest, "ATK_MANIFEST")

    # Verify Teacher bundle seals (full file-level)
    if authoritative:
        for bundle_root, label in [
            (args.calibration_teacher_bundle_root, "CAL_TEACHER"),
            (args.policy_teacher_bundle_root, "POL_TEACHER"),
            (args.heldout_teacher_bundle_root, "HELD_TEACHER"),
        ]:
            if bundle_root:
                verify_bundle_seal(bundle_root, label)

    source_status = discovery.get("identity_source_status", "UNKNOWN")
    cohort_membership = discovery.get("cohort_membership", {})

    allocation_receipt = None
    parent_cohort_manifest_sha = None
    allocation_algorithm_sha = None
    allocation_code_sha = None
    if source_status == "DETERMINISTIC_ALLOCATION":
        if args.deterministic_allocation_receipt and args.deterministic_allocation_receipt.is_file():
            allocation_receipt = load_strict_json(
                args.deterministic_allocation_receipt, "DETERMINISTIC_ALLOCATION_RECEIPT"
            )
        if args.parent_cohort_manifest and args.parent_cohort_manifest.is_file():
            parent_cohort_manifest_sha = sha256_file(args.parent_cohort_manifest)
        if args.allocation_algorithm_file and args.allocation_algorithm_file.is_file():
            allocation_algorithm_sha = sha256_file(args.allocation_algorithm_file)
        if args.allocation_code_file and args.allocation_code_file.is_file():
            allocation_code_sha = sha256_file(args.allocation_code_file)
    elif args.deterministic_allocation_receipt and args.deterministic_allocation_receipt.is_file():
        allocation_receipt = load_strict_json(
            args.deterministic_allocation_receipt, "DETERMINISTIC_ALLOCATION_RECEIPT"
        )
        if args.parent_cohort_manifest and args.parent_cohort_manifest.is_file():
            parent_cohort_manifest_sha = sha256_file(args.parent_cohort_manifest)
        if args.allocation_algorithm_file and args.allocation_algorithm_file.is_file():
            allocation_algorithm_sha = sha256_file(args.allocation_algorithm_file)
        if args.allocation_code_file and args.allocation_code_file.is_file():
            allocation_code_sha = sha256_file(args.allocation_code_file)

    # Per-split audit
    all_disjoint_errors = []; all_cov_issues = []; all_htc_errors = []
    all_cc_errors = defaultdict(list)  # per-role contract errors
    per_split = {}; pairwise_rows = []; cohort_rows = []

    for sk in expected:
        disjoint_errors = []; cov_issues = []; htc_local = []
        cc_local = defaultdict(list)  # contract errors per role

        sets_by_role = {}
        sets_by_role["checkpoint_training"] = extract_identities(training_ledger, "checkpoint_training", sk)
        sets_by_role["calibrator_fit"] = extract_identities(cal_manifest, "calibrator_fit", sk)
        sets_by_role["policy_selection"] = extract_identities(pol_manifest, "policy_selection", sk)
        sets_by_role["heldout_l3"] = extract_identities(held_manifest, "heldout_l3", sk)
        sets_by_role["attack_eval"] = extract_identities(atk_manifest, "attack_eval", sk)
        counts = {role: len(ids) for role, ids in sets_by_role.items()}

        # Empty-set guards
        for role, ids in sets_by_role.items():
            if len(ids) == 0:
                disjoint_errors.append(f"EMPTY_ROLE: {sk} {role} has 0 identities")

        # Gate 1
        check_pairwise_disjoint(sets_by_role, sk, disjoint_errors)
        check_training_provenance(training_ledger, sk, disjoint_errors)
        check_cohort_membership(sets_by_role, cohort_membership, sk, disjoint_errors)
        allocation_input = allocation_receipt if allocation_receipt is not None else cal_manifest
        if authoritative and source_status == "DETERMINISTIC_ALLOCATION":
            if allocation_receipt is None:
                disjoint_errors.append(f"ALLOC_RECEIPT_MISSING: {sk}")
            if parent_cohort_manifest_sha is None:
                disjoint_errors.append(f"ALLOC_PARENT_MANIFEST_MISSING: {sk}")
            if allocation_algorithm_sha is None:
                disjoint_errors.append(f"ALLOC_ALGORITHM_FILE_MISSING: {sk}")
            if allocation_code_sha is None:
                disjoint_errors.append(f"ALLOC_CODE_FILE_MISSING: {sk}")
        check_deterministic_allocation(
            allocation_input, sets_by_role, sk, authoritative, disjoint_errors,
            expected_parent_manifest_sha=parent_cohort_manifest_sha,
            expected_algorithm_sha=allocation_algorithm_sha,
            expected_code_sha=allocation_code_sha,
        )

        all_ids = set()
        for ids in sets_by_role.values(): all_ids |= ids
        total_unique = len(all_ids)
        if total_unique != sum(counts.values()):
            disjoint_errors.append(f"IDENTITY_DUPLICATION: {sk} {sum(counts.values()) - total_unique} duplicates")

        # Gate 2: Load Teacher labels and verify identity closure for C, P, H
        cal_rows = load_teacher_labels(args.calibration_teacher_bundle_root, sk, "CAL_LABELS") if args.calibration_teacher_bundle_root else None
        pol_rows = load_teacher_labels(args.policy_teacher_bundle_root, sk, "POL_LABELS") if args.policy_teacher_bundle_root else None
        h_rows = load_teacher_labels(args.heldout_teacher_bundle_root, sk, "HELD_LABELS") if args.heldout_teacher_bundle_root else None

        # C identity closure + contract checks
        verify_identity_closure(sets_by_role["calibrator_fit"], cal_rows, "CALIBRATION", sk, cov_issues)
        verify_step_closure(cal_rows, "CALIBRATION", sk, cov_issues)
        check_k10_parity(cal_rows, expected_k10, "CALIBRATION", sk, cc_local["calibration"])
        check_contract_sha_consistency(cal_rows, teacher_contract_sha, "CALIBRATION", sk, cc_local["calibration"])
        check_source_sha_validity(
            cal_rows, "CALIBRATION", sk, cc_local["calibration"],
            require_source_step_count=authoritative,
        )
        validate_head_label_types(cal_rows, "CALIBRATION", sk, cov_issues)
        compute_calibration_coverage_from_labels(cal_rows, sk, cov_issues)

        # P identity closure + contract checks
        verify_identity_closure(sets_by_role["policy_selection"], pol_rows, "POLICY", sk, cov_issues)
        verify_step_closure(pol_rows, "POLICY", sk, cov_issues)
        check_k10_parity(pol_rows, expected_k10, "POLICY", sk, cc_local["policy"])
        check_contract_sha_consistency(pol_rows, teacher_contract_sha, "POLICY", sk, cc_local["policy"])
        check_source_sha_validity(
            pol_rows, "POLICY", sk, cc_local["policy"],
            require_source_step_count=authoritative,
        )
        validate_k10_field_types(pol_rows, "POLICY", sk, cov_issues)
        compute_policy_coverage_from_labels(pol_rows, sk, cov_issues)

        # H identity closure + contract checks + step closure
        verify_identity_closure(sets_by_role["heldout_l3"], h_rows, "HELDOUT", sk, htc_local)
        verify_step_closure(h_rows, "HELDOUT", sk, htc_local)
        check_k10_parity(h_rows, expected_k10, "HELDOUT", sk, cc_local["heldout"])
        check_contract_sha_consistency(h_rows, teacher_contract_sha, "HELDOUT", sk, cc_local["heldout"])
        check_source_sha_validity(
            h_rows, "HELDOUT", sk, cc_local["heldout"],
            require_source_step_count=authoritative,
        )
        validate_k10_field_types(h_rows, "HELDOUT", sk, htc_local)

        # H K10 denominator
        if h_rows:
            by_ep = defaultdict(list)
            for r in h_rows: by_ep[r["canonical_parent_key"]].append(r)
            for ep_id in sorted(sets_by_role["heldout_l3"] & set(r["canonical_parent_key"] for r in h_rows)):
                ep_rows = sorted(by_ep[ep_id], key=lambda r: r["step"])
                T = len(ep_rows)
                if T >= 10:
                    eligible = ep_rows[:T - 9]
                    if not any(r.get("strict_k10_known_mask", False) for r in eligible):
                        htc_local.append(f"HELDOUT_K10_DENOM_EMPTY: {sk}/{ep_id}")
                else:
                    htc_local.append(f"HELDOUT_TOO_SHORT: {sk}/{ep_id} T={T}<10")

        # Cross-check manifest summaries against recomputed values
        if authoritative and cal_rows:
            manifest_sums = cal_manifest.get("calibration_head_summaries", {}).get(sk, {})
            if manifest_sums:
                for head in CALIBRATION_HEADS:
                    target_key = HEAD_TARGET_MAP[head]
                    known_key = HEAD_KNOWN_MAP[head]
                    actual_pos = sum(1 for r in cal_rows if r.get(known_key) and r.get(target_key))
                    summary_pos = manifest_sums.get(head, {}).get("known_positive", -1)
                    if summary_pos != -1 and actual_pos != summary_pos:
                        cov_issues.append(f"CALIBRATION_SUMMARY_MISMATCH: {sk}/{head} manifest={summary_pos} computed={actual_pos}")

        disjoint_ok = len(disjoint_errors) == 0; cov_ok = len(cov_issues) == 0; htc_ok = len(htc_local) == 0
        per_split[sk] = {"identity_disjointness_pass": disjoint_ok, "statistical_coverage_pass": cov_ok,
                         "heldout_teacher_closure_pass": htc_ok,
                         "disjointness_errors": disjoint_errors, "coverage_issues": cov_issues,
                         "htc_errors": htc_local, "contract_errors": dict(cc_local),
                         "identity_counts": counts, "total_unique": total_unique}
        all_disjoint_errors.extend(disjoint_errors); all_cov_issues.extend(cov_issues); all_htc_errors.extend(htc_local)
        for role_key in cc_local:
            all_cc_errors[role_key].extend(cc_local[role_key])

        for r1, r2 in PAIRWISE_CONSTRAINTS:
            s1, s2 = sets_by_role.get(r1, set()), sets_by_role.get(r2, set())
            pairwise_rows.append([sk, ROLE_LABELS[r1], ROLE_LABELS[r2], len(s1 & s2)])
        for role_name, ids in sets_by_role.items():
            ec = ROLE_TO_COHORT.get(role_name, "UNKNOWN")
            for eid in sorted(ids):
                actual = cohort_membership.get(eid, "UNKNOWN") if cohort_membership else "NO_LEDGER"
                cohort_rows.append([sk, ROLE_LABELS.get(role_name, role_name), eid, actual, ec, "" if actual == ec else "VIOLATION"])

    # Final classification
    disjointness_pass = len(all_disjoint_errors) == 0
    cal_cov_issues = [c for c in all_cov_issues if c.startswith("CALIBRATION")]
    pol_cov_issues = [c for c in all_cov_issues if c.startswith("POLICY")]
    cal_coverage_pass = len(cal_cov_issues) == 0
    pol_coverage_pass = len(pol_cov_issues) == 0
    htc_pass = len(all_htc_errors) == 0 and not any(all_cc_errors.get("heldout", []))
    cal_contract_integrity_pass = not any(all_cc_errors.get("calibration", []))
    pol_contract_integrity_pass = not any(all_cc_errors.get("policy", []))
    cp_contract_integrity_pass = cal_contract_integrity_pass and pol_contract_integrity_pass
    contract_integrity_pass = cp_contract_integrity_pass and not any(all_cc_errors.get("heldout", []))
    k10_pass = classify_k10_parity(dict(all_cc_errors), authoritative, expected_k10)
    coverage_status = classify_coverage(all_cov_issues, inputs_complete)
    verdict = classify_verdict(all_disjoint_errors, source_status, inputs_complete)
    phase_c = phase_c_authorization(
        verdict, cal_coverage_pass, pol_coverage_pass, htc_pass, k10_pass,
        authoritative, cp_contract_integrity_pass
    )

    overall_data_integrity = disjointness_pass and htc_pass and contract_integrity_pass
    overall_scientific = cal_coverage_pass and pol_coverage_pass and (k10_pass == "PASS" or not authoritative)
    phase_b_overall = "PASS" if (overall_data_integrity and overall_scientific) else "HOLD"

    # Receipt
    receipt = {"schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "validator_code_sha256": SELF_SHA,
               "status": "COMPLETE", "verdict": verdict,
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
               "calibration_contract_integrity_pass": cal_contract_integrity_pass,
               "policy_contract_integrity_pass": pol_contract_integrity_pass,
               "teacher_contract_integrity_pass": contract_integrity_pass,
               "identity_source_status": source_status, "mode": args.mode,
               "n_disjointness_errors": len(all_disjoint_errors), "n_coverage_issues": len(all_cov_issues),
               "n_htc_errors": len(all_htc_errors), "n_splits": len(expected),
               "input_manifests": {"identity_source_discovery_sha256": sha256_file(args.identity_source_discovery),
                   "checkpoint_training_ledger_sha256": sha256_file(args.checkpoint_training_ledger),
                   "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
                   "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
                   "heldout_l3_manifest_sha256": sha256_file(args.heldout_l3_manifest),
                   "attack_eval_manifest_sha256": sha256_file(args.attack_eval_manifest)},
               "per_split": per_split}
    if args.deterministic_allocation_receipt and args.deterministic_allocation_receipt.is_file():
        receipt["deterministic_allocation_receipt_sha256"] = sha256_file(
            args.deterministic_allocation_receipt
        )
    if args.parent_cohort_manifest and args.parent_cohort_manifest.is_file():
        receipt["parent_cohort_manifest_sha256"] = sha256_file(args.parent_cohort_manifest)
    if args.allocation_algorithm_file and args.allocation_algorithm_file.is_file():
        receipt["allocation_algorithm_file_sha256"] = sha256_file(args.allocation_algorithm_file)
    if args.allocation_code_file and args.allocation_code_file.is_file():
        receipt["allocation_code_file_sha256"] = sha256_file(args.allocation_code_file)
    if teacher_contract_sha: receipt["teacher_contract_sha256"] = teacher_contract_sha
    if expected_k10: receipt["expected_k10_schema"] = expected_k10
    if args.calibration_teacher_bundle_root:
        receipt["calibration_teacher_bundle_sha256"] = sha256_file(Path(args.calibration_teacher_bundle_root) / "SHA256SUMS")
    if args.policy_teacher_bundle_root:
        receipt["policy_teacher_bundle_sha256"] = sha256_file(Path(args.policy_teacher_bundle_root) / "SHA256SUMS")
    if args.heldout_teacher_bundle_root:
        htb = Path(args.heldout_teacher_bundle_root)
        receipt["heldout_teacher_bundle_sha256"] = sha256_file(htb / "SHA256SUMS") if (htb / "SHA256SUMS").is_file() else None
    if all_disjoint_errors: receipt["disjointness_errors"] = all_disjoint_errors
    if all_cov_issues: receipt["coverage_issues"] = all_cov_issues
    if all_htc_errors: receipt["heldout_teacher_closure_errors"] = all_htc_errors
    if any(all_cc_errors.values()): receipt["teacher_contract_errors"] = dict(all_cc_errors)

    (staging / "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2.json").write_text(json.dumps(receipt, indent=2) + "\n")

    write_csv(staging, "DEEPSEEK_PHASE_B_PAIRWISE_INTERSECTIONS_V1.csv", ["split","role_a","role_b","intersection_count"], pairwise_rows)
    write_csv(staging, "DEEPSEEK_PHASE_B_COHORT_MEMBERSHIP_V1.csv", ["split","role","identity","actual_cohort","expected_cohorts","status"], cohort_rows)
    cov_rows = []
    for sk in expected:
        ps = per_split[sk]
        for role in ["calibrator_fit","policy_selection","heldout_l3"]:
            role_issues = ps.get("coverage_issues", []) if role != "heldout_l3" else ps.get("htc_errors", [])
            prefix = "CALIBRATION" if role == "calibrator_fit" else ("POLICY" if role == "policy_selection" else "HELDOUT")
            status = "PASS" if not any(i.startswith(prefix) for i in role_issues) else "ISSUES"
            cov_rows.append([sk, role, ps["identity_counts"].get(role, 0), status])
    write_csv(staging, "DEEPSEEK_PHASE_B_STATISTICAL_COVERAGE_V1.csv", ["split","role","identity_count","coverage_status"], cov_rows)
    write_csv(staging, "DEEPSEEK_PHASE_B_HELDOUT_TEACHER_CLOSURE_V1.csv", ["split","heldout_identity_count","teacher_closure_status"],
              [[sk, per_split[sk]["identity_counts"].get("heldout_l3",0), "PASS" if len(per_split[sk].get("htc_errors",[]))==0 else "FAIL"] for sk in expected])

    seal_dir(staging); os.replace(staging, out_root)

    print(f"Phase B V3.1 Validation Complete")
    print(f"  Mode: {args.mode}  Verdict: {verdict}")
    print(f"  Identity Disjointness: {'PASS' if disjointness_pass else 'FAIL'} ({len(all_disjoint_errors)} errors)")
    print(f"  Statistical Coverage:  {coverage_status} ({len(all_cov_issues)} issues)")
    print(f"  Calibration Coverage:  {'PASS' if cal_coverage_pass else 'HOLD'}")
    print(f"  Policy Coverage:       {'PASS' if pol_coverage_pass else 'HOLD'}")
    print(f"  H Teacher Closure:     {'PASS' if htc_pass else 'HOLD'} ({len(all_htc_errors)} errors)")
    print(f"  Teacher Contract:      {'PASS' if contract_integrity_pass else 'HOLD'}")
    print(f"  K10 Contract Parity:   {k10_pass}")
    print(f"  Phase B Data Integrity:    {'PASS' if overall_data_integrity else 'HOLD'}")
    print(f"  Phase B Scientific:        {'PASS' if overall_scientific else 'HOLD'}")
    print(f"  Phase B Overall:           {phase_b_overall}")
    print(f"  CP Inference:          {phase_c['cp_inference_authorized']}")
    print(f"  L3 Data Ready:         {phase_c['heldout_l3_data_ready']}")
    print(f"  L3 Inference:          {phase_c['heldout_l3_inference_authorized']}")
    print(f"  Output: {out_root}")
    if all_disjoint_errors:
        print(f"\nDisjointness errors:"); [print(f"  {e}") for e in all_disjoint_errors[:10]]
    if all_cov_issues:
        print(f"\nCoverage issues:"); [print(f"  {c}") for c in all_cov_issues[:10]]
    if all_htc_errors:
        print(f"\nHTC errors:"); [print(f"  {e}") for e in all_htc_errors[:10]]

    if args.require_cp_authorization and not phase_c["cp_inference_authorized"]:
        return 2
    if not disjointness_pass: return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
