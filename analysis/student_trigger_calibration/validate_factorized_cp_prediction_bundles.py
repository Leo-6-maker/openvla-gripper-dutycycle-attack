#!/usr/bin/env python3
"""Validate C/P Student prediction bundles before calibration or threshold selection.

FAIL-CLOSED: any validation failure causes non-zero exit and no partial
authorization.  This validator runs BEFORE calibrator fitting or threshold
search — it proves the Student predictions are physically separate, identity-
closed, step-closed, schema-valid, checkpoint-bound, and free of Teacher leakage.

Authoritative mode additionally requires a PASS Phase B receipt with
cp_inference_authorized=true.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, sys, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
HEADS = ("grasp", "manipulation", "release")
REQUIRED_PREDICTION_FIELDS = (
    "canonical_parent_key", "step", "split_key",
    "checkpoint_sha256", "checkpoint_source_commit",
    "feature_order_sha256", "normalization_sha256",
    "runtime_source_sha256", "source_artifact_recursive_sha256",
    "source_episode_step_count",
    "grasp_logit", "grasp_probability",
    "manipulation_logit", "manipulation_probability",
    "release_logit", "release_probability",
)
FORBIDDEN_STUDENT_FIELDS = frozenset({
    "strict_k10_feasible", "strict_k10_known_mask",
    "grasp_established", "manipulation_active", "release_or_instability",
    "grasp_established_known_mask", "manipulation_active_known_mask",
    "release_or_instability_known_mask",
    "grasp_target", "manipulation_target", "release_target",
    "event_id", "event_role", "event_duration",
    "object_pose", "contact_pairs", "privileged_task_geometry",
    "attack_outcome", "official_success", "attack_condition",
    "teacher_phase", "teacher_label",
    "grasp_known_mask", "manipulation_known_mask", "release_known_mask",
})
LOGIT_PROB_TOLERANCE = 0.01
SELF_SHA = None


# ── utilities ──────────────────────────────────────────────────────────

def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()

def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)

def is_40char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 40 and all(c in "0123456789abcdef" for c in s)

def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, v))))


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    dups: list[str] = []
    def hook(pairs):
        seen = set(); result = {}
        for k, v in pairs:
            if k in seen: dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE_ERROR: {path} {e}")
    if dups:
        raise SystemExit(f"{label}_DUPLICATE_KEYS: {path} keys={dups[:5]}")
    if not isinstance(value, dict):
        raise SystemExit(f"{label}_NOT_OBJECT: {path}")
    return value


def load_strict_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"{label}_JSONL_MISSING: {path}")
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    with open(path) as f:
        for line_nr, line in enumerate(f, 1):
            if not line.strip(): continue
            dups: list[str] = []
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
            if not isinstance(r, dict):
                raise SystemExit(f"{label}_JSONL_NOT_OBJECT: {path}:{line_nr}")
            ep = r.get("canonical_parent_key")
            step = r.get("step")
            if not isinstance(ep, str) or not ep:
                raise SystemExit(f"{label}_IDENTITY_INVALID: {path}:{line_nr}")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise SystemExit(f"{label}_STEP_INVALID: {path}:{line_nr} step={step!r}")
            key = (ep, step)
            if key in seen_keys:
                raise SystemExit(f"{label}_DUP_EPISODE_STEP: {path}:{line_nr} {key}")
            seen_keys.add(key)
            rows.append(r)
    return rows


def verify_bundle_seal(bundle_root: Path, label: str) -> None:
    """Full seal: SHA256SUMS.sha256 + per-file SHA + no escapes/duplicates/extras."""
    bp = bundle_root.resolve()
    if bp.is_symlink():
        raise SystemExit(f"{label}_ROOT_SYMLINK: {bp}")
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if sums.is_symlink() or sidecar.is_symlink():
        raise SystemExit(f"{label}_SEAL_SYMLINK")
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED")
    expected = sha256_file(sums)
    actual_line = sidecar.read_text().strip().split()
    if not actual_line or actual_line[0] != expected:
        raise SystemExit(f"{label}_SIDECAR_BROKEN")
    listed: set[str] = set()
    with open(sums) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2:
                raise SystemExit(f"{label}_SEAL_PARSE: {line}")
            file_sha, rel = parts[0], " ".join(parts[1:])
            if not is_64char_hex(file_sha):
                raise SystemExit(f"{label}_SEAL_SHA_INVALID: {rel}")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise SystemExit(f"{label}_SEAL_PATH_ESCAPE: {rel}")
            target = bp / rel_path
            if target.is_symlink():
                raise SystemExit(f"{label}_SEAL_SYMLINK: {rel}")
            try:
                target.resolve().relative_to(bp)
            except ValueError:
                raise SystemExit(f"{label}_SEAL_PATH_ESCAPE: {rel}")
            if rel in listed:
                raise SystemExit(f"{label}_SEAL_DUP: {rel}")
            listed.add(rel)
            if not target.is_file() or sha256_file(target) != file_sha:
                raise SystemExit(f"{label}_SEAL_MISMATCH: {rel}")
    for p in bp.rglob("*"):
        if p.is_symlink():
            raise SystemExit(f"{label}_SEAL_SYMLINK: {p.relative_to(bp).as_posix()}")
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            rel = p.relative_to(bp).as_posix()
            if rel not in listed:
                raise SystemExit(f"{label}_SEAL_EXTRA: {rel}")


def extract_manifest_identities(manifest: dict[str, Any], role: str, split_key: str) -> set[str]:
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


# ── validation ─────────────────────────────────────────────────────────

def validate_prediction_schema(rows: list[dict[str, Any]], label: str) -> None:
    """Every row must have all required fields and no forbidden fields."""
    for i, r in enumerate(rows):
        for fld in REQUIRED_PREDICTION_FIELDS:
            if fld not in r:
                raise SystemExit(f"{label}_FIELD_MISSING: row={i} field={fld}")
        extra = FORBIDDEN_STUDENT_FIELDS & set(r)
        if extra:
            raise SystemExit(f"{label}_FORBIDDEN_FIELD: row={i} fields={sorted(extra)}")


def validate_numeric_constraints(rows: list[dict[str, Any]], label: str) -> None:
    """Logits finite, probabilities in [0,1], bool not accepted as number."""
    for i, r in enumerate(rows):
        for head in HEADS:
            logit = r[f"{head}_logit"]
            prob = r[f"{head}_probability"]
            if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
                raise SystemExit(f"{label}_LOGIT_INVALID: row={i} head={head} value={logit!r}")
            if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(float(prob)) or not 0.0 <= float(prob) <= 1.0:
                raise SystemExit(f"{label}_PROB_INVALID: row={i} head={head} value={prob!r}")
            expected = sigmoid(float(logit))
            err = abs(expected - float(prob))
            if err > LOGIT_PROB_TOLERANCE:
                raise SystemExit(f"{label}_LOGIT_PROB_MISMATCH: row={i} head={head} logit={logit} prob={prob} expected={expected:.6f} error={err:.6f}")


def validate_step_closure(rows: list[dict[str, Any]], label: str) -> None:
    """Per-identity: start at 0, contiguous, no gaps, count matches declared."""
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ep[r["canonical_parent_key"]].append(r)
    for ep_id, ep_rows in by_ep.items():
        ep_rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in ep_rows]
        if steps[0] != 0:
            raise SystemExit(f"{label}_STEP_START: {ep_id} first={steps[0]}")
        for expected, observed in enumerate(steps):
            if observed != expected:
                raise SystemExit(f"{label}_STEP_GAP: {ep_id} expected={expected} observed={observed}")
        declared_counts = {r.get("source_episode_step_count") for r in ep_rows}
        if len(declared_counts) != 1:
            raise SystemExit(f"{label}_STEP_COUNT_VARY: {ep_id} counts={declared_counts}")
        declared = next(iter(declared_counts))
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise SystemExit(f"{label}_STEP_COUNT_INVALID: {ep_id} declared={declared!r}")
        if declared != len(ep_rows):
            raise SystemExit(f"{label}_STEP_COUNT_MISMATCH: {ep_id} declared={declared} actual={len(ep_rows)}")


def validate_binding_uniformity(rows: list[dict[str, Any]], label: str) -> dict[str, str]:
    """Each binding field must have exactly one value across all rows."""
    binding_fields = (
        "checkpoint_sha256", "checkpoint_source_commit",
        "feature_order_sha256", "normalization_sha256",
        "runtime_source_sha256", "split_key",
    )
    values_by_field: dict[str, set[str]] = {f: set() for f in binding_fields}
    for r in rows:
        for f in binding_fields:
            v = r.get(f, "")
            if not isinstance(v, str) or not v:
                raise SystemExit(f"{label}_BINDING_MISSING: field={f}")
            values_by_field[f].add(v)
    binding: dict[str, str] = {}
    for f in binding_fields:
        if len(values_by_field[f]) != 1:
            raise SystemExit(f"{label}_BINDING_NONUNIFORM: field={f} values={sorted(values_by_field[f])}")
        binding[f] = next(iter(values_by_field[f]))
    return binding


def validate_identity_closure_list(
    prediction_identities: set[str],
    manifest_identities: set[str],
    role_label: str,
    split_key: str,
) -> None:
    """Prediction identities must exactly match manifest identities."""
    missing = manifest_identities - prediction_identities
    extra = prediction_identities - manifest_identities
    if missing:
        raise SystemExit(f"{role_label}_ID_MISSING: {split_key} n={len(missing)}: {sorted(missing)[:5]}")
    if extra:
        raise SystemExit(f"{role_label}_ID_EXTRA: {split_key} n={len(extra)}: {sorted(extra)[:5]}")


def validate_sha_format(value: str, label: str) -> None:
    if not is_64char_hex(value):
        raise SystemExit(f"{label}_SHA_INVALID: {value[:40]}")


def validate_commit_format(value: str, label: str) -> None:
    if not is_40char_hex(value):
        raise SystemExit(f"{label}_COMMIT_INVALID: {value[:40]}")


def validate_cp_physical_separation(
    c_root: Path, p_root: Path, c_ids: set[str], p_ids: set[str],
) -> None:
    """C and P bundles must be different directories, different seals, disjoint identities."""
    c_resolved = c_root.resolve()
    p_resolved = p_root.resolve()
    if c_resolved == p_resolved:
        raise SystemExit("CP_SAME_DIR: calibration and policy bundle roots must differ")
    c_seal = sha256_file(c_root / "SHA256SUMS")
    p_seal = sha256_file(p_root / "SHA256SUMS")
    if c_seal == p_seal:
        raise SystemExit("CP_SAME_SEAL: calibration and policy bundles have identical SHA256SUMS")
    overlap = c_ids & p_ids
    if overlap:
        raise SystemExit(f"CP_IDENTITY_OVERLAP: n={len(overlap)}: {sorted(overlap)[:5]}")


def validate_checkpoint_binding(
    rows: list[dict[str, Any]], checkpoint_manifest_root: Path, split_key: str, label: str,
) -> str:
    """Verify checkpoint SHA matches actual manifest file content."""
    manifest_path = checkpoint_manifest_root / split_key / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{label}_CHECKPOINT_MANIFEST_MISSING: {manifest_path}")
    cp_manifest = load_strict_json(manifest_path, f"{label}_CHECKPOINT_MANIFEST")
    declared_sha = cp_manifest.get("checkpoint_sha256", "")
    if not is_64char_hex(declared_sha):
        raise SystemExit(f"{label}_CHECKPOINT_MANIFEST_SHA_INVALID: {split_key}")
    declared_sha = declared_sha.lower()
    row_sha = next((r.get("checkpoint_sha256", "").lower() for r in rows), "").lower()
    if row_sha != declared_sha:
        raise SystemExit(f"{label}_CHECKPOINT_SHA_MISMATCH: {split_key} rows={row_sha[:16]} manifest={declared_sha[:16]}")
    return declared_sha


# ── Phase B receipt check ──────────────────────────────────────────────

def validate_phase_b_receipt(receipt_path: Path, authoritative: bool) -> dict[str, Any]:
    receipt = load_strict_json(receipt_path, "PHASE_B_RECEIPT")
    if receipt.get("schema") != "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2":
        raise SystemExit("PHASE_B_RECEIPT_SCHEMA_INVALID")
    if authoritative:
        required_pass = [
            ("cp_inference_authorized", True),
            ("phase_b_data_integrity", "PASS"),
            ("phase_b_scientific_coverage", "PASS"),
            ("k10_contract_parity", "PASS"),
        ]
        for field, expect in required_pass:
            actual = receipt.get(field)
            if actual != expect:
                raise SystemExit(f"PHASE_B_RECEIPT_{field.upper()}: expected {expect!r} got {actual!r}")
        if not receipt.get("calibration_coverage_pass"):
            raise SystemExit("PHASE_B_RECEIPT: calibration_coverage_pass=false")
        if not receipt.get("policy_coverage_pass"):
            raise SystemExit("PHASE_B_RECEIPT: policy_coverage_pass=false")
    return receipt


# ── Cross-role identity disjointness ──────────────────────────────────

def validate_cross_role_disjointness(
    c_ids: set[str], p_ids: set[str],
    t_ids: dict[str, set[str]], h_ids: dict[str, set[str]], a_ids: dict[str, set[str]],
    split_key: str,
) -> None:
    """C and P must be disjoint from T, H, A and from each other (checked earlier)."""
    for other_label, other_ids_dict in [("T", t_ids), ("H", h_ids), ("A", a_ids)]:
        other = other_ids_dict.get(split_key, set())
        for cp_label, cp_ids in [("C", c_ids), ("P", p_ids)]:
            overlap = cp_ids & other
            if overlap:
                raise SystemExit(f"IDENTITY_LEAKAGE: {split_key} {cp_label}∩{other_label}={len(overlap)}")


def validate_inference_not_run(output_root: Path) -> None:
    """Fail-closed: output root must not pre-exist."""
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {output_root}")


# ── output ─────────────────────────────────────────────────────────────

def write_csv_file(staging: Path, filename: str, headers: list[str], rows: list[list[Any]]):
    with open(staging / filename, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for row in rows: w.writerow(row)

def seal_output_dir(root: Path) -> str:
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    content = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    (root / "SHA256SUMS").write_text(content)
    seal = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    return seal


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-receipt", type=Path, required=True)
    ap.add_argument("--calibration-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--policy-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--checkpoint-training-ledger", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, default=None)
    ap.add_argument("--attack-eval-manifest", type=Path, default=None)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    ap.add_argument("--require-cp-ready", action="store_true")
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    authoritative = args.mode == "authoritative"
    out_root = args.output_root.resolve()
    validate_inference_not_run(out_root)

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)
    if authoritative and (len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS):
        raise SystemExit(f"SPLIT_ENFORCEMENT: authoritative mode requires exactly 12 unique frozen splits")

    # 1. Validate Phase B receipt
    receipt = validate_phase_b_receipt(args.phase_b_receipt, authoritative)

    # 2. Load manifests
    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    training_ledger = load_strict_json(args.checkpoint_training_ledger, "TRAINING_LEDGER")
    held_manifest = None
    atk_manifest = None
    if args.heldout_l3_manifest:
        held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")
    if args.attack_eval_manifest:
        atk_manifest = load_strict_json(args.attack_eval_manifest, "ATK_MANIFEST")

    # 3. Physical separation of C and P bundles
    c_root = args.calibration_prediction_bundle_root.resolve()
    p_root = args.policy_prediction_bundle_root.resolve()

    # 4. Verify bundle seals
    verify_bundle_seal(c_root, "C_PREDICTION_BUNDLE")
    verify_bundle_seal(p_root, "P_PREDICTION_BUNDLE")

    # 5. Verify feature-order and normalization contracts exist and are valid
    feature_order = load_strict_json(args.feature_order_contract, "FEATURE_ORDER")
    normalization = load_strict_json(args.normalization_contract, "NORMALIZATION")
    feature_sha = sha256_file(args.feature_order_contract)
    norm_sha = sha256_file(args.normalization_contract)

    # 6. Per-split validation
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    all_errors: list[str] = []
    per_split: dict[str, dict[str, Any]] = {}
    identity_closure_rows: list[list[Any]] = []
    step_closure_rows: list[list[Any]] = []
    binding_audit_rows: list[list[Any]] = []
    cp_predictions_ready = True

    for sk in expected:
        split_errors: list[str] = []

        # Extract identities from manifests
        c_manifest_ids = extract_manifest_identities(cal_manifest, "calibrator_fit", sk)
        p_manifest_ids = extract_manifest_identities(pol_manifest, "policy_selection", sk)
        t_ids = {sk: extract_manifest_identities(training_ledger, "checkpoint_training", sk)}
        h_ids = {sk: extract_manifest_identities(held_manifest, "heldout_l3", sk) if held_manifest else set()}
        a_ids = {sk: extract_manifest_identities(atk_manifest, "attack_eval", sk) if atk_manifest else set()}

        # Load prediction rows
        c_stream = c_root / sk / "predictions.jsonl"
        p_stream = p_root / sk / "predictions.jsonl"
        c_rows = load_strict_jsonl(c_stream, f"C_PRED_{sk}")
        p_rows = load_strict_jsonl(p_stream, f"P_PRED_{sk}")

        # Extract prediction identities
        c_pred_ids = {r["canonical_parent_key"] for r in c_rows}
        p_pred_ids = {r["canonical_parent_key"] for r in p_rows}

        try:
            # Schema validation
            validate_prediction_schema(c_rows, f"C_PRED_{sk}")
            validate_prediction_schema(p_rows, f"P_PRED_{sk}")

            # Numeric constraints
            validate_numeric_constraints(c_rows, f"C_PRED_{sk}")
            validate_numeric_constraints(p_rows, f"P_PRED_{sk}")

            # Step closure
            validate_step_closure(c_rows, f"C_PRED_{sk}")
            validate_step_closure(p_rows, f"P_PRED_{sk}")

            # Binding uniformity
            c_binding = validate_binding_uniformity(c_rows, f"C_PRED_{sk}")
            p_binding = validate_binding_uniformity(p_rows, f"P_PRED_{sk}")

            # Physical separation
            validate_cp_physical_separation(c_root, p_root, c_pred_ids, p_pred_ids)

            # Identity closure vs manifests
            validate_identity_closure_list(c_pred_ids, c_manifest_ids, "CALIBRATION", sk)
            validate_identity_closure_list(p_pred_ids, p_manifest_ids, "POLICY", sk)

            # Cross-role disjointness
            validate_cross_role_disjointness(c_pred_ids, p_pred_ids, t_ids, h_ids, a_ids, sk)

            # Checkpoint binding
            c_cp_sha = validate_checkpoint_binding(c_rows, args.checkpoint_manifest_root, sk, f"C_PRED_{sk}")
            p_cp_sha = validate_checkpoint_binding(p_rows, args.checkpoint_manifest_root, sk, f"P_PRED_{sk}")
            if c_cp_sha != p_cp_sha:
                raise SystemExit(f"CP_CHECKPOINT_DIVERGENT: {sk} C={c_cp_sha[:16]} P={p_cp_sha[:16]}")

            # Binding contract verification
            for binding, label in [(c_binding, "C"), (p_binding, "P")]:
                validate_sha_format(binding["checkpoint_sha256"], f"{label}_BINDING_{sk}")
                validate_commit_format(binding["checkpoint_source_commit"], f"{label}_BINDING_{sk}")
                validate_sha_format(binding["feature_order_sha256"], f"{label}_BINDING_{sk}")
                validate_sha_format(binding["normalization_sha256"], f"{label}_BINDING_{sk}")
                validate_sha_format(binding["runtime_source_sha256"], f"{label}_BINDING_{sk}")

                if binding["feature_order_sha256"] != feature_sha:
                    raise SystemExit(f"{label}_BINDING_FEATURE_SHA_MISMATCH: {sk}")
                if binding["normalization_sha256"] != norm_sha:
                    raise SystemExit(f"{label}_BINDING_NORMALIZATION_SHA_MISMATCH: {sk}")

            # Source SHA uniformity per identity
            for label, rows in [("C", c_rows), ("P", p_rows)]:
                by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for r in rows:
                    by_ep[r["canonical_parent_key"]].append(r)
                for ep_id, ep_rows in by_ep.items():
                    source_shas = {r.get("source_artifact_recursive_sha256") for r in ep_rows}
                    if len(source_shas) != 1:
                        raise SystemExit(f"{label}_PRED_{sk}_SOURCE_SHA_MULTIPLE: {ep_id}")
                    src = next(iter(source_shas))
                    if not is_64char_hex(src):
                        raise SystemExit(f"{label}_PRED_{sk}_SOURCE_SHA_INVALID: {ep_id}")

            identity_closure_rows.append([sk, "C", len(c_pred_ids), len(c_manifest_ids), "PASS" if c_pred_ids == c_manifest_ids else "FAIL"])
            identity_closure_rows.append([sk, "P", len(p_pred_ids), len(p_manifest_ids), "PASS" if p_pred_ids == p_manifest_ids else "FAIL"])
            step_closure_rows.append([sk, "C", len(c_rows), "PASS"])
            step_closure_rows.append([sk, "P", len(p_rows), "PASS"])
            binding_audit_rows.append([sk, "C", c_binding.get("checkpoint_sha256", "")[:16],
                                       c_binding.get("feature_order_sha256", "")[:16], "PASS"])
            binding_audit_rows.append([sk, "P", p_binding.get("checkpoint_sha256", "")[:16],
                                       p_binding.get("feature_order_sha256", "")[:16], "PASS"])

        except SystemExit as e:
            split_errors.append(str(e))
            cp_predictions_ready = False

        per_split[sk] = {
            "errors": split_errors,
            "c_identities": len(c_pred_ids),
            "p_identities": len(p_pred_ids),
            "pass": len(split_errors) == 0,
        }
        all_errors.extend(split_errors)

    # 7. Build receipt
    validation_receipt = {
        "schema": "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1",
        "validator_code_sha256": SELF_SHA,
        "status": "COMPLETE",
        "cp_predictions_ready": cp_predictions_ready,
        "mode": args.mode,
        "phase_b_receipt_sha256": sha256_file(args.phase_b_receipt),
        "cp_inference_authorized": receipt.get("cp_inference_authorized", False),
        "calibration_prediction_bundle_sha256": sha256_file(c_root / "SHA256SUMS"),
        "policy_prediction_bundle_sha256": sha256_file(p_root / "SHA256SUMS"),
        "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
        "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
        "checkpoint_training_ledger_sha256": sha256_file(args.checkpoint_training_ledger),
        "feature_order_contract_sha256": feature_sha,
        "normalization_contract_sha256": norm_sha,
        "n_errors": len(all_errors),
        "n_splits": len(expected),
        "per_split": per_split,
    }
    if held_manifest:
        validation_receipt["heldout_l3_manifest_sha256"] = sha256_file(args.heldout_l3_manifest)
    if atk_manifest:
        validation_receipt["attack_eval_manifest_sha256"] = sha256_file(args.attack_eval_manifest)
    if all_errors:
        validation_receipt["errors"] = all_errors[:50]

    (staging / "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1.json").write_text(
        json.dumps(validation_receipt, indent=2) + "\n")

    write_csv_file(staging, "CP_PREDICTION_IDENTITY_CLOSURE.csv",
                   ["split", "role", "prediction_identity_count", "manifest_identity_count", "closure_status"],
                   identity_closure_rows)
    write_csv_file(staging, "CP_PREDICTION_STEP_CLOSURE.csv",
                   ["split", "role", "total_rows", "step_closure_status"],
                   step_closure_rows)
    write_csv_file(staging, "CP_PREDICTION_BINDING_AUDIT.csv",
                   ["split", "role", "checkpoint_sha16", "feature_order_sha16", "binding_status"],
                   binding_audit_rows)

    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"CP Prediction Validation Complete")
    print(f"  Mode: {args.mode}  Ready: {cp_predictions_ready}")
    print(f"  Errors: {len(all_errors)}")
    for e in all_errors[:10]:
        print(f"    {e}")
    print(f"  Output: {out_root}")

    if not cp_predictions_ready:
        return 1
    if args.require_cp_ready and not cp_predictions_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
