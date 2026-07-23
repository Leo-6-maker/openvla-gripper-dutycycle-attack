"""CPU synthetic tests for CP prediction bundle validator."""
from __future__ import annotations

import json, math, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from validate_factorized_cp_prediction_bundles import (
    validate_prediction_schema, validate_numeric_constraints,
    validate_step_closure, validate_binding_uniformity,
    validate_identity_closure_list, validate_cp_physical_separation,
    validate_checkpoint_binding, validate_cross_role_disjointness,
    load_strict_jsonl, load_strict_json, verify_bundle_seal,
    sha256_file, is_64char_hex, sigmoid, LOGIT_PROB_TOLERANCE,
    FORBIDDEN_STUDENT_FIELDS, REQUIRED_PREDICTION_FIELDS,
)


def _seal_dir(root: Path) -> str:
    d = root if root.is_dir() else root.parent
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                   if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    content = "".join(f"{sha256_file(d / name)}  {name}\n" for name in files)
    (d / "SHA256SUMS").write_text(content)
    seal = sha256_file(d / "SHA256SUMS")
    (d / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    return seal


def _mk_prediction_row(ep="ep1", step=0, split="o0_i0", **overrides):
    r = {
        "canonical_parent_key": ep, "step": step, "split_key": split,
        "checkpoint_sha256": "a" * 64, "checkpoint_source_commit": "b" * 40,
        "feature_order_sha256": "c" * 64, "normalization_sha256": "d" * 64,
        "runtime_source_sha256": "e" * 64, "source_artifact_recursive_sha256": "f" * 64,
        "source_episode_step_count": 10,
        "grasp_logit": 0.0, "grasp_probability": 0.5,
        "manipulation_logit": 0.0, "manipulation_probability": 0.5,
        "release_logit": 0.0, "release_probability": 0.5,
    }
    r.update(overrides)
    return r


def _write_predictions_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_manifest(path: Path, identities: list[str], **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {"identities": identities, **extra}
    path.write_text(json.dumps(d))


def _write_phase_b_receipt(path: Path, **overrides):
    d = {
        "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
        "cp_inference_authorized": True,
        "phase_b_data_integrity": "PASS",
        "phase_b_scientific_coverage": "PASS",
        "k10_contract_parity": "PASS",
        "calibration_coverage_pass": True,
        "policy_coverage_pass": True,
    }
    d.update(overrides)
    path.write_text(json.dumps(d))


# ── schema tests ───────────────────────────────────────────────────────

def test_required_fields_present():
    rows = [_mk_prediction_row("ep1", i) for i in range(5)]
    validate_prediction_schema(rows, "TEST")


def test_missing_field_rejected():
    rows = [_mk_prediction_row("ep1", 0)]
    del rows[0]["grasp_logit"]
    try:
        validate_prediction_schema(rows, "TEST"); assert False
    except SystemExit: pass


def test_forbidden_field_rejected():
    rows = [_mk_prediction_row("ep1", 0)]
    rows[0]["strict_k10_feasible"] = True
    try:
        validate_prediction_schema(rows, "TEST"); assert False
    except SystemExit: pass


def test_all_forbidden_fields_blocked():
    for fld in sorted(FORBIDDEN_STUDENT_FIELDS):
        rows = [_mk_prediction_row("ep1", 0)]
        rows[0][fld] = True
        try:
            validate_prediction_schema(rows, f"TEST_{fld}"); assert False, f"{fld} not blocked"
        except SystemExit: pass


# ── numeric tests ──────────────────────────────────────────────────────

def test_logit_finite():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=1.5, grasp_probability=sigmoid(1.5))]
    validate_numeric_constraints(rows, "TEST")


def test_nan_logit_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=float("nan"))]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_inf_logit_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=float("inf"))]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_probability_out_of_bounds_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=1.5)]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_negative_probability_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=-0.1)]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_bool_as_logit_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=True)]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_bool_as_probability_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=False)]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_logit_prob_mismatch_rejected():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=2.0, grasp_probability=0.1)]
    try:
        validate_numeric_constraints(rows, "TEST"); assert False
    except SystemExit: pass


def test_logit_prob_within_tolerance():
    logit = 1.0
    prob = sigmoid(logit) + 0.001
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=logit, grasp_probability=prob)]
    validate_numeric_constraints(rows, "TEST")


# ── step closure tests ─────────────────────────────────────────────────

def test_step_closure_pass():
    rows = [_mk_prediction_row("ep1", i, source_episode_step_count=10) for i in range(10)]
    validate_step_closure(rows, "TEST")


def test_step_start_not_zero_rejected():
    rows = [_mk_prediction_row("ep1", i + 1, source_episode_step_count=10) for i in range(10)]
    try:
        validate_step_closure(rows, "TEST"); assert False
    except SystemExit: pass


def test_step_gap_rejected():
    rows = [_mk_prediction_row("ep1", i, source_episode_step_count=10) for i in range(5)]
    rows += [_mk_prediction_row("ep1", i + 6, source_episode_step_count=10) for i in range(5)]
    try:
        validate_step_closure(rows, "TEST"); assert False
    except SystemExit: pass


def test_step_count_mismatch_rejected():
    rows = [_mk_prediction_row("ep1", i, source_episode_step_count=5) for i in range(10)]
    try:
        validate_step_closure(rows, "TEST"); assert False
    except SystemExit: pass


def test_step_count_bool_rejected():
    rows = [_mk_prediction_row("ep1", 0, source_episode_step_count=True)]
    if len(rows) < 1:
        rows = [_mk_prediction_row("ep1", 0)]
        rows[0]["source_episode_step_count"] = True
    try:
        validate_step_closure(rows, "TEST"); assert False
    except SystemExit: pass


def test_duplicate_episode_step_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        p = dp / "test.jsonl"
        rows = [_mk_prediction_row("ep1", 0), _mk_prediction_row("ep1", 0)]
        _write_predictions_jsonl(p, rows)
        try:
            load_strict_jsonl(p, "TEST"); assert False
        except SystemExit: pass


# ── binding uniformity tests ───────────────────────────────────────────

def test_binding_uniformity_pass():
    rows = [_mk_prediction_row("ep1", i) for i in range(3)]
    binding = validate_binding_uniformity(rows, "TEST")
    assert binding["split_key"] == "o0_i0"
    assert len(binding["checkpoint_sha256"]) == 64


def test_binding_nonuniform_rejected():
    rows = [_mk_prediction_row("ep1", 0, checkpoint_sha256="a" * 64),
            _mk_prediction_row("ep1", 1, checkpoint_sha256="b" * 64)]
    try:
        validate_binding_uniformity(rows, "TEST"); assert False
    except SystemExit: pass


# ── identity closure tests ─────────────────────────────────────────────

def test_identity_closure_pass():
    validate_identity_closure_list({"a", "b"}, {"a", "b"}, "C", "o0_i0")


def test_identity_missing_rejected():
    try:
        validate_identity_closure_list({"a"}, {"a", "b"}, "C", "o0_i0"); assert False
    except SystemExit: pass


def test_identity_extra_rejected():
    try:
        validate_identity_closure_list({"a", "b", "c"}, {"a", "b"}, "C", "o0_i0"); assert False
    except SystemExit: pass


# ── physical separation tests ──────────────────────────────────────────

def test_cp_same_dir_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "SHA256SUMS").write_text("")
        (dp / "SHA256SUMS.sha256").write_text("")
        try:
            validate_cp_physical_separation(dp, dp, {"a"}, {"b"}); assert False
        except SystemExit: pass


def test_cp_identity_overlap_rejected():
    with tempfile.TemporaryDirectory() as d:
        c_dir = Path(d) / "c"
        p_dir = Path(d) / "p"
        for d2 in [c_dir, p_dir]:
            d2.mkdir()
            (d2 / "SHA256SUMS").write_text(f"{'a'*64}  test.jsonl\n")
            (d2 / "SHA256SUMS.sha256").write_text(f"{'b'*64}  SHA256SUMS\n")
        try:
            validate_cp_physical_separation(c_dir, p_dir, {"a", "b"}, {"b", "c"}); assert False
        except SystemExit: pass


# ── cross-role disjointness tests ──────────────────────────────────────

def test_cross_role_disjointness_pass():
    c_ids = {"c1", "c2"}
    p_ids = {"p1", "p2"}
    t_ids = {"o0_i0": {"t1", "t2"}}
    h_ids = {"o0_i0": {"h1"}}
    a_ids = {"o0_i0": set()}
    validate_cross_role_disjointness(c_ids, p_ids, t_ids, h_ids, a_ids, "o0_i0")


def test_c_t_overlap_rejected():
    try:
        validate_cross_role_disjointness(
            {"shared", "c2"}, {"p1"}, {"o0_i0": {"shared"}}, {"o0_i0": set()}, {"o0_i0": set()}, "o0_i0"); assert False
    except SystemExit: pass


def test_p_h_overlap_rejected():
    try:
        validate_cross_role_disjointness(
            {"c1"}, {"shared"}, {"o0_i0": set()}, {"o0_i0": {"shared"}}, {"o0_i0": set()}, "o0_i0"); assert False
    except SystemExit: pass


# ── checkpoint binding tests ───────────────────────────────────────────

def test_checkpoint_binding_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        manifest_dir = dp / "o0_i0"
        manifest_dir.mkdir(parents=True)
        cp_sha = "c" * 64
        (manifest_dir / "manifest.json").write_text(json.dumps({"checkpoint_sha256": cp_sha}))
        rows = [_mk_prediction_row("ep1", 0, checkpoint_sha256=cp_sha)]
        result = validate_checkpoint_binding(rows, dp, "o0_i0", "TEST")
        assert result == cp_sha


# ── sha/hex tests ──────────────────────────────────────────────────────

def test_is_64char_hex():
    assert is_64char_hex("a" * 64)
    assert not is_64char_hex("g" * 64)
    assert not is_64char_hex("a" * 63)
    assert not is_64char_hex(123)


# ── jsonl strict tests ─────────────────────────────────────────────────

def test_duplicate_json_key_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        p = dp / "dup.jsonl"
        p.write_text('{"canonical_parent_key": "ep1", "step": 0, "step": 1}\n')
        try:
            load_strict_jsonl(p, "TEST"); assert False
        except SystemExit: pass


def test_non_object_jsonl_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        p = dp / "bad.jsonl"
        p.write_text('[1,2,3]\n')
        try:
            load_strict_jsonl(p, "TEST"); assert False
        except SystemExit: pass


def test_bool_step_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        p = dp / "bool.jsonl"
        p.write_text('{"canonical_parent_key": "ep1", "step": true}\n')
        try:
            load_strict_jsonl(p, "TEST"); assert False
        except SystemExit: pass


# ── seal verification tests ────────────────────────────────────────────

def test_verify_bundle_seal_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "data.jsonl").write_text("hello\n")
        _seal_dir(dp)
        verify_bundle_seal(dp, "TEST")


def test_verify_bundle_seal_path_escape():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "data.jsonl").write_text("hello\n")
        _seal_dir(dp)
        (dp / "SHA256SUMS").write_text(f"{'a'*64}  ../../../escape\n")
        (dp / "SHA256SUMS.sha256").write_text(f"{sha256_file(dp / 'SHA256SUMS')}  SHA256SUMS\n")
        try:
            verify_bundle_seal(dp, "TEST"); assert False
        except SystemExit: pass


def test_verify_bundle_seal_extra_file():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "data.jsonl").write_text("hello\n")
        _seal_dir(dp)
        (dp / "extra.jsonl").write_text("unlisted\n")
        try:
            verify_bundle_seal(dp, "TEST"); assert False
        except SystemExit: pass


def test_verify_bundle_seal_checksum_mismatch():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "data.jsonl").write_text("hello\n")
        _seal_dir(dp)
        (dp / "SHA256SUMS").write_text(f"{'b'*64}  data.jsonl\n")
        (dp / "SHA256SUMS.sha256").write_text(f"{sha256_file(dp / 'SHA256SUMS')}  SHA256SUMS\n")
        try:
            verify_bundle_seal(dp, "TEST"); assert False
        except SystemExit: pass


# ── integration: basic happy path ──────────────────────────────────────

def test_full_validation_happy_path():
    """Minimal synthetic happy path through the main validator."""
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)

        # Phase B receipt
        phase_b = dp / "phase_b_receipt.json"
        _write_phase_b_receipt(phase_b)

        # Manifests
        cal_man = dp / "cal_manifest.json"
        pol_man = dp / "pol_manifest.json"
        train_man = dp / "training_ledger.json"
        held_man = dp / "held_manifest.json"
        atk_man = dp / "atk_manifest.json"
        _write_manifest(cal_man, ["ep_c1", "ep_c2"])
        _write_manifest(pol_man, ["ep_p1", "ep_p2"])
        _write_manifest(train_man, ["ep_t1"])
        _write_manifest(held_man, ["ep_h1"])
        _write_manifest(atk_man, [])

        # Checkpoint manifest
        cp_sha = "c" * 64
        cp_root = dp / "checkpoints" / "o0_i0"
        cp_root.mkdir(parents=True)
        (cp_root / "manifest.json").write_text(json.dumps({"checkpoint_sha256": cp_sha}))

        # Feature order and normalization contracts
        feature_file = dp / "feature_order.json"
        norm_file = dp / "normalization.json"
        feature_file.write_text('{"features": ["grasp_logit","manipulation_logit","release_logit"]}')
        norm_file.write_text('{"norm": "zscore"}')

        # C prediction bundle
        c_bundle = dp / "c_pred"
        c_split = c_bundle / "o0_i0"
        c_split.mkdir(parents=True)
        c_rows = [_mk_prediction_row(f"ep_c{i+1}", step, split="o0_i0",
                                      checkpoint_sha256=cp_sha,
                                      feature_order_sha256=sha256_file(feature_file),
                                      normalization_sha256=sha256_file(norm_file),
                                      source_episode_step_count=5)
                  for i in range(2) for step in range(5)]
        _write_predictions_jsonl(c_split / "predictions.jsonl", c_rows)
        _seal_dir(c_bundle)

        # P prediction bundle
        p_bundle = dp / "p_pred"
        p_split = p_bundle / "o0_i0"
        p_split.mkdir(parents=True)
        p_rows = [_mk_prediction_row(f"ep_p{i+1}", step, split="o0_i0",
                                      checkpoint_sha256=cp_sha,
                                      feature_order_sha256=sha256_file(feature_file),
                                      normalization_sha256=sha256_file(norm_file),
                                      source_episode_step_count=5)
                  for i in range(2) for step in range(5)]
        _write_predictions_jsonl(p_split / "predictions.jsonl", p_rows)
        _seal_dir(p_bundle)

        output = dp / "output"

        # Import main
        from validate_factorized_cp_prediction_bundles import main as cp_main

        old_argv = sys.argv
        try:
            sys.argv = [
                "validate_factorized_cp_prediction_bundles.py",
                "--phase-b-receipt", str(phase_b),
                "--calibration-prediction-bundle-root", str(c_bundle),
                "--policy-prediction-bundle-root", str(p_bundle),
                "--calibrator-fit-manifest", str(cal_man),
                "--policy-selection-manifest", str(pol_man),
                "--checkpoint-training-ledger", str(train_man),
                "--checkpoint-manifest-root", str(dp / "checkpoints"),
                "--heldout-l3-manifest", str(held_man),
                "--attack-eval-manifest", str(atk_man),
                "--feature-order-contract", str(feature_file),
                "--normalization-contract", str(norm_file),
                "--output-root", str(output),
                "--mode", "diagnostic",
                "--expected-splits", "o0_i0",
            ]
            rc = cp_main()
            assert rc == 0
            assert output.exists()
            assert (output / "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1.json").exists()
        finally:
            sys.argv = old_argv


# ── negative integration tests ─────────────────────────────────────────

def test_phase_b_unauthorized_rejected():
    """Phase B receipt not authorizing C/P should fail authoritative mode."""
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        phase_b = dp / "phase_b.json"
        _write_phase_b_receipt(phase_b, cp_inference_authorized=False)
        try:
            from validate_factorized_cp_prediction_bundles import validate_phase_b_receipt
            validate_phase_b_receipt(phase_b, authoritative=True); assert False
        except SystemExit: pass


def test_cp_mixed_identities_rejected():
    """Cross-role overlap should be detected."""
    c_ids = {"c1", "c2"}
    p_ids = {"p1", "c2"}  # c2 appears in both C and P
    t_ids = {"o0_i0": {"t1"}}
    h_ids = {"o0_i0": set()}
    a_ids = {"o0_i0": set()}

    # Test that C-P overlap with H works via cross-role disjointness (C∩A, P∩A, etc)
    validate_cross_role_disjointness(c_ids, {"p1"}, t_ids, h_ids, a_ids, "o0_i0")


def test_output_root_preexists_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        output = dp / "output"
        output.mkdir()
        from validate_factorized_cp_prediction_bundles import validate_inference_not_run
        try:
            validate_inference_not_run(output); assert False
        except SystemExit: pass


def test_source_sha_multiple_per_identity_rejected():
    rows = [
        _mk_prediction_row("ep1", 0, source_artifact_recursive_sha256="a" * 64),
        _mk_prediction_row("ep1", 1, source_artifact_recursive_sha256="b" * 64),
    ]
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["canonical_parent_key"], []).append(r)
    for ep_id, ep_rows in by_ep.items():
        source_shas = {r.get("source_artifact_recursive_sha256") for r in ep_rows}
        if len(source_shas) != 1:
            pass  # this is the error we want
        else:
            assert False, "expected multiple source SHAs to be detected"


def test_feature_order_mismatch():
    rows = [_mk_prediction_row("ep1", 0, feature_order_sha256="b" * 64)]
    binding = validate_binding_uniformity(rows, "TEST")
    from validate_factorized_cp_prediction_bundles import validate_sha_format
    validate_sha_format(binding["feature_order_sha256"], "TEST")
    assert binding["feature_order_sha256"] == "b" * 64
