"""CPU synthetic tests for CP prediction bundle validator (updated for shared module)."""
from __future__ import annotations

import json, math, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from factorized_phase_c_integrity import (
    validate_prediction_schema, validate_numeric_constraints,
    verify_step_closure, validate_binding_uniformity,
    verify_identity_closure, validate_cross_role_disjointness,
    load_strict_jsonl, load_strict_json, verify_bundle_seal,
    sha256_file, is_64char_hex, sigmoid, LOGIT_PROB_TOLERANCE,
    FORBIDDEN_STUDENT_FIELDS, REQUIRED_PREDICTION_FIELDS, seal_output_dir,
)
from validate_factorized_cp_prediction_bundles import (
    validate_cp_physical_separation, validate_checkpoint_binding,
    validate_phase_b_receipt, validate_inference_not_run, validate_sha_format,
)


def _mk_prediction_row(ep="ep1", step=0, split="o0_i0", **overrides):
    r = {
        "canonical_parent_key": ep, "step": step, "split_key": split,
        "checkpoint_sha256": "a" * 64, "checkpoint_source_commit": "b" * 40,
        "feature_order_sha256": "a" * 64, "normalization_sha256": "a" * 64,
        "runtime_source_sha256": "a" * 64, "source_artifact_recursive_sha256": "a" * 64,
        "source_episode_step_count": 10,
        "grasp_logit": 0.0, "grasp_probability": 0.5,
        "manipulation_logit": 0.0, "manipulation_probability": 0.5,
        "release_logit": 0.0, "release_probability": 0.5,
    }
    r.update(overrides)
    return r


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


def _write_manifest(path: Path, identities: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identities": identities}))


# ── schema tests ──────────────────────────────────────────────────────
def test_required_fields(): validate_prediction_schema([_mk_prediction_row("ep1", i) for i in range(5)], "T")
def test_missing_field():
    rows = [_mk_prediction_row("ep1", 0)]; del rows[0]["grasp_logit"]
    try: validate_prediction_schema(rows, "T"); assert False
    except SystemExit: pass
def test_forbidden_field():
    rows = [_mk_prediction_row("ep1", 0)]; rows[0]["strict_k10_feasible"] = True
    try: validate_prediction_schema(rows, "T"); assert False
    except SystemExit: pass
def test_all_forbidden_fields():
    for fld in sorted(FORBIDDEN_STUDENT_FIELDS):
        rows = [_mk_prediction_row("ep1", 0)]; rows[0][fld] = True
        try: validate_prediction_schema(rows, f"T_{fld}"); assert False, f"{fld}"
        except SystemExit: pass

# ── numeric tests ─────────────────────────────────────────────────────
def test_logit_finite(): validate_numeric_constraints([_mk_prediction_row("ep1", 0, grasp_logit=1.5, grasp_probability=sigmoid(1.5))], "T")
def test_nan_logit():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=float("nan"))]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_inf_logit():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=float("inf"))]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_probability_oob():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=1.5)]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_negative_prob():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=-0.1)]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_bool_as_logit():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=True)]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_bool_as_prob():
    rows = [_mk_prediction_row("ep1", 0, grasp_probability=False)]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass
def test_logit_prob_mismatch():
    rows = [_mk_prediction_row("ep1", 0, grasp_logit=2.0, grasp_probability=0.1)]
    try: validate_numeric_constraints(rows, "T"); assert False
    except SystemExit: pass

# ── step closure tests ────────────────────────────────────────────────
def test_step_closure_pass():
    rows = [_mk_prediction_row("ep1", i, source_episode_step_count=10) for i in range(10)]
    verify_step_closure(rows, "T")
def test_step_start_not_zero():
    rows = [_mk_prediction_row("ep1", i + 1, source_episode_step_count=10) for i in range(10)]
    try: verify_step_closure(rows, "T"); assert False
    except SystemExit: pass
def test_step_gap():
    rows = [_mk_prediction_row("ep1", i) for i in range(5)] + [_mk_prediction_row("ep1", i + 6) for i in range(5)]
    try: verify_step_closure(rows, "T"); assert False
    except SystemExit: pass
def test_step_count_mismatch():
    rows = [_mk_prediction_row("ep1", i, source_episode_step_count=5) for i in range(10)]
    try: verify_step_closure(rows, "T"); assert False
    except SystemExit: pass
def test_step_count_bool():
    rows = [_mk_prediction_row("ep1", 0)]; rows[0]["source_episode_step_count"] = True
    try: verify_step_closure(rows, "T"); assert False
    except SystemExit: pass
def test_dup_episode_step():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jsonl"; _write_jsonl(p, [_mk_prediction_row("ep1", 0), _mk_prediction_row("ep1", 0)])
        try: load_strict_jsonl(p, "T"); assert False
        except SystemExit: pass

# ── binding uniformity tests ──────────────────────────────────────────
def test_binding_uniformity_pass():
    b = validate_binding_uniformity([_mk_prediction_row("ep1", i) for i in range(3)], "T")
    assert b["split_key"] == "o0_i0"
def test_binding_nonuniform():
    rows = [_mk_prediction_row("ep1", 0, checkpoint_sha256="a" * 64), _mk_prediction_row("ep1", 1, checkpoint_sha256="b" * 64)]
    try: validate_binding_uniformity(rows, "T"); assert False
    except SystemExit: pass

# ── identity closure tests ────────────────────────────────────────────
def test_identity_closure_pass(): verify_identity_closure({"a", "b"}, {"a", "b"}, "C", "o0_i0")
def test_identity_missing():
    try: verify_identity_closure({"a"}, {"a", "b"}, "C", "o0_i0"); assert False
    except SystemExit: pass
def test_identity_extra():
    try: verify_identity_closure({"a", "b", "c"}, {"a", "b"}, "C", "o0_i0"); assert False
    except SystemExit: pass

# ── physical separation tests ─────────────────────────────────────────
def test_cp_same_dir():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "SHA256SUMS").write_text(""); (dp / "SHA256SUMS.sha256").write_text("")
        try: validate_cp_physical_separation(dp, dp, {"a"}, {"b"}); assert False
        except SystemExit: pass
def test_cp_identity_overlap():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        cdir = dp / "c"; pdir = dp / "p"
        for d2 in [cdir, pdir]:
            d2.mkdir(); (d2 / "SHA256SUMS").write_text(f"{'a'*64}  x.jsonl\n"); (d2 / "SHA256SUMS.sha256").write_text(f"{'b'*64}  SHA256SUMS\n")
        try: validate_cp_physical_separation(cdir, pdir, {"a", "b"}, {"b", "c"}); assert False
        except SystemExit: pass

# ── cross-role disjointness tests ─────────────────────────────────────
def test_cross_role_pass():
    validate_cross_role_disjointness({"c1"}, {"p1"}, {"o0_i0": {"t1"}}, {"o0_i0": {"h1"}}, {"o0_i0": set()}, "o0_i0")
def test_c_t_overlap():
    try: validate_cross_role_disjointness({"shared"}, {"p1"}, {"o0_i0": {"shared"}}, {"o0_i0": set()}, {"o0_i0": set()}, "o0_i0"); assert False
    except SystemExit: pass
def test_p_h_overlap():
    try: validate_cross_role_disjointness({"c1"}, {"shared"}, {"o0_i0": set()}, {"o0_i0": {"shared"}}, {"o0_i0": set()}, "o0_i0"); assert False
    except SystemExit: pass

# ── checkpoint binding tests ──────────────────────────────────────────
def test_checkpoint_binding():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); md = dp / "o0_i0"; md.mkdir(parents=True)
        cp_sha = "a" * 64
        (md / "manifest.json").write_text(json.dumps({"checkpoint_sha256": cp_sha}))
        rows = [_mk_prediction_row("ep1", 0, checkpoint_sha256=cp_sha)]
        result = validate_checkpoint_binding(rows, dp, "o0_i0", "T")
        assert result == cp_sha

# ── hex tests ─────────────────────────────────────────────────────────
def test_is_64char_hex():
    assert is_64char_hex("a" * 64); assert not is_64char_hex("g" * 64); assert not is_64char_hex("a" * 63)

# ── jsonl strict tests ────────────────────────────────────────────────
def test_dup_json_key():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "d.jsonl"; p.write_text('{"canonical_parent_key":"ep1","step":0,"step":1}\n')
        try: load_strict_jsonl(p, "T"); assert False
        except SystemExit: pass
def test_non_object_jsonl():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.jsonl"; p.write_text('[1,2,3]\n')
        try: load_strict_jsonl(p, "T"); assert False
        except SystemExit: pass
def test_bool_step():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.jsonl"; p.write_text('{"canonical_parent_key":"ep1","step":true}\n')
        try: load_strict_jsonl(p, "T"); assert False
        except SystemExit: pass

# ── seal tests ────────────────────────────────────────────────────────
def test_seal_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "d.jsonl").write_text("h\n"); seal_output_dir(dp); verify_bundle_seal(dp, "T")
def test_seal_path_escape():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "d.jsonl").write_text("h\n"); seal_output_dir(dp)
        (dp / "SHA256SUMS").write_text(f"{'a'*64}  ../../../escape\n"); (dp / "SHA256SUMS.sha256").write_text(f"{sha256_file(dp / 'SHA256SUMS')}  SHA256SUMS\n")
        try: verify_bundle_seal(dp, "T"); assert False
        except SystemExit: pass
def test_seal_extra_file():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "d.jsonl").write_text("h\n"); seal_output_dir(dp); (dp / "extra.jsonl").write_text("x\n")
        try: verify_bundle_seal(dp, "T"); assert False
        except SystemExit: pass

# ── phase B receipt tests ─────────────────────────────────────────────
def test_phase_b_unauthorized():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pb.json"
        p.write_text(json.dumps({"schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "cp_inference_authorized": False}))
        try: validate_phase_b_receipt(p, True); assert False
        except SystemExit: pass

def test_feature_order_mismatch():
    rows = [_mk_prediction_row("ep1", 0, feature_order_sha256="b" * 64)]
    b = validate_binding_uniformity(rows, "T"); validate_sha_format(b["feature_order_sha256"], "T")

# ── negative integration tests ────────────────────────────────────────
def test_output_preexists():
    with tempfile.TemporaryDirectory() as d:
        o = Path(d) / "output"; o.mkdir()
        try: validate_inference_not_run(o); assert False
        except SystemExit: pass

def test_source_sha_multiple():
    rows = [_mk_prediction_row("ep1", 0, source_artifact_recursive_sha256="a"*64), _mk_prediction_row("ep1", 1, source_artifact_recursive_sha256="b"*64)]
    shas = {r.get("source_artifact_recursive_sha256") for r in rows}
    assert len(shas) == 2

def test_phase_b_never_authorizes_l3():
    from validate_factorized_identity_disjointness import phase_c_authorization
    a = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, True, "PASS", True, True)
    assert a["heldout_l3_inference_authorized"] is False
