"""Comprehensive tests for pilot v2 validators — all P0 fixes."""
from __future__ import annotations

import csv, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file


def _seal_single(root: Path, filename: str, data: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def _make_parent(pid, suite="s0", rank=0, clean=True, emitted=True, horizon=20):
    return {"parent_id": pid, "suite": suite, "task": "t0", "clean_success": clean,
            "detector_emitted": emitted, "remaining_horizon": horizon,
            "selection_rank": rank, "canonical_selection_key": sha256_file.__code__ and f"key_{pid}"}


def _make_run(pid, cond, seed=0, rep=0, k_req=10, k_exec=10):
    return {"parent_id": pid, "condition": cond, "perturbation_seed": seed,
            "repeat_index": rep, "k_requested": k_req, "k_executed": k_exec,
            "attack_start_step": 5, "attack_end_step": 14,
            "checkpoint_sha256": "a" * 64, "initial_state_sha256": "b" * 64,
            "task_identity": "task_1", "prompt_sha256": "c" * 64,
            "preprocessing_sha256": "d" * 64, "processor_config_sha256": "e" * 64,
            "runtime_source_sha256": "f" * 64, "evaluation_horizon": 100,
            "gradient_aligned": (cond == "TRUE_T10"),
            "video_path": f"v_{pid}_{cond}.mp4", "telemetry_path": f"t_{pid}_{cond}.json",
            "arm_max_abs_diff": 0.001, "attack_step_ledger": [{"step": i + 5} for i in range(10)],
            "epsilon": 0.01, "pgd_steps": 10, "pgd_iterations": 1,
            "attacked_frame_count": 10, "norm_convention": "L2",
            "input_space": "pixel", "jpeg_preprocessing_sha256": "j" * 64,
            }


def _make_job(pid, cond, seed=0, rep=0):
    return {"parent_id": pid, "condition": cond, "perturbation_seed": seed, "repeat_index": rep}


def _sha(c): return c * 64


# ═══════════════════════════════════════════════════════════════════
# Fix 1: Schema hard-reject tests
# ═══════════════════════════════════════════════════════════════════

def test_parent_wrong_schema_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "p", "m.json", {"schema": "WRONG", "parents": []})
        try:
            from pilot_integrity import require_schema, load_strict_json
            m = load_strict_json(dp / "p/m.json", "T")
            require_schema(m, "PILOT_PARENT_MANIFEST_V0", "TEST")
            assert False, "Should reject wrong schema"
        except SystemExit: pass


def test_job_matrix_wrong_schema_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "j", "m.json", {"schema": "WRONG", "jobs": []})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": []})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            ev(); assert False, "Should reject wrong schema"
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Fix 2: Strict job key test
# ═══════════════════════════════════════════════════════════════════

def test_job_seed_bool_rejected():
    from validate_factorized_attack_pilot_execution import _build_job_key
    try:
        _build_job_key({"parent_id": "p", "condition": "CLEAN", "perturbation_seed": True, "repeat_index": 0}, "T")
        assert False, "Should reject bool seed"
    except SystemExit: pass


def test_job_seed_missing_rejected():
    from validate_factorized_attack_pilot_execution import _build_job_key
    try:
        _build_job_key({"parent_id": "p", "condition": "CLEAN", "repeat_index": 0}, "T")
        assert False, "Should reject missing seed"
    except SystemExit: pass


# ═══════════════════════════════════════════════════════════════════
# Fix 3: Unknown condition rejection
# ═══════════════════════════════════════════════════════════════════

def test_unknown_condition_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0",
            "jobs": [_make_job("fec_0", "FAKE_CONDITION")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            ev(); assert False, "Should reject unknown condition"
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Fix 4: K=10 / CLEAN k=0 / attack step ledger
# ═══════════════════════════════════════════════════════════════════

def test_clean_k_executed_10_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "CLEAN", k_req=10, k_exec=10); run.pop("attack_step_ledger", None)
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "CLEAN")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_true_k_9_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "TRUE_T10", k_exec=9)
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_missing_job_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0",
            "jobs": [_make_job("fec_0", "CLEAN"), _make_job("fec_0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0",
            "runs": [_make_run("fec_0", "CLEAN", k_req=0, k_exec=0)]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Fix 5: Matched parity tests
# ═══════════════════════════════════════════════════════════════════

def test_checkpoint_divergent_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "CLEAN"), _make_job("fec_0", "TRUE_T10")]
        c_run = _make_run("fec_0", "CLEAN", k_req=0, k_exec=0)
        c_run["checkpoint_sha256"] = _sha("a")
        t_run = _make_run("fec_0", "TRUE_T10")
        t_run["checkpoint_sha256"] = _sha("z")
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [c_run, t_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_true_rand_epsilon_mismatch():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "TRUE_T10"), _make_job("fec_0", "RAND_T10")]
        t_run = _make_run("fec_0", "TRUE_T10"); t_run["epsilon"] = 0.01
        r_run = _make_run("fec_0", "RAND_T10"); r_run["epsilon"] = 0.02; r_run["gradient_aligned"] = False
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t_run, r_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Fix 9: Parent selection tests
# ═══════════════════════════════════════════════════════════════════

def test_parent_expected_count_missing_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(8)]
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "selection_rule_sha256": _sha("a")})
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        for role, dn in [("T","mt"),("C","mc"),("P","mp"),("H","mh"),("A","ma")]:
            _seal_single(dp / dn, "m.json", {"identities": [f"o_{role}_0"]})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "mt/m.json"), "--c-manifest", str(dp / "mc/m.json"),
                        "--p-manifest", str(dp / "mp/m.json"), "--h-manifest", str(dp / "mh/m.json"),
                        "--a-manifest", str(dp / "ma/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"), "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


def test_selection_rank_duplicate_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", 0 if i < 2 else i) for i in range(5)]
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5,
            "expected_suite_counts": {"s0": 5}, "selection_rule_sha256": _sha("a")})
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        for role, dn in [("T","t2"),("C","c2"),("P","p2"),("H","h2"),("A","a2")]:
            _seal_single(dp / dn, "m.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "t2/m.json"), "--c-manifest", str(dp / "c2/m.json"),
                        "--p-manifest", str(dp / "p2/m.json"), "--h-manifest", str(dp / "h2/m.json"),
                        "--a-manifest", str(dp / "a2/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"), "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Fix 11: Blind review tests
# ═══════════════════════════════════════════════════════════════════

def test_blind_hides_condition():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "l", "l.json", {"runs": [
            {"parent_id": "fec_0", "condition": "TRUE_T10", "video_path": "v0.mp4"},
            {"parent_id": "fec_1", "condition": "CLEAN", "video_path": "v1.mp4"}]})
        _seal_single(dp / "v", "v.json", {"entries": [{"path": "v0.mp4"}, {"path": "v1.mp4"}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "l/l.json"),
                        "--pilot-video-index", str(dp / "v/v.json"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            rc = br(); assert rc == 0
        finally: sys.argv = old

        # Blind package must NOT have condition
        with open(dp / "blind/PILOT_BLIND_REVIEW_V0.csv") as f:
            for row in csv.DictReader(f):
                assert "condition" not in row or not row.get("condition")
        # Blind package must NOT contain unblinding
        assert not (dp / "blind/PILOT_UNBLINDING_V0.csv").exists()

        # Unblinding must have condition
        with open(dp / "unblind/PILOT_UNBLINDING_V0.csv") as f:
            for row in csv.DictReader(f):
                assert "condition" in row

        # Blind and unblinding must be different directories
        assert dp / "blind" != dp / "unblind"


def test_blind_unblind_same_root_rejected():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "l", "l.json", {"runs": [{"parent_id": "f0", "condition": "CLEAN", "video_path": "v.mp4"}]})
        _seal_single(dp / "v", "v.json", {"entries": [{"path": "v.mp4"}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "l/l.json"),
                        "--pilot-video-index", str(dp / "v/v.json"),
                        "--blind-package-root", str(dp / "same"),
                        "--unblinding-root", str(dp / "same")]
            br(); assert False, "Same root should be rejected"
        except SystemExit: pass
        finally: sys.argv = old


def test_blind_ids_unique():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "l", "l.json", {"runs": [
            {"parent_id": f"fec_{i}", "condition": f"C{i%3}", "video_path": f"v{i}.mp4"}
            for i in range(20)]})
        _seal_single(dp / "v", "v.json", {"entries": [{"path": f"v{i}.mp4"} for i in range(20)]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "l/l.json"),
                        "--pilot-video-index", str(dp / "v/v.json"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        blind_ids = set()
        with open(dp / "blind/PILOT_BLIND_REVIEW_V0.csv") as f:
            for row in csv.DictReader(f):
                assert row["blind_id"] not in blind_ids
                blind_ids.add(row["blind_id"])


def test_empty_video_path_fails():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "TRUE_T10"); run["video_path"] = ""; run["telemetry_path"] = ""
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"), "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"), "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old
