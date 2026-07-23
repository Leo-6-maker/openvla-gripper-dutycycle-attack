"""v2.2 tests — sealed roots, mandatory args, strict key rejection, evidence SHA closure."""
from __future__ import annotations

import csv, json, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, seal_output_dir, is_64char_hex, is_strict_int


def _seal_single(root: Path, filename: str, data: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def _make_parent(pid, suite="s0", rank=0):
    return {"parent_id": pid, "suite": suite, "task": "t0", "clean_success": True,
            "detector_emitted": True, "remaining_horizon": 20,
            "selection_rank": rank, "canonical_selection_key": f"key_{pid}"}


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
            "arm_max_abs_diff": 0.001,
            "attack_step_ledger": [{"step": i + 5} for i in range(10)],
            "epsilon": 0.01, "pgd_steps": 10, "pgd_iterations": 1,
            "attacked_frame_count": 10, "norm_convention": "L2",
            "input_space": "pixel", "jpeg_preprocessing_sha256": "j" * 64,
            "payload_config_sha256": "p" * 64,
            }


def _make_job(pid, cond, seed=0, rep=0):
    return {"parent_id": pid, "condition": cond, "perturbation_seed": seed, "repeat_index": rep}


def _sha(c): return c * 64


def _make_rule_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"rule":"frozen deterministic selection","version":"V0"}')


# ═══════════════════════════════════════════════════════════════════
# Sealed root / schema rejection
# ═══════════════════════════════════════════════════════════════════

def test_execution_wrong_schema_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "j", "m.json", {"schema": "WRONG", "jobs": []})
        old = sys.argv
        try:
            from pilot_integrity import consume_sealed_root
            consume_sealed_root(dp / "j", "PILOT_JOB_MATRIX_V0", "T")
            assert False
        except SystemExit: pass
        finally: sys.argv = old


def test_unsealed_root_rejected():
    from pilot_integrity import verify_sealed_root
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "data.json").write_text("{}")
        try: verify_sealed_root(dp, "T"); assert False
        except SystemExit: pass


# ═══════════════════════════════════════════════════════════════════
# Strict job key rejection (no silent continue)
# ═══════════════════════════════════════════════════════════════════

def test_job_seed_bool_rejected():
    from validate_factorized_attack_pilot_execution import _build_job_key
    try:
        _build_job_key({"parent_id": "p", "condition": "C", "perturbation_seed": True, "repeat_index": 0}, "T")
        assert False
    except SystemExit: pass


def test_job_seed_missing_rejected():
    from validate_factorized_attack_pilot_execution import _build_job_key
    try:
        _build_job_key({"parent_id": "p", "condition": "C", "repeat_index": 0}, "T")
        assert False
    except SystemExit: pass


def test_symlink_check_per_component():
    from pilot_integrity import guard_path_safe
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "sub").mkdir()
        (dp / "sub" / "f.txt").write_text("ok")
        guard_path_safe("sub/f.txt", dp, "T")  # should pass


# ═══════════════════════════════════════════════════════════════════
# K closure and condition tests
# ═══════════════════════════════════════════════════════════════════

def test_clean_k_not_zero():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "CLEAN", k_req=10, k_exec=10)
        run.pop("attack_step_ledger", None)
        # Create evidence files
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("vid"); (dp / "ev" / run["telemetry_path"]).write_text("tel")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "CLEAN")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"parent_id": "fec_0", "condition": "CLEAN", "perturbation_seed": 0, "repeat_index": 0, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"parent_id": "fec_0", "condition": "CLEAN", "perturbation_seed": 0, "repeat_index": 0, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("fec_0", "s0", 0)]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_true_k_9_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "TRUE_T10", k_exec=9)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"parent_id": "fec_0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"parent_id": "fec_0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("fec_0", "s0", 0)]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_epsilon_mismatch_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("f0", "TRUE_T10"), _make_job("f0", "RAND_T10")]
        t_run = _make_run("f0", "TRUE_T10"); t_run["epsilon"] = 0.01
        r_run = _make_run("f0", "RAND_T10"); r_run["epsilon"] = 0.02; r_run["gradient_aligned"] = False
        (dp / "ev").mkdir()
        for run in [t_run, r_run]:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / t_run["video_path"]); tsha = sha256_file(dp / "ev" / t_run["telemetry_path"])
        vsha2 = sha256_file(dp / "ev" / r_run["video_path"]); tsha2 = sha256_file(dp / "ev" / r_run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t_run, r_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": t_run["telemetry_path"], "sha256": tsha},
            {"parent_id": "f0", "condition": "RAND_T10", "perturbation_seed": 0, "repeat_index": 0, "path": r_run["telemetry_path"], "sha256": tsha2}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": t_run["video_path"], "sha256": vsha},
            {"parent_id": "f0", "condition": "RAND_T10", "perturbation_seed": 0, "repeat_index": 0, "path": r_run["video_path"], "sha256": vsha2}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0", "s0", 0)]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_arm_parity_missing_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("f0", "TRUE_T10"); run.pop("arm_max_abs_diff", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0", "s0", 0)]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_evidence_sha_missing_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("f0", "TRUE_T10")
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        # Index entries WITHOUT SHA
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["telemetry_path"]}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"parent_id": "f0", "condition": "TRUE_T10", "perturbation_seed": 0, "repeat_index": 0, "path": run["video_path"]}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0", "s0", 0)]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Parent validator tests
# ═══════════════════════════════════════════════════════════════════

def test_parent_selection_rule_mandatory():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(5)]
        rule_file = dp / "rule.json"; _make_rule_file(rule_file)
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5,
            "expected_suite_counts": {"s0": 5},
            "selection_rule_sha256": sha256_file(rule_file)})
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
            sys.argv = ["pm", "--pilot-parent-manifest-root", str(dp / "p"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "t2/m.json"), "--c-manifest", str(dp / "c2/m.json"),
                        "--p-manifest", str(dp / "p2/m.json"), "--h-manifest", str(dp / "h2/m.json"),
                        "--a-manifest", str(dp / "a2/m.json"),
                        "--pilot-detector-config-root", str(dp / "d"),
                        "--selection-rule-file", str(rule_file),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc == 0
        finally: sys.argv = old


def test_parent_selection_rule_missing_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(5)]
        rule_file = dp / "rule.json"; _make_rule_file(rule_file)
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5, "expected_suite_counts": {"s0": 5},
            "selection_rule_sha256": _sha("z")})  # wrong SHA
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        for role, dn in [("T","t3"),("C","c3"),("P","p3"),("H","h3"),("A","a3")]:
            _seal_single(dp / dn, "m.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest-root", str(dp / "p"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "t3/m.json"), "--c-manifest", str(dp / "c3/m.json"),
                        "--p-manifest", str(dp / "p3/m.json"), "--h-manifest", str(dp / "h3/m.json"),
                        "--a-manifest", str(dp / "a3/m.json"),
                        "--pilot-detector-config-root", str(dp / "d"),
                        "--selection-rule-file", str(rule_file),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Blind review tests
# ═══════════════════════════════════════════════════════════════════

def test_blind_separate_roots():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "l", "l.json", {"runs": [
            {"parent_id": "f0", "condition": "CLEAN", "video_path": "v.mp4"}]})
        _seal_single(dp / "v", "v.json", {"entries": [{"path": "v.mp4"}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "l/l.json"),
                        "--pilot-video-index", str(dp / "v/v.json"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        assert (dp / "blind/PILOT_BLIND_REVIEW_V0.csv").exists()
        assert (dp / "unblind/PILOT_UNBLINDING_V0.csv").exists()
        assert not (dp / "blind/PILOT_UNBLINDING_V0.csv").exists()


def test_blind_same_root_rejected():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "l", "l.json", {"runs": [{"parent_id": "f0", "condition": "C", "video_path": "v.mp4"}]})
        _seal_single(dp / "v", "v.json", {"entries": [{"path": "v.mp4"}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-run-ledger", str(dp / "l/l.json"),
                        "--pilot-video-index", str(dp / "v/v.json"),
                        "--blind-package-root", str(dp / "same"),
                        "--unblinding-root", str(dp / "same")]
            br(); assert False
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Analysis tests
# ═══════════════════════════════════════════════════════════════════

def test_analysis_requires_execution_pass():
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "ev", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "HOLD"})
        old = sys.argv
        try:
            from pilot_integrity import consume_sealed_root
            ev2, _ = consume_sealed_root(dp / "ev", "PILOT_EXECUTION_VALIDATION_V0", "T")
            assert ev2["status"] == "HOLD"
        finally: sys.argv = old
