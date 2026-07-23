"""Comprehensive tests for pilot validators — P0-2 through P0-11."""
from __future__ import annotations

import csv, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, seal_output_dir


def _seal_single(root: Path, filename: str, data: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def _make_parent(pid, suite="s0", rank=0, clean=True, emitted=True, horizon=20):
    return {"parent_id": pid, "suite": suite, "task": "t0", "clean_success": clean,
            "detector_emitted": emitted, "remaining_horizon": horizon,
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
            "video_path": f"v_{pid}_{cond}.mp4",
            "telemetry_path": f"t_{pid}_{cond}.json",
            "arm_max_abs_diff": 0.001, "detector_triggered": True}


def _make_job(pid, cond, seed=0, rep=0):
    return {"parent_id": pid, "condition": cond, "perturbation_seed": seed, "repeat_index": rep}


# ═══════════════════════════════════════════════════════════════════
# P0-2: Parent validator tests
# ═══════════════════════════════════════════════════════════════════

def test_parent_suite_quota_pass():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0" if i < 4 else "s1", i) for i in range(8)]
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 8,
            "expected_suite_counts": {"s0": 4, "s1": 4}})
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": "a" * 64, "detector_config_sha256": "b" * 64})
        for role, rdir in [("T","man_t"),("C","man_c"),("P","man_p"),("H","man_h"),("A","man_a")]:
            _seal_single(dp / rdir, "m.json", {"identities": [f"other_{role}_{j}" for j in range(10)]})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "man_t/m.json"), "--c-manifest", str(dp / "man_c/m.json"),
                        "--p-manifest", str(dp / "man_p/m.json"), "--h-manifest", str(dp / "man_h/m.json"),
                        "--a-manifest", str(dp / "man_a/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc == 0
        finally: sys.argv = old


def test_parent_wrong_count_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": [_make_parent(f"fec_{i}", "s0", i) for i in range(5)], "expected_parent_count": 8})
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": "a" * 64, "detector_config_sha256": "b" * 64})
        for role, rdir in [("T","tx"),("C","cx"),("P","px"),("H","hx"),("A","ax")]: _seal_single(dp / rdir, "m.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "tx/m.json"), "--c-manifest", str(dp / "cx/m.json"),
                        "--p-manifest", str(dp / "px/m.json"), "--h-manifest", str(dp / "hx/m.json"),
                        "--a-manifest", str(dp / "ax/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"), "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


def test_selection_rank_duplicate_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", 0 if i < 2 else i) for i in range(5)]
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": parents})
        _seal_single(dp / "f", "m.json", {"identities": [f"fec_{i}" for i in range(16)]})
        _seal_single(dp / "d", "c.json", {"paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": "a" * 64, "detector_config_sha256": "b" * 64})
        for role, rdir in [("T","tx"),("C","cx"),("P","px"),("H","hx"),("A","ax")]: _seal_single(dp / rdir, "m.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "tx/m.json"), "--c-manifest", str(dp / "cx/m.json"),
                        "--p-manifest", str(dp / "px/m.json"), "--h-manifest", str(dp / "hx/m.json"),
                        "--a-manifest", str(dp / "ax/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"), "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


def test_outcome_field_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        p = _make_parent("fec_0", "s0", 0); p["attack_outcome"] = "success"
        _seal_single(dp / "p", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [p]})
        _seal_single(dp / "f", "m.json", {"identities": ["fec_0"]})
        _seal_single(dp / "d", "c.json", {"paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": "a" * 64, "detector_config_sha256": "b" * 64})
        for role, rdir in [("T","tx"),("C","cx"),("P","px"),("H","hx"),("A","ax")]: _seal_single(dp / rdir, "m.json", {"identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest", str(dp / "p/m.json"),
                        "--reserved-fec-manifest", str(dp / "f/m.json"),
                        "--t-manifest", str(dp / "tx/m.json"), "--c-manifest", str(dp / "cx/m.json"),
                        "--p-manifest", str(dp / "px/m.json"), "--h-manifest", str(dp / "hx/m.json"),
                        "--a-manifest", str(dp / "ax/m.json"),
                        "--pilot-detector-config", str(dp / "d/c.json"), "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# P0-3/P0-4/P0-5: Execution validator tests
# ═══════════════════════════════════════════════════════════════════

def test_missing_job_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "CLEAN"), _make_job("fec_0", "TRUE_T10")]
        runs = [_make_run("fec_0", "CLEAN", k_req=0, k_exec=0)]
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"runs": runs})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_clean_k_executed_10_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "CLEAN")]
        runs = [_make_run("fec_0", "CLEAN", k_req=10, k_exec=10)]
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"runs": runs})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_true_k_9_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "TRUE_T10")]
        runs = [_make_run("fec_0", "TRUE_T10", k_exec=9)]
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"runs": runs})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_true_rand_epsilon_mismatch():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "TRUE_T10"), _make_job("fec_0", "RAND_T10")]
        true_run = _make_run("fec_0", "TRUE_T10"); true_run["epsilon"] = 0.01
        rand_run = _make_run("fec_0", "RAND_T10"); rand_run["epsilon"] = 0.02
        rand_run["gradient_aligned"] = False
        runs = [true_run, rand_run]
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"runs": runs})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_checkpoint_divergent_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = [_make_job("fec_0", "CLEAN"), _make_job("fec_0", "TRUE_T10")]
        clean_run = _make_run("fec_0", "CLEAN", k_req=0, k_exec=0)
        clean_run["checkpoint_sha256"] = "a" * 64
        true_run = _make_run("fec_0", "TRUE_T10")
        true_run["checkpoint_sha256"] = "z" * 64
        runs = [clean_run, true_run]
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"runs": runs})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


def test_empty_video_path_fails():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        run = _make_run("fec_0", "TRUE_T10"); run["video_path"] = ""; run["telemetry_path"] = ""
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "TRUE_T10")]})
        _seal_single(dp / "l", "m.json", {"runs": [run]})
        _seal_single(dp / "t", "m.json", {"entries": []})
        _seal_single(dp / "v", "m.json", {"entries": []})
        _seal_single(dp / "pm", "m.json", {"parents": [_make_parent("fec_0", "s0", 0)]})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix", str(dp / "j/m.json"),
                        "--pilot-run-ledger", str(dp / "l/m.json"),
                        "--pilot-telemetry-index", str(dp / "t/m.json"),
                        "--pilot-video-index", str(dp / "v/m.json"),
                        "--pilot-parent-manifest", str(dp / "pm/m.json"),
                        "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# P0-11: Blind review tests
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
                        "--pilot-video-index", str(dp / "v/v.json"), "--output-root", str(dp / "o")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        with open(dp / "o/PILOT_BLIND_REVIEW_V0.csv") as f:
            for row in csv.DictReader(f):
                assert "condition" not in row or not row.get("condition")
        with open(dp / "o/PILOT_UNBLINDING_V0.csv") as f:
            for row in csv.DictReader(f):
                assert "condition" in row


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
                        "--pilot-video-index", str(dp / "v/v.json"), "--output-root", str(dp / "o")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        blind_ids = set()
        with open(dp / "o/PILOT_BLIND_REVIEW_V0.csv") as f:
            for row in csv.DictReader(f):
                assert row["blind_id"] not in blind_ids
                blind_ids.add(row["blind_id"])
