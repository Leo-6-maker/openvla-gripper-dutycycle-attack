"""v2.3.1 tests — cross-artifact binding, duplicate rejection, strict rules, reachable decision tree, blind seal binding."""
from __future__ import annotations

import csv, json, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))

from pilot_integrity import sha256_file, seal_output_dir, is_64char_hex, is_strict_int, is_finite_number


def _seal_single(root: Path, filename: str, data: dict):
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def _sha(c): return c * 64


def _make_job(pid, cond, seed=0, rep=0, jid=None, mgid=None):
    return {"job_id": jid or f"j_{pid}_{cond}_{seed}_{rep}",
            "matched_group_id": mgid or f"g_{pid}_{rep}",
            "parent_id": pid, "condition": cond,
            "perturbation_seed": seed, "repeat_index": rep}


def _make_parent(pid, suite="s0", rank=0, csk=None):
    from validate_factorized_pilot_parent_manifest import _compute_canonical_key
    item = {"parent_id": pid, "suite": suite, "task": "t0", "clean_success": True,
            "detector_emitted": True, "remaining_horizon": 20,
            "selection_rank": rank}
    item["canonical_selection_key"] = csk or _compute_canonical_key(item)
    return item


def _make_run(pid, cond, jid=None, mgid=None, seed=0, rep=0, k_req=10, k_exec=10):
    return {"job_id": jid or f"j_{pid}_{cond}_{seed}_{rep}",
            "matched_group_id": mgid or f"g_{pid}_{rep}",
            "parent_id": pid, "condition": cond, "perturbation_seed": seed,
            "repeat_index": rep, "k_requested": k_req, "k_executed": k_exec,
            "attack_start_step": 5, "attack_end_step": 14,
            "attack_requested": (cond != "CLEAN"),
            "checkpoint_sha256": "a" * 64, "initial_state_sha256": "b" * 64,
            "task_identity": "task_1", "prompt_sha256": "c" * 64,
            "preprocessing_sha256": "d" * 64, "processor_config_sha256": "e" * 64,
            "runtime_source_sha256": "f" * 64, "evaluation_horizon": 100,
            "gradient_aligned": (cond == "TRUE_T10"),
            "payload_matches_TRUE": (cond == "RANDOM_TIME_T10"),
            "oracle_type": ("command_intervention" if cond == "COMMAND_OPEN_ORACLE" else None),
            "video_path": f"v_{pid}_{cond}.mp4", "telemetry_path": f"t_{pid}_{cond}.json",
            "arm_max_abs_diff": 0.001,
            "attack_step_ledger": [{"step": i + 5, "armed": True, "executed": True} for i in range(10)],
            "epsilon": 0.01, "pgd_steps": 10, "pgd_iterations": 1,
            "attacked_frame_count": 10, "norm_convention": "L2",
            "input_space": "pixel", "jpeg_preprocessing_sha256": "j" * 64,
            "payload_config_sha256": "p" * 64,
            }


def _make_rule_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"rule":"frozen deterministic selection","version":"V0"}')


def _make_go_rules(root: Path, **overrides):
    """Write a sealed GO/NO-GO rules root with all required fields."""
    root.mkdir(parents=True, exist_ok=True)
    rules_data = {"min_valid_pairs": 1, "min_oracle_physical_parents": 1,
                  "min_true_over_rand_parents": 1, "min_true_over_random_time_parents": 1,
                  "max_missing_evidence": 0, "require_all_conditions_per_group": True}
    rules_data.update(overrides)
    rules = {"schema": "PILOT_GO_NO_GO_RULES_V0", "rules": rules_data}
    (root / "go_rules.json").write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


# ═══════════════════════════════════════════════════════════════════
# Sealed root / schema / integrity primitives
# ═══════════════════════════════════════════════════════════════════

def test_wrong_schema_rejected():
    from pilot_integrity import consume_sealed_root
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "j", "m.json", {"schema": "WRONG", "jobs": []})
        try:
            consume_sealed_root(dp / "j", "PILOT_JOB_MATRIX_V0", "T"); assert False
        except SystemExit: pass


def test_unsealed_root_rejected():
    from pilot_integrity import verify_sealed_root
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "data.json").write_text("{}")
        try: verify_sealed_root(dp, "T"); assert False
        except SystemExit: pass


def test_is_finite_number():
    assert is_finite_number(0) is True
    assert is_finite_number(0.5) is True
    assert is_finite_number(-1) is True
    assert is_finite_number(True) is False
    assert is_finite_number(False) is False
    assert is_finite_number(float("nan")) is False
    assert is_finite_number(float("inf")) is False
    assert is_finite_number(float("-inf")) is False
    assert is_finite_number("0") is False
    assert is_finite_number(None) is False


# ═══════════════════════════════════════════════════════════════════
# Job ID / matched_group_id mandatory
# ═══════════════════════════════════════════════════════════════════

def test_job_id_missing_rejected():
    from validate_factorized_attack_pilot_execution import _validate_entry_id
    try:
        _validate_entry_id({"matched_group_id": "g0"}, "T"); assert False
    except SystemExit: pass


def test_matched_group_id_missing_rejected():
    from validate_factorized_attack_pilot_execution import _validate_entry_id
    try:
        _validate_entry_id({"job_id": "j0"}, "T"); assert False
    except SystemExit: pass


def test_job_matrix_dup_job_id_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "ev").mkdir()
        jobs = [_make_job("f0", "CLEAN", jid="j0", mgid="g0"),
                _make_job("f0", "TRUE_T10", jid="j0", mgid="g0")]  # DUP
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j")]
            ev(); assert False
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Exact video/telemetry index closure
# ═══════════════════════════════════════════════════════════════════

def test_extra_video_index_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j_f0_CLEAN_0_0"; mgid = "g_f0_0"
        job = _make_job("f0", "CLEAN", jid=jid, mgid=mgid)
        run = _make_run("f0", "CLEAN", jid=jid, mgid=mgid, k_req=0, k_exec=0)
        run.pop("attack_step_ledger", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [job]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha},
            {"job_id": "EXTRA_JOB", "matched_group_id": "g_extra", "path": "x.mp4", "sha256": _sha("x")}]})
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


def test_extra_telemetry_index_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j_f0_CLEAN_0_0"; mgid = "g_f0_0"
        job = _make_job("f0", "CLEAN", jid=jid, mgid=mgid)
        run = _make_run("f0", "CLEAN", jid=jid, mgid=mgid, k_req=0, k_exec=0)
        run.pop("attack_step_ledger", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [job]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha},
            {"job_id": "EXTRA_T", "matched_group_id": "g_extra", "path": "x.json", "sha256": _sha("x")}]})
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
# Cross-artifact immutable field binding
# ═══════════════════════════════════════════════════════════════════

def test_run_matched_group_divergence_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid_matrix = "g_matrix"; mgid_run = "g_different"
        job = _make_job("f0", "CLEAN", jid=jid, mgid=mgid_matrix)
        run = _make_run("f0", "CLEAN", jid=jid, mgid=mgid_run, k_req=0, k_exec=0)
        run["attack_requested"] = False; run.pop("attack_step_ledger", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [job]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid_matrix, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid_matrix, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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
# Duplicate condition in same matched_group
# ═══════════════════════════════════════════════════════════════════

def test_duplicate_condition_in_group_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", seed=0, jid="j_t0", mgid=mgid),
                _make_job("f0", "TRUE_T10", seed=1, jid="j_t1", mgid=mgid)]  # DUP condition in same group
        t0 = _make_run("f0", "TRUE_T10", jid="j_t0", mgid=mgid, seed=0)
        t1 = _make_run("f0", "TRUE_T10", jid="j_t1", mgid=mgid, seed=1)
        (dp / "ev").mkdir()
        for run in [t0, t1]:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha0 = sha256_file(dp / "ev" / t0["video_path"]); tsha0 = sha256_file(dp / "ev" / t0["telemetry_path"])
        vsha1 = sha256_file(dp / "ev" / t1["video_path"]); tsha1 = sha256_file(dp / "ev" / t1["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t0, t1]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": "j_t0", "matched_group_id": mgid, "path": t0["telemetry_path"], "sha256": tsha0},
            {"job_id": "j_t1", "matched_group_id": mgid, "path": t1["telemetry_path"], "sha256": tsha1}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j_t0", "matched_group_id": mgid, "path": t0["video_path"], "sha256": vsha0},
            {"job_id": "j_t1", "matched_group_id": mgid, "path": t1["video_path"], "sha256": vsha1}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0  # must reject duplicate condition in same group
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# attack_requested strict bool
# ═══════════════════════════════════════════════════════════════════

def test_attack_requested_missing_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid)
        run.pop("attack_requested", None)  # missing entirely
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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


def test_clean_attack_requested_wrong():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "CLEAN", jid=jid, mgid=mgid, k_req=0, k_exec=0)
        run["attack_requested"] = True  # WRONG for CLEAN
        run.pop("attack_step_ledger", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "CLEAN", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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
# NaN/Inf rejection
# ═══════════════════════════════════════════════════════════════════

def test_nan_tolerance_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "ev").mkdir(); (dp / "ev" / "v.mp4").write_text("v"); (dp / "ev" / "t.json").write_text("t")
        jid = "j0"; mgid = "g0"
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "CLEAN", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": float("nan")})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            ev(); assert False
        except SystemExit: pass
        finally: sys.argv = old


def test_nan_arm_deviation_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid); run["arm_max_abs_diff"] = float("nan")
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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


def test_negative_tolerance_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "ev").mkdir(); (dp / "ev" / "v.mp4").write_text("v"); (dp / "ev" / "t.json").write_text("t")
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "CLEAN", jid="j0", mgid="g0")]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": -0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            ev(); assert False
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Condition-specific contract rejections
# ═══════════════════════════════════════════════════════════════════

def test_rand_gradient_aligned_true_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "RAND_T10", jid=jid, mgid=mgid); run["gradient_aligned"] = True
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "RAND_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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


def test_random_time_payload_not_matching_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "RANDOM_TIME_T10", jid=jid, mgid=mgid)
        run["payload_matches_TRUE"] = False
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "RANDOM_TIME_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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


def test_oracle_not_command_intervention_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid=jid, mgid=mgid)
        run["oracle_type"] = "visual_attack"
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "COMMAND_OPEN_ORACLE", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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
# Attack-step ledger validation
# ═══════════════════════════════════════════════════════════════════

def test_attack_step_not_executed_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid)
        run["attack_step_ledger"] = [{"step": 5, "armed": True, "executed": True}] * 9 + [{"step": 14, "armed": True, "executed": False}]
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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


def test_attack_end_exceeds_horizon_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid)
        run["evaluation_horizon"] = 10; run["attack_end_step"] = 10
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
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
# Multi-seed matched_group pairing
# ═══════════════════════════════════════════════════════════════════

def test_multi_seed_same_matched_group():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", seed=0, jid="j_t0", mgid=mgid),
                _make_job("f0", "RAND_T10", seed=0, jid="j_r0", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t0", mgid=mgid)
        r_run = _make_run("f0", "RAND_T10", jid="j_r0", mgid=mgid); r_run["gradient_aligned"] = False
        (dp / "ev").mkdir()
        for run in [t_run, r_run]:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha_t = sha256_file(dp / "ev" / t_run["video_path"]); tsha_t = sha256_file(dp / "ev" / t_run["telemetry_path"])
        vsha_r = sha256_file(dp / "ev" / r_run["video_path"]); tsha_r = sha256_file(dp / "ev" / r_run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t_run, r_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": "j_t0", "matched_group_id": mgid, "path": t_run["telemetry_path"], "sha256": tsha_t},
            {"job_id": "j_r0", "matched_group_id": mgid, "path": r_run["telemetry_path"], "sha256": tsha_r}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j_t0", "matched_group_id": mgid, "path": t_run["video_path"], "sha256": vsha_t},
            {"job_id": "j_r0", "matched_group_id": mgid, "path": r_run["video_path"], "sha256": vsha_r}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc == 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Canonical key hard fail
# ═══════════════════════════════════════════════════════════════════

def test_canonical_key_mismatch_rejected():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(5)]
        parents[0]["canonical_selection_key"] = "WRONG_KEY_00000000"
        rule_file = dp / "rule.json"; _make_rule_file(rule_file)
        rule_sha = sha256_file(rule_file)
        _seal_single(dp / "par", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5,
            "expected_suite_counts": {"s0": 5}, "selection_rule_sha256": rule_sha})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        _seal_single(dp / "fec", "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": [f"fec_{i}" for i in range(16)]})
        for role, dn in [("T","mi_t"),("C","mi_c"),("P","mi_p"),("H","mi_h"),("A","mi_a")]:
            _seal_single(dp / dn, "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest-root", str(dp / "par"),
                        "--reserved-fec-manifest-root", str(dp / "fec"),
                        "--t-manifest-root", str(dp / "mi_t"), "--c-manifest-root", str(dp / "mi_c"),
                        "--p-manifest-root", str(dp / "mi_p"), "--h-manifest-root", str(dp / "mi_h"),
                        "--a-manifest-root", str(dp / "mi_a"),
                        "--pilot-detector-config-root", str(dp / "d"),
                        "--selection-rule-file", str(rule_file),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Parent validator with sealed identity manifests
# ═══════════════════════════════════════════════════════════════════

def test_parent_sealed_identity_manifests_pass():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(5)]
        rule_file = dp / "rule.json"; _make_rule_file(rule_file)
        rule_sha = sha256_file(rule_file)
        _seal_single(dp / "par", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5,
            "expected_suite_counts": {"s0": 5}, "selection_rule_sha256": rule_sha})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        _seal_single(dp / "fec", "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": [f"fec_{i}" for i in range(16)]})
        for role, dn in [("T","mi_t"),("C","mi_c"),("P","mi_p"),("H","mi_h"),("A","mi_a")]:
            _seal_single(dp / dn, "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest-root", str(dp / "par"),
                        "--reserved-fec-manifest-root", str(dp / "fec"),
                        "--t-manifest-root", str(dp / "mi_t"), "--c-manifest-root", str(dp / "mi_c"),
                        "--p-manifest-root", str(dp / "mi_p"), "--h-manifest-root", str(dp / "mi_h"),
                        "--a-manifest-root", str(dp / "mi_a"),
                        "--pilot-detector-config-root", str(dp / "d"),
                        "--selection-rule-file", str(rule_file),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc == 0
        finally: sys.argv = old


def test_parent_selection_rule_sha_mismatch():
    from validate_factorized_pilot_parent_manifest import main as pm
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        parents = [_make_parent(f"fec_{i}", "s0", i) for i in range(5)]
        rule_file = dp / "rule.json"; _make_rule_file(rule_file)
        _seal_single(dp / "par", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0",
            "parents": parents, "expected_parent_count": 5, "expected_suite_counts": {"s0": 5},
            "selection_rule_sha256": _sha("z")})
        _seal_single(dp / "d", "c.json", {"schema": "PILOT_DETECTOR_V0",
            "paper_authoritative": False, "attack_eval_consumed": False,
            "detector_checkpoint_sha256": _sha("a"), "detector_config_sha256": _sha("b"),
            "detector_feature_order_sha256": _sha("c"), "detector_normalization_sha256": _sha("d"),
            "detector_runtime_source_sha256": _sha("e")})
        _seal_single(dp / "fec", "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": [f"fec_{i}" for i in range(16)]})
        for role, dn in [("T","mi_t"),("C","mi_c"),("P","mi_p"),("H","mi_h"),("A","mi_a")]:
            _seal_single(dp / dn, "m.json", {"schema": "IDENTITY_MANIFEST_V0", "identities": []})
        old = sys.argv
        try:
            sys.argv = ["pm", "--pilot-parent-manifest-root", str(dp / "par"),
                        "--reserved-fec-manifest-root", str(dp / "fec"),
                        "--t-manifest-root", str(dp / "mi_t"), "--c-manifest-root", str(dp / "mi_c"),
                        "--p-manifest-root", str(dp / "mi_p"), "--h-manifest-root", str(dp / "mi_h"),
                        "--a-manifest-root", str(dp / "mi_a"),
                        "--pilot-detector-config-root", str(dp / "d"),
                        "--selection-rule-file", str(rule_file),
                        "--output-root", str(dp / "o")]
            rc = pm(); assert rc != 0
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# K-closure and execution validator
# ═══════════════════════════════════════════════════════════════════

def test_clean_k_not_zero():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j_fec_0_CLEAN_0_0"; mgid = "g_fec_0_0"
        run = _make_run("fec_0", "CLEAN", jid=jid, mgid=mgid, k_req=10, k_exec=10)
        run.pop("attack_step_ledger", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("vid"); (dp / "ev" / run["telemetry_path"]).write_text("tel")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "CLEAN", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
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


def test_true_k_9_rejected():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j_fec_0_TRUE_T10_0_0"; mgid = "g_fec_0_0"
        run = _make_run("fec_0", "TRUE_T10", jid=jid, mgid=mgid, k_exec=9)
        run["attack_step_ledger"] = [{"step": 5 + i, "armed": True, "executed": True} for i in range(9)]
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("fec_0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
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
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid); t_run["epsilon"] = 0.01
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["epsilon"] = 0.02; r_run["gradient_aligned"] = False
        (dp / "ev").mkdir()
        for run in [t_run, r_run]:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha_t = sha256_file(dp / "ev" / t_run["video_path"]); tsha_t = sha256_file(dp / "ev" / t_run["telemetry_path"])
        vsha_r = sha256_file(dp / "ev" / r_run["video_path"]); tsha_r = sha256_file(dp / "ev" / r_run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t_run, r_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": "j_t", "matched_group_id": mgid, "path": t_run["telemetry_path"], "sha256": tsha_t},
            {"job_id": "j_r", "matched_group_id": mgid, "path": r_run["telemetry_path"], "sha256": tsha_r}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j_t", "matched_group_id": mgid, "path": t_run["video_path"], "sha256": vsha_t},
            {"job_id": "j_r", "matched_group_id": mgid, "path": r_run["video_path"], "sha256": vsha_r}]})
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
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid); run.pop("arm_max_abs_diff", None)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
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
        jid = "j0"; mgid = "g0"
        run = _make_run("f0", "TRUE_T10", jid=jid, mgid=mgid)
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": [_make_job("f0", "TRUE_T10", jid=jid, mgid=mgid)]})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["telemetry_path"]}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": run["video_path"]}]})
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
# Parity mismatch → CSV consistency (budget valid from disposition)
# ═══════════════════════════════════════════════════════════════════

def test_parity_mismatch_csv_consistent():
    from validate_factorized_attack_pilot_execution import main as ev
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["checkpoint_sha256"] = _sha("z")
        (dp / "ev").mkdir()
        for run in [t_run, r_run]:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha_t = sha256_file(dp / "ev" / t_run["video_path"]); tsha_t = sha256_file(dp / "ev" / t_run["telemetry_path"])
        vsha_r = sha256_file(dp / "ev" / r_run["video_path"]); tsha_r = sha256_file(dp / "ev" / r_run["telemetry_path"])
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [t_run, r_run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": "j_t", "matched_group_id": mgid, "path": t_run["telemetry_path"], "sha256": tsha_t},
            {"job_id": "j_r", "matched_group_id": mgid, "path": r_run["telemetry_path"], "sha256": tsha_r}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j_t", "matched_group_id": mgid, "path": t_run["video_path"], "sha256": vsha_t},
            {"job_id": "j_r", "matched_group_id": mgid, "path": r_run["video_path"], "sha256": vsha_r}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "o")]
            rc = ev(); assert rc != 0
            with open(dp / "o/PILOT_DISPOSITION_V0.csv") as f:
                rows = list(csv.DictReader(f))
            with open(dp / "o/PILOT_BUDGET_PARITY_V0.csv") as f:
                budget = list(csv.DictReader(f))
            with open(dp / "o/PILOT_EXECUTION_VALIDATION_V0.json") as f:
                receipt = json.load(f)
            r_budget = [b for b in budget if b["condition"] == "RAND_T10"]
            assert len(r_budget) == 1
            assert r_budget[0]["disposition"] == "PARITY_MISMATCH"
            assert r_budget[0]["valid"] == "False"  # FIXED: valid derives from disposition
            r_disp = [r for r in rows if r["condition"] == "RAND_T10"]
            assert r_disp[0]["disposition"] == "PARITY_MISMATCH"
            assert receipt["status"] == "HOLD"
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Cross-receipt substitution rejection
# ═══════════════════════════════════════════════════════════════════

def test_cross_receipt_substitution_rejected():
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        run["attack_requested"] = False; run.pop("attack_step_ledger", None)
        run["official_success"] = True; run["gripper_opened"] = True
        (dp / "ev").mkdir(); (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        vsha = sha256_file(dp / "ev" / run["video_path"]); tsha = sha256_file(dp / "ev" / run["telemetry_path"])

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [run]})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": "j_c", "matched_group_id": mgid, "path": run["telemetry_path"], "sha256": tsha}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j_c", "matched_group_id": mgid, "path": run["video_path"], "sha256": vsha}]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        from validate_factorized_attack_pilot_execution import main as ev
        old = sys.argv
        try:
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            _seal_single(dp / "l2", "m.json", {"schema": "PILOT_RUN_LEDGER_V0",
                "runs": [{"job_id": "j_c", "matched_group_id": mgid, "parent_id": "f0", "condition": "CLEAN",
                          "k_requested": 0, "k_executed": 0, "official_success": False}]})

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l2"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            an(); assert False
        except SystemExit: pass
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# GO/NO-GO: attack ineffective → STOP; effective → CONTINUE
# ═══════════════════════════════════════════════════════════════════

def test_go_stop_when_attack_ineffective():
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid),
                _make_job("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid),
                _make_job("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = True; t_run["gripper_opened"] = True
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["official_success"] = True; r_run["gripper_opened"] = True
        rt_run = _make_run("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid)
        rt_run["official_success"] = True; rt_run["gripper_opened"] = True
        or_run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid)
        or_run["official_success"] = True; or_run["gripper_opened"] = True
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True; c_run["gripper_opened"] = True

        runs = [t_run, r_run, rt_run, or_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "STOP"
            assert gr["scientific_go_no_go_authorized"] is False
        finally: sys.argv = old


def test_go_continue_when_attack_effective():
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid),
                _make_job("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid),
                _make_job("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = False  # TRUE causes failure
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["official_success"] = True
        rt_run = _make_run("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid)
        rt_run["official_success"] = True  # TRUE beats RANDOM_TIME too
        or_run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid)
        or_run["official_success"] = False; or_run["gripper_opened"] = True
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True

        runs = [t_run, r_run, rt_run, or_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc == 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "CONTINUE"
        finally: sys.argv = old


def test_go_stop_timing_when_true_beats_rt_but_not_rand():
    """TRUE beats RANDOM_TIME but not RAND → timing matters, gradient doesn't → STOP_TIMING."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid),
                _make_job("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid),
                _make_job("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = False  # TRUE fails
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["official_success"] = False  # RAND also fails (random direction still works)
        rt_run = _make_run("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid)
        rt_run["official_success"] = True  # RANDOM_TIME succeeds (timing matters)
        or_run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid)
        or_run["official_success"] = False; or_run["gripper_opened"] = True
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True

        runs = [t_run, r_run, rt_run, or_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "STOP_TIMING"
        finally: sys.argv = old


def test_go_stop_window_when_oracle_fails():
    """ORACLE doesn't cause degradation → window not viable → STOP_WINDOW."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid),
                _make_job("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid),
                _make_job("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = False
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["official_success"] = True
        rt_run = _make_run("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid)
        rt_run["official_success"] = True
        or_run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid)
        or_run["official_success"] = True; or_run["gripper_opened"] = False  # Oracle doesn't cause damage
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True

        runs = [t_run, r_run, rt_run, or_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "STOP_WINDOW"
        finally: sys.argv = old


def test_go_rules_missing_field_rejected():
    """GO rules with missing required field → SystemExit."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "ev").mkdir()
        # Rules missing required field
        rules_root = dp / "gr"
        rules_root.mkdir(parents=True)
        rules = {"schema": "PILOT_GO_NO_GO_RULES_V0",
                 "rules": {"min_valid_pairs": 1}}  # missing 5 required fields
        (rules_root / "go_rules.json").write_text(json.dumps(rules))
        files = sorted(p for p in rules_root.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
        (rules_root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
        (rules_root / "SHA256SUMS.sha256").write_text(f"{sha256_file(rules_root / 'SHA256SUMS')}  SHA256SUMS\n")

        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "PASS",
            "input_seals": {}})
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": []})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        old = sys.argv
        try:
            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(rules_root),
                        "--output-root", str(dp / "o")]
            an(); assert False
        except SystemExit: pass
        finally: sys.argv = old


def test_go_stop_when_incomplete_groups():
    """Incomplete groups → hard STOP."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        # Only 2 of 5 conditions → incomplete group
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = False
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True

        runs = [t_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "STOP"
            assert any("INCOMPLETE_GROUPS" in b for b in gr["blocker_reasons"])
        finally: sys.argv = old


def test_go_modify_detector_when_insufficient_groups():
    """Insufficient groups → MODIFY_DETECTOR."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "ev").mkdir()
        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": []})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": []})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": []})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0",
            "status": "PASS", "input_seals": {}, "disposition_counts": {}, "n_expected_jobs": 0})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        _make_go_rules(dp / "gr", min_valid_pairs=4)  # need 4, have 0

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            # Create minimal PASS receipt
            _seal_single(dp / "ev_val2", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0",
                "status": "PASS", "input_seals": {}, "disposition_counts": {}, "n_expected_jobs": 0,
                "n_errors": 0, "allowed_conditions": []})

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "MODIFY_DETECTOR"
        finally: sys.argv = old


def test_go_modify_detector_true_beats_rand_but_not_rt():
    """TRUE beats RAND but not RANDOM_TIME → gradient matters, timing matters more → MODIFY_DETECTOR."""
    from analyze_factorized_attack_pilot import main as an
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        mgid = "g_f0_0"
        jobs = [_make_job("f0", "TRUE_T10", jid="j_t", mgid=mgid),
                _make_job("f0", "RAND_T10", jid="j_r", mgid=mgid),
                _make_job("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid),
                _make_job("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid),
                _make_job("f0", "CLEAN", jid="j_c", mgid=mgid)]
        t_run = _make_run("f0", "TRUE_T10", jid="j_t", mgid=mgid)
        t_run["official_success"] = False  # TRUE fails
        r_run = _make_run("f0", "RAND_T10", jid="j_r", mgid=mgid); r_run["gradient_aligned"] = False
        r_run["official_success"] = True  # RAND succeeds (gradient matters → TRUE beats RAND)
        rt_run = _make_run("f0", "RANDOM_TIME_T10", jid="j_rt", mgid=mgid)
        rt_run["official_success"] = False  # RANDOM_TIME also fails (timing matters equally)
        or_run = _make_run("f0", "COMMAND_OPEN_ORACLE", jid="j_or", mgid=mgid)
        or_run["official_success"] = False; or_run["gripper_opened"] = True
        c_run = _make_run("f0", "CLEAN", jid="j_c", mgid=mgid, k_req=0, k_exec=0)
        c_run["attack_requested"] = False; c_run.pop("attack_step_ledger", None)
        c_run["official_success"] = True

        runs = [t_run, r_run, rt_run, or_run, c_run]
        (dp / "ev").mkdir()
        for run in runs:
            (dp / "ev" / run["video_path"]).write_text("v"); (dp / "ev" / run["telemetry_path"]).write_text("t")
        sha_map = {}
        for run in runs:
            sha_map[run["job_id"]] = {
                "vsha": sha256_file(dp / "ev" / run["video_path"]),
                "tsha": sha256_file(dp / "ev" / run["telemetry_path"])}

        _seal_single(dp / "j", "m.json", {"schema": "PILOT_JOB_MATRIX_V0", "jobs": jobs})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": runs})
        _seal_single(dp / "t", "m.json", {"schema": "PILOT_TELEMETRY_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["telemetry_path"], "sha256": sha_map[r["job_id"]]["tsha"]} for r in runs]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": r["job_id"], "matched_group_id": mgid, "path": r["video_path"], "sha256": sha_map[r["job_id"]]["vsha"]} for r in runs]})
        _seal_single(dp / "pm", "m.json", {"schema": "PILOT_PARENT_MANIFEST_V0", "parents": [_make_parent("f0")]})
        _seal_single(dp / "arm", "m.json", {"schema": "PILOT_ARM_PARITY_PROTOCOL_V0", "max_abs_tolerance": 0.01})
        _make_go_rules(dp / "gr")

        old = sys.argv
        try:
            from validate_factorized_attack_pilot_execution import main as ev
            sys.argv = ["ev", "--pilot-job-matrix-root", str(dp / "j"), "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"), "--pilot-video-index-root", str(dp / "v"),
                        "--pilot-parent-manifest-root", str(dp / "pm"),
                        "--pilot-arm-parity-protocol-root", str(dp / "arm"),
                        "--evidence-root", str(dp / "ev"), "--output-root", str(dp / "ev_out")]
            ev()

            sys.argv = ["an", "--pilot-execution-validation-root", str(dp / "ev_out"),
                        "--pilot-job-matrix-root", str(dp / "j"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-telemetry-index-root", str(dp / "t"),
                        "--pilot-go-no-go-rules-root", str(dp / "gr"),
                        "--output-root", str(dp / "o")]
            rc = an()
            assert rc != 0
            with open(dp / "o/PILOT_AUTOMATED_GO_NO_GO_V0.json") as f:
                gr = json.load(f)
            assert gr["automated_recommendation"] == "MODIFY_DETECTOR"
        finally: sys.argv = old


# ═══════════════════════════════════════════════════════════════════
# Blind review tests (fixed: structural leak check, no raw-text scan)
# ═══════════════════════════════════════════════════════════════════

def test_blind_separate_roots():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jid = "j0"; mgid = "g0"
        (dp / "ev").mkdir()
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "PASS",
            "input_seals": {}, "disposition_counts": {}, "n_expected_jobs": 1, "n_errors": 0})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [
            {"job_id": jid, "matched_group_id": mgid, "parent_id": "f0", "condition": "CLEAN", "video_path": "v.mp4",
             "k_requested": 0, "k_executed": 0, "attack_requested": False}]})
        video_path = dp / "ev" / "v.mp4"; video_path.write_text("video_content")
        vsha = sha256_file(video_path)
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": "v.mp4", "sha256": vsha}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--evidence-root", str(dp / "ev"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        assert (dp / "blind/PILOT_BLIND_REVIEW_V0.csv").exists()
        assert (dp / "blind/videos").is_dir()
        assert (dp / "unblind/PILOT_UNBLINDING_V0.csv").exists()
        assert not (dp / "blind/PILOT_UNBLINDING_V0.csv").exists()
        # Structural check: blind JSON fields must not contain condition info
        with open(dp / "blind/PILOT_BLIND_REVIEW_V0.json") as f:
            blind_data = json.load(f)
        for entry in blind_data["entries"]:
            assert "condition" not in entry
            assert "parent_id" not in entry
            assert "video_path" not in entry
            assert "original" not in str(entry)
            # blind_id and video_file only
            assert set(entry.keys()).issubset({"blind_id", "video_file", "video_sha256",
                                               "reviewer_a_labels", "reviewer_b_labels"})


def test_blind_same_root_rejected():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "ev").mkdir()
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "PASS",
            "input_seals": {}, "disposition_counts": {}})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [
            {"job_id": "j0", "matched_group_id": "g0", "parent_id": "f0", "condition": "C", "video_path": "v.mp4",
             "k_requested": 0, "k_executed": 0}]})
        (dp / "ev/v.mp4").write_text("v")
        vsha = sha256_file(dp / "ev/v.mp4")
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j0", "matched_group_id": "g0", "path": "v.mp4", "sha256": vsha}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--evidence-root", str(dp / "ev"),
                        "--blind-package-root", str(dp / "same"),
                        "--unblinding-root", str(dp / "same")]
            br(); assert False
        except SystemExit: pass
        finally: sys.argv = old


def test_blind_requires_execution_pass():
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "HOLD"})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [
            {"job_id": "j0", "matched_group_id": "g0", "parent_id": "f0", "condition": "C", "video_path": "v.mp4",
             "k_requested": 0, "k_executed": 0}]})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": "j0", "matched_group_id": "g0", "path": "v.mp4", "sha256": _sha("v")}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--evidence-root", str(dp / "ev"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            br(); assert False
        except SystemExit: pass
        finally: sys.argv = old


def test_blind_no_raw_paths_structural():
    """Verify blind root structurally — check JSON field names and values, not raw text scan."""
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "ev").mkdir()
        jid = "j0"; mgid = "g0"
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "PASS",
            "input_seals": {}, "disposition_counts": {}, "n_expected_jobs": 1})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": [
            {"job_id": jid, "matched_group_id": mgid, "parent_id": "p1", "condition": "CLEAN", "video_path": "original_name.mp4",
             "k_requested": 0, "k_executed": 0, "attack_requested": False}]})
        (dp / "ev/original_name.mp4").write_text("video_content")
        vsha = sha256_file(dp / "ev/original_name.mp4")
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": [
            {"job_id": jid, "matched_group_id": mgid, "path": "original_name.mp4", "sha256": vsha}]})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--evidence-root", str(dp / "ev"),
                        "--blind-package-root", str(dp / "blind"),
                        "--unblinding-root", str(dp / "unblind")]
            rc = br(); assert rc == 0
        finally: sys.argv = old
        # Structural verification — parse JSON, check fields
        with open(dp / "blind/PILOT_BLIND_REVIEW_V0.json") as f:
            blind_data = json.load(f)
        for entry in blind_data["entries"]:
            # Must NOT contain raw path or parent info
            assert "original_name" not in json.dumps(entry)
            assert "p1" not in json.dumps(entry)
            assert "video_path" not in entry
            assert "video_reference" not in entry
            assert "original" not in str(entry)
            # video_file must be a random name (Bxxxxx.ext)
            vf = entry["video_file"]
            assert vf.startswith("B") and not vf.startswith("original")
        # Blind root CSV must not have raw path
        with open(dp / "blind/PILOT_BLIND_REVIEW_V0.csv") as f:
            csv_text = f.read()
        assert "original_name" not in csv_text
        assert "p1" not in csv_text


def test_blind_nested_roots_rejected():
    """Blind root nested inside unblinding root or vice versa → SystemExit."""
    from build_factorized_pilot_blind_review import main as br
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d); (dp / "ev").mkdir()
        _seal_single(dp / "ev_val", "m.json", {"schema": "PILOT_EXECUTION_VALIDATION_V0", "status": "PASS",
            "input_seals": {}, "disposition_counts": {}})
        _seal_single(dp / "l", "m.json", {"schema": "PILOT_RUN_LEDGER_V0", "runs": []})
        _seal_single(dp / "v", "m.json", {"schema": "PILOT_VIDEO_INDEX_V0", "entries": []})
        old = sys.argv
        try:
            sys.argv = ["br", "--pilot-execution-validation-root", str(dp / "ev_val"),
                        "--pilot-run-ledger-root", str(dp / "l"),
                        "--pilot-video-index-root", str(dp / "v"),
                        "--evidence-root", str(dp / "ev"),
                        "--blind-package-root", str(dp / "root"),
                        "--unblinding-root", str(dp / "root/sub")]
            br(); assert False
        except SystemExit: pass
        finally: sys.argv = old
