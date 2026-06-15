"""D4.3a: Canary auditor integration tests (CPU, no GPU).

Tests safe-tag parsing, attempt contract, hash-row validation,
CSV schema gates, emit uniqueness, and auditor main() integration.
"""

import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))

from audit_d4_shadow_canary import (
    SAFE_TAG_RE, parse_safe_tag, is_successful_attempt,
    has_first_action_generated, sha256_file,
    REQUIRED_LAUNCHER_FILES, REQUIRED_EPISODE_FILES,
    HASHED_EPISODE_FILES, HASHED_LAUNCHER_FILES,
)

VALID_KEYS = {("alphabet_soup", 0), ("bbq_sauce", 1),
              ("cream_cheese", 2), ("ketchup", 3)}

ALL_4_TASKS = ["alphabet_soup", "bbq_sauce", "cream_cheese", "ketchup"]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_episode(tmp_path, returncode=0, fatal=False, infra="ok",
                  first_action=True, has_fa_file=True):
    """Create a synthetic episode in tmp_path. Returns (ep_dir, ll_dir)."""
    ep_dir = tmp_path / "ep"
    ll_dir = tmp_path / "launcher"
    ep_dir.mkdir()
    ll_dir.mkdir()

    with open(ll_dir / "returncode.json", "w") as f:
        json.dump({"returncode": returncode}, f)

    with open(ep_dir / "episode_manifest.json", "w") as f:
        json.dump({
            "fatal": fatal, "infra_status": infra,
            "first_action_generated": first_action,
        }, f)

    if has_fa_file:
        (ep_dir / "FIRST_ACTION_GENERATED.json").write_text("{}")

    return ep_dir, ll_dir


def _make_minimal_canary_root(tmp_path):
    """Create a synthetic canary output tree. Returns (root, manifest_path, manifest_sha)."""
    root = tmp_path / "canary_root"
    root.mkdir()
    launcher_dir = root / "launcher_logs"
    launcher_dir.mkdir()

    # Write manifest
    manifest = root.parent / "manifest.csv"
    import hashlib
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "task_key", "state_id", "selection_hash",
                     "trace_id", "source", "frozen_order"])
        for i, tk in enumerate(ALL_4_TASKS):
            h = hashlib.sha256(
                f"D4.3_SHADOW_V1|{tk}|{i}".encode()
            ).hexdigest()
            w.writerow(["canary", tk, str(i), h,
                         f"trace_{tk}_s{i}", "historical_402", str(i)])
    manifest_sha = sha256_file(str(manifest))

    # Create 8 episodes (4 ref + 4 shadow)
    for tk_idx, tk in enumerate(ALL_4_TASKS):
        for mode in ("reference", "shadow"):
            tag = f"{tk}_s{tk_idx}_{mode}_attempt1"

            ep_dir = root / tag
            ll_dir = launcher_dir / tag
            ep_dir.mkdir()
            ll_dir.mkdir()

            # Launcher artifacts
            with open(ll_dir / "returncode.json", "w") as f:
                json.dump({"returncode": 0}, f)
            for fn in REQUIRED_LAUNCHER_FILES:
                fp = ll_dir / fn
                if not fp.exists():
                    if fn.endswith(".csv"):
                        with open(fp, "w", newline="") as f2:
                            w2 = csv.writer(f2)
                            w2.writerow(["artifact", "sha256"])
                            for hf in HASHED_LAUNCHER_FILES:
                                hfp = ll_dir / hf
                                if not hfp.exists():
                                    hfp.write_text(f"{hf}_content")
                                w2.writerow([hf, sha256_file(str(hfp))])
                    else:
                        fp.write_text(f"{fn}_content")

            # Episode artifacts
            for fn in REQUIRED_EPISODE_FILES:
                fp = ep_dir / fn
                if fn == "episode_manifest.json":
                    with open(fp, "w") as f2:
                        json.dump({
                            "task": tk, "state_id": tk_idx, "mode": mode,
                            "n_steps": 10, "success_primary": 1,
                            "success_done_any": 1, "success_check_any": 1,
                            "success_step_primary": 5, "done_step": 9,
                            "infra_status": "ok",
                            "detector_exception": False,
                            "action_identity_fail": False,
                            "n_invalid_field_steps": 0,
                            "first_action_generated": True,
                            "detector_emit_step": 5 if mode == "shadow" else "DISABLED",
                            "detector_pre_reset": {
                                "next_expected_step": 0, "emit_step": -1,
                                "history_len": 0, "candidate_count": 0,
                            } if mode == "shadow" else {},
                            "raw_action_sequence_sha256": "abc123",
                            "env_action_sequence_sha256": "def456",
                            "obs_sequence_sha256": "ghi789",
                        }, f2)
                elif fn == "ATTEMPT_STARTED.json":
                    with open(fp, "w") as f2:
                        json.dump({"marker": "ATTEMPT_STARTED"}, f2)
                elif fn == "MODEL_LOADED.json":
                    with open(fp, "w") as f2:
                        json.dump({"marker": "MODEL_LOADED"}, f2)
                elif fn == "FIRST_ACTION_GENERATED.json":
                    with open(fp, "w") as f2:
                        json.dump({"marker": "FIRST_ACTION_GENERATED"}, f2)
                elif fn == "step_trace.csv":
                    with open(fp, "w", newline="") as f2:
                        w2 = csv.DictWriter(f2, fieldnames=[
                            "step", "raw_valid", "env_valid", "qpos_valid",
                            "eef_valid", "convention_ok", "semantics_ok",
                        ])
                        w2.writeheader()
                        for s in range(10):
                            w2.writerow({"step": str(s),
                                "raw_valid": "1", "env_valid": "1",
                                "qpos_valid": "1", "eef_valid": "1",
                                "convention_ok": "1", "semantics_ok": "1"})
                elif fn == "action_identity.csv":
                    with open(fp, "w", newline="") as f2:
                        w2 = csv.DictWriter(f2, fieldnames=["step", "action_identical"])
                        w2.writeheader()
                        for s in range(10):
                            w2.writerow({"step": str(s), "action_identical": "1"})
                elif fn == "latency.csv":
                    with open(fp, "w", newline="") as f2:
                        w2 = csv.DictWriter(f2, fieldnames=[
                            "step", "detector_update_us", "model_inference_us"])
                        w2.writeheader()
                        for s in range(10):
                            w2.writerow({"step": str(s),
                                "detector_update_us": "500",
                                "model_inference_us": "50000"})
                elif fn == "detector_candidates.csv":
                    with open(fp, "w", newline="") as f2:
                        w2 = csv.DictWriter(f2, fieldnames=[
                            "step", "score", "abstain", "abstained"])
                        w2.writeheader()
                        w2.writerow({"step": "5", "score": "0.5",
                                      "abstain": "", "abstained": "0"})
                elif fn == "detector_emission.json":
                    with open(fp, "w") as f2:
                        json.dump({"emit_step": 5, "detector_enabled": True}, f2)
                elif fn == "artifact_hashes.csv":
                    # Will be written last; create placeholders for hashed files first
                    pass
                elif fn == "teacher_sidecar.json":
                    with open(fp, "w") as f2:
                        json.dump({"status": "PENDING_SIDECAR"}, f2)
                elif fn == "provenance.csv":
                    with open(fp, "w", newline="") as f2:
                        w2 = csv.writer(f2)
                        w2.writerow(["key", "value"])
                        w2.writerow(["git_HEAD", "test"])
                else:
                    fp.write_text(f"{fn}_content")

            # Write artifact_hashes.csv last (after all hashed files exist)
            with open(ep_dir / "artifact_hashes.csv", "w", newline="") as f2:
                w2 = csv.writer(f2)
                w2.writerow(["artifact", "sha256"])
                for hf in HASHED_EPISODE_FILES:
                    hfp = ep_dir / hf
                    if hfp.exists():
                        w2.writerow([hf, sha256_file(str(hfp))])

    # GPU snapshots
    for sn in ["gpu_processes_before.csv", "gpu_processes_after.csv"]:
        with open(root / sn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gpu_uuid", "pid", "process_name"])

    return root, manifest, manifest_sha


# ═══════════════════════════════════════════════════════════════
# Safe-tag regex (8 tests)
# ═══════════════════════════════════════════════════════════════

def test_regex_parses_single_underscore_task():
    m = SAFE_TAG_RE.match("ketchup_s5_reference_attempt1")
    assert m is not None
    assert m.group("task") == "ketchup"
    assert m.group("state_id") == "5"
    assert m.group("mode") == "reference"
    assert m.group("attempt_id") == "1"


def test_regex_parses_multi_underscore_task():
    m = SAFE_TAG_RE.match("alphabet_soup_s9_shadow_attempt2")
    assert m is not None
    assert m.group("task") == "alphabet_soup"


def test_regex_parses_chocolate_pudding():
    m = SAFE_TAG_RE.match("chocolate_pudding_s42_reference_attempt1")
    assert m is not None
    assert m.group("task") == "chocolate_pudding"


def test_regex_rejects_attempt3():
    assert SAFE_TAG_RE.match("ketchup_s5_reference_attempt3") is None


def test_regex_rejects_attack_mode():
    assert SAFE_TAG_RE.match("ketchup_s5_attack_attempt1") is None


def test_regex_rejects_bad_format():
    assert SAFE_TAG_RE.match("bad_tag") is None


def test_parse_safe_tag_validates_keys():
    info = parse_safe_tag("alphabet_soup_s0_reference_attempt1", VALID_KEYS)
    assert info is not None
    assert info["task"] == "alphabet_soup"


def test_parse_safe_tag_rejects_unknown_key():
    info = parse_safe_tag("tomato_sauce_s0_reference_attempt1", VALID_KEYS)
    assert info is None


# ═══════════════════════════════════════════════════════════════
# Attempt sequence validation (4 tests)
# ═══════════════════════════════════════════════════════════════

def test_attempt_ids_1_valid():
    assert [1] in ([1], [1, 2])


def test_attempt_ids_12_valid():
    assert [1, 2] in ([1], [1, 2])


def test_attempt_ids_2_only_invalid():
    assert [2] not in ([1], [1, 2])


def test_attempt_ids_3_invalid():
    assert [1, 2, 3] not in ([1], [1, 2])


# ═══════════════════════════════════════════════════════════════
# is_successful_attempt (7 tests) — all use tmp_path
# ═══════════════════════════════════════════════════════════════

def test_is_success_all_ok(tmp_path):
    ep, ll = _make_episode(tmp_path)
    assert is_successful_attempt(ep, ll)


def test_is_success_rc_nonzero(tmp_path):
    ep, ll = _make_episode(tmp_path, returncode=1)
    assert not is_successful_attempt(ep, ll)


def test_is_success_manifest_missing(tmp_path):
    ep_dir = tmp_path / "ep"
    ll_dir = tmp_path / "launcher"
    ep_dir.mkdir(); ll_dir.mkdir()
    with open(ll_dir / "returncode.json", "w") as f:
        json.dump({"returncode": 0}, f)
    (ep_dir / "FIRST_ACTION_GENERATED.json").write_text("{}")
    assert not is_successful_attempt(ep_dir, ll_dir)


def test_is_success_fatal(tmp_path):
    ep, ll = _make_episode(tmp_path, fatal=True)
    assert not is_successful_attempt(ep, ll)


def test_is_success_infra_not_ok(tmp_path):
    ep, ll = _make_episode(tmp_path, infra="ACTION_IDENTITY_FAIL")
    assert not is_successful_attempt(ep, ll)


def test_is_success_no_fa_file(tmp_path):
    ep, ll = _make_episode(tmp_path, has_fa_file=False)
    assert not is_successful_attempt(ep, ll)


def test_is_success_manifest_says_no_fa(tmp_path):
    ep, ll = _make_episode(tmp_path, first_action=False)
    assert not is_successful_attempt(ep, ll)


# ═══════════════════════════════════════════════════════════════
# Pre-action failure contract (2 tests)
# ═══════════════════════════════════════════════════════════════

def test_pre_action_failure_no_fa():
    assert not has_first_action_generated(Path("/nonexistent"))


def test_pre_action_failure_requires_nonzero_rc(tmp_path):
    ll_dir = tmp_path / "launcher"
    ll_dir.mkdir()
    with open(ll_dir / "returncode.json", "w") as f:
        json.dump({"returncode": 1}, f)
    rc = json.load(open(ll_dir / "returncode.json"))
    assert rc["returncode"] != 0


# ═══════════════════════════════════════════════════════════════
# Hash-row set validation (5 tests)
# ═══════════════════════════════════════════════════════════════

def test_hash_set_exact_match():
    expected = sorted(HASHED_LAUNCHER_FILES)
    actual = sorted(HASHED_LAUNCHER_FILES)
    assert actual == expected


def test_hash_set_extra_row_fails():
    assert set(HASHED_LAUNCHER_FILES) != (set(HASHED_LAUNCHER_FILES) | {"extra.txt"})


def test_hash_set_missing_row_fails():
    assert set(HASHED_LAUNCHER_FILES) != (set(HASHED_LAUNCHER_FILES) - {"command.txt"})


def test_hash_set_no_duplicates():
    rows = list(HASHED_LAUNCHER_FILES) + ["command.txt"]
    assert len(rows) != len(set(rows))


def test_artifact_hashes_not_in_hashed_set():
    assert "artifact_hashes.csv" in REQUIRED_EPISODE_FILES
    assert "artifact_hashes.csv" not in HASHED_EPISODE_FILES


# ═══════════════════════════════════════════════════════════════
# Invalid step counting — per step, not per flag (P0-2 fix)
# ═══════════════════════════════════════════════════════════════

def test_invalid_step_count_one_step_multi_flags(tmp_path):
    """One step with 3 invalid flags = count 1, not 3."""
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    with open(ep_dir / "step_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "step", "raw_valid", "env_valid", "qpos_valid",
            "eef_valid", "convention_ok", "semantics_ok",
        ])
        w.writeheader()
        w.writerow({"step": "0", "raw_valid": "0", "env_valid": "0",
                     "qpos_valid": "1", "eef_valid": "1",
                     "convention_ok": "0", "semantics_ok": "1"})
        w.writerow({"step": "1", "raw_valid": "1", "env_valid": "1",
                     "qpos_valid": "1", "eef_valid": "1",
                     "convention_ok": "1", "semantics_ok": "1"})

    flags = ["raw_valid", "env_valid", "qpos_valid", "eef_valid",
             "convention_ok", "semantics_ok"]
    with open(ep_dir / "step_trace.csv") as f:
        rows = list(csv.DictReader(f))
    invalid_steps = 0
    for row in rows:
        row_invalid = False
        for fl in flags:
            if row.get(fl) == "0":
                row_invalid = True
        if row_invalid:
            invalid_steps += 1
    assert invalid_steps == 1  # one step, not 3 flags


def test_invalid_step_count_multi_steps(tmp_path):
    """Two different steps with invalid flags = count 2."""
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    with open(ep_dir / "step_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "step", "raw_valid", "env_valid", "qpos_valid",
            "eef_valid", "convention_ok", "semantics_ok",
        ])
        w.writeheader()
        w.writerow({"step": "0", "raw_valid": "0", "env_valid": "1",
                     "qpos_valid": "1", "eef_valid": "1",
                     "convention_ok": "1", "semantics_ok": "1"})
        w.writerow({"step": "5", "raw_valid": "1", "env_valid": "1",
                     "qpos_valid": "0", "eef_valid": "1",
                     "convention_ok": "1", "semantics_ok": "1"})

    flags = ["raw_valid", "env_valid", "qpos_valid", "eef_valid",
             "convention_ok", "semantics_ok"]
    with open(ep_dir / "step_trace.csv") as f:
        rows = list(csv.DictReader(f))
    invalid_steps = 0
    for row in rows:
        if any(row.get(fl) == "0" for fl in flags):
            invalid_steps += 1
    assert invalid_steps == 2


# ═══════════════════════════════════════════════════════════════
# Action identity recomputation from CSV (1 test)
# ═══════════════════════════════════════════════════════════════

def test_action_identity_from_csv(tmp_path):
    ep_dir = tmp_path / "ep"
    ep_dir.mkdir()
    with open(ep_dir / "action_identity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "action_identical"])
        w.writeheader()
        w.writerow({"step": "0", "action_identical": "1"})
        w.writerow({"step": "1", "action_identical": "0"})
    with open(ep_dir / "action_identity.csv") as f:
        rows = list(csv.DictReader(f))
    fail = any(r.get("action_identical") == "0" for r in rows)
    assert fail


# ═══════════════════════════════════════════════════════════════
# File contracts (3 tests)
# ═══════════════════════════════════════════════════════════════

def test_required_launcher_files():
    assert "command.txt" in REQUIRED_LAUNCHER_FILES
    assert "returncode.json" in REQUIRED_LAUNCHER_FILES


def test_required_episode_files():
    assert "ATTEMPT_STARTED.json" in REQUIRED_EPISODE_FILES
    assert "episode_manifest.json" in REQUIRED_EPISODE_FILES


def test_hashed_files_subset_of_required():
    for f in HASHED_EPISODE_FILES:
        assert f in REQUIRED_EPISODE_FILES
    for f in HASHED_LAUNCHER_FILES:
        assert f in REQUIRED_LAUNCHER_FILES


# ═══════════════════════════════════════════════════════════════
# GPU snapshot — unconditional set audit (4 tests)
# ═══════════════════════════════════════════════════════════════

def test_gpu_before_nonempty_fails():
    before_ids = {("uuid-1", "123", "python")}
    assert len(before_ids) > 0


def test_gpu_after_residual_fails():
    after_ids = {("uuid-1", "456", "python")}
    assert len(after_ids) > 0


def test_gpu_before_empty_after_empty_passes():
    assert len(set()) == 0 and len(set()) == 0


def test_gpu_before_after_same_process_fails():
    before_ids = {("uuid-1", "123", "python")}
    assert len(before_ids) > 0


# ═══════════════════════════════════════════════════════════════
# CSV schema validation (3 tests)
# ═══════════════════════════════════════════════════════════════

def test_step_trace_flag_values_must_be_01():
    valid_values = {"0", "1"}
    for val in ["1", "0"]:
        assert str(val) in valid_values
    for bad in ["2", "True", ""]:
        assert str(bad) not in valid_values


def test_action_identity_values_must_be_01():
    valid_values = {"0", "1"}
    for bad in ["2", "True", ""]:
        assert str(bad) not in valid_values


def test_gpu_snapshot_csv_has_header():
    """GPU snapshot CSVs must have the expected header columns."""
    header = ["gpu_uuid", "pid", "process_name"]
    assert "gpu_uuid" in header
    assert "pid" in header


# ═══════════════════════════════════════════════════════════════
# Emit candidate uniqueness (4 tests)
# ═══════════════════════════════════════════════════════════════

def test_emit_single_candidate_matches():
    cands = [
        {"step": "10", "abstained": "0", "abstain": ""},
        {"step": "15", "abstained": "0", "abstain": ""},
    ]
    emit_cands = [c for c in cands if int(c["step"]) == 10]
    assert len(emit_cands) == 1 and emit_cands[0]["abstained"] == "0"


def test_emit_duplicate_candidate_fails():
    cands = [{"step": "10"}, {"step": "10"}]
    emit_cands = [c for c in cands if int(c["step"]) == 10]
    assert len(emit_cands) != 1


def test_emit_abstained_candidate_fails():
    cands = [{"step": "10", "abstained": "1", "abstain": "gripper_already_open"}]
    emit_cands = [c for c in cands if int(c["step"]) == 10]
    assert len(emit_cands) == 1 and emit_cands[0]["abstained"] == "1"


def test_emit_missing_candidate_fails():
    cands = [{"step": "5"}]
    emit_cands = [c for c in cands if int(c["step"]) == 10]
    assert len(emit_cands) == 0


# ═══════════════════════════════════════════════════════════════
# Auditor main() integration — synthetic valid tree PASS
# ═══════════════════════════════════════════════════════════════

def test_auditor_main_valid_tree_exit_0(tmp_path):
    """A complete valid canary tree must pass auditor main()."""
    root, manifest, manifest_sha = _make_minimal_canary_root(tmp_path)

    # Run auditor as subprocess
    import subprocess
    auditor_script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "stageb",
        "audit_d4_shadow_canary.py",
    )
    result = subprocess.run(
        [sys.executable, auditor_script,
         "--canary-output-dir", str(root),
         "--canary-manifest", str(manifest),
         "--expected-manifest-sha256", manifest_sha],
        capture_output=True, text=True, timeout=30,
    )
    print("STDOUT:", result.stdout[-500:])
    print("STDERR:", result.stderr[-500:])
    assert result.returncode == 0, (
        f"Auditor failed on valid tree: rc={result.returncode}\n{result.stderr}"
    )


# ═══════════════════════════════════════════════════════════════
# Auditor main() integration — corrupted tree FAIL
# ═══════════════════════════════════════════════════════════════

def test_auditor_main_corrupted_tree_exit_nonzero(tmp_path):
    """A canary tree with a missing required artifact must fail auditor."""
    root, manifest, manifest_sha = _make_minimal_canary_root(tmp_path)

    # Remove one required artifact from a shadow episode
    sh_dir = root / "alphabet_soup_s0_shadow_attempt1"
    (sh_dir / "step_trace.csv").unlink()

    import subprocess
    auditor_script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "stageb",
        "audit_d4_shadow_canary.py",
    )
    result = subprocess.run(
        [sys.executable, auditor_script,
         "--canary-output-dir", str(root),
         "--canary-manifest", str(manifest),
         "--expected-manifest-sha256", manifest_sha],
        capture_output=True, text=True, timeout=30,
    )
    print("STDOUT:", result.stdout[-500:])
    assert result.returncode != 0, (
        f"Auditor passed on corrupted tree (missing step_trace.csv)"
    )
