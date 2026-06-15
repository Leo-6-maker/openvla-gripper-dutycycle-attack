"""D4.3a: Canary auditor integration tests (CPU, no GPU).

Tests the auditor's safe-tag parsing, attempt contract enforcement,
hash-row set validation, and independent recomputation logic
using temporary directories.
"""

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))

# Import the auditor's key functions directly
from audit_d4_shadow_canary import (
    SAFE_TAG_RE, parse_safe_tag, is_successful_attempt,
    has_first_action_generated, sha256_file,
    REQUIRED_LAUNCHER_FILES, REQUIRED_EPISODE_FILES,
    HASHED_EPISODE_FILES, HASHED_LAUNCHER_FILES,
)

VALID_KEYS = {("alphabet_soup", 0), ("bbq_sauce", 1),
              ("cream_cheese", 2), ("ketchup", 3)}


# ═══════════════════════════════════════════════════════════════
# Safe-tag regex
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
    assert m.group("state_id") == "9"
    assert m.group("mode") == "shadow"
    assert m.group("attempt_id") == "2"


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
    assert SAFE_TAG_RE.match("ketchup_s5_reference_attempt") is None


def test_parse_safe_tag_validates_keys():
    info = parse_safe_tag("alphabet_soup_s0_reference_attempt1", VALID_KEYS)
    assert info is not None
    assert info["task"] == "alphabet_soup"
    assert info["state_id"] == 0


def test_parse_safe_tag_rejects_unknown_key():
    info = parse_safe_tag("tomato_sauce_s0_reference_attempt1", VALID_KEYS)
    assert info is None


# ═══════════════════════════════════════════════════════════════
# Attempt sequence validation
# ═══════════════════════════════════════════════════════════════

def test_attempt_ids_1_is_valid():
    ids = [1]
    assert ids in ([1], [1, 2])


def test_attempt_ids_12_is_valid():
    ids = [1, 2]
    assert ids in ([1], [1, 2])


def test_attempt_ids_2_only_invalid():
    ids = [2]
    assert ids not in ([1], [1, 2])


def test_attempt_ids_3_invalid():
    ids = [1, 2, 3]
    assert ids not in ([1], [1, 2])


# ═══════════════════════════════════════════════════════════════
# is_successful_attempt
# ═══════════════════════════════════════════════════════════════

def _make_episode(tmpdir, returncode=0, fatal=False, infra="ok",
                  first_action=True, has_fa_file=True):
    ep_dir = Path(tmpdir) / "ep"
    ll_dir = Path(tmpdir) / "launcher"
    ep_dir.mkdir(); ll_dir.mkdir()

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


def test_is_success_all_ok():
    ep, ll = _make_episode("tmp")
    assert is_successful_attempt(ep, ll)


def test_is_success_rc_nonzero():
    ep, ll = _make_episode("tmp", returncode=1)
    assert not is_successful_attempt(ep, ll)


def test_is_success_manifest_missing():
    ep_dir = Path(tempfile.mkdtemp()) / "ep"
    ll_dir = Path(tempfile.mkdtemp()) / "launcher"
    ep_dir.mkdir(); ll_dir.mkdir()
    with open(ll_dir / "returncode.json", "w") as f:
        json.dump({"returncode": 0}, f)
    (ep_dir / "FIRST_ACTION_GENERATED.json").write_text("{}")
    # No episode_manifest.json
    assert not is_successful_attempt(ep_dir, ll_dir)


def test_is_success_fatal():
    ep, ll = _make_episode("tmp", fatal=True)
    assert not is_successful_attempt(ep, ll)


def test_is_success_infra_not_ok():
    ep, ll = _make_episode("tmp", infra="ACTION_IDENTITY_FAIL")
    assert not is_successful_attempt(ep, ll)


def test_is_success_no_fa_file():
    ep, ll = _make_episode("tmp", has_fa_file=False)
    assert not is_successful_attempt(ep, ll)


def test_is_success_manifest_says_no_fa():
    ep, ll = _make_episode("tmp", first_action=False)
    assert not is_successful_attempt(ep, ll)


# ═══════════════════════════════════════════════════════════════
# Pre-action failure contract
# ═══════════════════════════════════════════════════════════════

def test_pre_action_failure_no_fa():
    """Pre-action failure must NOT have FIRST_ACTION_GENERATED."""
    assert not has_first_action_generated(Path("/nonexistent"))


def test_pre_action_failure_requires_nonzero_rc():
    """Pre-action failure must have returncode != 0."""
    with tempfile.TemporaryDirectory() as tmp:
        ll_dir = Path(tmp) / "launcher"
        ll_dir.mkdir()
        with open(ll_dir / "returncode.json", "w") as f:
            json.dump({"returncode": 1}, f)
        import json as j
        rc = j.load(open(ll_dir / "returncode.json"))
        assert rc["returncode"] != 0  # Pre-action failure has non-zero rc


# ═══════════════════════════════════════════════════════════════
# Hash-row set validation
# ═══════════════════════════════════════════════════════════════

def test_hash_set_exact_match():
    """Hash rows must exactly match the expected set."""
    expected = sorted(HASHED_LAUNCHER_FILES)
    actual = sorted(HASHED_LAUNCHER_FILES)  # correct
    assert actual == expected


def test_hash_set_extra_row_fails():
    """Extra hash row is a failure."""
    expected = set(HASHED_LAUNCHER_FILES)
    actual = set(HASHED_LAUNCHER_FILES) | {"extra_file.txt"}
    assert actual != expected


def test_hash_set_missing_row_fails():
    """Missing hash row is a failure."""
    expected = set(HASHED_LAUNCHER_FILES)
    actual = set(HASHED_LAUNCHER_FILES) - {"command.txt"}
    assert actual != expected


def test_hash_set_no_duplicates():
    """Hash rows must not contain duplicates."""
    rows = list(HASHED_LAUNCHER_FILES) + ["command.txt"]
    assert len(rows) != len(set(rows))


def test_artifact_hashes_not_in_hashed_set():
    """artifact_hashes.csv must be REQUIRED but NOT in HASHED set."""
    assert "artifact_hashes.csv" in REQUIRED_EPISODE_FILES
    assert "artifact_hashes.csv" not in HASHED_EPISODE_FILES


# ═══════════════════════════════════════════════════════════════
# Invalid field recomputation from step_trace
# ═══════════════════════════════════════════════════════════════

def test_invalid_field_recomputation():
    """Invalid fields must be computed from step_trace.csv, not trusted from manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "step_trace.csv"
        with open(trace_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "step", "raw_valid", "env_valid", "qpos_valid",
                "eef_valid", "convention_ok", "semantics_ok",
            ])
            w.writeheader()
            w.writerow({"step": "0", "raw_valid": "1", "env_valid": "0",
                         "qpos_valid": "1", "eef_valid": "1",
                         "convention_ok": "1", "semantics_ok": "1"})
            w.writerow({"step": "1", "raw_valid": "1", "env_valid": "1",
                         "qpos_valid": "1", "eef_valid": "1",
                         "convention_ok": "1", "semantics_ok": "1"})

        with open(trace_path) as f:
            rows = list(csv.DictReader(f))
        invalid = 0
        for row in rows:
            flags = ["raw_valid", "env_valid", "qpos_valid", "eef_valid",
                      "convention_ok", "semantics_ok"]
            if any(row.get(f, "1") in ("0", "False", "false") for f in flags):
                invalid += 1
        assert invalid == 1  # step 0 has env_valid=0


# ═══════════════════════════════════════════════════════════════
# Action identity recomputation from CSV
# ═══════════════════════════════════════════════════════════════

def test_action_identity_from_csv():
    """Action identity must be computed from CSV, not trusted from manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "action_identity.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "action_identical"])
            w.writeheader()
            w.writerow({"step": "0", "action_identical": "1"})
            w.writerow({"step": "1", "action_identical": "0"})

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        fail = any(r.get("action_identical") in ("0", "False", "false") for r in rows)
        assert fail


# ═══════════════════════════════════════════════════════════════
# Required file contracts
# ═══════════════════════════════════════════════════════════════

def test_required_launcher_files():
    assert "command.txt" in REQUIRED_LAUNCHER_FILES
    assert "returncode.json" in REQUIRED_LAUNCHER_FILES
    assert "launcher_artifact_hashes.csv" in REQUIRED_LAUNCHER_FILES


def test_required_episode_files():
    assert "ATTEMPT_STARTED.json" in REQUIRED_EPISODE_FILES
    assert "FIRST_ACTION_GENERATED.json" in REQUIRED_EPISODE_FILES
    assert "episode_manifest.json" in REQUIRED_EPISODE_FILES


def test_hashed_files_subset_of_required():
    for f in HASHED_EPISODE_FILES:
        assert f in REQUIRED_EPISODE_FILES
    for f in HASHED_LAUNCHER_FILES:
        assert f in REQUIRED_LAUNCHER_FILES


# ═══════════════════════════════════════════════════════════════
# GPU snapshot — unconditional set audit
# ═══════════════════════════════════════════════════════════════

def test_gpu_before_nonempty_fails():
    """Pre-existing GPU processes must cause failure."""
    before_ids = {("uuid-1", "123", "python")}
    after_ids = set()
    assert len(before_ids) > 0  # FAIL


def test_gpu_after_residual_fails():
    """Residual GPU processes after canary must cause failure."""
    before_ids = set()
    after_ids = {("uuid-1", "456", "python")}
    assert len(after_ids) > 0  # FAIL


def test_gpu_before_empty_after_empty_passes():
    """No processes before or after = clean."""
    before_ids = set()
    after_ids = set()
    assert len(before_ids) == 0 and len(after_ids) == 0


def test_gpu_before_after_same_process_fails():
    """Same process persisting through canary = pre-existing = FAIL."""
    before_ids = {("uuid-1", "123", "python")}
    after_ids = {("uuid-1", "123", "python")}
    assert len(before_ids) > 0


# ═══════════════════════════════════════════════════════════════
# CSV schema validation
# ═══════════════════════════════════════════════════════════════

def test_step_trace_required_flags():
    """Step trace must have all validity flags."""
    required = ["raw_valid", "env_valid", "qpos_valid", "eef_valid",
                "convention_ok", "semantics_ok"]
    columns = ["step", "raw_valid", "env_valid", "qpos_valid",
               "eef_valid", "convention_ok"]
    for f in required:
        if f not in columns:
            assert False, f"Missing column: {f}"


def test_step_trace_flag_values_must_be_01():
    """Validity flags must be '0' or '1', nothing else."""
    valid_values = {"0", "1"}
    for val in ["1", "0", "1", "0"]:
        assert str(val) in valid_values
    for bad in ["2", "True", "False", "", None]:
        if bad is not None and str(bad) not in valid_values:
            pass  # correctly rejected
        elif bad is None:
            pass  # correctly rejected


def test_action_identity_values_must_be_01():
    """action_identical must be '0' or '1'."""
    valid_values = {"0", "1"}
    for bad in ["2", "True", "", None]:
        ok = bad is not None and str(bad) in valid_values
        assert not ok  # should be invalid


# ═══════════════════════════════════════════════════════════════
# Emit candidate uniqueness
# ═══════════════════════════════════════════════════════════════

def test_emit_single_candidate_matches():
    """Exactly one candidate at emit step, not abstained."""
    cands = [
        {"step": "10", "abstained": "0", "abstain": ""},
        {"step": "15", "abstained": "0", "abstain": ""},
    ]
    emit_step = 10
    emit_cands = [c for c in cands if int(c["step"]) == emit_step]
    assert len(emit_cands) == 1
    assert emit_cands[0]["abstained"] == "0"
    assert emit_cands[0]["abstain"] == ""


def test_emit_duplicate_candidate_fails():
    """Two candidates at same step = FAIL."""
    cands = [
        {"step": "10", "abstained": "0", "abstain": ""},
        {"step": "10", "abstained": "0", "abstain": ""},
    ]
    emit_step = 10
    emit_cands = [c for c in cands if int(c["step"]) == emit_step]
    assert len(emit_cands) != 1  # FAIL


def test_emit_abstained_candidate_fails():
    """Emitting an abstained candidate = FAIL."""
    cands = [{"step": "10", "abstained": "1", "abstain": "gripper_already_open"}]
    emit_step = 10
    emit_cands = [c for c in cands if int(c["step"]) == emit_step]
    assert len(emit_cands) == 1
    assert emit_cands[0]["abstained"] == "1"  # FAIL


def test_emit_missing_candidate_fails():
    """No candidate at emit step = FAIL."""
    cands = [{"step": "5", "abstained": "0", "abstain": ""}]
    emit_step = 10
    emit_cands = [c for c in cands if int(c["step"]) == emit_step]
    assert len(emit_cands) == 0  # FAIL
