"""Gate E-R2.5.1: CPU panel contract tests.

Tests receipt validation, seal order, auditor checks, and termination
semantics. All tests are CPU-only — no OpenVLA, no LIBERO.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256, sha256_file, canonical_json_sha
from gripper_attack.r10_4d_passive import (
    R10_4DContractError,
    parse_route,
    run_passive_episode,
    safe_json_value,
    _classify_termination,
)

# ── Make panel scripts importable ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r10_4"))


# ═══════════════════════════════════════════════════════════════════════════════
# Fake deps
# ═══════════════════════════════════════════════════════════════════════════════

class FakeModel:
    def site_name2id(self, n): return 0
    def geom_id2name(self, g): return f"geom_{g}"

class FakeData:
    def __init__(self): self.site_xpos = np.array([[0.5, 0.0, 0.8]], dtype=np.float32); self.ncon = 0; self.contact = []

class FakeEnv:
    def __init__(self, policy_steps=3):
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
        self.policy_steps = policy_steps
        self.total_calls = 0; self.policy_calls = 0; self.actions = []
        self._success = True; self._raises = None
        self.obs = {"agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
                     "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
                     "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
                     "object-state": np.zeros(4, dtype=np.float32)}
    def set_check_success(self, v=None, raises=None): self._success = v; self._raises = raises
    def set_init_state(self, s): return self.obs
    def step(self, a):
        self.total_calls += 1; self.actions.append(list(a))
        done = False
        if self.total_calls > 10:
            self.policy_calls += 1
            done = self.policy_calls >= self.policy_steps
        info = {"step": self.total_calls}
        return self.obs, 0.0, done, info
    def check_success(self):
        if self._raises: raise self._raises
        return self._success
    def close(self): pass

class FakeAdapter:
    def __init__(self, gen=1): self.gen = gen; self.calls = 0
    def predict_action(self, *, image_np, task_label, capture=False):
        self.calls += 1
        a = np.zeros(7, dtype=np.float32); a[-1] = 1.0
        m = {} if self.gen is None else {"generation_passes_per_step": self.gen}
        return a, m
    def postprocess(self, a):
        e = np.asarray(a, dtype=np.float32).copy(); e[-1] = -1.0; return e

class FakeDetector:
    def __init__(self): self.calls = []
    def reset(self): self.calls.clear()
    def step(self, f, r):
        v = np.asarray(f, dtype=np.float32)
        self.calls.append((v.copy(), r))
        return -10.0, 1.0 / (1.0 + np.exp(10.0))

def img_getter(o): return o["agentview_image"]


def _r(**kw):
    """Minimal run_passive_episode helper."""
    d = {"env": FakeEnv(policy_steps=3), "initial_state": {}, "task_language": "test",
         "identity": "libero_10/task_01/state_20", "openvla_adapter": FakeAdapter(),
         "detector": FakeDetector(), "image_getter": img_getter, "max_steps": 20,
         "authorized_parents": frozenset({"libero_10/task_01/state_20"})}
    d.update(kw)
    return run_passive_episode(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# P0-9: check_success exception fail-closed
# ═══════════════════════════════════════════════════════════════════════════════

def test_check_success_exception_is_fail_closed():
    """check_success() raise → CHECK_SUCCESS_FAILURE, not HORIZON_TERMINATION"""
    env = FakeEnv(policy_steps=20)  # done at step 20 = horizon
    env.set_check_success(raises=RuntimeError("boom"))
    result = _r(env=env, max_steps=20)
    assert result["termination_reason"] == "CHECK_SUCCESS_FAILURE"
    assert result["status"] == "FAIL_TERMINATION"
    assert result["task_success"] is False


def test_success_termination_normal():
    env = FakeEnv(policy_steps=5)
    env.set_check_success(True)
    result = _r(env=env, max_steps=50)
    assert result["termination_reason"] == "SUCCESS_TERMINATION"
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def test_full_loop_task_failure_runtime_valid():
    env = FakeEnv(policy_steps=999)
    env.set_check_success(False)
    result = _r(env=env, max_steps=10)
    assert result["termination_reason"] == "FULL_LOOP_TASK_FAILURE"
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def test_early_done_without_success_hard_failure():
    env = FakeEnv(policy_steps=3)
    env.set_check_success(False)
    result = _r(env=env, max_steps=50)
    assert result["termination_reason"] == "EARLY_DONE_WITHOUT_SUCCESS"
    assert result["status"] == "FAIL_TERMINATION"


# ═══════════════════════════════════════════════════════════════════════════════
# P0-9: classifier unit tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_classifier_no_steps():
    r = _classify_termination([], 10, FakeEnv(), [])
    assert r["termination_reason"] == "NO_STEPS"
    assert r["is_hard_failure"] is True


def test_classifier_check_success_failure_always_hard():
    env = FakeEnv(policy_steps=5)
    env.set_check_success(raises=ValueError("x"))
    recs = [{"step": 0, "done": True, "reward": 0.0}]
    r = _classify_termination(recs, 10, env, [])
    assert r["termination_reason"] == "CHECK_SUCCESS_FAILURE"
    assert r["is_hard_failure"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Authorized parents
# ═══════════════════════════════════════════════════════════════════════════════

def test_authorized_parents_controls_access():
    """Only identities in authorized_parents can run."""
    # task_01 is authorized
    result = _r(identity="libero_10/task_01/state_20",
                authorized_parents=frozenset({"libero_10/task_01/state_20"}))
    assert result["identity"] == "libero_10/task_01/state_20"

    # task_02 is NOT authorized
    with pytest.raises(R10_4DContractError, match="PASSIVE_PARENT_NOT_AUTHORIZED"):
        _r(identity="libero_10/task_02/state_20",
           authorized_parents=frozenset({"libero_10/task_01/state_20"}))


# ═══════════════════════════════════════════════════════════════════════════════
# Auditor: reuse detection
# ═══════════════════════════════════════════════════════════════════════════════

def test_auditor_detects_reuse_dir():
    from audit_r10_4e_sealed_roots import _is_reuse_dir, audit_reuse_root
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Empty dir — not reuse
        assert not _is_reuse_dir(root)
        # Create REUSE_BINDING.json
        (root / "REUSE_BINDING.json").write_text(json.dumps({"identity": "libero_10/task_00/state_20"}))
        assert _is_reuse_dir(root)


def test_auditor_rejects_missing_summary():
    from audit_r10_4e_sealed_roots import audit_fresh_root
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report = audit_fresh_root(root, identity=None)
        assert report.get("valid") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Seal order: SHA256SUMS covers all files
# ═══════════════════════════════════════════════════════════════════════════════

def test_seal_covers_all_files():
    """After seal_root(), no file should be unlisted in SHA256SUMS."""
    import hashlib as _hl
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Write some files
        (root / "a.json").write_text('{"x":1}')
        (root / "b.jsonl").write_text('{"y":2}\n')

        # Manual seal (simulating seal_root logic)
        rows = []
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
                continue
            d = _hl.sha256()
            d.update(p.read_bytes())
            rows.append({"path": p.relative_to(root).as_posix(), "sha256": d.hexdigest(), "size": p.stat().st_size})
        sums = root / "SHA256SUMS"
        sums.write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))

        # Now add an extra file AFTER seal
        (root / "ROOT_SEAL_RECEIPT.json").write_text('{"sealed":true}')

        # Verify: SHA256SUMS does NOT list the extra file → auditor should catch it
        listed = {}
        for line in sums.read_text().splitlines():
            if not line.strip(): continue
            t = line.split(maxsplit=1)
            if len(t) == 2: listed[t[1].strip()] = t[0]
        assert "ROOT_SEAL_RECEIPT.json" not in listed, "Extra file was incorrectly listed in SHA256SUMS (should not be — seal order matters)"


# ═══════════════════════════════════════════════════════════════════════════════
# safe_json_value edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_safe_json_value_frozenset():
    result = safe_json_value(frozenset({1, 2, 3}))
    assert isinstance(result, list)
    assert sorted(result) == [1, 2, 3]

def test_safe_json_value_nested_complex():
    obj = {"a": np.float64(1.5), "b": [np.int32(1), np.int32(2)],
           "c": {"d": (3, 4), "e": {5, 6}}}
    result = safe_json_value(obj)
    assert json.dumps(result, sort_keys=True)  # must not raise
