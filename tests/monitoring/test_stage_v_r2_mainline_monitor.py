from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring import run_stage_v_r2_mainline_monitor as monitor


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_args(tmp_path: Path, repo: Path) -> object:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    old = tmp_path / "old"
    write_json(old / "ABORTED_INCOMPLETE.json", {"status": "ABORTED_INCOMPLETE", "accepted_parent_results": 0, "scientific_validity": 0})
    old_manifest = tmp_path / "old-manifest.json"
    write_json(old_manifest, {"parents": []})
    candidate = tmp_path / "candidate.json"
    write_json(candidate, {"parents": [{"canonical_parent_key": "libero_goal/task_00/state_48", "old_artifacts_reused": False}]})
    postmortem = tmp_path / "postmortem.json"
    policy = tmp_path / "policy.json"
    science = tmp_path / "science.json"
    for path in (postmortem, policy, science):
        write_json(path, {"status": "PASS"})
    return type("Args", (), {
        "repo_root": repo, "monitor_root": tmp_path / "monitor", "lock_path": tmp_path / "monitor.lock",
        "formal_root_parent": tmp_path / "formal", "expected_source_commit": commit, "expected_source_tree": tree,
        "old_root": old, "old_manifest": old_manifest, "old_manifest_sha256": sha(old_manifest),
        "candidate_manifest": candidate, "candidate_manifest_sha256": sha(candidate),
        "postmortem": postmortem, "postmortem_sha256": sha(postmortem),
        "timeout_policy": policy, "timeout_policy_sha256": sha(policy),
        "science_provenance": science, "science_provenance_sha256": sha(science),
        "qualification_root": None, "canary_root": None, "r2a_root": None, "v2_root": None, "stage_o_root": None,
        "required_gpus": 8, "excluded_gpus": [5], "protected_pids": [], "external_pid": 0,
        "canary_peak_mib": 1000, "minimum_ram_gib": 128, "poll_seconds": 1, "once": True,
    })()


def fake_system(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, int]]) -> None:
    monkeypatch.setattr(monitor, "_mem_snapshot", lambda: {
        "available_ram_gib": 900, "swap_used_bytes": 0, "swap_in": 0, "swap_out": 0, "oom_kill": 0,
    })
    monkeypatch.setattr(monitor, "read_gpu_snapshot", lambda: (rows, None))
    monkeypatch.setattr(monitor, "read_xid_status", lambda start: ("CLEAR", None))
    monkeypatch.setattr(monitor, "_compute_pids", lambda: set())


def gpu_rows(count: int = 8) -> list[dict[str, int]]:
    return [{"index": gpu, "memory_free_mib": 100000} for gpu in range(count)]


def test_resource_gate_never_approves_gpu5_or_silently_downgrades() -> None:
    result = monitor.resource_gate(
        gpu_rows=gpu_rows(), gpu_error=None,
        memory={"available_ram_gib": 900, "swap_used_bytes": 0, "swap_in": 0, "swap_out": 0, "oom_kill": 0},
        xid_status="CLEAR", xid_error=None, baseline_oom=0, required_gpus=8, excluded_gpus={5},
        protected_pids=set(), canary_peak_mib=1000, swap_bad_streak=0, minimum_ram_gib=128,
    )
    assert result["verdict"] == monitor.WAITING
    assert len(result["safe_gpus"]) == 7
    assert 5 not in result["safe_gpus"]


def test_pid_alive_is_true_only_for_live_pid() -> None:
    assert monitor.pid_alive(__import__("os").getpid()) is True
    assert monitor.pid_alive(2**31 - 1) is False


def test_monitor_waiting_does_not_create_formal_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = make_args(tmp_path, repo)
    fake_system(monkeypatch, gpu_rows())
    monkeypatch.setattr(monitor, "source_binding", lambda _: {"commit": args.expected_source_commit, "tree": args.expected_source_tree, "status_porcelain": ""})
    instance = monitor.MainlineMonitor(args)
    assert instance.tick() == monitor.WAITING
    assert not list((tmp_path / "formal").glob("STAGE_V_R2A_COUNTERFACTUAL_MAP_*"))
    state = json.loads((tmp_path / "monitor" / "STAGE_V_R2_MAINLINE_STATE.json").read_text(encoding="utf-8"))
    assert state["formal_root_created"] is False
    assert state["old_roots_reused"] is False
    assert state["eval160_reads"] == 0


def test_monitor_writes_one_receipt_per_state_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = make_args(tmp_path, repo)
    fake_system(monkeypatch, gpu_rows())
    monkeypatch.setattr(monitor, "source_binding", lambda _: {"commit": args.expected_source_commit, "tree": args.expected_source_tree, "status_porcelain": ""})
    instance = monitor.MainlineMonitor(args)
    assert instance.tick() == monitor.WAITING
    assert instance.tick() == monitor.WAITING
    receipts = list((tmp_path / "monitor" / "TRANSITION_RECEIPTS").glob("*.json"))
    assert len(receipts) == 1
    value = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert value["from_status"] == "NONE"
    assert value["to_status"] == monitor.WAITING
    assert len(value["state_sha256"]) == 64


def test_source_drift_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = make_args(tmp_path, repo)
    args.expected_source_commit = "wrong"
    fake_system(monkeypatch, gpu_rows(9))
    instance = monitor.MainlineMonitor(args)
    assert instance.tick() == "HARD_STOP"
    state = json.loads((tmp_path / "monitor" / "STAGE_V_R2_MAINLINE_STATE.json").read_text(encoding="utf-8"))
    assert "SOURCE_OR_TREE_MISMATCH" in state["hard_stop_reasons"]


def test_missing_preparation_binding_stays_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = make_args(tmp_path, repo)
    args.postmortem = None
    args.postmortem_sha256 = ""
    fake_system(monkeypatch, gpu_rows(9))
    monkeypatch.setattr(monitor, "source_binding", lambda _: {"commit": args.expected_source_commit, "tree": args.expected_source_tree, "status_porcelain": ""})
    instance = monitor.MainlineMonitor(args)
    assert instance.tick() == "PREPARATION"


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl behavior is covered by existing Windows fallback tests")
def test_duplicate_mainline_monitor_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    args = make_args(tmp_path, repo)
    fake_system(monkeypatch, gpu_rows())
    first = monitor.MainlineMonitor(args)
    first.lock.acquire(first.monitor_root)
    second = monitor.MainlineMonitor(args)
    with pytest.raises(monitor.MonitorError, match="DUPLICATE_MONITOR"):
        second.run()
    first.lock.close()
