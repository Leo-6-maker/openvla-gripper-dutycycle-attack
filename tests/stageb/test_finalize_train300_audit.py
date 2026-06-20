import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb import finalize_train300_audit as audit  # noqa: E402


REQUIRED = [
    "episode_manifest.json",
    "episode_summary.json",
    "step_telemetry.csv",
    "detector_telemetry.csv",
    "frame_index.csv",
    "agentview_frames_uint8.npz",
    "sim_state_stream.npz",
    "sim_state_manifest.json",
    "rollout_raw.mp4",
    "rollout_overlay.mp4",
    "video_manifest.json",
    "artifact_sha256.json",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _master() -> dict[str, str]:
    return {
        "canonical_key": "libero_spatial|0|10|0|CLEAN",
        "suite": "libero_spatial",
        "task_idx": "0",
        "state_id": "10",
        "eval_seed": "0",
        "condition": "CLEAN",
        "unnorm_key": "libero_spatial",
        "model_path": "/models/spatial",
    }


def _make_episode(tmp_path: Path, *, n_steps: int = 3) -> tuple[Path, dict[str, str], dict[str, str]]:
    ep = tmp_path / "episode"
    master = _master()
    manifest = {
        "suite": "libero_spatial",
        "task_idx": 0,
        "state_id": 10,
        "eval_seed": 0,
        "condition": "CLEAN",
        "attack_enabled": False,
        "source_commit": audit.EXPECTED_SOURCE_COMMIT,
        "unnorm_key": "libero_spatial",
        "model_path": "/models/spatial",
    }
    summary = {
        "suite": "libero_spatial",
        "task_idx": 0,
        "state_id": 10,
        "eval_seed": 0,
        "condition": "CLEAN",
        "vis_or_rand_run": False,
        "task_success": True,
        "n_steps": n_steps,
        "invalid_feature_steps": 0,
    }
    _write_json(ep / "episode_manifest.json", manifest)
    _write_json(ep / "episode_summary.json", summary)
    _write_json(
        ep / "sim_state_manifest.json",
        {
            "steps": n_steps,
            "arrays": {name: [n_steps, 1] for name in audit.REQUIRED_SIM_ARRAYS},
        },
    )
    _write_json(ep / "video_manifest.json", {"steps": n_steps})
    for rel in ["agentview_frames_uint8.npz", "sim_state_stream.npz", "rollout_raw.mp4", "rollout_overlay.mp4"]:
        (ep / rel).write_bytes(f"{rel}:{n_steps}".encode())
    rows = [{"step": i} for i in range(n_steps)]
    _write_csv(ep / "step_telemetry.csv", rows)
    _write_csv(ep / "detector_telemetry.csv", rows)
    _write_csv(ep / "frame_index.csv", rows)
    files = []
    for rel in REQUIRED:
        if rel == "artifact_sha256.json":
            continue
        files.append({"path": rel, "sha256": _sha(ep / rel), "size": (ep / rel).stat().st_size})
    _write_json(ep / "artifact_sha256.json", {"files": files, "recursive_sha256": "seal"})
    row = {"canonical_key": master["canonical_key"], "output_dir": str(ep)}
    return ep, master, row


def _patch_decoders(monkeypatch, *, n_steps=3):
    monkeypatch.setattr(audit, "mp4_frame_count", lambda path: (n_steps, ""))
    monkeypatch.setattr(audit, "npz_first_dim", lambda path, member: (n_steps, ""))


def _run(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    _, master, row = _make_episode(tmp_path)
    details, counters, problems = audit.audit_primary_episode(
        master["canonical_key"], master, row, REQUIRED
    )
    return details, counters, problems


def test_finalize_accepts_valid_episode_with_hardened_checks(tmp_path, monkeypatch):
    _, counters, problems = _run(tmp_path, monkeypatch)
    assert problems == []
    assert all(value == 0 for value in counters.values())


def test_identity_mismatch_fails(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    summary = json.loads((ep / "episode_summary.json").read_text())
    summary["state_id"] = 11
    _write_json(ep / "episode_summary.json", summary)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["identity_mismatch_count"] == 1
    assert "state_id" in details["identity_mismatches"]


def test_clean_contract_rejects_attack_flag(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    manifest = json.loads((ep / "episode_manifest.json").read_text())
    manifest["attack_enabled"] = True
    _write_json(ep / "episode_manifest.json", manifest)
    _, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["clean_contract_failure_count"] == 1


def test_invalid_feature_steps_fail(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    summary = json.loads((ep / "episode_summary.json").read_text())
    summary["invalid_feature_steps"] = 1
    _write_json(ep / "episode_summary.json", summary)
    _, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["invalid_feature_episode_count"] == 1


def test_sim_manifest_step_mismatch_fails(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    sim = json.loads((ep / "sim_state_manifest.json").read_text())
    sim["steps"] = 2
    _write_json(ep / "sim_state_manifest.json", sim)
    _, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["sim_manifest_step_mismatch_count"] == 1


def test_sim_array_missing_fails(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    sim = json.loads((ep / "sim_state_manifest.json").read_text())
    sim["arrays"].pop("qpos")
    _write_json(ep / "sim_state_manifest.json", sim)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["sim_array_missing_count"] == 1
    assert "qpos" in details["sim_array_missing_or_unreadable"]


def test_sim_array_length_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "mp4_frame_count", lambda path: (3, ""))

    def fake_npz(path, member):
        return (2, "") if member == "qvel" else (3, "")

    monkeypatch.setattr(audit, "npz_first_dim", fake_npz)
    _, master, row = _make_episode(tmp_path)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["sim_array_length_mismatch_count"] == 1
    assert "qvel" in details["sim_array_length_mismatch"]


def test_media_decode_failure_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "mp4_frame_count", lambda path: (None, "decode_error") if path.name == "rollout_raw.mp4" else (3, ""))
    monkeypatch.setattr(audit, "npz_first_dim", lambda path, member: (3, ""))
    _, master, row = _make_episode(tmp_path)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["media_decode_failure_count"] == 1
    assert "rollout_raw.mp4" in details["media_errors"]


def test_artifact_manifest_coverage_failure_fails(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    artifact = json.loads((ep / "artifact_sha256.json").read_text())
    artifact["files"] = [item for item in artifact["files"] if item["path"] != "sim_state_stream.npz"]
    _write_json(ep / "artifact_sha256.json", artifact)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["artifact_manifest_coverage_failure_count"] == 1
    assert "sim_state_stream.npz" in details["artifact_manifest_missing"]


def test_artifact_sha_mismatch_fails(tmp_path, monkeypatch):
    _patch_decoders(monkeypatch)
    ep, master, row = _make_episode(tmp_path)
    summary = json.loads((ep / "episode_summary.json").read_text())
    summary["mlp_emit_step"] = -1
    _write_json(ep / "episode_summary.json", summary)
    details, counters, _ = audit.audit_primary_episode(master["canonical_key"], master, row, REQUIRED)
    assert counters["artifact_sha_mismatch_count"] == 1
    assert "episode_summary.json" in details["artifact_sha_bad"]
