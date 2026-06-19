import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.cross_suite_layer1_resolver import (  # noqa: E402
    build_blind_review_manifest,
    build_dev_canary_manifest,
    build_review_package,
    load_ontology,
    load_step_rows,
    resolve_episode,
    run_resolver,
    validate_episode_rows,
)


ONTOLOGY = REPO / "configs" / "cross_suite_task_ontology_v1.yaml"


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _episode(tmp_path: Path, name: str, *, suite="libero_spatial", task_idx=0, state_id=0):
    ep = tmp_path / name
    _write_json(
        ep / "episode_manifest.json",
        {
            "condition": "CLEAN",
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "eval_seed": 0,
        },
    )
    _write_json(
        ep / "episode_summary.json",
        {
            "condition": "CLEAN",
            "suite": suite,
            "task_idx": task_idx,
            "state_id": state_id,
            "eval_seed": 0,
            "n_steps": 4,
            "task_success": True,
        },
    )
    _write_json(
        ep / "sim_state_manifest.json",
        {
            "arrays": {"qpos": [4, 2]},
            "metadata": {
                "body_names": ["world", "black_bowl_1_main", "plate_1_main", "robot0_base"],
                "site_names": ["plate_1_default_site", "gripper0_grip_site"],
                "joint_names": ["black_bowl_1_joint0"],
            },
        },
    )
    _write_csv(
        ep / "step_telemetry.csv",
        [
            {
                "step": 0,
                "raw_gripper": 1.0,
                "env_gripper": -1.0,
                "gripper_qpos": 0.03,
                "gripper_opening_proxy": 0.03,
                "eef_x": 0,
                "eef_y": 0,
                "eef_z": 0,
                "mlp_emit": 99,
                "mlp_triggered": True,
                "corridor_p": 0.9,
                "release_p": 0.1,
                "pred_phase": "release",
            },
            {
                "step": 1,
                "raw_gripper": 0.0,
                "env_gripper": 1.0,
                "gripper_qpos": 0.02,
                "gripper_opening_proxy": 0.02,
                "eef_x": 0,
                "eef_y": 0,
                "eef_z": 0,
                "mlp_emit": 99,
                "mlp_triggered": True,
                "corridor_p": 0.9,
                "release_p": 0.1,
                "pred_phase": "release",
            },
        ],
    )
    (ep / "rollout_raw.mp4").write_bytes(b"video")
    np.savez(ep / "sim_state_stream.npz", qpos=np.zeros((4, 2), dtype=np.float32))
    return ep


def _ledger_row(ep: Path, *, suite="libero_spatial", task_idx=0, state_id=0, success="True", sha="abc"):
    return {
        "canonical_key": f"{suite}|{task_idx}|{state_id}|0|CLEAN",
        "episode_path": str(ep),
        "suite": suite,
        "task_idx": str(task_idx),
        "state_id": str(state_id),
        "eval_seed": "0",
        "condition": "CLEAN",
        "status": "COMPLETE_VALID",
        "task_success": success,
        "n_steps": "4",
        "artifact_recursive_sha256": sha,
    }


def test_load_step_rows_drops_forbidden_detector_fields(tmp_path):
    ep = _episode(tmp_path, "ep")
    rows = load_step_rows(ep / "step_telemetry.csv")
    assert rows
    assert "raw_gripper" in rows[0]
    for forbidden in ["mlp_emit", "mlp_triggered", "corridor_p", "release_p", "pred_phase"]:
        assert forbidden not in rows[0]


def test_single_object_episode_resolves_to_manual_review_event(tmp_path):
    ep = _episode(tmp_path, "ep")
    ontology = load_ontology(ONTOLOGY)
    task = ontology[("libero_spatial", 0)]
    episode, events = resolve_episode(_ledger_row(ep), task, teacher_run_id="dev")
    assert episode["teacher_status"] == "ELIGIBLE_EVENT"
    assert episode["manual_review_required"] is True
    assert episode["object_binding_status"] in {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY"}
    assert episode["target_binding_status"] in {"BOUND_EXACT", "BOUND_BDDL_ONTOLOGY"}
    assert len(events) == 1
    assert events[0]["teacher_anchor_step"] == 1


def test_negative_and_multi_event_status_invariants(tmp_path):
    ep_goal = _episode(tmp_path, "goal", suite="libero_goal", task_idx=0)
    ep_l10 = _episode(tmp_path, "l10", suite="libero_10", task_idx=4)
    ontology = load_ontology(ONTOLOGY)
    neg, _ = resolve_episode(_ledger_row(ep_goal, suite="libero_goal", task_idx=0), ontology[("libero_goal", 0)], teacher_run_id="dev")
    multi, _ = resolve_episode(_ledger_row(ep_l10, suite="libero_10", task_idx=4), ontology[("libero_10", 4)], teacher_run_id="dev")
    assert neg["teacher_status"] == "CORRECT_SEMANTIC_ABSTAIN"
    assert multi["teacher_status"] == "MULTI_EVENT_AUDIT_ONLY"
    assert validate_episode_rows([neg, multi]) == []


def test_dev_and_blind_manifests_are_deterministic_and_disjoint(tmp_path):
    ontology = load_ontology(ONTOLOGY)
    rows = []
    for suite, task_idx in [
        ("libero_spatial", 0),
        ("libero_spatial", 1),
        ("libero_spatial", 2),
        ("libero_spatial", 3),
        ("libero_goal", 1),
        ("libero_goal", 2),
        ("libero_goal", 0),
        ("libero_goal", 5),
        ("libero_10", 5),
        ("libero_10", 5),
        ("libero_10", 4),
        ("libero_10", 6),
    ]:
        idx = len(rows)
        ep = _episode(tmp_path, f"ep{idx}", suite=suite, task_idx=task_idx, state_id=idx)
        rows.append(_ledger_row(ep, suite=suite, task_idx=task_idx, state_id=idx, sha=f"sha{idx}"))
    dev = build_dev_canary_manifest(rows, ontology)
    blind = build_blind_review_manifest(rows, ontology, exclude_keys={r["canonical_key"] for r in dev["selected"]}, count=4)
    assert dev["selected_count"] == 12
    assert {r["canonical_key"] for r in dev["selected"]}.isdisjoint({r["canonical_key"] for r in blind["selected"]})


def test_run_resolver_and_blind_package_outputs(tmp_path):
    ep = _episode(tmp_path, "ep")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "selected": [_ledger_row(ep)],
        },
    )
    out = tmp_path / "resolver"
    result = run_resolver(manifest, ONTOLOGY, out, teacher_run_id="dev")
    assert result["episode_count"] == 1
    assert result["event_count"] == 1
    package = build_review_package(manifest, out, tmp_path / "review")
    assert package["review_count"] == 1
    review_csv = (tmp_path / "review" / "blind_review_queue.csv").read_text(encoding="utf-8")
    assert "human_binding_accept" in review_csv
    assert "mlp_emit" not in review_csv
