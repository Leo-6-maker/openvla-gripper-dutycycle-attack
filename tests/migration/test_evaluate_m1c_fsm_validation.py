#!/usr/bin/env python3
"""Tests for P5 evaluator — episode-level aggregation."""
import json, tempfile, csv, sys
from pathlib import Path
import numpy as np, torch, pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))

from gripper_attack.sc5_detector_runtime_v1r import SC5_FEATURES, SC5_PHASES, SC5MLP, SC5DetectorRuntimeV1R
from scripts.migration.evaluate_m1c_fsm_validation import evaluate_fsm


def _make_ckpt(path):
    sd = SC5MLP(n_feat=25).state_dict()
    torch.save({"model_state": sd, "mean": np.zeros(25, dtype=np.float32),
                "std": np.ones(25, dtype=np.float32),
                "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
                "dataset_sha256": "0"*64, "split_mode": "frozen"}, str(path))


def _make_cells(tmpdir, episodes):
    cells = []
    for task, state, rows_data in episodes:
        d = Path(tmpdir) / f"task{task}_state{state}"
        d.mkdir(parents=True)
        with open(d / "step_telemetry.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows_data[0].keys())
            w.writeheader()
            w.writerows(rows_data)
        cells.append({"task": task, "state": state, "path": d})
    return cells


def _row(step, cp=0.9, rp=0.001, pp="stable_carry"):
    return {"step": str(step), "corridor_p": str(cp), "release_p": str(rp), "pred_phase": pp, "feat_valid": "True"}


def test_synthetic_four_cells():
    """1 TV triggers, 1 TV not, 2 NC not (arm too late for guard). Coverage=0.5, Abstain=1.0."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "fake.pt"; _make_ckpt(ckpt)
        cells = _make_cells(tmp, [
            (0,0, [_row(i) for i in range(20)]),                      # TV-A: triggers
            (1,0, [_row(i, cp=0.05) for i in range(20)]),             # TV-B: won't trigger
            (2,0, [_row(i, cp=0.05) for i in range(20)]),             # NC-A: abstain
            (3,0, [_row(i, cp=0.05) for i in range(19)] + [_row(19, cp=0.9)]),  # NC-B: arms at step 19, guard unmet
        ])
        import scripts.migration.evaluate_m1c_fsm_validation as ev; ev.CKPT = ckpt
        teacher = {(0,0): {"teacher_valid": True, "teacher_anchor": 5},
                   (1,0): {"teacher_valid": True, "teacher_anchor": -1},
                   (2,0): {"teacher_valid": False, "teacher_anchor": -1},
                   (3,0): {"teacher_valid": False, "teacher_anchor": -1}}
        r = evaluate_fsm(SC5DetectorRuntimeV1R, {"tau_corridor":0.3,"tau_release":0.3,"guard":5}, cells, teacher)
        assert r["n_tv"] == 2; assert r["n_nc"] == 2
        assert r["gates"]["coverage"] == 0.5
        assert r["gates"]["no_corridor_abstain"] == 1.0


def test_coverage_bounded():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "fake.pt"; _make_ckpt(ckpt)
        cells = _make_cells(tmp, [ (i,0, [_row(j) for j in range(20)]) for i in range(4) ])
        import scripts.migration.evaluate_m1c_fsm_validation as ev; ev.CKPT = ckpt
        teacher = {(i,0): {"teacher_valid": True, "teacher_anchor": 5} for i in range(4)}
        r = evaluate_fsm(SC5DetectorRuntimeV1R, {"tau_corridor":0.3,"tau_release":0.3,"guard":5}, cells, teacher)
        assert 0 <= r["gates"]["coverage"] <= 1


def test_r2_different_from_r1():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "fake.pt"; _make_ckpt(ckpt)
        cells = _make_cells(tmp, [ (0,0, [_row(i, cp=0.4) for i in range(20)]) ])
        import scripts.migration.evaluate_m1c_fsm_validation as ev; ev.CKPT = ckpt
        teacher = {(0,0): {"teacher_valid": True, "teacher_anchor": 5}}
        r1 = evaluate_fsm(SC5DetectorRuntimeV1R, {"tau_corridor":0.3,"tau_release":0.3,"guard":5}, cells, teacher)
        r2 = evaluate_fsm(SC5DetectorRuntimeV1R, {"tau_on":0.5,"tau_off":0.3,"n_candidate":3,"max_arm_age":50,"guard":5,"tau_release":0.3,"fsm_version":"v1r_r2"}, cells, teacher)
        assert r1["gates"]["coverage"] != r2["gates"]["coverage"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
