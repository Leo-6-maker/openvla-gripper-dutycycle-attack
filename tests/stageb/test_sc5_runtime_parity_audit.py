import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5MLP, SC5_FEATURES, SC5_PHASES
from scripts.stageb.audit_sc5_runtime_parity import audit, episode_emit_from_prediction_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint(path: Path, *, phase_name: str = "stable_carry") -> None:
    model = SC5MLP(n_feat=len(SC5_FEATURES))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.phase_head.bias[SC5_PHASES.index(phase_name)] = 3.0
        model.corridor_head.bias[0] = 3.0
        model.release_head.bias[0] = -3.0
    torch.save(
        {
            "feature_names": SC5_FEATURES,
            "phase_classes": SC5_PHASES,
            "mean": np.zeros(len(SC5_FEATURES), dtype=np.float32),
            "std": np.ones(len(SC5_FEATURES), dtype=np.float32),
            "model_state": model.state_dict(),
            "dataset_sha256": "dataset-sha",
            "split_mode": "provisional_cross_suite_frozen",
            "selected_tau_corridor": 0.9,
            "selected_tau_release": 0.1,
        },
        path,
    )


def test_episode_emit_from_prediction_rows_uses_frozen_fsm():
    rows = [
        {"step": 0, "pred_phase": "stable_carry", "corridor_p": 0.95, "release_p": 0.05},
        {"step": 4, "pred_phase": "stable_carry", "corridor_p": 0.95, "release_p": 0.05},
        {"step": 5, "pred_phase": "stable_carry", "corridor_p": 0.95, "release_p": 0.05},
    ]
    assert episode_emit_from_prediction_rows(rows, 0.9, 0.1, guard=5) == 5


def test_runtime_parity_audit_passes_on_matching_rows(tmp_path):
    ckpt = tmp_path / "model.pt"
    dataset = tmp_path / "dataset.csv"
    predictions = tmp_path / "predictions.csv"
    out = tmp_path / "out"
    _write_checkpoint(ckpt)
    corridor = float(torch.sigmoid(torch.tensor(3.0)))
    release = float(torch.sigmoid(torch.tensor(-3.0)))
    dataset_rows = []
    prediction_rows = []
    for step in range(6):
        base = {
            "episode_key": "libero_goal|0|20|0|CLEAN",
            "suite": "libero_goal",
            "task_idx": "0",
            "state_id": "20",
            "eval_seed": "0",
            "condition": "CLEAN",
            "dataset_split": "test",
            "step": step,
            "teacher_status": "ELIGIBLE_EVENT",
            "label_role": "primary_single_object_pick_place",
            "primary_or_supplementary": "primary",
            "teacher_phase": "stable_carry",
            "teacher_corridor_active": "1",
            "teacher_release_active": "0",
            "teacher_window_start": "0",
            "teacher_window_end": "5",
            "teacher_anchor_step": "0",
        }
        for feature in SC5_FEATURES:
            base[feature] = "0.0"
        dataset_rows.append(dict(base))
        prediction_rows.append(
            {
                "episode_key": base["episode_key"],
                "suite": base["suite"],
                "task_idx": base["task_idx"],
                "state_id": base["state_id"],
                "dataset_split": base["dataset_split"],
                "step": step,
                "teacher_status": base["teacher_status"],
                "label_role": base["label_role"],
                "primary_or_supplementary": base["primary_or_supplementary"],
                "teacher_phase": base["teacher_phase"],
                "teacher_corridor_active": "1",
                "teacher_release_active": "0",
                "teacher_window_start": "0",
                "teacher_window_end": "5",
                "teacher_anchor_step": "0",
                "pred_phase": "stable_carry",
                "corridor_p": corridor,
                "release_p": release,
            }
        )
    _write_csv(dataset, dataset_rows)
    _write_csv(predictions, prediction_rows)

    class Args:
        pass

    args = Args()
    args.dataset = str(dataset)
    args.predictions = str(predictions)
    args.checkpoint = str(ckpt)
    args.suite = "libero_goal"
    args.guard = 5
    args.tolerance = 1e-7
    args.write_all_rows = False
    args.output_dir = str(out)
    summary = audit(args)
    assert summary["parity_pass"] is True
    assert summary["emit_mismatch_count"] == 0
    assert summary["runtime_emit_positive_count"] == 1
