import csv
import json
from pathlib import Path

from scripts.stageb.audit_c2_control_ablation_application import build_rows, main


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_diff(path: Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["field"])
        writer.writeheader()
        for field in fields:
            writer.writerow({"field": field})


def make_c2_dir(tmp_path: Path) -> Path:
    root = tmp_path / "c2"
    write_json(
        root / "reference_mutable_control_state.json",
        {
            "robots": [
                {
                    "controller_goal": {"goal_pos": {"sha256": "a"}, "goal_ori": {"sha256": "b"}},
                    "interpolator_pos": None,
                    "interpolator_ori": None,
                    "action_history": {
                        "recent_actions": {
                            "dim": 7,
                            "last": {"sha256": "c"},
                            "current": {"sha256": "d"},
                        }
                    },
                }
            ],
            "env_counters": {"timestep": 61, "cur_time": 3.05},
            "qacc": {"sha256": "e"},
        },
    )
    write_json(root / "c2_control_state_ablation_summary.json", {"result": "C2_ONE_STEP_POST_ACTION_STILL_DIVERGES"})
    specs = {
        "A2_GOAL_STATE": ["robot0.controller.goal_pos", "robot0.controller.goal_ori"],
        "A3_GOAL_INTERPOLATOR_STATE": ["robot0.controller.goal_pos", "robot0.controller.goal_ori"],
        "A4_GOAL_INTERPOLATOR_ACTION_HISTORY": [
            "robot0.controller.goal_pos",
            "robot0.controller.goal_ori",
            "robot0.recent_actions.dim",
            "robot0.recent_actions.last",
            "env_inner.timestep",
        ],
        "A5_QACC_ABLATION": [
            "robot0.controller.goal_pos",
            "robot0.controller.goal_ori",
            "robot0.recent_actions.dim",
            "robot0.recent_actions.last",
            "env_inner.timestep",
            "mujoco.qacc",
        ],
    }
    for ablation, actions in specs.items():
        write_json(root / ablation / "applied_ablation.json", {"actions": actions})
        write_diff(root / ablation / "pre_diff.csv", ["robots[0].attrs.torques.sha256"])
    for ablation in ["A0_BASELINE", "A1_DERIVED_RECOMPUTE"]:
        write_json(root / ablation / "applied_ablation.json", {"actions": []})
        write_diff(root / ablation / "pre_diff.csv", ["robots[0].attrs.torques.sha256"])
    return root


def row_for(rows, ablation, group):
    return next(row for row in rows if row["ablation"] == ablation and row["requested_group"] == group)


def test_c2_application_auditor_marks_no_applicable_interpolator_state(tmp_path):
    rows = build_rows(make_c2_dir(tmp_path))

    a2_goal = row_for(rows, "A2_GOAL_STATE", "goal")
    assert a2_goal["status"] == "APPLIED_AND_MATCHED"
    assert a2_goal["was_effective_intervention"] is True

    a3_interp = row_for(rows, "A3_GOAL_INTERPOLATOR_STATE", "interpolator")
    assert a3_interp["status"] == "NO_APPLICABLE_INTERPOLATOR_STATE"
    assert a3_interp["applied_action_count"] == 0
    assert a3_interp["was_effective_intervention"] is False

    a4_history = row_for(rows, "A4_GOAL_INTERPOLATOR_ACTION_HISTORY", "action_history")
    assert a4_history["status"] == "APPLIED_AND_MATCHED"
    assert a4_history["applied_action_count"] >= 2

    a5_qacc = row_for(rows, "A5_QACC_ABLATION", "qacc")
    assert a5_qacc["status"] == "APPLIED_AND_MATCHED"
    assert a5_qacc["actual_applied_actions"] == "mujoco.qacc"


def test_c2_application_auditor_cli_writes_csv(tmp_path, monkeypatch):
    c2_dir = make_c2_dir(tmp_path)
    output = tmp_path / "out.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--c2-dir", str(c2_dir), "--output-csv", str(output)],
    )

    main()

    rows = list(csv.DictReader(output.open("r", encoding="utf-8")))
    assert row_for(rows, "A3_GOAL_INTERPOLATOR_STATE", "interpolator")["status"] == "NO_APPLICABLE_INTERPOLATOR_STATE"
