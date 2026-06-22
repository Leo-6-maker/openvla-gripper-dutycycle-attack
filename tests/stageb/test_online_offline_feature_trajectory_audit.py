import csv

from gripper_attack.sc5_detector_runtime import SC5_FEATURES
from scripts.stageb.audit_online_offline_feature_trajectory import audit


RAW_FIELDS = [
    "gripper_qpos",
    "gripper_opening_proxy",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
]


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(*, drift=False):
    offline = []
    online = []
    for step in range(3):
        base = {"episode_key": "libero_goal|4|1|0|CLEAN", "step": step}
        for name in RAW_FIELDS:
            base[name] = "0.1"
        for name in SC5_FEATURES:
            base[name] = "0.2"
        off = dict(base)
        on = {"step": step, "detector_state_after": "IDLE", "detector_emitted_after": "False"}
        for name in RAW_FIELDS:
            on[name] = "0.1"
        for name in SC5_FEATURES:
            on["f_" + name] = "0.2"
        if drift and step == 1:
            on["gripper_qpos"] = "0.3"
            on["f_gripper_qpos"] = "0.3"
        offline.append(off)
        online.append(on)
    return offline, online


def test_feature_trajectory_audit_classifies_match_and_drift(tmp_path):
    dataset = tmp_path / "dataset.csv"
    online = tmp_path / "online.csv"
    off, on = _rows(drift=False)
    _write_csv(dataset, off)
    _write_csv(online, on)

    class Args:
        pass

    args = Args()
    args.dataset = str(dataset)
    args.online_telemetry = str(online)
    args.episode_key = "libero_goal|4|1|0|CLEAN"
    args.output_dir = str(tmp_path / "match")
    args.tolerance = 1e-6
    assert audit(args)["classification"] == "FEATURE_TRAJECTORY_MATCH"

    off, on = _rows(drift=True)
    _write_csv(dataset, off)
    _write_csv(online, on)
    args.output_dir = str(tmp_path / "drift")
    summary = audit(args)
    assert summary["classification"] == "FEATURE_TRAJECTORY_DRIFT"
    assert summary["field_summary"]["gripper_qpos"]["count_gt_tolerance"] == 1
