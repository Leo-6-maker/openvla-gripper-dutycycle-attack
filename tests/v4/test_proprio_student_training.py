import torch

from src.utils.proprio_causal_student import ProprioCausalMLP, compute_loss, encode_dataset, indices_for_split, make_loader


def _synthetic_rows(n=12):
    rows = []
    for i in range(n):
        rows.append(
            {
                "run_id": f"r{i//2}",
                "suite": "libero_goal",
                "task_id": f"task_{i//3}",
                "task_name": "put bowl on plate",
                "state_id": str(i),
                "seed": "0",
                "episode_key": f"e{i}",
                "step_idx": "0",
                "mechanism_type": "pick_place_transfer" if i % 2 == 0 else "articulated_object",
                "parse_confidence": "high",
                "gripper_command": str(float(i % 2)),
                "gripper_qpos": "0.01",
                "gripper_width": "0.02",
                "eef_x": "",
                "eef_y": "",
                "eef_z": str(i / 10),
                "eef_vx": "",
                "eef_vy": "",
                "eef_vz": "0.01",
                "action_dx": "0",
                "action_dy": "0",
                "action_dz": "0",
                "action_gripper": str(1 if i % 2 else -1),
                "recent_close_streak": str(i % 3),
                "recent_open_streak": str((i + 1) % 2),
                "recent_gripper_flip_count": "0",
                "normalized_step": str(i / n),
                "teacher_phase": "carry" if i % 2 else "other",
                "teacher_hazard": "true" if i % 2 else "false",
                "teacher_release_safe": "false",
                "teacher_confidence": "high",
                "image_path": "",
                "visual_feature_path": "",
            }
        )
    return rows


def test_tiny_training_smoke_saves_checkpoint(tmp_path):
    rows = _synthetic_rows()
    data = encode_dataset(rows, "episode_key", seed=0)
    idx = indices_for_split(data, "train")
    loader = make_loader(data, idx, batch_size=4, shuffle=True)
    model = ProprioCausalMLP(input_dim=data.x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xb, phase, hazard, release, conf = next(iter(loader))
    loss = compute_loss(model(xb), phase, hazard, release, conf)
    loss.backward()
    opt.step()
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    assert ckpt.exists()


def test_training_has_no_visual_dependency():
    rows = _synthetic_rows()
    assert all(not r["image_path"] and not r["visual_feature_path"] for r in rows)
    data = encode_dataset(rows, "episode_key", seed=1)
    assert data.x.shape[0] == len(rows)

