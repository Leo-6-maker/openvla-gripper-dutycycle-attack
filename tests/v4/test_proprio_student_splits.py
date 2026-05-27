from src.utils.proprio_causal_student import assign_splits, split_summary


def _rows():
    rows = []
    for task in range(10):
        for ep in range(3):
            rows.append(
                {
                    "suite": "suite",
                    "task_id": f"task_{task}",
                    "episode_key": f"task_{task}::ep_{ep}",
                    "teacher_hazard": "true" if ep == 0 else "false",
                    "teacher_phase": "carry",
                }
            )
    return rows


def test_task_id_split_has_no_task_overlap():
    rows = _rows()
    split_by_ep = assign_splits(rows, "task_id", seed=1)
    task_by_split = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        task_by_split[split_by_ep[r["episode_key"]]].add(r["task_id"])
    assert task_by_split["train"].isdisjoint(task_by_split["val"])
    assert task_by_split["train"].isdisjoint(task_by_split["test"])
    assert task_by_split["val"].isdisjoint(task_by_split["test"])


def test_episode_key_split_has_no_episode_overlap_and_summary_rows():
    rows = _rows()
    split_by_ep = assign_splits(rows, "episode_key", seed=2)
    eps_by_split = {"train": set(), "val": set(), "test": set()}
    for ep, split in split_by_ep.items():
        eps_by_split[split].add(ep)
    assert eps_by_split["train"].isdisjoint(eps_by_split["val"])
    assert eps_by_split["train"].isdisjoint(eps_by_split["test"])
    summary = split_summary(rows, "episode_key", seed=2)
    assert summary
    assert {r["split_mode"] for r in summary} == {"episode_key"}

