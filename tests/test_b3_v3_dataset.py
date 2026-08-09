import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_formal import B3_HEADS  # noqa: E402
from gripper_attack.b3_v3_dataset import (  # noqa: E402
    B3Episode,
    B3EpisodeSampler,
    compute_fit_normalization,
    pad_episode_batch,
)


def _episode(index, *, suite="libero_object", task=0, state=0, with_9d=False):
    steps = 3 + index % 2
    x25 = torch.arange(steps * 25, dtype=torch.float32).reshape(steps, 25) + index
    targets = {head: torch.zeros(steps) for head in B3_HEADS}
    masks = {head: torch.ones(steps, dtype=torch.bool) for head in B3_HEADS}
    masks["retention_continuation_t10"][0] = False
    return B3Episode(
        canonical_parent_key=f"{suite}/task_{task:02d}/state_{state:02d}", suite=suite,
        task_idx=task, state_id=state, split="FIT_TRAIN", task_success=index % 2 == 0,
        features_25d=x25, targets=targets, known_masks=masks,
        valid_mask=torch.ones(steps, dtype=torch.bool),
        features_9d=torch.ones(steps, 9) if with_9d else None,
    )


def test_padding_preserves_episode_and_unknown_masks():
    first, second = _episode(0), _episode(1)
    batch = pad_episode_batch([first, second])
    assert batch.x25.shape == (2, 4, 25)
    assert batch.padding_mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert batch.known_masks["retention_continuation_t10"][0, 0].item() is False
    assert batch.episode_valid_mask[0, 3].item() is False


def test_25d9d_batch_cannot_mix_variants():
    with pytest.raises(ValueError, match="mix"):
        pad_episode_batch([_episode(0), _episode(1, with_9d=True)])


def test_sampler_interleaves_suite_task_groups_without_step_sampling():
    episodes = [
        _episode(0, suite="libero_object", task=0, state=0),
        _episode(1, suite="libero_object", task=0, state=1),
        _episode(2, suite="libero_spatial", task=1, state=0),
        _episode(3, suite="libero_spatial", task=1, state=1),
    ]
    sampler = B3EpisodeSampler(episodes, seed=3)
    order = sampler.ordered_indices(shuffle=False)
    assert sorted(order) == [0, 1, 2, 3]
    assert [(episodes[i].suite, episodes[i].task_idx) for i in order[:2]] == [
        ("libero_object", 0), ("libero_spatial", 1)
    ]


def test_normalization_is_fit_only_and_includes_failures():
    episodes = [_episode(0, state=0), _episode(1, state=1)]
    normalization = compute_fit_normalization(episodes)
    assert len(normalization.mean_25d) == 25
    assert all(value > 0 for value in normalization.std_25d)
    with pytest.raises(ValueError, match="FIT_TRAIN"):
        compute_fit_normalization([_episode(0, state=20)])
