import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_teacher_training_adapter import adapt_teacher_batch  # noqa: E402


def _row(*, unknown_t10=False):
    return {
        "grasp_support": True,
        "grasp_support_mask": True,
        "retention_active": False,
        "retention_active_mask": True,
        "retention_continuation_t10": None if unknown_t10 else True,
        "retention_unknown_mask": unknown_t10,
        "release_imminent": False,
        "release_imminent_mask": True,
    }


def test_adapter_inverts_t10_unknown_and_combines_padding_mask():
    records = [[_row(), _row(unknown_t10=True)], [_row(), _row()]]
    padding = torch.tensor([[True, True], [True, False]])
    targets, masks = adapt_teacher_batch(records, padding_mask=padding)

    assert targets["retention_continuation_t10"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert masks["retention_continuation_t10"].tolist() == [[True, False], [True, False]]
    assert masks["grasp_support"].tolist() == [[True, True], [True, False]]


def test_adapter_rejects_missing_head_or_mask():
    row = _row()
    row.pop("release_imminent_mask")
    with pytest.raises(ValueError, match="missing fields"):
        adapt_teacher_batch([[row]])
