import numpy as np, pytest
from gripper_attack.uncertainty import prefix_entropy, prefix_top2_margin

def test_entropy_shape():
    assert prefix_entropy(np.zeros((3, 4), dtype=np.float32)) > 0

def test_margin_top1_minus_top2():
    x = np.array([[1, 3, 2], [0, -1, 5]], dtype=np.float32)
    assert np.isclose(prefix_top2_margin(x), 3.0)

def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        prefix_entropy(np.zeros((4,)))
