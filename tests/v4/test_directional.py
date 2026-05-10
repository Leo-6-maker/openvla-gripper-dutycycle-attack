import numpy as np
from gripper_attack.directional import normalize_direction, compute_alignment

def test_normalize():
    g = normalize_direction(np.array([2,0,0,0,0,0,0], dtype=np.float32), [0,1,2,3,4,5])
    assert np.isclose(np.linalg.norm(g[[0,1,2,3,4,5]]), 1)

def test_alignment_sign():
    g = normalize_direction(np.array([1,0,0,0,0,0,0], dtype=np.float32), [0,1,2,3,4,5])
    assert compute_alignment(np.array([1,0,0,0,0,0,0], dtype=np.float32), g, [0,1,2,3,4,5])["alignment"] > 0
    assert compute_alignment(np.array([-1,0,0,0,0,0,0], dtype=np.float32), g, [0,1,2,3,4,5])["alignment"] < 0

def test_gripper_excluded_default_dims():
    g = normalize_direction(np.array([1,0,0,0,0,0,99], dtype=np.float32), [0,1,2,3,4,5])
    assert g[6] == 0
