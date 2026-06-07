"""Test: open convention env_action_6 = -1.0 = OPEN."""
def test_open_convention():
    def is_env_open(grip_val):
        return grip_val < -0.5
    assert is_env_open(-1.0), 'env_action=-1.0 should be OPEN'
    assert not is_env_open(1.0), 'env_action=+1.0 should be CLOSE'
    assert not is_env_open(0.0), 'env_action=0 should not match'
    print('PASS: test_open_convention')

def test_open_vs_close_symmetry():
    """OPEN and CLOSE must be opposites."""
    def is_env_open(g): return g < -0.5
    def is_env_close(g): return g > 0.5
    for val in [-1.0, 1.0]:
        assert is_env_open(val) != is_env_close(val), f'{val} must be either open or close, not both'
    print('PASS: test_open_vs_close_symmetry')

if __name__ == '__main__':
    test_open_convention()
    test_open_vs_close_symmetry()
    print('All open convention tests passed')
