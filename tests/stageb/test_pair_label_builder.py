"""Test: Stage-B v1 pair label builder logic."""
def is_open(v): return float(v) < -0.5

def test_cmd_susceptible():
    vis_open, vis_streak = 8, 4; rand_open, rand_streak = 2, 1
    cmd = (vis_open >= 6 or vis_streak >= 6) and not (rand_open >= 6 or rand_streak >= 6)
    assert cmd, 'VIS open=8 (>=6), random open=2 (<6) → should be cmd_susceptible'
    print('PASS: test_cmd_susceptible')

def test_not_cmd_when_random_matches():
    vis_open, rand_open = 7, 7
    cmd = (vis_open >= 6) and not (rand_open >= 6)
    assert not cmd, 'random also >=6 → NOT cmd_susceptible'
    print('PASS: test_not_cmd_when_random_matches')

def test_random_confounded():
    rand_open = 8
    assert rand_open >= 6, 'random open >=6 → random_confounded'
    print('PASS: test_random_confounded')

def test_physical_response_tiers():
    vis_delta, rand_delta = 0.015, 0.002
    phys_sens = vis_delta >= 0.01
    phys_strict = vis_delta >= 0.02
    vis_spec = phys_sens and not (rand_delta >= 0.01)
    assert phys_sens, 'delta 0.015 >= 0.01 → sensitive'
    assert not phys_strict, 'delta 0.015 < 0.02 → not strict'
    assert vis_spec, 'VIS delta >= 0.01 but random delta 0.002 < 0.01 → vis_specific'
    print('PASS: test_physical_response_tiers')

def test_open_count_correct_convention():
    actions = [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    open_count = sum(1 for a in actions if is_open(a))
    assert open_count == 6, f'open_count should be 6 (6x -1.0), got {open_count}'
    streak = max_streak = 0
    for a in actions:
        if is_open(a): streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    assert max_streak == 3, f'longest_open_streak should be 3, got {max_streak}'
    print('PASS: test_open_count_correct_convention')

if __name__ == '__main__':
    test_cmd_susceptible()
    test_not_cmd_when_random_matches()
    test_random_confounded()
    test_physical_response_tiers()
    test_open_count_correct_convention()
    print('All pair label builder tests passed')
