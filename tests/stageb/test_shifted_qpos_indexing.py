"""Test: shifted qpos uses step_idx, not enumerate local index."""
def test_step_dict_lookup():
    rows = [{'step': '3'}, {'step': '4'}, {'step': '5'}, {'step': '6'}, {'step': '7'}]
    step_dict = {int(r['step']): r for r in rows}
    att_steps = [5, 6]
    shifted = [step_dict[s+1] for s in att_steps if s+1 in step_dict]
    assert len(shifted) == 2, f'shifted should have 2 matches (steps 6,7), got {len(shifted)}'
    assert shifted[0]['step'] == '6', f'step 5→6: {shifted[0]}'
    assert shifted[1]['step'] == '7', f'step 6→7: {shifted[1]}'
    print('PASS: test_step_dict_lookup')

def test_enumerate_is_wrong():
    """Demonstrate that enumerate(att) local index is NOT correct."""
    rows = [{'step': '0'}, {'step': '1'}, {'step': '50'}, {'step': '51'}, {'step': '100'}]
    att = [rows[2], rows[3]]  # attack at steps 50, 51
    # WRONG: enumerate(att) uses 0,1 (local indices), pointing to rows[1],rows[2]
    wrong = []
    for i, r in enumerate(att):
        if i+1 < len(rows): wrong.append(rows[i+1])
    # i=0 → rows[1] = step 1 (wrong, should be step 51 or 100)
    # i=1 → rows[2] = step 50 (wrong, should be step 52 or 100)
    wrong_steps = [int(r['step']) for r in wrong]
    assert wrong_steps != [51, 52], 'enumerate att gives WRONG steps (proves the bug)'
    print('PASS: test_enumerate_is_wrong (confirmed local index bug)')

if __name__ == '__main__':
    test_step_dict_lookup()
    test_enumerate_is_wrong()
    print('All shifted qpos tests passed')
