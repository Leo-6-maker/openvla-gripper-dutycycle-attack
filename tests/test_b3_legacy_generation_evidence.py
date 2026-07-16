from detector.audit_b3_legacy_generation_evidence import classify_generation_state


def test_generation_field_categories_are_not_collapsed():
    assert classify_generation_state([1], 1) == "FIELD_PRESENT_VALUE_1"
    assert classify_generation_state([], 1) == "FIELD_MISSING"
    assert classify_generation_state([0], 1) == "FIELD_EXPLICIT_0"
    assert classify_generation_state([2], 1) == "FIELD_EXPLICIT_GT1"
    assert classify_generation_state([1, 0], 2) == "FIELD_EXPLICIT_0"
