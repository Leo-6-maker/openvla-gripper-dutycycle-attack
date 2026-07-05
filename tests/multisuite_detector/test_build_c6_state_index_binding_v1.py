from tools.multisuite_detector import build_c6_state_index_binding_v1 as m


def test_gate_name():
    assert m.GATE == "C6_1H_STATE_INDEX_BINDING_AUDIT_BUILD"
