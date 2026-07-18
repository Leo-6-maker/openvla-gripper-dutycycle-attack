from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.detector_v5.audit_v5_physics_task_decoder import build_object_slices, parse_bddl_objects


def test_bddl_object_order_and_fourteen_dimensional_slices():
    objects = parse_bddl_objects(
        """(:objects
        soup_1 - soup
        basket_1 basket_2 - basket
      )
    )"""
    )
    slices = build_object_slices(objects)
    assert [item["object_name"] for item in slices] == ["soup_1", "basket_1", "basket_2"]
    assert slices[0]["pos"] == [0, 3]
    assert slices[0]["quat"] == [3, 7]
    assert slices[0]["to_eef_pos"] == [7, 10]
    assert slices[0]["to_eef_quat"] == [10, 14]
    assert slices[1]["offset_start"] == 14
    assert slices[2]["offset_end_exclusive"] == 42
