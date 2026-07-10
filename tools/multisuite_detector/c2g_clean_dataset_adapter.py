"""Dataset adapter from clean Teacher-v2 rows to Detector-v2 model targets.

The adapter keeps teacher labels and privileged fields separate from student inputs,
provides the clean target-name mapping expected by the model, and derives explicit
fully-known-negative episode flags without converting unknown rows to negatives.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.gripper_attack.c2g_clean_window_schema import (
    assert_clean_student_feature_names,
    validate_clean_teacher_row,
)
from tools.multisuite_detector.c2g_dataset_scaffold import (
    assert_split_viability,
    split_label_coverage,
)


MODEL_TARGET_MAP = {
    "critical_window": "y_gripper_critical_window",
    "contact_grasp": "y_contact_or_grasp_stable",
    "close_intent": "y_clean_close_intent",
    "transport_constraint": "y_lift_transport_or_constraint",
    "release_safe": "y_release_safe",
    "grounding_confidence": "y_target_relevant",
    "window_start": "y_attack_start_b",
    "window_active": "y_gripper_critical_window",
}


def teacher_row_to_model_targets(row: Mapping[str, Any]) -> Dict[str, Dict[str, float | bool]]:
    """Convert a validated teacher row into clean model targets and masks."""

    validate_clean_teacher_row(row)
    known = bool(row["label_known_mask"])
    targets: Dict[str, float] = {}
    masks: Dict[str, bool] = {}
    for model_name, teacher_name in MODEL_TARGET_MAP.items():
        value = row[teacher_name]
        targets[model_name] = float(bool(value)) if known else 0.0
        masks[model_name] = known
    return {"targets": targets, "masks": masks}


def assert_student_feature_payload(
    feature_names: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Check feature names and ensure teacher fields are not copied into payloads."""

    assert_clean_student_feature_names(feature_names)
    expected = set(feature_names)
    for index, row in enumerate(rows):
        actual = set(row)
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        if unexpected or missing:
            raise ValueError(
                f"student feature row {index} schema mismatch "
                f"unexpected={unexpected} missing={missing}"
            )


def derive_episode_fully_known_negative(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    """Return true only when every row is known and none is critical."""

    state: dict[str, dict[str, bool]] = defaultdict(
        lambda: {"all_known": True, "positive": False}
    )
    for row in rows:
        episode = str(row["episode_key"])
        known = bool(row["label_known_mask"])
        positive = bool(row["y_gripper_critical_window"]) if known else False
        state[episode]["all_known"] = bool(state[episode]["all_known"] and known)
        state[episode]["positive"] = bool(state[episode]["positive"] or positive)
    return {
        episode: bool(values["all_known"] and not values["positive"])
        for episode, values in state.items()
    }


def clean_window_split_coverage(
    rows: Sequence[Dict[str, Any]],
    *,
    persistence_window: int = 3,
    persistence_required: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Use the mature split audit with the corrected clean-window label name."""

    return split_label_coverage(
        rows,
        label_key="y_gripper_critical_window",
        persistence_window=persistence_window,
        persistence_required=persistence_required,
    )


def assert_clean_window_split_viability(
    rows: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> None:
    coverage = clean_window_split_coverage(rows)
    assert_split_viability(coverage, **kwargs)
