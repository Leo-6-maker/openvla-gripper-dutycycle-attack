#!/usr/bin/env python3
"""Strict entry point for event-aware clean C2g collection.

The underlying release collector reuses the mature SC5 streaming adapter, preserves
canonical 25D ordering, and tracks the current structured goal target instead of a
single episode-level ``primary_target`` shortcut.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

from gripper_attack.c2f_siglip_detector_runtime import CANONICAL_25D_FEATURES
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2


def canonicalize_stream_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("streaming feature adapter must return a mapping")
    features = result.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("streaming feature result lacks a features mapping")
    missing = [name for name in CANONICAL_25D_FEATURES if name not in features]
    unexpected = sorted(set(features) - set(CANONICAL_25D_FEATURES))
    if missing or unexpected:
        raise ValueError(
            f"25D feature schema mismatch missing={missing} unexpected={unexpected}"
        )
    ordered = OrderedDict(
        (name, float(features[name]))
        for name in CANONICAL_25D_FEATURES
    )
    output = dict(result)
    output["features"] = ordered
    output["feature_names"] = list(CANONICAL_25D_FEATURES)
    return output


def install_canonical_order_patch() -> None:
    original = SC5StreamingFeatureAdapterV2.update
    if getattr(original, "_c2g_canonical_order_patch", False):
        return

    def patched(self: SC5StreamingFeatureAdapterV2, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return canonicalize_stream_result(original(self, *args, **kwargs))

    patched._c2g_canonical_order_patch = True  # type: ignore[attr-defined]
    SC5StreamingFeatureAdapterV2.update = patched  # type: ignore[assignment]


def main() -> int:
    install_canonical_order_patch()
    from scripts.stageb.collect_c2g_clean_window_rollouts_event_v2 import main as collector_main

    return int(collector_main())


if __name__ == "__main__":
    raise SystemExit(main())
