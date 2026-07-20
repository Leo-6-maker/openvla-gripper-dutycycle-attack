#!/usr/bin/env python3
"""Static and injected-control-flow R10.4-R4A audit; never loads a model."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256, feature_order_sha256


def _source_checks(root: Path) -> dict[str, object]:
    runtime_path = root / "src/gripper_attack/r10_4_runtime.py"
    runtime_text = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(runtime_text)
    unconditional_exit = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "exit"
        for node in ast.walk(tree)
    )
    external_image_normalization = "/ 255" in runtime_text or " /255" in runtime_text
    signature = inspect.signature(OfficialOpenVLAActionAdapter.predict_action)
    signature_pass = list(signature.parameters)[:3] == ["self", "image_np", "task_label"] and "capture" in signature.parameters
    return {
        "unconditional_exit_before_common_loop": unconditional_exit,
        "external_image_normalization": external_image_normalization,
        "official_adapter_signature": str(signature),
        "official_adapter_signature_pass": signature_pass,
        "feature_order_sha256": feature_order_sha256(),
        "feature_order_sha256_pass": feature_order_sha256() == FEATURE_ORDER_SHA256,
        "openvla_model_loaded": False,
        "detector_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = _source_checks(args.repo_root.resolve())
    result["status"] = "PASS" if not result["unconditional_exit_before_common_loop"] and not result["external_image_normalization"] and result["official_adapter_signature_pass"] and result["feature_order_sha256_pass"] else "HOLD"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
