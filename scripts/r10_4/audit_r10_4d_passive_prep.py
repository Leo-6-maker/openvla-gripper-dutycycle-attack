#!/usr/bin/env python3
"""Independent CPU/static audit for the R10.4D passive-smoke preparation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256
from gripper_attack.r10_4d_passive import FROZEN, RoutedGraspDetector, SUPPORTED_PARENT, parse_route


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/gripper_attack/r10_4d_passive.py"
RUNNER = ROOT / "scripts/r10_4/run_r10_4d_passive_smoke.py"
RECEIPT_BUILDER = ROOT / "scripts/r10_4/build_r10_4d_authorization_receipt.py"
PROTOCOL = ROOT / "configs/R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_V1.json"
TESTS = ROOT / "tests/test_r10_4d_passive.py"


def result(name: str, passed: bool, detail: str) -> tuple[str, bool, str]:
    return name, bool(passed), detail


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    for path in (RUNTIME, RUNNER, RECEIPT_BUILDER, PROTOCOL, TESTS):
        checks.append(result(f"FILE_{path.name}", path.is_file(), str(path)))

    runtime_source = RUNTIME.read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")
    builder_source = RECEIPT_BUILDER.read_text(encoding="utf-8")
    ast.parse(runtime_source)
    ast.parse(runner_source)
    ast.parse(builder_source)
    checks.append(result("PYTHON_AST_PARSE", True, "runtime/runner/receipt builder"))

    model = RoutedGraspDetector()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checks.append(result("MODEL_DUAL_HEAD", hasattr(model, "head_multi") and hasattr(model, "head_single"), "dual head"))
    checks.append(result("MODEL_PARAM_COUNT", parameter_count == 46658, str(parameter_count)))
    checks.append(result("ROUTE_PARENT", parse_route(SUPPORTED_PARENT) == "multi_object_transfer", SUPPORTED_PARENT))
    checks.append(result("ROUTE_FAIL_CLOSED", parse_route("libero_object/task_00/state_20") == "unsupported_abstain", "object abstains"))

    expected_frozen = {
        "input_dim": 25,
        "hidden_dim": 64,
        "num_layers": 2,
        "grasp_threshold": 0.5,
        "grasp_persistence": 3,
        "guard_param": 0.02,
        "max_episode_emits": 1,
    }
    for key, expected in expected_frozen.items():
        checks.append(result(f"FROZEN_{key}", FROZEN.get(key) == expected, repr(FROZEN.get(key))))

    checks.append(result(
        "GENERATION_NO_DEFAULT",
        'capture.get("generation_passes_per_step")' in runtime_source
        and 'capture.get("generation_passes_per_step",' not in runtime_source,
        "missing metadata cannot default to one",
    ))
    checks.append(result(
        "REAL_LOOP_HAS_MAX_STEPS",
        "for step in range(max_steps):" in runtime_source,
        "real path does not depend on recorded actions",
    ))
    checks.append(result(
        "ACTION_COPY_ONLY",
        "executed_action = clean_env_action.copy()" in runtime_source,
        "exact passive copy",
    ))
    checks.append(result(
        "ENV_STEP_EXECUTED_ACTION",
        "env.step(executed_action.tolist())" in runtime_source,
        "single clean action reaches env",
    ))
    detector_position = runtime_source.find("grasp_logit, grasp_probability = detector.step")
    sidecar_position = runtime_source.find("sidecar = dict(privileged_observer")
    checks.append(result(
        "PRIVILEGED_AFTER_DETECTOR",
        detector_position >= 0 and sidecar_position > detector_position,
        f"detector={detector_position} sidecar={sidecar_position}",
    ))
    checks.append(result(
        "PRIVILEGED_NOT_INPUT",
        'sidecar["detector_input"] = False' in runtime_source
        and '"privileged_runtime_input": False' in runtime_source,
        "sidecar isolated",
    ))

    required_flags = (
        "--model-path",
        "--detector-bundle",
        "--parent-manifest",
        "--authorization-receipt",
        "--output-root",
        "--upstream-root",
        "--gpu",
    )
    for flag in required_flags:
        checks.append(result(f"CLI_{flag[2:].replace('-', '_')}", flag in runner_source, flag))
    checks.append(result(
        "RECEIPT_BEFORE_MODEL_LOAD",
        runner_source.find("validate_authorization_receipt(") < runner_source.find("load_openvla(args.model_path"),
        "receipt validation precedes 7B load",
    ))
    checks.append(result(
        "OFFICIAL_OPENVLA_ADAPTER",
        "OfficialOpenVLAActionAdapter(" in runner_source,
        "official adapter",
    ))
    checks.append(result(
        "OFFICIAL_IMAGE_PATH",
        "get_libero_image(observation, 224)" in runner_source,
        "official image helper",
    ))
    checks.append(result(
        "OUTPUT_NON_OVERWRITE",
        "if args.output_root.exists():" in runner_source,
        "existing root rejected",
    ))
    checks.append(result(
        "ONE_PARENT_CONSTANT",
        SUPPORTED_PARENT in runtime_source and "episodes_authorized\": 1" in PROTOCOL.read_text(encoding="utf-8"),
        SUPPORTED_PARENT,
    ))

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    checks.append(result("PROTOCOL_SCHEMA", protocol.get("schema") == "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_PROTOCOL_V1", str(protocol.get("schema"))))
    checks.append(result("PROTOCOL_FEATURE_SHA", protocol.get("feature_order_sha256") == FEATURE_ORDER_SHA256, str(protocol.get("feature_order_sha256"))))
    checks.append(result("PROTOCOL_PASSIVE_ONLY", protocol.get("passive_only") is True, str(protocol.get("passive_only"))))
    for key in (
        "formal_training_authorized",
        "formal_attack_authorized",
        "command_open_authorized",
        "visual_attack_authorized",
        "random_attack_authorized",
        "second_episode_authorized",
        "parent_substitution_authorized",
        "threshold_or_fsm_change_authorized",
        "output_overwrite_authorized",
    ):
        checks.append(result(f"PROTOCOL_{key}", protocol.get(key) is False, repr(protocol.get(key))))

    failed = [check for check in checks if not check[1]]
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"\nR10.4D PREP AUDIT: {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        print("Failed checks:")
        for name, _passed, detail in failed:
            print(f"- {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
