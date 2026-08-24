"""CPU/mock/failure-injection qualification for the X1R2 executable contract."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/stage_x/run_stage_x1r_primary_matrix.py"
OUTPUT = ROOT / "reports/STAGE_X_X1R2_Q2_EXECUTABLE_CONTRACT_AUDIT_V1.json"
sys.path.insert(0, str(ROOT / "src"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT).strip()


def load_runner_helpers() -> dict[str, Any]:
    wanted = {
        "initial_exposure",
        "mark_model_inference_started",
        "mark_policy_action_materialized",
        "mark_env_step_started",
        "mark_env_step_completed",
        "mark_attack_invocation_started",
        "sync_attack_trace",
        "mark_adversarial_decode_started",
        "mark_adversarial_decode_completed",
    }
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace: dict[str, Any] = {"Any": Any, "Mapping": Mapping}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def assert_true(errors: list[str], condition: bool, name: str) -> None:
    if not condition:
        errors.append(name)


def run_failure_injection_cases(functions: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    def fresh() -> tuple[dict[str, Any], defaultdict[str, int]]:
        return functions["initial_exposure"](), defaultdict(int)

    exposure, counters = fresh()
    assert_true(errors, not exposure["model_inference_started"], "before_model_inference:model_start_false")
    assert_true(errors, not exposure["policy_action_materialized"], "before_model_inference:policy_false")
    assert_true(errors, not exposure["first_env_step_executed"], "before_model_inference:env_false")

    exposure, counters = fresh()
    functions["mark_model_inference_started"](exposure)
    functions["mark_policy_action_materialized"](exposure, counters)
    assert_true(errors, exposure["model_inference_started"], "after_clean_decode:model_start_true")
    assert_true(errors, exposure["policy_action_materialized"], "after_clean_decode:policy_true")
    assert_true(errors, not exposure["first_env_step_executed"], "after_clean_decode:env_false")
    assert_true(errors, exposure["rows_materialized"] == 0, "after_clean_decode:rows_zero")

    exposure, counters = fresh()
    functions["mark_model_inference_started"](exposure)
    functions["mark_policy_action_materialized"](exposure, counters)
    assert_true(errors, not exposure["env_step_started"], "before_env_step:env_start_false")
    assert_true(errors, not exposure["first_env_step_executed"], "before_env_step:first_env_false")

    exposure, counters = fresh()
    functions["mark_attack_invocation_started"](exposure, counters)
    assert_true(errors, exposure["attack_invocation_started"], "during_attack:invocation_true")
    assert_true(errors, not exposure["attack_result_returned"], "during_attack:returned_false")
    assert_true(errors, counters["pgd_calls"] == 1, "during_attack:pgd_call_recorded_at_start")

    exposure, counters = fresh()
    functions["mark_attack_invocation_started"](exposure, counters)
    functions["sync_attack_trace"](
        exposure,
        counters,
        {"attack_invocation_started": True, "attack_result_returned": True, "attack_result_accepted": False, "backward_invocation_count": 5, "loss_forward_count": 6},
    )
    assert_true(errors, exposure["attack_result_returned"], "after_attack_result:returned_true")
    assert_true(errors, not exposure["attack_result_accepted"], "after_attack_result:accepted_false")
    assert_true(errors, exposure["backward_invocation_count"] == 5, "after_attack_result:backward_count")
    assert_true(errors, counters["attack_backward_calls"] == 5, "after_attack_result:backward_counter")

    exposure, counters = fresh()
    functions["mark_adversarial_decode_started"](exposure)
    assert_true(errors, exposure["adversarial_decode_started"], "before_adv_decode:start_true")
    assert_true(errors, not exposure["attacked_action_materialized"], "before_adv_decode:action_false")
    functions["mark_adversarial_decode_completed"](exposure, counters)
    assert_true(errors, exposure["attacked_action_materialized"], "after_adv_decode:action_true")
    assert_true(errors, exposure["adversarial_decode_count"] == 1, "after_adv_decode:count_one")
    assert_true(errors, not exposure["first_env_step_executed"], "after_adv_decode:env_false")

    exposure, counters = fresh()
    functions["mark_env_step_started"](exposure, counters, attacked=True)
    assert_true(errors, exposure["env_step_started"], "attacked_env_step:start_true")
    assert_true(errors, exposure["attacked_env_step_started"], "attacked_env_step:attacked_start_true")
    assert_true(errors, not exposure["first_env_step_executed"], "attacked_env_step:first_env_false_before_return")
    functions["mark_env_step_completed"](exposure, counters, attacked=True)
    assert_true(errors, exposure["first_env_step_executed"], "attacked_env_step:first_env_true_after_return")
    assert_true(errors, exposure["attacked_env_step_completed"], "attacked_env_step:attacked_complete_true")
    assert_true(errors, counters["attacked_env_step_completed_count"] == 1, "attacked_env_step:completion_counter")

    return {"cases": 7, "status": "PASS" if not errors else "FAIL"}


def run_route_mock(errors: list[str]) -> dict[str, Any]:
    from gripper_attack.route_contract import (
        attach_route_debug,
        route_config_from_attack_config,
        validate_attack_request,
        validate_true_pgd_attack_result,
    )

    cfg = {
        "attack_optimizer": {
            "method": "token_prefix_pgd",
            "strict_route": True,
            "allow_fallback": False,
            "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            "target_token_id": 31745,
            "target_execution_class": "NATIVE_OPEN",
            "num_steps": 5,
            "epsilon": 0.03,
        }
    }
    route = route_config_from_attack_config(cfg)
    try:
        validate_attack_request(route, target_action_present=True)
        debug = attach_route_debug(
            {
                "adv_inputs": {"input_ids": object(), "pixel_values": object()},
                "num_backwards": 5,
                "num_loss_forwards": 6,
                "pixel_space": "processor_pixel_values",
                "pixel_budget_adv_inputs_linf": 0.02,
            },
            route,
            resolved_adapter_class="TokenPrefixPGDAttacker",
            fallback_used=False,
            target_action_present=True,
        )
        debug["x_adv_is_none"] = True
        debug["action_adv_is_none"] = True
        result = SimpleNamespace(
            debug=debug,
            attack_method="token_prefix_pgd_pixel_values_target_token_logratio_arm_v3",
            directional_loss_available=True,
            x_adv=None,
            action_adv=None,
            epsilon=0.03,
            observation_perturb_linf=0.02,
        )
        validate_true_pgd_attack_result(result, route)
    except Exception as exc:  # pragma: no cover - emitted in audit report
        errors.append(f"canonical_true_route:{type(exc).__name__}:{exc}")
    return {"strict_route": route.strict_route, "allow_fallback": route.allow_fallback, "target_token_id": route.target_token_id, "target_execution_class": route.target_execution_class}


def main() -> int:
    errors: list[str] = []
    runner_text = RUNNER.read_text(encoding="utf-8")
    functions = load_runner_helpers()
    exposure_cases = run_failure_injection_cases(functions, errors)
    route = run_route_mock(errors)
    static_checks = {
        "canonical_wrapper_only": "OpenVLAVisualAttacker(" in runner_text and "TokenPrefixPGDAttacker(" not in runner_text,
        "execution_trace_passed": "execution_trace=attack_trace" in runner_text,
        "full_episode_after_h_phys": "official_horizon_reached" in runner_text and "final_policy_steps_executed" in runner_text,
        "arm_isolation_literal": "ARM_TOKEN_ISOLATION_FAIL" in runner_text and "not arm_equal" in runner_text,
        "native_open_constants": "TARGET_TOKEN = 31745" in runner_text and 'TARGET_CLASS = "NATIVE_OPEN"' in runner_text,
        "rand_zero_optimizer_steps": '"optimizer_steps": 0' in runner_text and '"gradient_used": False' in runner_text,
        "durable_telemetry": "os.fsync(handle.fileno())" in runner_text and "persist_attack_tensor" in runner_text,
        "exposure_fields": all(name in runner_text for name in ("model_inference_started", "policy_action_materialized", "env_step_started", "env_step_completed", "attack_invocation_started", "attack_result_returned", "attack_result_accepted", "attacked_action_materialized", "attacked_env_step_started", "attacked_env_step_completed")),
    }
    for name, value in static_checks.items():
        assert_true(errors, value, f"static:{name}")
    report = {
        "schema": "STAGE_X_X1R2_Q2_EXECUTABLE_CONTRACT_AUDIT_V1",
        "status": "STAGE_X_X1R2_Q2_EXECUTABLE_CONTRACT_PASS" if not errors else "STAGE_X_X1R2_Q2_HOLD_EXECUTABLE_CONTRACT",
        "scope": "CPU/mock/failure-injection only; no checkpoint load, model inference, simulator reset/step, PGD outcome, V_phys, Eval160, or protected read",
        "source": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status_porcelain": git("status", "--porcelain")},
        "static_checks": static_checks,
        "failure_injection": exposure_cases,
        "canonical_route_mock": route,
        "errors": errors,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "model_inference_calls": 0, "env_reset_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "attack_outcome_reads": 0, "vphys_reads": 0, "physical_interventions": 0, "eval160_reads": 0, "protected_reads": 0},
        "next_gate": "STAGE_X_X1R2_Q3_ENGINEERING_ONLY_REAL_MODEL_QUALIFICATION" if not errors else "OWNER_REVIEW_X1R2_Q2_EXECUTABLE_CONTRACT_HOLD",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
