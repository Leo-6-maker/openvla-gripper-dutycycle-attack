from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/STAGE_X_X1R_T1D0R2_CLEAN_RUNTIME_AUTHORITY_V1.json"


def test_reviewed_ancestry_and_closed_authorization():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["reviewed_source"]["commit"] == "e9db0bfadf3822e8b3fc4771bbdd644e5c07b14e"
    assert config["reviewed_source"]["tree"] == "a2772e4b3a2e7dfd060a06bfe2462b7c728d1b14"
    assert all(value is False for key, value in config["authorization"].items() if key.endswith("_authorized"))
    assert config["authorization"]["next_gate"] == "CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED"


def test_d0r1_population_and_seed_ledger_are_frozen():
    rows = [json.loads(line) for line in (ROOT / "reports/STAGE_X_X1R_T1D0R1_PARENT_LEDGER_V1.json").read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 39
    assert all(row["clean_seed_namespace"] == "STAGE_X_X1R_T1D0R1_CLEAN_SEED_V1" for row in rows)
    assert "libero_goal/task_01" not in {row["canonical_parent_key"] for row in rows}
    assert sum(row["suite"] == "libero_10" for row in rows) == 10
    assert sum(row["suite"] == "libero_goal" for row in rows) == 9
    assert sum(row["suite"] == "libero_object" for row in rows) == 10
    assert sum(row["suite"] == "libero_spatial" for row in rows) == 10


def test_horizon_boundaries_and_pure_contract():
    module = ast.parse((ROOT / "src/gripper_attack/stage_x_x1r_d1_clean_runtime_contract.py").read_text(encoding="utf-8"))
    imports = [alias.name.split(".")[0] for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names]
    imports += [alias.name.split(".")[0] for node in ast.walk(module) if isinstance(node, ast.ImportFrom) for alias in node.names]
    assert not set(imports) & {"torch", "transformers", "gym", "mujoco", "robosuite"}
    namespace: dict[str, object] = {}
    exec(compile(module, "stage_x_x1r_d1_clean_runtime_contract.py", "exec"), namespace)
    assert namespace["legal_horizon"](0, 14) is False
    assert namespace["legal_horizon"](0, 15) is True
    assert namespace["legal_horizon"](5, 19) is False
    assert namespace["legal_horizon"](5, 20) is True
    assert namespace["HORIZONS"] == {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}


def test_required_runtime_forbidden_counters_are_split():
    text = (ROOT / "scripts/stage_x/audit_stage_x1r_t1d0r2_clean_runtime_authority.py").read_text(encoding="utf-8")
    for token in (
        "historical_replay_student_forward_calls",
        "openvla_weight_loads",
        "openvla_model_inference_calls",
        "prospective_parent_clean_rollouts",
        "env_step_calls",
        "pgd_calls",
        "protected_reads",
        "MISSING_SEALED_PER_STEP_REFERENCE",
    ):
        assert token in text
