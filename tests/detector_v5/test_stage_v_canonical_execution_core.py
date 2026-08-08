from __future__ import annotations

import json
from pathlib import Path

from gripper_attack.stage_v_canonical_execution_core import (
    CANONICAL_INIT_STATE_HASH_ALGORITHM,
    CANONICAL_INIT_STATE_SCHEMA,
    CONTRACT_FIELDS,
    CanonicalExecutionCore,
    PolicyStep,
    sha256_file,
    write_diagnostic_trace_artifacts,
    write_trace_artifacts,
)
from scripts.detector_v5.stage_v_rb1_runtime_equivalence import CONTRACT_FIELDS as VALIDATOR_CONTRACT_FIELDS, validate_receipt, verify_artifact_files
from scripts.detector_v5.audit_stage_v_rb1_receipt import audit


PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json").read_text())


class FakeEnv:
    def __init__(self) -> None:
        self.state = 0.0
        self.steps = 0

    def seed(self, seed: int) -> None:
        assert seed == 7

    def reset(self) -> dict[str, float]:
        self.state = -1.0
        self.steps = 0
        return self.observe()

    def set_init_state(self, value: list[float]) -> dict[str, float]:
        self.state = float(value[0])
        self.steps = 0
        return self.observe()

    def step(self, action: list[float]):
        self.state += float(action[0])
        self.steps += 1
        return self.observe(), float(self.state), self.steps >= 3, {"step": self.steps}

    def observe(self) -> dict[str, float]:
        return {"state": self.state}

    def get_state(self) -> float:
        return self.state

    def set_state(self, value: float) -> None:
        self.state = float(value)

    def close(self) -> None:
        pass


def _core() -> CanonicalExecutionCore:
    def policy(step: int, _obs: object, _label: str) -> PolicyStep:
        return PolicyStep(raw_action=[1.0, 0, 0, 0, 0, 0, 0], token_ids=(step, 101, 102, 103, 104, 105, 106), metadata={})

    identity = {
        "canonical_parent_key": "libero_10/task_00/state_00",
        "suite": "libero_10",
        "task_index": 0,
        "state_index": 0,
    }
    return CanonicalExecutionCore(
        env_factory=FakeEnv,
        policy=policy,
        action_postprocess=lambda action: list(action),
        initial_state=[0.0],
        identity=identity,
        task_label="open the box",
        seed=7,
        num_steps_wait=0,
        suite_horizon=3,
        observation_getter=lambda _env, obs, _step: obs,
        physical_state_getter=lambda env, _obs, _step: {"state": env.state},
    )


def test_same_core_clean_traces_are_mode_independent() -> None:
    assert CONTRACT_FIELDS == VALIDATOR_CONTRACT_FIELDS
    left = _core().run_clean_episode(mode="CLEAN_QUALIFICATION")
    right = _core().run_clean_episode(mode="COUNTERFACTUAL_CLEAN_PREFIX")
    assert left.initial_state_sha256 == right.initial_state_sha256
    assert left.actions == right.actions
    assert left.steps == right.steps


def test_snapshot_restore_replays_exact_actions() -> None:
    core = _core()
    env = FakeEnv()
    env.reset()
    snapshot = env.get_state()
    result = core.run_noop_continuation(env=env, snapshot=snapshot, actions=[[1.0, 0, 0, 0, 0, 0, 0]] * 2)
    assert len(result["snapshot_restore"]) == 1
    assert [row["step"] for row in result["noop_actions"]] == [0, 1]


def test_shared_core_exposes_clean_prefix_snapshot() -> None:
    result = _core().run_clean_prefix_and_snapshot(probe_step=1)
    assert result["probe_step"] == 1
    assert len(result["steps"]) == 2
    assert len(result["actions"]) == 2
    assert len(result["probe_state_sha256"]) == 64


def test_trace_artifacts_are_deterministic(tmp_path: Path) -> None:
    trace = _core().run_clean_episode(mode="CLEAN_QUALIFICATION")
    artifacts, hashes = write_trace_artifacts(tmp_path / "trace", trace)
    assert set(artifacts) == {"initial_state", "policy_token_trace", "postprocessed_action_trace", "observation_trace", "physical_state_trace"}
    assert hashes["postprocessed_action_trace_sha256"] == sha256_file(tmp_path / "trace" / "postprocessed_action_trace.jsonl")
    assert len(CONTRACT_FIELDS) == 20


def test_diagnostic_trace_artifacts_are_separate_from_v1_traces(tmp_path: Path) -> None:
    core = _core()
    core.diagnostic_getter = lambda env, _obs, step, _policy: {
        "full_sim_state": {"state": env.state},
        "policy_rgb_224": [[step]],
        "model_inputs": {"input_ids": [step], "attention_mask": [1]},
    }
    trace = core.run_clean_episode(mode="CLEAN_QUALIFICATION")
    common_artifacts, common_hashes = write_trace_artifacts(tmp_path / "trace", trace)
    diagnostic_artifacts, diagnostic_hashes = write_diagnostic_trace_artifacts(tmp_path / "trace", trace)
    assert set(common_artifacts) == {"initial_state", "policy_token_trace", "postprocessed_action_trace", "observation_trace", "physical_state_trace"}
    assert set(diagnostic_artifacts) == {"full_sim_state_trace", "policy_rgb_224_trace", "model_input_trace"}
    assert set(diagnostic_hashes) == {"full_sim_state_trace_sha256", "policy_rgb_224_trace_sha256", "model_input_trace_sha256"}


def test_producer_receipt_requires_independent_audit(tmp_path: Path) -> None:
    core = _core()
    trace = core.run_clean_episode(mode="CLEAN_QUALIFICATION")
    artifacts, hashes = write_trace_artifacts(tmp_path / "trace", trace)
    contract = {field: "value" for field in CONTRACT_FIELDS}
    contract.update({
        "clean_core_sha256": sha256_file(Path(__file__).resolve().parents[2] / "src/gripper_attack/stage_v_canonical_execution_core.py"),
        "seed": 7,
        "num_steps_wait": 0,
        "suite_horizon": 3,
        "initial_state_hash_algorithm": CANONICAL_INIT_STATE_HASH_ALGORITHM,
        "initial_state_identity_schema": CANONICAL_INIT_STATE_SCHEMA,
    })
    receipt = core.build_receipt(
        trace=trace,
        mode="CLEAN_QUALIFICATION",
        comparison_scope="CLEAN_PATH",
        contract=contract,
        trace_artifacts=artifacts,
        trace_hashes=hashes,
    )
    validate_receipt(receipt, PROTOCOL, require_independent_recompute=False)
    verify_artifact_files(receipt, tmp_path / "trace", PROTOCOL, require_independent_recompute=False)


def test_independent_auditor_recomputes_and_promotes_receipt(tmp_path: Path) -> None:
    core = _core()
    trace = core.run_clean_episode(mode="CLEAN_QUALIFICATION")
    artifacts, hashes = write_trace_artifacts(tmp_path / "trace", trace)
    contract = {field: "value" for field in CONTRACT_FIELDS}
    contract.update({
        "clean_core_sha256": sha256_file(Path(__file__).resolve().parents[2] / "src/gripper_attack/stage_v_canonical_execution_core.py"),
        "seed": 7,
        "num_steps_wait": 0,
        "suite_horizon": 3,
        "initial_state_hash_algorithm": CANONICAL_INIT_STATE_HASH_ALGORITHM,
        "initial_state_identity_schema": CANONICAL_INIT_STATE_SCHEMA,
    })
    receipt = core.build_receipt(
        trace=trace, mode="CLEAN_QUALIFICATION", comparison_scope="CLEAN_PATH",
        contract=contract, trace_artifacts=artifacts, trace_hashes=hashes,
    )
    producer = tmp_path / "RB1_PRODUCER_RECEIPT.json"
    audited = tmp_path / "RB1_INDEPENDENT_RECEIPT.json"
    producer.write_text(json.dumps(receipt), encoding="utf-8")
    result = audit(
        protocol_path=Path(__file__).resolve().parents[2] / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json",
        receipt_path=producer, artifact_root=tmp_path / "trace",
        core_path=Path(__file__).resolve().parents[2] / "src/gripper_attack/stage_v_canonical_execution_core.py",
        output_path=audited, repo=Path(__file__).resolve().parents[2],
    )
    assert result["verdict"] == "PASS"
    validate_receipt(json.loads(audited.read_text()), PROTOCOL)
