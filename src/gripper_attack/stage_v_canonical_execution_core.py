"""One clean execution path for Stage V qualification and science modes.

The module is deliberately dependency-injected: the server runner supplies the
official environment, processor/adapter and physical-state getter, while both
qualification and counterfactual clean-prefix modes call this same class.
No intervention is implemented here; the only alternate operation is replaying
the captured clean action sequence after snapshot/restore.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CANONICAL_INIT_STATE_SCHEMA = "STAGE_V_CANONICAL_INIT_STATE_V1"
CANONICAL_INIT_STATE_HASH_ALGORITHM = "sha256(canonical_json(STAGE_V_CANONICAL_INIT_STATE_V1))"
CANONICAL_TRACE_SCHEMA = "STAGE_V_CANONICAL_TRACE_V1"
DUMMY_OPEN_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
CONTRACT_FIELDS = (
    "clean_core_sha256", "source_commit", "source_tree", "runner_sha256",
    "model_tree_sha256", "processor_sha256", "tokenizer_sha256",
    "prompt_template", "unnorm_key", "seed", "num_steps_wait",
    "suite_horizon", "termination_predicate", "success_predicate",
    "reset_restore_contract", "action_decode_contract",
    "action_postprocess_contract", "gripper_semantics",
    "initial_state_hash_algorithm", "initial_state_identity_schema",
)


class CanonicalExecutionError(RuntimeError):
    """Raised when the shared clean path cannot produce exact evidence."""


DIAGNOSTIC_TRACE_FIELDS = (
    "full_sim_state_trace_sha256",
    "policy_rgb_224_trace_sha256",
    "model_input_trace_sha256",
)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_descriptor(value: Any) -> dict[str, Any] | None:
    """Return a lossless byte digest descriptor for numpy/torch-like values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
        dtype = str(getattr(value, "dtype", ""))
        shape = [int(item) for item in getattr(value, "shape", ())]
        try:
            raw = value.numpy().tobytes()
        except Exception:
            try:
                import torch
                raw = value.view(torch.uint8).numpy().tobytes()
            except Exception as exc:  # pragma: no cover - runtime-only device edge
                raise CanonicalExecutionError("ARRAY_BYTES_UNAVAILABLE") from exc
        return {"kind": "array", "dtype": dtype, "shape": shape, "raw_sha256": hashlib.sha256(raw).hexdigest()}
    if hasattr(value, "tobytes") and hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            raw = value.tobytes(order="C")
        except TypeError:
            raw = value.tobytes()
        return {
            "kind": "array",
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
    return None


def canonical_value(value: Any) -> Any:
    """Convert runtime values to finite, deterministic evidence values."""
    descriptor = _array_descriptor(value)
    if descriptor is not None:
        return descriptor
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalExecutionError("NONFINITE_TRACE_VALUE")
        return value
    if isinstance(value, Mapping):
        return {str(key): canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, bytes):
        return {"kind": "bytes", "length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if hasattr(value, "item"):
        return canonical_value(value.item())
    raise CanonicalExecutionError(f"UNSUPPORTED_TRACE_VALUE:{type(value).__name__}")


def canonical_initial_state_sha256(state: Any, identity: Mapping[str, Any]) -> str:
    """Hash identity plus exact state descriptors, never pickle or tolerances."""
    payload = {
        "schema": CANONICAL_INIT_STATE_SCHEMA,
        "identity": {str(key): canonical_value(value) for key, value in sorted(identity.items())},
        "state": canonical_value(state),
    }
    return canonical_sha256(payload)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except ImportError:  # pragma: no cover - numpy is present in the server env
        pass
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - CPU unit tests may omit torch
        pass


@dataclass(frozen=True)
class PolicyStep:
    raw_action: Any
    token_ids: tuple[int, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.token_ids) != 7:
            raise CanonicalExecutionError("POLICY_ACTION_TOKEN_DIMENSION_INVALID")
        try:
            if len(self.raw_action) != 7:
                raise CanonicalExecutionError("POLICY_ACTION_DIMENSION_INVALID")
        except TypeError as exc:
            raise CanonicalExecutionError("POLICY_ACTION_NOT_SEQUENCE") from exc

    @classmethod
    def from_value(cls, value: Any) -> "PolicyStep":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise CanonicalExecutionError("POLICY_RESULT_MUST_BE_POLICY_STEP_OR_MAPPING")
        raw = value.get("raw_action", value.get("action"))
        tokens = value.get("token_ids", value.get("action_token_ids"))
        if raw is None or tokens is None:
            raise CanonicalExecutionError("POLICY_RESULT_MISSING_ACTION_OR_TOKENS")
        token_ids = tuple(int(item) for item in tokens)
        return cls(raw_action=raw, token_ids=token_ids, metadata=dict(value.get("metadata", {})))


@dataclass
class EpisodeTrace:
    identity: Mapping[str, Any]
    initial_state_sha256: str
    steps: list[dict[str, Any]]
    actions: list[Any]
    terminal_outcome: str
    termination_step: int
    termination_reason: str


def _step_env(env: Any, action: Any) -> tuple[Any, Any, bool, Mapping[str, Any]]:
    result = env.step(action)
    if not isinstance(result, (tuple, list)) or len(result) not in (4, 5):
        raise CanonicalExecutionError("ENV_STEP_RESULT_INVALID")
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), info if isinstance(info, Mapping) else {}
    obs, reward, terminated, truncated, info = result
    return obs, reward, bool(terminated or truncated), info if isinstance(info, Mapping) else {}


class CanonicalExecutionCore:
    """Shared clean execution and exact trace writer.

    ``policy`` is called once per policy step and must return ``PolicyStep``.
    ``observation_getter`` and ``physical_state_getter`` are explicit so a
    future runner cannot silently substitute a different state representation.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[], Any],
        policy: Callable[[int, Any, str], PolicyStep | Mapping[str, Any]],
        action_postprocess: Callable[[Any], Any],
        initial_state: Any,
        identity: Mapping[str, Any],
        task_label: str,
        seed: int,
        num_steps_wait: int,
        suite_horizon: int,
        observation_getter: Callable[[Any, Any, int], Any] | None = None,
        physical_state_getter: Callable[[Any, Any, int], Any] | None = None,
        diagnostic_getter: Callable[[Any, Any, int, PolicyStep], Any] | None = None,
        raw_capture_getter: Callable[[Any, Any, int, PolicyStep], Any] | None = None,
        raw_capture_steps: Sequence[int] | None = None,
        success_predicate: Callable[[Any, Any, Mapping[str, Any], bool, int], bool] | None = None,
        termination_predicate: Callable[[Any, Any, Mapping[str, Any], bool, int], bool] | None = None,
        restore_initial_state: Callable[[Any, Any], Any] | None = None,
        snapshot_getter: Callable[[Any, Any], Any] | None = None,
        snapshot_restorer: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.action_postprocess = action_postprocess
        self.initial_state = copy.deepcopy(initial_state)
        self.identity = dict(identity)
        self.task_label = str(task_label)
        self.seed = int(seed)
        self.num_steps_wait = int(num_steps_wait)
        self.suite_horizon = int(suite_horizon)
        self.observation_getter = observation_getter or (lambda _env, obs, _step: obs)
        self.physical_state_getter = physical_state_getter
        self.diagnostic_getter = diagnostic_getter
        self.raw_capture_getter = raw_capture_getter
        self.raw_capture_steps = frozenset(int(step) for step in (raw_capture_steps or ()))
        self.success_predicate = success_predicate or (lambda _env, _obs, _info, done, _step: bool(done))
        self.termination_predicate = termination_predicate or (lambda _env, _obs, _info, done, _step: bool(done))
        self.restore_initial_state = restore_initial_state
        self.snapshot_getter = snapshot_getter
        self.snapshot_restorer = snapshot_restorer

    @property
    def initial_state_sha256(self) -> str:
        return canonical_initial_state_sha256(self.initial_state, self.identity)

    def _restore_initial(self, env: Any) -> Any:
        if hasattr(env, "seed"):
            env.seed(self.seed)
        obs = env.reset()
        if self.restore_initial_state is not None:
            restored = self.restore_initial_state(env, copy.deepcopy(self.initial_state))
        elif hasattr(env, "set_init_state"):
            restored = env.set_init_state(copy.deepcopy(self.initial_state))
        else:
            raise CanonicalExecutionError("INITIAL_STATE_RESTORE_UNBOUND")
        return obs if restored is None else restored

    def _physical_state(self, env: Any, obs: Any, step: int) -> Any:
        if self.physical_state_getter is None:
            raise CanonicalExecutionError("PHYSICAL_STATE_GETTER_REQUIRED")
        return self.physical_state_getter(env, obs, step)

    def _start_clean_env(self) -> tuple[Any, Any]:
        seed_everything(self.seed)
        env = self.env_factory()
        try:
            return env, self._restore_initial(env)
        except Exception:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            raise

    def _clean_step(self, env: Any, obs: Any, step: int) -> tuple[Any, dict[str, Any], Any, bool, bool]:
        observed = canonical_value(self.observation_getter(env, obs, step))
        policy_step = PolicyStep.from_value(self.policy(step, obs, self.task_label))
        diagnostics = None
        if self.diagnostic_getter is not None:
            diagnostics = canonical_value(self.diagnostic_getter(env, obs, step, policy_step))
        raw_capture = None
        if self.raw_capture_getter is not None and step in self.raw_capture_steps:
            raw_capture = copy.deepcopy(self.raw_capture_getter(env, obs, step, policy_step))
        raw_action = canonical_value(policy_step.raw_action)
        executed_action = self.action_postprocess(policy_step.raw_action)
        try:
            if len(executed_action) != 7:
                raise CanonicalExecutionError("POSTPROCESSED_ACTION_DIMENSION_INVALID")
        except TypeError as exc:
            raise CanonicalExecutionError("POSTPROCESSED_ACTION_NOT_SEQUENCE") from exc
        executed_value = canonical_value(executed_action)
        next_obs, reward, done, info = _step_env(env, executed_action)
        physical = canonical_value(self._physical_state(env, next_obs, step))
        success = bool(self.success_predicate(env, next_obs, info, done, step))
        row = {
            "schema": CANONICAL_TRACE_SCHEMA,
            "step": step,
            "observation": observed,
            "token_ids": [int(item) for item in policy_step.token_ids],
            "raw_action": raw_action,
            "postprocessed_action": executed_value,
            "reward": canonical_value(reward),
            "env_done": bool(done),
            "info": canonical_value(info),
            "physical_state": physical,
        }
        if diagnostics is not None:
            row["diagnostics"] = diagnostics
        if raw_capture is not None:
            row["raw_capture"] = raw_capture
        return next_obs, row, copy.deepcopy(executed_action), success, bool(self.termination_predicate(env, next_obs, info, done, step))

    def run_clean_episode(self, *, mode: str) -> EpisodeTrace:
        env, obs = self._start_clean_env()
        steps: list[dict[str, Any]] = []
        actions: list[Any] = []
        success = False
        reason = "official_horizon_exhausted"
        try:
            for _ in range(self.num_steps_wait):
                obs, _reward, _done, _info = _step_env(env, list(DUMMY_OPEN_ACTION))
            for step in range(self.suite_horizon):
                next_obs, row, action, success, terminated = self._clean_step(env, obs, step)
                actions.append(action)
                steps.append(row)
                obs = next_obs
                if terminated:
                    if success:
                        reason = "env_done_success"
                    else:
                        reason = "env_done_failure"
                    break
            else:
                success = bool(self.success_predicate(env, obs, {}, False, self.suite_horizon - 1))
                reason = "success_at_horizon" if success else "official_horizon_exhausted"
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if not steps:
            raise CanonicalExecutionError("EMPTY_POLICY_TRACE")
        return EpisodeTrace(
            identity=dict(self.identity),
            initial_state_sha256=self.initial_state_sha256,
            steps=steps,
            actions=actions,
            terminal_outcome="SUCCESS" if success else "TASK_FAILURE",
            termination_step=len(steps) - 1,
            termination_reason=reason,
        )

    def run_clean_prefix_and_snapshot(self, *, probe_step: int) -> dict[str, Any]:
        """Run the shared clean prefix, capture a bound snapshot, then close."""
        if probe_step < 0 or probe_step >= self.suite_horizon:
            raise CanonicalExecutionError("PROBE_STEP_OUT_OF_RANGE")
        env, obs = self._start_clean_env()
        steps: list[dict[str, Any]] = []
        actions: list[Any] = []
        try:
            for _ in range(self.num_steps_wait):
                obs, _reward, _done, _info = _step_env(env, list(DUMMY_OPEN_ACTION))
            for step in range(probe_step + 1):
                next_obs, row, action, _success, terminated = self._clean_step(env, obs, step)
                steps.append(row)
                actions.append(action)
                obs = next_obs
                if terminated and step != probe_step:
                    raise CanonicalExecutionError("CLEAN_PREFIX_TERMINATED_BEFORE_PROBE")
            snapshot = copy.deepcopy(self.capture_snapshot(env, obs))
            return {
                "identity": dict(self.identity),
                "initial_state_sha256": self.initial_state_sha256,
                "probe_step": probe_step,
                "probe_state_sha256": canonical_initial_state_sha256(snapshot, {"probe_step": probe_step, **self.identity}),
                "steps": steps,
                "actions": actions,
                "snapshot": snapshot,
            }
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def capture_snapshot(self, env: Any, obs: Any) -> Any:
        if self.snapshot_getter is not None:
            return self.snapshot_getter(env, obs)
        if hasattr(env, "get_state"):
            return env.get_state()
        sim = getattr(env, "sim", None)
        if sim is not None and hasattr(sim, "get_state"):
            return sim.get_state()
        raise CanonicalExecutionError("SNAPSHOT_GETTER_REQUIRED")

    def restore_snapshot(self, env: Any, snapshot: Any) -> Any:
        if self.snapshot_restorer is not None:
            restored = self.snapshot_restorer(env, copy.deepcopy(snapshot))
        elif hasattr(env, "set_state"):
            restored = env.set_state(copy.deepcopy(snapshot))
        else:
            sim = getattr(env, "sim", None)
            if sim is None or not hasattr(sim, "set_state"):
                raise CanonicalExecutionError("SNAPSHOT_RESTORER_REQUIRED")
            restored = sim.set_state(copy.deepcopy(snapshot))
            if hasattr(sim, "forward"):
                sim.forward()
        if restored is not None:
            return restored
        observe = getattr(env, "_get_observations", None)
        if callable(observe):
            return observe()
        observe = getattr(env, "observe", None)
        if callable(observe):
            return observe()
        raise CanonicalExecutionError("SNAPSHOT_RESTORE_OBSERVATION_UNBOUND")

    def run_noop_continuation(self, *, env: Any, snapshot: Any, actions: Sequence[Any], mode: str = "SNAPSHOT_RESTORE_NOOP") -> dict[str, Any]:
        """Restore a clean snapshot and replay captured clean actions exactly."""
        restored_obs = self.restore_snapshot(env, snapshot)
        restore_row = {
            "schema": CANONICAL_TRACE_SCHEMA,
            "snapshot_sha256": canonical_initial_state_sha256(snapshot, {"snapshot": True}),
            "restored_observation": canonical_value(self.observation_getter(env, restored_obs, 0)),
        }
        noop_rows: list[dict[str, Any]] = []
        obs = restored_obs
        for step, action in enumerate(actions):
            next_obs, reward, done, info = _step_env(env, copy.deepcopy(action))
            noop_rows.append({
                "schema": CANONICAL_TRACE_SCHEMA,
                "step": step,
                "postprocessed_action": canonical_value(action),
                "observation": canonical_value(self.observation_getter(env, obs, step)),
                "next_observation": canonical_value(self.observation_getter(env, next_obs, step)),
                "reward": canonical_value(reward),
                "env_done": bool(done),
                "info": canonical_value(info),
                "physical_state": canonical_value(self._physical_state(env, next_obs, step)),
            })
            obs = next_obs
        return {"snapshot_restore": [restore_row], "noop_actions": noop_rows}

    def build_receipt(
        self,
        *,
        trace: EpisodeTrace,
        mode: str,
        comparison_scope: str,
        contract: Mapping[str, Any],
        trace_artifacts: Mapping[str, Mapping[str, Any]],
        trace_hashes: Mapping[str, str],
        diagnostic_trace_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
        diagnostic_trace_hashes: Mapping[str, str] | None = None,
        probe_step: int | None = None,
        probe_state_sha256: str | None = None,
    ) -> dict[str, Any]:
        contract = dict(contract)
        missing = [field for field in CONTRACT_FIELDS if field not in contract]
        extra = sorted(set(contract) - set(CONTRACT_FIELDS))
        if missing or extra:
            raise CanonicalExecutionError(f"EXECUTION_CONTRACT_FIELDS_INVALID:missing={missing}:extra={extra}")
        if contract["clean_core_sha256"] != sha256_file(Path(__file__)):
            raise CanonicalExecutionError("CLEAN_CORE_DIGEST_NOT_BOUND_TO_THIS_MODULE")
        if contract["initial_state_hash_algorithm"] != CANONICAL_INIT_STATE_HASH_ALGORITHM:
            raise CanonicalExecutionError("INITIAL_STATE_HASH_ALGORITHM_NOT_CANONICAL")
        if contract["initial_state_identity_schema"] != CANONICAL_INIT_STATE_SCHEMA:
            raise CanonicalExecutionError("INITIAL_STATE_IDENTITY_SCHEMA_NOT_CANONICAL")
        receipt = {
            "schema": "STAGE_V_RB1_RUNTIME_RECEIPT_V1",
            "mode": str(mode),
            "comparison_scope": str(comparison_scope),
            **{key: self.identity[key] for key in ("canonical_parent_key", "suite", "task_index", "state_index")},
            "execution_contract": contract,
            "execution_contract_sha256": canonical_sha256(contract),
            "clean_core_sha256": contract["clean_core_sha256"],
            "initial_state_sha256": trace.initial_state_sha256,
            "trace_step_count": len(trace.steps),
            "termination_step": trace.termination_step,
            "terminal_outcome": trace.terminal_outcome,
            "termination_reason": trace.termination_reason,
            "trace_hashes": dict(trace_hashes),
            "trace_artifacts": dict(trace_artifacts),
            "independent_recompute": {"status": "PENDING", "recomputed": False},
            "clean_prefix_shared": True,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
            "attack_rollouts": 0,
            "intervention_applied_steps": 0,
            "counterfactual_open_steps": 0,
        }
        if comparison_scope == "NOOP_CONTINUATION":
            if probe_step is None or probe_state_sha256 is None:
                raise CanonicalExecutionError("NOOP_PROBE_IDENTITY_REQUIRED")
            receipt.update({"probe_step": int(probe_step), "probe_state_sha256": str(probe_state_sha256)})
        if diagnostic_trace_artifacts is not None or diagnostic_trace_hashes is not None:
            receipt["diagnostic_trace_artifacts"] = dict(diagnostic_trace_artifacts or {})
            receipt["diagnostic_trace_hashes"] = dict(diagnostic_trace_hashes or {})
            receipt["diagnostic_instrumentation"] = {
                "schema": "STAGE_V_RB1_DIAGNOSTIC_INPUT_TRACE_V1",
                "comparison_gate": "NOT_A_STAGE_V_RB1_V1_PASS_GATE",
            }
        return receipt


def write_trace_artifacts(root: Path, trace: EpisodeTrace, *, noop: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Write deterministic trace files and return their independently hashable manifest."""
    root.mkdir(parents=True, exist_ok=False)
    rows: dict[str, Sequence[Mapping[str, Any]]] = {
        "policy_token_trace": [{"step": row["step"], "token_ids": row["token_ids"], "raw_action": row["raw_action"]} for row in trace.steps],
        "postprocessed_action_trace": [{"step": row["step"], "postprocessed_action": row["postprocessed_action"]} for row in trace.steps],
        "observation_trace": [{"step": row["step"], "observation": row["observation"]} for row in trace.steps],
        "physical_state_trace": [{"step": row["step"], "physical_state": row["physical_state"]} for row in trace.steps],
    }
    if noop is not None:
        rows["snapshot_restore_trace"] = noop["snapshot_restore"]
        rows["noop_action_trace"] = noop["noop_actions"]
    initial_path = root / "initial_state.json"
    initial_path.write_bytes(_json({"schema": CANONICAL_INIT_STATE_SCHEMA, "identity": canonical_value(trace.identity), "initial_state_sha256": trace.initial_state_sha256}) + b"\n")
    artifacts: dict[str, dict[str, Any]] = {"initial_state": {"path": initial_path.name, "sha256": sha256_file(initial_path)}}
    hashes: dict[str, str] = {}
    field_names = {
        "policy_token_trace": "policy_token_trace_sha256",
        "postprocessed_action_trace": "postprocessed_action_trace_sha256",
        "observation_trace": "observation_trace_sha256",
        "physical_state_trace": "physical_state_trace_sha256",
        "snapshot_restore_trace": "snapshot_restore_trace_sha256",
        "noop_action_trace": "noop_action_trace_sha256",
    }
    for name, values in rows.items():
        path = root / f"{name}.jsonl"
        path.write_bytes(b"".join(_json(canonical_value(row)) + b"\n" for row in values))
        digest = sha256_file(path)
        artifacts[name] = {"path": path.name, "sha256": digest}
        hashes[field_names[name]] = digest
    return artifacts, hashes


def write_diagnostic_trace_artifacts(root: Path, trace: EpisodeTrace) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Write non-gating simulator and model-input diagnostics when supplied."""
    if not any("diagnostics" in row for row in trace.steps):
        raise CanonicalExecutionError("DIAGNOSTIC_TRACE_NOT_CAPTURED")
    rows = {
        "full_sim_state_trace": [
            {"step": row["step"], "full_sim_state": row["diagnostics"]["full_sim_state"]}
            for row in trace.steps
        ],
        "policy_rgb_224_trace": [
            {"step": row["step"], "policy_rgb_224": row["diagnostics"]["policy_rgb_224"]}
            for row in trace.steps
        ],
        "model_input_trace": [
            {"step": row["step"], "model_inputs": row["diagnostics"]["model_inputs"]}
            for row in trace.steps
        ],
    }
    field_names = {
        "full_sim_state_trace": "full_sim_state_trace_sha256",
        "policy_rgb_224_trace": "policy_rgb_224_trace_sha256",
        "model_input_trace": "model_input_trace_sha256",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    root.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        path = root / f"{name}.jsonl"
        path.write_bytes(b"".join(_json(canonical_value(row)) + b"\n" for row in values))
        digest = sha256_file(path)
        artifacts[name] = {"path": path.name, "sha256": digest}
        hashes[field_names[name]] = digest
    return artifacts, hashes


def _raw_bytes(value: Any) -> tuple[str, list[int], bytes] | None:
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
        dtype = str(getattr(value, "dtype", ""))
        shape = [int(item) for item in getattr(value, "shape", ())]
        try:
            raw = value.numpy().tobytes()
        except Exception:
            import torch
            raw = value.view(torch.uint8).numpy().tobytes()
        return dtype, shape, raw
    if hasattr(value, "tobytes") and hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            raw = value.tobytes(order="C")
        except TypeError:
            raw = value.tobytes()
        return str(value.dtype), [int(item) for item in value.shape], raw
    return None


def _raw_array_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        fields: list[tuple[str, Any]] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            child = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(_raw_array_fields(item, child))
        return fields
    if isinstance(value, (list, tuple)):
        fields: list[tuple[str, Any]] = []
        for index, item in enumerate(value):
            fields.extend(_raw_array_fields(item, f"{prefix}[{index}]"))
        return fields
    if _raw_bytes(value) is not None:
        return [(prefix, value)]
    return []


def write_raw_capture_artifacts(root: Path, trace: EpisodeTrace) -> dict[str, Any]:
    """Write prospective raw-byte sidecars; never enters V1 common traces."""
    captures = [row for row in trace.steps if "raw_capture" in row]
    if not captures:
        raise CanonicalExecutionError("RAW_CAPTURE_NOT_CAPTURED")
    root.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for row in captures:
        step = int(row["step"])
        step_root = root / f"step_{step:06d}"
        step_root.mkdir()
        raw_capture = row["raw_capture"]
        groups = {
            "raw_observation": {key: value for key, value in _raw_array_fields(raw_capture.get("raw_observation", {})) if len(getattr(value, "shape", ())) == 3},
            "policy_rgb_224": {"policy_rgb_224": raw_capture.get("policy_rgb_224")},
            "model_inputs": dict(_raw_array_fields(raw_capture.get("model_inputs", {}))),
        }
        for group, fields in groups.items():
            for field, value in sorted(fields.items()):
                if value is None:
                    continue
                payload = _raw_bytes(value)
                if payload is None:
                    raise CanonicalExecutionError(f"RAW_CAPTURE_VALUE_NOT_ARRAY:{group}:{field}")
                dtype, shape, raw = payload
                safe = field.replace(".", "__").replace("[", "_").replace("]", "")
                relative = Path(f"step_{step:06d}") / f"{group}__{safe}.bin"
                binary_path = root / relative
                binary_path.write_bytes(raw)
                descriptor = {
                    "schema": "STAGE_V_M1_RAW_ARRAY_DESCRIPTOR_V1",
                    "field": field,
                    "group": group,
                    "dtype": dtype,
                    "shape": shape,
                    "byte_order": "native",
                    "contiguous_order": "C",
                    "byte_length": len(raw),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "binary_path": relative.as_posix(),
                }
                descriptor_path = binary_path.with_suffix(".json")
                descriptor_path.write_bytes(_json(descriptor) + b"\n")
                entries.append({"step": step, **descriptor, "descriptor_path": descriptor_path.relative_to(root).as_posix()})
    return {"schema": "STAGE_V_M1_RAW_CAPTURE_MANIFEST_V1", "steps": sorted({int(row["step"]) for row in captures}), "entries": entries}
