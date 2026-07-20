"""R10.4-R4 runtime gates and replay-only helpers.

This module deliberately separates the executable runtime contract from model
loading.  The replay path never imports an OpenVLA checkpoint and never
creates a detector.  The passive path uses the same episode loop with injected
factories, so the fake CI path cannot hide a different control-flow branch.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .sc5_streaming_features_v2 import FEATURE_NAMES, SC5StreamingFeatureAdapterV2


ACTION_DIM = 7
NUM_STEPS_WAIT = 10
FEATURE_ORDER_SHA256 = "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366"
FEATURE_ABS_TOLERANCE = 1e-6
ACTION_ABS_TOLERANCE = 0.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R10_4ContractError(RuntimeError):
    """Fail-closed R4 contract violation."""


class HoldSourceIncomplete(R10_4ContractError):
    """A replay source does not contain all measured inputs required by R4B."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def pickle4_sha(value: Any) -> str:
    return sha256_bytes(pickle.dumps(value, protocol=4))


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or value in {"", "."} or any(part in {"", ".", ".."} for part in path.parts):
        raise R10_4ContractError(f"UNSAFE_CHECKSUM_PATH:{value!r}")
    return path.as_posix()


def verify_checksum_manifest(root: Path) -> dict[str, Any]:
    """Verify SHA256SUMS, its sidecar, every listed file, and exact closure."""

    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise R10_4ContractError(f"CHECKSUM_BUNDLE_INCOMPLETE:{root}")
    sidecar_rows = sidecar.read_text(encoding="utf-8").splitlines()
    sidecar_tokens = sidecar_rows[0].split() if sidecar_rows else []
    if len(sidecar_tokens) < 2 or sidecar_tokens[1] != sums.name:
        raise R10_4ContractError("CHECKSUM_SIDECAR_FORMAT_FAIL")
    if sidecar_tokens[0] != sha256_file(sums):
        raise R10_4ContractError("CHECKSUM_SIDECAR_DIGEST_FAIL")

    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split(maxsplit=1)
        if len(tokens) != 2 or not SHA256_RE.fullmatch(tokens[0]):
            raise R10_4ContractError(f"CHECKSUM_ROW_FORMAT_FAIL:{line!r}")
        relative = _safe_relative_path(tokens[1].lstrip(" *"))
        if relative in {sums.name, sidecar.name} or relative in listed:
            raise R10_4ContractError(f"CHECKSUM_DUPLICATE_OR_SELF:{relative}")
        listed[relative] = tokens[0]

    actual: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path in {sums, sidecar}:
            continue
        relative = path.relative_to(root).as_posix()
        actual[relative] = sha256_file(path)
    if set(actual) != set(listed):
        missing = sorted(set(listed) - set(actual))
        extra = sorted(set(actual) - set(listed))
        raise R10_4ContractError(f"CHECKSUM_FILE_SET_FAIL:missing={missing[:5]}:extra={extra[:5]}")
    mismatches = [name for name in sorted(listed) if actual[name] != listed[name]]
    if mismatches:
        raise R10_4ContractError(f"CHECKSUM_CONTENT_FAIL:{mismatches[:5]}")
    return {
        "root": str(root),
        "sha256sums_sha256": sha256_file(sums),
        "listed_file_count": len(listed),
        "file_set_sha256": canonical_json_sha(sorted(listed.items())),
    }


def verify_legacy_artifact_manifest(root: Path) -> dict[str, Any]:
    """Verify an immutable CLEAN artifact's legacy artifact_sha256.json closure."""

    root = root.resolve()
    manifest_path = root / "artifact_sha256.json"
    if not manifest_path.is_file():
        raise R10_4ContractError(f"ARTIFACT_MANIFEST_MISSING:{root}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files")
    if not isinstance(rows, list) or payload.get("recursive_sha256") != canonical_json_sha(rows):
        raise R10_4ContractError(f"ARTIFACT_MANIFEST_INVALID:{root}")
    listed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not SHA256_RE.fullmatch(str(row.get("sha256", ""))):
            raise R10_4ContractError("ARTIFACT_MANIFEST_ROW_INVALID")
        relative = _safe_relative_path(row["path"])
        if relative in listed or relative == manifest_path.name:
            raise R10_4ContractError(f"ARTIFACT_MANIFEST_DUPLICATE:{relative}")
        listed[relative] = row["sha256"]
    actual = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(listed):
        raise R10_4ContractError("ARTIFACT_FILE_SET_FAIL")
    if any(actual[name] != digest for name, digest in listed.items()):
        raise R10_4ContractError("ARTIFACT_FILE_DIGEST_FAIL")
    return {"root": str(root), "recursive_sha256": payload["recursive_sha256"], "file_count": len(rows)}


def feature_order_sha256() -> str:
    # The frozen S1 contract uses compact JSON separators.
    return canonical_json_sha(list(FEATURE_NAMES))


def validate_runtime_receipt(receipt: Mapping[str, Any], *, require_model_load: bool) -> None:
    """Validate a machine-built receipt before any model/device allocation."""

    required = {
        "schema": "R10_4_RUNTIME_AUTHORIZATION_RECEIPT_V1",
        "scope": "R10_4_R4_RUNTIME_INTEGRATION",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise R10_4ContractError(f"RUNTIME_RECEIPT_FIELD_FAIL:{key}")
    if receipt.get("feature_order_sha256") != FEATURE_ORDER_SHA256 or feature_order_sha256() != FEATURE_ORDER_SHA256:
        raise R10_4ContractError("RUNTIME_FEATURE_ORDER_BINDING_FAIL")
    if bool(receipt.get("model_load_authorized")) is not bool(require_model_load):
        raise R10_4ContractError("RUNTIME_MODEL_LOAD_SCOPE_FAIL")


def load_model_after_receipt(
    receipt: Mapping[str, Any], loader: Callable[[], Any]
) -> Any:
    """Only invoke an injected model loader after receipt validation."""

    validate_runtime_receipt(receipt, require_model_load=True)
    return loader()


@dataclass(frozen=True)
class RuntimeDependencies:
    env_factory: Callable[..., Any]
    task_initializer: Callable[[Any, Any, int], Any]
    image_getter: Callable[[Any], np.ndarray]
    feature_factory: Callable[[], Any] = SC5StreamingFeatureAdapterV2


class OfficialStreamingFeatureAdapter:
    """Environment-to-contract bridge around the frozen SC5 streamer.

    This class only extracts the already-defined physical fields and delegates
    all feature math and named ordering to ``SC5StreamingFeatureAdapterV2``.
    """

    def __init__(self) -> None:
        self.streamer = SC5StreamingFeatureAdapterV2()
        self.previous_eef: np.ndarray | None = None

    def reset(self) -> None:
        self.streamer.reset()
        self.previous_eef = None

    def update_from_env(self, observation: Any, env: Any, raw_action: np.ndarray, env_action: np.ndarray) -> dict[str, Any]:
        qpos = np.asarray(observation.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
        if qpos.size < 2:
            raise R10_4ContractError("RUNTIME_GRIPPER_QPOS_MISSING")
        try:
            site_id = env.sim.model.site_name2id("gripper0_grip_site")
            eef = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float32).reshape(3).copy()
        except Exception as exc:
            raise R10_4ContractError("RUNTIME_EEF_SITE_MISSING") from exc
        velocity = np.zeros(3, dtype=np.float32) if self.previous_eef is None else eef - self.previous_eef
        self.previous_eef = eef.copy()
        return self.streamer.update(
            step_id=self.streamer.next_expected_step,
            raw_gripper=float(raw_action[-1]),
            env_gripper=float(env_action[-1]),
            gripper_qpos=float(qpos[:2].sum()),
            gripper_opening_proxy=float(np.abs(qpos[:2]).sum()),
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
            action_dx=float(env_action[0]), action_dy=float(env_action[1]), action_dz=float(env_action[2]),
            action_gripper=float(raw_action[-1]),
        )


def initialize_env_once(env: Any, initial_state: Any, *, dummy_steps: int = NUM_STEPS_WAIT) -> Any:
    """Own reset, init-state, and dummy wait in exactly one place."""

    if getattr(env, "_r10_4_initialized", False):
        raise R10_4ContractError("RUNTIME_DOUBLE_INITIALIZATION")
    env._r10_4_initialized = True
    observation = env.set_init_state(initial_state)
    for _ in range(int(dummy_steps)):
        observation, _reward, _done, _info = env.step([0, 0, 0, 0, 0, 0, -1])
    return observation


def run_common_passive_loop(
    *,
    env: Any,
    initial_state: Any,
    task_language: str,
    adapter: Any,
    feature_adapter: Any,
    image_getter: Callable[[Any], np.ndarray],
    actions: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Common fake/real control flow; detector output is observation-only."""

    observation = initialize_env_once(env, initial_state)
    if hasattr(feature_adapter, "reset"):
        feature_adapter.reset()
    rows: list[dict[str, Any]] = []
    for step, recorded_action in enumerate(actions or []):
        image_np = np.asarray(image_getter(observation))
        if image_np.dtype != np.uint8:
            raise R10_4ContractError("RUNTIME_IMAGE_MUST_REMAIN_RAW_UINT8")
        clean_action, capture = adapter.predict_action(
            image_np=image_np,
            task_label=task_language,
            capture=True,
        )
        clean_action = np.asarray(clean_action, dtype=np.float32).reshape(-1)
        if clean_action.shape != (ACTION_DIM,):
            raise R10_4ContractError("RUNTIME_CLEAN_ACTION_SHAPE_FAIL")
        postprocess = getattr(adapter, "postprocess", None)
        executed = np.asarray(postprocess(clean_action) if callable(postprocess) else clean_action, dtype=np.float32).reshape(-1)
        if recorded_action is not None:
            expected = np.asarray(recorded_action, dtype=np.float32).reshape(-1)
            if expected.shape != (ACTION_DIM,) or not np.array_equal(executed, expected):
                raise R10_4ContractError("RUNTIME_REPLAY_ACTION_MISMATCH")
        if hasattr(feature_adapter, "update_from_env"):
            stream = feature_adapter.update_from_env(observation, env, clean_action, executed)
        else:
            stream = feature_adapter.update_from_observation(observation, executed)
        if not isinstance(stream, Mapping) or stream.get("valid") is not True:
            raise R10_4ContractError("RUNTIME_FEATURE_STREAM_INVALID")
        env_action = executed.copy()
        observation, _reward, done, _info = env.step(env_action.tolist())
        rows.append(
            {
                "step": step,
                "generation_passes_per_step": capture.get("generation_passes_per_step", 1),
                "features_25d": [float(stream["features"][name]) for name in FEATURE_NAMES],
                "clean_action_7d": executed.tolist(),
                "executed_action_7d": env_action.tolist(),
                "action_max_abs_error": float(np.max(np.abs(executed - env_action))),
                "done": bool(done),
            }
        )
        if done:
            break
    return rows


def official_env_dependencies(render_gpu_device_id: int) -> RuntimeDependencies:
    """Return official factory dependencies without constructing an env."""

    # Keep the CPU contract/import path usable on GitHub runners.  LIBERO is
    # imported only when the authorized replay path explicitly requests it.
    from .libero_v4_env_factory import build_v4_exact_env

    def factory(bddl_file: str, max_steps: int, dummy_steps: int) -> Any:
        return build_v4_exact_env(bddl_file, render_gpu_device_id, max_steps, dummy_steps)[0]

    def initializer(env: Any, initial_state: Any, dummy_steps: int) -> Any:
        return initialize_env_once(env, initial_state, dummy_steps=dummy_steps)

    def image_getter(observation: Any) -> np.ndarray:
        value = observation.get("agentview_image") if isinstance(observation, Mapping) else None
        if value is None:
            raise R10_4ContractError("RUNTIME_AGENTVIEW_IMAGE_MISSING")
        return np.asarray(value)

    return RuntimeDependencies(factory, initializer, image_getter)


__all__ = [
    "ACTION_DIM",
    "FEATURE_ABS_TOLERANCE",
    "FEATURE_NAMES",
    "FEATURE_ORDER_SHA256",
    "HoldSourceIncomplete",
    "R10_4ContractError",
    "RuntimeDependencies",
    "canonical_json_sha",
    "feature_order_sha256",
    "initialize_env_once",
    "load_model_after_receipt",
    "official_env_dependencies",
    "OfficialStreamingFeatureAdapter",
    "pickle4_sha",
    "run_common_passive_loop",
    "sha256_file",
    "validate_runtime_receipt",
    "verify_checksum_manifest",
    "verify_legacy_artifact_manifest",
]
