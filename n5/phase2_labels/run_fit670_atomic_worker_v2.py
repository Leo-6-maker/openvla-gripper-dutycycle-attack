"""Strict entry point for the legacy FIT670 atomic collector.

The implementation reuses the already-tested collector loop while replacing
its permissive V1 transition hook and enriching every entity with stable C1
logical identity. Formal collection must invoke this file, never the V1 worker
directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fit670_strict_contract as strict
import run_fit670_atomic_worker as legacy


def _arg_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
        return sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise strict.ContractViolation(f"missing required V2 argument: {name}") from exc


def _pop_arg(name: str) -> str:
    value = _arg_value(name)
    index = sys.argv.index(name)
    del sys.argv[index:index + 2]
    return value


LIBERO_ROOT = Path(_pop_arg("--libero-root")).resolve()
REPO_ROOT = Path(__file__).resolve().parents[2]
ARGS = {
    "shard_id": int(_arg_value("--shard-id")),
    "physical_gpu": int(_arg_value("--gpu")),
    "model_path": Path(_arg_value("--model-path")).resolve(),
    "official_worker": Path(_arg_value("--official-worker")).resolve(),
    "transition_root": Path(_arg_value("--transition-receipt")).resolve(),
    "allowlist": Path(_arg_value("--identity-allowlist")).resolve(),
    "shard_plan": Path(_arg_value("--shard-plan")).resolve(),
    "registry_root": Path(_arg_value("--registry-root")).resolve(),
    "alias_ledger": Path(_arg_value("--alias-ledger")).resolve(),
    "upstream_root": Path(_arg_value("--upstream-root")).resolve(),
    "output_root": Path(_arg_value("--output-root")).resolve(),
    "max_identities": int(_arg_value("--max-identities")),
}
COLLECTION_MODE = "canary" if ARGS["max_identities"] == 1 else "formal"
if ARGS["max_identities"] not in (0, 1):
    raise strict.ContractViolation("V2 max-identities must be exactly 0 or 1")
LIBERO_IMPORT_ORIGIN = strict.assert_import_origin("libero", LIBERO_ROOT)

SOURCE_FILES = {
    "fit670_strict_contract.py": Path(strict.__file__).resolve(),
    "run_fit670_atomic_worker_v2.py": Path(__file__).resolve(),
    "run_fit670_atomic_worker.py": Path(legacy.__file__).resolve(),
    "fit_collection_core.py": Path(legacy.__file__).resolve().parent / "fit_collection_core.py",
    "run_fit670_supervisor_v2.py": Path(__file__).resolve().parent / "run_fit670_supervisor_v2.py",
    "finalize_fit670_collection_v2.py": Path(__file__).resolve().parent / "finalize_fit670_collection_v2.py",
    "run_fit670_v2.sh": Path(__file__).resolve().parent / "run_fit670_v2.sh",
    "validate_fit670_canary_v2.py": Path(__file__).resolve().parent / "validate_fit670_canary_v2.py",
}

_transition_manifest = None
_, _allowlisted_identities = strict.validate_allowlist(ARGS["allowlist"])
_original_collect_entity = legacy.collect_entity
_original_load_resolutions = legacy.load_resolutions
_original_validate_shapes = legacy._validate_episode_shapes
_original_capture_one = legacy.capture_one_fit670_episode
_original_seal_root = legacy.seal_root


def seal_root_v2(root):
    root = Path(root)
    manifest_path = root / "WORKER_MANIFEST.json"
    if manifest_path.is_file():
        if _transition_manifest is None:
            raise strict.ContractViolation("worker seal attempted before strict transition")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "schema": "FIT670_ATOMIC_WORKER_V2",
                "transition_schema": strict.TRANSITION_SCHEMA,
                "identity_set_digest": _transition_manifest["identity_set_digest"],
                "shard_plan_sha256": _transition_manifest["shard_plan_sha256"],
                "collection_source_commit": _transition_manifest[
                    "collection_source_commit"
                ],
                "collection_source_tree": _transition_manifest[
                    "collection_source_tree"
                ],
                "libero_import_origin": LIBERO_IMPORT_ORIGIN,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return _original_seal_root(root)


def verify_transition_adapter(*_args, **_kwargs):
    global _transition_manifest
    registry_summary = ARGS["registry_root"].parent / "ENTITY_REGISTRY_V2_SUMMARY.json"
    _transition_manifest = strict.validate_transition_v2(
        ARGS["transition_root"],
        allowlist_path=ARGS["allowlist"],
        shard_plan_path=ARGS["shard_plan"],
        output_root=ARGS["output_root"],
        physical_gpu=ARGS["physical_gpu"],
        shard_id=ARGS["shard_id"],
        model_path=ARGS["model_path"],
        official_worker=ARGS["official_worker"],
        registry_summary=registry_summary,
        alias_ledger=ARGS["alias_ledger"],
        repo_root=REPO_ROOT,
        upstream_root=ARGS["upstream_root"],
        libero_root=LIBERO_ROOT,
        source_files=SOURCE_FILES,
        collection_mode=COLLECTION_MODE,
    )
    return _transition_manifest


def load_resolutions_v2(registry_path, allow_articulated=False):
    resolutions, relations = _original_load_resolutions(
        registry_path, allow_articulated=allow_articulated
    )
    if not isinstance(resolutions, dict):
        raise strict.ContractViolation(f"resolver output is not a dict: {registry_path}")
    for key, resolution in resolutions.items():
        if not isinstance(resolution, dict):
            raise strict.ContractViolation(f"invalid C1 resolution at {key}")
        logical = next(
            (
                resolution.get(name)
                for name in ("logical_name", "bddl_name", "name", "entity_name")
                if resolution.get(name)
            ),
            None,
        )
        if not logical:
            raise strict.ContractViolation(f"C1 resolution has no logical name at {key}")
        resolution["logical_name"] = logical
        resolution["resolution_kind"] = (
            resolution.get("resolution_kind") or resolution.get("resolution")
        )
    return resolutions, relations


def collect_entity_v2(model, data, resolution):
    return strict.enrich_entity_record(
        _original_collect_entity(model, data, resolution), resolution
    )


def validate_episode_shapes_v2(episode):
    if _transition_manifest is None:
        raise strict.ContractViolation("episode validation ran before strict transition")
    episode["schema"] = strict.EPISODE_SCHEMA
    episode["n_steps"] = episode.get("step_count")
    identity = _allowlisted_identities.get(episode.get("episode_id"))
    if identity is None:
        raise strict.ContractViolation("episode is not in frozen allowlist")
    episode["initial_state_sha256"] = identity["initial_state_sha256"]
    bindings = dict(episode.get("bindings") or {})
    bindings.update(
        {
            "identity_set_digest": _transition_manifest["identity_set_digest"],
            "shard_plan_sha256": _transition_manifest["shard_plan_sha256"],
            "collection_source_commit": _transition_manifest["collection_source_commit"],
            "collection_source_tree": _transition_manifest["collection_source_tree"],
            "transition_schema": strict.TRANSITION_SCHEMA,
        }
    )
    episode["bindings"] = bindings
    episode["episode_bindings"] = dict(bindings)
    for step, telemetry in enumerate(episode.get("telemetry") or []):
        entities = telemetry.get("entities")
        if not isinstance(entities, list):
            raise strict.ContractViolation(f"telemetry entities missing at step {step}")
        for entity in entities:
            strict._required(
                entity,
                (
                    "logical_name", "alias_to", "resolution_kind",
                    "binding_identity", "role", "entity_type", "entity_id",
                ),
                f"telemetry step {step}",
            )
        contacts = telemetry.get("contact_pairs")
        if not isinstance(contacts, list):
            raise strict.ContractViolation(f"contact_pairs missing at step {step}")
        if telemetry.get("contact_ncon_total") != len(contacts):
            raise strict.ContractViolation(f"contact closure mismatch at step {step}")
        if telemetry.get("contact_truncated") is not False:
            raise strict.ContractViolation(f"contact truncation at step {step}")
    return _original_validate_shapes(episode)


def capture_one_v2(module, suite, task_idx, state_id, collection_seed,
                   registry_dir, canonical_state, task, adapter, output_root,
                   gpu_info=None, provenance=None, episode_bindings=None,
                   save_student_rgb=True):
    episode_id = f"{suite}/task_{task_idx:02d}/state_{state_id:02d}"
    target = (
        Path(output_root) / "episodes" / suite
        / f"task_{task_idx:02d}" / f"state_{state_id:02d}"
    )
    if target.exists():
        if _transition_manifest is None:
            raise strict.ContractViolation("resume attempted before strict transition")
        identity = _allowlisted_identities.get(episode_id)
        if identity is None:
            raise strict.ContractViolation(f"resume identity is not allowlisted: {episode_id}")
        strict.validate_episode_v2(target, identity, _transition_manifest)
        return None, target
    return _original_capture_one(
        module, suite, task_idx, state_id, collection_seed,
        registry_dir, canonical_state, task, adapter, output_root,
        gpu_info=gpu_info,
        provenance=provenance,
        episode_bindings=episode_bindings,
        save_student_rgb=save_student_rgb,
    )


legacy.verify_transition = verify_transition_adapter
legacy.load_resolutions = load_resolutions_v2
legacy.collect_entity = collect_entity_v2
legacy._validate_episode_shapes = validate_episode_shapes_v2
legacy.capture_one_fit670_episode = capture_one_v2
legacy.seal_root = seal_root_v2

# The legacy main imports this symbol again from fit_transition. Replacing the
# module attribute alone is insufficient, so install the strict adapter there.
import fit_transition
fit_transition.verify_transition = verify_transition_adapter


def main() -> None:
    prefix = f".gpu_{ARGS['physical_gpu']}.worker_staging."
    try:
        legacy.main()
    except BaseException:
        for path in ARGS["output_root"].parent.glob(f"{prefix}*"):
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
        raise
    worker_root = ARGS["output_root"] / f"gpu_{ARGS['physical_gpu']}"
    manifest_path = worker_root / "WORKER_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("strict worker did not publish WORKER_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("n_fail") != 0:
        raise SystemExit(2)
    if manifest.get("n_success", 0) + manifest.get("n_skipped", 0) != manifest.get("n_assigned"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
