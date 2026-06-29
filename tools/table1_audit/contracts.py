from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tools.table1_audit.common import canonical_digest, is_valid_sha256, load_json, sha256_file


COMMON_FIELDS = {"schema_version", "self_sha256", "metadata", "source_path"}


@dataclass(frozen=True)
class LoadedContract:
    kind: str
    path: Path
    actual_sha256: str
    schema_version: str
    data: dict
    extras: dict

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "actual_sha256": self.actual_sha256,
            "schema_version": self.schema_version,
            "content_digest": canonical_digest(self.data),
            "extras": sorted(self.extras),
        }


class Contract:
    kind: ClassVar[str] = "contract"
    versions: ClassVar[set[str]] = set()
    required: ClassVar[set[str]] = set()
    formal_critical: ClassVar[set[str]] = set()

    @classmethod
    def load(cls, path: Path) -> LoadedContract:
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: contract must be a JSON object")
        version = str(data.get("schema_version") or "")
        if version not in cls.versions:
            raise ValueError(f"{path}: unsupported {cls.kind} schema_version {version!r}")
        missing = sorted(cls.required - data.keys())
        if missing:
            raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
        extras = {k: data[k] for k in sorted(set(data) - cls.required - COMMON_FIELDS)}
        for key, value in _walk(data):
            if key.endswith("_sha256") and key != "self_sha256":
                if isinstance(value, str) and value not in {"SERVER_SNAPSHOT_REQUIRED", "UNVERIFIED", "MISSING", "NOT_APPLICABLE"} and not is_valid_sha256(value):
                    raise ValueError(f"{path}: invalid sha256 field {key}")
        return LoadedContract(cls.kind, path.resolve(), sha256_file(path), version, data, extras)


class ManifestContract(Contract):
    kind = "manifest_contract"
    versions = {"manifest_contract.v1"}
    required = {
        "schema_version",
        "required_fields",
        "parent_fields",
        "replicate_field",
        "identity_allowlist",
        "result_denylist",
    }


class RuntimeLock(Contract):
    kind = "runtime_lock"
    versions = {"runtime_lock.v1"}
    required = {"schema_version", "required_sha256", "required_fields"}


class RetryPolicy(Contract):
    kind = "retry_policy"
    versions = {"retry_policy.v1"}
    required = {
        "schema_version",
        "legal_terminal_invalid_statuses",
        "terminal_reasons",
        "max_attempts",
    }


class RequiredArtifactSchema(Contract):
    kind = "required_artifact_schema"
    versions = {"required_artifact_schema.v1"}
    required = {"schema_version", "complete", "terminal_invalid"}


class ConditionSpec(Contract):
    kind = "condition_spec"
    versions = {"condition_spec.v1"}
    required = {
        "schema_version",
        "status",
        "condition_id",
        "allowed_output_root",
        "allowed_manifest_root",
        "fields",
        "clean_identity_allowlist",
        "clean_result_denylist",
        "bound_contract_sha256",
    }


class FreezeBundleContract(Contract):
    kind = "freeze_bundle_contract"
    versions = {"freeze_bundle_contract.v1"}
    required = {"schema_version", "required_files", "status_flow"}


def load_contract(path: Path, cls: type[Contract]) -> LoadedContract:
    return cls.load(path)


def _walk(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k), v
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
