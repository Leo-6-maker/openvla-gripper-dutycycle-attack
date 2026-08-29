"""Reconcile the Git LF copy with the frozen historical M1 manifest bytes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_SUITE_COUNT = 4
EXPECTED_FILES_PER_SUITE = 25


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_lf(raw: bytes) -> bytes:
    if b"\r\n" in raw:
        raw = raw.replace(b"\r\n", b"\n")
    if b"\r" in raw:
        raise RuntimeError("M1_MANIFEST_MIXED_OR_LONE_CR_LINE_ENDING")
    return raw


def _suite_summary(suite: str, value: dict[str, Any]) -> dict[str, Any]:
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FILES_PER_SUITE or int(value.get("files", -1)) != EXPECTED_FILES_PER_SUITE:
        raise RuntimeError(f"M1_MANIFEST_SUITE_FILE_COUNT_INVALID:{suite}")
    normalized_rows = []
    paths = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"M1_MANIFEST_ROW_INVALID:{suite}")
        relative = str(row["path"]).replace("\\", "/")
        if relative in paths:
            raise RuntimeError(f"M1_MANIFEST_DUPLICATE_PATH:{suite}:{relative}")
        paths.add(relative)
        size = int(row["size"])
        digest = str(row["sha256"])
        total += size
        normalized_rows.append({"path": relative, "size": size, "sha256": digest})
    if total != int(value["bytes"]):
        raise RuntimeError(f"M1_MANIFEST_SUITE_BYTE_TOTAL_INVALID:{suite}")
    return {
        "repo_id": str(value["repo_id"]),
        "revision": str(value["revision"]),
        "files": len(normalized_rows),
        "bytes": total,
        "file_authority_sha256": canonical_hash(normalized_rows),
    }


def reconcile(manifest_path: Path, z1_config_path: Path) -> dict[str, Any]:
    raw = manifest_path.read_bytes()
    lf = normalize_lf(raw)
    crlf = lf.replace(b"\n", b"\r\n")
    z1 = load_json(z1_config_path)
    expected_crlf_sha256 = str(z1["model_families"]["M1_OPENVLA_OFT"]["checkpoint_manifests_sha256"])
    lf_sha256 = sha256_bytes(lf)
    crlf_sha256 = sha256_bytes(crlf)
    if crlf_sha256 != expected_crlf_sha256:
        raise RuntimeError(f"M1_MANIFEST_CRLF_AUTHORITY_MISMATCH:{crlf_sha256}:{expected_crlf_sha256}")
    if crlf.replace(b"\r\n", b"\n") != lf:
        raise RuntimeError("M1_MANIFEST_REVERSE_NORMALIZATION_MISMATCH")
    lf_json = json.loads(lf.decode("utf-8"))
    crlf_json = json.loads(crlf.decode("utf-8"))
    if lf_json != crlf_json:
        raise RuntimeError("M1_MANIFEST_JSON_SEMANTIC_MISMATCH")
    suites = lf_json.get("suites")
    if not isinstance(suites, dict) or len(suites) != EXPECTED_SUITE_COUNT:
        raise RuntimeError("M1_MANIFEST_SUITE_SET_INVALID")
    suite_summary = {suite: _suite_summary(suite, value) for suite, value in sorted(suites.items())}
    return {
        "schema": "STAGE_AC_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_V1",
        "status": "STAGE_AC_AC2R1_M1_MANIFEST_BYTE_RECONCILIATION_PASS",
        "gate": "STAGE_Z_AC2R1_M1_MANIFEST_BYTE_AUTHORITY_RECONCILIATION_AND_PRE_GPU_REQUALIFICATION",
        "historical_z1_authority": {
            "manifest_name": manifest_path.name,
            "declared_sha256": expected_crlf_sha256,
            "representation": "CRLF",
            "bytes": len(crlf),
            "sha256": crlf_sha256,
        },
        "git_runtime_representation": {
            "manifest_name": manifest_path.name,
            "representation": "LF",
            "bytes": len(lf),
            "sha256": lf_sha256,
        },
        "deterministic_conversion": {
            "lf_to_crlf": "replace each LF byte with CRLF; reject lone CR and mixed endings",
            "crlf_to_lf": "replace each CRLF pair with LF",
            "forward_exact": crlf == (lf.replace(b"\n", b"\r\n")),
            "reverse_exact": crlf.replace(b"\r\n", b"\n") == lf,
            "json_payload_equal": lf_json == crlf_json,
            "semantic_payload_sha256": canonical_hash(lf_json),
            "newline_count": lf.count(b"\n"),
        },
        "suite_authority": suite_summary,
        "checkpoint_authority": {
            "suite_count": len(suite_summary),
            "files_per_suite": EXPECTED_FILES_PER_SUITE,
            "all_path_size_sha_rows_preserved": True,
            "verifier_policy": "Use the unchanged frozen Z1 per-file verifier against an exact CRLF runtime copy.",
        },
        "z1_config": {
            "name": z1_config_path.name,
            "bytes": z1_config_path.stat().st_size,
            "sha256": sha256_file(z1_config_path),
        },
        "scientific_firewall": {
            "model_inference_calls": 0,
            "env_step_calls": 0,
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "physical_endpoint_reads": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
        },
        "next_legal_action": "M1_AC2R1_THREE_PERMANENTLY_EXCLUDED_CANARY_CLEAN_ONLY_REQUALIFICATION",
    }


def validate_reconciliation(manifest_path: Path, reconciliation_path: Path, z1_config_path: Path) -> dict[str, Any]:
    artifact = load_json(reconciliation_path)
    if artifact.get("status") != "STAGE_AC_AC2R1_M1_MANIFEST_BYTE_RECONCILIATION_PASS":
        raise RuntimeError("M1_RECONCILIATION_NOT_PASS")
    current = reconcile(manifest_path, z1_config_path)
    for key in ("historical_z1_authority", "git_runtime_representation", "deterministic_conversion", "suite_authority"):
        if artifact.get(key) != current.get(key):
            raise RuntimeError(f"M1_RECONCILIATION_ARTIFACT_MISMATCH:{key}")
    return current


def materialize_historical_runtime_manifest(manifest_path: Path, reconciliation_path: Path, z1_config_path: Path, output_path: Path) -> dict[str, Any]:
    current = validate_reconciliation(manifest_path, reconciliation_path, z1_config_path)
    lf = normalize_lf(manifest_path.read_bytes())
    crlf = lf.replace(b"\n", b"\r\n")
    expected = str(current["historical_z1_authority"]["sha256"])
    if sha256_bytes(crlf) != expected:
        raise RuntimeError("M1_RUNTIME_MANIFEST_HISTORICAL_SHA256_INVALID")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_bytes() != crlf:
            raise RuntimeError(f"M1_RUNTIME_MANIFEST_APPEND_ONLY_CONFLICT:{output_path}")
    else:
        temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
        temporary.write_bytes(crlf)
        os.replace(temporary, output_path)
    return {"path": str(output_path), "bytes": len(crlf), "sha256": sha256_bytes(crlf), "source_lf_sha256": current["git_runtime_representation"]["sha256"]}
