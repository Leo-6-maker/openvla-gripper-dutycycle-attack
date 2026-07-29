"""Fail-closed audit for a sealed FIT670 V2 contact-complete canary.

The production entrypoint is intentionally inert until the source manifest says
PASS_ENGINEERING_CONSUMABLE_INPUT_GATE. This script never discovers or selects
identities from a directory; it consumes only the frozen manifest list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.v5_r3_teacher import (
    R3ContractError,
    canonicalize_fit670_episode,
    validate_contact_row,
)
from gripper_attack.seal_utils import rename_noreplace


CANARY_SCHEMA = "FIT670_V2_CANARY_CONTACT_TELEMETRY_V1"
CONSUMABLE_STATUS = "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"top-level seal missing: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"top-level seal sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid seal row: {line!r}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise ValueError(f"unsafe or missing sealed file: {name}")
        if sha256_file(root / relative) != digest:
            raise ValueError(f"sealed file mismatch: {name}")
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in sealed root: {path}")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise ValueError(f"sealed file closure mismatch: missing={sorted(actual-listed)} extra={sorted(listed-actual)}")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": len(listed)}


def _safe_relative_episode(root: Path, identity: str) -> Path:
    relative = Path("episodes")
    for part in identity.split("/"):
        if not part or part in {".", ".."}:
            raise ValueError(f"unsafe episode identity: {identity!r}")
        relative /= part
    relative /= "episode.json"
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"episode escapes input root: {identity!r}")
    return relative


def _identity_digest(identities: list[str]) -> str:
    payload = json.dumps(sorted(identities), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_transition_binding(path: Path, review: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path.resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"transition manifest missing: {manifest_path}")
    root = manifest_path.parent
    transition_seal = verify_seal(root)
    expected_sums_sha = review.get("transition_sha256sums_sha256")
    if expected_sums_sha != transition_seal["sha256sums_sha256"]:
        raise ValueError("transition seal binding mismatch")
    transition = json.loads(manifest_path.read_text(encoding="utf-8"))
    if transition.get("protected_overlap_verified") != 0 or transition.get("protected_payload_read") is not False:
        raise ValueError("transition protected-overlap contract is not closed")
    if transition.get("identity_set_digest") != review.get("identity_set_digest"):
        raise ValueError("transition identity-set digest mismatch")
    allowlist_path = transition.get("identity_allowlist_path")
    allowlist_sha = transition.get("identity_allowlist_file_sha256")
    if not isinstance(allowlist_path, str) or not re.fullmatch(r"[0-9a-f]{64}", str(allowlist_sha)):
        raise ValueError("transition identity allowlist binding missing")
    allowlist = Path(allowlist_path)
    if not allowlist.is_file() or sha256_file(allowlist) != allowlist_sha:
        raise ValueError("transition identity allowlist seal mismatch")
    allowlist_seal = verify_seal(allowlist.parent)
    if transition.get("identity_allowlist_root_sha256sums_sha256") != allowlist_seal["sha256sums_sha256"]:
        raise ValueError("transition identity allowlist root seal mismatch")
    allowlist_data = json.loads(allowlist.read_text(encoding="utf-8"))
    if allowlist_data.get("schema") != "FIT670_IDENTITY_ALLOWLIST_V1" or allowlist_data.get("protected_overlap") != 0:
        raise ValueError("identity allowlist schema/protected overlap is not closed")
    entries = allowlist_data.get("identities")
    if not isinstance(entries, list) or not all(isinstance(entry, Mapping) and entry.get("episode_id") for entry in entries):
        raise ValueError("identity allowlist entries are malformed")
    allowlist_ids = [str(entry["episode_id"]) for entry in entries]
    if len(set(allowlist_ids)) != len(allowlist_ids):
        raise ValueError("identity allowlist contains duplicates")
    canonical_entries = []
    for entry in entries:
        required = ("episode_id", "suite", "task_id", "state_id", "collection_seed", "initial_state_sha256")
        if any(key not in entry for key in required):
            raise ValueError("identity allowlist entry is incomplete")
        canonical_entries.append({key: entry[key] for key in required})
    computed_digest = hashlib.sha256(json.dumps(canonical_entries, sort_keys=True).encode()).hexdigest()
    if computed_digest != transition.get("identity_set_digest") or computed_digest != review.get("identity_set_digest") or computed_digest != allowlist_data.get("identity_set_digest"):
        raise ValueError("identity-set digest does not match allowlist/transition/review")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_path": str(manifest_path),
        "seal": transition_seal,
        "allowlist_sha256": allowlist_sha,
        "allowlist_path": str(allowlist),
        "allowlist_seal": allowlist_seal,
        "allowlist_ids": allowlist_ids,
        "allowlist_entries": {str(entry["episode_id"]): entry for entry in entries},
        "identity_set_digest": computed_digest,
    }


def _load_fit670_canary(input_root: Path, *, expected_count: int = 8, transition_manifest_path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Consume the sealed FIT670_EPISODE_V2 canary via its review/seal graph."""
    root = input_root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in FIT670 canary root: {path}")
        if ".staging." in path.name:
            raise ValueError(f"staging residue in FIT670 canary root: {path}")
    review_root = root / "CANARY_REVIEW_V2"
    review_seal = verify_seal(review_root)
    review_path = review_root / "CANARY_REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema") != "FIT670_CANARY_REVIEW_V2" or review.get("status") != CONSUMABLE_STATUS:
        raise ValueError("FIT670 canary review is not consumable")
    if review.get("n_episodes") != expected_count or len(review.get("episode_seals", {})) != expected_count:
        raise ValueError("FIT670 canary episode count mismatch")
    expected_shards = {str(index) for index in range(expected_count)}
    if review.get("n_shards") != expected_count or set(str(key) for key in review.get("worker_seals", {})) != expected_shards:
        raise ValueError("FIT670 canary shard count mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(review.get("collection_source_commit", ""))) or not re.fullmatch(r"[0-9a-f]{40}", str(review.get("collection_source_tree", ""))):
        raise ValueError("FIT670 canary source lineage is invalid")
    if transition_manifest_path is None:
        raise ValueError("FIT670 canary requires the sealed transition manifest")
    transition_binding = _verify_transition_binding(transition_manifest_path, review)
    for shard_id, expected in sorted(review["worker_seals"].items(), key=lambda item: int(item[0])):
        worker_root = root / f"gpu_{int(shard_id)}"
        worker_seal = verify_seal(worker_root)
        if worker_seal["sha256sums_sha256"] != expected:
            raise ValueError(f"FIT670 worker seal mismatch: {shard_id}")
    loaded: list[dict[str, Any]] = []
    identities: set[str] = set()
    for identity, expected_seal in sorted(review["episode_seals"].items()):
        relative = _safe_relative_episode(root, identity)
        episode_root = (root / relative).parent
        episode_seal = verify_seal(episode_root)
        if episode_seal["sha256sums_sha256"] != expected_seal:
            raise ValueError(f"FIT670 episode seal mismatch: {identity}")
        episode = json.loads((root / relative).read_text(encoding="utf-8"))
        if episode.get("episode_id") != identity or episode.get("attack_enabled") is not False or episode.get("teacher_labels_generated") is not False:
            raise ValueError(f"FIT670 episode binding/authorization mismatch: {identity}")
        expected_identity = f"{episode.get('suite')}/task_{int(episode.get('task_id')):02d}/state_{int(episode.get('state_id')):02d}"
        if expected_identity != identity:
            raise ValueError(f"FIT670 episode suite/task/state mismatch: {identity}")
        if identity in identities:
            raise ValueError(f"duplicate FIT670 identity: {identity}")
        if identity not in transition_binding["allowlist_ids"]:
            raise ValueError(f"FIT670 identity is outside the bound allowlist: {identity}")
        allowlist_entry = transition_binding["allowlist_entries"][identity]
        if episode.get("collection_seed") != allowlist_entry["collection_seed"] or episode.get("initial_state_sha256") != allowlist_entry["initial_state_sha256"]:
            raise ValueError(f"FIT670 episode seed/initial-state binding mismatch: {identity}")
        identities.add(identity)
        provenance = episode.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"FIT670 provenance missing: {identity}")
        for key, pattern in (("collector_commit", r"[0-9a-f]{40}"), ("collector_tree", r"[0-9a-f]{40}"), ("collector_script_sha256", r"[0-9a-f]{64}")):
            if not re.fullmatch(pattern, str(provenance.get(key, ""))):
                raise ValueError(f"FIT670 provenance {key} invalid: {identity}")
        if provenance["collector_commit"] != review["collection_source_commit"] or provenance["collector_tree"] != review["collection_source_tree"]:
            raise ValueError(f"FIT670 episode source lineage mismatch: {identity}")
        rows = canonicalize_fit670_episode(episode)
        for step, row in enumerate(rows):
            validate_contact_row(row, expected_step=step)
        loaded.append({
            "manifest": {
                "episode_id": identity,
                "suite": episode.get("suite"),
                "task_id": episode.get("task_id"),
                "state_id": episode.get("state_id"),
                "seed": episode.get("collection_seed"),
                "relative_path": relative.as_posix(),
                "step_count": len(rows),
                "source_sha256": sha256_file(root / relative),
                "source_commit": str(provenance["collector_commit"]),
                "source_tree": str(provenance["collector_tree"]),
                "source_command": str(provenance["collector_script_sha256"]),
                "environment": "official_a800",
                "episode_sha256sums_sha256": episode_seal["sha256sums_sha256"],
            },
            "rows": rows,
        })
    review_identity_set = {str(identity) for identity in review["episode_seals"]}
    if identities != review_identity_set:
        raise ValueError("FIT670 review/loaded episode identity closure mismatch")
    return {
        "schema": "FIT670_V2_CANARY_CONTACT_TELEMETRY_V1",
        "status": CONSUMABLE_STATUS,
        "source_schema": "FIT670_EPISODE_V2",
        "review_schema": review["schema"],
        "review_sha256sums_sha256": review_seal["sha256sums_sha256"],
        "protected_reads": False,
        "attack_enabled": False,
        "source_root": str(root),
        "collection_source_commit": review["collection_source_commit"],
        "collection_source_tree": review["collection_source_tree"],
        "identity_set_digest": review.get("identity_set_digest"),
        "episode_identity_digest": _identity_digest(sorted(identities)),
        "episode_identity_closure": "loaded_ids == review.episode_seals.keys() and loaded_ids subset of bound allowlist",
        "allowlist_identity_set_digest": transition_binding["identity_set_digest"],
        "identity_allowlist_sha256": transition_binding["allowlist_sha256"],
        "identity_allowlist_path": transition_binding["allowlist_path"],
        "transition_manifest_sha256": transition_binding["manifest_sha256"],
        "transition_manifest_path": transition_binding["manifest_path"],
        "transition_sha256sums_sha256": transition_binding["seal"]["sha256sums_sha256"],
        "transition_binding": transition_binding,
    }, loaded, {"sha256sums_sha256": review_seal["sha256sums_sha256"], "file_count": review_seal["file_count"], "seal_kind": "review_plus_worker_plus_episode"}


def load_consumable_episodes(input_root: Path, *, expected_count: int = 8, transition_manifest_path: Path | None = None, allow_synthetic_fixture: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = input_root.resolve()
    if (root / "CANARY_REVIEW_V2" / "CANARY_REVIEW.json").is_file():
        return _load_fit670_canary(root, expected_count=expected_count, transition_manifest_path=transition_manifest_path)
    root = input_root.resolve()
    seal = verify_seal(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("MANIFEST.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CANARY_SCHEMA:
        raise ValueError(f"unexpected canary schema: {manifest.get('schema')!r}")
    if manifest.get("status") != CONSUMABLE_STATUS:
        raise ValueError(f"input is not consumable: {manifest.get('status')!r}")
    if manifest.get("protected_reads") is not False or manifest.get("attack_enabled") is not False:
        raise ValueError("input manifest authorization is not FIT-only")
    if transition_manifest_path is None and not (allow_synthetic_fixture and manifest.get("synthetic_fixture") is True):
        raise ValueError("production input requires the sealed transition manifest")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise ValueError(f"expected exactly {expected_count} frozen episodes")
    identities: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("malformed episode manifest row")
        for key in ("episode_id", "suite", "task_id", "state_id", "seed", "relative_path", "step_count", "source_sha256", "source_commit", "source_tree", "source_command", "environment"):
            if key not in item or item[key] in (None, ""):
                raise ValueError(f"episode manifest missing {key}")
        identity = str(item["episode_id"])
        if identity in identities:
            raise ValueError(f"duplicate episode identity: {identity}")
        identities.add(identity)
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe episode path: {relative}")
        path = root / relative
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["source_sha256"])):
            raise ValueError(f"invalid episode source SHA: {identity}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(item["source_commit"])) or not re.fullmatch(r"[0-9a-f]{40}", str(item["source_tree"])):
            raise ValueError(f"invalid source lineage SHA: {identity}")
        if not path.is_file() or sha256_file(path) != str(item["source_sha256"]):
            raise ValueError(f"episode source seal mismatch: {identity}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != int(item["step_count"]):
            raise ValueError(f"episode step count mismatch: {identity}")
        for step, row in enumerate(rows):
            if row.get("episode_id") != identity:
                raise ValueError(f"episode identity mismatch: {identity} step={step}")
            validate_contact_row(row, expected_step=step)
        loaded.append({"manifest": item, "rows": rows})
    return manifest, loaded, seal


def audit(input_root: Path, output_root: Path | None = None, *, expected_count: int = 8, transition_manifest_path: Path | None = None, allow_synthetic_fixture: bool = False) -> dict[str, Any]:
    manifest, episodes, seal = load_consumable_episodes(input_root, expected_count=expected_count, transition_manifest_path=transition_manifest_path, allow_synthetic_fixture=allow_synthetic_fixture)
    report = {
        "schema": "V5_R3_CONTACT_INPUT_AUDIT_V1",
        "status": "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE",
        "input_schema": manifest["schema"],
        "input_status": manifest["status"],
        "identity_count": len(episodes),
        "step_count": sum(len(item["rows"]) for item in episodes),
        "source_sha256s": [item["manifest"]["source_sha256"] for item in episodes],
        "input_sha256sums_sha256": seal["sha256sums_sha256"],
        "protected_reads": 0,
        "attack_enabled": False,
        "teacher_labels_generated": False,
    }
    if output_root is not None:
        output = output_root.resolve()
        if output.exists():
            raise FileExistsError(output)
        staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
        if staging.exists():
            raise FileExistsError(staging)
        staging.mkdir(parents=True)
        (staging / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = [path for path in staging.rglob("*") if path.is_file()]
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in sorted(files)), encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        rename_noreplace(staging, output)
        report["output_root"] = str(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--transition-manifest", type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.input_root, args.output_root, expected_count=args.expected_count, transition_manifest_path=args.transition_manifest, allow_synthetic_fixture=args.synthetic_fixture), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
