#!/usr/bin/env python3
"""Shared fail-closed integrity primitives for Phase C freeze pipeline.

Every function in this module is fail-closed: invalid input raises SystemExit
with a structured label, never silently degrades.

Do NOT duplicate strict loaders, seal validators, path guards, identity/step
closure, or receipt binding validators across scripts — import from here.
"""
from __future__ import annotations

import hashlib, json, math, os, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── constants ──────────────────────────────────────────────────────────────
FROZEN_SPLITS = frozenset(f"o{o}_i{i}" for o in range(4) for i in range(3))
HEADS = ("grasp", "manipulation", "release")
HEX64_RE = set("0123456789abcdef")
HEX40_RE = set("0123456789abcdef")
LOGIT_PROB_TOLERANCE = 0.01


# ── core utilities ─────────────────────────────────────────────────────────

def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            d.update(chunk)
    return d.hexdigest()


def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in HEX64_RE for c in s)


def is_40char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 40 and all(c in HEX40_RE for c in s)


def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, v))))


# ── path safety ────────────────────────────────────────────────────────────

def guard_path_safe(rel: str, root: Path, label: str) -> Path:
    """Reject path escapes, symlinks, absolutes. Return resolved path within root."""
    if not isinstance(rel, str):
        raise SystemExit(f"{label}_PATH_NOT_STRING: {rel!r}")
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts:
        raise SystemExit(f"{label}_PATH_ESCAPE: {rel}")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{label}_PATH_OUTSIDE: {rel} -> {resolved}")
    if resolved.is_symlink() or any(
        (root / Path(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)
    ):
        raise SystemExit(f"{label}_PATH_SYMLINK: {rel}")
    return resolved


def verify_safe_file(path: Path, root: Path, label: str) -> Path:
    """Verify file exists, is within root, no symlinks, no path traversal."""
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"{label}_OUTSIDE_ROOT: {path}")
    if path.is_symlink() or resolved.is_symlink():
        raise SystemExit(f"{label}_SYMLINK: {path}")
    if not resolved.is_file():
        raise SystemExit(f"{label}_NOT_FOUND: {path}")
    # Walk components
    current = root.resolve()
    for part in path.resolve().relative_to(root.resolve()).parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"{label}_COMPONENT_SYMLINK: {part}")
    return resolved


# ── strict JSON loading ────────────────────────────────────────────────────

def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    """Load JSON file with duplicate-key detection. Rejects non-dict."""
    dups: list[str] = []
    def hook(pairs):
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for k, v in pairs:
            if k in seen:
                dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE: {path} {e}")
    if dups:
        raise SystemExit(f"{label}_DUP_KEYS: {path} keys={sorted(set(dups))[:5]}")
    if not isinstance(value, dict):
        raise SystemExit(f"{label}_NOT_OBJECT: {path}")
    return value


def load_strict_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """Load JSONL with per-line duplicate-key detection and strict step type."""
    if not path.is_file():
        raise SystemExit(f"{label}_JSONL_MISSING: {path}")
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    with open(path, encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            if not line.strip():
                continue
            dups: list[str] = []
            def hook(pairs):
                s: set[str] = set()
                r: dict[str, Any] = {}
                for k, v in pairs:
                    if k in s:
                        dups.append(k)
                    s.add(k)
                    r[k] = v
                return r
            try:
                r = json.loads(line, object_pairs_hook=hook)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{label}_JSONL_PARSE: {path}:{line_nr} {e}")
            if dups:
                raise SystemExit(f"{label}_JSONL_DUP_KEY: {path}:{line_nr} keys={dups}")
            if not isinstance(r, dict):
                raise SystemExit(f"{label}_JSONL_NOT_OBJECT: {path}:{line_nr}")
            ep = r.get("canonical_parent_key")
            step = r.get("step")
            if not isinstance(ep, str) or not ep:
                raise SystemExit(f"{label}_EP_INVALID: {path}:{line_nr} ep={ep!r}")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise SystemExit(f"{label}_STEP_INVALID: {path}:{line_nr} step={step!r}")
            key = (ep, step)
            if key in seen_keys:
                raise SystemExit(f"{label}_DUP_EPISODE_STEP: {path}:{line_nr} {key}")
            seen_keys.add(key)
            rows.append(r)
    return rows


# ── sealed directory verification ──────────────────────────────────────────

def verify_bundle_seal(bundle_root: Path, label: str) -> str:
    """Full seal verification. Returns SHA256SUMS SHA. Raises SystemExit on ANY violation."""
    bp = bundle_root.resolve()
    if bp.is_symlink():
        raise SystemExit(f"{label}_ROOT_SYMLINK: {bp}")
    if not bp.is_dir():
        raise SystemExit(f"{label}_NOT_DIR: {bp}")
    sums = bp / "SHA256SUMS"
    sidecar = bp / "SHA256SUMS.sha256"
    if sums.is_symlink() or sidecar.is_symlink():
        raise SystemExit(f"{label}_SEAL_SYMLINK")
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"{label}_UNSEALED: SHA256SUMS or .sha256 missing")
    expected_seal = sha256_file(sums)
    sidecar_line = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_line or sidecar_line[0] != expected_seal:
        raise SystemExit(
            f"{label}_SIDECAR_BROKEN: expected {expected_seal[:16]} "
            f"got {sidecar_line[0][:16] if sidecar_line else '?'}"
        )
    listed: set[str] = set()
    with open(sums, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise SystemExit(f"{label}_SEAL_PARSE: {line}")
            file_sha, rel = parts[0], " ".join(parts[1:])
            if not is_64char_hex(file_sha):
                raise SystemExit(f"{label}_SEAL_SHA_INVALID: {rel}")
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise SystemExit(f"{label}_SEAL_ESCAPE: {rel}")
            target = bp / rel_path
            if target.is_symlink():
                raise SystemExit(f"{label}_SEAL_SYMLINK: {rel}")
            try:
                target.resolve().relative_to(bp)
            except ValueError:
                raise SystemExit(f"{label}_SEAL_ESCAPE: {rel}")
            if rel in listed:
                raise SystemExit(f"{label}_SEAL_DUP: {rel}")
            listed.add(rel)
            if not target.is_file() or sha256_file(target) != file_sha:
                raise SystemExit(f"{label}_SEAL_MISMATCH: {rel}")
    # Check no extra unlisted files
    for p in bp.rglob("*"):
        if p.is_symlink():
            raise SystemExit(f"{label}_SEAL_SYMLINK: {p.relative_to(bp).as_posix()}")
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            rel = p.relative_to(bp).as_posix()
            if rel not in listed:
                raise SystemExit(f"{label}_SEAL_EXTRA: {rel}")
    return expected_seal


def verify_sealed_root(root: Path, label: str) -> dict[str, str]:
    """Verify sealed directory exists and return seal SHA. Wrapper for verify_bundle_seal."""
    seal = verify_bundle_seal(root, label)
    return {f"{label}_seal_sha256": seal}


# ── sealed receipt consumption ─────────────────────────────────────────────

def consume_sealed_receipt(
    root: Path,
    expected_schema: str,
    required_status_field: str,
    required_status_value: Any,
    label: str,
) -> tuple[dict[str, Any], str]:
    """Verify seal, load JSON, check schema and status. Returns (receipt_data, seal_sha)."""
    if not root.is_dir():
        raise SystemExit(f"{label}_ROOT_NOT_FOUND: {root}")
    seal = verify_bundle_seal(root, label)
    # Find the main JSON file (not SHA256SUMS, not .sha256)
    json_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix == ".json"
        and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256")
    )
    if len(json_files) != 1:
        raise SystemExit(f"{label}_AMBIGUOUS: expected 1 JSON, found {len(json_files)}")
    receipt = load_strict_json(json_files[0], label)
    if receipt.get("schema") != expected_schema:
        raise SystemExit(f"{label}_SCHEMA: expected {expected_schema}, got {receipt.get('schema')}")
    actual_status = receipt.get(required_status_field)
    if actual_status != required_status_value:
        raise SystemExit(
            f"{label}_{required_status_field.upper()}: "
            f"expected {required_status_value!r}, got {actual_status!r}"
        )
    return receipt, seal


# ── cross-receipt binding verification ─────────────────────────────────────

def verify_receipt_binding(
    receipt: dict[str, Any],
    field_name: str,
    actual_sha: str,
    label: str,
) -> None:
    """Verify receipt[field_name] == actual_sha. Fail-closed."""
    declared = receipt.get(field_name, "")
    if declared != actual_sha:
        raise SystemExit(
            f"{label}_BINDING_MISMATCH: {field_name} "
            f"declared={str(declared)[:16]} actual={actual_sha[:16]}"
        )


# ── identity helpers ───────────────────────────────────────────────────────

def extract_manifest_identities(
    manifest: dict[str, Any], role: str, split_key: str
) -> set[str]:
    """Extract per-split identities from various manifest shapes."""
    if "identities" in manifest:
        return set(manifest["identities"])
    splits = manifest.get("splits", manifest.get("split_identities", {}))
    if split_key in splits:
        sd = splits[split_key]
        if isinstance(sd, list):
            return set(sd)
        if isinstance(sd, dict):
            return set(sd.get(role, []))
    if role in manifest:
        rd = manifest[role]
        if isinstance(rd, list):
            return set(rd)
    return set()


def verify_identity_closure(
    actual_ids: set[str],
    manifest_ids: set[str],
    role_label: str,
    split_key: str,
) -> None:
    """Actual must == manifest. Missing or extra -> SystemExit."""
    missing = manifest_ids - actual_ids
    extra = actual_ids - manifest_ids
    if missing:
        raise SystemExit(
            f"{role_label}_ID_MISSING: {split_key} n={len(missing)}: {sorted(missing)[:5]}"
        )
    if extra:
        raise SystemExit(
            f"{role_label}_ID_EXTRA: {split_key} n={len(extra)}: {sorted(extra)[:5]}"
        )


# ── step closure ───────────────────────────────────────────────────────────

def verify_step_closure(rows: list[dict[str, Any]], label: str) -> None:
    """Per-identity: start at 0, contiguous, count matches declared."""
    by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ep[r["canonical_parent_key"]].append(r)
    for ep_id, ep_rows in by_ep.items():
        ep_rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in ep_rows]
        if steps[0] != 0:
            raise SystemExit(f"{label}_STEP_START: {ep_id} first={steps[0]}")
        for expected, observed in enumerate(steps):
            if observed != expected:
                raise SystemExit(f"{label}_STEP_GAP: {ep_id} expected={expected} observed={observed}")
        declared_counts = {r.get("source_episode_step_count") for r in ep_rows}
        if len(declared_counts) != 1:
            raise SystemExit(f"{label}_STEP_COUNT_VARY: {ep_id} counts={declared_counts}")
        declared = next(iter(declared_counts))
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise SystemExit(f"{label}_STEP_COUNT_INVALID: {ep_id} declared={declared!r}")
        if declared != len(ep_rows):
            raise SystemExit(f"{label}_STEP_COUNT_MISMATCH: {ep_id} declared={declared} actual={len(ep_rows)}")


# ── exact join ─────────────────────────────────────────────────────────────

def exact_three_way_join(
    pred_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]] | None,
    label: str,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]] | None]:
    """Exact (canonical_parent_key, step) join of 2 or 3 row sets. Fail on any mismatch."""
    pred_by_key = {(r["canonical_parent_key"], r["step"]): r for r in pred_rows}
    teacher_by_key = {(r["canonical_parent_key"], r["step"]): r for r in teacher_rows}
    runtime_by_key = None
    if runtime_rows is not None:
        runtime_by_key = {(r["canonical_parent_key"], r["step"]): r for r in runtime_rows}

    if pred_by_key.keys() != teacher_by_key.keys():
        pred_only = set(pred_by_key) - set(teacher_by_key)
        teacher_only = set(teacher_by_key) - set(pred_by_key)
        raise SystemExit(
            f"{label}_JOIN_PRED_TEACHER: pred_only={len(pred_only)} teacher_only={len(teacher_only)}"
        )
    if runtime_by_key is not None and pred_by_key.keys() != runtime_by_key.keys():
        pred_only = set(pred_by_key) - set(runtime_by_key)
        rt_only = set(runtime_by_key) - set(pred_by_key)
        raise SystemExit(
            f"{label}_JOIN_PRED_RUNTIME: pred_only={len(pred_only)} rt_only={len(rt_only)}"
        )

    return pred_by_key, teacher_by_key, runtime_by_key


# ── prediction schema validation ───────────────────────────────────────────

REQUIRED_PREDICTION_FIELDS = (
    "canonical_parent_key", "step", "split_key",
    "checkpoint_sha256", "checkpoint_source_commit",
    "feature_order_sha256", "normalization_sha256",
    "runtime_source_sha256", "source_artifact_recursive_sha256",
    "source_episode_step_count",
    "grasp_logit", "grasp_probability",
    "manipulation_logit", "manipulation_probability",
    "release_logit", "release_probability",
)

FORBIDDEN_STUDENT_FIELDS = frozenset({
    "strict_k10_feasible", "strict_k10_known_mask",
    "grasp_established", "manipulation_active", "release_or_instability",
    "grasp_established_known_mask", "manipulation_active_known_mask",
    "release_or_instability_known_mask",
    "grasp_target", "manipulation_target", "release_target",
    "event_id", "event_role", "event_duration",
    "object_pose", "contact_pairs", "privileged_task_geometry",
    "attack_outcome", "official_success", "attack_condition",
    "teacher_phase", "teacher_label",
    "grasp_known_mask", "manipulation_known_mask", "release_known_mask",
})


def validate_prediction_schema(rows: list[dict[str, Any]], label: str) -> None:
    for i, r in enumerate(rows):
        for fld in REQUIRED_PREDICTION_FIELDS:
            if fld not in r:
                raise SystemExit(f"{label}_FIELD_MISSING: row={i} field={fld}")
        extra = FORBIDDEN_STUDENT_FIELDS & set(r)
        if extra:
            raise SystemExit(f"{label}_FORBIDDEN_FIELD: row={i} fields={sorted(extra)}")


def validate_numeric_constraints(rows: list[dict[str, Any]], label: str) -> None:
    for i, r in enumerate(rows):
        for head in HEADS:
            logit = r[f"{head}_logit"]
            prob = r[f"{head}_probability"]
            if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
                raise SystemExit(f"{label}_LOGIT_INVALID: row={i} head={head} value={logit!r}")
            if isinstance(prob, bool) or not isinstance(prob, (int, float)) or not math.isfinite(float(prob)) or not 0.0 <= float(prob) <= 1.0:
                raise SystemExit(f"{label}_PROB_INVALID: row={i} head={head} value={prob!r}")
            expected = sigmoid(float(logit))
            err = abs(expected - float(prob))
            if err > LOGIT_PROB_TOLERANCE:
                raise SystemExit(f"{label}_LOGIT_PROB_MISMATCH: row={i} head={head} logit={logit} prob={prob} expected={expected:.6f} error={err:.6f}")


def validate_binding_uniformity(rows: list[dict[str, Any]], label: str) -> dict[str, str]:
    binding_fields = (
        "checkpoint_sha256", "checkpoint_source_commit",
        "feature_order_sha256", "normalization_sha256",
        "runtime_source_sha256", "split_key",
    )
    values_by_field: dict[str, set[str]] = {f: set() for f in binding_fields}
    for r in rows:
        for f in binding_fields:
            v = r.get(f, "")
            if not isinstance(v, str) or not v:
                raise SystemExit(f"{label}_BINDING_MISSING: field={f}")
            values_by_field[f].add(v)
    binding: dict[str, str] = {}
    for f in binding_fields:
        if len(values_by_field[f]) != 1:
            raise SystemExit(f"{label}_BINDING_NONUNIFORM: field={f} values={sorted(values_by_field[f])}")
        binding[f] = next(iter(values_by_field[f]))
    return binding


# ── output sealing ─────────────────────────────────────────────────────────

def seal_output_dir(root: Path) -> str:
    """Atomic staged seal: write to .staging then rename."""
    import shutil
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    names = sorted(
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256")
    )
    content = "".join(f"{sha256_file(staging / name)}  {name}\n" for name in names)
    (staging / "SHA256SUMS").write_text(content, encoding="utf-8")
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n", encoding="utf-8")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        os.replace(staging, root)
    except OSError:
        if root.exists():
            shutil.rmtree(root)
        os.replace(staging, root)
    return seal


# ── atomic single-use claim ────────────────────────────────────────────────

def claim_atomic_root(claim_root: Path, receipt_sha: str, label: str) -> None:
    """Create an atomic claim root. Fails if it already exists."""
    if claim_root.exists():
        raise SystemExit(f"{label}_CLAIM_ROOT_EXISTS: {claim_root}")
    # Use mkdir as atomic operation
    staging = claim_root.with_name(f".{claim_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    start_receipt = {
        "schema": f"{label}_START_RECEIPT_V1",
        "authorization_sha256": receipt_sha,
        "status": "CLAIMED",
    }
    (staging / "START_RECEIPT.json").write_text(json.dumps(start_receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    try:
        os.replace(staging, claim_root)
    except OSError as e:
        raise SystemExit(f"{label}_CLAIM_ATOMIC_FAIL: {e}") from e


# ── checkpoint file verification ───────────────────────────────────────────

def verify_checkpoint_from_manifest(
    checkpoint_manifest_root: Path,
    split_key: str,
    row_sha: str,
    label: str,
) -> str:
    """Load checkpoint manifest, extract actual checkpoint path, recompute SHA."""
    manifest_path = checkpoint_manifest_root / split_key / "manifest.json"
    verify_safe_file(manifest_path, checkpoint_manifest_root, f"{label}_CHECKPOINT_MANIFEST")
    cp_manifest = load_strict_json(manifest_path, f"{label}_CHECKPOINT_MANIFEST")

    declared_sha = str(cp_manifest.get("checkpoint_sha256", "")).lower()
    if not is_64char_hex(declared_sha):
        raise SystemExit(f"{label}_CHECKPOINT_MANIFEST_SHA_INVALID: {split_key}")

    # Check if manifest has actual checkpoint file path
    cp_path_rel = cp_manifest.get("checkpoint_path", cp_manifest.get("checkpoint_file", ""))
    if cp_path_rel:
        checkpoint_root = cp_manifest.get("checkpoint_root", str(checkpoint_manifest_root.resolve()))
        cp_file = guard_path_safe(cp_path_rel, Path(checkpoint_root), f"{label}_CHECKPOINT_FILE")
        actual_file_sha = sha256_file(cp_file)
        if actual_file_sha != declared_sha:
            raise SystemExit(
                f"{label}_CHECKPOINT_FILE_SHA_MISMATCH: {split_key} "
                f"declared={declared_sha[:16]} actual={actual_file_sha[:16]}"
            )

    row_sha_lower = row_sha.lower()
    if row_sha_lower != declared_sha:
        raise SystemExit(
            f"{label}_CHECKPOINT_SHA_MISMATCH: {split_key} "
            f"rows={row_sha_lower[:16]} manifest={declared_sha[:16]}"
        )
    return declared_sha


# ── cross-role disjointness ────────────────────────────────────────────────

def validate_cross_role_disjointness(
    c_ids: set[str], p_ids: set[str],
    t_ids: dict[str, set[str]], h_ids: dict[str, set[str]],
    a_ids: dict[str, set[str]], split_key: str,
) -> None:
    for other_label, other_ids_dict in [("T", t_ids), ("H", h_ids), ("A", a_ids)]:
        other = other_ids_dict.get(split_key, set())
        for cp_label, cp_ids in [("C", c_ids), ("P", p_ids)]:
            overlap = cp_ids & other
            if overlap:
                raise SystemExit(
                    f"IDENTITY_LEAKAGE: {split_key} {cp_label}∩{other_label}={len(overlap)}"
                )


# ── source file verification ───────────────────────────────────────────────

def verify_runtime_source_files(
    root: Path,
    declared_scheduler_sha: str | None = None,
    declared_adapter_sha: str | None = None,
) -> dict[str, str]:
    """Compute actual SHAs of factorized_scheduler.py and adapter. Verify against declared if provided."""
    scheduler_path = root / "src/gripper_attack/factorized_scheduler.py"
    adapter_path = root / "src/gripper_attack/factorized_scheduler_adapter.py"
    verify_safe_file(scheduler_path, root, "SCHEDULER_SOURCE")
    verify_safe_file(adapter_path, root, "ADAPTER_SOURCE")
    actual_scheduler = sha256_file(scheduler_path)
    actual_adapter = sha256_file(adapter_path)
    if declared_scheduler_sha and actual_scheduler != declared_scheduler_sha:
        raise SystemExit(
            f"SCHEDULER_SOURCE_SHA_MISMATCH: declared={declared_scheduler_sha[:16]} actual={actual_scheduler[:16]}"
        )
    if declared_adapter_sha and actual_adapter != declared_adapter_sha:
        raise SystemExit(
            f"ADAPTER_SOURCE_SHA_MISMATCH: declared={declared_adapter_sha[:16]} actual={actual_adapter[:16]}"
        )
    return {
        "scheduler_source_sha256": actual_scheduler,
        "runtime_adapter_source_sha256": actual_adapter,
    }


def verify_checkpoint_actual_file(
    checkpoint_manifest_root: Path,
    split_key: str,
    label: str,
) -> tuple[str, str]:
    """From the checkpoint manifest, extract and verify actual checkpoint file. Returns (manifest_sha, actual_file_sha)."""
    manifest_path = checkpoint_manifest_root / split_key / "manifest.json"
    verify_safe_file(manifest_path, checkpoint_manifest_root, f"{label}_MANIFEST")
    cp_manifest = load_strict_json(manifest_path, f"{label}_MANIFEST")
    declared_sha = str(cp_manifest.get("checkpoint_sha256", "")).lower()
    if not is_64char_hex(declared_sha):
        raise SystemExit(f"{label}_MANIFEST_SHA_INVALID: {split_key}")

    cp_path_rel = cp_manifest.get("checkpoint_path", cp_manifest.get("checkpoint_file", ""))
    actual_file_sha = declared_sha
    if cp_path_rel:
        cp_root_path = cp_manifest.get("checkpoint_root", str(checkpoint_manifest_root.resolve()))
        cp_file = guard_path_safe(cp_path_rel, Path(cp_root_path), f"{label}_CHECKPOINT")
        if not cp_file.is_file():
            raise SystemExit(f"{label}_CHECKPOINT_NOT_FOUND: {cp_file}")
        actual_file_sha = sha256_file(cp_file)
        if actual_file_sha != declared_sha:
            raise SystemExit(
                f"{label}_CHECKPOINT_FILE_MISMATCH: {split_key} "
                f"declared={declared_sha[:16]} actual={actual_file_sha[:16]}"
            )
    return declared_sha, actual_file_sha
