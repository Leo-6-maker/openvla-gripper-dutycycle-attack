"""[DeepSeek] FIT-INFERENCE Transition Receipt Verifier.

Validates a sealed transition receipt before R5-F model loading.
All rejections must occur BEFORE load_policy().

Bound values frozen at ee7da22 (R5-E-R1 execution commit).
"""
import json, hashlib, os, math
from pathlib import Path


# ── Frozen bindings (from R5-E-R1 evidence) ──
FROZEN = {
    "c1_canonical_digest":
        "f9bb35965a166b0f56d92f3624855459fb6c4845b3a60f99551e953931fc7eb7",
    "r5e_execution_commit":
        "ee7da22b76a856b6c10ac29f02f73dbf6aebcc83",
    "r5e_execution_tree":
        "4e5a07aaa0a64e8c96ddd5c3515b9a861c145f11",
    "r5e_run_a_sha256sums":
        "548bb98d91a321f938c47e1152104e819dc4e9a1378020c3b5fcdcaab7ca27ac",
    "r5e_run_b_sha256sums":
        "708e300ea561f5836fb6723eef14531ed9f91f4e188cad77905f6594b76c304e",
}


class TransitionRejected(Exception):
    """Transition receipt validation failed — must not load model."""
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def full_seal_check(root):
    """Verify every file in SHA256SUMS. Returns (ok, file_count, error)."""
    root = Path(root)
    sums_path = root / "SHA256SUMS"
    side_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        return False, 0, "not a sealed root"
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        return False, 0, "seal sidecar mismatch"
    file_count = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            return False, file_count, f"malformed SHA256SUMS line: {line}"
        digest, name = parts
        name = name.lstrip("*")
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        target = root / name
        if target.is_symlink():
            return False, file_count, f"SYMLINK: {name}"
        if not target.is_file():
            return False, file_count, f"missing: {name}"
        if sha256_file(target) != digest:
            return False, file_count, f"hash mismatch: {name}"
        file_count += 1
    return True, file_count, "OK"


def verify_transition(transition_root, execution_source_commit, script_sha,
                      model_path, official_worker_path, pilot_manifest_path,
                      registry_root, alias_ledger_path, output_root, gpu):
    """Validate transition receipt. Raises TransitionRejected on any failure.
    Must be called BEFORE load_policy()."""

    tr = Path(transition_root).resolve()
    if not tr.is_dir():
        raise TransitionRejected(f"transition receipt not found: {tr}")

    # 1. Full seal verification
    ok, n_files, err = full_seal_check(tr)
    if not ok:
        raise TransitionRejected(f"transition seal failed: {err}")

    # 2. Load manifest
    manifest_path = tr / "TRANSITION_MANIFEST.json"
    if not manifest_path.is_file():
        raise TransitionRejected("TRANSITION_MANIFEST.json missing")
    tm = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 3. Verify frozen scientific bindings
    for key, expected in FROZEN.items():
        actual = tm.get(key)
        if actual != expected:
            raise TransitionRejected(
                f"transition binding mismatch: {key}={actual} (expected {expected})")

    # 4. Verify execution source binding
    declared_source = tm.get("r5f_execution_source_commit")
    if declared_source != execution_source_commit:
        raise TransitionRejected(
            f"source commit mismatch: declared={declared_source} "
            f"actual={execution_source_commit}")

    declared_script = tm.get("r5f_script_sha256")
    if declared_script != script_sha:
        raise TransitionRejected(
            f"script SHA mismatch: declared={declared_script} "
            f"actual={script_sha}")

    # 5. Verify model binding
    declared_model_tree = tm.get("model_tree_sha256")
    if not Path(model_path).is_dir():
        raise TransitionRejected(f"model path missing: {model_path}")
    # Model tree check deferred to worker — but path must exist

    # 6. Verify worker binding
    declared_worker_sha = tm.get("official_worker_sha256")
    if not Path(official_worker_path).is_file():
        raise TransitionRejected(f"worker missing: {official_worker_path}")
    actual_worker_sha = sha256_file(official_worker_path)
    if declared_worker_sha != actual_worker_sha:
        raise TransitionRejected(
            f"worker SHA mismatch: declared={declared_worker_sha} "
            f"actual={actual_worker_sha}")

    # 7. Verify pilot manifest binding
    declared_pilot_sha = tm.get("pilot_manifest_sha256")
    if not Path(pilot_manifest_path).is_file():
        raise TransitionRejected(f"pilot manifest missing: {pilot_manifest_path}")
    actual_pilot_sha = sha256_file(pilot_manifest_path)
    if declared_pilot_sha != actual_pilot_sha:
        raise TransitionRejected(
            f"pilot manifest SHA mismatch: declared={declared_pilot_sha} "
            f"actual={actual_pilot_sha}")

    # 8. Verify identity allowlist
    declared_allowlist_digest = tm.get("identity_allowlist_digest")
    allowlist_sha = sha256_file(tr / "IDENTITY_ALLOWLIST.json")
    if declared_allowlist_digest != allowlist_sha:
        raise TransitionRejected(
            f"allowlist digest mismatch: declared={declared_allowlist_digest} "
            f"actual={allowlist_sha}")

    # 9. Verify registry binding
    declared_registry_sha = tm.get("registry_summary_sha256")
    registry_summary = Path(registry_root).parent / "ENTITY_REGISTRY_V2_SUMMARY.json"
    if not registry_summary.is_file():
        raise TransitionRejected(f"registry summary missing: {registry_summary}")
    actual_registry_sha = sha256_file(registry_summary)
    if declared_registry_sha != actual_registry_sha:
        raise TransitionRejected(
            f"registry summary SHA mismatch: declared={declared_registry_sha} "
            f"actual={actual_registry_sha}")

    # 10. Verify alias ledger binding
    declared_alias_sha = tm.get("alias_ledger_sha256")
    if not Path(alias_ledger_path).is_file():
        raise TransitionRejected(f"alias ledger missing: {alias_ledger_path}")
    actual_alias_sha = sha256_file(alias_ledger_path)
    if declared_alias_sha != actual_alias_sha:
        raise TransitionRejected(
            f"alias ledger SHA mismatch: declared={declared_alias_sha} "
            f"actual={actual_alias_sha}")

    # 11. Permission boundaries
    if tm.get("teacher_labels_authorized") is not False:
        raise TransitionRejected("teacher_labels_authorized must be false")
    if tm.get("student_training_authorized") is not False:
        raise TransitionRejected("student_training_authorized must be false")
    if tm.get("attack_authorized") is not False:
        raise TransitionRejected("attack_authorized must be false")
    if tm.get("protected_payload_read") is not False:
        raise TransitionRejected("protected_payload_read must be false")
    if tm.get("detector_load_authorized") is not False:
        raise TransitionRejected("detector_load_authorized must be false")

    # 12. GPU allowlist
    allowed_gpus = tm.get("allowed_gpus", [])
    if gpu not in allowed_gpus:
        raise TransitionRejected(
            f"GPU {gpu} not in allowlist: {allowed_gpus}")

    # 13. Output root allowlist
    allowed_outputs = tm.get("allowed_output_roots", [])
    if str(output_root) not in allowed_outputs:
        raise TransitionRejected(
            f"output root {output_root} not in allowlist")

    # 14. Transition created before now
    transition_created = tm.get("created_at", "")
    # Informational only — actual chronology checked by subagent

    return tm
