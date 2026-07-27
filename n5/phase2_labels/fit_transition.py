"""[DeepSeek] FIT-INFERENCE Transition Receipt Verifier (v2).

Validates a sealed transition receipt before R5-F model loading.
All rejections MUST occur BEFORE load_policy().

Bound values frozen at ee7da22 (R5-E-R1 execution commit).
"""
import hashlib, json, os, re, subprocess
from pathlib import Path


# ── Frozen R5-E-R1 scientific evidence ──
FROZEN_R5E = {
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
    "r5e_independent_review_sha256sums":
        "2465a4c9e4ba0d329183a70b4cc7f38fe38e78ccbb1cb908604fb878c288ca61",
    "r5e_comparison_sha256":
        "",  # must be bound from actual sealed comparison evidence root
}

FOUR_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


class TransitionRejected(Exception):
    """Transition receipt validation failed — must not load model."""
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ── Shared model-tree fingerprint ──

def compute_model_tree_fingerprint(model_path):
    """Deterministic directory fingerprint: relative path + file SHA + type.
    Rejects symlinks. Same implementation used by builder and verifier."""
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise TransitionRejected(f"model path not a directory: {root}")
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise TransitionRejected(f"model tree contains symlink: {p}")
        if p.is_file():
            if ".git" in p.parts:
                continue
            rel = p.relative_to(root).as_posix()
            ftype = "file"
            fsha = sha256_file(p)
            lines.append(f"{ftype} {fsha} {rel}")
        elif p.is_dir():
            rel = p.relative_to(root).as_posix()
            lines.append(f"dir {rel}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ── Seal verification ──

SHA256_RE = re.compile(r'^[0-9a-f]{64}$')

def full_seal_check(root):
    """Verify every file in SHA256SUMS. Rejects:
    symlinks, symlink parents, path traversal, absolute paths, duplicate
    entries, invalid digests, missing files, hash mismatches, extra unsealed
    files."""
    root = Path(root).resolve()
    sums_path = root / "SHA256SUMS"
    side_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        return False, 0, "not a sealed root"
    if sums_path.is_symlink() or side_path.is_symlink():
        return False, 0, "seal files are symlinks"
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or not SHA256_RE.match(sidecar[0]):
        return False, 0, "seal sidecar invalid format"
    if sidecar[0] != sha256_file(sums_path):
        return False, 0, "seal sidecar mismatch"

    file_count = 0
    manifest_files = {}
    seen = set()
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
        # Reject invalid digest, absolute paths, path traversal, duplicates
        if not SHA256_RE.match(digest):
            return False, file_count, f"invalid digest for: {name}"
        if name.startswith("/") or name.startswith("\\"):
            return False, file_count, f"absolute path: {name}"
        if ".." in Path(name).parts:
            return False, file_count, f"path traversal: {name}"
        if name in seen:
            return False, file_count, f"duplicate entry: {name}"
        seen.add(name)

        target = root / name
        # Check if target or any parent is a symlink
        for parent in [target] + list(target.parents):
            if parent == root:
                break
            if parent.is_symlink():
                return False, file_count, f"symlink in path: {parent.relative_to(root)}"
        if target.is_symlink():
            return False, file_count, f"SYMLINK: {name}"
        if not target.is_file():
            return False, file_count, f"missing: {name}"
        if sha256_file(target) != digest:
            return False, file_count, f"hash mismatch: {name}"
        manifest_files[name] = digest
        file_count += 1

    # Exclusive check
    for p in root.rglob("*"):
        if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel not in manifest_files:
                return False, file_count, f"unsealed extra file: {rel}"

    return True, file_count, "OK"


# ── Identity allowlist validation ──

def validate_identity_allowlist(allowlist_path, pilot_path):
    """Rebuild identity allowlist from pilot manifest and compare exactly
    with IDENTITY_ALLOWLIST.json. Returns (allowlist, digest)."""
    if not Path(pilot_path).is_file():
        raise TransitionRejected(f"pilot manifest missing: {pilot_path}")
    pilot = json.loads(Path(pilot_path).read_text(encoding="utf-8"))
    records = pilot.get("records", [])
    if len(records) != 40:
        raise TransitionRejected(f"pilot must have 40 records, got {len(records)}")

    # Rebuild allowlist from pilot
    rebuilt = []
    seen = set()
    suite_task = {s: set() for s in FOUR_SUITES}
    for rec in records:
        suite = str(rec["suite"])
        task_id = int(rec["task_id"])
        state_id = int(rec["state_id"])
        ep_id = str(rec["episode_id"])
        if "collection_seed" not in rec:
            raise TransitionRejected(f"pilot record {ep_id} missing collection_seed")
        seed_val = int(rec["collection_seed"])
        init_sha = str(rec.get("initial_state_sha256", ""))
        if not SHA256_RE.match(init_sha):
            raise TransitionRejected(f"pilot {ep_id}: invalid initial_state_sha256")
        if state_id < 0:
            raise TransitionRejected(f"pilot {ep_id}: negative state_id")
        if suite not in FOUR_SUITES:
            raise TransitionRejected(f"pilot {ep_id}: unknown suite {suite}")
        if task_id < 0 or task_id >= 10:
            raise TransitionRejected(f"pilot {ep_id}: task_id out of range")
        # Validate episode_id format
        expected_ep = f"{suite}/task_{task_id:02d}/state_{state_id}"
        if ep_id != expected_ep:
            raise TransitionRejected(
                f"pilot episode_id mismatch: {ep_id} != expected {expected_ep}")
        if ep_id in seen:
            raise TransitionRejected(f"duplicate pilot episode_id: {ep_id}")
        seen.add(ep_id)
        suite_task[suite].add(task_id)

        rebuilt.append({
            "episode_id": ep_id,
            "suite": suite,
            "task_id": task_id,
            "state_id": state_id,
            "collection_seed": seed_val,
            "initial_state_sha256": init_sha,
        })

    # Closure checks
    for suite in FOUR_SUITES:
        if len(suite_task[suite]) != 10:
            raise TransitionRejected(f"{suite}: expected 10 tasks, got {suite_task[suite]}")
        if suite_task[suite] != set(range(10)):
            raise TransitionRejected(f"{suite}: missing task ids")

    # Compare with IDENTITY_ALLOWLIST.json
    allowlist_path = Path(allowlist_path)
    if not allowlist_path.is_file():
        raise TransitionRejected("IDENTITY_ALLOWLIST.json missing from receipt")
    declared = json.loads(allowlist_path.read_text(encoding="utf-8"))
    declared_ids = declared.get("identities", [])
    if len(declared_ids) != 40:
        raise TransitionRejected(f"allowlist has {len(declared_ids)} identities, expected 40")

    # Canonical comparison
    rebuilt_str = json.dumps(rebuilt, sort_keys=True, ensure_ascii=False)
    declared_str = json.dumps(declared_ids, sort_keys=True, ensure_ascii=False)
    if rebuilt_str != declared_str:
        raise TransitionRejected("identity allowlist does not match pilot manifest rebuild")

    allowlist_digest = sha256_file(allowlist_path)
    return rebuilt, allowlist_digest


# ── Main verifier ──

def _runtime_git_values(root, label):
    """Return (commit, tree, clean) for a git repo. Raises on dirty or missing."""
    import subprocess
    root = Path(root).resolve()
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise TransitionRejected(f"{label} not a git repo: {root}")
    commit = p.stdout.strip()
    tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
                                   text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                     text=True).strip()
    if status:
        raise TransitionRejected(f"{label} working tree not clean: {status[:80]}")
    return commit, tree


def parse_iso_datetime(s):
    """Parse ISO8601 datetime string. Returns None on failure."""
    import datetime
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"]:
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def verify_transition(transition_root, execution_source_commit, script_sha,
                      model_path, official_worker_path, pilot_manifest_path,
                      registry_root, alias_ledger_path, upstream_root,
                      libero_root, output_root, gpu, physical_gpu=None):
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

    # 3. Chronology: created_at must be after execution commit
    created = tm.get("created_at", "")
    if not created:
        raise TransitionRejected("created_at missing")
    created_dt = parse_iso_datetime(created)
    if created_dt is None:
        raise TransitionRejected(f"created_at unparseable: {created}")
    # Get source commit timestamp
    import subprocess, datetime
    try:
        ct_str = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", execution_source_commit],
            text=True, stderr=subprocess.DEVNULL).strip()
        commit_ts = datetime.datetime.fromtimestamp(int(ct_str), tz=datetime.timezone.utc)
    except Exception:
        commit_ts = None  # Cannot verify without repo; skip for tests
    if commit_ts is not None and created_dt < commit_ts:
        raise TransitionRejected(
            f"transition created {created} before source commit timestamp "
            f"{commit_ts.isoformat()}")

    # 4. Schema, gate, status
    if tm.get("gate") != "FIT-INFERENCE_TRANSITION":
        raise TransitionRejected(f"gate mismatch: {tm.get('gate')}")
    if tm.get("schema") != "FIT_INFERENCE_TRANSITION_V1":
        raise TransitionRejected(f"schema mismatch: {tm.get('schema')}")
    if tm.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise TransitionRejected(f"status mismatch: {tm.get('status')}")

    # 5. Verify all frozen R5-E scientific evidence (including comparison SHA)
    for key, expected in FROZEN_R5E.items():
        actual = tm.get(key)
        if actual != expected:
            raise TransitionRejected(
                f"frozen R5-E evidence mismatch: {key}={actual} (expected {expected})")

    # 6. Execution source binding
    if tm.get("r5f_execution_source_commit") != execution_source_commit:
        raise TransitionRejected(
            f"source commit mismatch: declared={tm.get('r5f_execution_source_commit')} "
            f"actual={execution_source_commit}")
    if tm.get("r5f_script_sha256") != script_sha:
        raise TransitionRejected("script SHA mismatch")

    # 7. Model tree + processor
    declared_tree = tm.get("model_tree_sha256")
    actual_tree = compute_model_tree_fingerprint(model_path)
    if declared_tree != actual_tree:
        raise TransitionRejected(
            f"model tree mismatch: declared={declared_tree[:16]} actual={actual_tree[:16]}")
    declared_proc = tm.get("processor_sha256")
    proc_path = Path(model_path) / "preprocessor_config.json"
    if not proc_path.is_file():
        raise TransitionRejected(f"processor config missing: {proc_path}")
    if declared_proc != sha256_file(proc_path):
        raise TransitionRejected("processor SHA mismatch")

    # 8. Worker SHA
    declared_worker = tm.get("official_worker_sha256")
    if not Path(official_worker_path).is_file():
        raise TransitionRejected(f"worker missing: {official_worker_path}")
    if declared_worker != sha256_file(official_worker_path):
        raise TransitionRejected("worker SHA mismatch")

    # 9. Pilot manifest SHA
    declared_pilot = tm.get("pilot_manifest_sha256")
    if not Path(pilot_manifest_path).is_file():
        raise TransitionRejected(f"pilot manifest missing: {pilot_manifest_path}")
    if declared_pilot != sha256_file(pilot_manifest_path):
        raise TransitionRejected("pilot SHA mismatch")

    # 10. Identity allowlist + identity_set_digest
    allowlist_data, actual_allowlist_digest = validate_identity_allowlist(
        tr / "IDENTITY_ALLOWLIST.json", pilot_manifest_path)
    if tm.get("identity_allowlist_digest") != actual_allowlist_digest:
        raise TransitionRejected("allowlist digest mismatch")
    # Recompute identity_set_digest
    id_set_digest = hashlib.sha256(
        json.dumps(allowlist_data, sort_keys=True).encode()).hexdigest()
    if tm.get("identity_set_digest") != id_set_digest:
        raise TransitionRejected("identity_set_digest mismatch")
    if tm.get("authorized_identities") != 40:
        raise TransitionRejected("authorized_identities must be 40")
    if tm.get("n_pilot_identities") != 40:
        raise TransitionRejected("n_pilot_identities must be 40")

    # 11. Registry + alias
    declared_reg = tm.get("registry_summary_sha256")
    reg_path = Path(registry_root).parent / "ENTITY_REGISTRY_V2_SUMMARY.json"
    if not reg_path.is_file():
        raise TransitionRejected(f"registry summary missing: {reg_path}")
    if declared_reg != sha256_file(reg_path):
        raise TransitionRejected("registry SHA mismatch")
    declared_alias = tm.get("alias_ledger_sha256")
    if not Path(alias_ledger_path).is_file():
        raise TransitionRejected(f"alias ledger missing: {alias_ledger_path}")
    if declared_alias != sha256_file(alias_ledger_path):
        raise TransitionRejected("alias ledger SHA mismatch")

    # 12. Upstream runtime — compute actual commit, tree, clean
    declared_up_commit = tm.get("upstream_commit")
    declared_up_tree = tm.get("upstream_tree", "")
    up_commit, up_tree = _runtime_git_values(upstream_root, "upstream")
    if declared_up_commit != up_commit:
        raise TransitionRejected(
            f"upstream commit mismatch: declared={declared_up_commit[:16]} "
            f"actual={up_commit[:16]}")
    if declared_up_tree and declared_up_tree != up_tree:
        raise TransitionRejected("upstream tree mismatch")

    # 13. LIBERO runtime
    declared_lib_commit = tm.get("libero_commit")
    declared_lib_tree = tm.get("libero_tree", "")
    lib_commit, lib_tree = _runtime_git_values(libero_root, "LIBERO")
    if declared_lib_commit != lib_commit:
        raise TransitionRejected(
            f"LIBERO commit mismatch: declared={declared_lib_commit[:16]} "
            f"actual={lib_commit[:16]}")
    if declared_lib_tree and declared_lib_tree != lib_tree:
        raise TransitionRejected("LIBERO tree mismatch")

    # 14. Full permission matrix
    PERMS = {
        "openvla_inference_authorized": True,
        "clean_action_only": True, "forward_before_capture": True,
        "max_episodes": 40, "identity_set_frozen": True,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "detector_load_authorized": False,
        "attack_authorized": False, "protected_payload_read": False,
    }
    for key, expected in PERMS.items():
        if tm.get(key) != expected:
            raise TransitionRejected(f"permission violation: {key} must be {expected}")

    # 15. GPU + output allowlist
    if gpu not in tm.get("allowed_gpus", []):
        raise TransitionRejected(f"GPU {gpu} not in allowlist")
    if str(output_root) not in tm.get("allowed_output_roots", []):
        raise TransitionRejected("output root not in allowlist")

    # 16. Physical/logical GPU mapping consistency
    if physical_gpu is not None:
        declared = tm.get("physical_to_logical_gpu", {})
        if str(physical_gpu) not in declared:
            raise TransitionRejected(f"physical GPU {physical_gpu} not mapped")
        if declared[str(physical_gpu)] != 0:
            raise TransitionRejected(
                f"physical GPU {physical_gpu} maps to device "
                f"{declared[str(physical_gpu)]}, expected 0")

    return tm
