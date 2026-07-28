"""[DeepSeek] FIT670 Transition Receipt Builder — Gate F670-E.

Generates a sealed FIT670 transition receipt (FIT670_INFERENCE_TRANSITION_V1)
that validates 670 identities across 6 (or 8) GPUs.

Usage (server):
  python n5/phase2_labels/build_fit670_transition.py \
    --identity-allowlist /path/to/FIT670_IDENTITY_ALLOWLIST.json \
    --shard-plan /path/to/FIT670_GPU_SHARD_PLAN.json \
    --model-path /path/to/openvla-checkpoint \
    --official-worker /path/to/official_clean_worker.py \
    --registry-summary /path/to/ENTITY_REGISTRY_V2_SUMMARY.json \
    --alias-ledger /path/to/ALIAS_LEDGER.json \
    --upstream-root /path/to/openvla-upstream \
    --libero-root /path/to/libero \
    --allowed-output-root /path/to/output \
    --out /path/to/transition_root
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, uuid, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_transition import (
    compute_model_tree_fingerprint, sha256_file,
    FOUR_SUITES, FROZEN_R5E, verify_r5e_comparison_root,
)

SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--allowed-output-root", required=True)
    parser.add_argument("--r5e-comparison-root", type=Path, required=True)
    parser.add_argument("--n-gpus", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Verify input existence
    allowlist_path = Path(args.identity_allowlist).resolve()
    shard_plan_path = Path(args.shard_plan).resolve()
    if not allowlist_path.is_file():
        raise SystemExit(f"allowlist missing: {allowlist_path}")
    if not shard_plan_path.is_file():
        raise SystemExit(f"shard plan missing: {shard_plan_path}")

    # Load allowlist and verify
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    identities = allowlist.get("identities", [])
    if len(identities) != 670:
        raise SystemExit(f"expected 670 identities, got {len(identities)}")
    id_set_digest = allowlist.get("identity_set_digest", "")
    if not SHA256_RE.match(id_set_digest):
        raise SystemExit(f"invalid identity_set_digest: {id_set_digest[:20]}")

    # Load shard plan
    shard_plan = json.loads(shard_plan_path.read_text(encoding="utf-8"))
    n_shards = shard_plan.get("n_shards", args.n_gpus)
    if n_shards not in (6, 8):
        raise SystemExit(f"shard plan has {n_shards} shards, expected 6 or 8")

    # Verify R5-E comparison root (frozen evidence)
    comp_root = Path(args.r5e_comparison_root).resolve()
    ok, _, issues = verify_r5e_comparison_root(comp_root)
    if not ok:
        raise SystemExit(f"R5-E comparison root verification failed: {issues}")
    actual_comp_sha = sha256_file(comp_root / "SHA256SUMS")
    expected_comp_sha = FROZEN_R5E.get("r5e_comparison_sha256", "")
    if actual_comp_sha != expected_comp_sha:
        raise SystemExit(
            f"R5-E comparison SHA mismatch: {actual_comp_sha} vs {expected_comp_sha}")

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # ── Compute input SHAs ──
    model_tree = compute_model_tree_fingerprint(args.model_path)
    worker_sha = sha256_file(args.official_worker)
    registry_sha = sha256_file(args.registry_summary)
    alias_sha = sha256_file(args.alias_ledger)
    processor_sha = sha256_file(args.model_path / "preprocessor_config.json")
    allowlist_file_sha = sha256_file(allowlist_path)
    shard_plan_sha = sha256_file(shard_plan_path)
    upstream_commit = git_value(args.upstream_root, "rev-parse", "HEAD")
    upstream_tree = git_value(args.upstream_root, "rev-parse", "HEAD^{tree}")
    libero_commit = git_value(args.libero_root, "rev-parse", "HEAD")
    libero_tree = git_value(args.libero_root, "rev-parse", "HEAD^{tree}")

    # ── GPU allowlist ──
    allowed_physical_gpus = list(range(n_shards))
    physical_to_logical = {str(g): 0 for g in allowed_physical_gpus}

    # ── Build manifest ──
    manifest = {
        "gate": "FIT670-INFERENCE_TRANSITION",
        "schema": "FIT670_INFERENCE_TRANSITION_V1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_BEFORE_EXECUTION",
        "consumer_eligible": False,

        # Frozen R5-E evidence
        **FROZEN_R5E,

        # FIT670-specific
        "identity_pool": "D0-R2_DEV_POOL_670",
        "protected_overlap_verified": 0,
        "cross_gpu_trajectory_parity_guaranteed": False,
        "sdpa_bf16_non_determinism_documented": True,

        # Model
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": model_tree,
        "processor_sha256": processor_sha,

        # Worker
        "official_worker_path": str(Path(args.official_worker).resolve()),
        "official_worker_sha256": worker_sha,

        # Identity
        "identity_allowlist_path": str(allowlist_path),
        "identity_allowlist_file_sha256": allowlist_file_sha,
        "identity_allowlist_digest": allowlist_file_sha,
        "identity_set_digest": id_set_digest,
        "n_pilot_identities": 670,
        "authorized_identities": 670,

        # Shard plan
        "shard_plan_path": str(shard_plan_path),
        "shard_plan_sha256": shard_plan_sha,
        "n_shards": n_shards,
        "n_gpus": n_shards,

        # Registry
        "registry_summary_sha256": registry_sha,
        "alias_ledger_sha256": alias_sha,

        # Runtime
        "upstream_commit": upstream_commit,
        "upstream_tree": upstream_tree,
        "libero_commit": libero_commit,
        "libero_tree": libero_tree,

        # Permissions
        "allowed_physical_gpus": allowed_physical_gpus,
        "allowed_gpus": [0],
        "physical_to_logical_gpu": physical_to_logical,
        "allowed_output_roots": [args.allowed_output_root],
        "openvla_inference_authorized": True,
        "clean_action_only": True,
        "forward_before_capture": True,
        "max_episodes": 670,
        "identity_set_frozen": True,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "detector_load_authorized": False,
        "attack_authorized": False,
        "protected_payload_read": False,
    }

    # ── Seal ──
    staging = out.parent / f".{out.name}.transition_staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    published = False
    try:
        (staging / "TRANSITION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        seal_output(staging)
        staging.rename(out)
        published = True

        print(f"FIT670 transition sealed: {out}")
        print(f"  identities: 670")
        print(f"  shards: {n_shards}")
        print(f"  allowed GPUs: {allowed_physical_gpus}")
        print(f"  identity_set_digest: {id_set_digest}")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def seal_output(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return sums_sha


if __name__ == "__main__":
    main()
