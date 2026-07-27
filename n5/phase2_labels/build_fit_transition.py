"""[DeepSeek] FIT-INFERENCE Transition Receipt Builder.

Generates a sealed transition receipt that R5-F --transition-receipt validates.
Must be run AFTER R5-F execution source S is committed.
Binds all scientific evidence, model/worker/pilot SHAs, identity allowlist,
and permission boundaries.

Usage:
  python n5/phase2_labels/build_fit_transition.py \
    --out /path/to/transition_root \
    --r5f-source-commit <S> \
    --r5f-script-sha <sha> \
    --model-path /path/to/openvla-checkpoint \
    --official-worker /path/to/official_worker.py \
    --pilot-manifest /path/to/pilot_manifest.json \
    --registry-summary /path/to/c1_v2_r7/run_A/ENTITY_REGISTRY_V2_SUMMARY.json \
    --alias-ledger /path/to/c1_v2_r7/run_A/ALIAS_LEDGER.json \
    --allowed-gpus 6,7 \
    --allowed-output-root /path/to/r5f_output \
    --r5e-comparison-sha <sha>
"""
import argparse, hashlib, json, os, sys, time, uuid, shutil
from pathlib import Path
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# Frozen scientific evidence from R5-E-R1 (NOT configurable via CLI)
FROZEN_EVIDENCE = {
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
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r5f-source-commit", required=True)
    parser.add_argument("--r5f-script-sha", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--registry-summary", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--allowed-gpus", required=True)
    parser.add_argument("--allowed-output-root", required=True)
    parser.add_argument("--r5e-comparison-sha", default="")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # ── Compute input SHAs ──
    model_tree = _model_tree_sha(args.model_path)
    worker_sha = sha256_file(args.official_worker)
    pilot_sha = sha256_file(args.pilot_manifest)
    registry_sha = sha256_file(args.registry_summary)
    alias_sha = sha256_file(args.alias_ledger)
    processor_sha = sha256_file(args.model_path / "preprocessor_config.json")

    # ── Load pilot identities for allowlist ──
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    records = pilot.get("records", [])
    if len(records) != 40:
        raise SystemExit(f"pilot must have 40 records, got {len(records)}")

    identity_allowlist = []
    seen = set()
    for rec in records:
        suite = str(rec["suite"])
        task_id = int(rec["task_id"])
        state_id = int(rec["state_id"])
        ep_id = str(rec["episode_id"])
        seed_val = int(rec.get("collection_seed", 0))
        init_sha = str(rec.get("initial_state_sha256", ""))
        if ep_id in seen:
            raise SystemExit(f"duplicate episode_id: {ep_id}")
        seen.add(ep_id)
        identity_allowlist.append({
            "episode_id": ep_id,
            "suite": suite,
            "task_id": task_id,
            "state_id": state_id,
            "collection_seed": seed_val,
            "initial_state_sha256": init_sha,
        })

    # Verify closure
    suite_counts = {}
    for ident in identity_allowlist:
        suite_counts[ident["suite"]] = suite_counts.get(ident["suite"], 0) + 1
    for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        if suite_counts.get(suite, 0) != 10:
            raise SystemExit(f"{suite}: expected 10, got {suite_counts.get(suite, 0)}")

    # ── Build transition manifest ──
    allowed_gpus = [int(x.strip()) for x in args.allowed_gpus.split(",")]
    manifest = {
        "gate": "FIT-INFERENCE_TRANSITION",
        "schema": "FIT_INFERENCE_TRANSITION_V1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_BEFORE_EXECUTION",
        "consumer_eligible": False,

        # Scientific evidence
        **FROZEN_EVIDENCE,
        "r5e_comparison_sha256": args.r5e_comparison_sha,

        # Execution source binding
        "r5f_execution_source_commit": args.r5f_source_commit,
        "r5f_script_sha256": args.r5f_script_sha,

        # Model binding
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": model_tree,
        "processor_sha256": processor_sha,

        # Worker binding
        "official_worker_path": str(Path(args.official_worker).resolve()),
        "official_worker_sha256": worker_sha,

        # Pilot binding
        "pilot_manifest_path": str(Path(args.pilot_manifest).resolve()),
        "pilot_manifest_sha256": pilot_sha,
        "n_pilot_identities": len(records),

        # Registry binding
        "registry_summary_sha256": registry_sha,
        "alias_ledger_sha256": alias_sha,

        # Identity allowlist
        "identity_allowlist_digest": "",  # filled after allowlist JSON is written

        # Permission boundaries
        "authorized_identities": len(records),
        "allowed_gpus": allowed_gpus,
        "allowed_output_roots": [args.allowed_output_root],
        "openvla_inference_authorized": True,
        "clean_action_only": True,
        "forward_before_capture": True,
        "teacher_labels_authorized": False,
        "student_training_authorized": False,
        "attack_authorized": False,
        "detector_load_authorized": False,
        "protected_payload_read": False,
        "max_episodes": 40,
        "identity_set_frozen": True,
    }

    # ── Seal ──
    staging = out.parent / f".{out.name}.transition_staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)

    # Write allowlist
    allowlist_path = staging / "IDENTITY_ALLOWLIST.json"
    allowlist_path.write_text(json.dumps({
        "gate": "FIT-INFERENCE_IDENTITY_ALLOWLIST",
        "n_identities": len(identity_allowlist),
        "identity_set_digest": sha256_file.__func__,  # placeholder
        "identities": identity_allowlist,
    }, indent=2, sort_keys=True))

    # Recompute allowlist digest
    allowlist_digest = sha256_file(allowlist_path)
    manifest["identity_allowlist_digest"] = allowlist_digest

    (staging / "TRANSITION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))

    # Seal
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}" for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums)
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n")
    staging.rename(out)

    print(f"Transition sealed: {out}")
    print(f"  SHA256SUMS: {sums_sha}")
    print(f"  Allowlist digest: {allowlist_digest}")
    print(f"  Source commit: {args.r5f_source_commit}")


def _model_tree_sha(model_path):
    """Compute tree SHA of model checkpoint directory."""
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(model_path), "rev-parse", "HEAD^{tree}"],
        capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    # Fallback: hash all files
    files = sorted(Path(model_path).rglob("*"))
    h = hashlib.sha256()
    for fp in files:
        if fp.is_file() and ".git" not in fp.parts:
            h.update(fp.name.encode())
            h.update(sha256_file(fp).encode())
    return h.hexdigest()


if __name__ == "__main__":
    main()
