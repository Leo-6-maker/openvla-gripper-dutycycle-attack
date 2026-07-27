"""[DeepSeek] FIT-INFERENCE Transition Receipt Builder (v2).

Generates a sealed transition receipt that R5-F --transition-receipt validates.
Must be run AFTER R5-F execution source S is committed.
Uses shared compute_model_tree_fingerprint from fit_transition.

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
    --upstream-root /path/to/openvla-upstream \
    --libero-root /path/to/libero \
    --allowed-gpus 6,7 \
    --allowed-output-root /path/to/r5f_output \
    --r5e-comparison-sha <sha>
"""
import argparse, hashlib, json, os, sys, time, uuid, shutil, re, subprocess
from pathlib import Path

# Import shared fingerprint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_transition import (
    compute_model_tree_fingerprint, sha256_file,
    FOUR_SUITES, FROZEN_R5E,
)

SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def git_value(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True).strip()


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
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--allowed-gpus", required=True)
    parser.add_argument("--allowed-output-root", required=True)
    parser.add_argument("--r5e-comparison-sha", required=True)
    args = parser.parse_args()

    # Validate comparison SHA
    if not SHA256_RE.match(args.r5e_comparison_sha):
        raise SystemExit(f"r5e-comparison-sha invalid: {args.r5e_comparison_sha[:20]}")
    if not SHA256_RE.match(args.r5f_source_commit):
        raise SystemExit(f"r5f-source-commit invalid: {args.r5f_source_commit[:20]}")
    if not SHA256_RE.match(args.r5f_script_sha):
        raise SystemExit(f"r5f-script-sha invalid: {args.r5f_script_sha[:20]}")

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # ── Compute input SHAs ──
    model_tree = compute_model_tree_fingerprint(args.model_path)
    worker_sha = sha256_file(args.official_worker)
    pilot_sha = sha256_file(args.pilot_manifest)
    registry_sha = sha256_file(args.registry_summary)
    alias_sha = sha256_file(args.alias_ledger)
    processor_sha = sha256_file(args.model_path / "preprocessor_config.json")
    upstream_commit = git_value(args.upstream_root, "rev-parse", "HEAD")
    libero_fingerprint = compute_model_tree_fingerprint(args.libero_root)

    # ── Build identity allowlist from pilot ──
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    records = pilot.get("records", [])
    if len(records) != 40:
        raise SystemExit(f"pilot must have 40 records, got {len(records)}")

    identity_allowlist = []
    seen = set()
    suite_task = {s: set() for s in FOUR_SUITES}
    for rec in records:
        suite = str(rec["suite"])
        task_id = int(rec["task_id"])
        state_id = int(rec["state_id"])
        ep_id = str(rec["episode_id"])
        if "collection_seed" not in rec:
            raise SystemExit(f"pilot record {ep_id} missing collection_seed")
        seed_val = int(rec["collection_seed"])
        init_sha = str(rec.get("initial_state_sha256", ""))
        if not SHA256_RE.match(init_sha):
            raise SystemExit(f"pilot {ep_id}: invalid initial_state_sha256")
        if state_id < 0:
            raise SystemExit(f"pilot {ep_id}: negative state_id")
        expected_ep = f"{suite}/task_{task_id:02d}/state_{state_id}"
        if ep_id != expected_ep:
            raise SystemExit(f"pilot episode_id mismatch: {ep_id} != {expected_ep}")
        if ep_id in seen:
            raise SystemExit(f"duplicate pilot episode_id: {ep_id}")
        seen.add(ep_id)
        suite_task[suite].add(task_id)
        identity_allowlist.append({
            "episode_id": ep_id, "suite": suite,
            "task_id": task_id, "state_id": state_id,
            "collection_seed": seed_val, "initial_state_sha256": init_sha,
        })

    for suite in FOUR_SUITES:
        if len(suite_task[suite]) != 10:
            raise SystemExit(f"{suite}: expected 10 tasks, got {suite_task[suite]}")
        if suite_task[suite] != set(range(10)):
            raise SystemExit(f"{suite}: missing task ids")

    identity_set_digest = hashlib.sha256(
        json.dumps(identity_allowlist, sort_keys=True).encode()).hexdigest()

    # ── Build manifest ──
    allowed_gpus = [int(x.strip()) for x in args.allowed_gpus.split(",")]
    manifest = {
        "gate": "FIT-INFERENCE_TRANSITION",
        "schema": "FIT_INFERENCE_TRANSITION_V1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_BEFORE_EXECUTION",
        "consumer_eligible": False,

        # Frozen R5-E evidence
        **FROZEN_R5E,
        "r5e_comparison_sha256": args.r5e_comparison_sha,

        # Execution source
        "r5f_execution_source_commit": args.r5f_source_commit,
        "r5f_script_sha256": args.r5f_script_sha,

        # Model
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": model_tree,
        "processor_sha256": processor_sha,

        # Worker
        "official_worker_path": str(Path(args.official_worker).resolve()),
        "official_worker_sha256": worker_sha,

        # Pilot
        "pilot_manifest_path": str(Path(args.pilot_manifest).resolve()),
        "pilot_manifest_sha256": pilot_sha,
        "n_pilot_identities": 40,

        # Registry
        "registry_summary_sha256": registry_sha,
        "alias_ledger_sha256": alias_sha,

        # Runtime
        "upstream_commit": upstream_commit,
        "libero_fingerprint": libero_fingerprint,

        # Identity
        "identity_allowlist_digest": "",  # filled after writing
        "identity_set_digest": identity_set_digest,

        # Permissions
        "authorized_identities": 40,
        "allowed_gpus": allowed_gpus,
        "allowed_output_roots": [args.allowed_output_root],
        "openvla_inference_authorized": True,
        "clean_action_only": True,
        "forward_before_capture": True,
        "max_episodes": 40,
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

    allowlist_path = staging / "IDENTITY_ALLOWLIST.json"
    allowlist_path.write_text(json.dumps({
        "gate": "FIT-INFERENCE_IDENTITY_ALLOWLIST",
        "n_identities": 40,
        "identity_set_digest": identity_set_digest,
        "identities": identity_allowlist,
    }, indent=2, sort_keys=True))

    manifest["identity_allowlist_digest"] = sha256_file(allowlist_path)
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
    print(f"  Allowlist digest: {manifest['identity_allowlist_digest']}")
    print(f"  Source commit: {args.r5f_source_commit}")


if __name__ == "__main__":
    main()
