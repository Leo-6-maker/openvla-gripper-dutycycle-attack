#!/usr/bin/env python3
"""Build formal Factorized Student OOF Evaluation authorization (sealed root).

Binds: OOF training root, provenance reconciliation, eval protocol,
        prediction/evaluation/audit runners, source SHAs, checkpoint inventory.
"""
import argparse, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    with t.open("x") as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def verify_seal(root):
    s = root / "SHA256SUMS"; c = root / "SHA256SUMS.sha256"
    if not s.is_file() or not c.is_file():
        raise SystemExit(f"SEAL MISSING: {root}")
    if c.read_text().strip() != f"{sha256_file(s)}  SHA256SUMS":
        raise SystemExit(f"SEAL MISMATCH: {root}")
    for l in s.read_text().splitlines():
        d, _, n = l.partition("  "); t = root / n
        if not t.is_file() or sha256_file(t) != d:
            raise SystemExit(f"FILE MISMATCH: {root}/{n}")
    return sha256_file(s)


def write_seal(root):
    excl = {"SHA256SUMS", "SHA256SUMS.sha256"}
    fs = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in fs)
    _atomic_text(root / "SHA256SUMS", c)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


SEEDS = [42, 123, 456]
FOLD_IDS = [0, 1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--oof-training-root", type=Path, required=True,
                    help="OFFICIAL_V3_FACTORIZED_STUDENT_OOF_335048c_20260721")
    ap.add_argument("--provenance-recon-root", type=Path, required=True,
                    help="OFFICIAL_V3_FACTORIZED_STUDENT_OOF_PROVENANCE_RECONCILIATION_V1_20260721")
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--policy-intent-root", type=Path, default=None)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # Verify all input seals
    training_seal = verify_seal(args.oof_training_root.resolve())
    recon_seal = verify_seal(args.provenance_recon_root.resolve())
    teacher_seal = verify_seal(args.teacher_root.resolve())
    s1_seal = verify_seal(args.s1_root.resolve())
    fold_seal = verify_seal(args.fold_root.resolve())
    pi_seal = verify_seal(args.policy_intent_root.resolve()) if args.policy_intent_root else None

    # Verify reconciliation status
    recon = json.loads((args.provenance_recon_root / "reconciliation.json").read_text())
    if recon.get("status") != "PASS_FILESET_BOUND_FORMAL_OOF":
        raise SystemExit(f"Reconciliation status is {recon.get('status')}, not PASS_FILESET_BOUND_FORMAL_OOF")

    # Verify training authorization
    training_auth = json.loads((args.oof_training_root / "25D9D/fold0_seed42/authorization_receipt.json").read_text())

    # Evaluation protocol SHA
    eval_proto_path = ROOT / "configs/DETECTOR_V5_FACTORIZED_OOF_EVAL_PROTOCOL_V1.json"
    eval_proto = json.loads(eval_proto_path.read_text())
    if eval_proto.get("status") != "FROZEN_BEFORE_ANY_PREDICTION":
        raise SystemExit("Eval protocol status must be FROZEN_BEFORE_ANY_PREDICTION")
    eval_proto_sha = sha256_file(eval_proto_path)

    # Source SHAs
    src_dir = ROOT / "src/gripper_attack"
    script_dir = ROOT / "scripts/detector_v5"
    shas = {
        "dataset_sha": sha256_file(src_dir / "v5_factorized_dataset.py"),
        "model_sha": sha256_file(src_dir / "v5_factorized_student.py"),
        "loss_sha": sha256_file(src_dir / "v5_factorized_loss.py"),
        "predict_runner_sha": sha256_file(script_dir / "predict_factorized_oof.py"),
        "evaluate_runner_sha": sha256_file(script_dir / "evaluate_factorized_oof.py"),
        "audit_runner_sha": sha256_file(script_dir / "audit_factorized_oof_predictions.py"),
        "eval_protocol_sha": eval_proto_sha,
    }

    # Checkpoint inventory (verify all 24 exist and are sealed)
    checkpoint_inventory = []
    for mt in ["25D9D", "25D"]:
        for fold in FOLD_IDS:
            for seed in SEEDS:
                ckpt_dir = args.oof_training_root / mt / f"fold{fold}_seed{seed}"
                seal = verify_seal(ckpt_dir)
                checkpoint_inventory.append({
                    "model_type": mt, "fold_id": fold, "seed": seed,
                    "dir": str(ckpt_dir),
                    "seal": seal,
                })

    # Git HEAD
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    if len(head) != 40:
        raise SystemExit(f"invalid git HEAD: {head}")

    auth = {
        "schema": "DETECTOR_V5_FACTORIZED_OOF_EVAL_AUTHORIZATION_V1",
        "status": "FORMAL_OOF_EVALUATION_AUTHORIZED",
        "timestamp": "2026-07-21",
        "source_commit": head,
        "oof_training_root": str(args.oof_training_root.resolve()),
        "oof_training_root_seal": training_seal,
        "provenance_reconciliation_root": str(args.provenance_recon_root.resolve()),
        "provenance_reconciliation_seal": recon_seal,
        "teacher_root": str(args.teacher_root.resolve()),
        "teacher_root_seal": teacher_seal,
        "s1_root": str(args.s1_root.resolve()),
        "s1_root_seal": s1_seal,
        "fold_root": str(args.fold_root.resolve()),
        "fold_root_seal": fold_seal,
        "policy_intent_root": str(args.policy_intent_root.resolve()) if args.policy_intent_root else None,
        "policy_intent_root_seal": pi_seal,
        "eval_protocol_schema": eval_proto["schema"],
        **shas,
        "checkpoint_inventory": {
            "count": len(checkpoint_inventory),
            "checkpoints": checkpoint_inventory,
        },
        "hard_constraints": eval_proto["hard_constraints"],
        "gate_thresholds": eval_proto["oof_gate_thresholds"],
    }

    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    _atomic_text(staging / "authorization.json", json.dumps(auth, indent=2, sort_keys=True) + "\n")
    write_seal(staging)
    os.replace(staging, out)
    print(json.dumps({"status": "EVAL_AUTHORIZATION_SEALED", "commit": head, "root": str(out),
                       "checkpoints": len(checkpoint_inventory)}, indent=2))


if __name__ == "__main__":
    main()
