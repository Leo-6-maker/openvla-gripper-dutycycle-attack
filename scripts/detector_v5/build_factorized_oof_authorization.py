#!/usr/bin/env python3
"""Build formal Factorized Student OOF authorization (sealed root).

Verifies: Teacher/S1/F3/fold/policy-intent root seals, F3 status,
          Student protocol authority, exact Git HEAD.
Binds: all source SHA, model/loss/dataset/trainer/launcher.
Output: sealed authorization root.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--s1-root", type=Path, required=True)
    ap.add_argument("--f3-root", type=Path, required=True)
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--policy-intent-root", type=Path, required=True)
    args = ap.parse_args()

    out = args.output_root.resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    # Verify all input seals
    teacher_seal = verify_seal(args.teacher_root.resolve())
    s1_seal = verify_seal(args.s1_root.resolve())
    f3_seal = verify_seal(args.f3_root.resolve())
    fold_seal = verify_seal(args.fold_root.resolve())
    pi_seal = verify_seal(args.policy_intent_root.resolve()) if args.policy_intent_root else None

    # F3 status
    f3_audit = json.loads((args.f3_root / "geometry_audit.json").read_text())
    if f3_audit.get("status") != "PASS_FINAL_STUDENT_TRAINING":
        raise SystemExit(f"F3 status is {f3_audit.get('status')}, must be PASS_FINAL_STUDENT_TRAINING")

    # Student protocol still frozen
    proto = json.loads((ROOT / "configs/DETECTOR_V5_FACTORIZED_STUDENT_PROTOCOL_V1.json").read_text())
    if proto.get("formal_training_authorized") is not False:
        raise SystemExit("Student protocol already authorizes training")
    proto_sha = sha256_file(ROOT / "configs/DETECTOR_V5_FACTORIZED_STUDENT_PROTOCOL_V1.json")

    # Git HEAD + clean tree
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    if len(head) != 40:
        raise SystemExit(f"invalid git HEAD: {head}")
    # Only check tracked files — untracked dirs like envs/ are fine
    dirty = subprocess.check_output(
        ["git", "diff", "--stat", "HEAD"], cwd=str(ROOT), text=True
    ).strip()
    if dirty:
        raise SystemExit(f"tracked files must be unmodified for formal authorization:\n{dirty[:500]}")

    # Source SHAs
    src_dir = ROOT / "src/gripper_attack"
    script_dir = ROOT / "scripts/detector_v5"
    shas = {
        "dataset_sha": sha256_file(src_dir / "v5_factorized_dataset.py"),
        "model_sha": sha256_file(src_dir / "v5_factorized_student.py"),
        "loss_sha": sha256_file(src_dir / "v5_factorized_loss.py"),
        "trainer_sha": sha256_file(script_dir / "train_factorized_oof.py"),
        "launcher_sha": sha256_file(script_dir / "launch_factorized_oof.py"),
    }

    auth = {
        "schema": "DETECTOR_V5_FACTORIZED_OOF_AUTHORIZATION_V1",
        "status": "FORMAL_OOF_TRAINING_AUTHORIZED",
        "formal_oof_training_authorized": True,
        "full_fit_authorized": False,
        "cal_authorized": False,
        "check_authorized": False,
        "attack_authorized": False,
        "source_commit": head,
        "teacher_root": str(args.teacher_root.resolve()),
        "teacher_root_seal": teacher_seal,
        "s1_root": str(args.s1_root.resolve()),
        "s1_root_seal": s1_seal,
        "f3_root": str(args.f3_root.resolve()),
        "f3_root_seal": f3_seal,
        "fold_root": str(args.fold_root.resolve()),
        "fold_root_seal": fold_seal,
        "policy_intent_root": str(args.policy_intent_root.resolve()) if args.policy_intent_root else None,
        "policy_intent_root_seal": pi_seal,
        "student_protocol_schema": proto["schema"],
        "student_protocol_sha": proto_sha,
        **shas,
        "model_type_primary": "FACTORIZED_ROUTED_25D9D",
        "model_type_ablation": "FACTORIZED_ROUTED_25D",
        "training_config": {
            "folds": 4, "seeds": [42, 123, 456], "epochs": 30,
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 1e-5, "grad_clip": 5.0,
            "batch_size": 8, "dtype": "float32", "early_stopping": False,
            "checkpoint_epoch": 30, "normalization": "train_fold_only",
        },
    }

    staging = out.with_name(f".{out.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    _atomic_text(staging / "authorization.json", json.dumps(auth, indent=2, sort_keys=True) + "\n")
    write_seal(staging)
    os.replace(staging, out)
    print(json.dumps({"status": "AUTHORIZATION_SEALED", "commit": head, "root": str(out)}, indent=2))

if __name__ == "__main__":
    main()
