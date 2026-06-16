#!/usr/bin/env python3
"""Verify D5 v1 production bundle — strictly fail-closed.

Exits 0 only if ALL checks pass with zero warnings.
"""
import argparse, hashlib, json, os, subprocess, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def sha256_file(path):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(cmd, cwd=None):
    try:
        return subprocess.check_output(["git"] + cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def fail(msg, errors):
    print(f"  FAIL: {msg}", file=sys.stderr)
    return errors + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--strict", action="store_true", help="Check worktree clean + branch match")
    args = ap.parse_args()

    bundle = json.load(open(args.bundle))
    errors = 0

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Verifying bundle: {args.bundle}")
    print(f"  version={bundle['bundle_version']}  detector={bundle['detector_version']}")

    # ── 1. Bundle integrity ──
    if bundle["bundle_version"] != "1.0.0":
        errors = fail(f"bundle_version must be 1.0.0, got {bundle['bundle_version']}", errors)
    if bundle["parameters"]["tau"] != 0.050:
        errors = fail(f"tau must be 0.050, got {bundle['parameters']['tau']}", errors)

    # ── 2. All file artifacts MUST exist and match ──
    all_artifacts = {}
    all_artifacts.update(bundle.get("artifacts", {}))
    all_artifacts.update(bundle.get("evidence", {}))

    for name, art in all_artifacts.items():
        local_path = os.path.join(repo_root, art["path"])
        if os.path.exists(local_path):
            actual = sha256_file(local_path)
            if actual != art["sha256"]:
                errors = fail(f"{name}: SHA mismatch (expected {art['sha256'][:16]}... got {actual[:16]}...)", errors)
            else:
                print(f"  OK: {name}")
        else:
            # Try absolute path
            if os.path.exists(art["path"]):
                actual = sha256_file(art["path"])
                if actual != art["sha256"]:
                    errors = fail(f"{name}: remote SHA mismatch", errors)
                else:
                    print(f"  OK: {name} (remote)")
            else:
                errors = fail(f"{name}: file not found at {local_path} or {art['path']}", errors)

    # ── 3. Feature order ──
    expected = bundle["parameters"]["feature_names"]
    if len(expected) != 16:
        errors = fail(f"feature count {len(expected)} != 16", errors)
    else:
        print(f"  OK: feature_names count=16")

    # ── 4. Adapter source commit ──
    if not bundle["parameters"]["adapter_source_commit"].startswith("44bf7b86"):
        errors = fail("adapter_source_commit mismatch", errors)
    else:
        print(f"  OK: adapter_source_commit")

    # ── 5. GPU authorization ──
    for g in bundle["gpu"]["excluded_gpus"]:
        if g in bundle["gpu"].get("authorized_pairs", []):
            errors = fail(f"GPU{g} excluded but in authorized_pairs", errors)
    for g in ["3"]:
        if g not in bundle["gpu"]["quarantined_gpus"]:
            errors = fail(f"GPU{g} must be quarantined", errors)
    for g in ["0", "4"]:
        if g not in bundle["gpu"]["excluded_gpus"]:
            errors = fail(f"GPU{g} must be excluded", errors)
    print(f"  OK: GPU authorization matrix")

    # ── 6. Detector init + manifest consistency ──
    ckpt_info = bundle["artifacts"]["checkpoint"]
    cfg_info = bundle["artifacts"]["config"]
    if os.path.exists(ckpt_info["path"]) and os.path.exists(cfg_info["path"]):
        try:
            from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
            det = D5FrozenOnlineDetectorV1(ckpt_info["path"], cfg_info["path"])
            m = det.bound_manifest
            checks = [
                ("tau", m["tau"], 0.050),
                ("checkpoint_sha", m["checkpoint_sha"], ckpt_info["sha256"]),
                ("config_sha", m["config_sha"], cfg_info["sha256"]),
                ("runtime_sha", m["runtime_sha"], bundle["artifacts"]["runtime_file"]["sha256"]),
                ("adapter_sha", m["adapter_sha"], bundle["artifacts"]["adapter_file"]["sha256"]),
            ]
            for label, actual_val, expected_val in checks:
                if actual_val != expected_val:
                    errors = fail(f"manifest {label}: {str(actual_val)[:16]}... != {str(expected_val)[:16]}...", errors)
            print(f"  OK: detector init + manifest consistency")
        except Exception as e:
            errors = fail(f"detector init: {e}", errors)
    else:
        errors = fail("checkpoint or config not found — cannot init detector", errors)

    # ── 7. Strict mode: branch + worktree ──
    if args.strict:
        branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        expected_branch = bundle["release_branch"]
        if branch != expected_branch:
            errors = fail(f"branch mismatch: {branch} != {expected_branch}", errors)
        else:
            print(f"  OK: branch={branch}")

        status = git_value(["status", "--porcelain"], cwd=repo_root)
        if status.strip():
            errors = fail("worktree is dirty", errors)
        else:
            print(f"  OK: worktree clean")

        head = git_value(["rev-parse", "HEAD"], cwd=repo_root)
        if head != bundle["source_code_commit"]:
            errors = fail(f"HEAD {head[:16]}... != source_code_commit {bundle['source_code_commit'][:16]}...", errors)
        else:
            print(f"  OK: HEAD matches source_code_commit")

    if errors == 0:
        print(f"\n  Bundle VERIFIED: {args.bundle}")
        return 0
    else:
        print(f"\n  Bundle FAILED: {errors} error(s)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
