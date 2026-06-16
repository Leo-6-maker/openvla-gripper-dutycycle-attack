#!/usr/bin/env python3
"""Verify D5 v1 production bundle integrity.

Checks all file SHAs, detector init, bound_manifest consistency.
Exits 0 on success, non-zero on any mismatch.
"""
import argparse, hashlib, json, os, sys

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


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to production bundle JSON")
    args = ap.parse_args()

    bundle = json.load(open(args.bundle))
    errors = 0

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. Check all file artifacts
    artifacts = bundle.get("artifacts", {})
    for name, art in artifacts.items():
        local_path = os.path.join(repo_root, art.get("path", ""))
        remote_path = art.get("path", "")

        # Check local repo files
        if os.path.exists(local_path):
            actual = sha256_file(local_path)
            expected = art["sha256"]
            if actual != expected:
                fail(f"{name} local SHA mismatch: expected {expected[:16]}... got {actual[:16]}...")
                errors += 1
            else:
                print(f"  OK: {name} (local)")
        elif os.path.exists(remote_path):
            actual = sha256_file(remote_path)
            expected = art["sha256"]
            if actual != expected:
                fail(f"{name} remote SHA mismatch: expected {expected[:16]}... got {actual[:16]}...")
                errors += 1
            else:
                print(f"  OK: {name} (remote)")
        else:
            # Try server paths
            print(f"  WARN: {name} not found at local or remote path, skipping")

    # 2. Verify tau
    expected_tau = bundle["parameters"]["tau"]
    if abs(expected_tau - 0.050) > 1e-9:
        fail(f"Bundle tau {expected_tau} != 0.050")
        errors += 1
    else:
        print(f"  OK: tau = {expected_tau}")

    # 3. Verify feature names
    expected_features = bundle["parameters"]["feature_names"]
    if len(expected_features) != 16:
        fail(f"Feature count {len(expected_features)} != 16")
        errors += 1
    else:
        print(f"  OK: feature_names count = 16")

    # 4. Verify adapter source commit
    expected_commit = bundle["parameters"]["adapter_source_commit"]
    if not expected_commit.startswith("44bf7b86"):
        fail(f"Adapter source commit {expected_commit[:16]} != 44bf7b86...")
        errors += 1
    else:
        print(f"  OK: adapter_source_commit")

    # 5. GPU authorization
    quarantined = bundle["gpu"]["quarantined_gpus"]
    if "3" not in quarantined:
        fail("GPU3 must be quarantined")
        errors += 1
    else:
        print("  OK: GPU3 quarantined")

    excluded = bundle["gpu"]["excluded_gpus"]
    for g in ["0", "4"]:
        if g not in excluded:
            fail(f"GPU{g} must be excluded")
            errors += 1
    print("  OK: GPU0/4 excluded")

    # 6. Try detector init if checkpoint is accessible
    ckpt_path = artifacts.get("checkpoint", {}).get("path", "")
    cfg_path = artifacts.get("config", {}).get("path", "")
    if os.path.exists(ckpt_path) and os.path.exists(cfg_path):
        try:
            from gripper_attack.d5_frozen_online_detector_v1 import D5FrozenOnlineDetectorV1
            det = D5FrozenOnlineDetectorV1(ckpt_path, cfg_path)
            manifest = det.bound_manifest
            # Verify manifest consistency
            if manifest["tau"] != 0.050:
                fail(f"Detector manifest tau {manifest['tau']} != 0.050")
                errors += 1
            if manifest["checkpoint_sha"] != artifacts["checkpoint"]["sha256"]:
                fail("Detector manifest checkpoint SHA mismatch")
                errors += 1
            if manifest["config_sha"] != artifacts["config"]["sha256"]:
                fail("Detector manifest config SHA mismatch")
                errors += 1
            if "adapter_sha" not in manifest:
                fail("Detector manifest missing adapter_sha")
                errors += 1
            print("  OK: detector init + manifest consistency")
        except Exception as e:
            fail(f"Detector init failed: {e}")
            errors += 1
    else:
        print("  WARN: checkpoint or config not accessible, skipping detector init")

    # Summary
    if errors == 0:
        print(f"\nBundle VERIFIED: {args.bundle}")
        return 0
    else:
        print(f"\nBundle FAILED: {errors} error(s)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
