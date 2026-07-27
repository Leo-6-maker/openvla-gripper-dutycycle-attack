"""[DeepSeek] R4-A/B: Causal test — system LIBERO vs collector exact checkout.

Runs the canary state-forward in two environments:
  A: system LIBERO (current verifier)
  B: collector checkout @ /mnt/sdc/dty_user/pi0_openpi/third_party/libero

Hypothesis: body_origin drift is caused by LIBERO source version mismatch.
"""
import json, os, sys, hashlib, subprocess, time
from pathlib import Path

COLLECTOR_LIBERO = "/mnt/sdc/dty_user/pi0_openpi/third_party/libero"
VERIFIER_SCRIPT = "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/audit_grec_r3_state.py"
PYTHON_BIN = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"


def record_provenance(label, libero_root):
    """Record LIBERO checkout provenance."""
    import subprocess as sp
    result = {"label": label, "libero_root": str(Path(libero_root).resolve())}
    try:
        result["git_head"] = sp.check_output(
            ["git", "-C", libero_root, "rev-parse", "HEAD"], text=True).strip()
        result["git_tree"] = sp.check_output(
            ["git", "-C", libero_root, "rev-parse", "HEAD^{tree}"], text=True).strip()
        result["git_status"] = sp.check_output(
            ["git", "-C", libero_root, "status", "--porcelain"], text=True).strip()
        result["git_log_1"] = sp.check_output(
            ["git", "-C", libero_root, "log", "--oneline", "-1"], text=True).strip()
    except Exception as e:
        result["git_error"] = str(e)
    return result


def run_canary(label, libero_root=None):
    """Run the canary verifier. If libero_root is set, prepend to PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if libero_root:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(Path(libero_root).resolve()) + (":" + existing if existing else "")
    else:
        env.pop("PYTHONPATH", None)

    cmd = [PYTHON_BIN, "-c", """
import sys, os
# Verify libero location
import libero
print("LIBERO_FILE:", libero.__file__)
print("LIBERO_PATH:", os.path.dirname(os.path.dirname(libero.__file__)))

# Run the verifier
sys.argv = ["audit_grec_r3_state.py", "--mode", "canary"]
exec(open(sys.argv[0]).read())
"""]

    # Actually, simpler: run the script directly with PYTHONPATH
    cmd = [PYTHON_BIN, VERIFIER_SCRIPT, "--mode", "canary"]

    # But first verify the libero import location
    verify_cmd = [PYTHON_BIN, "-c",
        "import libero; print('LIBERO_FILE:'); print(libero.__file__)"]

    print(f"\n{'='*60}")
    print(f"Run: {label}")
    print(f"PYTHONPATH={env.get('PYTHONPATH', '(system)')}")
    print(f"{'='*60}")

    # Verify
    v = subprocess.run(verify_cmd, capture_output=True, text=True, env=env, timeout=30)
    print("Verify:", v.stdout.strip())

    # Run canary
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    elapsed = time.time() - start

    return {
        "label": label,
        "exit_code": r.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": "\n".join(r.stdout.splitlines()[-30:]) if r.stdout else "",
        "stderr_tail": "\n".join(r.stderr.splitlines()[-10:]) if r.stderr else "",
        "full_stdout": r.stdout,
    }


def main():
    print("=" * 60)
    print("[DeepSeek] R4-A/B: Exact Environment Causal Test")
    print("=" * 60)

    # Record provenance
    prov_a = record_provenance("system", os.path.dirname(os.path.dirname(
        __import__('libero').__file__))) if False else {"label": "system", "note": "cannot import libero here"}
    prov_b = record_provenance("collector", COLLECTOR_LIBERO)
    print(f"\nCollector LIBERO: {prov_b['git_head']} tree={prov_b['git_tree']}")
    print(f"  status: {prov_b.get('git_status', '?')[:80]}")

    # Run A: system LIBERO
    result_a = run_canary("A: system LIBERO")

    # Run B: collector LIBERO
    result_b = run_canary("B: collector LIBERO", COLLECTOR_LIBERO)

    # Report
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    for r in [result_a, result_b]:
        print(f"\n{r['label']}: exit={r['exit_code']} time={r['elapsed_s']:.1f}s")
        print(r['stdout_tail'])

    # Comparison
    if result_a["exit_code"] == 5 and result_b["exit_code"] == 0:
        print("\nCAUSE = LIBERO_SOURCE_VERSION_MISMATCH")
        print("Collector checkout PASS, system FAIL — version gap confirmed.")
    elif result_b["exit_code"] == 5:
        print("\nHOLD_RUNTIME_OR_STATE_DEPENDENCY")
        print("Collector checkout also FAILS — not a version issue alone.")
    else:
        print(f"\nA exit={result_a['exit_code']}, B exit={result_b['exit_code']}")


if __name__ == "__main__":
    main()
