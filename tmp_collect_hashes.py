"""Collect all E-R3a evidence hashes."""
import json, hashlib
from pathlib import Path

E3A = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_e_r3a_20260720")
OUT = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4e_e_r3a_output_20260720")
TASK00 = Path("/mnt/sdc/dty_user/openvla_attack_evidence/r10_4d_passive_smoke_output_20260720")

def sha(f):
    p = Path(f)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"

print("=== RECEIPT ===")
print("receipt_sha256:", sha(E3A / "r10_4e_e_r3a_receipt.json"))

print("\n=== EXTERNAL TASK00 ===")
print("task00_sha256sums_sha256:", sha(TASK00 / "SHA256SUMS"))

print("\n=== FRESH TASK01 ===")
print("task01_sha256sums_sha256:", sha(OUT / "libero_10_task_01_state_20/SHA256SUMS"))

print("\n=== PANEL SEAL ===")
print("panel_sha256sums_sha256:", sha(OUT / "SHA256SUMS"))

print("\n=== LEDGER REVISIONS ===")
for i in range(4):
    fn = "panel_ledger_rev{:04d}.json".format(i)
    s = sha(OUT / fn)
    print("  rev{:04d}: {}".format(i, s))

print("\n=== FINAL LEDGER ===")
print("panel_ledger.json:", sha(OUT / "panel_ledger.json"))

print("\n=== AUDIT ===")
audit_path = E3A / "r10_4e_e_r3a_audit_v2.json"
s = sha(audit_path)
print("audit_report_sha256:", s)
if Path(audit_path).is_file():
    audit = json.load(open(audit_path))
    print("audit_overall:", audit.get("overall"))
    comps = audit.get("components", {})
    for comp in ["external_task00", "panel_aggregate_seal", "reuse_binding",
                 "fresh_task01", "ledger_chain", "panel_summary", "panel_structure"]:
        c = comps.get(comp, {})
        v = c.get("valid")
        print("  {}: valid={}".format(comp, v))
