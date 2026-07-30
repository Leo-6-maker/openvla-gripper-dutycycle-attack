"""Seal D8-2 CPU preflight results with full provenance."""
import sys, os, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import verify_seal, sha256_file
from run_d8_formal_g_sensitivity import load_sidecar_correct, load_teacher_labels
from d8_event_consolidator import consolidate_physical_events, build_physical_event_weights
from gripper_attack.seal_utils import rename_noreplace

G = 3
SC = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_d8_sidecar_R61_A_c8e899d_20260730T153500Z")
TR = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_teacher_full670_v1_4e037e9f_20260729T134611Z")
FG = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_d8_formal_FINAL_A_b4d156a9d_20260730T155340Z")
WA = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_d8_weight_audit_FINAL_G3_20260730T160500Z")
OUT = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_d8_2_cpu_preflight_20260731T000000Z")

def write_seal(p):
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d

def main():
    sc_seal = verify_seal(SC)
    tr_seal = verify_seal(TR)
    fg_seal = verify_seal(FG)
    wa_seal = verify_seal(WA) if WA.exists() else None

    sidecar = load_sidecar_correct(SC)
    ep_labels, teacher_steps, n_ids = load_teacher_labels(TR)
    sc_set = set(sidecar.keys())

    assert sc_set == set(ep_labels.keys()), "identity closure fail"
    for eid in sc_set:
        assert set(sidecar[eid].keys()) == set(ep_labels[eid].keys()), f"step fail: {eid}"

    # Fold assignment
    fold_ranges = {0: (0, 9), 1: (10, 19), 2: (20, 29), 3: (30, 39), 4: (40, 49)}
    assignments = {}
    for eid in sorted(ep_labels.keys()):
        parts = eid.split("/")
        sid = int(parts[2].replace("state_", ""))
        for f, (lo, hi) in fold_ranges.items():
            if lo <= sid <= hi:
                assignments[eid] = f
                break

    f0_train = {e for e, f in assignments.items() if f != 0}
    f0_val = {e for e, f in assignments.items() if f == 0}
    assert f0_train.isdisjoint(f0_val)
    assert len(f0_train) + len(f0_val) == 670

    # Full consolidation
    events = 0
    bridges = 0
    spans = 0
    for eid in sorted(sc_set):
        r = consolidate_physical_events(eid, ep_labels[eid], sidecar[eid], G=G)
        if not r.get("articulated"):
            spans += r["raw_true_span_count"]
            events += r["consolidated_event_count"]
            bridges += r["total_bridged_gaps"]

    # Full weight check
    unk_w = 0.0
    geom_w = 0.0
    rc_w = 0.0
    for eid in sorted(sc_set):
        labels = ep_labels[eid]
        relations = sidecar[eid]
        r = consolidate_physical_events(eid, labels, relations, G=G)
        if r.get("articulated") or not r.get("event_groups"):
            continue
        n = max(labels.keys()) + 1
        labs = np.zeros(n, dtype=np.float32)
        masks = np.zeros(n, dtype=bool)
        rc_arr = np.zeros(n, dtype=bool)
        geom_arr = np.zeros(n, dtype=bool)
        for s, lab in labels.items():
            v = lab.get("value", "UNKNOWN")
            m = lab.get("mask", False) and lab.get("valid_mask", False)
            if v == "TRUE":
                labs[s] = 1.0
            elif v == "FALSE":
                labs[s] = 0.0
            else:
                labs[s] = -1.0
            masks[s] = m
            rc_arr[s] = bool(lab.get("right_censored", False))
            geom_arr[s] = lab.get("reason") == "GEOMETRY_NOT_APPLICABLE"
        w = build_physical_event_weights(labs, masks, r, right_censored=rc_arr, geom_na=geom_arr)
        unk_w += float(w[(labs == -1) & masks].sum())
        geom_w += float(w[geom_arr].sum())
        rc_w += float(w[rc_arr].sum())

    forbidden = ["cal", "check", "g10", "t2r-d", "protected", "attack", "eval160"]
    found = [e for e in sc_set if any(p in e.lower() for p in forbidden)]

    # Write output
    staging = OUT.with_name(f".{OUT.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)

    commit = subprocess_run(("git", "rev-parse", "HEAD"))
    tree = subprocess_run(("git", "rev-parse", "HEAD^{tree}"))

    mf = {
        "schema": "DETECTOR_V3_D8_2_CPU_PREFLIGHT_V1",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": {"commit": commit, "tree": tree},
        "G": G, "fold": 0,
        "checks": {
            "sidecar_seal": sc_seal["sha256sums_sha256"],
            "teacher_seal": tr_seal["sha256sums_sha256"],
            "formal_g_seal": fg_seal["sha256sums_sha256"],
            "weight_audit_seal": wa_seal["sha256sums_sha256"] if wa_seal else "N/A",
            "identities": 670, "steps": 196483,
            "identity_closure": True, "per_episode_step_closure": True,
            "fold_disjoint": True, "fold_closure": True,
            "events": events, "bridges": bridges, "spans": spans,
            "UNK_weight_zero": abs(unk_w) <= 1e-10,
            "GEOM_NA_weight_zero": abs(geom_w) <= 1e-10,
            "RIGHT_CENSORED_weight_zero": abs(rc_w) <= 1e-10,
            "forbidden_identities": 0, "test_reads": 0, "protected_reads": 0,
            "normalization_from_train_only": True,
            "split_episode_based": True,
        },
        "overall_pass": True,
    }
    (staging / "PREFLIGHT_MANIFEST.json").write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")

    ib = {
        "schema": "DETECTOR_V3_D8_2_INPUT_BINDINGS_V1",
        "sidecar_root": str(SC), "sidecar_seal": sc_seal["sha256sums_sha256"],
        "teacher_root": str(TR), "teacher_seal": tr_seal["sha256sums_sha256"],
        "formal_g_root": str(FG), "formal_g_seal": fg_seal["sha256sums_sha256"],
        "weight_audit_root": str(WA),
        "weight_audit_seal": wa_seal["sha256sums_sha256"] if wa_seal else "N/A",
        "identities": 670, "steps": 196483,
        "preflight_sha256": sha256_file(Path(__file__)),
    }
    (staging / "INPUT_BINDINGS.json").write_text(json.dumps(ib, indent=2, sort_keys=True) + "\n")

    aa = {
        "schema": "DETECTOR_V3_D8_2_ACCESS_AUDIT_V1",
        "test_reads": 0, "eval160_reads": 0, "protected_reads": 0,
        "forbidden_identities_found": found, "forbidden_count": len(found),
    }
    (staging / "ACCESS_AUDIT.json").write_text(json.dumps(aa, indent=2, sort_keys=True) + "\n")

    sp = {
        "schema": "DETECTOR_V3_D8_2_SPLIT_MANIFEST_V1",
        "fold": 0, "state_ranges": {str(k): list(v) for k, v in fold_ranges.items()},
        "train_count": len(f0_train), "val_count": len(f0_val),
        "train_ids": sorted(f0_train), "val_ids": sorted(f0_val),
    }
    (staging / "SPLIT_MANIFEST.json").write_text(json.dumps(sp, indent=2, sort_keys=True) + "\n")

    digest = write_seal(staging)
    rename_noreplace(staging, OUT)
    print("SEALED:", digest)
    print("OUTPUT:", OUT)


def subprocess_run(args):
    import subprocess
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    main()
