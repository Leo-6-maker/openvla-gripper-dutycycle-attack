#!/usr/bin/env python3
"""Gate F2-C0: Full C2G cross-artifact action-lineage audit.

Audits ALL steps across ALL 50 fold-0 multi-object validation episodes.
Produces confusion matrices, time-shift analysis, and numerical field comparison
for S1 raw_close vs Teacher candidate_close.

Fail-closed: shared derive_teacher_mask(), no field defaults.
"""

import json, sys
from collections import defaultdict
from pathlib import Path

OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
FOLD_ROOT = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
TEACHER_ROOT = OPS / "OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719/labels"
S1_ROOT = OPS / "OFFICIAL_V3_S1_FIT_V1_5e27d7c"
FEATURE_CONTRACT_SHA = "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366"


def jsonl(path):
    if not path.is_file():
        raise SystemExit("FILE_MISSING:{}".format(path))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit("FILE_EMPTY:{}".format(path))
    return [json.loads(l) for l in lines]


def derive_teacher_mask(record):
    """Frozen label derivation. All fields required — no defaults."""
    cc = record["candidate_close"]
    valid = record["student_valid"]
    known = record["known_mask"]
    sg = float(record["stable_grasp_score"])
    return bool(cc) and bool(valid) and bool(known) and float(sg) >= 0.3


def raw_is_close(raw):
    return float(raw) < 0.5


def expected_env_close(raw):
    """Official postprocess: env = -sign(2*raw-1). env>0 = close."""
    import numpy as np
    env = float(-np.sign(2.0 * float(raw) - 1.0))
    return env > 0.0


def main():
    manifest = json.loads((FOLD_ROOT / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json").read_text())
    if manifest.get("feature_order_sha256") != FEATURE_CONTRACT_SHA:
        raise SystemExit("FEATURE_CONTRACT_HASH_MISMATCH")
    f0 = [f for f in manifest["folds"] if f["fold_id"] == 0][0]
    val_ids = [i for i in f0["validation_identities"] if i.startswith("libero_10")]
    print("F2-C0: Action-Lineage Audit")
    print("  {} multi-object episodes".format(len(val_ids)))
    print("  Feature contract: {} OK".format(FEATURE_CONTRACT_SHA[:16]))

    # Verify all files exist
    for identity in val_ids:
        parts = identity.split("/")
        tp = TEACHER_ROOT / parts[0] / parts[1] / parts[2] / "physics_teacher_v21.jsonl"
        sp = S1_ROOT / parts[0] / parts[1] / parts[2] / "student_input_records.jsonl"
        if not tp.is_file():
            raise SystemExit("TEACHER_MISSING:{}".format(identity))
        if not sp.is_file():
            raise SystemExit("S1_MISSING:{}".format(identity))
    print("  All {} identities verified".format(len(val_ids)))

    # ── Matrix A: S1 raw_close vs Teacher cc —──────────────────────────────
    mat_a = defaultdict(int)  # (raw_close, teacher_cc) → count
    mat_b = defaultdict(int)  # (raw_close, teacher_mask) → count
    mat_c = defaultdict(int)  # (expected_env_close, teacher_cc) → count
    time_shift = defaultdict(lambda: defaultdict(int))  # offset → (raw_close, cc) → count
    raw_vals = []
    raw_cc_values = []

    total_steps = 0
    errors = 0

    for identity in val_ids:
        parts = identity.split("/")
        suite, task, state = parts
        teacher_recs = jsonl(TEACHER_ROOT / suite / task / state / "physics_teacher_v21.jsonl")
        s1_recs = jsonl(S1_ROOT / suite / task / state / "student_input_records.jsonl")
        T = len(s1_recs)
        if len(teacher_recs) != T:
            raise SystemExit("LENGTH_MISMATCH:{} T={} S={}".format(identity, len(teacher_recs), T))

        for t in range(T):
            tr = teacher_recs[t]
            sr = s1_recs[t]
            total_steps += 1

            s1_raw = float(sr["features_25d"][0])
            s1_close = raw_is_close(s1_raw)
            teacher_cc = bool(tr["candidate_close"])
            teacher_mask = derive_teacher_mask(tr)
            exp_env_close = expected_env_close(s1_raw)

            # Matrix A: raw_close × candidate_close
            mat_a[(s1_close, teacher_cc)] += 1

            # Matrix B: raw_close × full teacher mask
            mat_b[(s1_close, teacher_mask)] += 1

            # Matrix C: expected_env_close × candidate_close
            mat_c[(exp_env_close, teacher_cc)] += 1

            raw_vals.append(s1_raw)
            raw_cc_values.append(int(teacher_cc))

            # Time-shift: Teacher cc[t] vs S1 raw_close[t+k]
            for offset in range(-3, 4):
                tk = t + offset
                if 0 <= tk < T:
                    s1_k_raw = float(s1_recs[tk]["features_25d"][0])
                    s1_k_close = raw_is_close(s1_k_raw)
                    time_shift[offset][(s1_k_close, teacher_cc)] += 1

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Matrix A: S1 raw_close × Teacher candidate_close")
    print("  {} total steps".format(total_steps))
    print("=" * 65)
    for (rc, cc), count in sorted(mat_a.items()):
        pct = 100.0 * count / total_steps
        print("  raw_close={:<6s} cc={:<6s} → {:>8d} ({:>5.1f}%)".format(
            str(rc), str(cc), count, pct))

    agree = mat_a.get((True, True), 0) + mat_a.get((False, False), 0)
    disagree = mat_a.get((True, False), 0) + mat_a.get((False, True), 0)
    boundary = total_steps - agree - disagree
    print("  AGREEMENT:  {} ({:.1f}%)".format(agree, 100*agree/total_steps))
    print("  DISAGREEMENT: {} ({:.1f}%)".format(disagree, 100*disagree/total_steps))
    if disagree > agree:
        print("  >> GLOBAL INVERSION CONFIRMED <<")
        inversion_pct = 100.0 * disagree / total_steps
        print("  Inversion rate: {:.1f}%".format(inversion_pct))

    print("\nMatrix B: S1 raw_close × Teacher full mask (cc+valid+known+sg>=0.3)")
    for (rc, mask), count in sorted(mat_b.items()):
        pct = 100.0 * count / total_steps
        print("  raw_close={:<6s} mask={:<6s} → {:>8d} ({:>5.1f}%)".format(
            str(rc), str(mask), count, pct))

    print("\nMatrix C: expected_env_close(postprocess(S1 raw)) × Teacher cc")
    for (ec, cc), count in sorted(mat_c.items()):
        pct = 100.0 * count / total_steps
        print("  env_close={:<6s} cc={:<6s} → {:>8d} ({:>5.1f}%)".format(
            str(ec), str(cc), count, pct))
    agree_c = mat_c.get((True, True), 0) + mat_c.get((False, False), 0)
    disagree_c = mat_c.get((True, False), 0) + mat_c.get((False, True), 0)
    print("  AGREEMENT:  {} ({:.1f}%)".format(agree_c, 100*agree_c/total_steps))
    print("  DISAGREEMENT: {} ({:.1f}%)".format(disagree_c, 100*disagree_c/total_steps))

    print("\nS1 raw value stats:")
    import numpy as np
    ra = np.array(raw_vals)
    print("  min={:.4f}  max={:.4f}  mean={:.4f}  median={:.4f}".format(
        ra.min(), ra.max(), ra.mean(), np.median(ra)))
    raw_close_pct = 100.0 * (ra < 0.5).sum() / len(ra)
    print("  raw_close steps: {:.1f}%".format(raw_close_pct))
    cc_pct = 100.0 * sum(raw_cc_values) / len(raw_cc_values)
    print("  Teacher cc=True steps: {:.1f}%".format(cc_pct))

    print("\nTime-shift analysis: Teacher cc[t] vs S1 raw_close[t+k]")
    for offset in range(-3, 4):
        ts = time_shift[offset]
        agree_ts = ts.get((True, True), 0) + ts.get((False, False), 0)
        disagree_ts = ts.get((True, False), 0) + ts.get((False, True), 0)
        total_ts = agree_ts + disagree_ts
        if total_ts > 0:
            print("  k={:+d}: agree={:.1f}% disagree={:.1f}% (n={})".format(
                offset, 100*agree_ts/total_ts, 100*disagree_ts/total_ts, total_ts))

    # Write output
    output = {
        "schema": "R10_F2_C0_ACTION_LINEAGE_AUDIT_V1",
        "n_episodes": len(val_ids),
        "n_steps": total_steps,
        "matrix_a": {str(k): v for k, v in mat_a.items()},
        "matrix_b": {str(k): v for k, v in mat_b.items()},
        "matrix_c": {str(k): v for k, v in mat_c.items()},
        "inversion_rate": round(100.0 * disagree / total_steps, 2) if disagree > agree else None,
    }
    out_path = Path("/tmp/r10_f2_c0_action_lineage.json")
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    print("\nOutput: {}".format(out_path))


if __name__ == "__main__":
    main()
