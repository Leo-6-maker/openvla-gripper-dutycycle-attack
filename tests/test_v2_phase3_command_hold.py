#!/usr/bin/env python3
"""CPU preflight: validate Phase 3 anchor manifest and condition matrix before GPU."""
import json, sys, os

MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/v2_phase3_anchor_manifest.json'

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

K = manifest['K']
assert K == 10, f"K must be 10, got {K}"
errors = []

for state_key, state_data in manifest['states'].items():
    sid = state_key
    n_steps = state_data['n_steps']
    print(f"\n=== {sid} ===")
    print(f"  gc={state_data['gc_start']} sg={state_data['sg_start']} sc={state_data['sc_start']} rs={state_data['rs_start']}")

    for wname, winfo in state_data['windows'].items():
        anchor = winfo['anchor']
        valid = winfo['valid']

        if wname == 'W3_MID':
            sc = state_data['sc_start']; rs = state_data['rs_start']
            if anchor is None:
                print(f"  {wname}: ABSTAIN (no hazard corridor)")
                continue

        if not valid:
            print(f"  {wname}: INVALID — {winfo.get('reason', '?')}")
            continue

        window = winfo['window']
        assert window[0] == anchor, f"window start mismatch: {window[0]} != {anchor}"
        assert window[1] == anchor + K - 1, f"window end mismatch: {window[1]} != {anchor + K - 1}"
        assert window[0] >= 0, f"anchor negative: {anchor}"
        assert window[1] < n_steps, f"window exceeds episode: {window[1]} >= {n_steps}"

        # Phase ordering check
        if wname == 'W0_D5':
            pass  # grasp_close, no ordering constraint vs itself
        elif wname == 'W1_SG5':
            sg = state_data['sg_start']
            assert sg is not None, f"{wname} requires stable_grasp"
            assert anchor >= sg, f"{wname} anchor {anchor} < sg_start {sg}"
        elif wname == 'W2_SC5':
            sc = state_data['sc_start']
            assert sc is not None, f"{wname} requires stable_carry"
            assert anchor >= sc, f"{wname} anchor {anchor} < sc_start {sc}"
            assert state_data['obj_lifted_at_sc'], f"{wname}: object not lifted at sc_start"
        elif wname == 'W3_MID':
            sc = state_data['sc_start']; rs = state_data['rs_start']
            assert sc is not None and rs is not None, f"{wname} requires sc and rs"
            assert anchor >= sc + 5, f"{wname} anchor {anchor} < sc+5"
            assert anchor + K - 1 < rs, f"{wname} window crosses rs {rs}"

        print(f"  {wname}: anchor={anchor} window={window} OK")

    # Verify phase ordering
    gc = state_data['gc_start']; sg = state_data['sg_start']
    sc = state_data['sc_start']; rs = state_data['rs_start']
    if gc is not None and sg is not None:
        assert gc <= sg, f"gc({gc}) > sg({sg})"
    if sg is not None and sc is not None:
        assert sg <= sc, f"sg({sg}) > sc({sc})"
    if sc is not None and rs is not None:
        assert sc <= rs, f"sc({sc}) > rs({rs})"

print(f"\n({len(manifest['states'])} states, {K=})")
if errors:
    print(f"ERRORS: {errors}")
    sys.exit(1)
else:
    print("ALL PREFLIGHT CHECKS PASSED")
