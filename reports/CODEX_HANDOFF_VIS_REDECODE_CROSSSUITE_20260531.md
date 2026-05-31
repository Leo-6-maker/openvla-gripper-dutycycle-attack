# Codex Handoff — VIS Re-decode + CrossSuite Transfer Audit

**Date**: 2026-05-31 | **Status**: Ready for review

## 1. Server / Environment

```
ssh -J scene@10.60.133.3 liuyu@10.60.133.4
```

- Repo: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524`
- Conda env: `official_libero_20260525`
- GPU: 8× RTX 2080 Ti (11GB), GPU0 quarantined

## 2. Branches / Commits

| Location | Branch | SHA | Notes |
|----------|--------|-----|-------|
| **Production** | `exp/sustained-proxy-burst-control-20260530` | `07e13a0` | Object production — do not modify |
| **GitHub (source of truth)** | `exp/vis-token-prefix-redecode-and-crosssuite-audit-20260531` | `c24db50` | Review this branch |
| **Server** | `exp/vis-token-prefix-redecode-and-crosssuite-audit-20260531` | `bf4dd9c` | Patch-applied equivalent |

**Commits on GitHub branch:**
```
c24db50 Add Codex handoff: VIS re-decode status and cross-suite transfer audit
3763e4e Fix VIS PGD dtype handling: use model dtype for pixel_values
```

**Note**: Server SHA (`bf4dd9c`) and GitHub SHA (`c24db50`) differ because of local `git am` / format-patch recomputation. Codex should review GitHub branch `c24db50` as the authoritative source.

**PR status**: Not yet created.
PR creation URL: `https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/new/exp/vis-token-prefix-redecode-and-crosssuite-audit-20260531` (base: `exp/sustained-proxy-burst-control-20260530`)

## 3. Production Line (Unchanged)

- **Detector**: ProprioNoStep (13-dim proprio CausalTCNDetector)
- **Attack**: sustained_command_open_proxy_30
- **Scope**: LIBERO-Object only
- **Result**: High 0/10, Robust 10/10
- **Not modified** in this branch

## 4. Cross-Suite Transfer Result

| Suite | Trigger Rate | max_hs | Transfer |
|-------|-------------|--------|----------|
| Object | 94% | 0.9+ | Production |
| Spatial | 65% | up to 0.90 | Partial |
| Goal | 8% | up to 0.69 | Insufficient |

**Root cause**: eef_z distribution shift 5.7x (Object 0.19 → Spatial/Goal 1.08). Schema correct — no NaN/missing features. Not an engineering bug.

**Conclusion**: Cross-suite sus30 blocked. CrossSuite-ProprioNoStep-v2 needed (relative EEF features, multi-suite training).

## 5. VIS Engineering Status

### Fixed
- **dtype mismatch**: `attack_adapter.py` line 146 — `torch.float16` → `model_dtype` (1 line)
- **tokenization**: +1.0 and -1.0 map to different gripper bins (verified)
- **re-decode path**: `action_adv` is None by design; adversarial inputs in `debug['adv_inputs']` (pixel_values + input_ids)

### Findings
- PGD optimization works: target CE drops from 48 → 5
- Decoded gripper action unchanged at ε≤4/255 (token argmax doesn't flip)
- Arm dimensions drift (L2 0.76-1.52) despite gripper-only loss masking

### Conclusion
VIS PGD engineering path works, but small-epsilon token-prefix PGD insufficient for gripper steering. VIS rollout blocked.

## 6. Files Changed

| File | Change |
|------|--------|
| `src/gripper_attack/attack_adapter.py` | 1 line: fp16 → model dtype |

No other tracked files modified.

## 7. Tests

| Test | Result |
|------|--------|
| py_compile attack_adapter.py | PASS |
| py_compile runner | PASS |
| 13/13 sustained proxy tests | PASS |
| 6/6 success predicate tests | PASS |
| No large files committed | PASS |

## 8. Codex Review Focus

1. **dtype fix safety**: `next(self.model.parameters()).dtype` — safe for bf16/fp16/float32?
2. **Production semantics**: confirmed no changes to ProprioNoStep or sustained proxy
3. **No large artifacts**: confirmed no models/data/frames committed
4. **VIS re-decode interface**: `debug['adv_inputs']` documented but not formalized

## 9. Known Limitations

- VIS not rollout-ready (ε≤4/255 insufficient for token flip)
- Cross-suite not attack-ready (Goal 8% trigger)
- No detector-triggered VIS success claim
- No cross-suite universal ProprioNoStep claim

## 10. Valid Claims

- Object ProprioNoStep remains production
- Cross-suite zero-shot transfer is limited by distribution shift
- VIS dtype/tokenization/re-decode engineering has been debugged
- Small-epsilon token-prefix PGD changes loss but not decoded gripper token

## 11. Forbidden Claims

- VIS attack successful
- Command-layer sus30 equals VIS
- ProprioNoStep universal across LIBERO
- Cross-suite attack ready
- Detector oracle-optimal
- Universal attack

## 12. Next Recommended Work

**VIS (gated)**:
- Token flip threshold at higher eps (8/255, 16/255)
- Gripper-region/margin objectives
- Forced-window micro only after direction confirmed

**Cross-suite (gated)**:
- CrossSuite-ProprioNoStep-v2 with relative EEF features
- Clean teacher labels only
- Task-only / label-shuffle baselines

**Reporting**:
- Group meeting summary
- Final milestone report
