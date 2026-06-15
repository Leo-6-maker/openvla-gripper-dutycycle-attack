# D4.3A Clean-Shadow Canary Result

## Execution
- HEAD: e70507f282239597426bd4967a25da0acb362473
- Branch: exp/l12-production-streaming-adapter-20260615
- Server: klfy-SYS-4028GR-TR2
- GPU: physical 2,6
- Conda: openvla_official_libero_20260525

## Result: D4_SHADOW_CANARY_PASS

### Episodes (8/8 rc=0, 0 retries)
| State | Ref | Shadow | Steps | Emit |
|-------|-----|--------|-------|------|
| milk_s23 | 270s | 268s | 217 | step 49 |
| salad_dressing_s19 | 169s | 168s | 124 | abstain |
| bbq_sauce_s36 | 223s | 210s | 165 | step 49 |
| tomato_sauce_s6 | 197s | 205s | 149 | abstain |

### Gates
- Ref/Shadow steps: identical (4/4)
- Ref/Shadow success: identical (4/4)  
- Action identity: 0 mutations
- Invalid field steps: 0
- Detector exception: 0
- Abstain emission: 0
- GPU before/after: 0/0 residual

### Independent Auditor
204/204 PASS, 0 FAIL

## Status
- CPU_TESTS: PASS
- D4.2c_PARITY: PASS (46/46)
- STATE_FREEZE: PASS
- D4.3a_CANARY: PASS
- PANEL: NOT RUN
- ATTACK: NOT RUN
