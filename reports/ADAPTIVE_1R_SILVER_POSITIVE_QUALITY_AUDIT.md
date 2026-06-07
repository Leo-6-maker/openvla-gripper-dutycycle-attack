# Adaptive 1R Silver Positive Quality Audit

**Time**: 2026-06-06 12:58:52.384057
**VIS1R_DONE**: 21

## Summary

| Metric | Count |
|--------|-------|
| Total provisional silver | 21 |
| Mechanism: qpos bridge | 3 |
| Mechanism: token flip only | 0 |
| Mechanism: pending/parse error | 14 |
| Eligible silver | 3 |
| Dubious positive | 14 |

## Action

- Eligible silver rows may be considered for ablation with sample_weight=0.5.
- Dubious positives must NOT enter detector training.
- pending_negative_1r count: 0.

## Per-Candidate Detail

| ID | Task | Mechanism | VisOpen | Done | Steps | QposDelta | Eligible |
|----|------|-----------|---------|------|-------|-----------|----------|
| adp_0004 | milk_s5 | qpos_bridge_present_task_failu | 0 | False | 299 | 0.027929 | yes |
| adp_0005 | milk_s5 | qpos_bridge_present_task_failu | 0 | False | 299 | 0.036089 | yes |
| adp_0006 | milk_s8 | pending_audit_task_failure | 0 | False | 299 | 0.002986 | no |
| adp_0008 | milk_s1 | pending_audit_task_failure | 0 | False | 299 | 0.000455 | no |
| adp_0009 | milk_s1 | pending_audit | 0 | True | 195 | 0.000447 | no |
| adp_0011 | milk_s9 | pending_audit | 0 | True | 118 | 0.005238 | no |
| adp_0012 | milk_s9 | pending_audit | 0 | True | 167 | 0.000519 | no |
| adp_0014 | milk_s9 | pending_audit | 0 | True | 113 | 0.000457 | no |
| adp_0015 | ketchup_s0 | pending_audit | 0 | True | 145 | 0.000403 | no |
| adp_0016 | ketchup_s1 | pending_audit | 0 | True | 174 | 0.003099 | no |
| adp_0020 | ketchup_s2 | pending_audit | 0 | True | 144 | 0.008969 | no |
| adp_0021 | ketchup_s2 | pending_audit | 0 | True | 139 | 0.000536 | no |
| adp_0035 | ketchup_s4 | pending_audit | 0 | True | 148 | 0.000398 | no |
| adp_0036 | ketchup_s4 | pending_audit | 0 | True | 186 | 0.000388 | no |
| adp_0038 | alphabet_soup_s2 | pending_audit_task_failure | 0 | False | 299 | 0.000636 | no |
| adp_0039 | alphabet_soup_s3 | pending_audit_task_failure | 0 | False | 299 | 0.000373 | no |
| adp_0040 | alphabet_soup_s4 | qpos_bridge_present | 0 | True | 273 | 0.013017 | yes |
| adp_0041 | alphabet_soup_s4 | pending_audit | 0 | True | 142 | 0.009036 | no |
| adp_0042 | alphabet_soup_s4 | pending_audit | 0 | True | 139 | 0.000483 | no |
| adp_0043 | alphabet_soup_s4 | pending_audit | 0 | True | 153 | 0.000397 | no |
| adp_0044 | alphabet_soup_s6 | pending_audit | 0 | True | 153 | 0.000327 | no |
