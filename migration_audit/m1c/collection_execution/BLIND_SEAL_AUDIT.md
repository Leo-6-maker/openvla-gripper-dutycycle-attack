# M1C Blind Seal Audit

## Status: COMPROMISED

```text
Discovered: 2026-06-23 ~14:45 CST
Root cause: Collection script launched GPU6 blind pool alongside train/val.
            Aggregate progress reports included GPU6 success/emit counts.
Exposure:   12 blind cells had success and emit data inspected.
```

## Exposed Cells

All 12 cells from GPU6 (libero_object, B0 BF16+Eager):

| Task | State | Emit | Steps | Success |
|---|---|---|---|---|
| alphabet_soup | 38 | 115 | 247 | True |
| alphabet_soup | 39 | 184 | 400 | False |
| alphabet_soup | 40 | 101 | 228 | True |
| alphabet_soup | 41 | 100 | 218 | True |
| alphabet_soup | 42 | 130 | 200 | True |
| alphabet_soup | 43 | 77 | 168 | True |
| alphabet_soup | 44 | 75 | 199 | True |
| alphabet_soup | 45 | 173 | 231 | True |
| alphabet_soup | 46 | 98 | 163 | True |
| alphabet_soup | 47 | -1 | 400 | False |
| cream_cheese | 38 | 68 | 111 | True |
| cream_cheese | 39 | 83 | 137 | True |

## Disposition

```text
BLIND_SEAL = BROKEN
EXPOSED_CELLS = 12 (all alphabet_soup task 0 + cream_cheese task 1, states 38-47)
TARGET_BLIND_SIZE = 100 (10 tasks × 10 states)
REMAINING_UNEXPOSED_BLIND = 0 (all GPU6 cells marked compromised)
BUFFER_AVAILABLE = 20 (states 48-49 × 10 tasks)
```

## Remediation

1. All 12 exposed cells: `BLIND_COMPROMISED_DIAGNOSTIC_ONLY`
2. Files preserved but excluded from final blind evaluation
3. Replacement: buffer states 48-49 (20 cells) + need additional source
4. GPU6 reassigned to train overflow or C2 ablation
5. Future monitoring must NOT report blind per-episode metrics
6. Blind collection resumes only after: (a) new blind states acquired, OR (b) protocol amended for cross-suite blind

## Impact

- M1C final blind evaluation delayed
- Cross-suite clean corpus may serve as alternative blind source
- Train + validation collection unaffected (GPU2/3/4 continue)
