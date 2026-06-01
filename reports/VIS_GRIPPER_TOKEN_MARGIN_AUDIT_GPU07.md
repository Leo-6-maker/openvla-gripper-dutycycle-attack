# VIS Gripper Token Margin Audit

**Date**: 2026-06-01 | **GPU**: GPU7 (unstable for PGD)

## Margin Audit Results

| Frame | Open Rank | Open-Close Margin | Verdict |
|-------|-----------|-------------------|---------|
| ketchup_contact | **#2/255** | **-2.19** | PROMISING |
| tomato_contact | #30/255 | -13.47 | DIFFICULT |

## Key Finding

For ketchup contact frame (step 98):
- Top token: bin=127 (action=0.0, logit=16.12)
- Best open token: bin=252 (action=0.98, logit=13.94)
- **Only 2.19 logit gap** — CW-margin PGD can potentially bridge this

For tomato contact frame (step 134):
- Open token at rank #30 with -13.47 margin
- Much harder without large epsilon

## GPU Status

| GPU | Status | VIS PGD? |
|-----|--------|----------|
| 0 | OOM (fragmentation) | Maybe with PYTORCH_CUDA_ALLOC_CONF |
| 4 | QUARANTINED (fresh Xid13) | NO |
| 7 | UNSTABLE (misaligned address errors) | NO |
| 1-3,5-6 | Used for duration calibration | After completion |

## Conclusion

Open token is surprisingly close to the top for ketchup frame (rank #2, gap 2.19). CW-margin objective is promising but requires a healthy GPU for PGD. GPU7 confirmed unstable; GPU0 needs memory config. VIS PGD deferred until healthy GPU available.
