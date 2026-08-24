# Official V3 Detector V5 R2 Fold-0 handoff

Fold-0 was intentionally not promoted from the R2 stratified smoke.  The
required smoke conditions were not met: V5-A causal true-mixed top-1 was
`0.8889`, below the longest-window baseline `0.9290`, and causal pure-negative
abstention was `0/3`.

The corrected infrastructure is nevertheless sealed and reviewable:

- strict window geometry: 507 true mixed episodes in FIT;
- causal anchor API and per-step scheduler replay: PASS;
- active heads: utility, release, regrasp;
- support and uncertainty heads: disabled;
- one-shot compliance: 100% in the 200-episode replay;
- checkpoint and prediction roots: independently auditable;
- protected splits and attack outputs: not read.

No 600-episode Fold-0 run is authorized by the recorded R2 decision.  The
next scientific decision is whether to stop proprio-only V5-A and recover a
policy-intent or causal-visual source.  Such sources are currently absent;
V5-B/C/D remain HOLD.
