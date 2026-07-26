# Codex C3-S3 Geometry Observability Handoff

## Gate

`C3-S3 GEOMETRY OBSERVABILITY = HOLD`

The sealed C1-V2 task registry and coordinate-transform contract are closed. The numerical per-episode geometry contract is not closed because the authorized input set does not contain a development/confirmation geometry episode manifest, and this audit does not treat a non-empty manifest as replay evidence.

## Source and execution

- Branch: `codex/detector-completion-20260726`
- Code commit: `210ff02b916a4dcd46899683c4120225f46735ae`
- Code tree: `9e9a2c30f9917e758bffb420589d38feaa4df7aa`
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- C1 source commit: `beb0721d36bd27412cde7d60623b8cb2f671a4bf`
- C1 source tree: `bb7ef11fd8329ae2e8bb71a0bb4c6c4caaf2f7c3`
- Protected reads: `0`
- Model inference / training / rollout / attack: `NOT STARTED`

## Sealed evidence

Root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3/c3_s3_geometry_observability_210ff02b_20260726_2115`

- `SHA256SUMS` digest: `675c48b026cd1e5cf4cceacee00f9edfebdcad8d803529ab10b6dad5176f4444`
- `SHA256SUMS.sha256` file SHA: `ee16753b70baa3929febea8a64779b97a2ccf93fff8469c801e35e2db7882cfb`
- Summary SHA: `a957e3985755c3cb67721b53e5d4887f15a80616384b44c53210254c1c7561a1`
- Canonical rebuild A: `329a37ceed3a0a485c725bef37c2fee1287aacdb74da265489c484229a9f91bc`
- Canonical rebuild B: `329a37ceed3a0a485c725bef37c2fee1287aacdb74da265489c484229a9f91bc`
- Root seal: `PASS` (`sha256sum -c SHA256SUMS` and sidecar check)

## Results

- C1 task closure: `40/40`
- Relation rows: `46`
- Supported relation rows: `44`
- Non-placement exclusions: `2`
- Static fixture rows: `11`
- Dynamic reconstructable candidate rows: `31`
- Articulated unknown rows: `2` (white cabinet bottom; wooden cabinet top)
- Mapping completeness: `PASS`
- Transform identity/±90-degree/composition-inverse tests: `PASS`
- Silent fallback: `0`
- Unknown converted to negative: `0`
- Independent canonical digest equality: `PASS`
- Development/confirmation episode rows: `0`
- Static numerical replay error: `NOT EVALUATED`
- Dynamic numerical replay p99: `NOT EVALUATED`

The 40 model initializations were metadata-only (`action_steps=0`); no simulator rollout or model inference was performed.

## Blocking condition

The C1-V2 registry stores entity names, IDs, parent relationships and resolution decisions, but not per-episode canonical states, world-frame geometry, step bindings, or reconstruction error measurements. Therefore it cannot prove the C3-S3 numerical thresholds. C3-G and all later gates remain blocked. No data from CAL, G10, or T2R-D was read.

