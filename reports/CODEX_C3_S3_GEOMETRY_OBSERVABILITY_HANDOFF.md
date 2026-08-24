# Codex C3-S3 Geometry Observability Handoff

## Gate

`C3-S3 GEOMETRY OBSERVABILITY = HOLD`

The sealed C1-V2 task registry and coordinate-transform contract are closed. The numerical per-episode geometry contract is not closed because the authorized input set does not contain a development/confirmation geometry episode manifest, and this audit does not treat a non-empty manifest as replay evidence.

## Source and execution

- Branch: `codex/detector-completion-20260726`
- Code commit: `3ce6dac486f1c9c91b5246991a97bd67b6e55c61`
- Code tree: `f038d4dd8c800455abffd43c18eb6113177f8566`
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- C1 source commit: `beb0721d36bd27412cde7d60623b8cb2f671a4bf`
- C1 source tree: `bb7ef11fd8329ae2e8bb71a0bb4c6c4caaf2f7c3`
- Protected reads: `0`
- Model inference / training / rollout / attack: `NOT STARTED`

## Sealed evidence

Root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3/c3_s3_geometry_observability_3ce6dac4_20260726_2145`

- `SHA256SUMS` digest: `391772eda1ebc693b391628ceaae6cb0f8b29017cf9eded7a072f7a58b0e4c2d`
- `SHA256SUMS.sha256` file SHA: `adc43e398362c455c0f2349bc737ad6b068be340983e509a378cf13ac83f306b`
- Summary SHA: `3eeae73ea4eddcf6c99e169a89a5173e138a6db7e80944e003a60f0ed7297946`
- Canonical rebuild A: `eecd565eff524c2df51b7a97c900bd6967b35e3559430bbf2d1bc88957dbac32`
- Canonical rebuild B: `eecd565eff524c2df51b7a97c900bd6967b35e3559430bbf2d1bc88957dbac32`
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
