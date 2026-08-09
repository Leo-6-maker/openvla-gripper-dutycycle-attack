# C3-S3-I2 FIT input census

Status: `HOLD_PROVENANCE`

Code snapshot used for the metadata-only census: `5a750fdf6a2a64c4496626a7fb2f769f2db9b368`.

## Sealed outputs

- Existing R6/input inventory root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3_input_inventory_dd9af2e_20260726_2240`
  - `SHA256SUMS` SHA256: `c6fd38ce678830b465ca4757831802723e7b292f802bb459c5a48532d3a0855f`
  - `inventory.json` SHA256: `a210e2261234767f6c52ee657bb047d7833c41c5f9d4ed05ac099a31f9a9ab7e`
- Clean2000 metadata-only allocation root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/clean2000_allocation_5a750fd_20260726_2315`
  - `SHA256SUMS` SHA256: `f31f22b427b141f9909f74bbf1a9c2e6779bac867b593693cc37bfb08669fb4e`
  - `SHA256SUMS.sha256` SHA256: `666d0e171432e2013bbc6b0678f043b0261587d04ead6906743a7d789ecb5464`
  - `allocation_ledger.csv` SHA256: `cede0e735950e5a07dd7ccfeb3dd7cf733f3ec37431a8a1ee6fd14c46e8d3474`
  - `summary.json` SHA256: `feffc793d3bf8b5d117c61df1dd96f7f4e9037ee06e2c304efdc819ab1bed6ff`

## Observed metadata

The C2F Clean2000 root contains 12 worker shards (three per suite) and 12 `worker_manifest.jsonl` files. Reading those manifests only, without opening logs, shards, episode artifacts, or labels, yielded 2,000 identity records. The campaign root itself has no accepted top-level `SHA256SUMS`/sidecar seal and is not Official V3 trajectory-bound.

The allocation ledger therefore records all data capability and split/protected fields as `UNKNOWN_*`; it does not infer that the 2,000 identities are FIT, V22, train, validation, test, or replayable. The claimed 2,000/800 counts are preserved only as explicitly unaudited user claims.

No explicit sealed V22-800 identity root was mounted or supplied in this phase. Consequently overlap with V22, historical split allocation, protected status, and V23 labelability are not established.

## Gate decision

```text
C3-S3-I1                         = PASS
C3-S3-I2                         = HOLD_PROVENANCE
Clean2000 identity metadata      = 2000 observed, payloads not read
C2F campaign top-level seal      = MISSING
V22-800 root                      = NOT MOUNTED
FIT-only proof                   = NOT ESTABLISHED
C3-S3-I3                         = NOT STARTED
C3-S3-I4                         = NOT STARTED
CAL / G10 / T2R-D                 = NOT READ
MODEL INFERENCE / TRAINING        = NOT STARTED
ROLLOUT / ATTACK                  = NOT STARTED
```

Under the frozen gate sequence, `HOLD_PROVENANCE` stops C3-S3 here. No geometry replay or collection is authorized until an explicit FIT-only, V23-compatible source/reference allocation is sealed and mounted.
