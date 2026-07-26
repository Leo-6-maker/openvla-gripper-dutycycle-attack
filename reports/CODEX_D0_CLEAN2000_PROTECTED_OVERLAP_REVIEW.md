# D0 Clean2000 allocation review

## Decision

```text
D0                         = HOLD_PROVENANCE
protected overlap         = NONZERO (aggregate count 1330)
Clean2000 identity closure = PASS (2000/2000)
schema-only capability     = PARTIAL
V23 split freeze           = NOT CREATED
derived snapshot           = NOT CREATED
40-episode Teacher pilot  = NOT STARTED
```

The machine custodian read the Official V3 identity manifest and the two
pre-registered protected manifests only to compute aggregate identity-set
statistics. No protected identity value or manifest content was emitted into
the receipt or report.

## Input and aggregate results

| Input | Raw | Unique | Duplicate rows | Conflicting identities | Malformed |
|---|---:|---:|---:|---:|---:|
| Official Clean2000 manifest | 2000 | 2000 | 0 | 0 | 0 |
| G10 protected manifest | 1200 | 1200 | 0 | 0 | 0 |
| T2R-D protected manifest | 260 | 130 | 130 | 0 | 0 |

Aggregate protected union size is 1330. Clean2000/protected aggregate
overlap is 1330. The same custodian reports FIT_TRAIN/protected overlap 130
and non-FIT Clean2000/protected overlap 1200. Cross-manifest overlap is 0.
The nonzero Clean2000 overlap (and independently the nonzero FIT overlap) is
an immediate stop under the D0 contract; no split or capability result can
promote this input to a consumable FIT root.

The official identity manifest SHA256 is
`e6fdbfa7e81a150cfcf4a85ad300963d6b58dec5b84bf4334030f154315a3ffb`.
The source `clean/` directory has no accepted top-level SHA256SUMS pair, so
the source remains historical/derived metadata rather than an immutable
trajectory seal. The schema-only capability result is `PARTIAL`; no payload
was opened.

## Sealed custodian evidence

- Root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/d0_clean2000_517a957_20260727_0200`
- Custodian script SHA256:
  `530e9c75711ad0d55a1f82dc9a9909abaa13aac64fc6826408f2564cc826b6ad`
- `SHA256SUMS`: `5ed9a7d2f71a05806d94cca2c586368cfc035b136e78726c121d15f57568d8b8`
- `SHA256SUMS.sha256`: `2568f06ca6797654e40a1f29b18b656c8534d0028ff9ff9bdfc1f63cf3674138`
- `D0_FEASIBILITY_RECEIPT.json`: `0385da541d4ef7762119cc131f823980e883900478b56a20dcd0610f93b6dd96`
- Protected identity values emitted: `0`
- Payload read: `false`
- Model inference/training/rollout/attack: `false`

The failed earlier custodian attempt is retained as historical evidence and
is not used for the gate. The valid second receipt is the one cited above.

## Required stop

Do not freeze task-stratified V23 splits, create a derived data snapshot,
read protected payloads, or run the 40-episode FIT_DEV Teacher pilot until a
new authorized identity allocation resolves the aggregate overlap and closes
the source/capability contract.
