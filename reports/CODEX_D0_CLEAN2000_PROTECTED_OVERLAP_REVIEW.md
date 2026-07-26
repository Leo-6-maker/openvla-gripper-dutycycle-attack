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
overlap is 1330. Cross-manifest overlap is 0. The nonzero Clean2000 overlap
is an immediate stop under the D0 contract; no split or capability result can
promote this input to a consumable FIT root.

The official identity manifest SHA256 is
`e6fdbfa7e81a150cfcf4a85ad300963d6b58dec5b84bf4334030f154315a3ffb`.
The source `clean/` directory has no accepted top-level SHA256SUMS pair, so
the source remains historical/derived metadata rather than an immutable
trajectory seal. The schema-only capability result is `PARTIAL`; no payload
was opened.

## Sealed custodian evidence

- Root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/d0_clean2000_517a957_20260727_0145`
- Custodian script SHA256:
  `2b0930877e10e4b2dbfcb8c7cc64a27fe00327e7cb54aeada6b4645deab3296e`
- `SHA256SUMS`: `b2590d31f995346a7642940eb983c9c67b60aaaaee5b22b13c6f902ab3f5be5c`
- `SHA256SUMS.sha256`: `73979333304dffdde70df7b3a681dd4c7b22df3d36a0ef2f12c8bb2f18587cf9`
- `D0_FEASIBILITY_RECEIPT.json`: `838b9387c57d5c924ce5cb1fdf6174b07420aa3d2a5f70c1195183d42bc2cbd7`
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
