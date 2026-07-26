# D0-R2 Clean2000 DEV_POOL review

## Decision

```text
D0-R2 aggregate closure       = PASS
DEV_POOL                      = 670
protected union               = 1330
protected cross-overlap       = 0
schema-only capability        = REPLAY (670/670)
derived snapshot seal         = PASS
protected identity emission   = 0
protected payload emission    = false
```

The custodian ran from commit `c923b7b` in the official A800 Python
environment. It read the Clean2000 manifest and the two protected manifests
only for aggregate identity-set statistics. It did not emit protected
identity values or read protected episode payloads.

## Bound inputs

| Input | SHA256 |
|---|---|
| `OFFICIAL_CLEAN_2000_MANIFEST_V3.csv` | `e6fdbfa7e81a150cfcf4a85ad300963d6b58dec5b84bf4334030f154315a3ffb` |
| `G10_TEST_MANIFEST.json` | `06bd43fe5666b6bacb763e0ae7dbba12ef317911ba923b469dab8628cc8bb7ef` |
| `T2RD_CONFIRM_MANIFEST_V1.json` | `a126be308e932597e74247f260a219e9ee48c87b5b9716c1bc5750e118f749cc` |

The resulting sealed derived root is:

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/d0_r2_dev_pool_20260727_0240
```

Root `SHA256SUMS` SHA256:

```text
e4612df514e2953c691731b59e8cc080d0613ccb70eb93477dc32a5aa3c4ef1d
```

The root contains the non-protected complement manifest, suite/task counts,
capability audit, split plan, receipt, and both checksum files. Both
`sha256sum -c SHA256SUMS` and `sha256sum -c SHA256SUMS.sha256` passed.

## Aggregate checks

```text
Clean2000 unique                    = 2000
G10 ∪ T2R-D unique                  = 1330
Clean ∩ (G10 ∪ T2R-D)                = 1330
G10 ∩ T2R-D                          = 0
DEV_POOL unique                      = 670
DEV_POOL ∩ protected union           = 0
DEV_POOL ∪ protected union = Clean   = true
FIT_TRAIN ∩ protected                = 130
non-FIT Clean ∩ protected            = 1200
```

Per-suite counts are `libero_10=158`, `libero_goal=165`,
`libero_object=171`, and `libero_spatial=176`. The sealed
`DEV_POOL_PER_SUITE_TASK_COUNTS.csv` contains all 40 suite/task rows and sums
to 670.

## Capability boundary

The capability audit is schema-only. All 670 expected episode directories
were present, and all 670 had the required `episode_metadata.json` and
`step_records.jsonl`; the complete expected-file counts are sealed in
`CAPABILITY_AUDIT.json`. This is a replay/schema readiness result, not a V23
semantic-label result.

No model inference, Teacher materialization, Student training, rollout, CAL,
CHECK, or attack was run.

