# Codex C3-S3 I2-R1 provenance recovery hold

## Decision

`C3-S3-I2-R1 = HOLD_PROVENANCE`

The authorized R1 work stopped after the first unmet provenance condition. No
I3 input recovery, I4 numerical replay, or C3-G work was started. No model
inference, Student training, Full-FIT, rollout, or attack was run.

Code snapshot used for the corrected metadata-only allocation audit:

`67759b86dccc5394c9aaa7b9a664bdb88e915993`

## Correct Clean2000 source

The source identified by the Official V3 metadata is:

`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean`

The identity manifest is:

`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/manifests/OFFICIAL_CLEAN_2000_MANIFEST_V3.csv`

- manifest SHA256: `e6fdbfa7e81a150cfcf4a85ad300963d6b58dec5b84bf4334030f154315a3ffb`
- sidecar matches the manifest: `true`
- manifest sidecar file SHA256: `b2a4441fc63eaa3166ec9c75a9514e249e9248327129434d4a649d7a3b9e9659`
- source provenance SHA256: `506cf28ad5ebedaa96ebd10de0a166b7c77458e612a55df92b1b7189d1d49291`

The corrected audit used the official identity manifest only. It observed:

- raw records: `2000`
- unique identities: `2000`
- duplicates: `0`
- conflicts: `0`
- malformed rows: `0`
- each suite: `500`
- each task: `50`

The Official V3 horizon contract is suite-specific:

| Suite | Contract horizon |
|---|---:|
| `libero_10` | 520 |
| `libero_goal` | 300 |
| `libero_object` | 280 |
| `libero_spatial` | 220 |

These are protocol horizon values, not observed per-episode step counts. The
identity manifest contains no episode payload or actual `n_steps`, so no claim
about uniform episode length is made here.

## R1A — identity closure

`PASS`.

The updated auditor no longer uses `setdefault` to fold duplicate identities.
It emits raw, unique, duplicate, conflict, malformed, per-suite, per-task and
per-worker counts. The Official V3 manifest closed at 2000/2000 with all error
counts equal to zero.

Sealed metadata-only allocation root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/clean2000_allocation_official_v3_67759b8_20260726_2355`

- `SHA256SUMS` file SHA256: `be7442cc7d9fbb6348e77d4d4a4904c50f7591d2d1a2cd77cdd30890e16a07bd`
- `SHA256SUMS.sha256` file SHA256: `f445c787fffb6cc87bb1acf0ddda0f65a6fa96bc1b2de5b7d28a688fc5441112`
- sidecar content matches `SHA256SUMS`: `true`
- `summary.json` SHA256: `6d1191cac640b3e9c2fa95db6afc6c91bcccb61bb923371ff4eb3482b7c11803`
- `allocation_ledger.csv` SHA256: `39328393895a9815ccffe3f27a58faa9529278dc9865a8712aaebd22eb83ad1e`
- root `sha256sum -c SHA256SUMS`: `PASS`
- staging residue in final root/parent: `0`

## R1B — source binding

`HOLD_DERIVED_METADATA_ONLY`.

The Official V3 provenance record binds the following metadata:

- generation/source commit: `943b02749dce4414ec6791b15ceec87dbd3be1ba`
- generation/source tree: `d484dc74bbc9c457371537b89afa725a8c71957c640eacb63c15a58e27571e4c`
- runtime environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- upstream OpenVLA commit: `c8f03f48af692657d3060c19588038c7220e9af9`
- upstream LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- collector source SHA map: present in the sealed provenance metadata

The binding is not an original immutable raw-campaign seal because:

1. the Official V3 clean source directory has no accepted top-level
   `SHA256SUMS`/sidecar;
2. no generation command is present in the allowed metadata;
3. episode payloads were not opened or hashed;
4. no full materialization/NPZ SHA binding was available;
5. the historical `363,513` window count was not recomputed in this run.

Accordingly the result is explicitly `DERIVED_OFFICIAL_V3_METADATA_ONLY`,
not original immutable provenance.

The historical C2F parallel root was not substituted. It uses a different
collector/source lineage (`36712cc...`), has no accepted top-level seal, and
its historical metadata reports a different step geometry (maximum 300).
It is retained only as a rejected historical reference.

## R1C — derived snapshot

`NOT_EXECUTED_BLOCKED_BY_R1B`.

The old roots were not modified. Because R1B did not establish the required
source binding, no new `DERIVED_SEALED_SNAPSHOT` was created.

## R1D — V22-800 intersection

`HISTORICAL_NON_CONSUMABLE`.

No exact V22-800 identity manifest was found in the allowed non-protected
metadata roots. A historical G4 receipt references an 800-episode label
production, but its identity source is not mounted and it is not a complete
accepted V22 identity root. Therefore exact intersection was not computed.
Clean2000 was not invalidated merely because the V22 manifest was absent.

## R1E — FIT-only and protected-overlap proof

`HOLD`.

An explicit FIT-only identity proof and a protected-overlap-zero receipt were
not available. The main authorized audit did not read protected semantic or
episode payload content and did not compute protected overlap. During a
delegated audit, two protected manifest file SHA values were read for boundary
checking only:

- `g6_training_seal/G10_TEST_MANIFEST.json`
- `t2rd_confirmation_cohort/T2RD_CONFIRM_MANIFEST_V1.json`

No protected episode payload, labels, or semantic content were read, but this
incident means the overall round must not claim `protected_reads = 0`. The
protected SHA reads are also not a substitute for the required overlap receipt.

## Gate matrix

| Gate | Result |
|---|---|
| C3-S3-I2-R1A identity closure | `PASS` |
| C3-S3-I2-R1B source binding | `HOLD_DERIVED_METADATA_ONLY` |
| C3-S3-I2-R1C derived snapshot | `NOT EXECUTED` |
| C3-S3-I2-R1D V22 intersection | `HISTORICAL_NON_CONSUMABLE` |
| C3-S3-I2-R1E FIT/protected proof | `HOLD` |
| C3-S3-I2 overall | `HOLD_PROVENANCE` |
| C3-S3-I3 input recovery | `NOT STARTED / NOT AUTHORIZED` |
| C3-S3-I4 numerical replay | `NOT STARTED / NOT AUTHORIZED` |
| C3-G | `BLOCKED` |

Because R1B, R1D and R1E are not PASS, no I2 review packet was generated.

## Boundary declaration

- protected semantic/payload reads by the main run: `0`
- protected metadata SHA incident in delegated audit: `2 files`
- protected overlap computation: `NOT PERFORMED`
- model inference: `NOT STARTED`
- Student training: `NOT STARTED`
- Full-FIT: `NOT STARTED`
- CAL/CHECK: `NOT READ`
- rollout: `NOT STARTED`
- attack: `NOT STARTED`
- source artifact mutation: `0`
