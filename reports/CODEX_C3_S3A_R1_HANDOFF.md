# Codex C3-S3A-R1 remediation handoff

## Decision

`C3-S3A-R1 = PASS`.

The main execution and an independent reviewer both passed. This is a synthetic,
FIT-independent geometry numerical-validity fixture. It is not Clean2000,
Teacher, Student, rollout, or attack evidence.

## Frozen code

- branch: `codex/detector-completion-20260726`
- commit: `fc18cd965d237c4d40ae60dc8f25be2d8dc98a29`
- tree: `fd50ed1350fa41032a5c9ec1118e064932f43334`
- server worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/codex-detector-completion-0b723`
- worktree state: clean, detached at the commit above
- Python: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`
- compile and targeted contract tests: `29 passed`, `0 failed`, `0 errors`

## Fresh sealed evidence

Dataset root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3a_r1/C3_S3_GEOMFIT_SYNTHETIC_V1_fc18cd9_20260727_0005`

- dataset `SHA256SUMS`: `11b523775d872103c379182c693873d974532165108b5f12c974e07fda684a04`
- allowlist SHA256: `44d7d723a9b66b97f253b0de05053da25167e1ac75650963bb9a5fad3ea109cf`
- dataset manifest SHA256: `59a6d36a1a746287274053178c7ab12cb91136913e8b424f43afa50a44896b41`

Evidence root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3a_r1/evidence_fc18cd9_20260727_0005`

| artifact | result | `SHA256SUMS` SHA256 |
|---|---|---|
| `smoke_static` | PASS, 11 relations | `9d9c63c78d624072a7390ef243f56848deb417d8059afdea358df158f9ffb290` |
| `smoke_dynamic` | PASS, 31 relations | `fc277e5c5579e7c261d916a4648f88ace2889513f4c9d237c79073d12e5ae88c` |
| `smoke_articulated` | PASS, 2 relations | `15100b0646a3be9036f7babbfe8b6717aef16dddbfd67952d3a977373d09472e` |
| `negative_translation` | PASS_NEGATIVE_CONTROL | `c1fc181432191d306a47ea07bfb1d680589731764e5208579f3e3c557da1b5bd` |
| `negative_rotation` | PASS_NEGATIVE_CONTROL | `684c7a7ce3924d3e6bac3436941c0cb2d3ac4a9d16caf401e652a4fef402e987` |
| `negative_local-transform` | PASS_NEGATIVE_CONTROL | `43d7b3f587dddd169b8f60dd414171df91e26709dea97709396c92261177b862` |
| `negative_qpos` | PASS_NEGATIVE_CONTROL | `1a9aed42f946b4128a62e294e1927331e6ed182765d24289bf40d68acdebc178` |
| `negative_joint-axis` | PASS_NEGATIVE_CONTROL | `53ddf9097ccc0dee3937de9b25d0eac555c3ece6dc132b15f8e87fd29eef1971` |
| `run_A` | PASS | `2261d6280c17c1d86f81a031a355c5e51821173af1a4ceb78fb2ef1c310e52b8` |
| `run_B` | PASS | `968cffb4345322ed524b69520991bd93b779b9ece1e88c3a8ae1d9c5e7ecd824` |
| `comparison` | PASS | `ade13282632c59923a1b97b804578e0bc1bb0ff08c109bf46e5236facfb978ab` |

Both full runs have canonical digest:

`d3886e048edb99eb0ae9e8a6315dff45df943c713414494140f3068f289b74fd`

## Contract result

- relation coverage: `44/44` (`11 STATIC / 31 DYNAMIC / 2 ARTICULATED`)
- exact source/reference steps: `110 static`, `3300 dynamic/articulated`
- articulated chain: `200/200` steps; qpos, axis, limits, qpos index and ancestor chain verified
- static position max: `0 m`
- static rotation max: `0 rad`
- dynamic position p99: `5.72e-17 m`
- dynamic rotation p99: `0 rad`
- unknown articulated: `0`
- source-only fault controls: all five rejected as expected
- A/B canonical: identical
- source mutation: `0`
- protected reads: `0`
- model inference, training, rollout and attack: not run

The source reconstruction and MuJoCo reference are separate computation chains;
the old shared `_step_poses`/`os.replace` path is not used by the new runner.

## Independent review

The independent reviewer used a separate implementation and directly recomputed
pose errors, seals, relation coverage, denominators, thresholds, articulated
chains, A/B equality, and a source-only translation fault. It reported:

`C3-S3A-R1 INDEPENDENT REVIEW = PASS`

No protected or historical experiment roots were read.

## Boundary

This PASS only releases the authorized C3-G-DEV stage. C3-T, Clean2000
relabeling, Teacher/Student training, CAL/CHECK, rollout and attack remain HOLD.
