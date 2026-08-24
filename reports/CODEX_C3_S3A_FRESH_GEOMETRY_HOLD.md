# C3-S3A fresh geometry numerical validity — independent review hold

## Final decision

```text
MAIN SYNTHETIC EXECUTION       = PASS
INDEPENDENT REVIEW              = HOLD
C3-S3A                         = HOLD_INDEPENDENT_REVIEW
C3-G-DEV                       = NOT STARTED
D0 Official V3 Clean2000       = HOLD
```

The main execution completed the authorized synthetic-only run, but the
independent review found unresolved contract issues. Therefore the combined
Gate is not PASS and no C3-G-DEV work is authorized in this round.

## Main execution evidence

Code snapshot:

`f47f7f2a6dc78b01dc41bd7a5f10a47d4f8d5c28`

Official environment:

`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

MuJoCo version: `3.9.0`.

Targeted official-environment tests: `26 passed`, `0 failed`, `0 errors`.

Synthetic dataset root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3a_synthetic/C3_S3_GEOMFIT_SYNTHETIC_V1_f47f7f2_20260726_2340`

- root SHA256SUMS digest: `31c8937ce057abc1b7a4da8ef492ebe275b961c96a0a6cd1f903cb1fd309acde`
- dataset MANIFEST SHA256: `449fc5481cf833235de6e587db32b858ec2fa31da42044a7f67c913db5fdcaec`
- dataset manifest SHA256: `ef201f9424cad9999434511107be2fdb7356b109dc87a7253babd4bfacb01f8e`
- allowlist SHA256: `d995fe6bce258bf1647626ddb2f9ad94597766693ed7fd180c6872ee846db1b2`
- relation plan: `44 = 11 STATIC + 31 DYNAMIC + 2 ARTICULATED`
- static minimum: `10` configurations/relation
- dynamic/articulated minimum: `100` samples/relation

Smoke roots:

| Smoke | Result | Root SHA256SUMS |
|---|---|---|
| static | PASS | `419030198d7ccab47df54fda5e2f5a8ef5df18a5b3c0a5c6a30597ba0a9dc5de` |
| dynamic | PASS | `e5d3161d294bfd8ecccc19da4387648b89e7e783eb3e25b34c6d56cfcd0e6050` |
| articulated | PASS | `4297d68938a04e5a34552c901ac769f44340ae82f042a97a101f865d935981bf` |

Full runs:

| Run | Result | Root SHA256SUMS | Canonical digest |
|---|---|---|---|
| `run_A` | PASS | `3f4be51b98a7c31ad25120556e7c218b59e21e5d9462f6fe6f42ef05b9134412` | `73fb11340a6ddd777dd903f005135dda7af2b21ada77436a74760143cf24862e` |
| `run_B` | PASS | `8e15abe515fae3898b0a8a0618e7248b89e7e783eb3e25b34c6d56cfcd0e6050` | `73fb11340a6ddd777dd903f005135dda7af2b21ada77436a74760143cf24862e` |

Comparison root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3a_synthetic/comparison_f47f7f2_20260726_2340`

- result: `PASS` for the implemented comparison
- comparison SHA256SUMS: `c3fa11d29a7294d8d85dd44b09a473c1c9de6e70ed99f4250a931ccf40c75bb9`
- canonical identical: `true`
- coverage identical: `true`
- metrics identical: `true`

Main-run numerical values:

- supported coverage: `44/44`
- static position denominator: `110`
- static rotation denominator: `110`
- dynamic position p99 denominator: `3300`
- dynamic rotation p99 denominator: `3300`
- static position max: `0.0 m`
- static rotation max: `0.0 rad`
- dynamic position p99: `5.594315114139762e-17 m`
- dynamic rotation p99: `0.0 rad`
- articulated unknown: `0`
- threshold violations: `0`

## Independent review blockers

The independent reviewer returned `HOLD` for the following reasons:

1. The two articulated relations are labeled `ARTICULATED` in the fixture
   plan, but their per-step reconstruction kind is `DYNAMIC`. Thus
   `unknown_articulated=0` does not yet constitute an explicit articulated
   reconstruction proof under the contract.
2. Source and reference records both derive their pose inputs from the shared
   `_step_poses()` helper. Different declared method/code strings are not
   sufficient to prove independent computation chains when the numerical
   inputs share the same generator.
3. The sealing helper checks `final.exists()` before `os.replace`; a concurrent
   creator can still introduce a clobber race. Strict no-clobber semantics are
   not closed.
4. `compare_runs()` validates run seals and equality but does not independently
   revalidate the allowlist and denied-root closure.
5. The current targeted tests do not cover the complete synthetic smoke,
   run_A/run_B, comparison, and seal boundary as an independent contract.

These are implementation/contract holds, not evidence of Clean2000 damage.

## Boundary declaration

- Clean2000 payload read: `0`
- protected/history episode root mounted: `0`
- OpenVLA loaded: `0`
- policy rollout: `0`
- Teacher labeling: `0`
- Student training: `0`
- attack: `0`
- source artifact mutation: `0`
- C3-G-DEV: `NOT STARTED`

All synthetic roots and failed/held review evidence are retained. No further
Gate was entered after the independent HOLD.
