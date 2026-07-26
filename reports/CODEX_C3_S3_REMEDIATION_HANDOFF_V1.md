# Codex C3-S3 remediation handoff

## Current gate

`C3-S3 MAPPING = PASS; C3-S3 NUMERICAL OBSERVABILITY = HOLD`

The remediation now has an explicit input contract. Only the sealed H0.3-R6
task registry is allowlisted. No FIT-only episode geometry telemetry root with
an independent world-frame reference is currently mounted, so the numerical
replay gate remains fail-closed.

## Code and environment

- Branch: `codex/detector-completion-20260726`
- Code commit: `f34c1878bc6bc2da943ad17d1b6d06a8676b0e87`
- Tree: `0910b6fd6728f1320e033a89a87f3c31216f27c8`
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- C1 source commit: `beb0721d36bd27412cde7d60623b8cb2f671a4bf`
- Protected reads: `0`
- Model inference, Student training, rollout and attack: `NOT STARTED`

## Input inventory

Allowlist:

`configs/C3_S3_ALLOWED_INPUTS_V1.json`

- Allowlist SHA256: `5440d43a8f54ea665beb316d4a959e05841590a55f97a51a771fcd5e99445af9`
- Allowed C1/R6 root: sealed, `ARTIFACT_MANIFEST.json` SHA
  `8f1b6d890b21a54206e1dd4f6606901e2441b5c537e9ff6c3ba4a89d0bcea2b0`
- Allowed episode geometry roots: `0`
- Candidate C2F root was inspected only at top-level metadata. It exposes
  `logs/` and `shards/`, has no accepted top-level seal, and is not Official
  V3 trajectory-bound; it is rejected as a C3-S3 input.
- Official all-state clean and historical T2R/CAL/G10/CHECK paths remain
  denied by contract and were not read for semantics.

Inventory evidence root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3_input_inventory_f34c187_20260726_2345`

- `SHA256SUMS` digest: `9790e4e7927335e4d48a6c8cd8a22fa05fb940277768b80d4775e212179767c7`
- `inventory.json` SHA256: `ba7ffa60298abdce13ad65a512ce40ef8586274adf9e759eb9bb7cc21f9c0fca`
- Status: `HOLD_INPUTS_MISSING`

## C3-S3 evidence

Current metadata-only evidence root:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3/c3_s3_geometry_observability_660688a_20260726_2300`

- `SHA256SUMS` digest: `0033a4041d32aaaf62c3340e2cc442c11d5152673a5b64ebfd1a98d797ff5dff`
- `SHA256SUMS.sha256` file SHA: `7806eaf88f47f473e590d38a23d09803997602b0895f4988ae78dc32b50aed81`
- `MANIFEST.json` SHA256: `d7269e954f0c18b5a49c3185030871c34e3a7ff27faf9e8f4751993ecd488ef9`
- `summary.json` SHA256: `ddd24069aebb3d1cfb2777cadf8956dc8d6deb74bcb2b53114570b8fe0d5dd13`
- Canonical rebuild A/B: `10d80dc380de34b4750dd5cac820d5e6584dabb0663c6e74c262e5e2dc8ea25f`
- C1 relation rows: `46`; supported: `44`; non-placement: `2`
- Articulated unknown rows: `2` (white cabinet bottom; wooden cabinet top)
- Episode rows: `0`
- Static/dynamic numerical replay: `HOLD_INPUTS_MISSING`
- Root seal: `PASS`

The two articulated rows remain unknown. They are not converted to negative
and do not silently reduce the supported denominator.

## Implemented contract

- Explicit root allowlist with manifest SHA binding.
- Exact path and regular-file checks; symlink components and denied roots fail
  closed.
- Exact episode, step `0..T-1`, and entity joins; duplicate/missing/misaligned
  records fail closed.
- Static and dynamic parent-pose × local-pose reconstruction.
- Quaternion sign equivalence and geodesic error
  `2*acos(abs(dot(q_pred,q_ref)))`.
- Explicit p99 method and denominator.
- Independent reference-chain requirement; no Teacher/event/outcome fallback.
- Staging plus atomic rename and sealed output roots.

## Gate and boundaries

`C3-S3 = HOLD` because no allowlisted episode telemetry/reference root exists.
`C3-G = BLOCKED`. No protected split semantics were read. No model inference,
training, rollout, shadow, or attack was started.
