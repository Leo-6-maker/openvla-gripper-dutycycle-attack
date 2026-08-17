# STAGE X X1R T0 — Existing PGD Alignment Review

Status: `SEALED_BEFORE_CODE_CHANGE`

This matrix is the first T0 artifact. It was sealed before adding the T0
audit script, protocol, or tests. The task remains read-only/clean-only:
actual PGD optimization, attacked images/actions, `env.step`, V_phys reads,
new M4, X2, timing matrix, Eval160, and protected evaluation are forbidden.

## Current finding

The existing repository contains a real token-prefix visual PGD path:

`processor pixel_values -> frozen victim loss -> fp32 projected sign update ->
model-dtype cast -> OpenVLA re-decode`.

The canonical upstream action tokenizer is not the same as the project helper.
The official `ActionTokenizer` uses 256 bin edges and `np.digitize`, while
`TokenPrefixPGDAttacker.action_to_token_ids()` uses nearest-neighbor matching
against 255 bin centers. This creates an endpoint mismatch without implying
that the PGD gradient primitive itself is wrong.

For the already observed `31744`/`31745` rows, the current evidence supports
this narrower statement only:

- `31744` is the native endpoint token emitted by the checkpoint's canonical
  digitization for the upper endpoint;
- the project helper maps the decoded last bin center back to `31745`;
- both decode to numerically close OPEN actions under the model decoder;
- token IDs are not equivalent, and historical direct generated token IDs are
  not recoverable from the old artifacts.

The final root-cause label remains pending the complete T0 differential census,
but the source-level mechanism is already identified as a tokenizer authority
defect, not a license to relax token parity.

## Evidence boundary

Current parity runtime: `fa42bf5244984f6a47eb91922d38fbbb91cd16c4`, tree
`a1cb0496b4265ec2100b610459d0e982fdac26b9`.

Canonical upstream tokenizer SHA256:
`fdc98fcbf5b0926ef2181db71946d23ffbfa052cf8443dc933d52c42a191352c`.

All historical Stage V/Stage VI-B2 launch-time victim weights remain
`NOT_IDENTIFIABLE`. Current checkpoint hashes are prospective only.

## Allowed next work

1. Run the requested CPU/static differential census against the exact
   checkpoint-local/native tokenizer for all four suites.
2. Run clean teacher-forced causal-row checks and synthetic CW/Linf/dtype
   checks only.
3. Run the existing eight Q00 rows through fresh clean forward processes only,
   with no PGD or environment steps, to test token determinism.
4. Seal T0 reports and publish a stacked Draft PR for owner/GPT review.

No versioned helper repair or future X1R authority is issued by this review.
