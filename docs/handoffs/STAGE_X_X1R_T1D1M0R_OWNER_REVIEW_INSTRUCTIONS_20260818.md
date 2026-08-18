# T1-D1M0R owner review instructions

Use only the safe human-facing sheet and the fixed review material:

- `reports/STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.csv`
- `M001` through `M014` review clips and frame strips at the durable paths in
  the sheet
- the frozen D1M0 rubric

Review rows in order `M001` → `M014`. Do not open the private mapping, D1R
receipts, telemetry, Student scores, physical scores, or any attack/V_phys
artifact while labeling.

The sheet is sealed with blank `contact_label`, `reason_code`, `reviewer`,
`review_timestamp`, and `optional_short_note` fields. Do not change the task
instruction or clip/strip paths and hashes.

Use only the frozen labels:

```text
PASS    -> contact_label=PASS, reason_code blank
FAIL    -> contact_label=FAIL, exactly one frozen FAIL reason code
ABSTAIN -> contact_label=ABSTAIN, reason_code blank
```

For `FAIL`, use exactly one of:

```text
PRECONTACT_OR_APPROACH
WRONG_OR_IRRELEVANT_OBJECT_PART
RELEASE_ALREADY_STARTED
RELEASE_SAFE_OR_INDEPENDENTLY_SUPPORTED
CONTACT_ALREADY_LOST_OR_SLIPPING
OTHER_CLEARLY_NON_GRIPPER_DEPENDENT
```

Use `ABSTAIN` when the fixed material cannot support a defensible PASS/FAIL
judgment. An optional short note may explain the ambiguity; do not add a new
reason code.

After completing all 14 rows, make a new copy named
`STAGE_X_X1R_T1D1M1_OWNER_LABEL_SUBMISSION_V1.csv`, leave the sealed sheet
unchanged, and report the filled file's raw SHA256. Do not map labels to parent
identities, calculate a final eligible count, or run any attack. Return the
filled submission for a separate D1M1 review.

`Eval160` and protected evaluation remain unread and unauthorized.
