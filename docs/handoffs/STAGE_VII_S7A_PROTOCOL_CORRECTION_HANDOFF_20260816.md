# Stage VII S7-A protocol correction

The first S7-A engineering run at
`STAGE_VII_S7A_CANDIDATE_DEVELOPMENT_20260816T080303Z` is preserved but is
`NONCONSUMABLE_FOR_STAGE_VII_PROMOTION`.

Reason: its runner implemented direct class-weighted BCE but omitted two
requirements from the frozen Stage VII candidate contract:

- within-parent pairwise ranking auxiliary loss;
- suite-balanced and parent-balanced sampling.

That run's negative result must not be used to seal the Stage VII scientific
decision. The corrected runner now binds:

- small multidose causal TCN;
- direct T3/T5/T10 V_phys supervision;
- class-balanced BCE;
- within-parent, same-dose pairwise ranking loss with frozen weight `0.25`;
- deterministic inverse suite/parent/row weighting;
- fixed threshold `0.69` and abstains masked as non-negatives.

The earlier decision root remains immutable historical engineering evidence but
is superseded for scientific promotion by the corrected prospective run.
No fresh M4, timing matrix, Eval160 read, or protected read occurred.
