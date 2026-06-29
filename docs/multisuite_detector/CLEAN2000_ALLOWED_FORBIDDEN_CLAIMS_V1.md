# CLEAN2000 Allowed and Forbidden Claims V1

## Allowed Claims

1. "The Object-only frozen detector evaluates zero-shot transfer from LIBERO-Object to three unseen suites."
2. "Balanced pooled training improves supervised multi-suite interpolation relative to Object-only."
3. "LOSO evaluates held-out-suite generalization under identical architecture and feature contract."
4. "Window-level and episode-level metrics are reported separately; frame accuracy alone is insufficient."
5. "All checkpoints are selected on validation-suite metrics only; test-suite performance is reported without re-selection."
6. "The privileged teacher oracle provides an empirical upper bound for event detection under this feature contract."

## Forbidden Claims

1. "Mixed-suite training achieves high F1, therefore zero-shot generalization holds." (Confuses interpolation with generalization)
2. "High F1 on pooled test data proves cross-suite generalization." (Pooled test contains in-distribution suites)
3. "The mixed detector validates the current TRUE_T10 detector." (Different training data, different claims)
4. "Clean-success-only evaluation proves safe abstention." (Must evaluate on safety/abstention set)
5. "LOSO test-suite F1 > X proves the detector generalizes to all unseen manipulation tasks." (Only 4 suites tested)
6. "Because Object-only F1 is low on LIBERO-10, the TRUE_T10 held-out results are invalid." (Different detector, different evaluation)
7. "Balanced pooled detector should replace Object-only for TRUE_T10." (TRUE_T10 is pre-registered with frozen Object-only detector)

## Required Qualifiers

- All LOSO results must state: "Test suite was completely excluded from training, normalization, threshold selection, and early stopping."
- All Object-only results must state: "Checkpoint, normalization, and threshold are byte-identical to the SC5 Object detector used in TRUE_T10."
- All pooled results must state: "This is a supervised detector; it does not support zero-shot generalization claims."
