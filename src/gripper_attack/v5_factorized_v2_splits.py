"""V2 inner-CV split resolver — single source of truth for train/val identity assignment.

Protocol: 3-fold identity-level CV.
  inner_val   = the specified inner fold (≈1/3 of outer_train)
  inner_train = the other two inner folds (≈2/3 of outer_train)
"""
from pathlib import Path
from typing import Any


def resolve_inner_train_val_ids(
    split_bundle: dict[str, Any],
    outer_fold: int,
    inner_fold: int,
) -> tuple[set[str], set[str]]:
    """Return (inner_train_ids, inner_val_ids) for a given outer/inner fold.

    inner_val   = identities in the specified inner fold
    inner_train = identities in the other two inner folds
    """
    fold_key = f"fold_{outer_fold}"
    fold_data = split_bundle["splits"][fold_key]
    inner_folds = fold_data["inner_folds"]

    inner_val_ids = set(inner_folds[inner_fold]["identities"])
    inner_train_ids = set()
    for i, inner in enumerate(inner_folds):
        if i != inner_fold:
            inner_train_ids.update(inner["identities"])

    return inner_train_ids, inner_val_ids


def get_outer_val_ids(
    split_bundle: dict[str, Any],
    outer_fold: int,
) -> set[str]:
    """Return outer validation identities (never used in inner-CV)."""
    fold_key = f"fold_{outer_fold}"
    return set(split_bundle["splits"][fold_key]["outer_val_identities"])


def validate_inner_split(
    split_bundle: dict[str, Any],
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    """Validate an inner split: no overlap, correct proportions, no contamination."""
    inner_train, inner_val = resolve_inner_train_val_ids(split_bundle, outer_fold, inner_fold)
    outer_val = get_outer_val_ids(split_bundle, outer_fold)

    issues = []
    overlap = inner_train & inner_val
    if overlap:
        issues.append(f"train/val overlap: {len(overlap)} identities")

    val_contamination = inner_val & outer_val
    if val_contamination:
        issues.append(f"inner-val contains outer-val: {len(val_contamination)} identities")

    train_contamination = inner_train & outer_val
    if train_contamination:
        issues.append(f"inner-train contains outer-val: {len(train_contamination)} identities")

    total = len(inner_train) + len(inner_val)
    expected = sum(len(inner["identities"]) for inner in split_bundle["splits"][f"fold_{outer_fold}"]["inner_folds"])
    if total != expected:
        issues.append(f"total identities {total} != expected {expected}")

    train_pct = len(inner_train) / max(1, total)
    if not (0.60 <= train_pct <= 0.75):
        issues.append(f"train fraction {train_pct:.2f} outside [0.60, 0.75]")

    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "inner_train_count": len(inner_train),
        "inner_val_count": len(inner_val),
        "outer_val_count": len(outer_val),
        "train_fraction": train_pct,
        "issues": issues,
        "valid": len(issues) == 0,
    }


__all__ = ["resolve_inner_train_val_ids", "get_outer_val_ids", "validate_inner_split"]
