"""V5-R2 development entrypoint.

The implementation intentionally reuses the guarded FIT-only smoke runner;
the new entrypoint exists so stratified development runs are not confused
with the historical 32-episode smoke.
"""

from train_v5_proprio_fit_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
