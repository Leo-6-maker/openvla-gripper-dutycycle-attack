# Baseline Protocol V1

Status: PLANNING_ONLY

## Main Baselines

- Clean exact-prefix replay.
- RAND_DIRECTION with same timing, epsilon, K, prefix, and preprocessing.
- RANDOM_TIME with same payload, epsilon, K, prefix, and legal random window.
- Adapted TMA-OPEN with same victim, epsilon, K, prefix, preprocessing, and denominator.

## Mechanism Baselines

- SHUFFLED_GRADIENT.
- UNTARGETED_PGD / UMA-style PGD.
- EARLY_SHIFT.
- ARM_TARGETED.
- COMMAND_OPEN_ORACLE.

## Threat-Model Boundary

UADA/UPA/TMA-style inference-time visual attacks are empirical baselines.
FreezeVLA, backdoor, text, patch, and unrelated threat models belong in a
capability table, not an ASR leaderboard.
