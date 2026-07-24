# Statistical Analysis Plan V1

Status: PLANNING_ONLY

## Primary Contrasts

- Ours vs RAND_DIRECTION.
- Ours vs RANDOM_TIME.
- Ours vs Adapted TMA-OPEN.

Ours vs Clean is an attack-effect baseline, not part of the primary
multiple-testing family.

## Binary Outcomes

Report numerator, denominator, paired risk difference, 95% cluster bootstrap
CI, exact McNemar test, and Holm correction.

## Continuous Outcomes

Use paired bootstrap or paired nonparametric tests for open duty, qpos/width
response, arm NAD, actual Linf, and runtime.

Cluster bootstrap units are task and parent. Perturbation seeds are not
independent task samples.

## No-Emit Rule

Detector no-emission stays in ITT:

```text
detector no emit -> attack not executed -> parent outcome retained
```

Emitted-only results are auxiliary.
