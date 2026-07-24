# Compute Budget V1

Status: PLANNING_ONLY

This is a sizing placeholder, not execution authorization.

## Proposed Formal Matrix

```text
target 20 primary parents per suite
maximum 3 parents per task
minimum 5 eligible tasks where available
4 suites x 20 parents = up to 80 parents
```

Main formal conditions:

```text
Clean
Ours
RAND_DIRECTION
RANDOM_TIME
Adapted TMA-OPEN
```

Main matrix: 80 parents x 5 conditions = 400 suffix branches.

Seed robustness: 5 parents per suite, 4 attack conditions, 2 extra seeds =
160 suffix branches.

Mechanism subset: 5 parents per suite, 5 controls = 100 suffix branches.

Total formal suffix budget: 660, plus 80 prefix acquisitions.

## Budget Gate

Before execution, replace this placeholder with measured runtime, GPU memory,
disk output size, retry allowance, and storage headroom. Do not reserve GPUs or
start jobs from this document.
