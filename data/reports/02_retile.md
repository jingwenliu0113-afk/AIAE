# CP-SAT re-tiling benchmark

- structures sampled: 150 (seed 0)
- time limit per solve: 10.0s
- total solves: 1129
- wall clock: 234s
- exact-cover verification failures: **0**

## Feasibility by dropped part

| dropped | solves | feasible | median s | p95 s | max s | median bricks in->out |
|---|---:|---:|---:|---:|---:|---|
| `1x1` | 148 | **9%** | 0.01 | 0.21 | 0.46 | 65 -> 53 |
| `1x2` | 150 | **100%** | 0.11 | 0.54 | 2.56 | 92 -> 85 |
| `1x4` | 138 | **100%** | 0.11 | 0.88 | 5.66 | 97 -> 83 |
| `1x6` | 137 | **100%** | 0.12 | 0.52 | 1.00 | 96 -> 82 |
| `1x8` | 128 | **100%** | 0.12 | 0.44 | 0.90 | 95 -> 81 |
| `2x2` | 147 | **100%** | 0.12 | 0.61 | 0.97 | 91 -> 81 |
| `2x4` | 137 | **100%** | 0.11 | 0.53 | 0.94 | 95 -> 81 |
| `2x6` | 144 | **100%** | 0.12 | 1.29 | 10.35 | 94 -> 91 |

## Overall

- feasible: **88.0%** of 1129 solves
- statuses: {'INFEASIBLE': 135, 'OPTIMAL': 994}

- hit the 10.0s limit: **0**
