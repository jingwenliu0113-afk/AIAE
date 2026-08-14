# F-oracle: tiling with the shape already known

**This is an oracle upper bound, not a deployable method and not a fair end-to-end comparison against arms A-E.** The target voxel shape is read from the test reference build. Nothing predicts it: no retrieval, no train-side index, no generative model. A real system has to recover the shape from the caption, and deleting that step is the whole point of an oracle -- it bounds what the optimisation stage can do when the stage before it is perfect.

Using a test-split reference shape is the definition of this arm, stated here so it is never mistaken for a leak or for a product result. The intended reading is the gap between F-oracle and F-pipeline, which measures what shape acquisition costs. F-pipeline is not built yet, so that gap is not available and this number stands alone.

## Setup

The environment the **solve** ran in. On a re-render this is the stored run's environment, not the machine that rebuilt the page.

- `platform`: macOS-26.6.1-arm64-arm-64bit-Mach-O
- `machine`: arm64
- `python`: 3.13.9
- `ortools`: 9.15.6755
- `solver`: CP-SAT (ortools)
- `time_limit_seconds`: 10.0
- `seed`: 0
- `workers`: 1
- `objective`: minimise brick count subject to exact cover + part budget
- `shape_source`: test reference build (oracle; nothing predicts it)
- `retrieval_used`: False
- `model_used`: False
- `split`: test
- `source_file`: data/processed/counterfactual_test.jsonl
- `source_sha256`: 917b4386f21c469b8c68248044ae4387e03c43c8bccd66b7b1fc67a53c4bdd62
- `task_signature`: e25fcc7bdbb6d6e1ea24c83db6ebebc3d39f425241e6ed384cf8917ba52404aa
- `wall_seconds`: 2770.1

Rendered 2026-08-13T23:41:43+08:00 on Python 3.13.9 / OR-Tools 9.15.6755. A re-render is refused unless these match the solve, so the two agree here by construction rather than by assumption.

### Provenance: some fields were backfilled, not measured

**`provenance_backfilled: true`.** This run predates part of the record-keeping around it, so the fields below were established afterwards by checking inputs that had not changed. They are reconstructions and are labelled as such -- none of them is a measurement taken while the solver ran.

| field | how it got here |
|---|---|
| `platform` | written by the run itself, during the solve |
| `machine` | written by the run itself, during the solve |
| `solver` | written by the run itself, during the solve |
| `time_limit_seconds` | written by the run itself, during the solve |
| `seed` | written by the run itself, during the solve |
| `workers` | written by the run itself, during the solve |
| `objective` | written by the run itself, during the solve |
| `shape_source` | written by the run itself, during the solve |
| `retrieval_used` | written by the run itself, during the solve |
| `model_used` | written by the run itself, during the solve |
| `split` | written by the run itself, during the solve |
| `source_file` | written by the run itself, during the solve |
| `wall_seconds` | written by the run itself, during the solve |
| `source_sha256` | **backfilled afterwards** |
| `task_signature` | **backfilled afterwards** |
| `python` | **backfilled afterwards** |
| `ortools` | **backfilled afterwards** |

The solve ran 2026-08-13 on this machine and venv. `source_sha256` and `task_signature` were recomputed from inputs shown unchanged: the source file's digest still equals the one report 04 recorded when it was generated, its mtime (19:03) predates the run, and all 1,600 stored reference brick counts match freshly derived tasks. `python` and `ortools` were read from the venv, whose installs (2026-07-21 and 2026-08-13 14:03) both predate the solve, so no version change could have intervened. All four are reconstructions from evidence, not readings taken during the solve.


## Unit of evaluation

A pair contributes 2 roles x 4 inventory framings = 8 rows, and the two roles are two tilings of **one** shape. A per-row mean therefore weights every shape eight times, and duplicated geometry more still. All four units are reported rather than one being picked; the headline is `unique_task`, the finest unit that is not a duplicate of another. Grouped units count only if every row in the group does.

The two connectivity columns have **different denominators** and are not interchangeable. The yield is over every unit, so a timed-out task counts against it -- it is an end-to-end number, not a property of the tilings. The conditional rate is over only the units that fully succeeded, and is the one that describes the tilings.

| unit | n | all accepted | rate | solved-and-connected | yield (/n) | all connected given all accepted | rate | multiplicity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sample` | 1600 | 1496 | 93.50% | 1124 | 70.25% | 1124/1496 | 75.13% | {1: 1600} |
| `unique_task` | 1476 | 1384 | 93.77% | 1034 | 70.05% | 1034/1384 | 74.71% | {1: 1352, 2: 124} |
| `pair` | 200 | 153 | 76.50% | 69 | 34.50% | 69/153 | 45.10% | {8: 200} |
| `unique_geometry` | 178 | 136 | 76.40% | 58 | 32.58% | 58/136 | 42.65% | {8: 156, 16: 22} |

## What acceptance means

A run is accepted when all four hold, each re-derived from the returned bricks rather than trusted from the solver:

- `voxel_exact` -- placed cells equal the reference occupancy exactly
- `collision_free` -- no cell covered twice
- `within_inventory` -- no canonical part exceeds its quantity (`4x1` counts against `1x4`)
- `parts_legal` -- every part is in the 8-part vocabulary

**Stud connectivity is not an acceptance condition here.** The model minimises brick count subject to exact cover and the budget; it has no connectivity constraint, so a minimal tiling of a connected shape can fall into pieces. That is a limit of the formulation, not an infeasibility, so it is reported as its own rate above and never folded into acceptance. Connectivity-aware tiling remains future work.

## Feasibility is 100% by construction -- the acceptance rate is not

Every variant's stock is a superset of what its own reference build used, so that build is a witness: a solution always exists. The run bears this out with **0 INFEASIBLE**. A high acceptance rate here restates the setup and must not be quoted as an achievement.

What the acceptance rate does measure is the *solver budget*. All 104 failures are timeouts at the 10.0s limit -- the search ran out of time on a problem known to have an answer. Raising the limit would raise the rate, so this number describes this configuration, not the difficulty of the task. The informative results are below: status, cost, and how often a solved output comes apart -- reported for all accepted tilings, and separately for the `OPTIMAL` subset, which is the only one whose members are actually minimum-brick tilings.

## Solver status

| status | n |
|---|---:|
| `OPTIMAL` | 1399 |
| `FEASIBLE` | 97 |
| `INFEASIBLE` | 0 |
| `UNKNOWN` | 104 |

- solved (solver returned a tiling): 1496/1600
- accepted (solved **and** passed all four checks): 1496/1600
- solved **and** stud-connected, reported separately and never folded into acceptance: 1124/1600

| failure | n |
|---|---:|
| timeout | 104 |

## Bricks against the reference build

The oracle cannot beat the reference and does not: the reference tiling was itself produced by the same minimum-count objective on the same shape, with *more* freedom (no part budget). The oracle solves a constrained version of an already-minimal problem, so matching is the best available outcome and the interesting question is how often it falls short.

- proved optimal and matched the reference exactly: **1399/1399** of all `OPTIMAL` runs
- used more bricks than the reference: 44, of which 44 are `FEASIBLE` (the search was cut off before it proved a minimum, not a worse optimum)
- used fewer: 0 -- as expected, this cannot happen

| min | median | p95 | max | fewer | equal | more |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 22 | 0 | 1452 | 44 |

## The finding: a proved minimum is often not one structure

Each figure below names its denominator, because the same count over a different population means a different thing.

| figure | value | what it is |
|---|---:|---|
| solved-and-connected yield | 1124/1600 = **70.25%** | end-to-end over every task attempted; timeouts count against it |
| connectivity among successes | 1124/1496 = **75.1%** | over accepted tilings only -- the rate that describes the tilings |
| per-geometry yield | 58/178 = **32.6%** | geometries whose tasks *all* solved **and** all came out connected; **not** a connectivity rate, since an unsolved task sinks the geometry |
| per-geometry, conditional | 58/136 = **42.6%** | of the geometries whose tasks all succeeded, the share where all tilings are also connected |

### Restricted to proved minima

Only `OPTIMAL` runs are minimum-brick tilings. A `FEASIBLE` result is whatever the search held when the clock ran out, so it must not be called a minimum and is excluded here.

- of 1399 proved minima, **1059 (75.7%)** are a single connected structure
- **340 tilings are provably minimum and disconnected** -- the objective was satisfied exactly and the result is still not one buildable structure
- (`FEASIBLE`, not minima: 97, of which 65 connected)

Every reference build was required to be stud-connected when the counterfactual data was generated, so a connected tiling exists for each of these shapes within each of these inventories. The solver returns a disconnected one anyway, because nothing in the objective asks otherwise: fewest bricks and one connected structure are different optimisation problems, and this arm answers the first. The 340 proved-minimum-but-disconnected cases are the cleanest statement of that -- not a search failure, but the correct answer to the question actually being asked. This is the measured cost of the missing constraint, and the argument for making connectivity-aware tiling the next work on the F track.


## Solve time

- total: 2747.1s over 1600 tasks
- median 0.143s, p95 10.002s, max 10.096s
- time limit 10.0s, seed 0, workers 1
