# Counterfactual dataset

- seed 0, max source bricks 150, attempt cap 12x target
- variants per target: exact, loose, distractor, mixed
- wall clock: 9050s

**Acceptance gate**: both arms solver-feasible, exact voxel cover, inventory legal, dropped part absent everywhere, and a single connected component **under stud coupling alone**. The baseplate is not a part, carries no inventory and is never written out, so it never counts towards connection; it appears only as the ground-single metric below. Support is reported, not enforced.

Rather than drop a source when one chosen part fails, every droppable part is tried in a seeded order until one yields a stud-connected counterfactual.

## Yield

| split | pairs kept | target met | attempts | yield | samples | objects |
|---|---:|:--:|---:|---:|---:|---:|
| train | 1200 | yes | 8789 | 13.7% | 9600 | 1083 |
| val | 200 | yes | 1515 | 13.2% | 1600 | 139 |
| test | 200 | yes | 1473 | 13.6% | 1600 | 174 |

## Failure reasons

| split | control disconnected | control retile failed | counterfactual disconnected for all 1 droppable parts | counterfactual disconnected for all 2 droppable parts | counterfactual disconnected for all 3 droppable parts | counterfactual disconnected for all 4 droppable parts | counterfactual disconnected for all 5 droppable parts | counterfactual disconnected for all 6 droppable parts | counterfactual disconnected for all 7 droppable parts | no distractor pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5780 | 6 | 1 | 3 | 8 | 7 | 25 | 28 | 51 | 1680 |
| val | 1027 | 0 | 0 | 2 | 0 | 0 | 10 | 4 | 9 | 263 |
| test | 955 | 0 | 0 | 0 | 1 | 2 | 3 | 8 | 10 | 294 |

## Components (stud coupling gates; baseplate is a metric)

| split | targets | stud single | ground single | parts tried (mean/max) |
|---|---:|---:|---:|---|
| train | 2400 | 2400 | 2400 | 1.6 / 6 |
| val | 400 | 400 | 400 | 1.5 / 6 |
| test | 400 | 400 | 400 | 1.5 / 6 |

Every kept target is stud-single by construction. Ground-single is shown to make explicit that the weaker criterion is not what was applied.


## Support (reported, not gated)

| split | targets | fully supported | rate |
|---|---:|---:|---:|
| train | 2400 | 706 | 29.4% |
| val | 400 | 127 | 31.8% |
| test | 400 | 117 | 29.2% |

## Distractor effectiveness

Every distractor and mixed sample must add at least one part the target does not use.

| split | extra-type counts | types added |
|---|---|---|
| train | {1: 1834, 2: 1406, 3: 1560} | {'2x2': 1596, '1x4': 1548, '1x8': 1544, '1x6': 1426, '1x1': 1200, '2x4': 1174, '2x6': 516, '1x2': 322} |
| val | {1: 378, 2: 222, 3: 200} | {'1x1': 260, '2x2': 254, '1x4': 250, '1x8': 218, '1x6': 188, '2x4': 132, '2x6': 64, '1x2': 56} |
| test | {1: 336, 2: 218, 3: 246} | {'2x2': 274, '1x4': 242, '1x8': 224, '1x6': 218, '1x1': 212, '2x4': 192, '2x6': 76, '1x2': 72} |

## Dropped part distribution

| split | 1x2 | 1x4 | 1x6 | 1x8 | 2x2 | 2x4 | 2x6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 196 | 174 | 186 | 111 | 186 | 175 | 172 |
| val | 27 | 33 | 26 | 27 | 24 | 30 | 33 |
| test | 27 | 32 | 24 | 18 | 31 | 31 | 37 |

## Cross-split object overlap (must all be 0)

- `train|val`: **0**
- `test|train`: **0**
- `test|val`: **0**

## Files

| file | bytes | sha256 |
|---|---:|---|
| `data/processed/counterfactual_train.jsonl` | 19699144 | `fb3d5c7931e7745d0095f938780255933e6e8dfef1a3ec662c87c14b4514c3bf` |
| `data/processed/counterfactual_val.jsonl` | 3220298 | `0da84b3281006db34213b8fc9342cdb9bcb355a9db23917f74e3840279cfbdd5` |
| `data/processed/counterfactual_test.jsonl` | 3316802 | `917b4386f21c469b8c68248044ae4387e03c43c8bccd66b7b1fc67a53c4bdd62` |
