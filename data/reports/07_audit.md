# Dataset audit (recomputed from files)

Everything below is recalculated from the JSONL, not read from the stored `checks` field. It shares the project's parser and predicates with the generator, so it verifies that the checks were applied and labelled correctly -- not that those predicates are themselves right.

Each pair contributes two distinct tilings (control and counterfactual); the four inventory framings reuse the same geometry, so samples = 4 x unique targets.

| split | pairs | unique targets | samples | objects | bytes |
|---|---:|---:|---:|---:|---:|
| train | 1200 | **2400** | 9600 | 1083 | 19699144 |
| val | 200 | **400** | 1600 | 139 | 3220298 |
| test | 200 | **400** | 1600 | 174 | 3316802 |

| split | sha256 |
|---|---|
| train | `fb3d5c7931e7745d0095f938780255933e6e8dfef1a3ec662c87c14b4514c3bf` |
| val | `0da84b3281006db34213b8fc9342cdb9bcb355a9db23917f74e3840279cfbdd5` |
| test | `917b4386f21c469b8c68248044ae4387e03c43c8bccd66b7b1fc67a53c4bdd62` |

## Components

Stud coupling is the gate. The baseplate column shows the weaker criterion that is deliberately *not* applied.

| split | unique targets | stud single | ground single | pairs both stud-connected |
|---|---:|---:|---:|---:|
| train | 2400 | 2400 | 2400 | 1200/1200 |
| val | 400 | 400 | 400 | 200/200 |
| test | 400 | 400 | 400 | 200/200 |

## Support (reported, not gated; over unique targets)

| split | unique targets | fully supported |
|---|---:|---:|
| train | 2400 | 29.4% |
| val | 400 | 31.8% |
| test | 400 | 29.2% |

## Droppable parts tried before success

- train: {1: 784, 2: 250, 3: 96, 4: 48, 5: 12, 6: 10}
- val: {1: 139, 2: 40, 3: 9, 4: 6, 5: 4, 6: 2}
- test: {1: 140, 2: 37, 3: 14, 4: 6, 5: 2, 6: 1}

## Distractor extra-type histogram (0 must be absent)

- train: {1: 1834, 2: 1406, 3: 1560}
- val: {1: 378, 2: 222, 3: 200}
- test: {1: 336, 2: 218, 3: 246}

## Cross-split object overlap

- `train|val`: **0**
- `test|train`: **0**
- `test|val`: **0**

## Result

- checks failed: **0**
