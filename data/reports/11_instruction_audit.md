# Instruction data audit

Recomputed from the JSONL at tokenizer `AvaLovelace/BrickGPT` @ `19737def7bfe5950b2a466825ad7c6d74b7eafe3`. **Every check covers every row of both arms** -- token counts included, re-derived rather than sampled.

| split | rows per arm | rows re-tokenised (both arms) | targets using a rotated spelling |
|---|---:|---:|---:|
| train | 9584 | 19168 | 9456 |
| val | 1600 | 3200 | 1576 |
| test | 1600 | 3200 | 1592 |
| **all** | | **25568** | |

Rotated targets are the point of the rotation rule: their parts are counted against the canonical line (`4x1` against `1x4`), and the check above confirms none of them overdraws.

## Removals against the counterfactual file

Rebuilt from `counterfactual_{split}.jsonl`, not from the build report. Every sample absent from the instruction data must belong to a pair that is absent *entirely* -- both roles, all four framings, in both arms. A pair left half-removed would strand a control without its counterfactual.

| split | source samples | missing (inv) | missing (noinv) | pairs touched | over-budget rows in them | removal sound |
|---|---:|---:|---:|---:|---|---|
| train | 9600 | 16 | 16 | 2 | 2, 2 | **yes** |
| val | 1600 | 0 | 0 | 0 | - | **yes** |
| test | 1600 | 0 | 0 | 0 | - | **yes** |

`removal sound` covers every removal condition together: whole pairs only, identical across arms, no row invented, and at least one genuinely over-budget row per removed pair (re-tokenised here, not read from the build report). Any one of them failing turns the column.

## Result

- checks failed: **0**
