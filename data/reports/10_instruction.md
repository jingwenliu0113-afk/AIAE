# Instruction format

One template for every arm; the inventory block is the only difference between the unconditioned and conditioned arms.

`inv` is training data for the conditioned arms. `noinv` is **arm A's matched evaluation counterpart, not arm-A training data**: the same objects, captions and targets with the block removed, so the unconditioned baseline is scored on identical material. Its rows repeat because the four framings of a target share one unconditioned prompt; each duplicate still carries the inventory hidden from it, which compliance is scored against, so evaluation weights by multiplicity.

Two documented deviations from the section 9.7 sketch, both to keep the arms comparable: the body is BrickGPT's own instruction rather than a fresh `### Request` preamble (it carries the allowed dimensions and one-unit-tall rule the checkpoint was trained against), and parts are named `1x1` rather than `brick_1x1`, reusing the model's existing size vocabulary. The spelling still need not match: the model may emit `4x1` against a listed `1x4`, and the canonical mapping draws both from the same quantity -- which is what the rotation rule states.

- tokenizer: `AvaLovelace/BrickGPT` @ `19737def7bfe5950b2a466825ad7c6d74b7eafe3` (pinned)
- prompt tokens are masked out of the loss (`labels = -100`)
- target ends with EOS
- wall clock 33s

## Sequence lengths

| arm / split | rows | prompt (median) | target (median) | total median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| `inv/train` | 9584 | 273 | 581 | 858 | 1439 | 2044 |
| `inv/val` | 1600 | 274 | 561 | 837 | 1396 | 1783 |
| `inv/test` | 1600 | 274 | 601 | 872 | 1424 | 1648 |
| `noinv/train` | 9584 | 176 | 581 | 761 | 1344 | 1944 |
| `noinv/val` | 1600 | 177 | 561 | 740 | 1303 | 1683 |
| `noinv/test` | 1600 | 177 | 601 | 773 | 1329 | 1548 |

## Coverage by budget

Exact counts, not rounded rates. Pairs containing over-2048 rows were removed before writing -- the whole pair, both roles and all four framings, not just the row that measured long. Targets are never truncated.

| arm / split | <= 1024 | <= 2048 | <= 4096 | over 2048 | pairs dropped |
|---|---|---|---|---:|---:|
| `inv/train` | 6838/9584 (71.35%) | 9584/9584 (100.00%) | 9584/9584 (100.00%) | **0** | 2 |
| `inv/val` | 1172/1600 (73.25%) | 1600/1600 (100.00%) | 1600/1600 (100.00%) | **0** | 0 |
| `inv/test` | 1120/1600 (70.00%) | 1600/1600 (100.00%) | 1600/1600 (100.00%) | **0** | 0 |
| `noinv/train` | 7640/9584 (79.72%) | 9584/9584 (100.00%) | 9584/9584 (100.00%) | **0** | 2 |
| `noinv/val` | 1352/1600 (84.50%) | 1600/1600 (100.00%) | 1600/1600 (100.00%) | **0** | 0 |
| `noinv/test` | 1252/1600 (78.25%) | 1600/1600 (100.00%) | 1600/1600 (100.00%) | **0** | 0 |

### What was removed

Four distinct counts, listed separately because they are easy to conflate. The trigger rows are what tripped the budget check; the pair is the unit actually removed, so the number of rows that leave the JSONL is several times larger than the number that measured long.

| split | over-budget trigger rows | pairs dropped | source samples removed | instruction rows removed (both arms) |
|---|---|---:|---:|---:|
| `train` | inv 4, noinv 0 | 2 | 16 | 32 |
| `val` | inv 0, noinv 0 | 0 | 0 | 0 |
| `test` | inv 0, noinv 0 | 0 | 0 | 0 |

A pair is 2 roles x 4 inventory framings = 8 counterfactual samples, and each surviving sample is written once per arm.


## Cost of the inventory block

Per-row paired difference `inv - noinv` over all 12784 rows. A difference of medians would not answer this: the cost depends on how many part lines an inventory has.

| min | median | p95 | max |
|---:|---:|---:|---:|
| +58 | +100 | +107 | +107 |

## Rows, unique prompts and multiplicity

`inv` also shows a little repetition. That is not the framings collapsing: it is two `structure_id`s of the same object whose voxel occupancy is identical, so re-tiling returns the same target and the caption is the same object's. Every such group sits inside one object, so it cannot cross the split boundary -- the column below counts any group that does.

`noinv` is arm A's matched evaluation counterpart, not arm-A training data. Removing the block collapses a target's four inventory framings to one prompt, so rows repeat; each duplicate still carries the inventory hidden from its prompt, which is what compliance is scored against. Evaluation weights by multiplicity.

| arm / split | rows | unique prompt-target | multiplicity | dup groups spanning objects |
|---|---:|---:|---|---:|
| `inv/train` | 9584 | 9106 | {1: 8628, 2: 478} | **0** |
| `inv/val` | 1600 | 1326 | {1: 1052, 2: 274} | **0** |
| `inv/test` | 1600 | 1476 | {1: 1352, 2: 124} | **0** |
| `noinv/train` | 9584 | 2270 | {4: 2144, 8: 126} | **0** |
| `noinv/val` | 1600 | 328 | {4: 256, 8: 72} | **0** |
| `noinv/test` | 1600 | 367 | {4: 334, 8: 33} | **0** |

## Files

| file | sha256 |
|---|---|
| `data/processed/instruct_inv_train.jsonl` | `423b1745e99ad36cb12df5cb5f9e33d12cb72b256f5a43f2ee2c7230c2e8aac5` |
| `data/processed/instruct_inv_val.jsonl` | `5bdf882b4361f3cbd518de5e3af95422699ca636c9436a409fc47ef01965d4c6` |
| `data/processed/instruct_inv_test.jsonl` | `085c6900a328c1ccdb7496ae9af22ffb383038bf8c70af10213a3561a297dbbd` |
| `data/processed/instruct_noinv_train.jsonl` | `2db0d9740961fd5e85e1811970f2bd620f3e02a200bf3c825bae4996e2a70839` |
| `data/processed/instruct_noinv_val.jsonl` | `f8a104ee12f4082e615a2e15f17381708564bddae3af74073c3d194abae206e3` |
| `data/processed/instruct_noinv_test.jsonl` | `5ee28ef30468fe6adef6300bb8834ed65bf83c42355f318e291f21e2c182bc8f` |
