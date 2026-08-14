# LoRA smoke test (2,000 rows)

**A plumbing test, not an experiment.** It checks that a reproducible fine-tune runs end to end from the right starting point with a correct loss mask. It compares no hyperparameters, selects no checkpoint, and never reads the test split. The generations at the end are smoke observations: **no claim is made here about inventory compliance or generalisation**, which need the A-E protocol, multiple seeds and the hard-gate arms.

## Where training starts

Not from bare Llama. BrickGPT is published as a LoRA adapter, so training a fresh adapter on the base model would discard the published checkpoint while still looking like fine-tuning.

- base model: `meta-llama/Llama-3.2-1B-Instruct`
- published adapter: `AvaLovelace/BrickGPT` @ `19737def7bfe5950b2a466825ad7c6d74b7eafe3` (r=32, alpha=16)
- tokenizer: `AvaLovelace/BrickGPT` @ pinned revision, resolved separately from the adapter
- **start_from**: published BrickGPT adapter merged into the base weights, then a new LoRA attached on top; the published adapter is not modified and is not what gets saved
- merge verified to change weights: `True` (a fingerprint over q_proj/v_proj is compared before and after; the run aborts if the merge is a no-op)

## Configuration (declared before the run)

| setting | value |
|---|---|
| `rank` | 16 |
| `alpha` | 32 |
| `dropout` | 0.05 |
| `learning_rate` | 0.0001 |
| `target_modules` | ['q_proj', 'v_proj'] |
| `batch_size` | 1 |
| `grad_accum` | 8 |
| `max_length` | 2048 |
| `epochs` | 1 |
| `seed` | 0 |
| `dtype` | bfloat16 |
| `quantization` | none (bf16); 4-bit deliberately not used on MPS |
| `effective_batch` | 8 |

- trainable parameters: **1,703,936** of 1,237,518,336 (0.138%)
- trainable tensors: 64
- no quantisation: bf16 throughout. 4-bit was not used -- there is no dependable bitsandbytes path on Apple Silicon, and section 9.8 treats QLoRA as a memory option rather than a requirement.

## Data

| split | samples | pairs | objects |
|---|---:|---:|---:|
| train | 2000 | 250 | 244 |
| val | 320 | 40 | 38 |

- selection: 250 whole pairs by seeded shuffle of sorted pair ids (seed 0); every row of a chosen pair is included
- roles (train): {'control': 1000, 'counterfactual': 1000}
- variants (train): {'distractor': 500, 'exact': 500, 'loose': 500, 'mixed': 500}
- validation comes from the **val split only**; the test split is not opened by this script (`test_split_read: False`)
- train/val `object_id` overlap: **0** (244 vs 38 objects)
- rows truncated at max_length 2048: **0** (longest row 1836 tokens)

## Loss

| point | value |
|---|---:|
| validation before training | 1.1186 |
| validation after training | 0.2626 |
| train, first logged window | 0.9419 |
| train, last logged window | 0.2324 |

Validation is computed with the same masking as training, over 320 held-out rows. Two points on a single short run are a pipeline signal, not evidence that the model learned the task.

| epoch | rows seen | optimizer steps | train loss |
|---:|---:|---:|---:|
| 0 | 50 | 6 | 0.9419 |
| 0 | 100 | 12 | 0.7464 |
| 0 | 150 | 18 | 0.6005 |
| 0 | 200 | 25 | 0.5479 |
| 0 | 250 | 31 | 0.4785 |
| 0 | 300 | 37 | 0.4518 |
| 0 | 350 | 43 | 0.4308 |
| 0 | 400 | 50 | 0.3677 |
| 0 | 450 | 56 | 0.3943 |
| 0 | 500 | 62 | 0.3564 |
| 0 | 550 | 68 | 0.3613 |
| 0 | 600 | 75 | 0.3013 |
| 0 | 650 | 81 | 0.3100 |
| 0 | 700 | 87 | 0.3201 |
| 0 | 750 | 93 | 0.3221 |
| 0 | 800 | 100 | 0.2944 |
| 0 | 850 | 106 | 0.3022 |
| 0 | 900 | 112 | 0.3130 |
| 0 | 950 | 118 | 0.2975 |
| 0 | 1000 | 125 | 0.2950 |
| 0 | 1050 | 131 | 0.2909 |
| 0 | 1100 | 137 | 0.2743 |
| 0 | 1150 | 143 | 0.2924 |
| 0 | 1200 | 150 | 0.2694 |
| 0 | 1250 | 156 | 0.2646 |
| 0 | 1300 | 162 | 0.2579 |
| 0 | 1350 | 168 | 0.2683 |
| 0 | 1400 | 175 | 0.2698 |
| 0 | 1450 | 181 | 0.2775 |
| 0 | 1500 | 187 | 0.2561 |
| 0 | 1550 | 193 | 0.2604 |
| 0 | 1600 | 200 | 0.2478 |
| 0 | 1650 | 206 | 0.2493 |
| 0 | 1700 | 212 | 0.2276 |
| 0 | 1750 | 218 | 0.2340 |
| 0 | 1800 | 225 | 0.2529 |
| 0 | 1850 | 231 | 0.2399 |
| 0 | 1900 | 237 | 0.2465 |
| 0 | 1950 | 243 | 0.2326 |
| 0 | 2000 | 250 | 0.2324 |

## Cost

- `train_seconds`: 20732.7
- `total_seconds`: 26812.9
- `rows`: 2000
- `optimizer_steps`: 250
- `seconds_per_row_median`: 6.005
- `seconds_per_row_mean`: 10.366
- `seconds_per_optimizer_step`: 82.931
- `tokens_seen`: 1811906
- `tokens_per_second`: 87.4
- `peak_process_rss_gb`: 1.41
- `mps_allocated_gb_end`: 2.34

### The run gets progressively slower

**13.1x.** The first windows run at 2.85s/row and the last at 37.37s/row, with a worst window of 95.0s/row. This is why the mean (10.366s) sits so far above the median (6.005s), and why the progress line's ETA kept rising instead of falling: it extrapolates a rate that no longer holds.

The trend is noisy rather than smooth -- the table below goes up and down by a factor of two or three between neighbouring windows -- so this is a rising, erratic cost with a severe tail, not a clean curve. Only the direction and the size of the endpoints are being claimed.

Row length is ruled out: rows are shuffled, so long and short ones are spread evenly through the run.

**Memory is not ruled out.** The two figures recorded are `peak_process_rss_gb` = 1.41 and `mps_allocated_gb_end` = 2.34, and they say only that the process's peak resident set was small and that PyTorch's *tracked* MPS allocation was small **at the moment the run ended**. Neither is a peak of MPS usage, and neither covers driver-side or IOKit allocations, unified-memory pressure, compaction, swap, or fragmentation inside the allocator -- an allocator that is degrading typically still reports a modest tracked total. So memory remains a live candidate alongside anything else. **The cause is not isolated here and this report claims none.** The next step is a short diagnostic over 100-200 rows recording current *and* driver MPS allocation, system memory pressure, sequence length and per-phase timing -- not another full run.

| rows | seconds/row | train loss |
|---:|---:|---:|
| 50 | 2.04 | 0.9419 |
| 250 | 6.4 | 0.4785 |
| 500 | 12.88 | 0.3564 |
| 750 | 6.06 | 0.3221 |
| 1000 | 4.84 | 0.2950 |
| 1250 | 11.76 | 0.2646 |
| 1500 | 7.31 | 0.2561 |
| 1750 | 9.96 | 0.2340 |
| 2000 | 95.0 | 0.2324 |

**Consequence for the plan.** Section 9.8 asks whether local MPS is fast enough to be the real training environment. On this evidence, not as it stands: 2,000 rows for one epoch took 5.8 hours, and the full inventory-conditioned set is 9,584 rows, which at three epochs is fourteen times this work. A flat 2.85s/row would put that near 23 hours; the observed curve would put it far higher. Isolating the slowdown -- or restarting the process periodically, or moving to the Kaggle GPU fallback the workflow already allows -- has to come before the A-E runs, and is a finding of this smoke test rather than a detail of it.


DataLoader workers were chosen by measurement, not raised on principle: {'0': 0.014, '2': 25.947, '4': 51.51} seconds for two passes over 64 rows at 0/2/4 workers, so **0** was used. Rows are encoded once up front and held in memory, which leaves a worker process almost nothing to do.

## Environment

- `platform`: macOS-26.6.1-arm64-arm-64bit-Mach-O
- `machine`: arm64
- `python`: 3.13.9
- `torch`: 2.13.0
- `transformers`: 5.15.0
- `peft`: 0.20.0
- `device`: mps

## Provenance

What this run read and what it produced, so a later reader can tell whether the report still describes the files on disk.

| item | value |
|---|---|
| `instruct_inv_train.jsonl` sha256 | `423b1745e99ad36cb12df5cb5f9e33d12cb72b256f5a43f2ee2c7230c2e8aac5` |
| `instruct_inv_val.jsonl` sha256 | `5bdf882b4361f3cbd518de5e3af95422699ca636c9436a409fc47ef01965d4c6` |
| train selection digest | `92cda251219f79869f039c51dba81fad541ba2d36923bd48a9b959b775729654` |
| val selection digest | `e0e1b39a570584d5bb7459bb1fab7012e23fcfe5238e32b931d79b8c6adffa76` |
| training-order digest (epoch 0) | `91f73fe0d3ce55544d2770f2b12ffe2081150818d38e8c413d39c79a8a86b506` |
| manifest sha256 | `894acf3bdd3929943cfda0b1bb0fe1a75bea94621614e3f715ffa716b80ff6b8` |
| base model | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` |
| published adapter | `AvaLovelace/BrickGPT` @ `19737def7bfe5950b2a466825ad7c6d74b7eafe3` |
| tokenizer | `AvaLovelace/BrickGPT` @ `19737def7bfe5950b2a466825ad7c6d74b7eafe3` |
| saved adapter sha256 | `645cc6d9acf5aebc2301003567f7c0c3e9a6e08861d223117b3a81a3b7cccc69` |

### Code provenance: an unrecoverable gap

**The exact source that produced this run was never recorded.** The exact source commit and file digests at training start were never recorded: this run predates that capture. `de6b51e` is the first post-training tidy-up commit (2026-08-14 08:36:27, 44 minutes after the checkpoint was written at 07:52:40) and it contains edits made after training, so it is NOT the code that produced this run and must not be cited as such. Nothing recoverable identifies the exact training source; this is a permanent provenance gap for this run, not something a later check can close. Later runs capture HEAD, the dirty flag and file digests before the model loads.

**Selection digest and training-order digest are different things.** The selection digest covers *which* rows were chosen and their fixed listing order. The training-order digest covers the order they were actually fed to the model after `torch.randperm`: two runs can select identical rows and still present them in different orders, which is a different curriculum over the same data.

*RECONSTRUCTED after the run by replaying torch.randperm(seed=0) under torch 2.13.0, the same version the run used. It was not recorded while training. Valid only while that RNG behaviour holds; a torch upgrade would invalidate the reconstruction without any warning. Later runs record this digest as the shuffle happens.*

### Backfilled, not measured during the run

These fields did not exist when this run executed and were established afterwards from evidence. They are reconstructions and are labelled as such rather than presented as readings taken while training.

| field | how it got here |
|---|---|
| `instruction_sha256` | **backfilled after the run** |
| `base_revision` | **backfilled after the run** |
| `adapter_sha256` | **backfilled after the run** |
| `train_selection_digest` | **backfilled after the run** |
| `val_selection_digest` | **backfilled after the run** |
| `manifest_sha256` | **backfilled after the run** |
| `training_order_digest` | **backfilled after the run** |

The run predates the manifest, the base-revision pin and this block. `instruction_sha256` and `adapter_sha256` are computed from the files the run itself read and wrote, unchanged since. `base_revision`: the local cache holds exactly one Llama-3.2-1B-Instruct snapshot (9213176...), its mtime (2026-08-13 14:33) predates the training run (2026-08-14, ~02:00), and a cold start against that snapshot reproduces a coherent loss (0.1037) where the wrong load path gives 0.5609. Together these strongly support that this run used 9213176... -- but it was NOT recorded at training time, and the evidence cannot rule out a snapshot that was since deleted, or a different cache configuration (HF_HOME, a shared cache, an offline copy) in effect during the run. It is a well-supported inference, not a measurement. The selection digests are recomputed by the same seeded, deterministic sampler the run used.


## Generations before and after (smoke observation only)

4 fixed prompts from the **val** split, greedy decoding, no inventory gate, 400 new tokens max. Greedy because sampling on this machine has a measured MPS defect on sparse distributions (report 06) and a smoke check should not also be a sampling experiment. Before-training output is the merged BrickGPT: a freshly initialised LoRA has `lora_B = 0` and is exactly the identity, so the two rows differ only by what training changed.

**These are not evidence of compliance or quality.** They exist to show the generation path still produces brick syntax after training.

Two neutral observations, recorded because the next round should look at them rather than because anything is concluded here. Every output on both sides parses as bricks, so the syntax survived training. And with one exception before training, none of these generations emitted EOS inside 400 tokens -- they were cut off by the budget. Whether that is the prompt, the greedy decode, the short run, or nothing at all is not established by four samples.

### `054812ab-2129-4804-88a4-1f5c218a3835:0:control:distractor`

- before: 400 tokens in 7.14s
- after: 400 tokens in 20.38s

```text
BEFORE:
1x1 (4,19,0)
1x2 (4,17,0)
1x1 (4,12,0)
1x4 (4,8,0)
2x4 (2,15,0)
2x6 (2,7,0)
1x1 (1,19,0)
1x2 (1,17,0)
1x1 (1,12,0)
1x4 (1,8,0)
1x1 (4,18,1)
1x2 (4,16,1)
1x8 (4,8,1)
1x1 (3,8,1)
1x2 (3,6,1)
2x1 (2,18,1)
2x1 (2,9,1)
1x1 (2,7,1)
2x1 (2,4,1)
4x1 (1,19,1)
1x1 (1,18,1)
1x2 (1,16,1)
1x8 (1,8,1)
1x1 (4,19,2)
1x2 (4,17,2)
1x8 (4,9,2)
1x1 (4,8,2)
1x1 (3,4,2)
2x1 (2,18,2)
2x2 (2,8,2)
1x2 (2,3,2)
1x8 (1,12,2)
1x1 (1,11,2)
1x4 (1,7,2)
4x1 (1,7,2)
1x1 (1,6,2)
1x1 (1,5,2)
1x1 (4,19,3)
1x2 (4,17,3)
1x1 (4,7,3)

AFTER:
1x1 (1,14,0)
1x2 (1,15,0)
1x1 (2,19,0)
1x2 (2,17,0)
1x1 (3,19,0)
1x2 (3,17,0)
1x1 (1,14,1)
1x2 (1,15,1)
1x1 (2,19,1)
1x2 (2,17,1)
1x1 (3,19,1)
1x2 (3,17,1)
1x1 (1,14,2)
1x2 (1,15,2)
1x1 (2,19,2)
1x2 (2,17,2)
1x1 (3,19,2)
1x2 (3,17,2)
1x1 (1,14,3)
1x2 (1,15,3)
1x1 (2,19,3)
1x2 (2,17,3)
1x1 (3,19,3)
1x2 (3,17,3)
1x1 (1,14,4)
1x2 (1,15,4)
1x1 (2,19,4)
1x2 (2,17,4)
1x1 (3,19,4)
1x2 (3,17,4)
1x1 (1,14,5)
1x2 (1,15,5)
1x1 (2,19,5)
1x2 (2,17,5)
1x1 (3,19,5)
1x2 (3,17,5)
1x1 (1,14,6)
1x2 (1,15,6)
1x1 (2,19,6)
1x2 (2,17,6)
```

### `054812ab-2129-4804-88a4-1f5c218a3835:0:control:exact`

- before: 271 tokens in 4.83s
- after: 400 tokens in 8.11s

```text
BEFORE:
1x1 (2,18,0)
1x6 (2,12,0)
1x8 (2,4,0)
1x1 (1,3,0)
2x1 (0,18,0)
2x2 (0,16,0)
2x6 (0,10,0)
2x4 (0,6,0)
2x1 (0,5,0)
2x1 (0,4,0)
1x1 (2,9,1)
1x4 (2,5,1)
2x6 (1,14,1)
2x4 (1,10,1)
1x8 (1,2,1)
1x8 (0,12,1)
1x8 (0,4,1)
1x1 (2,19,2)
1x2 (2,17,2)
1x8 (2,9,2)
2x4 (0,16,2)
2x1 (0,15,2)
2x6 (0,9,2)
1x1 (2,14,3)
1x2 (2,12,3)
2x1 (0,14,3)
2x2 (0,12,3)

AFTER:
1x1 (1,14,0)
1x2 (1,15,0)
1x1 (2,19,0)
1x2 (2,17,0)
1x1 (3,19,0)
1x2 (3,17,0)
1x1 (1,14,1)
1x2 (1,15,1)
1x1 (2,19,1)
1x2 (2,17,1)
1x1 (3,19,1)
1x2 (3,17,1)
1x1 (1,14,2)
1x2 (1,15,2)
1x1 (2,19,2)
1x2 (2,17,2)
1x1 (3,19,2)
1x2 (3,17,2)
1x1 (1,14,3)
1x2 (1,15,3)
1x1 (2,19,3)
1x2 (2,17,3)
1x1 (3,19,3)
1x2 (3,17,3)
1x1 (1,14,4)
1x2 (1,15,4)
1x1 (2,19,4)
1x2 (2,17,4)
1x1 (3,19,4)
1x2 (3,17,4)
1x1 (1,14,5)
1x2 (1,15,5)
1x1 (2,19,5)
1x2 (2,17,5)
1x1 (3,19,5)
1x2 (3,17,5)
1x1 (1,14,6)
1x2 (1,15,6)
1x1 (2,19,6)
1x2 (2,17,6)
```

### `054812ab-2129-4804-88a4-1f5c218a3835:0:control:loose`

- before: 400 tokens in 7.1s
- after: 400 tokens in 905.32s

```text
BEFORE:
1x1 (4,19,0)
1x2 (4,17,0)
1x1 (4,12,0)
1x2 (4,10,0)
1x1 (3,17,0)
2x4 (3,13,0)
2x6 (2,7,0)
2x1 (1,16,0)
2x4 (1,12,0)
1x1 (1,11,0)
1x2 (1,9,0)
1x1 (4,18,1)
1x8 (4,10,1)
1x1 (3,8,1)
2x1 (2,18,1)
2x6 (2,12,1)
2x1 (2,11,1)
2x2 (2,9,1)
1x1 (2,7,1)
1x8 (1,9,1)
1x1 (4,17,2)
1x4 (4,13,2)
1x1 (3,18,2)
2x1 (3,12,2)
2x2 (3,10,2)
1x1 (3,9,2)
1x2 (3,7,2)
1x1 (2,18,2)
1x1 (2,15,2)
1x2 (2,13,2)
1x1 (2,10,2)
1x4 (2,6,2)
2x2 (1,16,2)
1x1 (1,9,2)
2x4 (0,12,2)
2x1 (0,11,2)
2x1 (4,16,3)
2x2 (4,14,3)
1x1 (4,10,3)
1x1 (3,16,3)

AFTER:
1x1 (1,14,0)
1x2 (1,15,0)
1x1 (2,19,0)
1x2 (2,17,0)
1x1 (3,19,0)
1x2 (3,17,0)
1x1 (1,14,1)
1x2 (1,15,1)
1x1 (2,19,1)
1x2 (2,17,1)
1x1 (3,19,1)
1x2 (3,17,1)
1x1 (1,14,2)
1x2 (1,15,2)
1x1 (2,19,2)
1x2 (2,17,2)
1x1 (3,19,2)
1x2 (3,17,2)
1x1 (1,14,3)
1x2 (1,15,3)
1x1 (2,19,3)
1x2 (2,17,3)
1x1 (3,19,3)
1x2 (3,17,3)
1x1 (1,14,4)
1x2 (1,15,4)
1x1 (2,19,4)
1x2 (2,17,4)
1x1 (3,19,4)
1x2 (3,17,4)
1x1 (1,14,5)
1x2 (1,15,5)
1x1 (2,19,5)
1x2 (2,17,5)
1x1 (3,19,5)
1x2 (3,17,5)
1x1 (1,14,6)
1x2 (1,15,6)
1x1 (2,19,6)
1x2 (2,17,6)
```

### `054812ab-2129-4804-88a4-1f5c218a3835:0:control:mixed`

- before: 400 tokens in 7.26s
- after: 400 tokens in 8.23s

```text
BEFORE:
1x1 (4,19,0)
1x2 (4,17,0)
2x2 (4,15,0)
2x6 (4,9,0)
1x1 (4,8,0)
1x2 (4,6,0)
1x1 (3,15,0)
1x4 (3,11,0)
2x6 (2,5,0)
2x1 (1,15,0)
2x4 (1,11,0)
1x1 (1,10,0)
1x4 (1,6,0)
1x1 (0,15,0)
1x8 (0,7,0)
1x1 (5,16,1)
1x2 (5,14,1)
1x1 (5,11,1)
1x2 (5,9,1)
2x6 (3,11,1)
2x4 (3,7,1)
1x1 (3,6,1)
1x2 (3,4,1)
1x1 (2,16,1)
1x2 (2,14,1)
1x1 (2,8,1)
1x4 (2,4,1)
2x1 (1,13,1)
2x4 (1,9,1)
1x1 (1,7,1)
1x2 (1,5,1)
2x2 (0,15,1)
1x1 (0,14,1)
1x1 (0,11,1)
1x2 (0,9,1)
2x1 (0,8,1)
1x1 (4,15,2)
1x2 (4,13,2)
1x1 (4,10,2)
1x2 (4,8,2)

AFTER:
1x1 (1,2,0)
1x1 (1,3,0)
1x1 (1,18,0)
1x1 (1,19,0)
1x1 (2,2,0)
1x1 (2,3,0)
1x1 (2,18,0)
1x1 (2,19,0)
1x1 (1,2,1)
1x1 (1,3,1)
1x1 (1,2,1)
1x1 (1,18,1)
1x1 (1,19,1)
1x1 (2,2,1)
1x1 (2,3,1)
1x1 (2,18,1)
1x1 (2,19,1)
1x1 (1,2,2)
1x1 (1,3,2)
1x1 (1,2,2)
1x1 (1,18,2)
1x1 (1,19,2)
1x1 (2,2,2)
1x1 (2,3,2)
1x1 (2,18,2)
1x1 (2,19,2)
1x1 (1,2,3)
1x1 (1,3,3)
1x1 (1,2,3)
1x1 (1,18,3)
1x1 (1,19,3)
1x1 (2,2,3)
1x1 (2,3,3)
1x1 (2,18,3)
1x1 (2,19,3)
1x1 (1,2,4)
1x1 (1,3,4)
1x1 (1,2,4)
1x1 (1,18,4)
1x1 (1,19,4)
```

## Checkpoint

- adapter: `artifacts/checkpoints/lora_smoke/` (gitignored; only the script, config, tests and this report are committed)
- the published adapter is merged into the base and is **not** modified; what is saved is our delta alone
- saved without tokenizer files, which is why the loader addresses tokenizer and adapter separately
