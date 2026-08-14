# MPS speed diagnostic (<= 200 rows)

Report 13 measured a 13.1x slowdown across 2,000 rows and named no cause. This is the short follow-up that tries to localise it: same starting point, LoRA config, data seed and training-order rule, but at most 200 rows per condition, with each phase timed separately and memory sampled as it goes. Optimizer updates do run, in memory, on real training rows -- what is not done is keeping them: no checkpoint is written and `artifacts/checkpoints/lora_smoke/` is untouched.

## Method

- `time.perf_counter()`, with `torch.mps.synchronize()` at every phase boundary. Without the sync an MPS call returns as soon as the work is *enqueued*, so the timing would be attributed to whichever phase happened to wait for it.
- phases: `collate_h2d`, `forward`, `backward`, `optimizer`; anything left over is reported as `_unattributed` rather than absorbed.
- memory sampled every 5 rows: PyTorch's tracked allocation *and* the driver's, which can diverge.
- windows of 20 rows.

### Conditions, declared before running

| condition | difference |
|---|---|
| `continuous` | one uninterrupted run |
| `empty_cache` | the same rows in the same order, with `torch.mps.empty_cache()` every 10 rows |

A process-restart condition is **not** included. Restarting resets the model and optimizer too, so a speedup would confound a fresh process with a fresh model; measuring it properly needs its own design stating which state carries over. Reporting it here would produce a number that reads as a fix without being one.

### Stop conditions

- 3 consecutive rows over 30s
- or 45 minutes in one condition

Hitting one is a normal outcome: partial results are kept and the reason is reported. The goal is to localise a cost, not to finish training.

## Environment

- `platform`: macOS-26.6.1-arm64-arm-64bit-Mach-O
- `machine`: arm64
- `python`: 3.13.9
- `torch`: 2.13.0
- `transformers`: 5.15.0
- `peft`: 0.20.0
- `device`: mps
- `dtype`: bfloat16
- `max_rows_per_condition`: 200
- `window`: 20
- `memory_sample_every`: 5
- `empty_cache_every`: 10
- `seed`: 0
- `grad_accum`: 8
- `condition_order`: ['continuous', 'empty_cache']
- `single_process_fixed_order`: True
- `stop_slow_row_seconds`: 30.0
- `stop_max_seconds`: 2700.0
- `phases`: ['collate_h2d', 'forward', 'backward', 'optimizer']
- `schema_version`: 1
- `stop_slow_row_streak`: 3
- `loss_decimals_stored`: 4
- `row_timing_decimals_stored`: 4
- `schema_1_note`: This run did not record per-row sample ids, per-row end-to-end time, the split of between-row overhead, or the loss at full precision. Those gaps are rendered as unknown rather than reconstructed.
- `condition_definitions`: {'continuous': 'one uninterrupted run', 'empty_cache': 'the same rows in the same order, with `torch.mps.empty_cache()` every 10 rows'}

Baseline memory before any condition ran:

```json
{
  "mps_current_allocated_gb": 0.0,
  "mps_driver_allocated_gb": 0.0,
  "mps_recommended_max_gb": 37.44,
  "swap_used_gb": 0.884,
  "memory_pressure_percent_free": 93,
  "peak_process_rss_gb": 0.699,
  "free_plus_inactive_gb": 7.474
}
```


## Condition: `continuous`

- run order in the process: **0** (0 = first)
- rows completed: **200** of 200 requested
- stopped early: `None`
- end-to-end: 684.3s = 3.422s/row, **including** the between-row work broken out below
- model compute: 682.41s = 3.412s/row, the summed timed regions only
- between-row overhead: 1.89s; its split into clears, probes and the rest is not recorded for this run
- scheduled `empty_cache()`: **0** calls -- this is the intervention under test
- teardown `empty_cache()`: **not recorded for this run**. This run never counted them, so how many were made -- if any -- is not known and no number is assumed here. By design such a clear would fall outside the timed region and outside every figure above, which is why it is counted apart from the scheduled clears in the first place.

**The window figures below report both**: model compute is the summed timed regions, end-to-end adds the between-row work. `empty_cache()` and the memory probes run outside the timed region, so the two columns must not be quoted against each other.

### Where the time went

| phase | total s | mean s | median s | max s | share |
|---|---:|---:|---:|---:|---:|
| `collate_h2d` | 0.534 | 0.0027 | 0.002 | 0.0788 | 0.1% |
| `forward` | 427.587 | 2.1379 | 1.1566 | 10.9192 | 62.7% |
| `backward` | 253.066 | 1.2653 | 0.7601 | 6.8614 | 37.1% |
| `optimizer` | 0.941 | 0.0047 | 0.0 | 0.1202 | 0.1% |
| _unattributed_ | 0.282 | | | | 0.0% |

### Per window (raw)

`compute` is the timed regions; `end-to-end` adds the between-row work. A run that did not record per-row end-to-end shows `not recorded for this run` rather than a figure copied from the compute column.

| window | rows | compute s | compute s/row | end-to-end s | end-to-end s/row | tokens | supervised | tok/s | mean seq |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1-20 | 30.016 | 1.501 | - | - | 19995 | 14540 | 666.2 | 999.8 |
| 1 | 21-40 | 32.453 | 1.623 | - | - | 17043 | 11670 | 525.2 | 852.1 |
| 2 | 41-60 | 68.053 | 3.403 | - | - | 17596 | 12260 | 258.6 | 879.8 |
| 3 | 61-80 | 53.122 | 2.656 | - | - | 15645 | 10200 | 294.5 | 782.2 |
| 4 | 81-100 | 71.319 | 3.566 | - | - | 18866 | 13430 | 264.5 | 943.3 |
| 5 | 101-120 | 53.567 | 2.678 | - | - | 16655 | 11250 | 310.9 | 832.8 |
| 6 | 121-140 | 91.57 | 4.578 | - | - | 18143 | 12790 | 198.1 | 907.1 |
| 7 | 141-160 | 91.369 | 4.568 | - | - | 20304 | 14830 | 222.2 | 1015.2 |
| 8 | 161-180 | 86.985 | 4.349 | - | - | 18753 | 13300 | 215.6 | 937.6 |
| 9 | 181-200 | 103.957 | 5.198 | - | - | 17011 | 11540 | 163.6 | 850.5 |

### Memory as it went

`peak process RSS` is `ru_maxrss`: a high-water mark for the process, not a current reading. `free+inactive` is what `vm_stat` allows adding up; inactive pages are reclaimable, so it is neither free memory nor available memory in the everyday sense.

| row | elapsed s | MPS current GB | MPS driver GB | recommended max GB | peak process RSS GB | free+inactive GB | swap GB | mem pressure % free |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.67 | 2.347 | 9.774 | 37.44 | 0.922 | 4.982 | 0.884 | 72 |
| 5 | 6.84 | 2.347 | 20.424 | 37.44 | 0.936 | 2.449 | 0.884 | 52 |
| 10 | 14.37 | 2.416 | 27.039 | 37.44 | 1.017 | 2.997 | 0.884 | 49 |
| 15 | 21.52 | 2.416 | 34.225 | 37.44 | 1.037 | 2.214 | 0.884 | 49 |
| 20 | 30.12 | 2.396 | 41.481 | 37.44 | 1.037 | 1.229 | 1.102 | 42 |
| 25 | 37.11 | 2.393 | 46.011 | 37.44 | 1.037 | 0.995 | 2.398 | 16 |
| 30 | 43.53 | 2.393 | 44.913 | 37.44 | 1.037 | 1.323 | 2.305 | 29 |
| 35 | 53.29 | 2.393 | 48.527 | 37.44 | 1.037 | 1.759 | 6.607 | 40 |
| 40 | 62.69 | 2.371 | 50.537 | 37.44 | 1.037 | 1.603 | 6.244 | 29 |
| 45 | 71.28 | 2.396 | 50.498 | 37.44 | 1.037 | 1.254 | 5.901 | 29 |
| 50 | 97.2 | 2.393 | 52.299 | 37.44 | 1.037 | 1.453 | 7.082 | 24 |
| 55 | 106.71 | 2.393 | 52.284 | 37.44 | 1.037 | 1.974 | 7.549 | 35 |
| 60 | 130.89 | 2.393 | 54.585 | 37.44 | 1.037 | 1.968 | 9.154 | 33 |
| 65 | 141.93 | 2.396 | 51.288 | 37.44 | 1.037 | 0.817 | 8.827 | 27 |
| 70 | 156.97 | 2.396 | 51.324 | 37.44 | 1.037 | 0.985 | 6.944 | 20 |
| 75 | 175.4 | 2.393 | 51.666 | 37.44 | 1.037 | 1.407 | 8.756 | 27 |
| 80 | 184.21 | 2.371 | 51.99 | 37.44 | 1.037 | 1.745 | 9.005 | 27 |
| 85 | 192.71 | 2.393 | 52.295 | 37.44 | 1.037 | 0.694 | 9.437 | 12 |
| 90 | 201.08 | 2.394 | 52.037 | 37.44 | 1.037 | 1.16 | 8.742 | 32 |
| 95 | 218.7 | 2.394 | 52.17 | 37.44 | 1.037 | 0.676 | 8.417 | 10 |
| 100 | 255.72 | 2.393 | 51.843 | 37.44 | 1.037 | 0.872 | 7.348 | 14 |
| 105 | 270.71 | 2.396 | 51.601 | 37.44 | 1.037 | 1.369 | 7.657 | 26 |
| 110 | 281.81 | 2.396 | 51.673 | 37.44 | 1.037 | 1.215 | 8.384 | 35 |
| 115 | 293.9 | 2.396 | 52.066 | 37.44 | 1.037 | 0.928 | 7.93 | 30 |
| 120 | 309.52 | 2.371 | 51.763 | 37.44 | 1.037 | 1.204 | 7.209 | 30 |
| 125 | 335.06 | 2.396 | 51.596 | 37.44 | 1.037 | 1.168 | 6.987 | 24 |
| 130 | 365.47 | 2.393 | 52.555 | 37.44 | 1.037 | 0.703 | 8.746 | 11 |
| 135 | 386.58 | 2.393 | 51.02 | 37.44 | 1.037 | 0.975 | 6.888 | 17 |
| 140 | 401.32 | 2.396 | 51.003 | 37.44 | 1.037 | 1.55 | 7.556 | 26 |
| 145 | 426.66 | 2.393 | 52.438 | 37.44 | 1.037 | 1.036 | 8.516 | 17 |
| 150 | 435.77 | 2.393 | 52.431 | 37.44 | 1.037 | 0.687 | 7.778 | 14 |
| 155 | 457.32 | 2.393 | 52.23 | 37.44 | 1.037 | 0.81 | 7.307 | 18 |
| 160 | 492.91 | 2.371 | 54.499 | 37.44 | 1.037 | 1.514 | 9.162 | 25 |
| 165 | 527.19 | 2.396 | 51.988 | 37.44 | 1.037 | 1.646 | 7.739 | 31 |
| 170 | 545.43 | 2.393 | 51.744 | 37.44 | 1.037 | 0.549 | 9.005 | 9 |
| 175 | 568.81 | 2.393 | 51.865 | 37.44 | 1.037 | 0.573 | 8.699 | 8 |
| 180 | 580.13 | 2.393 | 52.147 | 37.44 | 1.037 | 0.967 | 8.698 | 16 |
| 185 | 613.11 | 2.396 | 51.924 | 37.44 | 1.037 | 0.568 | 9.075 | 8 |
| 190 | 638.04 | 2.396 | 51.797 | 37.44 | 1.037 | 0.568 | 8.609 | 9 |
| 195 | 659.32 | 2.396 | 51.712 | 37.44 | 1.037 | 0.443 | 8.635 | 7 |
| 200 | 684.26 | 2.371 | 51.968 | 37.44 | 1.037 | 0.593 | 9.24 | 9 |


## Condition: `empty_cache`

- run order in the process: **1** (0 = first)
- rows completed: **200** of 200 requested
- stopped early: `None`
- end-to-end: 278.04s = 1.39s/row, **including** the between-row work broken out below
- model compute: 269.08s = 1.345s/row, the summed timed regions only
- between-row overhead: 8.96s; its split into clears, probes and the rest is not recorded for this run
- scheduled `empty_cache()`: **20** calls -- this is the intervention under test
- teardown `empty_cache()`: **not recorded for this run**. This run never counted them, so how many were made -- if any -- is not known and no number is assumed here. By design such a clear would fall outside the timed region and outside every figure above, which is why it is counted apart from the scheduled clears in the first place.

**The window figures below report both**: model compute is the summed timed regions, end-to-end adds the between-row work. `empty_cache()` and the memory probes run outside the timed region, so the two columns must not be quoted against each other.

### Where the time went

| phase | total s | mean s | median s | max s | share |
|---|---:|---:|---:|---:|---:|
| `collate_h2d` | 0.271 | 0.0014 | 0.0007 | 0.0109 | 0.1% |
| `forward` | 126.573 | 0.6329 | 0.6117 | 1.5978 | 47.1% |
| `backward` | 141.652 | 0.7083 | 0.6898 | 1.7598 | 52.7% |
| `optimizer` | 0.366 | 0.0018 | 0.0 | 0.0192 | 0.1% |
| _unattributed_ | 0.218 | | | | 0.1% |

### Per window (raw)

`compute` is the timed regions; `end-to-end` adds the between-row work. A run that did not record per-row end-to-end shows `not recorded for this run` rather than a figure copied from the compute column.

| window | rows | compute s | compute s/row | end-to-end s | end-to-end s/row | tokens | supervised | tok/s | mean seq |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1-20 | 29.662 | 1.483 | - | - | 19995 | 14540 | 674.1 | 999.8 |
| 1 | 21-40 | 24.665 | 1.233 | - | - | 17043 | 11670 | 691.0 | 852.1 |
| 2 | 41-60 | 25.579 | 1.279 | - | - | 17596 | 12260 | 687.9 | 879.8 |
| 3 | 61-80 | 22.508 | 1.125 | - | - | 15645 | 10200 | 695.1 | 782.2 |
| 4 | 81-100 | 27.669 | 1.383 | - | - | 18866 | 13430 | 681.8 | 943.3 |
| 5 | 101-120 | 24.362 | 1.218 | - | - | 16655 | 11250 | 683.7 | 832.8 |
| 6 | 121-140 | 28.54 | 1.427 | - | - | 18143 | 12790 | 635.7 | 907.1 |
| 7 | 141-160 | 32.586 | 1.629 | - | - | 20304 | 14830 | 623.1 | 1015.2 |
| 8 | 161-180 | 28.558 | 1.428 | - | - | 18753 | 13300 | 656.7 | 937.6 |
| 9 | 181-200 | 24.953 | 1.248 | - | - | 17011 | 11540 | 681.7 | 850.5 |

### Memory as it went

`peak process RSS` is `ru_maxrss`: a high-water mark for the process, not a current reading. `free+inactive` is what `vm_stat` allows adding up; inactive pages are reclaimable, so it is neither free memory nor available memory in the everyday sense.

| row | elapsed s | MPS current GB | MPS driver GB | recommended max GB | peak process RSS GB | free+inactive GB | swap GB | mem pressure % free |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.52 | 2.349 | 9.877 | 37.44 | 2.979 | 7.859 | 3.197 | 71 |
| 5 | 6.79 | 2.349 | 20.518 | 37.44 | 2.979 | 5.194 | 3.189 | 51 |
| 10 | 14.66 | 2.418 | 4.783 | 37.44 | 2.979 | 8.095 | 3.158 | 84 |
| 15 | 21.83 | 2.418 | 21.666 | 37.44 | 2.979 | 5.155 | 3.142 | 63 |
| 20 | 30.61 | 2.395 | 4.883 | 37.44 | 2.979 | 8.791 | 3.111 | 82 |
| 25 | 37.13 | 2.39 | 18.717 | 37.44 | 2.979 | 5.515 | 3.095 | 54 |
| 30 | 43.21 | 2.39 | 4.22 | 37.44 | 2.979 | 8.049 | 3.088 | 75 |
| 35 | 49.61 | 2.387 | 18.053 | 37.44 | 2.979 | 5.643 | 3.08 | 62 |
| 40 | 56.14 | 2.373 | 4.25 | 37.44 | 2.979 | 7.653 | 3.072 | 85 |
| 45 | 62.15 | 2.409 | 15.135 | 37.44 | 2.979 | 6.386 | 3.072 | 65 |
| 50 | 70.01 | 2.403 | 4.088 | 37.44 | 2.979 | 6.535 | 3.072 | 62 |
| 55 | 74.73 | 2.403 | 13.635 | 37.44 | 2.979 | 6.754 | 3.048 | 64 |
| 60 | 82.65 | 2.39 | 4.326 | 37.44 | 2.979 | 6.88 | 3.041 | 86 |
| 65 | 88.07 | 2.39 | 15.447 | 37.44 | 2.979 | 6.37 | 3.002 | 66 |
| 70 | 93.92 | 2.39 | 4.318 | 37.44 | 2.979 | 8.809 | 3.002 | 82 |
| 75 | 100.34 | 2.386 | 18.127 | 37.44 | 2.979 | 5.605 | 3.002 | 56 |
| 80 | 106.03 | 2.373 | 4.265 | 37.44 | 2.979 | 9.026 | 3.002 | 84 |
| 85 | 112.27 | 2.388 | 17.156 | 37.44 | 2.979 | 5.809 | 3.002 | 59 |
| 90 | 118.49 | 2.384 | 4.246 | 37.44 | 2.979 | 5.977 | 3.002 | 79 |
| 95 | 124.78 | 2.384 | 16.43 | 37.44 | 2.979 | 5.962 | 2.939 | 58 |
| 100 | 134.54 | 2.382 | 4.025 | 37.44 | 2.979 | 8.096 | 2.916 | 78 |
| 105 | 139.94 | 2.382 | 15.482 | 37.44 | 2.979 | 6.181 | 2.916 | 63 |
| 110 | 146.6 | 2.382 | 4.025 | 37.44 | 2.979 | 7.602 | 2.916 | 72 |
| 115 | 152.61 | 2.381 | 16.035 | 37.44 | 2.979 | 6.051 | 2.916 | 62 |
| 120 | 159.76 | 2.373 | 4.025 | 37.44 | 2.979 | 6.806 | 2.916 | 68 |
| 125 | 166.94 | 2.407 | 16.207 | 37.44 | 2.979 | 5.975 | 2.916 | 61 |
| 130 | 175.71 | 2.396 | 4.072 | 37.44 | 2.979 | 7.003 | 2.916 | 86 |
| 135 | 182.14 | 2.396 | 14.467 | 37.44 | 2.979 | 6.428 | 2.916 | 64 |
| 140 | 189.18 | 2.395 | 4.287 | 37.44 | 2.979 | 8.963 | 2.916 | 84 |
| 145 | 198.45 | 2.39 | 23.691 | 37.44 | 2.979 | 4.076 | 2.814 | 44 |
| 150 | 205.43 | 2.39 | 4.31 | 37.44 | 2.979 | 5.805 | 2.783 | 64 |
| 155 | 212.6 | 2.389 | 17.322 | 37.44 | 2.979 | 5.631 | 2.783 | 62 |
| 160 | 222.64 | 2.373 | 5.56 | 37.44 | 2.979 | 5.726 | 2.588 | 81 |
| 165 | 230.84 | 2.405 | 22.096 | 37.44 | 2.979 | 4.518 | 2.588 | 50 |
| 170 | 237.4 | 2.401 | 4.049 | 37.44 | 2.979 | 8.018 | 2.588 | 77 |
| 175 | 244.59 | 2.401 | 16.353 | 37.44 | 2.979 | 5.785 | 2.588 | 63 |
| 180 | 252.1 | 2.391 | 4.064 | 37.44 | 2.979 | 5.773 | 2.588 | 66 |
| 185 | 259.84 | 2.389 | 19.334 | 37.44 | 2.979 | 5.033 | 2.588 | 52 |
| 190 | 265.87 | 2.389 | 4.064 | 37.44 | 2.979 | 8.411 | 2.58 | 80 |
| 195 | 272.12 | 2.392 | 17.076 | 37.44 | 2.979 | 5.587 | 2.58 | 58 |
| 200 | 277.96 | 2.373 | 4.271 | 37.44 | 2.979 | 8.787 | 2.58 | 84 |


## Comparison

`end-to-end s/row` includes between-row overhead; `compute s/row` and the window columns are the timed regions only.

| condition | order | rows | end-to-end s/row | compute s/row | first window (compute) | last window (compute) | stopped |
|---|---:|---:|---:|---:|---:|---:|---|
| `continuous` | 0 | 200 | 3.422 | 3.412 | 1.501 | 5.198 | no |
| `empty_cache` | 1 | 200 | 1.39 | 1.345 | 1.483 | 1.248 | no |

## Reading this

**The long tail did reproduce within this window.** The last window is more than twice the first, so the effect starts early enough to study at this scale.

### What moved with it

The two conditions ran the same rows in the same order. Checked rather than assumed: **200/200** rows match on token and supervised-token counts, and **200/200** produce **the same loss at the stored precision of 4 decimal places** -- which bounds agreement at that precision and says nothing about the digits below it.

What differs is the timing, the memory, **and the conditions they started from**: both ran in one process in the fixed order `continuous -> empty_cache`, each rebuilding the model and optimizer from the same seed, but the second inherited whatever swap, thermal, OS and process state the first left behind. Order is therefore confounded with condition here.

| | `continuous` | `empty_cache` |
|---|---:|---:|
| s/row, first window | 1.501 | 1.483 |
| s/row, last window | **5.198** | **1.248** |
| MPS *tracked* allocation | 2.347-2.416 GB | 2.349-2.418 GB |
| MPS *driver* min / max / end GB | 9.774 / **54.585** / 51.968 | 4.025 / 23.691 / 4.271 |
| samples over the 37.44 GB recommended max | **37/41** | 0/41 |
| swap start / end GB | 0.884 / **9.24** | 3.197 / 2.58 |
| least free+inactive GB seen | **0.443** | 4.076 |
| memory pressure % free, start / min / end | 72 / **7** / 9 | 71 / 44 / 84 |

Two observations, stated at the strength the design supports.

**The tracked figure would never have shown this.** PyTorch's `current_allocated_memory` sits at 2.347-2.416 GB in `continuous` and 2.349-2.418 GB in `empty_cache` -- flat in both, and within 0.002 GB of each other. The driver figure is where the growth is, and in `continuous` it runs to 54.585 GB, past the 37.44 GB the system recommends, while swap grows to 9.24 GB and free+inactive pages fall to 0.443 GB -- reclaimable pages, not a reading of memory sitting available. Report 13 read a flat tracked figure and a small RSS as 'not memory exhaustion'; on this evidence that reading was wrong, and the correction made to it last round was warranted.

**The condition that cleared the cache did not degrade.** With `empty_cache()` every 10 rows, driver allocation stayed a sawtooth that never reached the recommended max and the per-row time did not rise. That is a co-occurrence in one ordered pair of runs, not a demonstrated fix. The `forward` phase is where the cost sits: 2.14s mean in `continuous` against 0.63s in `empty_cache`, on the same rows in the same order.

**This is a strong short-range signal, not an isolated cause.** Under a single fixed order with n=1 per condition, periodic `empty_cache()` and the absence of degradation occurred together, and the degradation moved with driver allocation, swap and memory pressure. That is co-occurrence plus a mitigation that worked once. It does **not** rule out the fixed order itself -- `empty_cache` ran second, on a machine the first condition had already loaded -- and it does **not** establish the internal mechanism: retained cache, fragmentation, unified-memory pressure and swap thrash are all consistent with these readings and none is separated here. Nor does it show this accounts for report 13's 13.1x over 2,000 rows; this run is 200 rows.

The order confound is the first thing a follow-up should remove, by running the conditions in both orders or in fresh processes.

## Provenance

**Part of this record was reconstructed after the run, and is labelled as such.** The gaps below cannot be closed retrospectively; later runs capture all of it before the model loads.

**The code that actually ran was not recorded at start, and the numeric JSON predates the final version of this script** -- the script was edited after the run to correct method and naming. So the measurements are from one version and the report is rendered by another. This gap cannot be closed retrospectively: no HEAD, dirty flag or file digest was captured before the model loaded. Later runs capture all of it up front via `capture_provenance()`.

| item | value | when recorded |
|---|---|---|
| `selection_digest` | `92cda251219f79869f039c51dba81fad541ba2d36923bd48a9b959b775729654` | **backfilled** |
| `training_order_digest` | `9c2cf55a0056d937801071f2a9c5317c0a1794d1f8c8976cd389b489fa270b62` | **backfilled** |
| `base_model` | `meta-llama/Llama-3.2-1B-Instruct` | **backfilled** |
| `base_revision` | `9213176726f574b556790deb65791e0c5aa438b6` | **backfilled** |
| `published_adapter_revision` | `19737def7bfe5950b2a466825ad7c6d74b7eafe3` | **backfilled** |
| `tokenizer_revision` | `19737def7bfe5950b2a466825ad7c6d74b7eafe3` | **backfilled** |
| `instruct_inv_train.jsonl` | `423b1745e99ad36cb12df5cb5f9e33d12cb72b256f5a43f2ee2c7230c2e8aac5` | **backfilled** |
| `continuous` input order | not recorded for this run | -- |
| `empty_cache` input order | not recorded for this run | -- |

The data digest is computed from the file the run read, unchanged since (it still matches report 10). The selection and training-order digests are recomputed by replaying the same deterministic sampler and `torch.randperm(seed=0)` under the same torch version -- valid only while that RNG behaviour holds. The revisions are the values `src/model_ids.py` supplied then and now; the run did not record them. `scheduled_empty_cache_calls` is derived from the schedule, not counted during the run; teardown clears were never counted at all, so how many were made -- if any -- is recorded as unknown rather than assumed. `phases` is read off the stored per-row records. `stop_slow_row_streak` is the unchanged default the run used, which the original report already stated in prose. `condition_definitions` is derived from the stored run rather than from the current script: `continuous` records no clear schedule and `empty_cache` records one of every 10 rows, and the two arms are already verified to have run the same rows in the same order. The wording the original report used for the second arm was corrected in the same round that added this field, so the description stored here is the corrected one rather than the one that was rendered at the time. None of these is a measurement taken at run time.

### Limits of this diagnostic

- Timings are wall clock on a shared machine. Nothing here isolates thermal state, other processes, or OS-level scheduling, and no claim is made about any of them.
- The memory columns are readings, not explanations. A rising driver figure against a flat tracked figure is *consistent with* allocator growth; it does not establish it.
- One run per condition, in one process, in a fixed order. The gap measured is far larger than window-to-window noise, but order is confounded with condition and n=1 cannot separate them.
- The second condition did not start from the first's initial conditions: swap, thermal state and OS state carried over. Only the model and optimizer were rebuilt from the same seed.
- The mechanism is not identified. The output is a narrowed search space and a mitigation that co-occurred with the absence of the slowdown once, not a diagnosis.
- Under this design `empty_cache()` cannot be said to remove the slowdown: the degradation did not appear in the condition that cleared the cache, and the same condition also ran second. Whether clearing holds over thousands of rows, and what it costs when it does, is the next thing to measure -- and it needs both orders, or fresh processes, before it can be measured at all.
