# StableText2Brick EDA

- train rows: 42604
- test rows: 4785

## 1. Axis assignment

- bricks parsed: 5097330
- structures that failed to parse: 0
- out-of-bounds under `h->x, w->y` (implemented): **0**
- out-of-bounds under `h->y, w->x`: **838563**

## 2. Dimension spellings

- distinct raw spellings: **14**
- distinct parts after rotation normalisation: **8**

| raw spelling | count |
|---|---:|
| `1x1` | 930775 |
| `1x2` | 615295 |
| `2x1` | 612169 |
| `2x2` | 603326 |
| `6x2` | 565982 |
| `2x6` | 537204 |
| `8x1` | 191997 |
| `1x4` | 165336 |
| `4x1` | 163722 |
| `4x2` | 153577 |
| `2x4` | 153195 |
| `1x8` | 150546 |
| `6x1` | 129021 |
| `1x6` | 125185 |

| canonical part | count | share |
|---|---:|---:|
| `1x2` | 1227464 | 24.1% |
| `2x6` | 1103186 | 21.6% |
| `1x1` | 930775 | 18.3% |
| `2x2` | 603326 | 11.8% |
| `1x8` | 342543 | 6.7% |
| `1x4` | 329058 | 6.5% |
| `2x4` | 306772 | 6.0% |
| `1x6` | 254206 | 5.0% |

- parts outside declared PART_VOCAB: none

## 3. Structures per object_id

- distinct object_id: **28259**
- mean structures per object: 1.68

| structures | #objects |
|---:|---:|
| 1 | 9469 |
| 2 | 18620 |
| 3 | 2 |
| 4 | 167 |
| 6 | 1 |

## 4. Counterfactual supply (the load-bearing number)

- object_id with >1 structure: **18790**
- ...whose variants have identical inventories: 11944
- ...differing in counts only: 5595
- ...differing in which part *types* are used: **1251**

Examples of type-set differences:

- `3e64a7042776f19adc9938c9fceb2ffd`
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x4,2x6}
  - {1x1,1x2,1x4,1x8,2x2,2x4,2x6}
- `1101146651cd32a1bd09c0f277d16187`
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x4,2x6}
  - {1x1,1x2,1x4,1x8,2x2,2x4,2x6}
- `6193a59df632dc4fd9b53420a5458c53`
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x4,2x6}
  - {1x1,1x2,1x4,1x8,2x2,2x4,2x6}
- `6256db826fbb31add7e7281b421bca5`
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x4,2x6}
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x6}
- `644f11d3687ab3ba2ade7345ab5b0cf6`
  - {1x1,1x2,1x4,1x6,1x8,2x2,2x4,2x6}
  - {1x1,1x2,1x4,1x8,2x2,2x4,2x6}

## 5. Train/test leakage

- object_id appearing in both splits: **0**

## 6. Structure size and sanity

- bricks per structure: min 6, p25 66, median 92, p75 128, p95 252, max 409
- structures with internal collisions: **0**
- distinct category_id: 21

| category_id | rows |
|---|---:|
| `04379243` | 11157 |
| `03001627` | 9564 |
| `02958343` | 6533 |
| `04256520` | 5549 |
| `04530566` | 3048 |
| `02828884` | 2792 |
| `02924116` | 1557 |
| `03467517` | 1515 |
| `02876657` | 939 |
| `03593526` | 806 |
| `04468005` | 743 |
| `02871439` | 695 |
| `03991062` | 581 |
| `02818832` | 335 |
| `03928116` | 330 |
| `03797390` | 310 |
| `02880940` | 269 |
| `04460130` | 213 |
| `02801938` | 168 |
| `02942699` | 168 |
| `02843684` | 117 |
