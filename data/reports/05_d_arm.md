# D arm: hard inventory gate

Compliance measurement of the constraint layer on the base checkpoint. **Not** an A/B/C/D/E comparison -- that needs a fixed prompt set, matched physics settings and multiple samples per arm.

The prompt states the same inventory the gate enforces, both taken from one snapshot before decoding starts. An earlier version of this script built the gate by hand and left the block off, so it measured a hard gate over an unconditioned prompt -- a configuration no arm uses. The counts below were produced after that was fixed.

## Environment

- `platform`: macOS-26.6.1-arm64-arm-64bit-Mach-O
- `machine`: arm64
- `torch`: 2.13.0
- `device`: mps
- `dtype`: torch.bfloat16
- `base`: meta-llama/Llama-3.2-1B-Instruct
- `adapter`: AvaLovelace/BrickGPT
- `temperature`: 0.6
- `max_bricks`: 40
- `prompt`: inventory-conditioned (the B-E form), matching the gate

## Compliance

- runs: 24 (3 inventories x 8 seeds)
- tokens audited against their slot: 5832
- **grammar violations: 0**
- **type-compliance failures: 0/24**
- **count-compliance failures: 0/24**
- parse rate: 24/24
- parsed bricks == gate ledger: 24/24
- prompt block == gate opening ledger: 24/24
- wall clock: 103s

## Per run

| caption | seed | bricks | termination | remaining | coll | comps | unsup |
|---|---:|---:|---|---|---:|---:|---:|
| A simple chair. | 0 | 24 | inventory_exhausted | (empty) | 0 | 5 | 4 |
| A simple chair. | 1 | 24 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A simple chair. | 2 | 24 | inventory_exhausted | (empty) | 0 | 4 | 3 |
| A simple chair. | 3 | 24 | inventory_exhausted | (empty) | 0 | 3 | 2 |
| A simple chair. | 4 | 24 | inventory_exhausted | (empty) | 0 | 2 | 3 |
| A simple chair. | 5 | 24 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A simple chair. | 6 | 24 | inventory_exhausted | (empty) | 0 | 10 | 9 |
| A simple chair. | 7 | 24 | inventory_exhausted | (empty) | 0 | 3 | 2 |
| A small car. | 0 | 9 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A small car. | 1 | 9 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A small car. | 2 | 9 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A small car. | 3 | 9 | inventory_exhausted | (empty) | 1 | 1 | 0 |
| A small car. | 4 | 9 | inventory_exhausted | (empty) | 1 | 1 | 0 |
| A small car. | 5 | 9 | inventory_exhausted | (empty) | 3 | 1 | 0 |
| A small car. | 6 | 9 | inventory_exhausted | (empty) | 1 | 1 | 0 |
| A small car. | 7 | 9 | inventory_exhausted | (empty) | 0 | 1 | 0 |
| A table. | 0 | 40 | max_bricks | {'1x8': 5, '2x2': 8, '2x4': 17} | 4 | 1 | 0 |
| A table. | 1 | 40 | max_bricks | {'1x8': 9, '2x2': 1, '2x4': 20} | 0 | 4 | 3 |
| A table. | 2 | 40 | max_bricks | {'2x2': 13, '2x4': 17} | 0 | 8 | 7 |
| A table. | 3 | 40 | max_bricks | {'1x8': 5, '2x2': 5, '2x4': 20} | 0 | 1 | 0 |
| A table. | 4 | 40 | max_bricks | {'1x2': 5, '1x8': 2, '2x2': 6, '2x4': 17} | 0 | 10 | 9 |
| A table. | 5 | 40 | max_bricks | {'1x2': 8, '1x8': 9, '2x4': 13} | 0 | 7 | 6 |
| A table. | 6 | 40 | max_bricks | {'1x8': 6, '2x2': 10, '2x4': 14} | 2 | 37 | 36 |
| A table. | 7 | 40 | max_bricks | {'1x8': 8, '2x2': 3, '2x4': 19} | 0 | 4 | 3 |

Collisions, components and unsupported bricks are *not* gated by this layer -- placement legality belongs to the rejection stage, which is not implemented yet. They are recorded to show what the inventory gate alone does and does not buy.
