# Sparse-distribution sampling by device

## Environment

- `platform`: macOS-26.6.1-arm64-arm-64bit-Mach-O
- `machine`: arm64
- `processor`: arm
- `python`: 3.13.9
- `torch`: 2.13.0
- `mps_available`: True
- `mps_built`: True

## Setup

- vocabulary 128256, 20 candidates left finite after masking
- 4000 draws per device, seed 0, dtype bfloat16

## Result: mask the full row, then sample (what `generate` does)

| device | draws | outside support | rate |
|---|---:|---:|---:|
| `cpu` | 4000 | **0** | 0.00% |
| `mps` | 4000 | **24** | 0.60% |

## Result: restrict to candidates, then normalise (what this project does)

| device | draws | outside support | rate |
|---|---:|---:|---:|
| `cpu` | 4000 | **0** | 0.00% |
| `mps` | 4000 | **0** | 0.00% |

## Reading this

A brick is ten tokens. At the masked-row rate measured above, a 60-brick structure draws 600 times, so even a fraction of a percent corrupts a large share of generations; the observed symptom was a coordinate slot emitting a word.

This measures the environment listed above and nothing else. It is not a general statement about Apple Silicon, other torch releases, or other samplers. Re-run after upgrading torch.
