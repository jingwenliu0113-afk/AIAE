# Corpus structure study

- seed 0; eligible rows sorted by `structure_id`, then sampled
- eligibility: <= 150 bricks (same as the generator)
- `paired source` and `paired retile` are the **same 60 shapes**; `source 400` is the wider sample and is *not* comparable to them one-to-one
- wall clock: 16s

## All three populations

| population | n | stud connected | stud + baseplate | >=1 unsupported |
|---|---:|---|---|---|
| `source 400` | 400 | **369/400 = 92.2%** | 400/400 = 100.0% | 335/400 = 83.8% |
| `paired source 60` | 60 | **53/60 = 88.3%** | 60/60 = 100.0% | 54/60 = 90.0% |
| `paired retile 60` | 60 | **23/60 = 38.3%** | 29/60 = 48.3% | 56/60 = 93.3% |

## Support, on the shapes that can be compared

Restricted to the 60 paired shapes:

- corpus tiling: 54/60 = 90.0%
- CP-SAT re-tiling: 56/60 = 93.3%
- difference: **+2 of 60 structures**

That is a small difference on a small sample. It is an observation about this sample, not evidence that re-tiling causes unsupported bricks; no causal claim is made either way. Support is reported rather than gated in the generator.

## Connectivity

Stud coupling alone is the project definition and the acceptance gate. The baseplate column is an anchoring metric only: it merges components that share no studs, so a model held together only by it would come apart when lifted.

On the paired 60, corpus tilings are 53/60 = 88.3% single-component against 23/60 = 38.3% for our re-tilings.

The current re-tiling formulation is associated with that drop, but the cause is not isolated. Solver strategy alone does not account for it: per-layer reaches 38.3% and a joint solve 33.3% on the same shapes (scripts/09_stagger_ablation.py), so switching to a joint model does not recover the gap. Candidate factors -- the fewest-bricks objective, tie-breaking between equally optimal tilings, and per-layer independence -- have not been separated.

## Raw counts

- `source 400`: {"n": 400, "stud_connected": 369, "ground_connected": 400, "with_unsupported": 335, "total_unsupported": 3016, "total_bricks": 33352, "median_stud_components": 1}
- `paired source 60`: {"n": 60, "stud_connected": 53, "ground_connected": 60, "with_unsupported": 54, "total_unsupported": 433, "total_bricks": 5265, "median_stud_components": 1}
- `paired retile 60`: {"n": 60, "stud_connected": 23, "ground_connected": 29, "with_unsupported": 56, "total_unsupported": 405, "total_bricks": 4377, "median_stud_components": 2, "unsolved": 0}
