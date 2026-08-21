# Stagger ablation

- seed 0, 60 shapes, eligibility <= 150 bricks
- sampling rule identical to `scripts/08_corpus_structure_study.py`
- solver time limit 20.0s, single worker
- measurements reused from a previous run

| condition | solved | stud-connected | rate | mean bricks | seconds |
|---|---:|---:|---:|---:|---:|
| `per-layer` | 60/60 | 23 | 38.3% | 73.0 | 7.0 |
| `joint` | 60/60 | 20 | 33.3% | 73.0 | 11.2 |
| `joint+stagger` | 53/60 | 11 | 20.8% | 130.8 | 985.2 |

## What these numbers do and do not say

`joint+stagger` left **7 of 60** shapes unfinished at the 20s budget. Those come back `UNKNOWN` -- the solver ran out of time. They are **not** disconnected results, and nothing here shows a connected tiling does not exist for them.

Two rates, neither of which is a connectivity rate on its own:

- over each condition's own solved subset: `joint` 33.3%, `joint+stagger` 20.8% (-12.6%) -- different denominators
- solved **and** connected, over all 60 attempted: `joint` 33.3%, `joint+stagger` 18.3% (-15.0%) -- an end-to-end yield that mixes solver success with connectivity

Mean brick count (73 vs 131) is computed over those same different solved subsets, with an unknown status mix. It does not support a claim that stagger inflates brick count; the shapes it could not finish are simply absent from its column.

## Engineering conclusion

At a 20s budget the staggered formulation solves fewer shapes (53/60 against 60/60) and costs 88x the wall clock for the same sample. That is sufficient to keep it out of the production path, and it is the only conclusion drawn here.

Whether the constraint helps or hurts connectivity when given enough time is not answered by this benchmark. Settling it would need a budget large enough for both conditions to solve every shape. Sample of 60 shapes.
