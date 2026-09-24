# Model Card — BrickAgain (public research snapshot)

This card describes the models in **this repository**, which is the archived
research track. The current product — BrickNet real parts, 14,583 part types —
is not published here and is **not** described by this card. Numbers from the
two tracks measure different systems and must never be placed side by side.

Last updated 2026-09-18.

---

## 1. What is here

| Model | What it does | Where |
|---|---|---|
| Inventory-gated generator | Text description plus a finite parts inventory → a brick structure, with the inventory enforced at decode time | `src/generation/`, `src/constraints/` |
| CP-SAT re-tiler | Re-lays a fixed voxel shape under a different inventory | `src/data/retile.py` |
| Vision classifier | Eight-way brick photo classification, ResNet-18 head | `src/vision/` |
| Visual-stress recogniser | Reads rendered builds back for evaluation | `src/eval/visual_stress.py` |

All four operate on **eight rectangular part types** — `1x1 1x2 1x4 1x6 1x8
2x2 2x4 2x6` — derived from `AvaLovelace/StableText2Brick`.

## 2. Intended use

Research and teaching: reproducing the experiments recorded in
`data/reports/`, inspecting how an inventory constraint can be enforced by a
decoder rather than learned, and reusing the CP-SAT re-tiling and evaluation
code.

The load-bearing idea is that **the inventory guarantee comes from the
program, not the model**. `src/constraints/inventory_decode.py` rejects any
step that would exceed the declared stock, so a model that has never seen the
constraint still cannot violate it. That property is worth reading the code
for; the generation quality is not.

## 3. Out of scope

- **Not a product.** The public snapshot does not contain the BrickNet
  dataset, weights or product modules. It will not start a working
  application, by design — see the first paragraph of `README.md`.
- **Not a safety system.** Structures are checked for geometric validity and
  stud connectivity only. Nothing here predicts whether a build will stand up,
  survive handling, or be safe for a child. No physical assembly has been
  tested.
- **Not a claim about real photographs.** The vision classifier was fitted on
  a small photographed set; it has not been evaluated on images from other
  cameras, lighting or backgrounds.
- **Not affiliated** with the LEGO Group, the BrickGPT authors, Meta, or
  LDraw.org.

## 4. Known limitations

- **Eight parts is not LEGO.** Axle, hinge, ball and clip connections do not
  exist in this vocabulary. A structure that is valid here can be impossible
  with real parts, and the reverse.
- **Connectivity is stud coupling only.** Two columns joined by a beam at the
  top are connected in reality and disconnected here.
- **Feasibility is not uniform across parts.** CP-SAT re-tiling succeeds for
  88.0% of structures overall, but per-part feasibility ranges widely — `1x1`
  restriction succeeds 8.8% of the time
  (`reports/figures/03_retile_feasibility.png`).
- **Staggering was tried and rejected.** Requiring staggered courses cost
  140x the solve time and *reduced* stud connectivity from 38.3% to 20.8%
  (`reports/figures/04_stagger_ablation.png`). The published pipeline does not
  stagger, and structures may therefore have aligned vertical seams.
- **The generation arm has no end-to-end number.** `data/reports/12_f_oracle.md`
  is an oracle: the target shape is read from the reference build rather than
  predicted. It bounds what the optimisation stage can do given a perfect
  stage before it, and is explicitly not a deployable result.

## 5. Known biases in the data

- **Part frequency is heavily skewed.** `1x2` accounts for 1.23M of 5.10M
  placements; `1x6` for 254K — roughly five to one
  (`reports/figures/01_part_distribution.png`). A model fitted on this corpus
  will reach for `1x2` and `2x6` first, and that is a property of the corpus,
  not of good building.
- **Variants are mostly not counterfactual.** Of 18,790 objects with more than
  one variant, 11,944 have an *identical* inventory and only 1,251 use
  different part types (`reports/figures/02_variant_breakdown.png`). Natural
  variation is not a substitute for generated counterfactuals; this is why
  `src/data/retile.py` exists.
- **The corpus is one source.** Everything derives from StableText2Brick's
  captioned builds. Whatever that set over- or under-represents is inherited
  whole, and has not been audited for it.

## 6. Evaluation, and how to read it

Reports live in `data/reports/` as a JSON and a Markdown file per experiment.
Every figure in `reports/figures/` is regenerated from those JSON files by
`scripts/73_figures.py`, so a number in this card can be traced to a file.

Read them with these caveats:

- An **oracle** arm is an upper bound, not a score.
- A **feasibility rate** is about the solver, not about a model.
- Results in `data/reports/60_visual_stress/` are measured against *rendered*
  builds, so they bound recognition of the renderer's output, not of photos.

## 7. Splits and leakage

Splits are assigned by `object_id`, never by structure: a re-tiled variant of
an object is the same object, and splitting on the variant would place a
model's twin in training and its sibling in validation.
`src/data/splits.py` derives every structure's split from its object, and
`tests/test_no_leakage.py` holds that property together with the cases that
must fail if it breaks.

The assignment is a pure function of a salted SHA-256 of the id, so it does
not reshuffle between runs or between machines.

## 8. Reproducibility

- Seeds are fixed; hyperparameters live in `configs/vision.yaml` and are read
  by `src/config.py`, with `tests/test_config.py` proving the file is actually
  read rather than shadowed by the compiled-in defaults.
- `scripts/73_figures.py` writes figures deterministically, but the bytes
  are only reproducible on the platform that drew them: on ubuntu-latest an
  unchanged rebuild came out different in all four PNGs, with matplotlib
  pinned. So CI does not redraw. It checks `reports/figures/figures.json` --
  the SHA-256 of the generator, of every report each figure read, and of each
  PNG -- which gives the same answer on any machine and still fails when a
  report moves without a redraw.
- Dataset and model artefacts are **not** in this repository. Several
  experiments in `data/reports/` therefore cannot be re-executed from a
  public checkout; their recorded results and digests are what travels.

## 9. Licence and provenance

MIT (`LICENSE`). Third-party components, their licences and the items
deliberately not redistributed are listed in `THIRD_PARTY_NOTICES.md`.
