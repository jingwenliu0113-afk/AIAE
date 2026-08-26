# BrickAgain

BrickAgain is a work-in-progress research prototype for generating simplified
brick structures from text while respecting a finite parts inventory. The core
track uses the eight rectangular parts represented by StableText2Brick and
combines deterministic data preparation, CP-SAT re-tiling, inventory-gated
decoding, LoRA training utilities, evaluation, and LDraw export.

> 這是尚在進行中的研究原型。目前公開的重點是可審查的程式、測試與實驗紀錄，
> 不代表完整的最終產品或已完成的學術結論。

BrickAgain is an independent research project and is not affiliated with or
endorsed by the LEGO Group, the BrickGPT authors, Meta, or LDraw.org.

## Current status

Implemented and tested:

- StableText2Brick parsing and orientation normalization into eight inventory
  part types.
- Object-level train/validation/test split checks.
- Deterministic CP-SAT counterfactual re-tiling and dataset audits.
- Transactional inventory accounting and inventory-gated decoding.
- BrickGPT inference integration, LoRA smoke-training utilities, and protected
  local-adapter loading order.
- LDraw export aligned with BrickGPT reference vectors.
- An opt-in collision and connectivity gate for decoding, off by default,
  which makes a colliding placement unreachable rather than detected. It has
  never been formally evaluated -- see below.
- A command-line demonstration that checks, exports and draws a brick list on
  the CPU with no model, no GPU and no network. It does not generate on that
  path: the brick list comes from a stored fixture, from you, or from a decode,
  and the report names which. See [SHOWCASE.md](SHOWCASE.md).
- Stored reports with explicit provenance and limitations.

Still open:

- **The collision and connectivity gate has never been formally evaluated.**
  It is implemented, reviewed and tested, and it is opt-in and off by default.
  No metric has ever been computed with it enabled, so enabling it is not
  evidence that any success rate moved, in either direction. The one relevant
  precedent points the other way: the inventory gate *lowered* the marginal
  in-bounds and collision-free rates in the four-arm comparison, because
  constraining one axis moves the others.
- Connectivity here means 2-D footprint overlap between adjacent layers. It
  is not support and not a physics result: nothing in this repository checks
  centre of mass, moments, or whether a model stands up. That analysis needs
  a solver this project does not have.
- The MPS slowdown study currently provides a strong short-run signal, not a
  causal result; its first run used a fixed condition order with one run per
  condition.
- This repository does not distribute Llama, BrickGPT, or locally trained
  model weights.

**The model-research track is closed, not paused.** The full A–E
hyperparameter comparison was never completed and is not scheduled: one
declared hyperparameter setting was run, a four-arm functional comparison ran
once on cases frozen in advance, and a project model was selected from that
and frozen. No further training, tuning or model reselection is planned, and
the four-arm numbers are not published here. Reading "incomplete" as "in
progress" would be reading a decision as a backlog.

That is a statement about the research track, not about delivery. The minimum
non-UI delivery has completed independent technical review: manual stock can
drive a train-only existing-work comparison or a
lexical-retrieval-plus-CP-SAT baseline, and a selected structure can be
checked, exported as LDraw, and drawn as a CPU 3-D geometric preview. Static
delivery requires both ground contact and adjacent-layer connectivity. Ground
contact is not physics and not stability; connectivity is not support. This is
delivery plumbing, not a new model result. The two-page UI was explicitly
excluded from that minimum scope; it was authorised separately afterwards and
is now implemented, which changes what is built and not what was measured.
This public release has completed independent technical review.

See [BRICKAGAIN_PROJECT_WORKFLOW.md](BRICKAGAIN_PROJECT_WORKFLOW.md) for the
research plan. The running status file is not published: it is a working
record of the private research tree, and it names evidence that is not here.
What survives of it is in the aggregate reports under `data/reports/`, which
are published in full.

## Demonstration

```bash
./.venv/bin/python scripts/26_showcase.py --sample tower
```

A brick list goes in with an inventory and a brief; every deterministic check
runs on it, and out come LDraw, a per-layer plan view, and optionally a CPU
3-D geometric preview with `--preview FILE`. That path needs no model, no GPU
and no network — it checks and exports a brick list rather than producing one.
Only `--generate` loads a model.

It **measures nothing**: no number it prints is a metric, and none of it is
comparable to any stored evaluation. Every report states where its bricks came
from, whether the token count was measured or derived, and whether the
termination was measured, stated, supplied or unavailable. Full operating
instructions, the stored briefs, and what the output does and does not mean
are in [SHOWCASE.md](SHOWCASE.md).

For the minimum non-UI product path, including manual inventory, train-only
existing-work comparison, the minimum F-pipeline, 3-D preview, and the
read-only closure of the stored Success evidence, see
[DELIVERY.md](DELIVERY.md):

```bash
./.venv/bin/python scripts/27_delivery.py \
  --mode compare \
  --caption "This train features a streamlined, elongated rectangular body composed of uniformly arranged bricks. The top is flat with evenly spaced small cylindrical protrusions, providing a cohesive and structured appearance." \
  --inventory "1x2:1,2x4:1,2x6:5" --top-n 1
```

The full tested exit-zero compare and F-pipeline commands, including LDraw and
preview outputs, are in `DELIVERY.md`. This comparison uses a deterministic
lexical baseline, not a multilingual embedding model. The catalogue rows are
checked against the frozen object-level split manifest as well as their
`split=train` labels. The F-pipeline is implemented but has not been formally
evaluated; neither path produces a metric. Its default train catalogue is
private processed data and is deliberately absent from the public snapshot.

## The two-page interface

```bash
./.venv/bin/python scripts/29_ui.py
```

A local, CPU-only, offline two-page interface over the same delivery path,
bound to loopback and nothing else. Page one takes a brief and a manual stock
of the eight parts and picks one of the two methods; page two shows the method
actually used, the provenance, the per-candidate evidence, every deterministic
check, the stock used and left, a CPU 3-D geometric preview, and an LDraw
download — the last two only when a result actually passes the static delivery
checks. It is built on the standard library and Jinja2, which are already
pinned, so it adds no dependency and needs no build step.

It composes the existing delivery path rather than reimplementing it: each
submission goes through `scripts/27_delivery.py`'s own payload, so the page and
the command line cannot disagree about what is deliverable. It loads no model,
offers no decode, never enables the placement gate, reads no frozen evaluation
case, writes nothing into `artifacts/`, and produces no metric. Operating
instructions, the failure paths and the full boundary are in [UI.md](UI.md).

The minimum two-page interface was excluded from the earlier minimum delivery
by explicit decision, and was authorised separately afterwards. It is
implemented. This public release has completed independent technical review,
covering the local interface and its delivery boundaries, not model
effectiveness.

## The full interface

```bash
./.venv/bin/python scripts/35_full_ui.py
```

Four pages, still local, still offline, still bound to loopback and nothing
else: inventory and request, photograph recognition and correction, result and
delivery, build steps. Three entries — multilingual retrieval over the
train-only catalogue, the minimum F-pipeline, and one demonstration decode with
the archived project model.

It adds pages rather than replacing them: the two-page interface above is
unchanged and still available, and the full one **subclasses** its request
handler, so every transport refusal is the same code rather than a copy of it.
What is new is one more accepted content type for a photograph, with its own
bounded parser; routes for the corrections and the build steps; and a bounded
in-process store, because a four-page flow has state.

An uploaded photograph is recognised, every detection can be corrected by
hand, and each item keeps the model's prediction, the operator's edit and the
adopted value separately, so a corrected inventory can never be mistaken for a
measured one. Colours are assigned deterministically from a `(part, colour)`
stock and can never exceed it. The build order re-verifies bounds, collisions,
stock and the accumulated structure after every step.

The project model is verified against `runs/project_model.json` before any
weight is read; nothing here retrains, tunes or reselects it. The placement
gate is opt-in, off by default, and labelled as never formally evaluated
whenever it is on. There is no Phase 3C, no frozen evaluation case, no
Success@K and no metric. Full description in [VISION.md](VISION.md).

## Image recognition, retrieval, colour and build steps

The eight-class image work, the multilingual retrieval index, the
deterministic colour assignment and the build-order planner are described in
[VISION.md](VISION.md), together with the two public image datasets they use,
their licences, the frozen vision split and what the numbers do and do not
mean. In short: it is a new task on new data with a new frozen split, it has
nothing to do with the frozen Phase 2 cases, and its results describe that
public archive and those eight classes rather than an arbitrary pile of bricks.

Two claims made in the first write-up did not survive review and are withdrawn
in that document's erratum: the detection figure labelled `AP@50` ranks a
shared localiser's boxes by a *classifier's* confidence, so it is neither a
detector comparison nor an eight-class mAP; and the checkpoint's selection
from among several configurations has no artefact on this machine to support
it. Read the erratum before quoting any detection number.

```bash
./.venv/bin/python scripts/30_vision_data.py --audit     # what would be fetched
./.venv/bin/python scripts/31_vision_split.py --verify   # the frozen split
./.venv/bin/python scripts/33_vision_eval.py --validation --classification
./.venv/bin/python scripts/34_rag_index.py --check       # retrieval readiness
```

The images themselves are not in this repository. They are read from a public
mirror into a local directory the allowlist denies, and this project does not
redistribute them.

## Repository layout

```text
src/            Core parsing, inventory, constraints, training, evaluation,
                generation, demonstration, delivery, interface, and LDraw
                modules, plus the vision, retrieval, colour and assembly
                packages
scripts/        Numbered reproducible experiments and audits
tests/          Offline and model-dependent tests
data/reports/   Aggregate experiment reports and provenance
artifacts/ldraw Example generated LDraw files
```

Not here: raw and processed data, the frozen object-level split manifest
(`data/splits/`, a per-object identifier list), per-record report JSON,
per-run session evidence, model checkpoints, credentials and local render
caches. Nor the image datasets, their per-image manifests, the frozen vision
splits, or any fitted vision weights. See the allowlist for the exact rule.

### Public snapshot history

This repository has its own root history, exported through the allowlist
rather than filtered out of the development history. It is not a rewrite of a
private history and it does not share commits with one: private development
commit identifiers and author metadata are not published, here or anywhere
else in this repository.

One consequence is worth stating plainly. Commit identifiers quoted inside the
older reports describe the development tree they were written in, and are not
resolvable here. That does not weaken those reports: what they rest on is the
stored file digests and the provenance limitations they state about
themselves, and both are unchanged.

## Environment

The recorded development environment is Python 3.13.9 on Apple Silicon with
PyTorch 2.13.0 and MPS. `requirements.txt` is an exact environment snapshot,
not a claim that every dependency combination has been tested on every
platform.

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Some model-backed scripts require separately granted access to
`meta-llama/Llama-3.2-1B-Instruct` and a local Hugging Face login. Credentials
must remain outside this repository.

## Tests

The model-free suite can be run without network access:

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ./.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Exact pass and skip counts are deliberately not written down here. They
differ between this snapshot and the private tree, and a number in a README
is a number that goes stale silently. Run the suite and read the total.

**This repository is a public snapshot, and it skips more than the private
research tree does.** Three things are worth separating:

1. **The private development tree** runs everything. It holds the processed
   instruction data, the frozen split manifest, and the per-run evidence for
   every session that has been executed.
2. **This public snapshot** publishes aggregate results only. Per-record
   report JSON, dataset identifiers, session journals, launch records,
   watchdog logs, source snapshots and failure evidence are not published, so
   the tests that read those files skip here. Every one of those skips says
   `artifact-only:` and nothing else does — run with `-rs` to see them.
3. **Tokenizer-dependent skips** are unrelated to either. The decoding,
   generation-loop and instruction tests need the `AvaLovelace/BrickGPT`
   tokenizer, which is public and ungated; those skips depend only on whether
   it can be fetched from the network or served from a local cache, and are
   not about model access approval.

A skip that does *not* carry one of those two reasons is a bug, not a
boundary.

The boundary is written down once, as an allowlist, in
[`scripts/17_public_snapshot.py`](scripts/17_public_snapshot.py): an unlisted
path is withheld rather than published, and the build refuses before copying
anything if a file is still undecided or carries an unreviewed finding. You
can run it here — `--summary`, `--scan` and `--audit` all work against this
snapshot on its own.

The release gate that *checks* the boundary is not published. It runs in the
private tree before a build: it pins the set of allowed skips by test node id,
and it builds an evidence-free tree, breaks a production rule inside it and
requires the run to fail — because tolerating absent evidence must never
become tolerating a defect. That gate asserts against evidence this snapshot
does not have, so shipping it would ship a test that can only be kept green by
ignoring it.

### What is not here

The detailed per-run evidence stays in the private research tree. This
repository carries the aggregate reports and the code that produced them, not
the rows. Where a report cites a digest of evidence that is not published, the
digest is still checkable against the private tree and is stated as such
rather than presented as reproducible from what is here.

## Data and reproducibility

The numbered scripts document the order in which the stored reports were
produced. Existing reports preserve what was actually recorded at run time and
label later reconstructions explicitly. They must not be silently upgraded
with provenance that an older run did not capture.

The model and tokenizer revisions used by the current model paths are pinned in
`src/model_ids.py`. The historical StableText2Brick runs did not record an
upstream dataset revision, so exact upstream dataset revision provenance for
those existing reports remains unavailable. Future data runs should pin and
record the dataset revision before execution.

## External data and models

- [StableText2Brick](https://huggingface.co/datasets/AvaLovelace/StableText2Brick)
  supplies the public simplified-brick dataset.
- [BrickGPT](https://github.com/AvaLovelace1/BrickGPT) supplies the published
  reference implementation and adapter used by the baseline.
- [Llama 3.2](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
  is a gated, separately licensed base model and is not redistributed here.
- [LDraw](https://www.ldraw.org/) defines the output format; the LDraw parts
  library itself is not included in this repository.
- Two public LEGO image datasets from Gdansk University of Technology's Bridge
  of Knowledge, both CC BY 4.0, supply the single-brick classification and
  multi-brick detection data:
  [10.34808/rcza-jy08](https://doi.org/10.34808/rcza-jy08) and
  [10.34808/anq4-rn44](https://doi.org/10.34808/anq4-rn44), described in
  [this paper](https://www.nature.com/articles/s41597-023-02682-2). The images
  are not redistributed here; see [VISION.md](VISION.md) for the versions,
  digests and the eight-class selection rule.
- [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) (Apache-2.0)
  is the pinned backbone the eight-class head is fitted on, and
  [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
  (MIT) is the pinned multilingual embedding for retrieval. Neither is
  redistributed here; both are pinned by revision in
  `src/vision/model_ids.py`.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing code,
data, model derivatives, or generated artifacts.

## Contributing and security

Contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md). Please report
security issues using the process in [SECURITY.md](SECURITY.md), not a public
issue containing exploit details or credentials.

## License

Original BrickAgain source code is released under the [MIT License](LICENSE).
Third-party data, model materials, formats, and adapted code remain governed by
their own licenses as documented in `THIRD_PARTY_NOTICES.md`.
