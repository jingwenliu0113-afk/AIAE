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
- Stored reports with explicit provenance and limitations.

Still in progress:

- The formal A–E comparison is incomplete.
- Collision and support rejection are not yet integrated into the hard gate.
- The MPS slowdown study currently provides a strong short-run signal, not a
  causal result; its first run used a fixed condition order with one run per
  condition.
- This repository does not distribute Llama, BrickGPT, or locally trained
  model weights.

See [BRICKAGAIN_PROJECT_WORKFLOW.md](BRICKAGAIN_PROJECT_WORKFLOW.md) for the
research plan. The running status file is not published: it is a working
record of the private research tree, and it names evidence that is not here.
What survives of it is in the aggregate reports under `data/reports/`, which
are published in full.

## Repository layout

```text
src/            Core parsing, inventory, constraints, training, evaluation,
                generation, and LDraw modules
scripts/        Numbered reproducible experiments and audits
tests/          Offline and model-dependent tests
data/reports/   Aggregate experiment reports and provenance
artifacts/ldraw Example generated LDraw files
```

Not here: raw and processed data, the frozen object-level split manifest
(`data/splits/`, a per-object identifier list), per-record report JSON,
per-run session evidence, model checkpoints, credentials and local render
caches. See the allowlist for the exact rule.

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
