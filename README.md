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

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the evidence-backed current
state and [BRICKAGAIN_PROJECT_WORKFLOW.md](BRICKAGAIN_PROJECT_WORKFLOW.md) for
the research plan.

## Repository layout

```text
src/            Core parsing, inventory, constraints, training, evaluation,
                generation, and LDraw modules
scripts/        Numbered reproducible experiments and audits
tests/          Offline and model-dependent tests
data/reports/   Stored experiment reports and provenance
data/splits/    Frozen object-level split manifest
artifacts/ldraw Example generated LDraw files
```

Raw/processed data, model checkpoints, Hugging Face credentials, and local
render caches are intentionally excluded from Git.

### Public snapshot history

The first public-ready snapshot was exported from the local development tree
at base commit `4544273`, together with the reviewed release-preparation
changes. The public repository intentionally starts without the local `.git`
directory so author email addresses embedded in the private development
history are not published. Historical commit identifiers stored in older
reports therefore describe the original development provenance but may not be
resolvable in the public repository; stored file digests and the reports'
explicit provenance limitations remain unchanged.

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

In the reviewed development tree this produces 550 passed and 30 skipped
tests. A clean public snapshot intentionally omits ignored processed
instruction data, so the same command there produces 548 passed and 32
skipped tests; the additional two skips are replay checks that require those
local digest inputs. The remaining skips cover optional tokenizer/cache-
dependent paths.

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
