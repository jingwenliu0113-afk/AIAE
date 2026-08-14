# Contributing to BrickAgain

BrickAgain is an active research prototype. Small, reviewable changes with
tests and explicit evidence are preferred.

## Before opening a change

1. Read `PROJECT_STATUS.md` and the relevant section of
   `BRICKAGAIN_PROJECT_WORKFLOW.md`.
2. Check that the proposed change stays within the eight-part core scope unless
   it is clearly labelled as an extension.
3. Do not commit raw or processed datasets, model weights, checkpoints,
   credentials, personal data, or third-party material without verified
   redistribution permission.
4. Do not rewrite stored experimental provenance or turn a later inference
   into a measurement that was not recorded at run time.

## Development setup

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  ./.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

Model-backed scripts require separately downloaded models and may require gated
access. The default test suite must remain useful without network access.

## Change requirements

- Add or update tests for behavior changes.
- Keep train/validation/test objects disjoint.
- Preserve canonical part normalization and stud-only connectivity semantics.
- Record source names, revisions, configuration, seeds, digests, and stopping
  conditions before a new experiment begins whenever they are available.
- State limitations and denominators next to reported results.
- Run the offline test command and `git diff --check` before requesting review.

For security-sensitive reports, follow `SECURITY.md` instead of opening a
public issue.
