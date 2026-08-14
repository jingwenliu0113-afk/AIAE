"""The model identities and pinned revisions, in one place.

Arms A/B/D run the published checkpoint through ``src.generation.brickgpt``;
arms C/E run a locally trained adapter through ``src.training.lora``. If those
two paths resolve *different* base weights, every A-E comparison silently
measures that difference as well as the thing under test -- and nothing about
the code would look wrong, because each side is individually consistent.

So the ids live here, imported by both. This module deliberately depends on
nothing: inference must not have to import the training package to know which
base model it is using.

Three revisions, pinned separately because they move separately:

``BASE_REVISION``
    The Llama snapshot. A LoRA delta is only meaningful against the exact
    weights it was fitted to, so this matters most for the trained arms -- but
    the untrained arms have to load the same weights or the comparison is not
    controlled.
``ADAPTER_REVISION``
    The published BrickGPT adapter.
``TOKENIZER_REVISION``
    Addressed separately from the adapter because a locally trained adapter is
    saved without tokenizer files; resolving the tokenizer from an adapter
    directory would fail for exactly the checkpoints this project produces.
"""

from __future__ import annotations

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
#: The snapshot in the local cache that every arm resolves to. Pinned so an
#: upstream push cannot move the weights under a trained adapter, and so the
#: trained and untrained arms cannot drift apart.
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"

ADAPTER = "AvaLovelace/BrickGPT"
ADAPTER_REVISION = "19737def7bfe5950b2a466825ad7c6d74b7eafe3"

#: Defaults to the published repo, which does ship tokenizer files, and can be
#: pointed elsewhere without touching the adapter.
TOKENIZER = ADAPTER
TOKENIZER_REVISION = ADAPTER_REVISION

#: Marker file written beside every locally trained adapter. Named here so
#: inference can recognise one without importing the training package.
LOCAL_ADAPTER_MANIFEST = "brickagain_manifest.json"

__all__ = [
    "BASE_MODEL", "BASE_REVISION", "ADAPTER", "ADAPTER_REVISION",
    "TOKENIZER", "TOKENIZER_REVISION", "LOCAL_ADAPTER_MANIFEST",
]
