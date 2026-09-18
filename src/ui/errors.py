"""The interface's refusal type, alone, so importing it costs nothing.

``UiError`` used to live in :mod:`src.ui.app`, which imports
:mod:`src.demo.showcase` and :mod:`src.rendering.preview`.  Anything that
wanted only the exception pulled both of those into its static import closure
-- and ``scripts/62_visual_stress_pbr.py`` did, because the brick recogniser it
calls raises ``UiError``.  The V2 PBR corpus and the visual-stress runs pin
that closure by SHA-256, so editing a *notice string* in the interface moved a
render archive's source manifest and made it unverifiable, twice.

An exception class has no dependencies.  Keeping it in a leaf module is what
stops a docstring from invalidating a rendered corpus.  :mod:`src.ui.app`
re-exports it, so every existing ``from src.ui.app import UiError`` is
unchanged.
"""

from __future__ import annotations


class UiError(ValueError):
    """A submitted form the UI refuses, with a message a reader can act on.

    Distinct from :class:`~src.delivery.pipeline.DeliveryError` and
    :class:`~src.demo.showcase.ShowcaseError` only in where it was raised;
    all three are rendered the same way, and none of them ever reaches the
    browser as a traceback.
    """
