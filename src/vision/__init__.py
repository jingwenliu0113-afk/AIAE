"""Image work: the eight brick classes, from a photograph to an inventory.

Two tasks, on two public datasets, with one class list shared with the rest of
the project:

* single-brick classification -- a traditional-CV baseline and a fine-tuned
  ImageNet backbone, scored separately on real photographs and on renders;
* multi-brick detection and counting -- deterministic segmentation for the
  boxes, then the single-brick classifier on each crop.

Three boundaries hold throughout the package:

**The class list is not restated here.**  It is derived from
``src.data.bricks.PART_VOCAB`` and ``src.rendering.ldr.PART_TO_LDRAW`` in
:mod:`src.vision.classes`, so a vision label, an inventory item and an LDraw
part cannot drift apart.

**Nothing here is a Phase 2 number.**  This is a new task on new data with a
new frozen split.  No figure produced by this package may be placed beside a
Core Success@K, and nothing here touches the 160 frozen cases.

**The heavy dependencies are optional and lazily imported.**  ``torch`` and
``transformers`` are reached only inside the functions that need them, so the
parser, the inventory engine, the delivery path, the UI and their tests keep
importing and passing on a machine with neither installed.
"""
