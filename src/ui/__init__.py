"""The minimum two-page local UI for the BrickAgain delivery path.

The UI is a *composition layer* and nothing more.  Retrieval, CP-SAT
re-tiling, the deterministic checks, the LDraw writer and the CPU 3-D preview
all stay where they already are: this package builds a request, hands it to
``scripts/27_delivery.py``'s own ``make_payload``, and renders whatever came
back.  There is deliberately no second copy of "what counts as deliverable"
here -- a UI that decides that for itself is how a demonstration and a command
line come to disagree while both look right.

It runs on the CPU, offline, bound to loopback only.  It loads no weights,
reaches no decoder, reads no frozen Phase 2 case, and produces no metric.
"""
