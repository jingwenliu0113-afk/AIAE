"""One structure, checked and exported: the project's demonstration.

An inventory and a text brief go in. A brick list comes out. Every
deterministic check this project has runs on it, and the result can be written
as LDraw, drawn as a layer-by-layer plan view, and passed to the separate
CPU-only 3-D geometric preview writer.

Read the following before reading any output of this module.

**It measures nothing.** No number printed here is a metric. Nothing here is
comparable to the frozen Phase 2 results, and no result of this module may be
quoted as evidence about arm B, C, D or E, about Core Success@K, or about
whether any component helps. The Phase 2 comparison ran once, on 160 cases
frozen in advance, and those cases are not touched from here.

**Where the brick list came from is recorded, not assumed.** There are exactly
three modes, they are mutually exclusive, and every report names the one that
produced it:

``sample``
    A stored brief that ships with the code: hand-written, fixed, and used
    without modification. No model runs. Its caption, inventory and
    termination are the fixture's own and cannot be overridden -- a changed
    sample is not that sample, so changing one produces a
    ``supplied_bricks`` report labelled as a variant instead.

``supplied_bricks``
    Brick text from anywhere -- a decoder somewhere else, an operator's
    keyboard, a variant of a stored brief. Nothing about it was measured
    here, so the token count is derived from the grammar and the termination
    is either what the operator stated or recorded as unavailable. It is
    never called measured, and a measured token count is refused in this
    mode rather than accepted and relabelled.

``decoded``
    This process ran a decoder. The token count and the termination are the
    ones the loop reported, and the report carries the identity of the
    weights, the device, the sampling settings and the gate configuration,
    because a decoded result without them cannot be re-run.

**The placement gate is opt-in, unevaluated, and reachable only by decoding.**
``placement`` can only be true on a ``decoded`` report, because the gate is a
property of a decode that happened -- claiming it over text that arrived some
other way would be claiming a decode nobody ran. When it is on, the report
carries the connectivity mode and the gate's own counters. The gate itself has
**never been formally evaluated**: Phase 3C is not authorised, no metric has
ever been computed with it on, and turning it on says nothing about Core
Success@K in either direction. The one relevant precedent points the other way
-- ``InventoryGate`` *lowered* the marginal ``in_bounds`` and ``collision_free``
rates in Phase 2, because constraining one axis moves the others.

**Connectivity is not support and not physics.** ``stud_only_connected`` is
2-D footprint overlap between adjacent layers. It does not check centre of
mass, moments, or whether a model stands up. The separate ``unsupported``
count is the scorer's own descriptive count of bricks with nothing directly
beneath them, reported with the scorer's own words, and it is not a stability
result either.

**Where the checks come from.** :func:`src.eval.scoring.score_generation` --
the scorer used for the frozen Phase 2 evaluation, imported and called, not
reimplemented. A second copy of "what counts as a collision" is how the demo
and the evaluation come to disagree while both look right.

**A check nobody can decide is not a failed check.** Two of the ten read the
termination, and supplied text may not have one. Those come back as ``None``
-- in the report itself, not only in the printed rendering -- because a
consumer reading the JSON has to get the same three answers a reader gets:
passed, failed, or nobody can say. Writing ``false`` there and a caveat
elsewhere would hand every machine reader a failure that never happened, in
the direction that happens to look cautious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.constraints.placement_decode import (CONNECTIVITY_MODES,
                                              PLACEMENT_STOP_REASONS)
from src.data.bricks import PART_VOCAB, WORLD, Brick, layers
from src.eval.scoring import EOS_TOKENS, score_generation
from src.generation.brickgpt import TOKENS_PER_BRICK, BrickGate, parse_output
from src.generation.prompt import build_prompt
from src.rendering.ldr import write_ldr

# ---------------------------------------------------------------------------
# The three modes, and the vocabulary of provenance
# ---------------------------------------------------------------------------

MODE_SAMPLE = "sample"
MODE_SUPPLIED = "supplied_bricks"
MODE_DECODED = "decoded"
MODES = (MODE_SAMPLE, MODE_SUPPLIED, MODE_DECODED)

#: How the token count was arrived at. Only a decode measures one.
TOKENS_MEASURED = "measured"
TOKENS_DERIVED = "derived"

#: How the termination was arrived at. ``stated`` is a fixture declaring its
#: own; ``operator-supplied`` is a person typing one; ``unavailable`` is the
#: honest answer when neither happened.
TERM_MEASURED = "measured"
TERM_STATED = "stated"
TERM_OPERATOR = "operator-supplied"
TERM_UNAVAILABLE = "unavailable"

#: Every reason a decoder in this project can stop for, from the gates
#: themselves rather than a list kept here.
TERMINATIONS: tuple[str, ...] = BrickGate.STOP_REASONS + PLACEMENT_STOP_REASONS

#: The two checks that read the termination. With none available they are
#: undeterminable, and so is the conjunction over all ten.
TERMINATION_DEPENDENT = ("termination_accepted", "deterministic_core_success")

#: Counters only :meth:`PlacementRules.counters` produces. A caller claiming
#: the gate ran has to bring these, which is why ``placement`` cannot be
#: asserted over text that was never decoded here.
PLACEMENT_COUNTER_KEYS = ("bricks_placed", "eos_deferrals",
                          "candidates_masked", "connectivity")

PLACEMENT_NOTICE = (
    "placement gate: ON (opt-in). Collision is masked, so a collision is "
    "unreachable rather than detected. This has never been formally "
    "evaluated -- Phase 3C is not authorised -- and it is not evidence that "
    "anything improved."
)

STANDING_NOTICE = (
    "This is a demonstration. It measures nothing, it is not comparable to "
    "the frozen Phase 2 evaluation, and no number here is a metric."
)


class ShowcaseError(ValueError):
    """Bad input to the demonstration. Nothing was produced."""


# ---------------------------------------------------------------------------
# Inventory, entered by hand
# ---------------------------------------------------------------------------

INVENTORY_HELP = (
    "comma-separated part:count, e.g. '2x4:10,1x2:8'. Parts are the eight in "
    f"the vocabulary: {' '.join(PART_VOCAB)}. Either spelling of a rotated "
    "part names the same stock, so 4x1 and 1x4 may not both be given."
)


def parse_inventory(spec: str) -> dict[str, int]:
    """``"2x4:10,1x2:8"`` -> ``{"2x4": 10, "1x2": 8}``.

    Rotations are normalised, because ``1x4`` and ``4x1`` are one inventory
    item and accepting both as separate entries is the silent halving this
    project has a non-negotiable decision about. Giving both is refused rather
    than summed: which of the two counts was meant is not this parser's guess
    to make.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ShowcaseError(f"an inventory is required; {INVENTORY_HELP}")
    out: dict[str, int] = {}
    seen: dict[str, str] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        part, sep, count = item.partition(":")
        if not sep:
            raise ShowcaseError(f"{item!r} is not part:count; {INVENTORY_HELP}")
        part, count = part.strip(), count.strip()
        try:
            h, _, w = part.partition("x")
            canonical = f"{min(int(h), int(w))}x{max(int(h), int(w))}"
        except ValueError:
            raise ShowcaseError(f"{part!r} is not a part name") from None
        if canonical not in PART_VOCAB:
            raise ShowcaseError(
                f"{part!r} is not one of the eight parts: "
                f"{' '.join(PART_VOCAB)}")
        if not count.isdigit():
            raise ShowcaseError(f"{count!r} is not a whole number of {part}")
        n = int(count)
        if n < 1:
            raise ShowcaseError(f"{part}:{n} stocks nothing")
        if canonical in out:
            raise ShowcaseError(
                f"{part!r} and {seen[canonical]!r} are the same part "
                f"({canonical}) and draw on the same stock; give it once")
        out[canonical] = n
        seen[canonical] = part
    if not out:
        raise ShowcaseError(f"an inventory is required; {INVENTORY_HELP}")
    return {p: out[p] for p in PART_VOCAB if p in out}


def remaining(initial: dict[str, int], used: dict[str, int]) -> dict[str, int]:
    """What is left, including the parts that went negative.

    Negative is kept rather than clamped: a demonstration that renders an
    overdraw as zero has hidden the only thing worth seeing.
    """
    return {p: initial[p] - used.get(p, 0) for p in initial}


# ---------------------------------------------------------------------------
# The plan view
# ---------------------------------------------------------------------------

SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
EMPTY = "."


def plan_view(bricks: list[Brick], *, legend: bool = True) -> str:
    """A per-layer plan of the footprint, drawn over the model's bounding box.

    One row per x, one column per y, lowest layer first -- the order the model
    is built in. Each brick gets a symbol so a reader can see which cell
    belongs to which brick, and overlapping cells print ``*`` rather than
    silently taking the last writer: an overlap is the one thing a plan view
    of a brick model must never smooth over.

    Not a render. There is no renderer in this project and this is not
    pretending to be one -- it is the occupancy, drawn.
    """
    if not bricks:
        return "(no bricks)"
    xs = [b.x for b in bricks] + [b.x + b.h - 1 for b in bricks]
    ys = [b.y for b in bricks] + [b.y + b.w - 1 for b in bricks]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    by_layer = layers(bricks)
    index = {id(b): i for i, b in enumerate(bricks)}

    out = []
    for z in sorted(by_layer):
        grid = [[EMPTY] * (y1 - y0 + 1) for _ in range(x1 - x0 + 1)]
        for b in by_layer[z]:
            mark = SYMBOLS[index[id(b)] % len(SYMBOLS)]
            for cx, cy in b.footprint:
                cell = grid[cx - x0][cy - y0]
                grid[cx - x0][cy - y0] = "*" if cell != EMPTY else mark
        out.append(f"z={z}")
        for row, cells in enumerate(grid):
            out.append(f"  x={row + x0:<3}" + " ".join(cells))
        out.append("")
    out.append(f"x {x0}..{x1} down, y {y0}..{y1} across, world {WORLD}. "
               "'*' marks a cell claimed twice.")
    if legend:
        if len(bricks) <= len(SYMBOLS):
            out.append("legend:")
            for i, b in enumerate(bricks):
                out.append(f"  {SYMBOLS[i]} {b}")
        else:
            out.append(f"legend omitted: {len(bricks)} bricks, "
                       f"{len(SYMBOLS)} symbols")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The stored briefs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A stored brief, inventory and brick list, for running the demo dry.

    **Hand-written, not generated.** These exist so the checking, export and
    plan-view path can be run and read without weights, a GPU or a network.
    Not one of them came out of a model, none is a target, and no result
    computed from one is evidence about any model.

    Frozen, and used whole. The caption, the inventory, the text and the
    termination belong together: a report on this text against a different
    inventory is a report about something else, so overriding any of them
    produces a ``supplied_bricks`` variant instead of a modified sample.
    ``termination`` is *stated* by the fixture, which is not a measurement
    and is labelled as such wherever it appears.
    """

    name: str
    caption: str
    inventory: dict[str, int]
    text: str
    termination: str
    shows: str


SAMPLES: dict[str, Sample] = {
    "tower": Sample(
        name="tower",
        caption="a small tower",
        inventory={"2x4": 4, "2x2": 2},
        text="2x4 (0,0,0)\n2x4 (0,0,1)\n2x2 (0,1,2)\n",
        termination="normal_eos",
        shows="everything passes: payable, in bounds, no overlap, connected.",
    ),
    "overdrawn": Sample(
        name="overdrawn",
        caption="a wall that costs more than the pile",
        inventory={"2x4": 2},
        text="2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)\n",
        termination="normal_eos",
        shows="three bricks against a stock of two. Everything else passes, "
              "so the overdraw is the only failure: it is shown part by "
              "part and the remaining count goes negative.",
    ),
    "collision": Sample(
        name="collision",
        caption="two bricks in the same place",
        inventory={"2x4": 4},
        text="2x4 (0,0,0)\n2x4 (0,2,0)\n",
        termination="normal_eos",
        shows="an overlap inside one layer: the plan view marks it '*' and "
              "the collision check names the pair. Connectivity fails too, "
              "and not incidentally -- same-layer contact is not a "
              "connection, so two overlapping bricks are two components.",
    ),
    "in-pieces": Sample(
        name="in-pieces",
        caption="two towers that never touch",
        inventory={"2x2": 4},
        text="2x2 (0,0,0)\n2x2 (0,0,1)\n2x2 (10,10,0)\n2x2 (10,10,1)\n",
        termination="normal_eos",
        shows="two components: legal bricks, nothing overlapping, and "
              "stud_only_connected false.",
    ),
}


def sample(name: str) -> Sample:
    if name not in SAMPLES:
        raise ShowcaseError(
            f"{name!r} is not a stored brief; have "
            f"{', '.join(sorted(SAMPLES))}")
    return SAMPLES[name]


# ---------------------------------------------------------------------------
# What a decoder reported
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decoded:
    """Everything a decode has to bring back for its report to stand up.

    Built by :func:`generate` and nowhere else in this package -- a scanning
    test asserts that -- because every field here is a claim about something
    that happened in a process, and a report claiming a decode it cannot
    describe is worse than one that admits the text arrived by other means.

    The settings are not decoration. A decoded result without the weights it
    ran on, the device, the seed, the sampling temperature and the budgets
    cannot be re-run, and a demonstration nobody can re-run is a screenshot.
    """

    n_tokens: int
    termination: str
    model: str
    device: str
    seed: int
    temperature: float
    max_bricks: int
    max_tokens: int
    placement: bool = False
    connectivity: str | None = None
    counters: dict | None = None
    adapter: str | None = None

    def __post_init__(self):
        if isinstance(self.n_tokens, bool) or \
                not isinstance(self.n_tokens, int) or self.n_tokens < 1:
            raise ShowcaseError(
                f"a decode reported n_tokens={self.n_tokens!r}; a measured "
                "count is a positive whole number")
        if self.termination not in TERMINATIONS:
            raise ShowcaseError(
                f"termination={self.termination!r} is not one of "
                f"{list(TERMINATIONS)}")
        if self.placement:
            if self.connectivity not in CONNECTIVITY_MODES:
                raise ShowcaseError(
                    "a placement-gated decode must record its connectivity "
                    f"mode; got {self.connectivity!r}")
            missing = [k for k in PLACEMENT_COUNTER_KEYS
                       if not isinstance(self.counters, dict)
                       or k not in self.counters]
            if missing:
                raise ShowcaseError(
                    "a placement-gated decode must carry the gate's own "
                    f"counters; these are missing: {missing}. Claiming the "
                    "gate ran without them is claiming a decode nobody can "
                    "check.")
        else:
            if self.connectivity is not None:
                raise ShowcaseError(
                    "connectivity describes the placement gate, and the "
                    "placement gate was not on")
            if self.counters is not None:
                raise ShowcaseError(
                    "these counters are the placement gate's own "
                    "bookkeeping, and the placement gate was not on. The "
                    "stock gate keeps none, so a report claiming both no "
                    "gate and its counters is describing two different runs.")

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "adapter": self.adapter,
            "device": self.device,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_bricks": self.max_bricks,
            "max_tokens": self.max_tokens,
            "placement": self.placement,
            "connectivity": self.connectivity,
            "gate_counters": self.counters,
        }


# ---------------------------------------------------------------------------
# Deriving what was not measured, and saying so
# ---------------------------------------------------------------------------


def tokens_if_decoded(text: str, termination: str | None) -> int:
    """What the grammar says this text would have cost, had it been decoded.

    Derived, never measured. Every brick is ten tokens, and a run that ended
    on EOS spends one more -- so the arithmetic depends on how the run ended,
    which is exactly the thing supplied text may not know. With no termination
    the body alone is counted and no EOS is assumed, because assuming one is
    assuming the answer.

    The consequence is stated rather than hidden: the scorer compares the
    token count with the brick count, so on a derived count that comparison is
    true by construction and tests nothing. On a decoded count it is the
    decoder's own arithmetic and the check is real.
    """
    bricks, _ = parse_output(text)
    body = len(bricks) * TOKENS_PER_BRICK
    ends_on_eos = termination in ("normal_eos", "inventory_exhausted")
    return body + (EOS_TOKENS if ends_on_eos else 0)


def _report(*, mode: str, caption: str, inventory: dict[str, int], text: str,
            provenance: dict, n_tokens: int, termination: str,
            undeterminable: tuple[str, ...] = (),
            decoded: Decoded | None = None) -> dict:
    """The one place a report is built. Every mode arrives here."""
    if mode not in MODES:
        raise ShowcaseError(f"mode={mode!r} is not one of {list(MODES)}")
    if not isinstance(caption, str) or not caption.strip():
        raise ShowcaseError("a caption is required")
    if not inventory:
        raise ShowcaseError("an inventory is required")

    scored = score_generation(text, inventory=inventory, n_tokens=n_tokens,
                              termination=termination)
    bricks, unparsed = parse_output(text)
    used = scored["inventory"]["used"]
    placement = bool(decoded and decoded.placement)

    # Three states, in the data. ``checks_determinable`` stays as the
    # explicit companion, but the value a caller reads first is already
    # None -- a reader who never looks at the companion cannot come away
    # with a failure this module did not observe.
    checks = {name: (None if name in undeterminable else value)
              for name, value in scored["checks"].items()}
    determinable = {name: name not in undeterminable for name in checks}

    return {
        "kind": "brickagain.showcase",
        "notice": STANDING_NOTICE,
        "placement_notice": PLACEMENT_NOTICE if placement else None,
        "provenance": {"mode": mode, **provenance,
                       "decode": decoded.as_dict() if decoded else None},
        "request": {
            "caption": caption,
            "inventory": dict(inventory),
            "placement_gate": placement,
            "prompt": build_prompt(caption, inventory),
        },
        "result": {
            "n_tokens": n_tokens,
            "termination": termination,
            "text": text,
            "n_bricks": len(bricks),
            "unparsed_lines": unparsed,
            "lines_all_parsed": not unparsed and bool(bricks),
            "token_brick_note": scored["parse"]["token_brick_note"],
            "tokens_match_complete_bricks":
                scored["parse"]["tokens_match_complete_bricks"],
        },
        "inventory": {**scored["inventory"],
                      "remaining": remaining(dict(inventory), used)},
        "checks": checks,
        "checks_determinable": determinable,
        "geometry": scored["geometry"],
        "connectivity": scored["connectivity"],
        "unsupported": scored["support"],
        "ldraw": scored["ldraw"],
        "plan_view": plan_view(bricks),
        "scored_by": "src.eval.scoring.score_generation",
    }


def inspect_sample(name: str) -> dict:
    """A stored brief, used whole. Nothing about it may be overridden."""
    brief = sample(name)
    return _report(
        mode=MODE_SAMPLE, caption=brief.caption,
        inventory=dict(brief.inventory), text=brief.text,
        n_tokens=tokens_if_decoded(brief.text, brief.termination),
        termination=brief.termination,
        provenance={
            "text_origin": f"sample:{brief.name}",
            "caption_source": MODE_SAMPLE,
            "inventory_source": MODE_SAMPLE,
            "sample": {"name": brief.name, "shows": brief.shows},
            "tokens": {"source": TOKENS_DERIVED,
                       "note": "derived from the grammar; the fixture was "
                               "never decoded, so no count was measured"},
            "termination": {"source": TERM_STATED,
                            "note": "declared by the fixture, not measured"},
            "variant_of": None,
            "changed_from_sample": None,
        })


def inspect_supplied(caption: str, inventory: dict[str, int], text: str, *,
                     origin: str, termination: str | None = None,
                     variant_of: str | None = None,
                     n_tokens: int | None = None,
                     placement: bool = False,
                     counters: dict | None = None) -> dict:
    """Brick text from elsewhere. Nothing here was measured in this process.

    ``n_tokens``, ``placement`` and ``counters`` exist only to be refused.
    They are the three claims a supplied report cannot make, and taking them
    as parameters means a caller that passes one gets an error naming it
    rather than a report quietly missing what it thought it had said.
    """
    if n_tokens is not None:
        raise ShowcaseError(
            "supplied brick text has no measured token count: nothing "
            "decoded it here. Use the decoded mode for a decode, or drop "
            "the count and let the report say it was derived.")
    if placement:
        raise ShowcaseError(
            "the placement gate is a property of a decode, and this text was "
            "not decoded here. Claiming it over supplied text would be "
            "claiming a decode nobody ran.")
    if counters is not None:
        raise ShowcaseError(
            "gate counters come from a gate that ran, and no gate ran over "
            "supplied text")
    if termination is not None and termination not in TERMINATIONS:
        raise ShowcaseError(
            f"termination={termination!r} is not one of {list(TERMINATIONS)}; "
            "leave it out to have the report record it as unavailable")

    known = termination is not None
    effective = termination if known else TERM_UNAVAILABLE
    note = ("stated by the operator, not measured: nothing here decoded this "
            "text" if known else
            "not available: the text arrived without one and this module "
            "will not assume the run ended on EOS")
    return _report(
        mode=MODE_SUPPLIED, caption=caption, inventory=dict(inventory),
        text=text, n_tokens=tokens_if_decoded(text, termination),
        termination=effective,
        undeterminable=() if known else TERMINATION_DEPENDENT,
        provenance={
            "text_origin": origin,
            "caption_source": "operator",
            "inventory_source": "operator",
            "sample": None,
            "tokens": {"source": TOKENS_DERIVED,
                       "note": "derived from the grammar; nothing decoded "
                               "this text here"},
            "termination": {"source": TERM_OPERATOR if known
                            else TERM_UNAVAILABLE, "note": note},
            "variant_of": variant_of,
            "changed_from_sample": changed_from_sample(
                variant_of, caption, inventory, termination),
        })


def changed_from_sample(name: str | None, caption, inventory,
                        termination) -> list[str] | None:
    """Which of the fixture's fields the operator replaced.

    Recorded so a variant cannot be read as the brief it started from. The
    text is the only thing a variant keeps; everything else is the operator's.
    """
    if name is None:
        return None
    brief = sample(name)
    changed = []
    if caption != brief.caption:
        changed.append("caption")
    if dict(inventory) != dict(brief.inventory):
        changed.append("inventory")
    if termination != brief.termination:
        changed.append("termination")
    return changed


def inspect_decoded(caption: str, inventory: dict[str, int], text: str, *,
                    decoded: Decoded) -> dict:
    """A decode this process ran, with everything needed to run it again."""
    if not isinstance(decoded, Decoded):
        raise ShowcaseError(
            "a decoded report needs a Decoded record; nothing else may claim "
            "a measured token count")
    return _report(
        mode=MODE_DECODED, caption=caption, inventory=dict(inventory),
        text=text, n_tokens=decoded.n_tokens, termination=decoded.termination,
        decoded=decoded,
        provenance={
            "text_origin": "decoder",
            "caption_source": "operator",
            "inventory_source": "operator",
            "sample": None,
            "tokens": {"source": TOKENS_MEASURED,
                       "note": "the count the decode loop spent"},
            "termination": {"source": TERM_MEASURED,
                            "note": "the reason the decode loop stopped"},
            "variant_of": None,
            "changed_from_sample": None,
        })


# ---------------------------------------------------------------------------
# Rendering the report
# ---------------------------------------------------------------------------

CHECK_ORDER = (
    "parse_success", "known_parts", "type_compliance", "inventory_valid",
    "in_bounds", "collision_free", "stud_only_connected", "touches_ground",
    "ldraw_serializable", "termination_accepted",
)


def _tick(report: dict, name: str) -> str:
    value = report["checks"][name]
    if value is None:
        return "n/a"
    return "pass" if value else "FAIL"


def format_report(report: dict, *, show_prompt: bool = False,
                  show_plan: bool = True) -> str:
    """The demonstration, as a reader sees it."""
    prov, req = report["provenance"], report["request"]
    res, inv = report["result"], report["inventory"]
    out = [
        "=" * 72,
        "BrickAgain demonstration",
        "=" * 72,
        report["notice"],
    ]
    if report["placement_notice"]:
        out.append(report["placement_notice"])

    out += [
        "",
        "-- provenance " + "-" * 58,
        f"  mode        : {prov['mode']}",
        f"  brick text  : {prov['text_origin']}",
        f"  caption     : from {prov['caption_source']}",
        f"  inventory   : from {prov['inventory_source']}",
        f"  tokens      : {res['n_tokens']} ({prov['tokens']['source']})"
        f" -- {prov['tokens']['note']}",
        f"  termination : {res['termination']}"
        f" ({prov['termination']['source']}) -- {prov['termination']['note']}",
    ]
    if prov["variant_of"]:
        out.append(f"  variant of  : sample:{prov['variant_of']}, operator "
                   f"replaced {', '.join(prov['changed_from_sample']) or 'nothing'}")
    if prov["sample"]:
        out.append(f"  shows       : {prov['sample']['shows']}")
    if prov["decode"]:
        d = prov["decode"]
        out += [
            f"  weights     : {d['model']}"
            + (f"  adapter {d['adapter']}" if d["adapter"] else ""),
            f"  device      : {d['device']}   seed {d['seed']}"
            f"   temperature {d['temperature']}",
            f"  budgets     : max_bricks {d['max_bricks']}"
            f"   max_tokens {d['max_tokens']}",
            f"  gate        : placement {d['placement']}"
            + (f", connectivity {d['connectivity']}" if d["placement"] else ""),
        ]

    out += [
        "",
        f"brief      : {req['caption']}",
        "inventory  : " + ", ".join(f"{p}:{n}"
                                    for p, n in req["inventory"].items()),
        "",
        "-- result " + "-" * 62,
        f"bricks     : {res['n_bricks']}",
        "",
        res["text"].rstrip("\n") or "(empty)",
    ]
    if res["unparsed_lines"]:
        out += ["", "unparsed lines:"] + [f"  {line!r}"
                                          for line in res["unparsed_lines"]]
    if res["token_brick_note"]:
        out += ["", f"token/brick note: {res['token_brick_note']}"]

    out += ["", "-- inventory " + "-" * 59]
    for part, n in req["inventory"].items():
        left = inv["remaining"][part]
        flag = "  OVERDRAWN" if left < 0 else ""
        out.append(f"  {part:<5} stocked {n:>3}   used "
                   f"{inv['used'].get(part, 0):>3}   left {left:>4}{flag}")
    if inv["type_violations"]:
        out.append("  parts used that were never stocked: "
                   + ", ".join(inv["type_violations"]))
    out.append(f"  overdrawn total: {inv['count_overflow_amount']}")

    out += ["", "-- checks " + "-" * 62]
    for name in CHECK_ORDER:
        out.append(f"  {_tick(report, name):<4}  {name}")
    out.append(f"  {_tick(report, 'deterministic_core_success'):<4}"
               "  deterministic_core_success  (all of the above)")
    undecided = [n for n, value in report["checks"].items() if value is None]
    if undecided:
        out += [
            "",
            "  n/a means this module does not have the answer, not that the "
            "answer is no:",
            f"  {', '.join(sorted(undecided))} read the termination, and "
            "none was available.",
        ]

    geo, con = report["geometry"], report["connectivity"]
    out += ["", "-- detail " + "-" * 62]
    if geo["colliding_pairs"]:
        out.append("  colliding brick pairs: "
                   + ", ".join(str(tuple(p)) for p in geo["colliding_pairs"]))
    if geo["out_of_bounds_indices"]:
        out.append(f"  out of bounds: {geo['out_of_bounds_indices']}")
    out += [
        f"  connectivity : {con['criterion']}",
        f"  components   : {con['n_components']}"
        f"   touches ground: {con['touches_ground']}",
        f"  unsupported  : {report['unsupported']['unsupported_brick_count']}"
        f" brick(s) -- {report['unsupported']['note']}",
    ]
    if report["ldraw"]["error"]:
        out.append(f"  LDraw error  : {report['ldraw']['error']}")

    if show_plan:
        out += ["", "-- plan view " + "-" * 59, report["plan_view"]]
    if show_prompt:
        out += ["", "-- prompt " + "-" * 62, req["prompt"]]
    out += ["", f"checks computed by {report['scored_by']}", ""]
    return "\n".join(out)


def passed(report: dict) -> bool | None:
    """True, False, or None when a check nobody could decide is in the way.

    Reads the value, not the companion map: the two agree by construction and
    the value is the one a JSON consumer sees.
    """
    value = report["checks"]["deterministic_core_success"]
    return None if value is None else bool(value)


def write_ldraw(report: dict, path) -> Path:
    """Write the LDraw file, refusing when the structure would not serialise."""
    if not report["ldraw"]["serializable"]:
        raise ShowcaseError(
            "this structure does not serialise to LDraw"
            + (f": {report['ldraw']['error']}" if report["ldraw"]["error"]
               else " (no bricks)"))
    bricks, _ = parse_output(report["result"]["text"])
    return write_ldr(path, bricks)


# ---------------------------------------------------------------------------
# The model path
# ---------------------------------------------------------------------------

MODEL_PUBLISHED = "published"
MODEL_PROJECT = "project"
MODELS = (MODEL_PUBLISHED, MODEL_PROJECT)

#: Where :func:`project_adapter_dir` looks. Written by ``24_project_model.py``
#: and not published: a public checkout has no ``runs/``, so the project model
#: is unavailable there and the refusal says exactly that.
POINTER = "runs/project_model.json"


def project_adapter_dir(root=None) -> Path:
    """The adapter the project model pointer names, or a refusal saying why."""
    import json

    root = Path(root or Path(__file__).resolve().parents[2])
    pointer = root / POINTER
    if not pointer.is_file():
        raise ShowcaseError(
            f"{POINTER} is not here, so the project model cannot be located. "
            "It is written by scripts/24_project_model.py in the private "
            "research tree and is not published. Use the published model, or "
            "run the demonstration on stored brick text instead.")
    body = json.loads(pointer.read_text(encoding="utf-8"))
    adapter = root / body["adapter"]["path"]
    if not adapter.is_dir():
        raise ShowcaseError(f"{POINTER} names {adapter}, which is not here")
    return adapter


def generate(caption: str, inventory: dict[str, int], *,
             model: str = MODEL_PROJECT, placement: bool = False,
             connectivity: str = "off", device: str = "mps",
             seed: int = 0, temperature: float = 0.6,
             max_bricks: int = 80, max_tokens: int = 800,
             local_files_only: bool = True, root=None) -> dict:
    """Decode one structure with real weights. Requires them to be present.

    Deliberately **not** one of the frozen arms. It loads through the same two
    loaders arms B/D and C/E use -- ``load_merged_brickgpt`` and
    ``load_finetuned(verify_digest=True)`` -- because that load order is the
    one correct one and a second spelling of it is how a demonstration ends up
    showing a model nobody trained. But it declares no arm identity, writes no
    result row and produces no metric, and nothing it returns may be quoted
    beside a Phase 2 number.

    ``placement`` routes to the placement entry point and nowhere else, and
    the gate that comes back is checked to be a placement gate before the
    report is allowed to say so. Reading the flag and trusting it would let a
    wiring mistake publish a claim about a gate that never ran.

    Imports are local: this module has to import cleanly, and its tests have
    to run, on a machine with no torch, no weights and no network.
    """
    if model not in MODELS:
        raise ShowcaseError(f"model={model!r} is not one of {list(MODELS)}")
    if connectivity not in CONNECTIVITY_MODES:
        raise ShowcaseError(
            f"connectivity={connectivity!r} is not one of "
            f"{list(CONNECTIVITY_MODES)}")
    if not placement and connectivity != "off":
        raise ShowcaseError(
            "connectivity configures the placement gate, and the placement "
            "gate was not asked for")

    import torch

    from src.constraints.inventory_decode import (InventoryGate,
                                                  generate_raw_with_inventory)
    from src.constraints.placement_decode import (InventoryPlacementGate,
                                                  PlacementGate,
                                                  generate_raw_with_placement)
    from src.generation.brickgpt import BrickGPT, load_tokenizer
    from src.inventory.engine import Inventory
    from src.model_ids import TOKENIZER, TOKENIZER_REVISION
    from src.training.lora import load_finetuned, load_merged_brickgpt

    adapter = None
    tok = load_tokenizer(TOKENIZER, TOKENIZER_REVISION,
                         local_files_only=local_files_only)
    if model == MODEL_PUBLISHED:
        weights, _ = load_merged_brickgpt(dtype=torch.bfloat16,
                                          local_files_only=local_files_only)
        weights = weights.to(device).eval()
    else:
        adapter = project_adapter_dir(root)
        weights, _ = load_finetuned(adapter, dtype=torch.bfloat16,
                                    device=device, verify_digest=True,
                                    local_files_only=local_files_only)
    gpt = BrickGPT.from_loaded(weights, tok, device=device)

    stock = Inventory.from_parts(dict(inventory))
    kw = dict(max_bricks=max_bricks, max_tokens=max_tokens,
              temperature=temperature, seed=seed)
    # Both branches decode under an inventory, so both have an exact gate
    # they must come back with. The check is not paranoia about the two
    # functions: it is what stops a wiring change from publishing a report
    # whose provenance block describes a gate that did not run.
    if placement:
        raw, gate = generate_raw_with_placement(
            gpt, caption, inventory=stock, enabled=True,
            connectivity=connectivity, **kw)
        if not isinstance(gate, InventoryPlacementGate):
            extra = (" A bare PlacementGate here would mean stock was never "
                     "enforced, which is the one guarantee this project does "
                     "not trade." if isinstance(gate, PlacementGate) else "")
            raise ShowcaseError(
                "the placement entry point returned a "
                f"{type(gate).__name__}; a decode under an inventory has to "
                f"come back with an InventoryPlacementGate.{extra}")
    else:
        raw, gate = generate_raw_with_inventory(gpt, caption, stock, **kw)
        if isinstance(gate, InventoryPlacementGate):
            raise ShowcaseError(
                "the stock entry point returned an InventoryPlacementGate; "
                "the report would say the placement gate was off while a "
                "placement gate was what ran")
        if not isinstance(gate, InventoryGate):
            raise ShowcaseError(
                f"the stock entry point returned a {type(gate).__name__}; "
                "without an InventoryGate nothing enforced the inventory, "
                "and this report is about decoding under one")

    record = Decoded(
        n_tokens=raw.n_tokens, termination=raw.termination, model=model,
        adapter=str(adapter) if adapter else None, device=device, seed=seed,
        temperature=temperature, max_bricks=max_bricks, max_tokens=max_tokens,
        placement=placement, connectivity=connectivity if placement else None,
        counters=gate.counters() if placement else None)
    return inspect_decoded(caption, dict(inventory), raw.text, decoded=record)
