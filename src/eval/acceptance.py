"""The core acceptance contract for arms B, C, D and E, frozen before it runs.

This is not a generic evaluation harness. It is one comparison, written down
in full before any number from it exists, because the comparison is only worth
anything if nothing about it can be chosen after the fact:

===  =====================  ================  ==============================
arm  model                  prompt            hard gate
===  =====================  ================  ==============================
B    published BrickGPT     inventory block   none
C    ``final_H2``           inventory block   none
D    published BrickGPT     inventory block   :class:`InventoryGate`
E    ``final_H2``           inventory block   the same :class:`InventoryGate`
===  =====================  ================  ==============================

Two contrasts, laid on the same 160 cases: **B - C** is what the local
fine-tune did on its own, and **D - E** is what it did once inventory
violations were made unreachable. Every other input is shared -- one
tokenizer, one set of pinned revisions, one caption, one inventory, one seed
list, one temperature, one brick budget, one token budget, one K. A setting
that varied between arms would be measured as if it were the treatment.

Five separations the rest of the module exists to hold.

**The published model is loaded merged.** Arms B and D go through
``load_merged_brickgpt()``, which is the base weights with the published
adapter merged in -- the very thing ``final_H2`` was fitted on top of. The
alternative, ``BrickGPT(adapter=ADAPTER)``, leaves the published adapter as a
live PEFT wrapper, so B/D and C/E would differ by the local delta *and* by
whether the published adapter had been merged. Arms C and E go through
``load_finetuned(..., verify_digest=True)``, the one correct cold start; the
obvious wrong spelling, ``BrickGPT(adapter=<local path>)``, lands the delta on
bare Llama and fails silently. :func:`build_interface` is the only door, and
it takes its loaders by injection so a test can prove which one was used
without a GPU anywhere near the process.

**The GPU decodes; the Mac decides.** The node produces raw text, a token
count, seconds, a termination reason and a gate ledger, and forms no opinion
about whether any of it is a brick. The parser, the checkers and the LDraw
writer run here, on the machine whose results are quoted. ``--run`` refuses to
start anywhere but WSL2 with CUDA; ``--materialize``, ``--verify`` and
``--score`` refuse to start anywhere but the Mac.

**Every number has one formula, written down before it exists.**
:data:`METRIC_SPEC` is that list -- numerator, denominator and definition for
each -- and it is inside :func:`contract_digest`. Two overflow rates are
reported because they answer different questions and folding them into one
would silently pick an answer: a macro rate over cases and a micro rate over
bricks.

**The order of the run is frozen too.** The 20 selected pairs split by index
parity into two groups of 10; one runs ``B, D, C, E`` and the other
``C, E, B, D``, so neither arm of either contrast is always the warmer one.
:data:`STEP_ORDER` is the whole schedule, the runner takes a step number and
refuses to skip one or to run one twice, and a fixed warm-up runs before every
step and is never recorded as a measurement.

**The test split is opened once, deliberately, and never travels.** The plan
carries six fields per case -- ``sample_id``, ``pair_id``, ``role``,
``variant``, ``caption``, ``inventory`` -- and :func:`plan_leak_problems`
refuses anything else, by name and by running the project's own brick parser
over every string in it. The targets stay on this machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform as _platform
from dataclasses import dataclass
from pathlib import Path

from src.data.bricks import PART_VOCAB, parse_bricks
from src.generation.prompt import build_prompt
from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                           BASE_REVISION, TOKENIZER, TOKENIZER_REVISION)
from src.training.longrun import canonical_json, digest_obj
from src.training.session import sha256_file, write_once_json

ROOT = Path(__file__).resolve().parents[2]

KIND = "brickagain.core_eval"
CONTRACT_VERSION = 2
PLAN_KIND = "brickagain.core_eval_plan"
PLAN_SCHEMA_VERSION = 2
EVIDENCE_KIND = "brickagain.core_eval_step"
EVIDENCE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# The four arms
# ---------------------------------------------------------------------------

#: The published checkpoint, with its adapter merged into the base weights.
PUBLIC_MODEL = "published_brickgpt_merged"

#: The project model. Its name, its revisions and the three digests that
#: identify it on disk are all in this contract, so a node can check the
#: weights it was handed against the plan it was handed and needs no second
#: document that the pack does not bind.
FINAL_MODEL = "final_H2"

#: The three files that make an adapter directory loadable, and what each must
#: hash to. Taken from ``runs/project_model.json``, which is how the project
#: chose this checkpoint; frozen here so the check does not depend on that
#: file travelling. A digest carried inside the contract is bound by the pack
#: manifest like every other byte of it.
FINAL_ADAPTER_SHA256: dict[str, str] = {
    "adapter_model.safetensors":
        "8cc1c8b6bb56462e378a0149ecead2769a824eacc53f83db179edb529510cbac",
    "brickagain_manifest.json":
        "1931b9d999cb689d1cfe4302aa5919bc49e7d1ab9562e580f6b6ebf0d87688fc",
    "adapter_config.json":
        "47b23f67d771556f1e0c06e31ad24b01737d926d957e9c56a9caa193fd80718d",
}

FINAL_ADAPTER_FILES: tuple[str, ...] = tuple(sorted(FINAL_ADAPTER_SHA256))

LOADER_PUBLIC = "src.training.lora.load_merged_brickgpt"
LOADER_FINAL = "src.training.lora.load_finetuned(verify_digest=True)"

#: The one prompt form every arm uses. Arm A's un-conditioned prompt is a
#: different experiment and is not part of this acceptance run.
PROMPT_FORM = "inventory"

GATE_NONE = "none"
GATE_INVENTORY = "src.constraints.inventory_decode.InventoryGate"


@dataclass(frozen=True)
class Arm:
    """One arm: which weights, which prompt, which gate. Nothing else."""

    name: str
    model: str
    loader: str
    #: Which prompt form, not a prompt. Named ``prompt_form`` because
    #: ``prompt`` is a forbidden field name -- the plan may not carry prompt
    #: text -- and a plan that named an arm's *form* ``prompt`` would trip its
    #: own leak check, which is the check working rather than a nuisance.
    prompt_form: str
    gate: str
    contrast: str

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


ARM_ORDER: tuple[str, ...] = ("B", "C", "D", "E")

ARMS: dict[str, Arm] = {
    "B": Arm("B", PUBLIC_MODEL, LOADER_PUBLIC, PROMPT_FORM, GATE_NONE,
             "the published model told about the inventory and trusted to "
             "respect it"),
    "C": Arm("C", FINAL_MODEL, LOADER_FINAL, PROMPT_FORM, GATE_NONE,
             "the same, after the local fine-tune; B - C is what training "
             "did on its own"),
    "D": Arm("D", PUBLIC_MODEL, LOADER_PUBLIC, PROMPT_FORM, GATE_INVENTORY,
             "the published model with inventory violations made "
             "unreachable"),
    "E": Arm("E", FINAL_MODEL, LOADER_FINAL, PROMPT_FORM, GATE_INVENTORY,
             "the same gate over the fine-tuned model; D - E is what "
             "training did once the gate already guaranteed the inventory"),
}

#: The two contrasts this run exists to produce, and the order of subtraction.
#: Written down so no report can quietly reverse a sign.
CONTRASTS: tuple[tuple[str, str], ...] = (("B", "C"), ("D", "E"))


def arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"{name!r} is not one of {list(ARM_ORDER)}")
    return ARMS[name]


def contrast_name(a: str, b: str) -> str:
    return f"{a}-{b}"


# ---------------------------------------------------------------------------
# The settings every arm shares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """One object, handed to all four arms.

    There is deliberately no per-arm override and no way to construct a
    second one from a command line. A field here is a thing that is the same
    in B, C, D and E, and the only way to make it differ between them is to
    edit this class -- which changes the contract digest, which every result
    row carries, which the validator checks.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3)
    temperature: float = 0.6
    max_bricks: int = 80
    #: 80 bricks x 10 tokens. Equal to the brick budget on purpose: the two
    #: ceilings coincide, so a run that stops on budget always stops on a
    #: brick boundary and never leaves a half-written brick behind.
    max_tokens: int = 800
    device: str = "cuda"
    dtype: str = "bfloat16"
    #: Explicit, never inherited from the environment. Both the published and
    #: the final model must resolve every byte from the local cache or refuse.
    local_files_only: bool = True
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    published_adapter: str = ADAPTER
    published_adapter_revision: str = ADAPTER_REVISION
    tokenizer: str = TOKENIZER
    tokenizer_revision: str = TOKENIZER_REVISION

    @property
    def k(self) -> int:
        """K is the seed count, not a second number that could disagree."""
        return len(self.seeds)

    def as_dict(self) -> dict:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["seeds"] = list(out["seeds"])
        out["k"] = self.k
        return out


SETTINGS = Settings()


def settings_for(name: str) -> dict:
    """The settings for one arm. Identical for all four, by construction."""
    arm(name)
    return SETTINGS.as_dict()


def settings_digest() -> str:
    return digest_obj(SETTINGS.as_dict())


#: The final model, as one block for the contract and the plan to carry.
def final_model_document() -> dict:
    return {
        "name": FINAL_MODEL,
        "loader": LOADER_FINAL,
        "base_model": SETTINGS.base_model,
        "base_revision": SETTINGS.base_revision,
        "published_adapter": SETTINGS.published_adapter,
        "published_adapter_revision": SETTINGS.published_adapter_revision,
        "tokenizer": SETTINGS.tokenizer,
        "tokenizer_revision": SETTINGS.tokenizer_revision,
        "adapter_files": dict(FINAL_ADAPTER_SHA256),
        "weights_travel_in_the_pack": False,
        "note": ("The weights are named on the node's command line and "
                 "checked against these three digests before anything loads. "
                 "The digests are in the contract, so they are bound by the "
                 "pack manifest; nothing here reads a pointer the pack does "
                 "not carry."),
    }


# ---------------------------------------------------------------------------
# The canonical metric specification
#
# Every number this run reports appears here with its numerator, its
# denominator and the sentence that defines it, and the whole block is inside
# contract_digest(). A metric whose definition is not in the digest is a
# metric that can be redefined after the numbers exist, which is the failure
# the frozen contract is for.
# ---------------------------------------------------------------------------

#: Quantile probabilities, frozen. Reported for timing and nothing else.
QUANTILE_PROBS: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

#: One algorithm, named and implemented here rather than borrowed. ``numpy``
#: alone offers nine, ``statistics`` two more, and they disagree by more than
#: rounding on samples this size -- so "the median seconds" is not a
#: well-defined number until this line is part of the contract.
QUANTILE_METHOD = (
    "linear interpolation on the ascending sample: h = (n - 1) * q, "
    "result = x[floor(h)] + (x[ceil(h)] - x[floor(h)]) * (h - floor(h)); "
    "n == 0 gives null, n == 1 gives that value. This is R type 7, which is "
    "numpy's default 'linear' method."
)


def quantile(values, q: float) -> float | None:
    """The frozen quantile. Pure, total, and the only one used anywhere."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q={q!r} is not a probability")
    h = (len(xs) - 1) * q
    lo, hi = math.floor(h), math.ceil(h)
    return xs[lo] + (xs[hi] - xs[lo]) * (h - lo)


def quantiles(values, probs=QUANTILE_PROBS) -> dict:
    return {f"p{int(round(p * 100)):02d}": quantile(values, p) for p in probs}


#: The termination reasons a core success may end on. ``max_bricks`` and
#: ``max_tokens`` are not failures of the model, but they are budgets running
#: out mid-thought, and a structure cut off by a ceiling is not one the model
#: said it had finished.
ACCEPTED_TERMINATIONS: tuple[str, ...] = ("normal_eos", "inventory_exhausted")

#: The checks that must all hold for one draw to be a deterministic core
#: success, in the order the scorer evaluates them. Named once; the scorer
#: reads this list rather than repeating it, so the conjunction and its
#: specification cannot drift apart.
CORE_SUCCESS_CHECKS: tuple[str, ...] = (
    "parse_success",
    "known_parts",
    "type_compliance",
    "inventory_valid",
    "in_bounds",
    "collision_free",
    "stud_only_connected",
    "touches_ground",
    "ldraw_serializable",
    "termination_accepted",
)

METRIC_SPEC: dict = {
    "parse_success": {
        "scope": "draw",
        "type": "boolean",
        "definition": (
            "at least one brick parsed, no unparsed line, and the token count "
            "agrees with the number of whole bricks: a run ending on EOS "
            "spent 10*bricks + 1 tokens, and a run ending on a budget spent a "
            "multiple of 10. A count that is neither means the decoder "
            "stopped inside a brick and is reported, never rounded."),
    },
    "known_parts": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("every parsed brick's canonical part is one of the "
                       "eight in PART_VOCAB."),
    },
    "type_compliance": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("every canonical part used appears in the case "
                       "inventory with a positive quantity. This is about "
                       "which parts, not how many."),
    },
    "count_overflow_amount": {
        "scope": "draw",
        "type": "count",
        "formula": "sum_p max(0, U_p - I_p)",
        "definition": ("bricks drawn beyond stock, summed over canonical "
                       "parts. U_p is the usage of part p in this draw and "
                       "I_p its quantity in the case inventory."),
    },
    "count_overflow_rate": {
        "scope": "draw",
        "type": "rate",
        "numerator": "sum_p max(0, U_p - I_p)",
        "denominator": "max(1, sum_p U_p)",
        "definition": ("the per-draw overflow rate, with the denominator "
                       "floored at one so a draw that used nothing is 0.0 "
                       "rather than undefined."),
    },
    "macro_count_overflow_rate": {
        "scope": "arm",
        "type": "rate",
        "numerator": "sum over draws of count_overflow_rate",
        "denominator": "number of draws",
        "definition": ("the unweighted mean of the per-draw rates. Every "
                       "draw counts once regardless of how many bricks it "
                       "emitted, so a short draw that overdrew everything is "
                       "not diluted by a long clean one."),
    },
    "micro_count_overflow_rate": {
        "scope": "arm",
        "type": "rate",
        "numerator": "sum over draws of sum_p max(0, U_p - I_p)",
        "denominator": "max(1, sum over draws of sum_p U_p)",
        "definition": ("the pooled rate over all bricks the arm emitted. "
                       "Reported beside the macro rate rather than instead "
                       "of it: they answer different questions and a single "
                       "'overflow rate' would be a silent choice between "
                       "them."),
    },
    "inventory_valid": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("type_compliance holds and count_overflow_amount is "
                       "zero. Reported separately from both, per the "
                       "workflow: the three must not be collapsed into one "
                       "inventory-compliance figure."),
    },
    "in_bounds": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("every brick satisfies Brick.in_bounds() in the 20x20"
                       "x20 world."),
    },
    "collision_free": {
        "scope": "draw",
        "type": "boolean",
        "definition": "find_collisions() returns no pair sharing a cell.",
    },
    "stud_only_connected": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("is_connected(bricks, ground=False): one component "
                       "under stud coupling between adjacent layers. The "
                       "ground is never a connector -- a baseplate is not a "
                       "part, carries no inventory and is not written out, "
                       "and joining everything resting on z == 0 would pass "
                       "an assembly that falls apart when lifted."),
    },
    "touches_ground": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("any brick at z == 0. A separate reading from "
                       "connectivity, reported on its own, and required for "
                       "core success because a structure floating in the air "
                       "is not one anybody can build."),
    },
    "ldraw_serializable": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("to_ldr() returns without raising and the draw has at "
                       "least one brick."),
    },
    "termination_accepted": {
        "scope": "draw",
        "type": "boolean",
        "definition": (f"termination is one of {list(ACCEPTED_TERMINATIONS)}."),
    },
    "deterministic_core_success": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("the conjunction of " + ", ".join(CORE_SUCCESS_CHECKS)
                       + ". Nothing else enters it: no stability, no "
                         "semantics, nothing rendered."),
    },
    "core_success_at_k": {
        "scope": "case",
        "type": "rate",
        "numerator": ("cases with at least one deterministic_core_success "
                      "among their K seeds"),
        "denominator": ("cases with all K seeds present. A case missing a "
                        "seed is reported as incomplete and is in neither "
                        "the numerator nor the denominator: 'at least one of "
                        "four' said over three seeds is a different quantity "
                        "wearing the same name."),
        "definition": "Core Success@K with K = 4.",
    },
    "unsupported_brick_count": {
        "scope": "draw",
        "type": "count",
        "definition": ("bricks above z == 0 with no brick directly beneath "
                       "them. DESCRIPTIVE ONLY. Not stability, not a physics "
                       "result, and not part of core success."),
    },
    "unsupported_brick_rate": {
        "scope": "draw",
        "type": "rate",
        "numerator": "unsupported_brick_count",
        "denominator": "max(1, number of bricks)",
        "definition": "descriptive only, as above.",
    },
    "seconds": {
        "scope": "draw",
        "type": "duration",
        "excludes": ("model load, tokenizer resolution, the warm-up, and "
                     "every check that runs before the first cell"),
        "definition": ("wall-clock seconds spent inside the decode loop for "
                       "this one draw, stored exactly as measured. Loading "
                       "the weights is NOT in it: each step is its own "
                       "process and loads from cold, so a load counted here "
                       "would be charged to whichever cell happened to be "
                       "first. Not rounded either: a rounded duration cannot "
                       "be subtracted from another and still be the "
                       "difference that was measured."),
    },
    "seconds_summary": {
        "scope": "arm",
        "type": "summary",
        "definition": ("n, total, mean, min, max and the frozen quantiles "
                       f"{list(QUANTILE_PROBS)} over that arm's draws. "
                       + QUANTILE_METHOD),
    },
    "paired_seconds_delta": {
        "scope": "contrast",
        "type": "summary",
        "definition": ("for every (case_id, seed) present in both arms, "
                       "seconds(a) - seconds(b); then n, mean and the frozen "
                       "quantiles over those differences. Paired rather than "
                       "a difference of means, because the pairing is what "
                       "removes the case-to-case variation."),
    },
    "contrast_delta": {
        "scope": "contrast",
        "type": "difference",
        "definition": ("value(a) - value(b), an absolute difference. Both "
                       "arms' raw values and both denominators are reported "
                       "beside it. No ratio, no percentage change and no "
                       "significance claim is produced: this run has one "
                       "sample of each cell and nothing here estimates a "
                       "variance."),
    },
}

#: The specification, as one value. Computed once at import because
#: :func:`contract_digest` is called for every result row the validator reads
#: and re-serialising the whole table 2,560 times says nothing new. Binding
#: the digest binds the table: any edit to a numerator, a denominator or a
#: definition moves this, which moves the contract digest.
METRIC_SPEC_DIGEST = digest_obj(METRIC_SPEC)

#: The strata every contrast is reported over. ``role`` and ``variant`` come
#: from the case, which is where the plan already carries them.
STRATA: tuple[str, ...] = ("overall", "role", "variant")

ROLES: tuple[str, ...] = ("control", "counterfactual")
VARIANTS: tuple[str, ...] = ("exact", "loose", "distractor", "mixed")


# ---------------------------------------------------------------------------
# The scorer's own source, digested
# ---------------------------------------------------------------------------

#: Every module a reported number passes through on the Mac: the parser and
#: the structural checks, the LDraw writer, the decode-layer parse entry
#: point, the scorer, and this contract. Digested so a score record says
#: which code produced it.
#:
#: Deliberately *not* inside contract_digest(). The contract digest is what a
#: plan and a result row are bound to, and a result row is raw text a GPU
#: produced -- it does not depend on the scorer at all. Folding the scorer's
#: source into it would invalidate finished GPU work every time a checker
#: gained a comment, and re-scoring stored text with corrected scorer code is
#: a legitimate act that has to stay possible. What must not be possible is
#: doing it without saying so, which is what recording and checking this
#: digest at score time prevents.
SCORER_SOURCES: tuple[str, ...] = (
    "src/data/bricks.py",
    "src/rendering/ldr.py",
    "src/generation/brickgpt.py",
    "src/eval/scoring.py",
    "src/eval/acceptance.py",
)


def scorer_manifest(root=None) -> dict:
    """``path -> sha256`` for every module a reported number passes through.

    A file that is not there gets ``None`` rather than a plausible-looking
    digest of nothing: "the digest was never taken" and "the digest matches"
    are opposite answers.
    """
    root = Path(root or ROOT)
    out = {}
    for rel in SCORER_SOURCES:
        path = root / rel
        out[rel] = sha256_file(path) if path.is_file() else None
    return out


def scorer_manifest_digest(root=None) -> str:
    return digest_obj(scorer_manifest(root))


def scorer_manifest_problems(recorded, root=None) -> list[str]:
    """Does the code on this machine still match what a record names?"""
    if not isinstance(recorded, dict):
        return [f"the scorer manifest is a {type(recorded).__name__}, not an "
                "object"]
    current = scorer_manifest(root)
    problems = []
    for rel in SCORER_SOURCES:
        want, got = recorded.get(rel), current.get(rel)
        if want is None:
            problems.append(f"{rel}: the record pins no digest for it")
        elif got is None:
            problems.append(f"{rel}: named by the record and absent here")
        elif want != got:
            problems.append(
                f"{rel}: hashes to {got[:16]}... and the record was made "
                f"against {want[:16]}...; the numbers in that record were "
                "produced by different code")
    extra = sorted(set(recorded) - set(SCORER_SOURCES))
    if extra:
        problems.append(f"the record names {extra}, which this contract does "
                        "not count as scorer source")
    return problems


# ---------------------------------------------------------------------------
# The counters that do not exist
# ---------------------------------------------------------------------------

#: The counters the workflow names and this project has not built. They are
#: reported as ``null`` with ``implemented: false`` and never as ``0``: zero
#: is a measurement saying the thing did not happen, and there is no
#: rejection layer here for it to have not happened in.
UNIMPLEMENTED_COUNTERS: tuple[str, ...] = (
    "candidate_rejections",
    "brick_retries",
    "previous_brick_backtracks",
    "physics_rollbacks",
)


def unimplemented_counters() -> dict:
    return {name: {"value": None, "implemented": False}
            for name in UNIMPLEMENTED_COUNTERS}


def counter_problems(counters) -> list[str]:
    """A counter that is present, unimplemented and numeric is a lie."""
    if not isinstance(counters, dict):
        return [f"counters is a {type(counters).__name__}, not an object"]
    problems = []
    missing = [n for n in UNIMPLEMENTED_COUNTERS if n not in counters]
    if missing:
        problems.append(f"counters does not report {missing}")
    for name in UNIMPLEMENTED_COUNTERS:
        entry = counters.get(name)
        if not isinstance(entry, dict):
            problems.append(f"counter {name} is not an object")
            continue
        if entry.get("implemented") is not False:
            problems.append(
                f"counter {name} claims implemented={entry.get('implemented')!r}; "
                "no rejection or rollback layer exists in this run")
        if entry.get("value") is not None:
            problems.append(
                f"counter {name} reports value {entry.get('value')!r}. An "
                "unimplemented counter is null; 0 would be a measurement "
                "saying the thing was counted and did not happen.")
    return problems


#: Names this run must not produce, in a key or anywhere else. Each is a
#: claim the evidence cannot support: there is no physics model, no
#: calibrated semantic threshold, and nothing is rendered.
FORBIDDEN_METRIC_TERMS: tuple[str, ...] = (
    "stability", "semantic_success", "semantic success",
    "full_success", "full success", "render_quality", "render quality",
)


# ---------------------------------------------------------------------------
# The frozen execution schedule
#
# Order is a variable. Whichever arm runs second runs on a warmer machine, a
# fuller allocator and a different cache state, and report 15 measured that
# effect at 4x on this project's own hardware. A schedule chosen while the run
# is under way is a variable nobody controlled, so it is chosen here.
#
# The 20 selected pairs split by their index in the plan: even indices are one
# group of 10, odd the other. One group runs B, D, C, E; the other C, E, B, D.
# So B precedes C in one group and follows it in the other, and the same for
# D and E -- neither arm of either contrast is systematically the warmer one.
#
# Eight steps, 320 cells each (10 pairs x 8 cases x 4 seeds), 2,560 in total.
#
# **Eight model loads, one per step.** An earlier draft of this comment said
# three, on the reasoning that consecutive steps needing the same weights are
# adjacent. They are -- but each step is its own ``--run`` invocation and its
# own process, so each one loads from cold and nothing is carried over. No
# cross-step cache is provided and none should be: a resident model would make
# a step's timings depend on which step preceded it in the same process, which
# is the confound the schedule exists to remove. Loading is not measured
# either way -- see the ``seconds`` metric, which covers the decode loop alone.
# ---------------------------------------------------------------------------

#: Every preflight gate that must be present *and* true before a step may
#: have run. Present as well as true: a check that is merely absent used to
#: pass here, so deleting a gate from the evidence was a way of satisfying it.
#:
#: The names are ``src.training.gpu_node.preflight``'s own keys, not prettier
#: synonyms -- ``offline`` rather than ``offline_env``, ``allocator_config``
#: rather than ``allocator``. A gate named something the preflight never emits
#: is a gate that can only be absent, which is exactly the hole above. A test
#: drives the real preflight and fails if any name here stops appearing.
REQUIRED_PREFLIGHT_GATES: tuple[str, ...] = (
    "platform",
    "cuda",
    "torch_cuda_build",
    "gpu_model",
    "vram",
    "system_ram",
    "offline",
    "allocator_config",
    "pack",
    "dependencies",
)

GROUPS: tuple[str, ...] = ("even", "odd")

GROUP_ARM_ORDER: dict[str, tuple[str, ...]] = {
    "even": ("B", "D", "C", "E"),
    "odd": ("C", "E", "B", "D"),
}

STEP_ORDER: tuple[tuple[str, str], ...] = tuple(
    (group, name) for group in GROUPS for name in GROUP_ARM_ORDER[group])

N_STEPS = len(STEP_ORDER)

#: A fixed warm-up before every step, on a caption that is not in the test
#: split and an inventory that is not any case's. It is decoded and thrown
#: away: no cell is written, no second is recorded as a measurement, and the
#: seconds it took go into the step's evidence explicitly marked as excluded.
#:
#: Warming on a real case would be the alternative and is worse twice over --
#: that case's first draw would be the only one measured cold, and its cell
#: would have been decoded twice with only one kept.
WARMUP: dict = {
    "caption": "A small stack of bricks, used only to warm the device.",
    "inventory": {"1x2": 4, "2x2": 4, "2x4": 4},
    "seeds": (9001, 9002),
    "generations": 2,
    "when": "immediately before the first measured cell of every step",
    "recorded_as_measurement": False,
    "policy": ("the same two decodes at the same two seeds before all eight "
               "steps, whether or not the weights were just reloaded, so "
               "every step's first measured cell meets the same warm state"),
}


def group_for_index(index: int) -> str:
    """Which group the pair at this position in the plan belongs to."""
    return GROUPS[index % len(GROUPS)]


def ordered_pair_ids(plan: dict) -> list[str]:
    """The selected pairs, in plan order, each exactly once."""
    seen: list[str] = []
    for case in plan["cases"]:
        if case["pair_id"] not in seen:
            seen.append(case["pair_id"])
    return seen


def plan_schedule(plan: dict) -> dict:
    """The whole schedule for one plan, derived rather than stored twice."""
    pairs = ordered_pair_ids(plan)
    groups = {name: [p for i, p in enumerate(pairs)
                     if group_for_index(i) == name] for name in GROUPS}
    return {
        "groups": groups,
        "group_arm_order": {k: list(v) for k, v in GROUP_ARM_ORDER.items()},
        "steps": [{"step_index": i, "group": g, "arm": a,
                   "pairs": len(groups[g])}
                  for i, (g, a) in enumerate(STEP_ORDER)],
        "warmup": {**WARMUP, "seeds": list(WARMUP["seeds"])},
        "rationale": (
            "even-indexed pairs run B, D, C, E and odd-indexed pairs run "
            "C, E, B, D, so neither arm of B - C nor of D - E is always the "
            "one that ran second."),
    }


def step(index: int) -> tuple[str, str]:
    if not isinstance(index, int) or isinstance(index, bool) \
            or not 0 <= index < N_STEPS:
        raise ValueError(f"step {index!r} is not one of 0..{N_STEPS - 1}")
    return STEP_ORDER[index]


def step_cases(plan: dict, index: int) -> list[dict]:
    """The cases one step covers: the pairs of its group, in plan order."""
    group, _name = step(index)
    wanted = set(plan_schedule(plan)["groups"][group])
    return [c for c in plan["cases"] if c["pair_id"] in wanted]


def step_cells(plan: dict, index: int, *,
               settings: Settings = SETTINGS) -> list[tuple]:
    """Every cell one step must produce, in the order it must produce them."""
    _group, name = step(index)
    digest = plan["plan_digest"]
    return [(digest, case["case_id"], name, seed)
            for case in step_cases(plan, index)
            for seed in settings.seeds]


# ---------------------------------------------------------------------------
# The frozen selection from the test split
# ---------------------------------------------------------------------------

TEST_FILE = "data/processed/instruct_inv_test.jsonl"

#: What that file must hash to. Carried here rather than computed at use
#: time: a digest taken from the file it authenticates agrees with whatever
#: the file happens to be.
EXPECTED_TEST_SHA256 = (
    "085c6900a328c1ccdb7496ae9af22ffb383038bf8c70af10213a3561a297dbbd")

SELECTION_SEED = 0
N_PAIRS = 20
ROWS_PER_PAIR = 8
N_CASES = N_PAIRS * ROWS_PER_PAIR          # 160
SELECTOR = "src.training.lora.sample_pairs"
SELECTION_RULE = ("whole pairs by seeded shuffle of sorted pair ids; every "
                  "row of a chosen pair is included")

#: Where a materialised plan lives, so the pack allowlist can name it and the
#: node can find it. Deliberately not under ``runs/``: the pack verifier
#: ignores that tree as run output, and a plan the verifier ignores is a plan
#: that can be swapped after the pack was audited.
PLAN_PATH = "gpu_plans/core_eval_plan.json"

#: The only fields a case may carry. Everything the row also has -- the
#: target, the reference brick list, the parts it actually used, the object
#: id, the token counts -- is an answer or an identifier, and none of it is
#: needed to generate.
CASE_FIELDS: tuple[str, ...] = (
    "case_id", "sample_id", "pair_id", "role", "variant", "caption",
    "inventory", "prompt_sha256", "inventory_digest",
)

#: Named as well as excluded, so a reader can see *which* answer fields were
#: thought about rather than trusting that the allowlist was complete.
FORBIDDEN_CASE_FIELDS: tuple[str, ...] = (
    "target", "bricks", "bricks_txt", "used", "object_id", "split",
    "dropped_part", "prompt", "n_tokens", "n_prompt_tokens",
    "n_target_tokens", "reference", "solution",
)

#: Every top-level key a plan has, so a plan with an extra one is refused
#: rather than carried.
PLAN_FIELDS: tuple[str, ...] = (
    "kind", "schema_version", "contract_version", "contract_digest",
    "settings_digest", "source", "arms", "settings", "final_model",
    "schedule", "scorer_source_manifest", "scorer_source_manifest_digest",
    "cases", "carries", "note", "plan_digest",
)


class PlanRefused(RuntimeError):
    """The plan was not built, and nothing was written."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_inventory(inventory: dict) -> dict:
    """Vocabulary order, positive counts only, integers.

    Fixed order so two identical inventories digest identically and the order
    a JSON file happened to use carries no signal.
    """
    if not isinstance(inventory, dict):
        raise PlanRefused(f"inventory is a {type(inventory).__name__}, not an "
                          "object")
    out = {}
    for part in PART_VOCAB:
        n = inventory.get(part, 0)
        if isinstance(n, bool) or not isinstance(n, int):
            raise PlanRefused(f"inventory quantity for {part} is {n!r}")
        if n < 0:
            raise PlanRefused(f"inventory quantity for {part} is negative")
        if n:
            out[part] = n
    unknown = sorted(set(inventory) - set(PART_VOCAB))
    if unknown:
        raise PlanRefused(f"inventory names parts outside the vocabulary: "
                          f"{unknown}")
    if not out:
        raise PlanRefused("inventory is empty")
    return out


def prompt_sha256(caption: str, inventory: dict) -> str:
    """The digest of the prompt the node must rebuild.

    The prompt text itself is not carried. The node builds it from the
    caption and the inventory with the project's own builder and reports what
    it got; if that disagrees with this, the case was not the case.
    """
    return sha256_text(build_prompt(caption, canonical_inventory(inventory)))


def build_case(row: dict) -> dict:
    """One case, from one row of the test split. Six fields plus two digests."""
    inventory = canonical_inventory(row["inventory"])
    caption = row["caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise PlanRefused(f"{row.get('sample_id')!r} has no caption")
    for field in ("sample_id", "pair_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise PlanRefused(f"a row has no {field}")
    if row.get("role") not in ROLES:
        raise PlanRefused(f"{row['sample_id']!r} has role {row.get('role')!r}")
    if row.get("variant") not in VARIANTS:
        raise PlanRefused(
            f"{row['sample_id']!r} has variant {row.get('variant')!r}")
    return {
        "case_id": row["sample_id"],
        "sample_id": row["sample_id"],
        "pair_id": row["pair_id"],
        "role": row["role"],
        "variant": row["variant"],
        "caption": caption,
        "inventory": inventory,
        "prompt_sha256": prompt_sha256(caption, inventory),
        "inventory_digest": digest_obj(inventory),
    }


def plan_leak_problems(body) -> list[str]:
    """Everything in this plan that must not leave the Mac.

    Two independent checks, because either alone fails open. The field names
    catch a column copied across wholesale; running the project's own brick
    parser over every string catches a target pasted into a field with an
    innocent name, which is the shape a mistake actually takes.
    """
    problems: list[str] = []
    forbidden = set(FORBIDDEN_CASE_FIELDS)
    allowed = set(CASE_FIELDS)

    for i, case in enumerate((body or {}).get("cases") or []):
        if not isinstance(case, dict):
            problems.append(f"case {i} is a {type(case).__name__}, not an object")
            continue
        extra = sorted(set(case) - allowed)
        if extra:
            problems.append(f"case {case.get('case_id', i)!r} carries {extra}, "
                            f"which is not among {sorted(allowed)}")
        for name in sorted(set(case) & forbidden):
            problems.append(f"case {case.get('case_id', i)!r} carries the "
                            f"answer field {name!r}")

    def walk(node, trail=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in forbidden:
                    problems.append(f"{trail}.{k} is a forbidden field".lstrip("."))
                yield from walk(v, f"{trail}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{trail}[{i}]")
        elif isinstance(node, str):
            yield trail, node

    for trail, value in walk(body or {}):
        if parse_bricks(value, strict=False):
            problems.append(
                f"{trail.lstrip('.')} parses as one or more bricks. A plan "
                "carries prompts, never answers.")
    return problems


def _case_fields_from_file(path: Path, wanted: set[str]) -> dict[str, dict]:
    """``sample_id -> {caption, inventory}`` for the chosen ids only.

    A second pass rather than a widened ``Row``: :func:`sample_pairs` is the
    frozen selection and must keep operating on exactly the rows it always
    has. Nothing but the caption and the inventory is kept from this pass.
    """
    out: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id in wanted:
                out[sample_id] = {"caption": row.get("caption"),
                                  "inventory": row.get("inventory") or {}}
    return out


def source_document(sha256: str, n_cases: int) -> dict:
    return {
        "file": TEST_FILE,
        "sha256": sha256,
        "selector": SELECTOR,
        "selection": SELECTION_RULE,
        "seed": SELECTION_SEED,
        "pairs": N_PAIRS,
        "rows_per_pair": ROWS_PER_PAIR,
        "cases": n_cases,
        "roles": list(ROLES),
        "variants": list(VARIANTS),
    }


def materialize_plan(root=None) -> dict:
    """Open the test split once and turn 20 whole pairs into 160 cases.

    Refuses before reading a row if the file does not hash to
    :data:`EXPECTED_TEST_SHA256`, and refuses after building if anything about
    the result is not exactly what the contract describes -- including its own
    ``plan_problems``, so nothing is written that could not then be read back.
    """
    from src.training.lora import read_rows, sample_pairs

    root = Path(root or ROOT)
    path = root / TEST_FILE
    if not path.is_file():
        raise PlanRefused(f"{TEST_FILE} is not on this machine")

    actual = sha256_file(path)
    if actual != EXPECTED_TEST_SHA256:
        raise PlanRefused(
            f"{TEST_FILE} hashes to {actual[:16]}..., not the "
            f"{EXPECTED_TEST_SHA256[:16]}... this contract pins. The frozen "
            "selection would name different rows; refusing to open it.")

    rows = sample_pairs(read_rows(path), n_pairs=N_PAIRS, seed=SELECTION_SEED)
    if len(rows) != N_CASES:
        raise PlanRefused(
            f"the frozen selection produced {len(rows)} rows, not {N_CASES}")

    extra = _case_fields_from_file(path, {r.sample_id for r in rows})
    cases = []
    for row in rows:
        found = extra.get(row.sample_id)
        if not found:
            raise PlanRefused(f"{row.sample_id} vanished between the two passes")
        cases.append(build_case({
            "sample_id": row.sample_id, "pair_id": row.pair_id,
            "role": row.role, "variant": row.variant,
            "caption": found["caption"], "inventory": found["inventory"],
        }))

    body = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": contract_digest(),
        "settings_digest": settings_digest(),
        "source": source_document(actual, len(cases)),
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "settings": SETTINGS.as_dict(),
        "final_model": final_model_document(),
        # The scorer that was approved for this plan, frozen with it. The
        # cost is deliberate and worth stating: this is inside plan_digest,
        # which every result row carries, so editing any of the five modules
        # invalidates a materialised plan and the cells measured against it.
        # Re-materialising is then the only honest way forward, because a
        # number produced by a different parser is a different number.
        # ``ROOT``, not the ``root`` the data was read from. The manifest
        # describes the code that is running, which lives where this module
        # lives; a test that points ``root`` at a temporary tree is still
        # scored by the checkers in this repository.
        "scorer_source_manifest": scorer_manifest(ROOT),
        "scorer_source_manifest_digest": scorer_manifest_digest(ROOT),
        "cases": cases,
        "carries": list(CASE_FIELDS),
        "note": ("Prompts only. The targets, the reference brick lists and "
                 "the parts they use stay on the Mac and are not in this "
                 "file."),
    }
    body["schedule"] = plan_schedule(body)
    body["plan_digest"] = plan_digest(body)

    problems = plan_problems(body)
    if problems:
        raise PlanRefused("refusing to write the plan:\n  - "
                          + "\n  - ".join(problems))
    return body


def plan_digest(body: dict) -> str:
    """One value over everything that decides what will be generated."""
    return digest_obj({
        "kind": body.get("kind"),
        "schema_version": body.get("schema_version"),
        "contract_digest": body.get("contract_digest"),
        "settings_digest": body.get("settings_digest"),
        "source": body.get("source"),
        "arms": body.get("arms"),
        "final_model": body.get("final_model"),
        "schedule": body.get("schedule"),
        "scorer_source_manifest": body.get("scorer_source_manifest"),
        "cases": body.get("cases"),
    })


def write_plan(path, body: dict) -> str:
    """Write once. A second plan at the same name is a refusal, not an update."""
    return write_once_json(Path(path), body)


def read_plan(path) -> dict:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = plan_problems(body)
    if problems:
        raise PlanRefused("this plan does not describe the frozen contract:\n"
                          "  - " + "\n  - ".join(problems))
    return body


def _case_problems(case, index) -> list[str]:
    """One case, against the schema the contract freezes for it."""
    where = f"case {case.get('case_id', index)!r}"
    if not isinstance(case, dict):
        return [f"case {index} is a {type(case).__name__}, not an object"]

    problems = []
    got, want = set(case), set(CASE_FIELDS)
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        problems.append(f"{where} has fields {sorted(got)}, not exactly "
                        f"{sorted(want)}"
                        + (f"; missing {missing}" if missing else "")
                        + (f"; unexpected {extra}" if extra else ""))
        if missing:
            return problems

    for field in ("case_id", "sample_id", "pair_id", "caption"):
        value = case.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{where}: {field} is {value!r}, which is not a "
                            "non-empty string")
    if isinstance(case.get("case_id"), str) \
            and case.get("case_id") != case.get("sample_id"):
        problems.append(
            f"{where}: case_id and sample_id differ ({case.get('case_id')!r} "
            f"vs {case.get('sample_id')!r}). They are one identifier written "
            "twice; two values means a result row can be attributed to a row "
            "of the split it did not come from.")
    if case.get("role") not in ROLES:
        problems.append(f"{where}: role {case.get('role')!r} is not one of "
                        f"{list(ROLES)}")
    if case.get("variant") not in VARIANTS:
        problems.append(f"{where}: variant {case.get('variant')!r} is not one "
                        f"of {list(VARIANTS)}")

    try:
        inventory = canonical_inventory(case.get("inventory"))
    except PlanRefused as exc:
        return problems + [f"{where}: {exc}"]
    if case["inventory"] != inventory:
        problems.append(f"{where}: the inventory is not in canonical form "
                        "(vocabulary order, positive integer quantities)")
    if case.get("inventory_digest") != digest_obj(inventory):
        problems.append(f"{where}: records an inventory digest its own "
                        "inventory does not produce")
    if isinstance(case.get("caption"), str) and case["caption"].strip():
        if case.get("prompt_sha256") != prompt_sha256(case["caption"],
                                                      inventory):
            problems.append(f"{where}: records a prompt digest that its own "
                            "caption and inventory do not produce")
    return problems


def plan_problems(body) -> list[str]:
    """Everything that stops a file from being the plan this contract means.

    Every check, every time -- a plan that is wrong in three ways is worth
    knowing about in three ways. The list is long because the alternative is
    a plan that is checked for the two things somebody remembered.
    """
    if not isinstance(body, dict):
        return [f"the plan is a {type(body).__name__}, not an object"]
    problems: list[str] = []

    got, want = set(body), set(PLAN_FIELDS)
    if got != want:
        problems.append(f"the plan's top-level fields are {sorted(got)}, not "
                        f"exactly {sorted(want)}")
    if body.get("kind") != PLAN_KIND:
        problems.append(f"kind is {body.get('kind')!r}, not {PLAN_KIND!r}")
    if body.get("schema_version") != PLAN_SCHEMA_VERSION:
        problems.append(f"schema_version is {body.get('schema_version')!r}, "
                        f"not {PLAN_SCHEMA_VERSION}")
    if body.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version is {body.get('contract_version')!r}, "
                        f"not {CONTRACT_VERSION}")
    if body.get("contract_digest") != contract_digest():
        problems.append(
            f"the plan was made against contract "
            f"{str(body.get('contract_digest'))[:16]}..., and this code is "
            f"{contract_digest()[:16]}...")
    if body.get("settings_digest") != settings_digest():
        problems.append("the plan's settings digest is not the one this code "
                        "would run")

    # The frozen blocks, compared whole. A field-by-field comparison would
    # pass a plan that gained an arm, or an arm that gained a field.
    if body.get("arms") != {name: ARMS[name].as_dict() for name in ARM_ORDER}:
        problems.append("the plan's arm definitions are not the four this "
                        "contract freezes")
    if body.get("settings") != SETTINGS.as_dict():
        problems.append("the plan's settings block is not the frozen one")
    if body.get("final_model") != final_model_document():
        problems.append("the plan's final_model block is not the frozen one; "
                        "the node checks the adapter against this, so a "
                        "drifted block is a check against the wrong digests")
    if body.get("carries") != list(CASE_FIELDS):
        problems.append(f"the plan says it carries {body.get('carries')!r}, "
                        f"not {list(CASE_FIELDS)}")

    # Self-consistency only. Whether the scorer on *this* machine still
    # matches is a question for score time, and asking it here would make a
    # plan unreadable on the node the moment a Mac-side checker gained a
    # comment -- including for ``--run``, which never scores anything.
    manifest = body.get("scorer_source_manifest")
    if not isinstance(manifest, dict) or set(manifest) != set(SCORER_SOURCES):
        problems.append(
            f"the plan's scorer_source_manifest names "
            f"{sorted(manifest) if isinstance(manifest, dict) else manifest!r}, "
            f"not the {list(SCORER_SOURCES)} this contract counts as scorer "
            "source")
    elif any(not isinstance(v, str) or len(v) != 64 for v in manifest.values()):
        problems.append("the plan's scorer_source_manifest holds something "
                        "that is not a digest")
    elif body.get("scorer_source_manifest_digest") != digest_obj(manifest):
        problems.append("the plan records a scorer_source_manifest_digest its "
                        "own manifest does not produce")

    cases = body.get("cases")
    if not isinstance(cases, list) or len(cases) != N_CASES:
        problems.append(
            f"the plan holds "
            f"{len(cases) if isinstance(cases, list) else '?'} cases, not "
            f"{N_CASES}")
        return problems + plan_leak_problems(body)

    source = body.get("source")
    if source != source_document(str((source or {}).get("sha256")),
                                 len(cases)):
        problems.append(
            "the plan's source block is not the frozen selection metadata "
            f"(expected file, selector, rule, seed {SELECTION_SEED}, "
            f"{N_PAIRS} pairs of {ROWS_PER_PAIR}, {N_CASES} cases, and the "
            "declared roles and variants)")
    if (source or {}).get("sha256") != EXPECTED_TEST_SHA256:
        problems.append("the plan was made from a test file this contract "
                        "does not pin")

    for i, case in enumerate(cases):
        problems += _case_problems(case, i)

    ids = [c.get("case_id") for c in cases if isinstance(c, dict)]
    if len(set(ids)) != len(ids):
        problems.append("two cases share a case_id")

    pairs: dict[str, list[dict]] = {}
    for case in cases:
        if isinstance(case, dict):
            pairs.setdefault(case.get("pair_id"), []).append(case)
    if len(pairs) != N_PAIRS:
        problems.append(f"the plan holds {len(pairs)} pairs, not {N_PAIRS}")
    expected_pair = {(role, variant) for role in ROLES for variant in VARIANTS}
    for pair_id in sorted(pairs, key=str):
        members = pairs[pair_id]
        if len(members) != ROWS_PER_PAIR:
            problems.append(f"pair {pair_id!r} has {len(members)} cases, not "
                            f"{ROWS_PER_PAIR}")
            continue
        composition = {(c.get("role"), c.get("variant")) for c in members}
        if composition != expected_pair:
            problems.append(
                f"pair {pair_id!r} is not the {len(ROLES)} roles x "
                f"{len(VARIANTS)} variants the dataset builds; a pair that is "
                "eight rows of the wrong eight is a split pair with the right "
                "count")

    if body.get("schedule") != plan_schedule(body):
        problems.append(
            "the plan's schedule is not the one this contract derives from "
            "its own case order; the run order is frozen, not recorded")

    if body.get("plan_digest") != plan_digest(body):
        problems.append("plan_digest does not recompute from the plan")
    return problems + plan_leak_problems(body)


def plan_scorer_problems(plan: dict, root=None) -> list[str]:
    """Is the scorer on this machine the one this plan was approved with?

    Asked once, immediately before scoring. ``plan_problems`` deliberately
    checks the manifest only against itself, because the node reads the same
    plan and never scores; this is the half that compares it against the
    files that are about to produce the numbers.
    """
    recorded = (plan or {}).get("scorer_source_manifest")
    problems = scorer_manifest_problems(recorded, root)
    if problems:
        return ["the scorer this plan was made with is not the scorer on "
                "this machine:"] + problems
    if plan.get("scorer_source_manifest_digest") != scorer_manifest_digest(root):
        return [f"the plan pins scorer manifest "
                f"{str(plan.get('scorer_source_manifest_digest'))[:16]}... and "
                f"this machine's is {scorer_manifest_digest(root)[:16]}..."]
    return []


def case_index(body: dict) -> dict[str, dict]:
    return {c["case_id"]: c for c in body["cases"]}


# ---------------------------------------------------------------------------
# The contract document
# ---------------------------------------------------------------------------

def contract_document(root=None) -> dict:
    """The whole contract, as one object a report can embed verbatim."""
    return {
        "kind": KIND,
        "contract_version": CONTRACT_VERSION,
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "arm_order": list(ARM_ORDER),
        "contrasts": [contrast_name(a, b) for a, b in CONTRASTS],
        "shared_settings": SETTINGS.as_dict(),
        "settings_are_identical_across_arms": True,
        "final_model": final_model_document(),
        "selection": {
            **source_document(EXPECTED_TEST_SHA256, N_CASES),
            "expected_sha256": EXPECTED_TEST_SHA256,
        },
        "schedule": {
            "steps": [{"step_index": i, "group": g, "arm": a}
                      for i, (g, a) in enumerate(STEP_ORDER)],
            "group_arm_order": {k: list(v)
                                for k, v in GROUP_ARM_ORDER.items()},
            "grouping": ("the selected pairs in plan order, split by index "
                         "parity into two groups of 10"),
            "cells_per_step": len(GROUPS) and
                              (N_PAIRS // len(GROUPS)) * ROWS_PER_PAIR
                              * SETTINGS.k,
            "warmup": {**WARMUP, "seeds": list(WARMUP["seeds"])},
            "model_loads": (
                "one per step, eight in total. Each step is its own process "
                "and loads from cold; no model is cached across steps, "
                "deliberately, because a resident model would make a step's "
                "timings depend on which step preceded it in the same "
                "process. Load time is not part of the reported seconds."),
            "runner_refuses": ["a step out of order",
                               "a step whose predecessor has all its cells "
                               "but no completion record",
                               "a step that already has cells, without "
                               "--resume",
                               "a step that is complete and sealed",
                               "any cell of a later step existing first"],
            "evidence": (
                "one immutable attempt record per attempt, written after the "
                "preflight and the warm-up and before the first cell; every "
                "cell names its attempt; one completion record per step, "
                "listing every attempt. A step whose cells are complete but "
                "which was never sealed may be sealed without decoding "
                "anything."),
        },
        "plan_carries": list(CASE_FIELDS),
        "plan_refuses": list(FORBIDDEN_CASE_FIELDS),
        "metrics": METRIC_SPEC,
        "strata": list(STRATA),
        "quantiles": {"probabilities": list(QUANTILE_PROBS),
                      "method": QUANTILE_METHOD},
        "scoring": {
            "where": "Mac; the node decodes and does not parse",
            "connectivity": "stud coupling only, ground=False",
            "touches_ground": "reported separately and required for core success",
            "accepted_terminations": list(ACCEPTED_TERMINATIONS),
            "core_success_checks": list(CORE_SUCCESS_CHECKS),
            "unimplemented_counters": list(UNIMPLEMENTED_COUNTERS),
            "required_preflight_gates": list(REQUIRED_PREFLIGHT_GATES),
            "not_reported": list(FORBIDDEN_METRIC_TERMS),
            "source_manifest": scorer_manifest(root),
            "source_manifest_digest": scorer_manifest_digest(root),
            "source_manifest_note": (
                "the modules every reported number passes through, digested. "
                "Deliberately outside contract_digest: a result row is raw "
                "text a GPU produced and does not depend on the scorer, so "
                "folding this in would invalidate finished GPU work whenever "
                "a checker gained a comment. It is recorded in every score "
                "record and re-checked when one is read."),
        },
        "platforms": {
            "run": "WSL2 with CUDA; no CPU and no MPS fallback",
            "materialize": "Mac only",
            "verify": "Mac only",
            "score": "Mac only",
        },
    }


def contract_digest() -> str:
    """Everything that decides what the four arms are and what is computed.

    The prose does not enter it; the metric specification does. A run whose
    numerator changed after the fact would otherwise carry the same digest as
    the run it is compared against.
    """
    return digest_obj({
        "kind": KIND,
        "contract_version": CONTRACT_VERSION,
        "arms": {name: {k: v for k, v in ARMS[name].as_dict().items()
                        if k != "contrast"} for name in ARM_ORDER},
        "contrasts": [list(c) for c in CONTRASTS],
        "settings": SETTINGS.as_dict(),
        "final_model": {"name": FINAL_MODEL,
                        "adapter_files": dict(FINAL_ADAPTER_SHA256)},
        "selection": {"file": TEST_FILE, "sha256": EXPECTED_TEST_SHA256,
                      "selector": SELECTOR, "rule": SELECTION_RULE,
                      "seed": SELECTION_SEED, "pairs": N_PAIRS,
                      "rows_per_pair": ROWS_PER_PAIR, "cases": N_CASES,
                      "roles": list(ROLES), "variants": list(VARIANTS)},
        "schedule": {"steps": [list(s) for s in STEP_ORDER],
                     "group_arm_order": {k: list(v)
                                         for k, v in GROUP_ARM_ORDER.items()},
                     "warmup": {**WARMUP, "seeds": list(WARMUP["seeds"])}},
        "metrics_digest": METRIC_SPEC_DIGEST,
        "strata": list(STRATA),
        "quantiles": {"probabilities": list(QUANTILE_PROBS),
                      "method": QUANTILE_METHOD},
        "core_success_checks": list(CORE_SUCCESS_CHECKS),
        "accepted_terminations": list(ACCEPTED_TERMINATIONS),
        "unimplemented_counters": list(UNIMPLEMENTED_COUNTERS),
        "required_preflight_gates": list(REQUIRED_PREFLIGHT_GATES),
        "case_fields": list(CASE_FIELDS),
    })


# ---------------------------------------------------------------------------
# Which machine may do which thing
# ---------------------------------------------------------------------------

class DeviceRefused(RuntimeError):
    """The device this run requires is not here, and there is no substitute."""


class PlatformRefused(RuntimeError):
    """This machine is not the machine this stage runs on."""


MAC_SYSTEM = "Darwin"

#: The stages that belong on the Mac, and the one that does not. Split by
#: what each needs rather than by preference: decoding needs the GPU, and
#: everything else needs the machine whose results are quoted.
MAC_ONLY_MODES: tuple[str, ...] = ("materialize", "verify", "score")
NODE_ONLY_MODES: tuple[str, ...] = ("run",)


def resolve_device(torch_mod) -> str:
    """CUDA or nothing.

    No CPU fallback and no MPS fallback. A run that quietly changes device has
    quietly become a different experiment, and its numbers will be compared
    against ones from the experiment it stopped being -- which is the same
    rule ``src.training.gpu_node`` applies to training, for the same reason.
    """
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise DeviceRefused(
            f"this run requires {SETTINGS.device!r} and it is not available. "
            "There is no CPU or MPS fallback: B/C/D/E must be decoded on one "
            "device or they are not comparable.")
    return SETTINGS.device


def mac_only_problems(mode: str, *, system=None) -> list[str]:
    """``materialize``/``verify``/``score`` run here and nowhere else.

    Not a preference. ``--materialize`` opens the test split, which never
    leaves this machine; ``--score`` produces the numbers the report quotes
    and must produce them under one parser, one checker and one LDraw writer.
    Running either on the node would put a second answer on a second machine.
    """
    if mode not in MAC_ONLY_MODES:
        return [f"{mode!r} is not one of the Mac-only stages "
                f"{list(MAC_ONLY_MODES)}"]
    system = _platform.system() if system is None else system
    if system != MAC_SYSTEM:
        return [f"--{mode} runs on the Mac only, and this is {system!r}. The "
                "execution node decodes; it does not open the test split and "
                "it does not score."]
    return []


def node_only_problems(mode: str, probe_reading) -> list[str]:
    """``--run`` needs WSL2 and CUDA, both read rather than assumed."""
    if mode not in NODE_ONLY_MODES:
        return [f"{mode!r} is not one of the node-only stages "
                f"{list(NODE_ONLY_MODES)}"]
    p = probe_reading or {}
    problems = []
    system = p.get("os_system")
    if system is None:
        problems.append("the operating system could not be read, and a check "
                        "that cannot be evaluated has not been satisfied")
    elif system != "Linux":
        problems.append(f"--run executes on the node, which is WSL2 Ubuntu; "
                        f"this is {system!r}")
    elif p.get("wsl2") is None:
        problems.append("the kernel release could not be read, so this cannot "
                        "be shown to be WSL2")
    elif not p.get("wsl2"):
        problems.append("this is Linux but not WSL2: "
                        + str(p.get("wsl_evidence") or "the kernel release "
                              "does not name a Microsoft build"))
    if p.get("torch_cuda_build") in (None, ""):
        problems.append("this torch has no CUDA build, so it can only run on "
                        "the CPU. There is no CPU fallback here.")
    if not p.get("cuda_available"):
        problems.append("CUDA is not available; a run on another device is a "
                        "different experiment")
    return problems


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def final_adapter_problems(adapter_dir, *, expected=None) -> list[str]:
    """The three digests the contract pins, against the directory on disk.

    ``load_finetuned`` already checks the adapter against the manifest beside
    it. That is a different question: it proves the checkpoint is internally
    consistent, and a directory somebody rebuilt is internally consistent too.
    This asks whether it is the checkpoint the contract named -- and the
    contract travels inside the pack manifest, so the answer is bound to the
    pack the node verified rather than to a pointer file nobody digested.
    """
    expected = dict(expected or FINAL_ADAPTER_SHA256)
    problems: list[str] = []
    missing = [n for n in FINAL_ADAPTER_FILES if n not in expected]
    if missing:
        problems.append(f"the contract pins no digest for {missing}")

    directory = Path(adapter_dir)
    if not directory.is_dir():
        return problems + [f"{directory.name!r} is not a directory on this "
                           "machine"]
    for name in FINAL_ADAPTER_FILES:
        want = expected.get(name)
        blob = directory / name
        if not blob.is_file():
            problems.append(f"{name} is missing from the adapter directory")
            continue
        if want is None:
            continue
        got = sha256_file(blob)
        if got != want:
            problems.append(
                f"{name} hashes to {got[:16]}..., not the {want[:16]}... this "
                f"contract pins for {FINAL_MODEL}")
    return problems


def plan_final_adapter_problems(plan: dict, adapter_dir) -> list[str]:
    """The same check, driven by the plan the node was handed.

    The node reads the plan out of the pack it verified, so this is the whole
    chain: pack digest carried by hand, plan digested inside the manifest,
    adapter digests inside the plan, weights on disk checked against them.
    """
    block = (plan or {}).get("final_model") or {}
    if block != final_model_document():
        return ["the plan's final_model block is not the frozen one; refusing "
                "to check the weights against digests that already drifted"]
    return final_adapter_problems(adapter_dir,
                                  expected=block.get("adapter_files"))


def default_loaders() -> dict:
    """The real loaders, resolved late so importing this module loads nothing."""
    from src.generation.brickgpt import BrickGPT, load_tokenizer
    from src.training.lora import load_finetuned, load_merged_brickgpt

    return {
        "tokenizer": load_tokenizer,
        "merged": load_merged_brickgpt,
        "finetuned": load_finetuned,
        "interface": BrickGPT.from_loaded,
    }


def build_interface(name: str, *, device: str, adapter_dir=None,
                    loaders=None, dtype=None):
    """The one door to a loaded arm. Returns ``(interface, info)``.

    ``BrickGPT(adapter=...)`` is never called from here, for either kind of
    arm. For B/D it would leave the published adapter unmerged, so B/D and
    C/E would differ by that as well as by the local delta; for C/E it would
    land the local delta on bare Llama, which loads, generates and is wrong.

    ``loaders`` is injected so a test can assert *which* loader ran, with
    which arguments, without a GPU, a network or a checkpoint.
    """
    spec = arm(name)
    loaders = loaders or default_loaders()
    if dtype is None:                       # resolved late, like the loaders
        import torch

        dtype = getattr(torch, SETTINGS.dtype)

    tok = loaders["tokenizer"](SETTINGS.tokenizer, SETTINGS.tokenizer_revision,
                               local_files_only=SETTINGS.local_files_only)

    if spec.model == PUBLIC_MODEL:
        model, info = loaders["merged"](
            dtype=dtype, local_files_only=SETTINGS.local_files_only)
        model = model.to(device).eval()
    else:
        if adapter_dir is None:
            raise ValueError(
                f"arm {name} runs {FINAL_MODEL} and no adapter directory was "
                "given; the weights do not travel in the pack and must be "
                "named explicitly")
        model, info = loaders["finetuned"](
            adapter_dir, dtype=dtype, device=device, verify_digest=True,
            local_files_only=SETTINGS.local_files_only)

    return loaders["interface"](model, tok, device=device), info


def model_identity(name: str) -> dict:
    """What a result row must say about the weights that produced it.

    Derived from the contract rather than passed in. A row therefore cannot
    record digests of its own choosing and still validate: the expected value
    is the frozen one, on both sides of the comparison.
    """
    spec = arm(name)
    return {
        "model": spec.model,
        "loader": spec.loader,
        "base_model": SETTINGS.base_model,
        "base_revision": SETTINGS.base_revision,
        "published_adapter": SETTINGS.published_adapter,
        "published_adapter_revision": SETTINGS.published_adapter_revision,
        "adapter_files": (dict(FINAL_ADAPTER_SHA256)
                          if spec.model != PUBLIC_MODEL else None),
    }


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------

def gate_ledger(spec: Arm, opening: dict, gate=None) -> dict:
    """What the node reports about the gate, and nothing it had to parse.

    ``accepted_parts`` is the gate's own ledger and exists only where there is
    a gate. For B and C it is ``null`` rather than ``[]``: an empty list would
    say the gate accepted nothing, and there was no gate.
    """
    accepted = None
    remaining = None
    if spec.gate != GATE_NONE and gate is not None:
        accepted = list(getattr(gate, "accepted", []))
        remaining = getattr(gate, "inventory").as_dict()
    return {
        "gate": spec.gate,
        "opening_inventory": dict(opening),
        "accepted_parts": accepted,
        "remaining_inventory": remaining,
        "counters": unimplemented_counters(),
    }


def _decode(interface, spec: Arm, caption: str, opening: dict, kw):
    """One decode, gated or not, through the one paired path either way."""
    from src.constraints.inventory_decode import generate_raw_with_inventory
    from src.inventory.engine import Inventory

    if spec.gate == GATE_NONE:
        return interface.generate_raw(caption, inventory=opening, **kw), None
    return generate_raw_with_inventory(
        interface, caption, Inventory.from_parts(dict(opening)), **kw)


def warm_up(interface, name: str, *, settings: Settings = SETTINGS) -> dict:
    """The frozen warm-up. Decoded, timed, and never a measurement.

    Returned so the step's evidence can say it happened and how long it took;
    nothing here writes a cell, and the seconds below appear in no arm
    summary and no paired delta.
    """
    spec = arm(name)
    opening = canonical_inventory(WARMUP["inventory"])
    seconds = []
    for seed in WARMUP["seeds"]:
        raw, _gate = _decode(interface, spec, WARMUP["caption"], opening, {
            "max_bricks": settings.max_bricks,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature, "seed": seed})
        seconds.append(float(raw.seconds))
    return {
        "generations": len(seconds),
        "seeds": list(WARMUP["seeds"]),
        "caption_is_from_the_test_split": False,
        "seconds": seconds,
        "excluded_from_every_reported_number": True,
        "policy": WARMUP["policy"],
    }


def run_case(interface, case: dict, name: str, seed: int, *,
             settings: Settings = SETTINGS, plan_digest_value: str = "",
             step_index: int | None = None, group: str | None = None,
             attempt_id: str | None = None,
             attempt_digest: str | None = None) -> dict:
    """Decode one (case, arm, seed) cell. Raw text out; nothing parsed.

    The inventory reaches the model twice -- as the block it reads and, for
    D/E, as the counter the gate spends -- and
    :func:`generate_raw_with_inventory` is what keeps those the same numbers.
    """
    spec = arm(name)
    opening = canonical_inventory(case["inventory"])
    raw, gate = _decode(interface, spec, case["caption"], opening, {
        "max_bricks": settings.max_bricks, "max_tokens": settings.max_tokens,
        "temperature": settings.temperature, "seed": seed})

    built = build_prompt(case["caption"], opening)
    return {
        "plan_digest": plan_digest_value,
        "case_id": case["case_id"],
        "arm": name,
        "seed": seed,
        "step_index": step_index,
        "group": group,
        # Which run of this step produced the cell. A resume opens a new
        # attempt and the cells already on disk keep the id of the attempt
        # that actually measured them -- they are never rewritten, so they
        # cannot be re-attributed to the process that came along afterwards.
        "attempt_id": attempt_id,
        "attempt_digest": attempt_digest,
        "raw_text": raw.text,
        "n_tokens": raw.n_tokens,
        # Stored exactly as measured. A rounded duration cannot be subtracted
        # from another and still be the difference that was measured, and
        # every timing number this run reports is a difference.
        "seconds": float(raw.seconds),
        "termination": raw.termination,
        "truncated": bool(raw.truncated),
        "gate": gate_ledger(spec, opening, gate),
        "prompt_sha256": sha256_text(built),
        "inventory_digest": digest_obj(opening),
        "contract_digest": contract_digest(),
        "settings_digest": settings_digest(),
        "model": model_identity(name),
    }


# ---------------------------------------------------------------------------
# Results: append-only, one cell each, no second opinion about any of them
# ---------------------------------------------------------------------------

RESULT_FIELDS: tuple[str, ...] = (
    "plan_digest", "case_id", "arm", "seed", "step_index", "group",
    "attempt_id", "attempt_digest",
    "raw_text", "n_tokens", "seconds", "termination", "truncated", "gate",
    "prompt_sha256", "inventory_digest", "contract_digest", "settings_digest",
    "model",
)

#: Every termination the decoder can report. The test suite pins this list to
#: ``BrickGate.STOP_REASONS`` rather than importing it, because that lives
#: behind torch and this module loads nothing.
TERMINATIONS: tuple[str, ...] = (
    "normal_eos", "inventory_exhausted", "max_bricks", "max_tokens")


class ResultsRefused(RuntimeError):
    """A write that would have replaced, duplicated or reordered a measurement."""


def cell_key(row: dict) -> tuple:
    return (row.get("plan_digest"), row.get("case_id"), row.get("arm"),
            row.get("seed"))


def read_cells(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def read_cells_lenient(path) -> tuple[list[dict], list[str]]:
    """Every readable row, plus a complaint for each that is not.

    The validator needs to *report* a truncated line rather than die on it: a
    process killed mid-append leaves exactly that, and "the results file
    cannot be parsed" is the least useful possible description of it.
    """
    path = Path(path)
    if not path.is_file():
        return [], [f"{path.name} does not exist"]
    rows, problems = [], []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {i} is not readable as JSON ({exc}); a "
                            "partial line is a run that stopped mid-append")
            continue
        if not isinstance(row, dict):
            problems.append(f"line {i} is a {type(row).__name__}, not an object")
            continue
        rows.append(row)
    return rows, problems


def results_problems(path) -> list[str]:
    """Everything that stops this results file from being readable at all.

    Separate from :func:`validate_results` and called before it, because the
    callers that must not proceed on a damaged file are the ones that never
    run the full validator: the runner reaches ``known_keys`` -- and so
    ``read_cells``, which *raises* -- before it has checked anything, and the
    traceback it produced was after the arguments were accepted and before
    anything said why.
    """
    if not Path(path).is_file():
        # Not damage. Before step 0 writes its first cell there is no file,
        # and a guard that refused that would refuse every run at its start.
        # Absence is caught where it means something: as missing cells.
        return []
    _rows, problems = read_cells_lenient(path)
    return problems


def known_keys(path) -> set:
    """The keys already on disk, for a caller that is about to append many.

    Raises on a damaged file, deliberately: every caller is expected to have
    run :func:`results_problems` first, and silently skipping an unreadable
    line here would let a run append a cell that already exists.
    """
    return {cell_key(r) for r in read_cells(path)}


def append_cell(path, row: dict, *, known: set | None = None) -> None:
    """Append one cell, or refuse. There is no third outcome.

    ``known`` is the caller's own set of keys, seeded from the file and
    updated here. Without it every append re-reads the whole file, which is
    correct and quadratic -- 320 cells a step, each carrying up to a kilobyte
    of raw text. With it a long run stays linear and the answer is the same,
    because the set starts as exactly what the file contained.

    Neither form defends against a second process appending concurrently; the
    check-then-append window exists either way. What defends against that is
    :func:`validate_results`, which re-reads everything afterwards and reports
    a duplicate as a duplicate.
    """
    path = Path(path)
    missing = [f for f in RESULT_FIELDS if f not in row]
    if missing:
        raise ResultsRefused(f"a result row without {missing} is not a "
                             "measurement of anything")
    key = cell_key(row)
    seen = known if known is not None else {cell_key(r)
                                            for r in read_cells(path)}
    if key in seen:
        raise ResultsRefused(
            f"{key} is already recorded. A cell is measured once; "
            "replacing it would swap one measurement for another with "
            "nothing in the file saying so.")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json({f: row[f] for f in RESULT_FIELDS}) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    seen.add(key)
    return None


def expected_cells(plan: dict, *, arms=ARM_ORDER,
                   settings: Settings = SETTINGS) -> list[tuple]:
    """Every cell this plan predetermines, in schedule order."""
    wanted = set(arms)
    out: list[tuple] = []
    for index in range(N_STEPS):
        _group, name = STEP_ORDER[index]
        if name in wanted:
            out += step_cells(plan, index, settings=settings)
    return out


def missing_cells(path, plan: dict, *, arms=ARM_ORDER,
                  settings: Settings = SETTINGS) -> list[tuple]:
    """What a resume may still do, and nothing else.

    A resume fills gaps. It never re-runs a cell that is there and never
    overwrites one, so the set it is allowed to touch is exactly this.
    """
    have = known_keys(path)
    return [key for key in expected_cells(plan, arms=arms, settings=settings)
            if key not in have]


def result_row_problems(row: dict, plan_cases: dict, *, arms,
                        settings: Settings) -> list[str]:
    """One row, against the plan and the contract it claims to belong to."""
    problems: list[str] = []
    where = f"{row.get('case_id')!r}/{row.get('arm')}/seed {row.get('seed')}"

    missing = [f for f in RESULT_FIELDS if f not in row]
    if missing:
        return [f"{where}: missing {missing}"]
    extra = sorted(set(row) - set(RESULT_FIELDS))
    if extra:
        problems.append(f"{where}: carries unexpected fields {extra}")

    name = row["arm"]
    if name not in arms:
        problems.append(f"{where}: arm {name!r} is not one of {list(arms)}")
        return problems
    if row["seed"] not in settings.seeds:
        problems.append(f"{where}: seed {row['seed']!r} is not one of "
                        f"{list(settings.seeds)}")
    case = plan_cases.get(row["case_id"])
    if case is None:
        problems.append(f"{where}: no such case in the plan")
        return problems

    if row["contract_digest"] != contract_digest():
        problems.append(f"{where}: was produced under a different contract")
    if row["settings_digest"] != settings_digest():
        problems.append(f"{where}: was produced under different settings")
    if row["prompt_sha256"] != case["prompt_sha256"]:
        problems.append(f"{where}: the prompt it generated from is not the "
                        "prompt the plan pins for this case")
    if row["inventory_digest"] != case["inventory_digest"]:
        problems.append(f"{where}: the inventory it was given is not the "
                        "inventory the plan pins for this case")
    if row["model"] != model_identity(name):
        problems.append(
            f"{where}: the weights it records are not the weights arm {name} "
            "is defined to run")

    if not isinstance(row["raw_text"], str):
        problems.append(f"{where}: raw_text is not text")
    if isinstance(row["n_tokens"], bool) or not isinstance(row["n_tokens"], int) \
            or row["n_tokens"] < 1:
        problems.append(f"{where}: n_tokens is {row['n_tokens']!r}")
    if isinstance(row["seconds"], bool) or \
            not isinstance(row["seconds"], (int, float)) or row["seconds"] < 0:
        problems.append(f"{where}: seconds is {row['seconds']!r}")
    if row["termination"] not in TERMINATIONS:
        problems.append(f"{where}: termination {row['termination']!r} is not "
                        f"one of {list(TERMINATIONS)}")
    if not isinstance(row["truncated"], bool):
        problems.append(f"{where}: truncated is not a boolean")
    if not isinstance(row["attempt_id"], str) or not row["attempt_id"].strip():
        problems.append(f"{where}: names no attempt, so nothing says which "
                        "run of this step measured it")
    if not isinstance(row["attempt_digest"], str) \
            or len(row["attempt_digest"]) != 64:
        problems.append(f"{where}: attempt_digest is "
                        f"{row['attempt_digest']!r}, not a digest")

    ledger = row["gate"]
    if not isinstance(ledger, dict):
        problems.append(f"{where}: the gate ledger is not an object")
    else:
        spec = ARMS[name]
        if ledger.get("gate") != spec.gate:
            problems.append(
                f"{where}: ran under gate {ledger.get('gate')!r} and arm "
                f"{name} is defined with {spec.gate!r}")
        if ledger.get("opening_inventory") != case["inventory"]:
            problems.append(f"{where}: the gate opened on an inventory the "
                            "plan does not pin")
        accepted = ledger.get("accepted_parts")
        if spec.gate == GATE_NONE and accepted is not None:
            problems.append(f"{where}: arm {name} has no gate, so its ledger "
                            "cannot list accepted parts")
        if spec.gate != GATE_NONE and not isinstance(accepted, list):
            problems.append(f"{where}: a gated arm must report the parts its "
                            "gate accepted")
        problems += [f"{where}: {p}"
                     for p in counter_problems(ledger.get("counters"))]
    return problems


def _schedule_problems_for_rows(rows, plan: dict, *, arms,
                                settings: Settings) -> list[str]:
    """Did each row come from the step the schedule assigns it?"""
    placement = {}
    for index in range(N_STEPS):
        group, name = STEP_ORDER[index]
        if name not in set(arms):
            continue
        for key in step_cells(plan, index, settings=settings):
            placement[key] = (index, group)

    problems = []
    for row in rows:
        key = cell_key(row)
        want = placement.get(key)
        if want is None:
            continue                       # reported elsewhere as unexpected
        index, group = want
        if row.get("step_index") != index or row.get("group") != group:
            problems.append(
                f"{row.get('case_id')!r}/{row.get('arm')}/seed "
                f"{row.get('seed')}: records step "
                f"{row.get('step_index')!r}/{row.get('group')!r} and the "
                f"frozen schedule puts it in step {index}/{group!r}")
    return problems


def validate_results(path, plan: dict, *, arms=ARM_ORDER,
                     settings: Settings = SETTINGS) -> list[str]:
    """Everything wrong with a results file, against the plan it claims.

    Seven refusals: a cell that is missing, a cell recorded twice, a cell no
    plan predetermined, an arm or a setting that drifted, a prompt or
    inventory digest that is not the one this case pins, a row that is not a
    complete measurement, and a row attributed to a step the frozen schedule
    does not put it in. None of them can be switched off, and a run that is
    merely unfinished is reported as unfinished rather than scored.
    """
    problems = list(plan_problems(plan))
    rows, read_problems = read_cells_lenient(path)
    problems += read_problems

    cases = case_index(plan) if not problems or "cases" in plan else {}
    seen: dict[tuple, int] = {}
    for row in rows:
        problems += result_row_problems(row, cases, arms=arms,
                                        settings=settings)
        if row.get("plan_digest") != plan.get("plan_digest"):
            problems.append(
                f"{row.get('case_id')!r}/{row.get('arm')}/seed "
                f"{row.get('seed')}: belongs to plan "
                f"{str(row.get('plan_digest'))[:16]}..., not this one")
            continue
        key = cell_key(row)
        seen[key] = seen.get(key, 0) + 1

    for key, count in sorted(seen.items(), key=lambda kv: str(kv[0])):
        if count > 1:
            problems.append(f"{key[1]!r}/{key[2]}/seed {key[3]} is recorded "
                            f"{count} times")

    predetermined = set(expected_cells(plan, arms=arms, settings=settings))
    for key in sorted(set(seen) - predetermined, key=str):
        problems.append(f"{key[1]!r}/{key[2]}/seed {key[3]} is not a cell this "
                        "plan predetermines")
    absent = [key for key in expected_cells(plan, arms=arms, settings=settings)
              if key not in seen]
    if absent:
        shown = ", ".join(f"{k[1]}/{k[2]}/seed {k[3]}" for k in absent[:5])
        problems.append(
            f"{len(absent)} predetermined cells were never measured "
            f"(e.g. {shown}). An incomplete grid is not a result.")

    problems += _schedule_problems_for_rows(rows, plan, arms=arms,
                                            settings=settings)
    return problems


# ---------------------------------------------------------------------------
# Evidence: attempts, and the completion that closes a step
#
# A step is not one event. It is one or more *attempts* -- a process that
# passed the preflight, warmed the device and then wrote cells until it
# finished or died -- and one *completion*, written when every cell of the
# step exists.
#
# That shape is forced by what a resume actually is. The single record the
# first draft wrote could only be written at the end, so a step killed at cell
# 200 left 200 measured cells and nothing at all saying which machine, which
# pack or which preflight produced them; and the record the resume eventually
# wrote described its own process, silently claiming all 320. Now the attempt
# record is written immutably *before the first cell*, every cell names the
# attempt that measured it, and a resume opens a new attempt rather than
# adopting the old cells.
# ---------------------------------------------------------------------------

ATTEMPT_KIND = "brickagain.core_eval_attempt"
COMPLETION_KIND = "brickagain.core_eval_step"
EVIDENCE_SCHEMA_VERSION = 2

ATTEMPT_FIELDS: tuple[str, ...] = (
    "kind", "schema_version", "attempt_id", "attempt_index", "step_index",
    "group", "arm", "plan_digest", "contract_digest", "settings_digest",
    "pack_digest", "dependency_digest", "preflight", "platform",
    "provenance", "warmup", "adapter", "cells_missing_at_start",
    "started_at",
)

COMPLETION_FIELDS: tuple[str, ...] = (
    "kind", "schema_version", "step_index", "group", "arm", "plan_digest",
    "contract_digest", "settings_digest", "cells_expected", "cells_recorded",
    "attempts", "sealed_at", "sealed_without_decoding",
)

ATTEMPT_REFERENCE_FIELDS: tuple[str, ...] = (
    "attempt_id", "attempt_digest", "cells_written", "started_at",
    "pack_digest", "dependency_digest",
)


def package_provenance() -> dict:
    """The versions every number here was produced under."""
    from importlib.metadata import version

    def maybe(name):
        try:
            return version(name)
        except Exception:
            return None

    return {"python": _platform.python_version(),
            "torch": maybe("torch"), "transformers": maybe("transformers"),
            "peft": maybe("peft"), "accelerate": maybe("accelerate")}


def step_slug(index: int) -> str:
    group, name = step(index)
    return f"step_{index:02d}_{group}_{name}"


def completion_path(directory, index: int) -> Path:
    return Path(directory) / f"{step_slug(index)}.json"


def attempt_path(directory, index: int, attempt_index: int) -> Path:
    return Path(directory) / f"{step_slug(index)}_attempt_{attempt_index:02d}.json"


def attempt_id_for(index: int, attempt_index: int) -> str:
    return f"{step_slug(index)}#attempt{attempt_index:02d}"


def existing_attempts(directory, index: int) -> list[Path]:
    """Every attempt file for one step, in the order they were opened."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    prefix = f"{step_slug(index)}_attempt_"
    return sorted(p for p in directory.iterdir()
                  if p.name.startswith(prefix) and p.name.endswith(".json"))


def next_attempt_index(directory, index: int) -> int:
    return len(existing_attempts(directory, index))


def attempt_digest(body: dict) -> str:
    """The attempt record as one value, for a result row to point at."""
    return digest_obj({f: body.get(f) for f in ATTEMPT_FIELDS})


def build_attempt_evidence(*, index: int, attempt_index: int, plan: dict,
                           probe_reading: dict, preflight_result: dict,
                           pack_digest: str, dependency_digest: str,
                           warmup: dict, adapter_dir=None,
                           cells_missing_at_start: int,
                           started_at: str) -> dict:
    """Everything an attempt has proved, written before it measures anything.

    The two digests are the ones the operator carried by hand. They are not
    recomputed from anything the pack supplied, because a digest stored with
    the thing it authenticates is rewritten by whoever rewrote that thing --
    the node's own preflight is what compares them against the machine.
    """
    group, name = step(index)
    p = probe_reading or {}
    return {
        "kind": ATTEMPT_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "attempt_id": attempt_id_for(index, attempt_index),
        "attempt_index": attempt_index,
        "step_index": index,
        "group": group,
        "arm": name,
        "plan_digest": plan["plan_digest"],
        "contract_digest": contract_digest(),
        "settings_digest": settings_digest(),
        "pack_digest": pack_digest,
        "dependency_digest": dependency_digest,
        "preflight": {
            "passed": bool(preflight_result.get("passed")),
            "failed": list(preflight_result.get("failed") or []),
            "checks": {k: v.get("passed")
                       for k, v in (preflight_result.get("checks")
                                    or {}).items()},
        },
        "platform": {
            "os_system": p.get("os_system"), "wsl2": p.get("wsl2"),
            "device": SETTINGS.device, "dtype": SETTINGS.dtype,
            "cuda_available": p.get("cuda_available"),
            "torch_cuda_build": p.get("torch_cuda_build"),
            "gpu_name": p.get("gpu_name"),
            "vram_total_gb": p.get("vram_total_gb"),
            "system_ram_gb": p.get("system_ram_gb"),
            "allocator_backend": p.get("allocator_backend"),
            "offline_env": p.get("offline_env"),
        },
        "provenance": package_provenance(),
        "warmup": warmup,
        "adapter": ({"directory_name": Path(adapter_dir).name,
                     "files": dict(FINAL_ADAPTER_SHA256),
                     "checked_against": "the plan's final_model block"}
                    if adapter_dir is not None else None),
        "cells_missing_at_start": cells_missing_at_start,
        "started_at": started_at,
    }


def attempt_problems(body, plan: dict, *, expected_pack_digest,
                     expected_dependency_digest) -> list[str]:
    """Is this an attempt that may be believed?

    Both carried digests are required and have no defaults, for the reason
    ``gpu_node.preflight`` gives: a trust check with a default is a trust
    check somebody forgets to pass, and it fails open when they do.
    """
    from src.training import pack as pack_module

    if not isinstance(body, dict):
        return [f"the attempt evidence is a {type(body).__name__}, not an "
                "object"]
    problems: list[str] = []
    got, want = set(body), set(ATTEMPT_FIELDS)
    if got != want:
        problems.append(f"the attempt fields are {sorted(got)}, not exactly "
                        f"{sorted(want)}")
    if body.get("kind") != ATTEMPT_KIND:
        problems.append(f"kind is {body.get('kind')!r}, not {ATTEMPT_KIND!r}")
    if body.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        problems.append(f"schema_version is {body.get('schema_version')!r}")

    index = body.get("step_index")
    try:
        group, name = step(index)
    except ValueError as exc:
        problems.append(str(exc))
        group = name = None
    if group is not None:
        if body.get("group") != group or body.get("arm") != name:
            problems.append(
                f"step {index} is {group}/{name} in the frozen schedule and "
                f"this attempt says {body.get('group')!r}/{body.get('arm')!r}")
        attempt_index = body.get("attempt_index")
        if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) \
                or attempt_index < 0:
            problems.append(f"attempt_index is {attempt_index!r}")
        elif body.get("attempt_id") != attempt_id_for(index, attempt_index):
            problems.append(
                f"attempt_id {body.get('attempt_id')!r} is not the id step "
                f"{index} attempt {attempt_index} is given")

    if body.get("plan_digest") != plan.get("plan_digest"):
        problems.append("the attempt names a different plan")
    if body.get("contract_digest") != contract_digest():
        problems.append("the attempt was written under a different contract")
    if body.get("settings_digest") != settings_digest():
        problems.append("the attempt was written under different settings")

    problems += [f"pack digest: {p}" for p in
                 pack_module.expected_digest_problems(expected_pack_digest)]
    problems += [f"dependency digest: {p}" for p in
                 pack_module.expected_digest_problems(
                     expected_dependency_digest, what="dependency digest")]
    if body.get("pack_digest") != expected_pack_digest:
        problems.append(
            f"the attempt ran against pack "
            f"{str(body.get('pack_digest'))[:16]}..., not the digest carried "
            "from the build machine")
    if body.get("dependency_digest") != expected_dependency_digest:
        problems.append(
            "the attempt ran against dependencies "
            f"{str(body.get('dependency_digest'))[:16]}..., not the digest "
            "carried from the build machine")

    problems += _preflight_problems(body.get("preflight"))
    problems += _platform_problems(body.get("platform"))

    provenance = body.get("provenance") or {}
    for required in ("python", "torch", "transformers", "peft"):
        if not provenance.get(required):
            problems.append(f"the attempt records no {required} version")

    problems += _warmup_problems(body.get("warmup"))

    if name is not None:
        adapter = body.get("adapter")
        if ARMS[name].model == PUBLIC_MODEL:
            # Both directions, because only one of them was checked before.
            # B and D load the published weights and no local delta at all,
            # so an adapter recorded here does not mean "extra provenance":
            # it means a local checkpoint was on the machine and something
            # pointed at it during a step that must not have used one.
            if adapter is not None:
                problems.append(
                    f"step {index} runs the published model and the attempt "
                    f"records an adapter ({adapter!r}). Arms B and D take no "
                    "local delta; an adapter here is the wrong weights.")
        elif not isinstance(adapter, dict):
            problems.append(f"step {index} runs {FINAL_MODEL} and the attempt "
                            "records no adapter")
        elif adapter.get("files") != dict(FINAL_ADAPTER_SHA256):
            problems.append("the adapter digests recorded are not the three "
                            "this contract pins")

    missing = body.get("cells_missing_at_start")
    if not isinstance(missing, int) or isinstance(missing, bool) or missing < 0:
        problems.append(f"cells_missing_at_start is {missing!r}")
    if not isinstance(body.get("started_at"), str) \
            or not body.get("started_at"):
        problems.append("the attempt records no start time")
    return problems


def _preflight_problems(result) -> list[str]:
    """Every required gate present, and every one of them true.

    Present as well as true. The first version checked ``if gate in checks``,
    so an evidence file that simply did not mention a gate satisfied it --
    deleting a check was the cheapest way to pass.
    """
    if not isinstance(result, dict):
        return [f"the preflight record is a {type(result).__name__}, not an "
                "object"]
    problems = []
    if result.get("passed") is not True:
        problems.append(
            f"the node's preflight did not pass; it failed "
            f"{result.get('failed')!r}. A gate that did not pass is not a "
            "gate that was waived.")
    checks = result.get("checks")
    if not isinstance(checks, dict):
        return problems + ["the preflight records no checks at all"]
    for gate in REQUIRED_PREFLIGHT_GATES:
        if gate not in checks:
            problems.append(
                f"the preflight record does not contain the gate {gate!r}. An "
                "absent gate is not a satisfied one; this is the whole reason "
                "the required set is written down.")
        elif checks[gate] is not True:
            problems.append(f"the preflight check {gate!r} did not pass")
    return problems


def _platform_problems(block) -> list[str]:
    if not isinstance(block, dict):
        return [f"the platform record is a {type(block).__name__}, not an "
                "object"]
    problems = []
    if block.get("os_system") != "Linux" or block.get("wsl2") is not True:
        problems.append("the attempt did not run on WSL2")
    if block.get("cuda_available") is not True:
        problems.append("the attempt did not run with CUDA available")
    if not block.get("torch_cuda_build"):
        problems.append("the torch that ran the attempt has no CUDA build")
    if block.get("device") != SETTINGS.device \
            or block.get("dtype") != SETTINGS.dtype:
        problems.append(
            f"the attempt records device {block.get('device')!r} and dtype "
            f"{block.get('dtype')!r}, not {SETTINGS.device!r}/"
            f"{SETTINGS.dtype!r}")
    return problems


def _warmup_problems(warmup) -> list[str]:
    if not isinstance(warmup, dict):
        return [f"the warm-up record is a {type(warmup).__name__}, not an "
                "object"]
    problems = []
    if warmup.get("generations") != WARMUP["generations"]:
        problems.append(
            f"the attempt records {warmup.get('generations')!r} warm-up "
            f"generations and the policy is {WARMUP['generations']}")
    if warmup.get("excluded_from_every_reported_number") is not True:
        problems.append("the attempt does not record its warm-up as excluded")
    if warmup.get("caption_is_from_the_test_split") is not False:
        problems.append("the warm-up caption is not declared as coming from "
                        "outside the test split")
    return problems


# ---------------------------------------------------------------------------
# Completion: the record that closes a step
# ---------------------------------------------------------------------------

def attempt_reference(body: dict, cells_written: int) -> dict:
    """The completion's entry for one attempt, derived from that attempt.

    Every field is read with ``.get``. A JSON file that parses and is missing
    ``attempt_id`` is a thing that happens -- a process killed mid-write, an
    edit by hand -- and it has to come out as a refusal from the validator
    rather than as a ``KeyError`` from the sealer, which would abort the seal
    with a traceback and no explanation.
    """
    body = body if isinstance(body, dict) else {}
    return {
        "attempt_id": body.get("attempt_id"),
        "attempt_digest": attempt_digest(body),
        "cells_written": cells_written,
        "started_at": body.get("started_at"),
        "pack_digest": body.get("pack_digest"),
        "dependency_digest": body.get("dependency_digest"),
    }


def build_step_completion(*, index: int, plan: dict, attempts: list[dict],
                          cells_recorded: int, sealed_at: str,
                          sealed_without_decoding: bool,
                          settings: Settings = SETTINGS) -> dict:
    """Every attempt this step took, and the fact that it is now closed."""
    group, name = step(index)
    return {
        "kind": COMPLETION_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "step_index": index,
        "group": group,
        "arm": name,
        "plan_digest": plan["plan_digest"],
        "contract_digest": contract_digest(),
        "settings_digest": settings_digest(),
        "cells_expected": len(step_cells(plan, index, settings=settings)),
        "cells_recorded": cells_recorded,
        "attempts": attempts,
        "sealed_at": sealed_at,
        # True when this invocation decoded nothing: the cells were already
        # complete and only the closing record was missing. Sealing is then a
        # read and a write, never a re-measurement.
        "sealed_without_decoding": sealed_without_decoding,
    }


def completion_problems(body, plan: dict, *, index: int,
                        settings: Settings = SETTINGS) -> list[str]:
    if not isinstance(body, dict):
        return [f"the completion record is a {type(body).__name__}, not an "
                "object"]
    problems = []
    got, want = set(body), set(COMPLETION_FIELDS)
    if got != want:
        problems.append(f"the completion fields are {sorted(got)}, not "
                        f"exactly {sorted(want)}")
    if body.get("kind") != COMPLETION_KIND:
        problems.append(f"kind is {body.get('kind')!r}, not "
                        f"{COMPLETION_KIND!r}")
    if body.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        problems.append(f"schema_version is {body.get('schema_version')!r}")

    group, name = step(index)
    if body.get("step_index") != index or body.get("group") != group \
            or body.get("arm") != name:
        problems.append(
            f"the completion says step {body.get('step_index')!r} "
            f"{body.get('group')!r}/{body.get('arm')!r} and the frozen "
            f"schedule says {index} {group}/{name}")
    if body.get("plan_digest") != plan.get("plan_digest"):
        problems.append("the completion names a different plan")
    if body.get("contract_digest") != contract_digest():
        problems.append("the completion was written under a different contract")
    if body.get("settings_digest") != settings_digest():
        problems.append("the completion was written under different settings")

    expected = len(step_cells(plan, index, settings=settings))
    if body.get("cells_expected") != expected:
        problems.append(f"the completion expects {body.get('cells_expected')!r} "
                        f"cells and the step is {expected}")
    if body.get("cells_recorded") != expected:
        problems.append(
            f"the completion records {body.get('cells_recorded')!r} of "
            f"{expected} cells; a step is closed when it is complete")

    attempts = body.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return problems + ["the completion lists no attempts, so nothing says "
                           "which process measured these cells"]
    ids = []
    total = 0
    for i, reference in enumerate(attempts):
        if not isinstance(reference, dict):
            problems.append(f"attempt {i} in the completion is not an object")
            continue
        if set(reference) != set(ATTEMPT_REFERENCE_FIELDS):
            problems.append(f"attempt {i} in the completion has fields "
                            f"{sorted(reference)}, not "
                            f"{sorted(ATTEMPT_REFERENCE_FIELDS)}")
            continue
        ids.append(reference["attempt_id"])
        written = reference["cells_written"]
        if not isinstance(written, int) or isinstance(written, bool) \
                or written < 0:
            problems.append(f"attempt {reference['attempt_id']!r} records "
                            f"cells_written {written!r}")
        else:
            total += written
    if len(set(ids)) != len(ids):
        problems.append("two attempts in the completion share an id")
    if total != body.get("cells_recorded"):
        problems.append(
            f"the attempts account for {total} cells and the completion "
            f"records {body.get('cells_recorded')!r}; every cell belongs to "
            "exactly one attempt")
    if not isinstance(body.get("sealed_at"), str) or not body.get("sealed_at"):
        problems.append("the completion records no seal time")
    if not isinstance(body.get("sealed_without_decoding"), bool):
        problems.append("the completion does not say whether sealing decoded "
                        "anything")
    return problems


# ---------------------------------------------------------------------------
# Where a step is, and whether it may be touched
# ---------------------------------------------------------------------------

STEP_UNSTARTED = "unstarted"
STEP_PARTIAL = "partial"
STEP_UNSEALED = "cells_complete_unsealed"
STEP_SEALED = "sealed"


def step_state(results, evidence_dir, plan: dict, index: int, *,
               settings: Settings = SETTINGS) -> str:
    """One of four states. The runner's whole decision comes from this."""
    cells = step_cells(plan, index, settings=settings)
    have = known_keys(results)
    done = sum(1 for c in cells if c in have)
    if evidence_dir is not None and completion_path(evidence_dir,
                                                    index).is_file():
        return STEP_SEALED
    if done == len(cells):
        return STEP_UNSEALED
    return STEP_PARTIAL if done else STEP_UNSTARTED


def step_problems(path, plan: dict, index: int, *, resume: bool,
                  evidence_dir=None,
                  settings: Settings = SETTINGS) -> list[str]:
    """May this step be touched now? The schedule is an order, not a suggestion.

    An earlier step counts as finished when it is *sealed*, not merely when
    its cells exist: a step whose completion was never written has no record
    saying which attempts produced it, and starting the next one on top of
    that buries the gap. With no ``evidence_dir`` the ordering falls back to
    cell completeness, which is all a caller without evidence can ask.
    """
    try:
        step(index)
    except ValueError as exc:
        return [str(exc)]

    have = known_keys(path)
    problems: list[str] = []

    for earlier in range(index):
        group, name = STEP_ORDER[earlier]
        cells = step_cells(plan, earlier, settings=settings)
        absent = [c for c in cells if c not in have]
        if absent:
            problems.append(
                f"step {earlier} ({group}/{name}) is {len(absent)} cells "
                f"short. The schedule is frozen: whichever arm runs second "
                f"runs warmer, so a step taken out of order is a comparison "
                f"nobody controlled.")
        elif evidence_dir is not None and not completion_path(
                evidence_dir, earlier).is_file():
            problems.append(
                f"step {earlier} ({group}/{name}) has all its cells and no "
                "completion record. Seal it before starting another step: "
                "until it is sealed nothing says which attempts measured it.")

    for later in range(index + 1, N_STEPS):
        group, name = STEP_ORDER[later]
        present = [c for c in step_cells(plan, later, settings=settings)
                   if c in have]
        if present:
            problems.append(
                f"step {later} ({group}/{name}) already holds {len(present)} "
                f"cells and comes after this one; the order was already "
                f"broken before this call")

    state = step_state(path, evidence_dir, plan, index, settings=settings)
    if state == STEP_SEALED:
        group, name = STEP_ORDER[index]
        problems.append(f"step {index} ({group}/{name}) is complete and "
                        "sealed; nothing here re-runs a measured cell")
    elif state == STEP_PARTIAL and not resume:
        mine = step_cells(plan, index, settings=settings)
        done = sum(1 for c in mine if c in have)
        problems.append(
            f"step {index} already holds {done} of its {len(mine)} cells. "
            "Pass --resume to open a new attempt and fill only what is "
            "missing; nothing here ever re-runs or replaces a cell.")
    return problems


# ---------------------------------------------------------------------------
# The whole evidence chain, read back
# ---------------------------------------------------------------------------

def _read_json(path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"{Path(path).name} is not readable ({exc})"


def evidence_manifest(evidence_dir, plan: dict, *, arms=ARM_ORDER) -> dict:
    """``file name -> sha256`` for every evidence file this run rests on."""
    out: dict[str, str | None] = {}
    for index in range(N_STEPS):
        _group, name = STEP_ORDER[index]
        if name not in set(arms):
            continue
        paths = [completion_path(evidence_dir, index)]
        paths += existing_attempts(evidence_dir, index)
        for path in paths:
            out[path.name] = sha256_file(path) if path.is_file() else None
    return dict(sorted(out.items()))


def evidence_manifest_digest(evidence_dir, plan: dict, *,
                             arms=ARM_ORDER) -> str:
    return digest_obj(evidence_manifest(evidence_dir, plan, arms=arms))


def step_chain_problems(evidence_dir, plan: dict, index: int, *,
                        expected_pack_digest, expected_dependency_digest,
                        results_path=None, rows=None,
                        require_completion: bool = True,
                        settings: Settings = SETTINGS) -> list[str]:
    """One step's whole chain: cells -> rows -> attempts -> completion.

    The single place any of this is decided. Three callers need the same
    answer and used to get three different partial ones: ``--verify`` walked
    the chain, the sealer checked almost none of it before writing the record
    that closes a step, and the runner checked only that a predecessor's
    completion *file existed*. A validator that three callers approximate
    separately is three validators.

    ``require_completion=False`` is the sealer's form: everything except the
    completion, which is the thing it is about to write. With it ``True`` the
    completion must be there and must agree with the attempts and the rows.

    Nothing here raises. Every way a file can disappoint -- absent,
    unreadable, valid JSON of the wrong shape, valid JSON missing the field
    the next line was going to read -- comes back as a sentence, because the
    sealer's alternative to a sentence is a traceback after the cells were
    measured and before the record that vouches for them.
    """
    try:
        group, name = step(index)
    except ValueError as exc:
        return [str(exc)]
    label = f"step {index} ({group}/{name})"

    if evidence_dir is None:
        return [f"{label}: no evidence directory was given, so nothing can be "
                "said about which machine produced these cells, under which "
                "pack, or whether the preflight passed"]
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.is_dir():
        return [f"{label}: {evidence_dir.name!r} is not a directory, so this "
                "step has no evidence"]

    problems: list[str] = []
    if rows is None:
        rows, read_issues = read_cells_lenient(results_path)
        problems += [f"{label}: {p}" for p in read_issues]

    # ---- placement: which rows belong to this step, and do they say so ---
    #
    # Two sources, unioned, because either alone leaves a gap. Selecting only
    # rows that *claim* this step misses a cell of this step filed under
    # another one -- it then shows up as a missing cell, which is true but
    # says nothing about the row that is actually wrong. Selecting only by
    # cell key misses a row that claims this step while carrying another
    # step's arm, because a cell key already contains the arm.
    cells = step_cells(plan, index, settings=settings)
    cell_set = set(cells)
    relevant = [row for row in rows
                if row.get("step_index") == index or cell_key(row) in cell_set]

    placed = []
    for row in relevant:
        where = (f"{row.get('case_id')!r}/{row.get('arm')}/seed "
                 f"{row.get('seed')}")
        wrong = []
        if row.get("step_index") != index:
            wrong.append(f"step_index {row.get('step_index')!r}")
        if row.get("group") != group:
            wrong.append(f"group {row.get('group')!r}")
        if row.get("arm") != name:
            wrong.append(f"arm {row.get('arm')!r}")
        if wrong:
            problems.append(
                f"{label}: {where} records {', '.join(wrong)}, and the frozen "
                f"schedule puts this step at {index}/{group!r}/{name!r}")
            continue
        if cell_key(row) not in cell_set:
            problems.append(f"{label}: {where} claims this step and is not "
                            "one of its cells")
            continue
        placed.append(row)

    # ---- the cells this step is defined to hold --------------------------
    mine = placed
    counts: dict[tuple, int] = {}
    for row in mine:
        key = cell_key(row)
        counts[key] = counts.get(key, 0) + 1

    absent = [c for c in cells if c not in counts]
    if absent:
        shown = ", ".join(f"{c[1]}/seed {c[3]}" for c in absent[:5])
        problems.append(f"{label}: {len(absent)} of {len(cells)} cells are "
                        f"missing (e.g. {shown})")
    for key in sorted((k for k, n in counts.items() if n > 1), key=str):
        problems.append(f"{label}: {key[1]!r}/seed {key[3]} is recorded "
                        f"{counts[key]} times")

    # ---- the rows themselves ---------------------------------------------
    cases = case_index(plan) if isinstance(plan.get("cases"), list) else {}
    for row in mine:
        problems += [f"{label}: {p}" for p in result_row_problems(
            row, cases, arms=(name,), settings=settings)]

    # ---- every attempt ----------------------------------------------------
    attempt_bodies: dict[str, dict] = {}
    for path in existing_attempts(evidence_dir, index):
        body, issue = _read_json(path)
        if issue:
            problems.append(f"{label}: {issue}")
            continue
        if not isinstance(body, dict):
            problems.append(f"{label}: {path.name} holds a "
                            f"{type(body).__name__}, not an object")
            continue
        problems += [f"{label} {path.name}: {p}" for p in attempt_problems(
            body, plan, expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest)]
        key = body.get("attempt_id")
        if not isinstance(key, str) or not key.strip():
            problems.append(f"{label}: {path.name} names no attempt_id, so no "
                            "cell can point at it and no completion can list "
                            "it")
            continue
        if key in attempt_bodies:
            problems.append(f"{label}: two attempt files claim {key!r}")
        attempt_bodies[key] = body
    if not attempt_bodies:
        problems.append(f"{label}: there are no attempt records, so nothing "
                        "says which process measured its cells")

    # ---- how many cells each attempt actually wrote ----------------------
    written: dict[str, int] = {}
    for row in mine:
        key = row.get("attempt_id")
        written[key] = written.get(key, 0) + 1
    for key in sorted(set(written) - set(attempt_bodies), key=str):
        problems.append(f"{label}: {written[key]} cells name attempt {key!r}, "
                        "which has no record here")
    for row in mine:
        body = attempt_bodies.get(row.get("attempt_id"))
        if body is None:
            continue
        if row.get("attempt_digest") != attempt_digest(body):
            problems.append(
                f"{label}: {row.get('case_id')!r}/seed {row.get('seed')} "
                "records an attempt digest that is not the digest of the "
                "attempt it names")
            break

    # ---- the completion ---------------------------------------------------
    completion_file = completion_path(evidence_dir, index)
    if not require_completion:
        if completion_file.is_file():
            problems.append(f"{label}: a completion record already exists; a "
                            "step is sealed once")
        return problems
    if not completion_file.is_file():
        problems.append(
            f"{label} was never sealed: its cells may exist but no completion "
            "record says which attempts measured them")
        return problems
    completion, issue = _read_json(completion_file)
    if issue:
        return problems + [f"{label}: {issue}"]
    if not isinstance(completion, dict):
        return problems + [f"{label}: the completion holds a "
                           f"{type(completion).__name__}, not an object"]
    problems += [f"{label}: {p}" for p in completion_problems(
        completion, plan, index=index, settings=settings)]

    listed: dict[str, dict] = {}
    entries = completion.get("attempts")
    for i, reference in enumerate(entries if isinstance(entries, list) else []):
        if not isinstance(reference, dict) \
                or not isinstance(reference.get("attempt_id"), str):
            problems.append(f"{label}: attempt entry {i} in the completion "
                            "names no attempt_id")
            continue
        listed[reference["attempt_id"]] = reference

    for key in sorted(set(listed) - set(attempt_bodies)):
        problems.append(f"{label}: the completion lists attempt {key!r} and "
                        "no such attempt record is here")
    for key in sorted(set(attempt_bodies) - set(listed)):
        problems.append(
            f"{label}: attempt {key!r} left a record and the completion does "
            "not list it; a step lists every attempt it took")

    # Field by field against the reference the attempt itself produces. The
    # digest alone was checked before, which left every other field of the
    # entry -- when it started, which pack, how many cells -- as free text
    # nobody compared against anything.
    for key in sorted(set(listed) & set(attempt_bodies)):
        expected = attempt_reference(attempt_bodies[key], written.get(key, 0))
        if listed[key] == expected:
            continue
        differing = sorted(
            field for field in ATTEMPT_REFERENCE_FIELDS
            if listed[key].get(field) != expected.get(field))
        problems.append(
            f"{label}: the completion's entry for {key!r} differs from the "
            f"attempt record in {differing or sorted(set(listed[key]) ^ set(expected))}; "
            "an entry is derived from its attempt and the cells that name it, "
            "not written beside them")

    for key in sorted(set(written) - set(listed), key=str):
        problems.append(f"{label}: {written[key]} cells name attempt {key!r}, "
                        "which this step's completion does not list")
    return problems


def predecessor_problems(evidence_dir, plan: dict, index: int, *,
                         expected_pack_digest, expected_dependency_digest,
                         results_path=None, rows=None,
                         settings: Settings = SETTINGS) -> list[str]:
    """Replay the whole chain for every step before this one.

    Checking that a predecessor's completion *file exists* is not checking
    the predecessor. The order is frozen so that step N runs on a machine
    whose earlier steps are accounted for; a predecessor whose attempt
    records contradict its completion, or whose preflight failed, is not
    accounted for, and step N would be measured on top of it.

    Run before the model is loaded and before the warm-up, because the point
    of noticing is to not spend either.
    """
    if index <= 0:
        return []                       # step 0 has nothing behind it
    problems: list[str] = []
    if rows is None and results_path is not None:
        rows, read_issues = read_cells_lenient(results_path)
        # Not discarded. A predecessor cannot be replayed against a results
        # file that cannot be read, and "the rows I could parse look fine" is
        # not the same answer as "the rows are fine".
        problems += [f"results: {p}" for p in read_issues]
    for earlier in range(index):
        problems += step_chain_problems(
            evidence_dir, plan, earlier,
            expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest,
            results_path=results_path, rows=rows, require_completion=True,
            settings=settings)
    return problems


def evidence_problems(evidence_dir, plan: dict, *, results_path,
                      expected_pack_digest, expected_dependency_digest,
                      arms=ARM_ORDER,
                      settings: Settings = SETTINGS) -> list[str]:
    """The chain, end to end, for every step of the arms asked about.

    Nothing here is optional. A results file with no evidence beside it says
    nothing about which machine produced it, under which pack, or whether the
    preflight passed -- and a score computed from it would be a number with no
    provenance at all.
    """
    if evidence_dir is None:
        return ["no evidence directory was given, so nothing can be said "
                "about which machine produced these results, under which "
                "pack, or whether the preflight passed"]
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.is_dir():
        return [f"{evidence_dir.name!r} is not a directory, so this run has "
                "no evidence"]

    rows, read_issues = read_cells_lenient(results_path)
    wanted = set(arms)
    problems: list[str] = [f"results: {p}" for p in read_issues]
    for index in range(N_STEPS):
        if STEP_ORDER[index][1] not in wanted:
            continue
        problems += step_chain_problems(
            evidence_dir, plan, index,
            expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest,
            rows=rows, require_completion=True, settings=settings)
    return problems
