"""Phase 3C: what the placement layer does to legality, and nothing else.

Phase 3A froze the rejection layer's *semantics* and Phase 3B implemented it.
Both were reviewed; neither was ever measured. PROJECT_STATUS has said since
the thirty-ninth round that Phase 3C may not begin until three things are
frozen -- the semantics (done, Phase 3A), the **data source** and the
**acceptance criteria** (both still open). This module is those two, written
before a single sample exists, plus the plan, the seal, the receipt and the
scorer that make the result checkable by somebody who does not trust it.

---------------------------------------------------------------------------
The question
---------------------------------------------------------------------------

With the model, the checkpoint, the adapter, the prompt, the inventory, the
cases, the seeds, the decode budget and the execution environment held
identical, how much does :class:`PlacementGate` change generation legality?

Three arms, one variable:

===  ==========================================================  ============
arm  what decodes                                                gate
===  ==========================================================  ============
A    ``final_H2``, inventory prompt, nothing enforced            none
B    the same, with stock made unspendable past its quantity     InventoryGate
C    the same, plus collision and connectivity                   InventoryPlacementGate
===  ==========================================================  ============

**The primary contrast is C - B.** A is the floor: it is what the model does
when it is told about the inventory and simply believed. B is Phase 2's arm E
on cases Phase 2 never saw. C is B with the layer under test switched on, and
the difference between them is the only thing this run is designed to
measure. A - B and C - A are reported because they are free once the cells
exist, not because the design isolates them.

---------------------------------------------------------------------------
What arm C is, exactly, and what that costs the answer
---------------------------------------------------------------------------

Arm C is ``InventoryPlacementGate(enabled=True, connectivity="final_eos")``.
That is **two mechanisms**, and this design cannot separate them:

* the collision mask, which makes an overlapping placement unreachable; and
* the connectivity deferral, which withholds EOS from the candidate list
  while the model is in more than one piece, at most
  :data:`~src.constraints.placement_decode.MAX_EOS_DEFERRALS` times.

A four-arm design with collision alone in a third arm would separate them.
This run has three arms because three is what was asked for, so **C - B is
the effect of the layer as a whole and no claim here decomposes it**. What is
reported instead is the layer's own counters -- candidates masked per slot,
EOS deferrals, and the ``connectivity_unmet`` and ``space_exhausted``
terminations -- which say how much of each mechanism actually fired. Those
are descriptions of what the layer did, not an attribution of the outcome to
one half of it.

---------------------------------------------------------------------------
What this run may not be read as saying
---------------------------------------------------------------------------

**1. No target number.** Nothing here predicts, promises or requires 60%,
85%, or any other figure. There is no threshold whose crossing counts as
success. The acceptance criterion frozen below is *that the comparison was
run as specified and every number in it can be re-derived from the stored
samples* -- not that any number reached a value.

**2. ``stud_only_connected`` and ``unsupported_brick_count`` are geometric.**
Connectivity is adjacent-layer 2-D footprint overlap with ``ground=False``.
It does not check centre of mass, moments, or whether anything stands up
under gravity. Real stability analysis needs a solver this project does not
have. **Neither may be called support, stability or physics**, here or in the
report, and a test scans this module and the report generator for those
words.

**3. Collision-freedom in arm C is a construction, not a finding.** The mask
makes a colliding placement unreachable, so ``collision_free`` is 1.0 in arm
C by the same argument that makes ``inventory_valid`` 1.0 in arms B and C.
Reporting it is bookkeeping -- it proves the layer was actually on -- and it
is not evidence that the layer improved anything. The metrics that can move
in either direction are the ones downstream of it: ``stud_only_connected``,
``touches_ground``, ``termination_accepted``, Core Success, and the latency.

**4. Constraining one axis moves the others, and Phase 2 measured that
happening.** ``InventoryGate`` *lowered* the marginal ``in_bounds`` and
``collision_free`` rates (D against B, E against C). So a fall in some rate
in arm C is an expected shape of result, not a defect in the layer or in this
run, and the report says so before it says anything else.

**5. These are 160 cases from one test split.** The intervals below describe
variation across *these* cases under a frozen resampling rule. They are not a
claim about LEGO captions in general, and no hypothesis test was
pre-registered.

---------------------------------------------------------------------------
Where the cases come from, and why they are clean
---------------------------------------------------------------------------

The same test split Phase 2 drew from, ``instruct_inv_test.jsonl``, pinned by
SHA-256. Phase 2 used 20 of its 200 pairs and PROJECT_STATUS records that
those 160 cases **may never be an independent test set again**. So Phase 3C
takes 20 pairs from what is left, under three exclusions applied in order:

1. **the pairs themselves** -- Phase 2's 20 pair ids;
2. **the group** -- every pair whose ``object_id`` appears in any Phase 2
   pair. Pairs are 8 rows of one object, and two pairs can share an object:
   Phase 2's 20 pairs cover only 19 objects. Excluding at pair level alone
   would leave a pair of an object Phase 2 already generated against, which
   is the group-level leak this exclusion exists to close;
3. **the caption** -- every pair carrying a caption whose SHA-256 matches any
   Phase 2 caption, in case the same text reaches two objects.

Among what survives, no two selected pairs may share an ``object_id`` either,
so the 20 cases-groups are 20 distinct objects.

**What is disclosed rather than excluded.** ``scripts/12_f_oracle.py`` ran
CP-SAT over the whole of ``counterfactual_test.jsonl`` -- the same objects,
in their reference geometry. It ran no model, made no selection and tuned
nothing (``model_used: false``, ``retrieval_used: false`` in its own report);
it measured whether an exact cover exists. Excluding it would leave no test
split at all. It is recorded in the audit as a data-level touch, and it is
the reason the audit reports *two* independence claims rather than one.

**val is not opened.** Not by the audit, not by the selection, not by the
scorer. Checkpoint selection used val; this run reads the split membership
file to prove the objects it picked are test objects and never opens
``instruct_inv_val.jsonl``. The audit records ``val_opened: false`` and a
test asserts no module reachable from here names that file.

---------------------------------------------------------------------------
The acceptance criteria
---------------------------------------------------------------------------

Frozen before generation, and none of them is a number the run has to reach:

1. all 1,920 cells present, one per (case, arm, seed), none missing, none
   duplicated, none decoded twice with one kept;
2. the three arms ran the same 160 cases, the same 4 seeds, the same prompt
   digests, the same inventories and the same decode budget -- checked cell
   by cell against the plan rather than asserted;
3. every sample file published write-once and covered by a seal the Mac
   verified against a digest carried by a second route;
4. every reported number re-derived on the Mac from the stored per-case
   samples, by the scorer this contract pins, and equal to what the node's
   run implies;
5. the environment the node reports equals the environment this contract
   pinned, field by field;
6. an execution authorization, written after the pack was built and before
   anything reached the node, binds the pack digest, the dependency digest,
   the adapter, both gate sources and every module the decode path can
   reach -- and the file table the pack digest was taken over is archived
   with it, so the value can be recomputed later with no pack present;
7. **every stored row names the execution manifest that is actually in the
   directory.** The field existed from the start and nothing compared it, so
   a bundle whose rows pointed at another run -- or at nothing -- passed the
   receipt and was scored.

A run that satisfies those is a run whose numbers may be quoted. What the
numbers *are* is the finding, whatever it turns out to be.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from src.data.bricks import PART_VOCAB
from src.eval import acceptance
from src.eval.acceptance import (FINAL_ADAPTER_SHA256, FINAL_MODEL,
                                 LOADER_FINAL, PlanRefused, canonical_inventory,
                                 canonical_json, digest_obj,
                                 plan_leak_problems, quantile, quantiles,
                                 sha256_text)
from src.generation.prompt import build_prompt
# Imported at module level, deliberately. ``pack`` does not import this
# module -- that is what keeps the Phase 3C generation out of ``pack.py`` and
# out of the V1 visual run's closure -- so the dependency runs one way only.
from src.training import pack as pack_module
from src.training.session import sha256_file

ROOT = Path(__file__).resolve().parents[2]

KIND = "brickagain.phase3c"
CONTRACT_VERSION = 1
PLAN_KIND = "brickagain.phase3c_plan"
PLAN_SCHEMA_VERSION = 1
MEMBERSHIP_KIND = "brickagain.phase3c_case_membership"
AUDIT_KIND = "brickagain.phase3c_isolation_audit"
MANIFEST_KIND = "brickagain.phase3c_execution_manifest"
SAMPLES_KIND = "brickagain.phase3c_samples"
SEAL_KIND = "brickagain.phase3c_seal"
RECEIPT_KIND = "brickagain.phase3c_receipt"
SCORES_KIND = "brickagain.phase3c_scores"


# ---------------------------------------------------------------------------
# The three arms
# ---------------------------------------------------------------------------

#: One prompt form for all three arms: caption plus the inventory block, built
#: by the project's one builder. Arm A is *told* about the stock and simply
#: believed; it is not a different prompt. Changing the prompt between arms
#: would put a second variable beside the gate.
PROMPT_FORM = "inventory"

GATE_NONE = "none"
GATE_INVENTORY = "src.constraints.inventory_decode.InventoryGate"
GATE_INVENTORY_PLACEMENT = \
    "src.constraints.placement_decode.InventoryPlacementGate"

#: Arm C's connectivity mode. ``off`` would make arm C the collision mask
#: alone; ``final_eos`` is the whole layer, which is what "does PlacementGate
#: help" asks about. The cost of the choice is stated in the docstring: C
#: bundles two mechanisms and this design cannot separate them.
CONNECTIVITY_MODE = "final_eos"


@dataclass(frozen=True)
class Arm:
    """One arm: which weights, which prompt form, which gate. Nothing else."""

    name: str
    model: str
    loader: str
    prompt_form: str
    gate: str
    #: ``None`` where there is no placement layer, so a reader can see that
    #: the field is absent rather than set to something inert.
    connectivity: str | None
    note: str

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


ARM_ORDER: tuple[str, ...] = ("A", "B", "C")

ARMS: dict[str, Arm] = {
    "A": Arm("A", FINAL_MODEL, LOADER_FINAL, PROMPT_FORM, GATE_NONE, None,
             "the project model told about the inventory and trusted to "
             "respect it; nothing is enforced"),
    "B": Arm("B", FINAL_MODEL, LOADER_FINAL, PROMPT_FORM, GATE_INVENTORY,
             None,
             "the same, with stock unspendable past its quantity. This is "
             "Phase 2's arm E on cases Phase 2 never saw"),
    "C": Arm("C", FINAL_MODEL, LOADER_FINAL, PROMPT_FORM,
             GATE_INVENTORY_PLACEMENT, CONNECTIVITY_MODE,
             "the same again, with collision masked and EOS withheld while "
             "the model is in more than one piece. C - B is what this run "
             "exists to measure"),
}

#: ``value(a) - value(b)``, written down so no report can reverse a sign. The
#: first entry is the one the run is for.
CONTRASTS: tuple[tuple[str, str], ...] = (("C", "B"), ("B", "A"), ("C", "A"))
PRIMARY_CONTRAST: tuple[str, str] = CONTRASTS[0]


def arm(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"{name!r} is not one of {list(ARM_ORDER)}")
    return ARMS[name]


def contrast_name(a: str, b: str) -> str:
    return f"{a}-{b}"


# ---------------------------------------------------------------------------
# The settings all three arms share
# ---------------------------------------------------------------------------

#: Phase 2's settings object, reused rather than restated: same seeds, same
#: temperature, same brick and token budgets, same device, same dtype, same
#: offline policy, same base weights and tokenizer revisions. Reusing it is
#: what makes "the decode budget is the same" checkable instead of asserted,
#: and it keeps arm B numerically comparable to Phase 2's arm E.
SETTINGS = acceptance.SETTINGS


def settings_digest() -> str:
    return acceptance.settings_digest()


def settings_for(name: str) -> dict:
    """The settings for one arm. Identical for all three, by construction."""
    arm(name)
    return SETTINGS.as_dict()


def final_model_document() -> dict:
    """The one model every arm loads, and the digests that identify it."""
    return acceptance.final_model_document()


# ---------------------------------------------------------------------------
# The execution environment, pinned before anything runs
# ---------------------------------------------------------------------------

#: What the node must be. Not a description of what it happened to be: the
#: runner compares its own probe against this field by field and refuses
#: before loading anything if they differ, and the execution manifest records
#: both sides. Taken from the environment Phase 2 measured on, so arm B and
#: Phase 2's arm E differ by the cases and nothing else.
EXECUTION_ENVIRONMENT: dict = {
    "os_system": "Linux",
    "wsl2": True,
    "device": "cuda",
    "dtype": "bfloat16",
    "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
    "torch_cuda_build": "13.0",
    "python": "3.12.3",
    "torch": "2.13.0+cu130",
    "transformers": "5.15.0",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "offline_env": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    },
    "local_files_only": True,
    "note": ("The Mac decodes nothing. MPS and CUDA do not agree bit for bit "
             "and a three-arm contrast split across them would measure the "
             "device as well as the gate."),
}

#: The fields of :data:`EXECUTION_ENVIRONMENT` a node probe must reproduce
#: exactly. ``vram`` and ``system_ram`` are deliberately absent: they vary by
#: hundredths between boots and the preflight already bounds them.
PINNED_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "os_system", "wsl2", "device", "dtype", "gpu_name", "torch_cuda_build",
    "python", "torch", "transformers", "peft", "accelerate", "offline_env",
)


# ---------------------------------------------------------------------------
# Termination, and the one place this contract corrects the Phase 2 scorer
# ---------------------------------------------------------------------------

#: A draw may be a core success only if it ended one of these two ways. Same
#: pair Phase 2 froze. ``max_bricks`` and ``max_tokens`` are budgets running
#: out mid-thought; ``space_exhausted`` and ``connectivity_unmet`` are the
#: placement layer giving up, which is a real outcome and not the model
#: saying it had finished.
ACCEPTED_TERMINATIONS: tuple[str, ...] = ("normal_eos", "inventory_exhausted")

#: Every termination arm C can produce. The two new ones come from
#: :data:`src.constraints.placement_decode.PLACEMENT_STOP_REASONS`.
TERMINATIONS: tuple[str, ...] = (
    "normal_eos", "inventory_exhausted", "max_bricks", "max_tokens",
    "space_exhausted", "connectivity_unmet",
)

#: The terminations that spend a token on EOS, which is what decides whether
#: ``n_tokens`` should be ``10n + 1`` or ``10n``.
#:
#: **This is the one place Phase 3C does not simply reuse the Phase 2
#: scorer.** ``src.eval.scoring.expected_complete_bricks`` decides that
#: question by asking whether the termination is *accepted*, which was correct
#: while the only two accepted reasons were also the only two EOS reasons.
#: Arm C breaks that coincidence: ``space_exhausted`` and
#: ``connectivity_unmet`` both leave EOS as the only candidate, the model
#: samples it, and the draw ends on ``10n + 1`` tokens while being an
#: unaccepted termination. Scored by the Phase 2 rule those draws would fail
#: ``parse_success`` for arithmetic reasons and arm C would be charged with
#: parse failures it did not have.
#:
#: ``src/eval/scoring.py`` is **not edited** to fix this. Its bytes are inside
#: the ``scorer_source_manifest_digest`` that Phase 2's plan pins, and moving
#: them would make Phase 2's own ``--score`` refuse to run -- retiring a
#: finished result's replayability to make a new one convenient. Phase 3C
#: carries the correction instead, in :func:`correct_token_accounting`, and a
#: test asserts the correction is a no-op on all four Phase 2 terminations.
EOS_TERMINATIONS: tuple[str, ...] = (
    "normal_eos", "inventory_exhausted", "space_exhausted",
    "connectivity_unmet",
)

TOKENS_PER_BRICK = 10
EOS_TOKENS = 1


def expected_complete_bricks(n_tokens: int, termination: str
                             ) -> tuple[int | None, str | None]:
    """How many whole bricks that many tokens can be, or why they cannot.

    :data:`EOS_TERMINATIONS` decides the arithmetic, not
    :data:`ACCEPTED_TERMINATIONS`. Everything else is the Phase 2 rule.
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) \
            or n_tokens < 1:
        return None, f"n_tokens is {n_tokens!r}"
    if termination in EOS_TERMINATIONS:
        body = n_tokens - EOS_TOKENS
        if body % TOKENS_PER_BRICK:
            return None, (f"{n_tokens} tokens ending on {termination} is not "
                          f"{TOKENS_PER_BRICK}n + 1; the run stopped inside a "
                          "brick")
        return body // TOKENS_PER_BRICK, None
    if n_tokens % TOKENS_PER_BRICK:
        return None, (f"{n_tokens} tokens ending on {termination} is not a "
                      f"multiple of {TOKENS_PER_BRICK}; the run stopped "
                      "inside a brick")
    return n_tokens // TOKENS_PER_BRICK, None


def correct_token_accounting(scored: dict, *, n_tokens: int,
                             termination: str) -> dict:
    """Re-decide the token arithmetic of one scored draw, and what follows.

    Called on the output of :func:`src.eval.scoring.score_generation`. Only
    four values can move -- the expected brick count, the note that explains
    it, ``tokens_match_complete_bricks`` and ``parse_success`` -- and
    ``deterministic_core_success`` is then re-derived from the corrected
    check table rather than left as the scorer computed it.

    Everything else the scorer said stands untouched: the parse itself, the
    parts, the inventory arithmetic, the geometry, the connectivity, the
    LDraw pass and the termination verdict do not depend on this.
    """
    out = json.loads(json.dumps(scored))          # never mutate the caller's
    expected, note = expected_complete_bricks(n_tokens, termination)
    parse = out["parse"]
    consistent = expected is not None and expected == parse["n_bricks"]
    parse_success = (parse["n_bricks"] > 0
                     and parse["n_unparsed_lines"] == 0
                     and consistent)
    parse["expected_complete_bricks"] = expected
    parse["token_brick_note"] = note
    parse["tokens_match_complete_bricks"] = consistent
    parse["parse_success"] = parse_success
    out["checks"]["parse_success"] = parse_success
    core = all(bool(out["checks"][name])
               for name in acceptance.CORE_SUCCESS_CHECKS)
    out["checks"]["deterministic_core_success"] = core
    out["deterministic_core_success"] = core
    return out


# ---------------------------------------------------------------------------
# What this run reports, with its numerator, its denominator and its sentence
# ---------------------------------------------------------------------------

#: The per-draw booleans, in the order the scorer evaluates them. Reused from
#: Phase 2 rather than restated: an arm-B number that is not comparable to
#: Phase 2's arm E because a check was quietly redefined is worse than no
#: comparison at all.
DRAW_CHECKS: tuple[str, ...] = acceptance.CORE_SUCCESS_CHECKS
CORE_SUCCESS_CHECKS: tuple[str, ...] = acceptance.CORE_SUCCESS_CHECKS

#: Two-sided 95%. Written as a constant because "95%" appears in the report
#: and a report whose stated level and computed level can disagree is a report
#: with an unfixable footnote.
CI_LEVEL = 0.95
WILSON_Z = 1.959963984540054

#: The paired bootstrap, frozen whole. Cases are the resampling unit because
#: cases are what is independent here: the four seeds of one case share a
#: caption and an inventory, and resampling draws would treat them as four
#: independent observations of the same thing.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_METHOD = (
    "paired case bootstrap: resample the N cases with replacement "
    f"{BOOTSTRAP_RESAMPLES} times using random.Random({BOOTSTRAP_SEED}) and "
    "randrange(N) per draw, recompute both arms' statistics on the resampled "
    "cases, take the difference, and report the "
    f"{(1 - CI_LEVEL) / 2:.3f} and {1 - (1 - CI_LEVEL) / 2:.3f} percentiles "
    "of those differences by the frozen quantile function. Percentile "
    "method; no bias correction and no acceleration.")

WILSON_METHOD = (
    "Wilson score interval at z = 1.959963984540054: "
    "(p + z^2/2n +- z*sqrt((p(1-p) + z^2/4n)/n)) / (1 + z^2/n). Chosen over "
    "the normal approximation because several of these rates are at or near "
    "0 and 1, where the normal interval leaves the unit interval.")

METRIC_SPEC: dict = {
    "draw_checks": {
        "scope": "draw",
        "type": "boolean table",
        "definition": ("the ten Phase 2 core-success conjuncts, computed by "
                       "src.eval.scoring.score_generation with this "
                       "contract's token accounting applied on top "
                       f"(see EOS_TERMINATIONS): {list(DRAW_CHECKS)}. The "
                       "definitions are Phase 2's, unchanged, so arm B is "
                       "comparable to Phase 2's arm E."),
    },
    "parseable": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("the report's name for parse_success: at least one "
                       "brick parsed, no unparsed line, and the token count "
                       "agrees with the number of whole bricks under "
                       "EOS_TERMINATIONS."),
    },
    "inventory_valid": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("every part used is stocked and nothing is drawn "
                       "beyond its quantity. 1.0 by construction in arms B "
                       "and C; reported so the gate can be seen to have been "
                       "on."),
    },
    "collision_free": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("find_collisions() returns no pair sharing a cell. "
                       "1.0 by construction in arm C: the mask makes a "
                       "colliding placement unreachable. That is bookkeeping, "
                       "not a finding."),
    },
    "stud_only_connected": {
        "scope": "draw",
        "type": "boolean",
        "definition": ("is_connected(bricks, ground=False): one component "
                       "under adjacent-layer footprint overlap. GEOMETRIC. "
                       "Not support, not stability, not physics."),
    },
    "core_success_at_1": {
        "scope": "case",
        "type": "rate",
        "numerator": ("cases whose draw at the first frozen seed "
                      f"({SETTINGS.seeds[0]}) is a deterministic core "
                      "success"),
        "denominator": "cases with that seed present",
        "definition": ("one declared attempt, not the best of four and not "
                       "an average over seeds. The seed is fixed by the "
                       "contract so it cannot be chosen after the numbers "
                       "exist."),
    },
    "core_success_at_4": {
        "scope": "case",
        "type": "rate",
        "numerator": ("cases with at least one deterministic core success "
                      "among their K seeds"),
        "denominator": ("cases with all K seeds present. A case missing a "
                        "seed is reported as incomplete and is in neither "
                        "part of the fraction."),
        "definition": f"Core Success@K with K = {SETTINGS.k}.",
    },
    "generation_failure_rate": {
        "scope": "arm",
        "type": "rate",
        "numerator": ("draws that produced nothing usable: no brick parsed, "
                      "or an unparsed line, or a token count that is not a "
                      "whole number of bricks"),
        "denominator": "draws recorded for that arm",
        "definition": ("the complement of parseable, named separately "
                       "because it is the thing an operator wants counted. "
                       "A cell the node never wrote is not in this rate: it "
                       "is a missing cell, reported on its own, and the run "
                       "does not close with any."),
    },
    "termination_reasons": {
        "scope": "arm",
        "type": "histogram",
        "definition": (f"counts over {list(TERMINATIONS)}. The last two "
                       "reach only arm C."),
    },
    "gate_masked_candidates": {
        "scope": "draw",
        "type": "count",
        "definition": ("candidates the placement layer removed from a "
                       "candidate list, summed over slots 0, 2, 4, 6 and 8, "
                       "and also reported per slot. **This layer masks; it "
                       "does not reject.** An illegal placement is never "
                       "sampled and then thrown away, so there is no "
                       "rejection count to report and the four rejection "
                       "counters stay null with implemented: false. Reading "
                       "this number as rejections would claim an observation "
                       "the layer never makes."),
    },
    "eos_deferrals": {
        "scope": "draw",
        "type": "count",
        "definition": ("times EOS was withheld from a candidate list while "
                       "the model was in more than one piece. NOT a count of "
                       "times the model asked to stop: the layer decides "
                       "what may be sampled and never learns what would have "
                       "been sampled from a list it did not offer."),
    },
    "seconds": {
        "scope": "draw",
        "type": "duration",
        "excludes": "model load, tokenizer resolution and the warm-up",
        "definition": ("wall-clock seconds inside the decode loop for one "
                       "draw, from time.perf_counter(), stored unrounded."),
    },
    "paired_case_result": {
        "scope": "case",
        "type": "record",
        "definition": ("for every case, every arm's per-seed check table, "
                       "its Core Success@1 and @4, and the C - B verdict for "
                       "that case. This is what makes the run auditable one "
                       "case at a time rather than only in aggregate."),
    },
    "contrast_delta": {
        "scope": "contrast",
        "type": "difference",
        "definition": ("value(a) - value(b), an absolute difference, with "
                       "both raw values and both denominators beside it. No "
                       "ratio and no percentage change."),
    },
    "wilson_interval": {
        "scope": "arm",
        "type": "interval",
        "definition": WILSON_METHOD,
    },
    "paired_bootstrap_interval": {
        "scope": "contrast",
        "type": "interval",
        "definition": BOOTSTRAP_METHOD,
    },
    "discordant_cases": {
        "scope": "contrast",
        "type": "counts",
        "definition": ("for a case-level boolean, the two off-diagonal "
                       "counts: cases where a succeeded and b did not, and "
                       "the reverse. Reported as raw counts. **No McNemar "
                       "p-value is computed and no significance is claimed** "
                       "-- the counts are what a reader needs to see how much "
                       "of the difference is real disagreement rather than "
                       "two rates that happen to differ."),
    },
}

METRIC_SPEC_DIGEST = digest_obj(METRIC_SPEC)

#: Words this run may not use about connectivity or the unsupported count.
#: A test scans this module, the script and the generated report for them.
FORBIDDEN_METRIC_TERMS: tuple[str, ...] = (
    "physically stable", "physical stability", "structurally sound",
    "load bearing", "load-bearing", "stability verified", "support verified",
)

STRATA: tuple[str, ...] = ("overall", "role", "variant")
ROLES: tuple[str, ...] = acceptance.ROLES
VARIANTS: tuple[str, ...] = acceptance.VARIANTS


# ---------------------------------------------------------------------------
# The scorer's own source, digested
# ---------------------------------------------------------------------------

#: Every module a reported number passes through on the Mac. Phase 2's five,
#: plus this one -- the token-accounting correction lives here, so a score
#: record that did not name this file would not say what produced it.
SCORER_SOURCES: tuple[str, ...] = tuple(sorted(
    set(acceptance.SCORER_SOURCES) | {"src/eval/phase3c.py"}))


def scorer_manifest(root=None) -> dict:
    base = Path(root or ROOT)
    return {name: sha256_file(base / name) for name in SCORER_SOURCES}


def scorer_manifest_digest(root=None) -> str:
    return digest_obj(scorer_manifest(root))


def scorer_manifest_problems(recorded, root=None) -> list[str]:
    """Whether the code running now is the code the plan was approved with."""
    if not isinstance(recorded, dict):
        return [f"the scorer manifest is a {type(recorded).__name__}, "
                "not an object"]
    current = scorer_manifest(root)
    problems = []
    for name in sorted(set(current) | set(recorded)):
        want, got = recorded.get(name), current.get(name)
        if want is None:
            problems.append(f"{name} is scored by code the plan does not name")
        elif got is None:
            problems.append(f"{name} is named by the plan and is not here")
        elif want != got:
            problems.append(f"{name} hashes to {got[:16]}..., and the plan "
                            f"was approved against {want[:16]}...")
    return problems


# ---------------------------------------------------------------------------
# Case selection: the source, the exclusions, the draw
# ---------------------------------------------------------------------------

TEST_FILE = acceptance.TEST_FILE
EXPECTED_TEST_SHA256 = acceptance.EXPECTED_TEST_SHA256

#: Phase 2's materialised plan. Read to learn what it used, verified against
#: both its own digest and a SHA-256 pinned here, so "the cases Phase 2 used"
#: cannot be narrowed by editing the file that says so.
PHASE2_PLAN = "gpu_plans/core_eval_plan.json"
#: The value ``PROJECT_STATUS`` published for that file when Phase 2 sealed
#: it, which is the second route: the digest is not taken from the file it
#: describes. An earlier draft of this module carried
#: ``9c1a2e8cbcb00b4c...`` here, which matches neither the file on disk nor
#: any value in the record, and which made :func:`read_phase2_plan` refuse
#: every plan including the real one. It was never executed against the file
#: it pins, so nothing downstream ever depended on it.
EXPECTED_PHASE2_PLAN_SHA256 = (
    "e0303a5b0f5815a25090773bc84eb0d91122ee20c6a8ba064481b9626ad226c3")
EXPECTED_PHASE2_PLAN_DIGEST = (
    "a761fe77cce43ae68820e291a30edf9bec0da59ce594834200a8ee4a71225272")

#: The split membership file, read to prove every selected object is a test
#: object. It carries object ids and their split and no captions, so reading
#: it is not reading a split.
SPLIT_FILE = "data/splits/object_splits.json"

#: The group key. A pair is 8 rows of one object; two pairs can be two
#: variants of the same object, which is why the exclusion is by object and
#: not by pair.
GROUP_KEY = "object_id"

SELECTION_SEED = 20260829
N_PAIRS = 20
ROWS_PER_PAIR = 8
N_CASES = N_PAIRS * ROWS_PER_PAIR                 # 160

EXCLUSIONS: tuple[str, ...] = (
    "every pair id used by the Phase 2 core evaluation",
    "every pair whose object_id appears in any Phase 2 pair",
    "every pair carrying a caption whose SHA-256 matches a Phase 2 caption",
)

SELECTION_RULE = (
    "sort the eligible pair ids; shuffle that list once with "
    f"random.Random({SELECTION_SEED}).shuffle; walk it in order taking a "
    "pair whenever its object_id has not already been taken, and stop at "
    f"{N_PAIRS} pairs. Every row of a taken pair is a case, ordered by pair "
    "then by sample_id. The seed is today's date, declared here before the "
    "draw was run, and the first draw it produced is the one used.")

#: What a case may carry. Phase 2's list, unchanged: the plan travels to a
#: machine that must not see an answer.
CASE_FIELDS: tuple[str, ...] = acceptance.CASE_FIELDS
FORBIDDEN_CASE_FIELDS: tuple[str, ...] = acceptance.FORBIDDEN_CASE_FIELDS

# ---------------------------------------------------------------------------
# Where a generation of the frozen plan set lives
# ---------------------------------------------------------------------------
#
# **Generations, not versions in place.** The contract changed after the
# first plan set was written -- the execution authorization did not exist,
# and this module's own bytes are inside the scorer manifest the plan pins --
# so that set can no longer authorise a run. It is *superseded*, not
# replaced: its files stay exactly where they are, unmodified, and the
# reason is recorded in the new generation's own directory.
#
# The archive is under ``data/phase3c/`` rather than ``gpu_plans/`` for one
# reason: ``gpu_plans/`` is in ``.gitignore``, so a set living only there is
# a set that exists on one disk. These files are the record of what a result
# was produced against and they have to survive the machine.
#
# They are private-only. The plan carries the test split's captions and
# inventories; ``17_public_snapshot.py`` denies the whole tree and
# ``tests/test_public_snapshot.py`` proves both directions -- that no case
# content reaches the published tree, and that the denial is what keeps it
# out rather than an accident of the allowlist.

GENERATION = "gen09"
ARCHIVE_ROOT = "data/phase3c/frozen"
ARCHIVE_DIR = f"{ARCHIVE_ROOT}/{GENERATION}"

#: Where the node's plan is staged. Not the archive: a pack may carry nothing
#: under ``data/`` -- ``pack.build`` refuses it as a last check over its own
#: deny table, and that check is worth more than having one location. So the
#: archive is the record and this is a ``copy_once`` copy of it, held to the
#: same bytes by :func:`staged_copy_problems`. The name carries the
#: generation, so a later one cannot land on it.
#:
#: **The authorization is deliberately not staged.** It names the digest of
#: the pack, so a copy of it inside that pack would put the value and the
#: thing it authenticates in one parcel -- the failure GPU_NODE.md already
#: refuses for ``pack_digest``. It travels by hand from the archive, and the
#: node is pointed at it with ``--authorization``.
NODE_PLAN_PATH = f"gpu_plans/phase3c_{GENERATION}_plan.json"

PLAN_PATH = f"{ARCHIVE_DIR}/plan.json"
MEMBERSHIP_PATH = f"{ARCHIVE_DIR}/case_membership.json"
AUDIT_PATH = f"{ARCHIVE_DIR}/isolation_audit.json"
AUTHORIZATION_PATH = f"{ARCHIVE_DIR}/execution_authorization.json"
CONTRACT_SNAPSHOT_PATH = f"{ARCHIVE_DIR}/contract_at_freeze.json"

#: The pack manifest, copied write-once into the archive at authorisation.
#:
#: The gap it closes. gen02's grant named ``pack_digest 2cdbbfc8...`` and
#: nothing on this machine could produce those bytes again: the pack was
#: built into a scratch directory, the digest was read off the build, and
#: then editing continued -- so the value was a number in a file with no
#: path back to what it described. A digest nobody can re-derive is a digest
#: nobody can check, which is the same failure as not having one.
#:
#: This file is the file table the digest was taken over: every included
#: path with its own SHA-256 and byte count, plus ``files_digest`` and
#: ``data_digest``. From it ``pack.pack_digest`` recomputes the authorised
#: value with no pack present, and every row can be checked against the tree
#: at the commit that froze the generation.
PACK_EVIDENCE_PATH = f"{ARCHIVE_DIR}/pack_manifest.json"

#: The versioned supersession record, one per generation, write-once. It says
#: which generations may not be executed and why, without touching a byte of
#: the archives it is about.
SUPERSESSION_PATH = f"{ARCHIVE_ROOT}/supersession_{GENERATION}.json"

#: Where an attempt that reached a node and produced nothing is kept.
#:
#: A failed attempt is the one artefact with no digest chain of its own:
#: there are no cells to seal, so there is no seal, so there is no receipt
#: and nothing re-derives anything. What it has instead is an index that
#: names every file it holds with that file's size and SHA-256, and a
#: read-only listing of the node directory taken afterwards. Neither is
#: trustworthy alone -- an index can be edited to agree with itself, and a
#: listing is just text -- so :func:`failed_attempt_problems` checks them
#: against each other, against the bytes on disk, and against the frozen
#: generation the attempt ran under.
FAILED_ATTEMPT_ROOT = "data/phase3c/failed_attempts"
FAILED_ATTEMPT_KIND = "brickagain.phase3c_failed_attempt"

#: Schema 2 records an attempt that produced **no cell at all** -- gen04.
#: Schema 3 records one that produced some and then stopped -- gen08, whose
#: step 0 completed 320 cells before step 1 refused. They are separate
#: versions rather than one relaxed version because schema 2's central claim
#: is "every member is absent", and a schema able to express "some members
#: are populated" cannot also enforce that. gen04's bytes stay at 2 and are
#: still held to every check they were written under.
FAILED_ATTEMPT_SCHEMA_VERSION = 3
FAILED_ATTEMPT_SCHEMA_VERSIONS: tuple[int, ...] = (2, 3)

#: The block each schema states its outcome in. A record carrying both would
#: satisfy whichever is checked, so carrying both is itself a problem.
ZERO_CELL_PROOF = "proof_no_cell_was_written"
PARTIAL_PROOF = "proof_of_what_was_written"

#: The states a sample member can be found in, and the distinction the whole
#: zero-cell claim rests on. ``absent`` is a stronger statement than
#: ``zero_byte``: a zero-byte file is something a process created and left
#: empty, and creating one to demonstrate a zero-cell result would destroy
#: the evidence it claimed to provide.
MEMBER_STATES: tuple[str, ...] = ("absent", "zero_byte", "populated")

#: Every generation that may not be executed, kept for the record. Each entry
#: names where its files still are; none of them is edited or deleted.
SUPERSEDED: tuple[dict, ...] = (
    {
        "generation": "gen08",
        "written": "2026-08-31",
        "plan": "data/phase3c/frozen/gen08/plan.json",
        "membership": "data/phase3c/frozen/gen08/case_membership.json",
        "audit": "data/phase3c/frozen/gen08/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen08/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen08/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "88bb26ee8c8dbc62941de5002f36fe1443eba4c080fd40cef9d617a01bd8b645",  # noqa: E501
        "pack_digest_it_names": "35b2692cb8d07801ab9c1529965f1918f9b712a9f157d4ec3d3a194563d6b4c7",  # noqa: E501
        "pack_evidence_digest": "ac1bba09c86922d34f2e7ea33ad5c2743ace09a46d882362b4cb55b30796939e",  # noqa: E501
        "superseded_because": (
            "it reached the node and stopped part way through. Every carried "
            "digest matched, the pack verified with no problems, the "
            "environment matched the contract field by field, and step 0 "
            "(even/A) completed its 320 cells and exited 0. Step 1 (even/B) "
            "then failed on its FIRST measured cell with AttributeError: "
            "'InventoryGate' object has no attribute 'counters'. gate_ledger "
            "called gate.counters() on whatever the decode returned; arm A "
            "names no gate and returned before the call, arm C's "
            "InventoryPlacementGate defines the method, and arm B's "
            "InventoryGate does not. The suite missed it because its "
            "stand-in gate implemented counters(), making the substitute a "
            "superset of the real object -- the same shape as gen04. gen09 "
            "checks the actual gate object's exact class, fills arm B's "
            "ledger from that object without inventing placement counters, "
            "and refuses arm C if its counters are missing or in the wrong "
            "connectivity mode. Steps 2-5 never started. The fix edits "
            "src/eval/phase3c.py, which this plan's scorer manifest pins, so "
            "the freeze has to be redone."),
        "cells_produced": 320,
        "sealed": False,
        "reached_a_node": True,
        "attempted_execution":
            "data/phase3c/failed_attempts/"
            "gen08_gate_ledger_defect_20260831T044425Z",
        "attempted_execution_index_sha256": "f98a47d1b3e822054e4bac11aa95fcde05f6a76e5b8c8a7b0efbd198034608c5",  # noqa: E501
        "archive_retained_unmodified": True,
        "may_be_used_for": (
            "the record only. Its 320 cells are one arm of one group: they "
            "answer no contrast this contract asks about, they were produced "
            "under a source closure the fix changes, and they may not be "
            "carried into gen09, sealed, scored or cited."),
        "not_deleted_because": (
            "it is the record of the first Phase 3C execution to produce any "
            "cell at all, and of the interface defect that stopped it."),
    },
    {
        "generation": "gen07",
        "written": "2026-08-30",
        "plan": "data/phase3c/frozen/gen07/plan.json",
        "membership": "data/phase3c/frozen/gen07/case_membership.json",
        "audit": "data/phase3c/frozen/gen07/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen07/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen07/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "08defec87812b01f185207852d2868948a8cbe87ea6633a18e0d21535b966401",  # noqa: E501
        "pack_digest_it_names": "6a191cfce9919fadce8fc112344e282f142de3c891067c74988224181ab8e49b",  # noqa: E501
        "pack_evidence_digest": "d6fd7f6533d9838223a1e68a7dc6b049c06256d93333fb32975c044bbb83ed18",  # noqa: E501
        "superseded_because": (
            "its failed-attempt evidence was present and correctly bound, "
            "but the validator required only the ordered generation names. "
            "Removing both attempted_execution and its index SHA-256 from "
            "gen04, then recomputing supersession_digest, still passed both "
            "supersession_problems and the whole archive check. gen08 "
            "requires every not-executable entry to be exactly the record "
            "this module declares, so the binding cannot disappear as a "
            "self-consistent pair. The fix edits src/eval/phase3c.py, which "
            "the plan's scorer manifest pins, so the freeze has to be redone."),
        "cells_produced": 0,
        "sealed": False,
        "reached_a_node": False,
        "archive_retained_unmodified": True,
        "may_be_used_for": (
            "the record only. No run may cite it; it authorised nothing that "
            "ran and produced no measurement."),
        "not_deleted_because": (
            "it is the record of the freeze whose failed-attempt binding was "
            "correct in its own bytes but not required by its validator."),
    },
    {
        "generation": "gen06",
        "written": "2026-08-30",
        "plan": "data/phase3c/frozen/gen06/plan.json",
        "membership": "data/phase3c/frozen/gen06/case_membership.json",
        "audit": "data/phase3c/frozen/gen06/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen06/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen06/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "a16d5a80ad7514203207df86ca527889268e1b505df804d6546e02d2f4a09419",  # noqa: E501
        "pack_digest_it_names": "6086c00fc21dec12e9c36bd72d9b51d361f5753f066c30415e2f82635944e296",  # noqa: E501
        "pack_evidence_digest": "9a4f798dcc5cbd9bfc114bf70d8a8d67f8ee499265525bf5eaf28ae3b991d353",  # noqa: E501
        "superseded_because": (
            "two things it left fail-open, both reproduced. Its runner "
            "checked which gate the warm-up *should* use and never checked "
            "which one it did: warm_up returned a record and mode_run "
            "printed it, so a warm-up that ran another arm's gate, another "
            "arm's seeds, or failed outright would have been followed by "
            "1,920 measured cells with nothing anywhere recording the "
            "mismatch -- a warm-up is decoded and discarded, so no seal, "
            "receipt or score would ever have re-derived it. And its gen04 "
            "failed-attempt record was a document nothing read: the "
            "supersession entry named a directory by path, so an index "
            "edited to agree with itself, a zero-byte sample planted to "
            "stand for an absent one, or a seal appearing in that directory "
            "would all have passed. gen07 checks the warm-up against the "
            "arm spec before the first cell, and validates the failed "
            "attempt by reading it -- bound by its index's own SHA-256. "
            "Both fixes move scripts/59_phase3c.py and src/eval/phase3c.py, "
            "and the second is pinned by this plan's scorer manifest."),
        "cells_produced": 0,
        "sealed": False,
        "reached_a_node": False,
        "archive_retained_unmodified": True,
        "may_be_used_for": (
            "the record only. No run may cite it; it authorised nothing and "
            "nothing ran under it."),
        "not_deleted_because": (
            "it is the record of the freeze that cut the pack/V1 cycle, and "
            "of two checks that were described and not performed."),
    },
    {
        "generation": "gen05",
        "written": "2026-08-30",
        "plan": "data/phase3c/frozen/gen05/plan.json",
        "membership": "data/phase3c/frozen/gen05/case_membership.json",
        "audit": "data/phase3c/frozen/gen05/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen05/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen05/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "75e7e4ac73e46272ef0b7b014e3c171e5a93a1508d69304f0c4cf6db724d49cc",  # noqa: E501
        "pack_digest_it_names": "47f30b59a79c105fb57c5bbc20ba42d8fad659e2b8b86d0c9a9dc6c0d97b9e73",  # noqa: E501
        "pack_evidence_digest": "d4e263676cba488b048a94fc8dee448872fed930dd8f534b00945c7ab4411d89",  # noqa: E501
        # Superseded by a dependency cycle between two frozen artefacts,
        # not by anything wrong inside it. It never reached a node.
        "superseded_because": (
            "it was frozen inside a cycle between two artefacts that pin "
            "each other's source. Naming the current staged plan in "
            "src/training/pack.py meant a Phase 3C bump edited that file, "
            "and that file is inside the V1 visual run's import closure, "
            "whose frozen source manifest pins its SHA-256 -- so freezing "
            "gen05 invalidated V1's stored gen04 runs. Re-freezing V1 to "
            "repair that edits src/eval/visual_stress.py, which the Phase 3C "
            "pack carries, which invalidates this generation's pack digest "
            "and the authorization bound to it. Each repair broke the other. "
            "The cycle is cut by moving the generation out of pack.py and "
            "into a pointer document, and cutting it edits both pack.py and "
            "src/eval/phase3c.py -- the second is pinned by this plan's "
            "scorer manifest, so the freeze has to be redone once more."),
        "cells_produced": 0,
        "sealed": False,
        "reached_a_node": False,
        "archive_retained_unmodified": True,
        "may_be_used_for": (
            "the record only. No run may cite it; it authorised nothing and "
            "nothing ran under it."),
        "not_deleted_because": (
            "it is the evidence of the cycle: the one generation whose own "
            "contents were never in question and which still could not be "
            "used."),
    },
    {
        "generation": "gen04",
        "written": "2026-08-30",
        "plan": "data/phase3c/frozen/gen04/plan.json",
        "membership": "data/phase3c/frozen/gen04/case_membership.json",
        "audit": "data/phase3c/frozen/gen04/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen04/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen04/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "808708f7eba1ba8fb4a5da5f3b0f7bb30e73fbad30d313446fae8482a8fbc678",  # noqa: E501
        "pack_digest_it_names": "4af2895805fff8a0ed23f106c2657f435c72539d6cc8ff13690362dd0206e05f",  # noqa: E501
        "pack_evidence_digest": "7e93636f3d277ba7ca9891a46cd5c1fc33a86f135284bcd6ec4ab1450eb9f10c",  # noqa: E501
        # The one generation superseded by a defect outside its own
        # evidence. gen01-gen03 were replaced because something they bound
        # was not bound tightly enough; gen04 bound everything correctly and
        # was defeated by the runner.
        "superseded_because": (
            "its evidence chain held and its runner did not. The node run "
            "was attempted on 2026-08-30: every carried digest matched, "
            "every pinned environment field matched, the pack verified with "
            "zero problems, and step 0 then raised KeyError because "
            "scripts/59_phase3c.py resolved this phase's arm names through "
            "Phase 2's acceptance registry, in which 'A' does not exist. "
            "Only arm A fails loudly there. Arm B would have resolved to "
            "the published model with no gate and arm C to the fine-tuned "
            "model with no gate, while the measured cells went on using "
            "this phase's gates -- so the warm-up and the measurement would "
            "have run different decoders, and C - B would have measured the "
            "fine-tune rather than PlacementGate. The attempt produced 0 "
            "cells and no seal. Repairing it moves scripts/59_phase3c.py "
            "and src/eval/phase3c.py, and the second is pinned by this "
            "plan's scorer manifest, so the freeze has to be redone."),
        "attempted_execution": (
            "data/phase3c/failed_attempts/"
            "gen04_runner_defect_20260830T070331Z"),
        # The binding, rather than the path. ``attempted_execution_problems``
        # reads that index, requires it to be *these* bytes, and then runs
        # ``failed_attempt_problems`` over the directory. A record that named
        # a directory and nothing else could describe a sealed run and still
        # agree with itself.
        "attempted_execution_index_sha256": "88e89e3459ea9a72b6be58927e84f881713a450c692b210bdb86900f12d31505",  # noqa: E501
        "cells_produced": 0,
        "sealed": False,
        "archive_retained_unmodified": True,
        "may_be_used_for": (
            "the record only. No run may cite it, and its one attempted "
            "execution produced no measurement there would be anything to "
            "cite."),
        "not_deleted_because": (
            "it is the evidence of what the fourth freeze bound, and of the "
            "one defect a complete evidence chain could not catch: nothing "
            "it recorded was wrong about the data, only about which "
            "registry the runner read."),
    },
    {
        "generation": "gen03",
        "written": "2026-08-30",
        "plan": "data/phase3c/frozen/gen03/plan.json",
        "membership": "data/phase3c/frozen/gen03/case_membership.json",
        "audit": "data/phase3c/frozen/gen03/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen03/execution_authorization.json",
        "pack_evidence": "data/phase3c/frozen/gen03/pack_manifest.json",
        "plan_digest": "e3b591879bd06856295a22dab4e6d863a6ff1880b81e5e5f5d7701096377b2aa",  # noqa: E501
        "contract_digest": "887640775ed723d18db4629591b34818250aca630bb12e743afb2d7f195fa2a3",  # noqa: E501
        "authorization_digest": "876e8fff5b716e1ddf2e0d8d28bf53ce1a3c8f2bd65c0aa6e6ae196f024b4889",  # noqa: E501
        "pack_digest_it_names": "20935518724f83af5173f1090a5f3a02d2d5fa6ce713bdeab410b083f9888507",  # noqa: E501
        "pack_evidence_digest": "0dbad12b20ccbee34fec7ea2cb6fac88c63b2d9d94b6928e7dea5f154d7a22be",  # noqa: E501
        "superseded_because": (
            "its receipt compared a chosen subset of fields rather than a "
            "whole expected record, so a field nobody thought to check was "
            "a field nothing checked; its grant archived the pack file "
            "table without binding that table's digest, so a grant and an "
            "evidence file could describe different packs and each hold on "
            "its own; its contract snapshot carried a payload nothing "
            "compared to the digest beside it; its score record had no "
            "digest of its own and its identity covered eleven named "
            "fields, leaving the resampling rule, the primary contrast and "
            "the scorer manifest free to differ unnoticed; and its "
            "published members were compared as parsed objects, so a "
            "reformatted artefact passed as unchanged. Closing those moves "
            "src/eval/phase3c.py and the contract, both of which its plan "
            "pins."),
        "may_be_used_for": "the record only. No run may cite it.",
        "not_deleted_because": (
            "it is the evidence of what the third freeze bound, and the "
            "reason the fourth one binds more."),
    },
    {
        "generation": "gen02",
        "written": "2026-08-29",
        "plan": "data/phase3c/frozen/gen02/plan.json",
        "membership": "data/phase3c/frozen/gen02/case_membership.json",
        "audit": "data/phase3c/frozen/gen02/isolation_audit.json",
        "authorization":
            "data/phase3c/frozen/gen02/execution_authorization.json",
        "plan_digest": "89f4392fcd9806641578ad6999db349be0f73122f2c15e70cd819c5e211d9168",  # noqa: E501
        "contract_digest": "1570d082cf2c3297972a579ef7b6b7a1e344574dfc9520159ae35945c961a36a",  # noqa: E501
        "authorization_digest": "7e65308ea392637db570aebb4cd1844347e8df396dd802cbc1ae65a3590ad59f",  # noqa: E501
        "pack_digest_it_names": "2cdbbfc80ec4730216790282fc7758d8d044439e53de06fac24b4a4ac2954328",  # noqa: E501
        "superseded_because": (
            "the pack digest its grant names cannot be reproduced from any "
            "state of this tree that was kept: the pack was built into a "
            "scratch directory, the value was read off that build, and the "
            "file table it was taken over was never archived. An "
            "independent reviewer rebuilding a 60-file pack from the commit "
            "that froze it gets a different value, and there is no way to "
            "tell which bytes the authorised one described. A grant whose "
            "central binding nobody can re-derive does not bind anything, "
            "so gen02 may not authorise a run. Separately, the row-level "
            "manifest binding, the report chain and the CLI's own "
            "--authorization path were all fixed after it was written, and "
            "those fixes move src/eval/phase3c.py, which its plan pins."),
        "may_be_used_for": "the record only. No run may cite it.",
        "not_deleted_because": (
            "it is the evidence of what the second freeze bound and of why "
            "an unreproducible digest is not an acceptable binding."),
    },
    {
        "generation": "gen01",
        "written": "2026-08-29",
        "plan": "gpu_plans/phase3c_plan.json",
        "membership": "gpu_plans/phase3c_case_membership.json",
        "audit": "gpu_plans/phase3c_isolation_audit.json",
        # Whole, not split across two lines: a SHA-256 broken in half
        # leaves two 32-character hex strings, which is the dataset
        # identifier shape the public-snapshot audit refuses -- correctly,
        # since it cannot tell the halves from the thing they resemble.
        "plan_digest": "89f4392fcd9806641578ad6999db349be0f73122f2c15e70cd819c5e211d9168",  # noqa: E501
        "scorer_source_manifest_digest": "e10bffd4475f70e7b4213c7fc96ec125b2c46c4ba7b550d23fee2c9fa1b6e7cb",  # noqa: E501
        "superseded_because": (
            "it predates the execution grant, so nothing in it bound "
            "the pack digest, the dependency digest or the gate source; a "
            "manifest naming any well-formed digest would have been "
            "accepted. Its scorer manifest also pins an earlier "
            "src/eval/phase3c.py, so it cannot authorise a run against this "
            "one."),
        "plan_digest_shared_with": (
            "gen02. Those two ask for the same generation -- same cases, "
            "same order, same seeds, same budget, same arms -- and "
            "plan_digest deliberately excludes the scorer, exactly as Phase "
            "2's does, because a raw generation does not depend on the "
            "scorer. What separated them was "
            "scorer_source_manifest_digest and whether an execution "
            "grant exists at all -- gen01 has none, and a run cannot be "
            "authorised by a document that does not exist. gen03 differs "
            "from both, because its contract gained two acceptance "
            "criteria -- the grant with its archived pack file table, and "
            "the row-level binding to the execution manifest -- and "
            "contract_digest is inside plan_digest."),
        "may_be_used_for": "the record only. No run may cite it.",
        "not_deleted_because": (
            "a superseded plan is evidence about how this phase was frozen "
            "the first time, and deleting it would make that unreviewable."),
    },
)

#: The four names inside a run directory.
MANIFEST_NAME = "execution_manifest.json"
SEAL_NAME = "seal.json"
RECEIPT_NAME = "receipt.json"
SCORES_NAME = "scores.json"

#: What a published report consists of. Defined here rather than only in
#: :mod:`src.eval.phase3c_report` because ``rehash`` has to know them: the
#: report is written into the run directory, and a file the seal does not
#: name is otherwise reported as a stray. One vocabulary, two readers.
REPORT_NAME = "phase3c_report.md"
SUCCESS_INDEX_NAME = "success_cases.json"
FAILURE_INDEX_NAME = "failure_cases.json"
REPRODUCE_NAME = "reproduce.md"
PUBLISHED_NAMES: tuple[str, ...] = (REPORT_NAME, SUCCESS_INDEX_NAME,
                                    FAILURE_INDEX_NAME, REPRODUCE_NAME)


def read_test_rows(root=None) -> list[dict]:
    """The test split, once, after its SHA-256 matches what is pinned."""
    path = Path(root or ROOT) / TEST_FILE
    if not path.is_file():
        raise PlanRefused(f"{TEST_FILE} is not on this machine")
    actual = sha256_file(path)
    if actual != EXPECTED_TEST_SHA256:
        raise PlanRefused(
            f"{TEST_FILE} hashes to {actual[:16]}..., not the "
            f"{EXPECTED_TEST_SHA256[:16]}... this contract pins. The frozen "
            "exclusions would name different rows; refusing to open it.")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_phase2_plan(root=None) -> dict:
    """Phase 2's plan, verified two ways before a word of it is believed."""
    path = Path(root or ROOT) / PHASE2_PLAN
    if not path.is_file():
        raise PlanRefused(f"{PHASE2_PLAN} is not on this machine; the "
                          "exclusions cannot be computed without it")
    actual = sha256_file(path)
    if actual != EXPECTED_PHASE2_PLAN_SHA256:
        raise PlanRefused(
            f"{PHASE2_PLAN} hashes to {actual[:16]}..., not the "
            f"{EXPECTED_PHASE2_PLAN_SHA256[:16]}... this contract pins")
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("plan_digest") != EXPECTED_PHASE2_PLAN_DIGEST:
        raise PlanRefused(
            f"{PHASE2_PLAN} records plan_digest "
            f"{str(body.get('plan_digest'))[:16]}..., not the "
            f"{EXPECTED_PHASE2_PLAN_DIGEST[:16]}... this contract pins")
    if acceptance.plan_digest(body) != body["plan_digest"]:
        raise PlanRefused(f"{PHASE2_PLAN} carries a plan_digest that is not "
                          "the digest of its own contents")
    return body


def _pairs(rows) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["pair_id"], []).append(row)
    return out


def isolation_audit(root=None) -> dict:
    """Everything that decides which cases are eligible, and why.

    Read-only and self-contained: it opens the test split, the Phase 2 plan
    and the split membership file, and nothing else. **It never opens val**,
    and it says so in its own output.
    """
    root = Path(root or ROOT)
    rows = read_test_rows(root)
    phase2 = read_phase2_plan(root)
    pairs = _pairs(rows)

    for pair_id, members in pairs.items():
        if len(members) != ROWS_PER_PAIR:
            raise PlanRefused(f"pair {pair_id!r} has {len(members)} rows, "
                              f"not {ROWS_PER_PAIR}")

    pair_objects = {p: sorted({r[GROUP_KEY] for r in rs})
                    for p, rs in pairs.items()}
    multi = sorted(p for p, objs in pair_objects.items() if len(objs) != 1)
    if multi:
        raise PlanRefused(f"{len(multi)} pairs span more than one "
                          f"{GROUP_KEY}; the group key is not a group")
    pair_object = {p: objs[0] for p, objs in pair_objects.items()}

    used_pairs = sorted({c["pair_id"] for c in phase2["cases"]})
    used_samples = sorted({c["sample_id"] for c in phase2["cases"]})
    used_objects = sorted({pair_object[p] for p in used_pairs
                           if p in pair_object})
    used_captions = {sha256_text(c["caption"]) for c in phase2["cases"]}

    excluded_pair = sorted(set(used_pairs))
    excluded_group = sorted(p for p in pairs
                            if p not in set(used_pairs)
                            and pair_object[p] in set(used_objects))
    already = set(excluded_pair) | set(excluded_group)
    excluded_caption = sorted(
        p for p in pairs if p not in already
        and any(sha256_text(r["caption"]) in used_captions for r in pairs[p]))

    eligible = sorted(set(pairs) - already - set(excluded_caption))

    splits = json.loads((root / SPLIT_FILE).read_text(encoding="utf-8"))
    objects = splits.get("objects") or {}
    off_split = sorted({pair_object[p] for p in eligible
                        if objects.get(pair_object[p]) != "test"})

    # Duplicate captions inside the eligible set: not contamination, but two
    # selected pairs with the same caption would be one case counted twice.
    caption_owner: dict[str, list[str]] = {}
    for p in eligible:
        for r in pairs[p]:
            caption_owner.setdefault(sha256_text(r["caption"]), []).append(p)
    shared_captions = sorted(
        {tuple(sorted(set(owners))) for owners in caption_owner.values()
         if len(set(owners)) > 1})

    object_pairs: dict[str, list[str]] = {}
    for p in eligible:
        object_pairs.setdefault(pair_object[p], []).append(p)
    reused_objects = sorted(o for o, ps in object_pairs.items() if len(ps) > 1)

    return {
        "kind": AUDIT_KIND,
        "source": {
            "file": TEST_FILE,
            "sha256": EXPECTED_TEST_SHA256,
            "rows": len(rows),
            "pairs": len(pairs),
            "objects": len(set(pair_object.values())),
            "rows_per_pair": ROWS_PER_PAIR,
        },
        "prior_use": {
            "phase_2_core_eval": {
                "plan": PHASE2_PLAN,
                "plan_digest": EXPECTED_PHASE2_PLAN_DIGEST,
                "pairs": len(used_pairs),
                "cases": len(used_samples),
                "objects": len(used_objects),
                "verdict": ("a model evaluation on this split; every case, "
                            "every object of every case and every caption of "
                            "every case is excluded below"),
            },
            "f_oracle": {
                "script": "scripts/12_f_oracle.py",
                "file": "data/processed/counterfactual_test.jsonl",
                "covers": "the whole test split, all 200 pairs",
                "model_used": False,
                "retrieval_used": False,
                "verdict": ("a CP-SAT feasibility study of the reference "
                            "geometries. It ran no model, selected nothing "
                            "and tuned nothing, so it does not make a case "
                            "unusable for a model evaluation -- but it did "
                            "read these objects and is DISCLOSED rather than "
                            "excluded, because excluding it would leave no "
                            "test split at all."),
            },
            "bricknet_six_arm": {
                "cases": "data/bricknet/frozen/eval_cases.json",
                "split": "the BrickNet extension's own val captions",
                "verdict": ("a different corpus with a different part "
                            "vocabulary. The audit checks that no caption "
                            "digest is shared with this split."),
            },
        },
        "exclusions": {
            "rules": list(EXCLUSIONS),
            "by_pair_id": excluded_pair,
            "by_group": excluded_group,
            "by_caption": excluded_caption,
            "counts": {
                "by_pair_id": len(excluded_pair),
                "by_group": len(excluded_group),
                "by_caption": len(excluded_caption),
            },
        },
        "eligible": {
            "pairs": len(eligible),
            "cases_available": len(eligible) * ROWS_PER_PAIR,
            "pair_ids": eligible,
            "objects_in_more_than_one_eligible_pair": reused_objects,
            "pairs_sharing_a_caption": [list(t) for t in shared_captions],
        },
        "checks": {
            "every_eligible_object_is_a_test_object": not off_split,
            "objects_not_in_the_test_split": off_split,
            "eligible_pairs_are_enough_for_the_frozen_draw":
                len(eligible) >= N_PAIRS,
            "val_opened": False,
            "val_file": "data/processed/instruct_inv_val.jsonl",
            "test_split_opened": True,
        },
    }


def bricknet_caption_overlap(rows, root=None) -> list[str]:
    """Caption digests shared with the BrickNet six-arm evaluation set.

    Separate from :func:`isolation_audit` because the file is an extension
    artifact that may legitimately be absent; an empty list from a missing
    file would be a claim the audit did not earn, so this raises instead.
    """
    path = Path(root or ROOT) / "data/bricknet/frozen/eval_cases.json"
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; the BrickNet overlap check "
                          "cannot be answered by assuming it is empty")
    theirs = {c.get("caption_sha256")
              for c in json.loads(path.read_text(encoding="utf-8"))}
    mine = {sha256_text(r["caption"]): r["pair_id"] for r in rows}
    return sorted(mine[d] for d in (theirs & set(mine)))


def select_pairs(audit: dict) -> list[str]:
    """The frozen draw. Deterministic given the audit's eligible list."""
    eligible = list(audit["eligible"]["pair_ids"])
    if len(eligible) < N_PAIRS:
        raise PlanRefused(
            f"the exclusions leave {len(eligible)} eligible pairs and the "
            f"frozen draw needs {N_PAIRS}. The rule is not relaxed to make "
            "the number up.")
    shuffled = sorted(eligible)
    random.Random(SELECTION_SEED).shuffle(shuffled)
    taken: list[str] = []
    seen_objects: set[str] = set()
    lookup = {p: o for p, o in _audit_pair_objects(audit).items()}
    for pair_id in shuffled:
        obj = lookup[pair_id]
        if obj in seen_objects:
            continue
        seen_objects.add(obj)
        taken.append(pair_id)
        if len(taken) == N_PAIRS:
            break
    if len(taken) != N_PAIRS:
        raise PlanRefused(
            f"only {len(taken)} pairs have distinct {GROUP_KEY}s and the "
            f"frozen draw needs {N_PAIRS}")
    return taken


def _audit_pair_objects(audit: dict) -> dict[str, str]:
    """``pair_id -> object_id`` for the eligible pairs.

    Recomputed from the group index the audit carries rather than stored
    twice; the audit lists which objects own more than one eligible pair, and
    the selection needs the whole mapping.
    """
    return audit["_pair_object"]


def build_case(row: dict) -> dict:
    """One case, from one row. Phase 2's builder, reused verbatim."""
    return acceptance.build_case(row)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

GROUPS: tuple[str, ...] = ("even", "odd")

#: Even-indexed pairs run A, B, C; odd-indexed pairs run C, B, A. So neither
#: side of the primary contrast is always the one that ran second, and a
#: drift that depends on order shows up as a difference between the groups
#: rather than as a difference between the arms.
GROUP_ARM_ORDER: dict[str, tuple[str, ...]] = {
    "even": ("A", "B", "C"),
    "odd": ("C", "B", "A"),
}

STEP_ORDER: tuple[tuple[str, str], ...] = tuple(
    (group, name) for group in GROUPS for name in GROUP_ARM_ORDER[group])

N_STEPS = len(STEP_ORDER)

#: The same warm-up Phase 2 used, on a caption that is in no split. Decoded,
#: timed and thrown away.
WARMUP: dict = dict(acceptance.WARMUP)


def group_for_index(index: int) -> str:
    return GROUPS[index % len(GROUPS)]


def ordered_pair_ids(plan: dict) -> list[str]:
    seen: list[str] = []
    for case in plan["cases"]:
        if case["pair_id"] not in seen:
            seen.append(case["pair_id"])
    return seen


def plan_schedule(plan: dict) -> dict:
    pairs = ordered_pair_ids(plan)
    groups = {name: [p for i, p in enumerate(pairs)
                     if group_for_index(i) == name] for name in GROUPS}
    return {
        "groups": groups,
        "group_arm_order": {k: list(v) for k, v in GROUP_ARM_ORDER.items()},
        "steps": [{"step_index": i, "group": g, "arm": a,
                   "pairs": len(groups[g]),
                   "cells": len(groups[g]) * ROWS_PER_PAIR * SETTINGS.k}
                  for i, (g, a) in enumerate(STEP_ORDER)],
        "warmup": {**WARMUP, "seeds": list(WARMUP["seeds"])},
        "rationale": (
            "even-indexed pairs run A, B, C and odd-indexed pairs run C, B, "
            "A, so neither arm of C - B is always the one that ran second."),
    }


def step(index: int) -> tuple[str, str]:
    if not isinstance(index, int) or isinstance(index, bool) \
            or not 0 <= index < N_STEPS:
        raise ValueError(f"step {index!r} is not one of 0..{N_STEPS - 1}")
    return STEP_ORDER[index]


def step_cases(plan: dict, index: int) -> list[dict]:
    group, _name = step(index)
    wanted = set(plan_schedule(plan)["groups"][group])
    return [c for c in plan["cases"] if c["pair_id"] in wanted]


def step_cells(plan: dict, index: int) -> list[tuple]:
    """Every cell one step must produce, in the order it must produce them."""
    _group, name = step(index)
    digest = plan["plan_digest"]
    return [(digest, case["case_id"], name, seed)
            for case in step_cases(plan, index)
            for seed in SETTINGS.seeds]


def step_name(index: int) -> str:
    group, name = step(index)
    return f"step_{index:02d}_{group}_{name}"


def samples_member(index: int) -> str:
    return f"samples/{step_name(index)}.jsonl"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

PLAN_FIELDS: tuple[str, ...] = (
    "kind", "schema_version", "contract_version", "contract_digest",
    "settings_digest", "source", "selection", "arms", "settings",
    "final_model", "execution_environment", "schedule",
    "scorer_source_manifest", "scorer_source_manifest_digest",
    "case_membership_digest", "audit_digest", "cases", "carries",
    "acceptance_criteria", "note", "plan_digest",
)

ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "all cells present: one row per (case, arm, seed), none missing, none "
    "duplicated",
    "the three arms ran the same cases, seeds, prompt digests, inventories "
    "and decode budget, checked row by row against the plan",
    "every sample file published write-once and covered by a seal the Mac "
    "verified against a digest carried by a second route",
    "every reported number re-derived on the Mac from the stored samples by "
    "the scorer this plan pins",
    "the environment the node reports equals the environment this contract "
    "pinned, field by field",
    "an execution authorization, written after the pack was built and before "
    "anything reached the node, binds the pack digest, the dependency "
    "digest, the adapter, both gate sources and every module the decode path "
    "can reach; the file table the pack digest was taken over is archived "
    "beside it, so the authorised value can be recomputed with no pack "
    "present",
    "every stored row names the execution manifest that is in the run "
    "directory, checked at run, resume, seal, receipt and score",
    "no threshold: no number in this run is required to reach any value",
)


def source_document(n_cases: int, n_pairs: int) -> dict:
    return {
        "file": TEST_FILE,
        "sha256": EXPECTED_TEST_SHA256,
        # Not ``split``. That name is in ``FORBIDDEN_CASE_FIELDS``, and
        # ``plan_leak_problems`` walks the whole plan by key name rather than
        # by depth, so a field called ``split`` anywhere in it is refused --
        # correctly, since the check cannot know that this one holds the word
        # "test" and not a row's own split label. The leak rule is not
        # loosened to admit a field; the field is named so the rule can stay
        # exactly as strict as Phase 2 left it.
        "source_split": "test",
        "group_key": GROUP_KEY,
        "exclusions": list(EXCLUSIONS),
        "selection": SELECTION_RULE,
        "seed": SELECTION_SEED,
        "pairs": n_pairs,
        "rows_per_pair": ROWS_PER_PAIR,
        "cases": n_cases,
        "roles": list(ROLES),
        "variants": list(VARIANTS),
        "phase_2_cases_reused": False,
        "val_opened": False,
    }


def contract_document(root=None) -> dict:
    """The whole contract, as one value. Everything a number depends on."""
    return {
        "kind": KIND,
        "contract_version": CONTRACT_VERSION,
        "question": ("with model, checkpoint, adapter, prompt, inventory, "
                     "cases, seeds, decode budget and execution environment "
                     "held identical, how much does PlacementGate change "
                     "generation legality"),
        "primary_contrast": list(PRIMARY_CONTRAST),
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "contrasts": [list(c) for c in CONTRASTS],
        "settings": SETTINGS.as_dict(),
        "settings_digest": settings_digest(),
        "final_model": final_model_document(),
        "execution_environment": EXECUTION_ENVIRONMENT,
        "pinned_environment_fields": list(PINNED_ENVIRONMENT_FIELDS),
        "prompt_form": PROMPT_FORM,
        "terminations": list(TERMINATIONS),
        "accepted_terminations": list(ACCEPTED_TERMINATIONS),
        "eos_terminations": list(EOS_TERMINATIONS),
        "core_success_checks": list(CORE_SUCCESS_CHECKS),
        "metric_spec": METRIC_SPEC,
        "metric_spec_digest": METRIC_SPEC_DIGEST,
        "strata": list(STRATA),
        "roles": list(ROLES),
        "variants": list(VARIANTS),
        "uncertainty": {
            "level": CI_LEVEL,
            "wilson": WILSON_METHOD,
            "bootstrap": BOOTSTRAP_METHOD,
            "resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "significance_test": None,
            "note": ("no hypothesis test was pre-registered and none is run. "
                     "The intervals describe variation across these 160 "
                     "cases under a frozen resampling rule and say nothing "
                     "about captions outside them."),
        },
        "selection": {
            "file": TEST_FILE,
            "sha256": EXPECTED_TEST_SHA256,
            "phase_2_plan": PHASE2_PLAN,
            "phase_2_plan_digest": EXPECTED_PHASE2_PLAN_DIGEST,
            "group_key": GROUP_KEY,
            "exclusions": list(EXCLUSIONS),
            "rule": SELECTION_RULE,
            "seed": SELECTION_SEED,
            "pairs": N_PAIRS,
            "rows_per_pair": ROWS_PER_PAIR,
            "cases": N_CASES,
        },
        "schedule": {
            "groups": list(GROUPS),
            "group_arm_order": {k: list(v)
                                for k, v in GROUP_ARM_ORDER.items()},
            "steps": N_STEPS,
            "warmup": {**WARMUP, "seeds": list(WARMUP["seeds"])},
        },
        "acceptance_criteria": list(ACCEPTANCE_CRITERIA),
        "forbidden_metric_terms": list(FORBIDDEN_METRIC_TERMS),
        "case_fields": list(CASE_FIELDS),
        "scorer_sources": list(SCORER_SOURCES),
    }


def contract_digest(root=None) -> str:
    """One value over everything the numbers depend on.

    The scorer's *content* is deliberately outside it, exactly as in Phase 2:
    a raw generation does not depend on the scorer, and folding the scorer in
    would invalidate finished GPU work whenever a checker gained a comment.
    The scorer manifest is recorded and checked at score time instead.
    """
    return digest_obj(contract_document(root))


def plan_digest(body: dict) -> str:
    return digest_obj({
        "kind": body.get("kind"),
        "schema_version": body.get("schema_version"),
        "contract_digest": body.get("contract_digest"),
        "settings_digest": body.get("settings_digest"),
        "source": body.get("source"),
        "selection": body.get("selection"),
        "arms": body.get("arms"),
        "final_model": body.get("final_model"),
        "execution_environment": body.get("execution_environment"),
        "schedule": body.get("schedule"),
        "case_membership_digest": body.get("case_membership_digest"),
        "audit_digest": body.get("audit_digest"),
        "cases": body.get("cases"),
        "acceptance_criteria": body.get("acceptance_criteria"),
    })


def membership_document(cases: list[dict], pairs: list[str],
                        audit_digest: str) -> dict:
    """Who is in this evaluation, as a document of its own.

    Separate from the plan because membership is the claim most worth being
    able to hand somebody on its own: it carries no caption and no inventory,
    only the identifiers, their strata and the digests that tie each case to
    the prompt it will produce.
    """
    body = {
        "kind": MEMBERSHIP_KIND,
        "source_file": TEST_FILE,
        "source_sha256": EXPECTED_TEST_SHA256,
        "group_key": GROUP_KEY,
        "selection_seed": SELECTION_SEED,
        "selection_rule": SELECTION_RULE,
        "exclusions": list(EXCLUSIONS),
        "audit_digest": audit_digest,
        "pairs": list(pairs),
        "cases": [{"case_id": c["case_id"], "pair_id": c["pair_id"],
                   "role": c["role"], "variant": c["variant"],
                   "prompt_sha256": c["prompt_sha256"],
                   "inventory_digest": c["inventory_digest"]}
                  for c in cases],
        "n_pairs": len(pairs),
        "n_cases": len(cases),
    }
    body["membership_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "membership_digest"})
    return body


def materialize_plan(root=None) -> tuple[dict, dict, dict]:
    """Open the test split once and turn the frozen draw into a plan.

    Returns ``(plan, membership, audit)``. Nothing is written here; the caller
    decides where, and write-once decides whether.
    """
    root = Path(root or ROOT)
    rows = read_test_rows(root)
    audit = isolation_audit(root)

    overlap = bricknet_caption_overlap(rows, root)
    audit["prior_use"]["bricknet_six_arm"]["shared_caption_pairs"] = overlap
    if overlap:
        raise PlanRefused(
            f"{len(overlap)} pairs share a caption with the BrickNet "
            "evaluation set; the two evaluations would not be independent")

    pairs = _pairs(rows)
    audit["_pair_object"] = {p: sorted({r[GROUP_KEY] for r in rs})[0]
                             for p, rs in pairs.items()}
    chosen = select_pairs(audit)
    audit.pop("_pair_object")
    audit["selected"] = {
        "pairs": chosen,
        "n_pairs": len(chosen),
        "n_cases": len(chosen) * ROWS_PER_PAIR,
    }
    audit["audit_digest"] = digest_obj(
        {k: v for k, v in audit.items() if k != "audit_digest"})

    cases = []
    for pair_id in chosen:
        for row in sorted(pairs[pair_id], key=lambda r: r["sample_id"]):
            cases.append(build_case(row))
    if len(cases) != N_CASES:
        raise PlanRefused(f"the frozen draw produced {len(cases)} cases, "
                          f"not {N_CASES}")

    membership = membership_document(cases, chosen, audit["audit_digest"])

    body = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": contract_digest(root),
        "settings_digest": settings_digest(),
        "source": source_document(len(cases), len(chosen)),
        "selection": {"pairs": chosen, "seed": SELECTION_SEED,
                      "rule": SELECTION_RULE},
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "settings": SETTINGS.as_dict(),
        "final_model": final_model_document(),
        "execution_environment": EXECUTION_ENVIRONMENT,
        "scorer_source_manifest": scorer_manifest(ROOT),
        "scorer_source_manifest_digest": scorer_manifest_digest(ROOT),
        "case_membership_digest": membership["membership_digest"],
        "audit_digest": audit["audit_digest"],
        "cases": cases,
        "carries": list(CASE_FIELDS),
        "acceptance_criteria": list(ACCEPTANCE_CRITERIA),
        "note": ("Prompts only. The targets, the reference brick lists and "
                 "the parts they use stay on the Mac and are not in this "
                 "file."),
    }
    body["schedule"] = plan_schedule(body)
    body["plan_digest"] = plan_digest(body)

    problems = plan_problems(body, root=root)
    if problems:
        raise PlanRefused("refusing to write the plan:\n  - "
                          + "\n  - ".join(problems))
    return body, membership, audit


def plan_problems(body, *, root=None, check_scorer=True) -> list[str]:
    """Everything wrong with a plan, in one list. No early return."""
    problems: list[str] = []
    if not isinstance(body, dict):
        return [f"the plan is a {type(body).__name__}, not an object"]

    extra = sorted(set(body) - set(PLAN_FIELDS))
    missing = sorted(set(PLAN_FIELDS) - set(body))
    if extra:
        problems.append(f"the plan carries fields the contract does not "
                        f"name: {extra}")
    if missing:
        problems.append(f"the plan is missing {missing}")

    if body.get("kind") != PLAN_KIND:
        problems.append(f"kind is {body.get('kind')!r}, not {PLAN_KIND!r}")
    if body.get("schema_version") != PLAN_SCHEMA_VERSION:
        problems.append(f"schema_version is {body.get('schema_version')!r}")
    if body.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version is {body.get('contract_version')!r}")
    if body.get("contract_digest") != contract_digest(root):
        problems.append("contract_digest is not this contract's digest; the "
                        "plan was made under different rules")
    if body.get("settings_digest") != settings_digest():
        problems.append("settings_digest is not this contract's settings")
    if body.get("execution_environment") != EXECUTION_ENVIRONMENT:
        problems.append("execution_environment is not the one pinned here")

    arms = body.get("arms") or {}
    if sorted(arms) != sorted(ARM_ORDER):
        problems.append(f"the plan names arms {sorted(arms)}, not "
                        f"{sorted(ARM_ORDER)}")
    for name in sorted(set(arms) & set(ARM_ORDER)):
        if arms[name] != ARMS[name].as_dict():
            problems.append(f"arm {name} in the plan is not arm {name} here")

    cases = body.get("cases")
    if not isinstance(cases, list):
        problems.append("cases is not a list")
        return problems
    if len(cases) != N_CASES:
        problems.append(f"the plan has {len(cases)} cases, not {N_CASES}")
    ids = [c.get("case_id") for c in cases if isinstance(c, dict)]
    if len(set(ids)) != len(ids):
        problems.append("the plan repeats a case_id")
    pairs = {c.get("pair_id") for c in cases if isinstance(c, dict)}
    if len(pairs) != N_PAIRS:
        problems.append(f"the plan covers {len(pairs)} pairs, not {N_PAIRS}")

    for case in cases:
        if not isinstance(case, dict):
            problems.append("a case is not an object")
            continue
        try:
            inventory = canonical_inventory(case.get("inventory") or {})
        except PlanRefused as exc:
            problems.append(f"{case.get('case_id')!r}: {exc}")
            continue
        want = sha256_text(build_prompt(case.get("caption") or "", inventory))
        if case.get("prompt_sha256") != want:
            problems.append(f"{case.get('case_id')!r} carries a "
                            "prompt_sha256 that is not the digest of the "
                            "prompt its caption and inventory build")
        if case.get("inventory_digest") != digest_obj(inventory):
            problems.append(f"{case.get('case_id')!r} carries an "
                            "inventory_digest that is not its inventory's")
        if case.get("role") not in ROLES:
            problems.append(f"{case.get('case_id')!r} has role "
                            f"{case.get('role')!r}")
        if case.get("variant") not in VARIANTS:
            problems.append(f"{case.get('case_id')!r} has variant "
                            f"{case.get('variant')!r}")

    problems.extend(plan_leak_problems(body))

    if body.get("schedule") != plan_schedule(body):
        problems.append("the schedule is not the one this contract derives "
                        "from these cases")
    if check_scorer:
        problems.extend(scorer_manifest_problems(
            body.get("scorer_source_manifest"), root))
        if body.get("scorer_source_manifest_digest") != \
                digest_obj(body.get("scorer_source_manifest")):
            problems.append("scorer_source_manifest_digest is not the digest "
                            "of the manifest beside it")
    if list(body.get("acceptance_criteria") or []) != list(
            ACCEPTANCE_CRITERIA):
        problems.append("acceptance_criteria is not the frozen list")
    if body.get("plan_digest") != plan_digest(body):
        problems.append("plan_digest is not the digest of the plan's own "
                        "contents")
    return problems


def read_plan(path, *, root=None, check_scorer=True) -> dict:
    path = Path(path)
    if not path.is_file():
        raise PlanRefused(f"there is no plan at {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PlanRefused(f"{path} is not valid JSON ({exc}); a truncated "
                          "plan is not a plan") from exc
    problems = plan_problems(body, root=root, check_scorer=check_scorer)
    if problems:
        raise PlanRefused(f"{path} does not verify:\n  - "
                          + "\n  - ".join(problems))
    return body


def case_index(plan: dict) -> dict[str, dict]:
    return {c["case_id"]: c for c in plan["cases"]}


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

RESULT_FIELDS: tuple[str, ...] = (
    "plan_digest", "case_id", "arm", "seed", "step_index", "group",
    "manifest_digest", "raw_text", "n_tokens", "seconds", "termination",
    "truncated", "gate", "prompt_sha256", "inventory_digest",
    "contract_digest", "settings_digest", "model",
)


def model_identity() -> dict:
    """What a row must say about the weights. One model, so one answer."""
    return {
        "model": FINAL_MODEL,
        "loader": LOADER_FINAL,
        "base_model": SETTINGS.base_model,
        "base_revision": SETTINGS.base_revision,
        "published_adapter": SETTINGS.published_adapter,
        "published_adapter_revision": SETTINGS.published_adapter_revision,
        "adapter_files": dict(FINAL_ADAPTER_SHA256),
    }


# ---------------------------------------------------------------------------
# Loading an arm, and proving that what loaded is what the arm names
# ---------------------------------------------------------------------------
#
# Why this lives here and not in ``acceptance``. gen04's node run reached the
# node, passed every digest, every environment field and the whole preflight,
# and then died on ``KeyError: 'A' is not one of ['B', 'C', 'D', 'E']``: the
# runner had handed a Phase 3C arm name to Phase 2's registry. Only arm A is
# absent from that registry, so only arm A failed loudly. Arm B would have
# resolved to Phase 2's B -- the *published* model with *no* gate -- and arm C
# to Phase 2's C -- the fine-tuned model with *no* gate -- while the measured
# cells went on using Phase 3C's gates through :func:`run_case`. The run would
# have finished, looked ordinary, and answered a different question.
#
# So Phase 3C gets its own door, reading its own frozen ``ARMS``. There is no
# second table to keep in step: :func:`arm` is the only source, here as in
# :func:`run_case`, and every arm of this phase runs ``FINAL_MODEL``.


def default_loaders() -> dict:
    """The real loaders, resolved late so importing this module loads nothing.

    Phase 2's set, deliberately reused rather than restated -- but note that
    Phase 3C never opens the ``merged`` door, because no arm here runs the
    published model. It is present only because the mapping is shared.
    """
    return acceptance.default_loaders()


def build_interface(name: str, *, device: str, adapter_dir,
                    loaders=None, dtype=None):
    """The one door to a loaded Phase 3C arm. Returns ``(interface, info)``.

    All three arms load the same ``final_H2`` weights through the same
    loader, from the same adapter directory. That is the design: the only
    thing that differs between A, B and C is the gate, and the gate is not
    applied here -- it is applied by :func:`decode`, which every measured
    cell and :func:`warm_up` both go through.

    ``loaders`` is injected so a test can assert *which* loader ran, with
    which arguments, without a GPU, a network or a checkpoint.
    """
    spec = arm(name)                    # A/B/C, from the frozen spec
    if spec.model != FINAL_MODEL:
        raise PlanRefused(
            f"arm {name} names model {spec.model!r}; every arm of this phase "
            f"runs {FINAL_MODEL!r} and this loader builds nothing else")
    if adapter_dir is None:
        raise PlanRefused(
            f"arm {name} runs {FINAL_MODEL} and no adapter directory was "
            "given; the weights do not travel in the pack and must be named "
            "explicitly")
    loaders = loaders or default_loaders()
    if dtype is None:                   # resolved late, like the loaders
        import torch

        dtype = getattr(torch, SETTINGS.dtype)

    tok = loaders["tokenizer"](SETTINGS.tokenizer, SETTINGS.tokenizer_revision,
                               local_files_only=SETTINGS.local_files_only)
    model, info = loaders["finetuned"](
        adapter_dir, dtype=dtype, device=device, verify_digest=True,
        local_files_only=SETTINGS.local_files_only)
    return loaders["interface"](model, tok, device=device), info


def adapter_digests(adapter_dir) -> dict:
    """The three adapter files as they are on disk, hashed here and now.

    Read from the directory that was actually loaded, so the comparison is
    against the weights in play rather than against a value copied out of the
    contract into a variable named after it.
    """
    base = Path(adapter_dir)
    out = {}
    for name in sorted(FINAL_ADAPTER_SHA256):
        path = base / name
        out[name] = sha256_file(path) if path.is_file() else None
    return out


def observed_model_identity(info, adapter_dir) -> dict:
    """The row's model block, built from what the loader actually returned.

    Every value here comes from the load: the four revision fields from the
    loader's own ``info``, the adapter digests from the directory on disk.
    Nothing is copied from the contract. :func:`verified_model_identity` then
    requires this to equal :func:`model_identity`, so a row can only claim
    ``final_H2`` when ``final_H2`` is what loaded.
    """
    info = info if isinstance(info, dict) else {}
    return {
        "model": FINAL_MODEL,
        "loader": LOADER_FINAL,
        "base_model": info.get("base_model"),
        "base_revision": info.get("base_revision"),
        "published_adapter": info.get("published_adapter"),
        "published_adapter_revision": info.get("published_adapter_revision"),
        "adapter_files": adapter_digests(adapter_dir),
    }


def loaded_identity_problems(info, adapter_dir) -> list[str]:
    """Whether the weights that loaded are the ones this phase measures.

    ``load_finetuned`` is the only loader that reports ``local_adapter``,
    ``merge_changed_weights`` and the full four-step ``load_order``;
    ``load_merged_brickgpt`` -- the published model -- reports none of the
    first and a shorter order. Checking for them is therefore not a
    formality: it is what separates "the fine-tuned weights are loaded" from
    "something loaded and the row will say final_H2 anyway", which is exactly
    the substitution gen04's runner would have made for arm B.
    """
    from src.training.lora import LOAD_ORDER

    if not isinstance(info, dict):
        return [f"the loader returned a {type(info).__name__} where a load "
                "record was expected; nothing can be checked against it"]

    problems: list[str] = []
    local = info.get("local_adapter")
    if local is None:
        problems.append(
            "the load record has no local_adapter, which is what "
            "load_finetuned reports and load_merged_brickgpt does not. The "
            "published model is not an arm of this phase.")
    elif Path(str(local)).resolve() != Path(adapter_dir).resolve():
        problems.append(
            f"the load record names local_adapter {local!r}, which is not "
            f"the {str(adapter_dir)!r} this step was told to load")

    if tuple(info.get("load_order") or ()) != tuple(LOAD_ORDER):
        problems.append(
            f"the load order was {info.get('load_order')!r}, not "
            f"{list(LOAD_ORDER)!r}; the local delta must sit on the merged "
            "published adapter and not on bare Llama")

    if info.get("merge_changed_weights") is not True:
        problems.append(
            "the load record does not say the published adapter changed any "
            "weight when it merged, so the base may be bare Llama")

    for field in ("base_model", "base_revision", "published_adapter",
                  "published_adapter_revision"):
        want = getattr(SETTINGS, field)
        got = info.get(field)
        if got != want:
            problems.append(
                f"the load record says {field}={got!r}; the contract pins "
                f"{want!r}")

    on_disk = adapter_digests(adapter_dir)
    for name in sorted(FINAL_ADAPTER_SHA256):
        want, got = FINAL_ADAPTER_SHA256[name], on_disk.get(name)
        if got is None:
            problems.append(f"{name} is not in {adapter_dir}")
        elif got != want:
            problems.append(
                f"{name} on disk digests to {got[:16]}..., not the "
                f"{want[:16]}... this phase authorised")
    return problems


def verified_model_identity(info, adapter_dir) -> dict:
    """The model block for every row of this step, or a refusal.

    Called after the interface is built and **before** the warm-up and the
    first cell, so a step that loaded the wrong weights writes nothing at
    all rather than a member's worth of rows that name weights they did not
    come from.
    """
    problems = loaded_identity_problems(info, adapter_dir)
    if problems:
        raise PlanRefused("; ".join(problems))
    observed = observed_model_identity(info, adapter_dir)
    if observed != model_identity():
        raise PlanRefused(
            "the load matched every field checked individually and still "
            "does not equal the frozen model identity, so the row block "
            "cannot be written")
    return observed


def gate_class(gate_name: str):
    """The class an arm's declared gate names, resolved from that string.

    The arm spec states its gate as a dotted path, and this is where that
    string is turned into the class the decode is required to have built.
    Resolving it rather than importing two names at the top is what makes
    the spec the authority: a spec naming a gate that does not exist, or
    naming one class while the decode returns another, is a refusal here
    instead of a row that says whatever the spec said.
    """
    if gate_name == GATE_NONE:
        return None
    module_path, _, cls_name = gate_name.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    except (ImportError, AttributeError) as exc:
        raise PlanRefused(
            f"the gate this arm names, {gate_name}, does not resolve to a "
            f"class ({type(exc).__name__}: {exc})") from exc


def observed_gate(spec: Arm, gate) -> dict:
    """What the object a decode actually returned reports about itself.

    This is the one place the *real* gate object is judged, and both the
    warm-up and every measured cell go through it, so neither can report a
    gate the decode did not build.

    gen08 died here. ``gate_ledger`` called ``gate.counters()`` on whatever
    it was handed; arm A short-circuits on ``GATE_NONE`` and never reached
    the call, arm C's ``InventoryPlacementGate`` has the method, and arm B's
    ``InventoryGate`` does not -- so step 0 wrote 320 cells and step 1 died
    on ``AttributeError`` before its first. The test suite missed it because
    its stand-in gate implemented ``counters()``, making the substitute a
    superset of the object the node passes.

    The type check is **exact and not** ``isinstance``:
    ``InventoryPlacementGate`` subclasses ``InventoryGate``, so an
    ``isinstance`` test would let arm C's gate pass as arm B's and quietly
    measure the placement layer inside the arm that is supposed to be
    stock-only -- which is the ``C - B`` contrast measuring nothing.

    Raises :class:`PlanRefused` unless the object is exactly this arm's
    gate, so a step whose decode built the wrong one writes no row at all.
    """
    expected = gate_class(spec.gate)

    if expected is None:
        if gate is not None:
            raise PlanRefused(
                f"arm {spec.name} names no gate and the decode returned a "
                f"{type(gate).__module__}.{type(gate).__name__}")
        # No gate ran, so there is nothing to report and no counter to read.
        # ``unimplemented_counters`` is the honest answer here and only here.
        return {"gate": GATE_NONE, "connectivity": None,
                "accepted_parts": None, "remaining_inventory": None,
                "counters": acceptance.unimplemented_counters()}

    if gate is None:
        raise PlanRefused(
            f"arm {spec.name} runs through {spec.gate} and the decode "
            "returned no gate, so nothing enforced the inventory")
    if type(gate) is not expected:               # noqa: E721 -- exact
        raise PlanRefused(
            f"arm {spec.name} runs through {spec.gate} and the decode "
            f"returned a {type(gate).__module__}.{type(gate).__name__}")

    observed = {
        "gate": spec.gate,
        "connectivity": None,
        "accepted_parts": list(gate.accepted),
        "remaining_inventory": gate.inventory.as_dict(),
        # Arm B is stock alone: it has no placement layer, so it has no
        # placement counters to report. Saying so is not the same as saying
        # the layer ran and masked nothing, which is why this stays the
        # unimplemented block rather than becoming zeroes.
        "counters": acceptance.unimplemented_counters(),
    }
    if spec.gate != GATE_INVENTORY_PLACEMENT:
        return observed

    # Arm C has the layer, so its counters are a measurement and a missing
    # or unreadable one is a refusal. Falling back to the unimplemented
    # block here would turn "the placement layer did not report" into "there
    # is no placement layer", which is the defect one arm to the left.
    reader = getattr(gate, "counters", None)
    if not callable(reader):
        raise PlanRefused(
            f"arm {spec.name} runs the placement layer and its gate does "
            "not report counters")
    try:
        counters = reader()
    except Exception as exc:                     # noqa: BLE001 - fail closed
        raise PlanRefused(
            f"arm {spec.name}'s gate could not report its counters "
            f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(counters, dict):
        raise PlanRefused(
            f"arm {spec.name}'s gate reported a {type(counters).__name__} "
            "for its counters, not a record")
    if counters.get("connectivity") != spec.connectivity:
        raise PlanRefused(
            f"arm {spec.name} runs connectivity {spec.connectivity!r} and "
            f"its gate is in {counters.get('connectivity')!r}")
    observed["connectivity"] = counters["connectivity"]
    observed["counters"] = counters
    return observed


def gate_ledger(spec: Arm, opening: dict, gate=None) -> dict:
    """What the node reports about the gate, and nothing it had to parse.

    Every field but ``opening_inventory`` comes from :func:`observed_gate`,
    which reads the object the decode returned. ``connectivity`` in
    particular is arm C's gate's own state rather than a copy of the spec,
    so a row cannot claim a mode the gate was not in.
    """
    observed = observed_gate(spec, gate)
    return {
        "gate": observed["gate"],
        "connectivity": observed["connectivity"],
        "opening_inventory": dict(opening),
        "accepted_parts": observed["accepted_parts"],
        "remaining_inventory": observed["remaining_inventory"],
        "counters": observed["counters"],
    }


def decode(interface, spec: Arm, caption: str, opening: dict, kw):
    """One decode, through the one entry point each arm names."""
    from src.constraints.inventory_decode import generate_raw_with_inventory
    from src.constraints.placement_decode import generate_raw_with_placement
    from src.inventory.engine import Inventory

    if spec.gate == GATE_NONE:
        return interface.generate_raw(caption, inventory=opening, **kw), None
    inventory = Inventory.from_parts(dict(opening))
    if spec.gate == GATE_INVENTORY:
        return generate_raw_with_inventory(interface, caption, inventory, **kw)
    return generate_raw_with_placement(
        interface, caption, inventory=inventory, enabled=True,
        connectivity=spec.connectivity, **kw)


def decode_kwargs() -> dict:
    return {"max_bricks": SETTINGS.max_bricks,
            "max_tokens": SETTINGS.max_tokens,
            "temperature": SETTINGS.temperature}


def warm_up(interface, name: str) -> dict:
    """The frozen warm-up, through this arm's own gate. Never a measurement.

    It goes through :func:`decode` -- the same function every measured cell
    goes through -- so the warm state the first cell meets was produced under
    the same gate the cell will run under. gen04's runner used Phase 2's
    warm-up here, which for arm C meant warming the device with an ungated
    decode and then measuring a gated one; the two differ in how much work a
    step does per token, so it was not the warm-up the contract describes.

    ``gate`` and ``connectivity`` are returned so a step's evidence can say
    which gate the warm-up ran under, rather than leaving it to be assumed.
    """
    spec = arm(name)
    opening = canonical_inventory(WARMUP["inventory"])
    seconds = []
    observed = None
    for seed in WARMUP["seeds"]:
        raw, gate = decode(interface, spec, WARMUP["caption"], opening,
                           {**decode_kwargs(), "seed": seed})
        # Every generation, not only the last: a warm-up that built the
        # right gate once and the wrong one afterwards left the device in a
        # state the first cell does not match, and reporting the spec here
        # would have hidden it. ``observed_gate`` refuses rather than
        # returns, so a wrong gate stops the step before any cell.
        observed = observed_gate(spec, gate)
        seconds.append(float(raw.seconds))
    if observed is None:
        raise PlanRefused(
            f"arm {name}'s warm-up decoded nothing, so no gate was observed")
    return {
        "generations": len(seconds),
        "seeds": list(WARMUP["seeds"]),
        "arm": name,
        # From the gate the decode returned, not from the spec that asked
        # for it. gen06 reported the spec back to itself.
        "gate": observed["gate"],
        "connectivity": observed["connectivity"],
        "caption_is_from_the_test_split": False,
        "seconds": seconds,
        "excluded_from_every_reported_number": True,
        "policy": WARMUP["policy"],
    }


#: Every field a warm-up record must carry, and what it must equal. Two of
#: them are constants rather than arm-dependent, and both are claims the run
#: makes about its own honesty: that the warm-up caption is in no split, and
#: that its seconds reach no reported number.
WARMUP_FIELDS: tuple[str, ...] = (
    "arm", "gate", "connectivity", "seeds", "generations",
    "caption_is_from_the_test_split", "excluded_from_every_reported_number",
    "policy", "seconds",
)


def expected_warm_up(name: str) -> dict:
    """What a warm-up of this arm must report, derived rather than described.

    Built from :func:`arm` and :data:`WARMUP` -- the same two sources
    :func:`warm_up` reads -- so the comparison cannot drift into describing
    a third thing. ``seconds`` is absent here because its *values* are a
    measurement of this machine; only its length is fixed, and
    :func:`warm_up_problems` checks that against ``generations``.
    """
    spec = arm(name)
    return {
        "arm": name,
        "gate": spec.gate,
        "connectivity": spec.connectivity,
        "seeds": list(WARMUP["seeds"]),
        "generations": len(WARMUP["seeds"]),
        "caption_is_from_the_test_split": False,
        "excluded_from_every_reported_number": True,
        "policy": WARMUP["policy"],
    }


def warm_up_problems(warmup, name: str) -> list[str]:
    """Whether the warm-up that just ran is the warm-up the contract names.

    Checked **after** the warm-up returns and **before** the first cell,
    because a warm-up is the one part of a step that is decoded and then
    discarded: nothing downstream re-derives it, no digest covers it, and a
    step whose warm-up ran a different gate from its cells leaves no trace
    of that in any sample row. gen04's runner warmed arm C through Phase 2's
    ungated spec and would have measured it gated; the rows would have been
    written, sealed, verified and scored without a single check noticing.

    Fails closed on a missing field, a field of the wrong type, and a field
    of the right type with the wrong value -- and on extra fields, because a
    record carrying something this contract does not name was produced by
    something other than :func:`warm_up`.
    """
    if not isinstance(warmup, dict):
        return [f"the warm-up returned a {type(warmup).__name__}, not a "
                "record; nothing can be checked against it"]

    problems: list[str] = []
    extra = sorted(set(warmup) - set(WARMUP_FIELDS))
    missing = sorted(set(WARMUP_FIELDS) - set(warmup))
    if extra:
        problems.append(f"the warm-up record carries fields this contract "
                        f"does not name: {extra}")
    if missing:
        problems.append(f"the warm-up record is missing {missing}")

    expected = expected_warm_up(name)
    for field in sorted(set(expected) & set(warmup)):
        want, got = expected[field], warmup[field]
        if field == "seeds":
            got = list(got) if isinstance(got, (list, tuple)) else got
        # ``bool`` is an ``int`` in Python, so a record saying
        # ``generations: True`` would compare equal to 1 without this.
        if isinstance(want, bool) != isinstance(got, bool) \
                or type(want) is not type(got) and not (
                    isinstance(want, list) and isinstance(got, list)):
            problems.append(
                f"the warm-up reports {field}={got!r} "
                f"({type(got).__name__}); this arm's warm-up reports a "
                f"{type(want).__name__}")
            continue
        if got != want:
            problems.append(
                f"the warm-up reports {field}={got!r} and this arm's "
                f"warm-up is {want!r}")

    seconds = warmup.get("seconds")
    if "seconds" in warmup:
        if not isinstance(seconds, list) \
                or not all(isinstance(s, float) for s in seconds):
            problems.append("the warm-up's seconds are not a list of floats, "
                            "so it did not time what it decoded")
        elif len(seconds) != expected["generations"]:
            problems.append(
                f"the warm-up timed {len(seconds)} generation(s) and reports "
                f"{warmup.get('generations')!r}; the contract's warm-up is "
                f"{expected['generations']}")
        elif warmup.get("generations") == expected["generations"] \
                and any(s < 0 for s in seconds):
            problems.append("the warm-up reports a negative duration")
    return problems


def run_case(interface, case: dict, name: str, seed: int, *,
             plan_digest_value: str, step_index: int, group: str,
             manifest_digest: str, model: dict) -> dict:
    """Decode one (case, arm, seed) cell. Raw text out; nothing parsed.

    ``model`` is required and is the block :func:`verified_model_identity`
    returned for the weights this step actually loaded. It is not defaulted
    to :func:`model_identity`: a row that filled its own model block from the
    contract would say ``final_H2`` whatever had loaded, which is precisely
    what gen04's arm B would have done.
    """
    spec = arm(name)
    opening = canonical_inventory(case["inventory"])
    raw, gate = decode(interface, spec, case["caption"], opening,
                       {**decode_kwargs(), "seed": seed})
    built = build_prompt(case["caption"], opening)
    return {
        "plan_digest": plan_digest_value,
        "case_id": case["case_id"],
        "arm": name,
        "seed": seed,
        "step_index": step_index,
        "group": group,
        "manifest_digest": manifest_digest,
        "raw_text": raw.text,
        "n_tokens": raw.n_tokens,
        "seconds": float(raw.seconds),
        "termination": raw.termination,
        "truncated": bool(raw.truncated),
        "gate": gate_ledger(spec, opening, gate),
        "prompt_sha256": sha256_text(built),
        "inventory_digest": digest_obj(opening),
        "contract_digest": contract_digest(),
        "settings_digest": settings_digest(),
        "model": dict(model),
    }


def _is_digest(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value.lower()))


def row_problems(row, plan_cases: dict, *, plan: dict,
                 step_index: int | None = None,
                 manifest_digest: str | None = None) -> list[str]:
    """Whether one stored row is a cell this plan predetermined.

    ``manifest_digest`` is the digest of the execution manifest actually
    read and judged for this run, and passing it is what binds a row to the
    document saying which pack, weights and gate produced it. Before this,
    ``manifest_digest`` was a required *field* that nothing ever compared:
    a row could name any manifest, or one manifest could be swapped for
    another, and every layer -- run, resume, seal, receipt, score -- agreed.
    """
    problems: list[str] = []
    if not isinstance(row, dict):
        return [f"a row is a {type(row).__name__}, not an object"]
    extra = sorted(set(row) - set(RESULT_FIELDS))
    missing = sorted(set(RESULT_FIELDS) - set(row))
    if extra:
        problems.append(f"a row carries {extra}")
    if missing:
        problems.append(f"a row is missing {missing}")
    case_id = row.get("case_id")
    case = plan_cases.get(case_id)
    if case is None:
        problems.append(f"{case_id!r} is not a case in this plan")
        return problems
    name = row.get("arm")
    if name not in ARM_ORDER:
        problems.append(f"{case_id!r} names arm {name!r}")
        return problems
    if row.get("plan_digest") != plan["plan_digest"]:
        problems.append(f"{case_id!r}/{name} was produced under a different "
                        "plan")
    if row.get("contract_digest") != plan["contract_digest"]:
        problems.append(f"{case_id!r}/{name} was produced under a different "
                        "contract")
    if row.get("settings_digest") != plan["settings_digest"]:
        problems.append(f"{case_id!r}/{name} was produced under different "
                        "settings")
    stated = row.get("manifest_digest")
    if not _is_digest(stated):
        problems.append(f"{case_id!r}/{name} names execution manifest "
                        f"{str(stated)[:24]!r}, which is not a SHA-256")
    elif manifest_digest is not None and stated != manifest_digest:
        problems.append(
            f"{case_id!r}/{name} was produced under execution manifest "
            f"{stated[:16]}... and the manifest in this run digests to "
            f"{str(manifest_digest)[:16]}...")
    if row.get("seed") not in SETTINGS.seeds:
        problems.append(f"{case_id!r}/{name} has seed {row.get('seed')!r}")
    if row.get("prompt_sha256") != case["prompt_sha256"]:
        problems.append(f"{case_id!r}/{name} was prompted with something "
                        "other than this case's prompt")
    if row.get("inventory_digest") != case["inventory_digest"]:
        problems.append(f"{case_id!r}/{name} was offered a different "
                        "inventory")
    if row.get("model") != model_identity():
        problems.append(f"{case_id!r}/{name} names different weights")
    if row.get("termination") not in TERMINATIONS:
        problems.append(f"{case_id!r}/{name} ended on "
                        f"{row.get('termination')!r}")
    spec = arm(name)
    gate = row.get("gate") or {}
    if gate.get("gate") != spec.gate:
        problems.append(f"{case_id!r}/{name} ran gate {gate.get('gate')!r}, "
                        f"not {spec.gate!r}")
    if gate.get("connectivity") != spec.connectivity:
        problems.append(f"{case_id!r}/{name} ran connectivity "
                        f"{gate.get('connectivity')!r}, not "
                        f"{spec.connectivity!r}")
    if spec.gate == GATE_NONE and row.get("termination") in (
            "space_exhausted", "connectivity_unmet", "inventory_exhausted"):
        problems.append(f"{case_id!r}/{name} has no gate and ended on "
                        f"{row.get('termination')!r}, which only a gate "
                        "writes")
    seconds = row.get("seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) \
            or seconds < 0:
        problems.append(f"{case_id!r}/{name} has seconds {seconds!r}")
    n_tokens = row.get("n_tokens")
    if not isinstance(n_tokens, int) or isinstance(n_tokens, bool) \
            or n_tokens < 1:
        problems.append(f"{case_id!r}/{name} has n_tokens {n_tokens!r}")
    if step_index is not None and row.get("step_index") != step_index:
        problems.append(f"{case_id!r}/{name} claims step "
                        f"{row.get('step_index')!r}, not {step_index}")
    return problems


def read_rows(path) -> list[dict]:
    """Every row of a samples file, refusing a half-written last line."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                raise PlanRefused(
                    f"{path}: line {n} has no newline; the file was cut off "
                    "mid-write and a partial sample is not a sample")
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise PlanRefused(f"{path}: line {n} is not valid JSON "
                                  f"({exc})") from exc
    return rows


def cell_key(row: dict) -> tuple:
    return (row.get("case_id"), row.get("arm"), row.get("seed"))


def step_problems(rows, plan: dict, index: int,
                  *, manifest_digest: str | None = None) -> list[str]:
    """Whether these rows are exactly the cells the step was to produce.

    ``manifest_digest`` is threaded through to :func:`row_problems`, so a
    caller that has read and judged the run's execution manifest binds every
    row of the step to it rather than to a field the row states about itself.
    """
    problems: list[str] = []
    cases = case_index(plan)
    want = {(c, a, s) for _d, c, a, s in step_cells(plan, index)}
    seen: dict[tuple, int] = {}
    for row in rows:
        problems.extend(row_problems(row, cases, plan=plan, step_index=index,
                                     manifest_digest=manifest_digest))
        key = cell_key(row)
        seen[key] = seen.get(key, 0) + 1
    duplicated = sorted(k for k, n in seen.items() if n > 1)
    if duplicated:
        problems.append(f"{len(duplicated)} cells appear more than once, "
                        f"first {duplicated[0]}")
    missing = sorted(want - set(seen))
    if missing:
        problems.append(f"{len(missing)} cells the step must produce are "
                        f"absent, first {missing[0]}")
    extra = sorted(set(seen) - want)
    if extra:
        problems.append(f"{len(extra)} cells do not belong to this step, "
                        f"first {extra[0]}")
    return problems


# ---------------------------------------------------------------------------
# Statistics: the two frozen procedures, and nothing else
# ---------------------------------------------------------------------------

def wilson(successes: int, n: int) -> dict:
    """The frozen Wilson score interval. Total: n == 0 gives nulls."""
    if n <= 0:
        return {"value": None, "low": None, "high": None, "n": 0,
                "successes": successes}
    p = successes / n
    z = WILSON_Z
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"value": p, "low": max(0.0, centre - half),
            "high": min(1.0, centre + half), "n": n, "successes": successes}


def paired_bootstrap(values_a, values_b) -> dict:
    """A percentile interval for ``mean(a) - mean(b)`` over paired cases.

    ``values_a`` and ``values_b`` are per-case numbers in the same order --
    one entry per case, for the same cases. Resampling is over case indices,
    so both arms are resampled together and the pairing survives.
    """
    a, b = list(values_a), list(values_b)
    if len(a) != len(b):
        raise ValueError("the two arms do not have the same cases")
    n = len(a)
    point = (sum(a) / n - sum(b) / n) if n else None
    if n < 2:
        return {"delta": point, "low": None, "high": None, "n": n,
                "resamples": 0, "method": BOOTSTRAP_METHOD}
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sa = sb = 0.0
        for _j in range(n):
            i = rng.randrange(n)
            sa += a[i]
            sb += b[i]
        deltas.append(sa / n - sb / n)
    lo = (1 - CI_LEVEL) / 2
    return {
        "delta": point,
        "low": quantile(deltas, lo),
        "high": quantile(deltas, 1 - lo),
        "n": n,
        "resamples": BOOTSTRAP_RESAMPLES,
        "method": BOOTSTRAP_METHOD,
    }


def discordant(flags_a, flags_b) -> dict:
    """The two off-diagonal counts of a paired boolean. Counts only."""
    a, b = list(flags_a), list(flags_b)
    if len(a) != len(b):
        raise ValueError("the two arms do not have the same cases")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {
        "a_only": only_a,
        "b_only": only_b,
        "both": sum(1 for x, y in zip(a, b) if x and y),
        "neither": sum(1 for x, y in zip(a, b) if not x and not y),
        "n": len(a),
        "note": ("raw counts. No McNemar statistic and no p-value is "
                 "computed; no significance is claimed."),
    }


# ---------------------------------------------------------------------------
# Scoring: from stored samples to every number the report quotes
# ---------------------------------------------------------------------------

def score_row(row: dict, case: dict) -> dict:
    """One stored sample, scored, with this contract's token accounting."""
    from src.eval.scoring import score_generation

    if row["case_id"] != case["case_id"]:
        raise ValueError(f"row {row['case_id']!r} scored against case "
                         f"{case['case_id']!r}")
    inventory = canonical_inventory(case["inventory"])
    scored = score_generation(row["raw_text"], inventory=inventory,
                              n_tokens=row["n_tokens"],
                              termination=row["termination"])
    scored = correct_token_accounting(scored, n_tokens=row["n_tokens"],
                                      termination=row["termination"])
    gate = row.get("gate") or {}
    counters = gate.get("counters") or {}
    masked = counters.get("candidates_masked") or {}
    return {
        "case_id": row["case_id"],
        "pair_id": case["pair_id"],
        "role": case["role"],
        "variant": case["variant"],
        "arm": row["arm"],
        "seed": row["seed"],
        "step_index": row.get("step_index"),
        "group": row.get("group"),
        "seconds": row["seconds"],
        "n_tokens": row["n_tokens"],
        "gate_masked_candidates": counters.get("candidates_masked_total"),
        "gate_masked_by_slot": {str(k): v for k, v in masked.items()},
        "eos_deferrals": counters.get("eos_deferrals"),
        "bricks_placed": counters.get("bricks_placed"),
        **scored,
    }


def _rate(scores, name: str) -> dict:
    n = len(scores)
    hits = sum(1 for s in scores if s["checks"][name])
    out = wilson(hits, n)
    out["numerator"] = hits
    out["denominator"] = n
    return out


def _by_case(scores) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for s in scores:
        out.setdefault(s["case_id"], {})[s["seed"]] = s
    return out


def case_success_flags(scores, *, k: int) -> tuple[dict, dict, list]:
    """``(at_1, at_4, incomplete)`` keyed by case id.

    ``at_1`` reads the first frozen seed and nothing else; ``at_4`` needs all
    K seeds present, and a case missing one is in neither.
    """
    seeds = tuple(SETTINGS.seeds[:k])
    first = seeds[0]
    by_case = _by_case(scores)
    at_1: dict[str, bool] = {}
    at_4: dict[str, bool] = {}
    incomplete: list[str] = []
    for case_id, per_seed in sorted(by_case.items()):
        if first in per_seed:
            at_1[case_id] = bool(
                per_seed[first]["checks"]["deterministic_core_success"])
        if all(s in per_seed for s in seeds):
            at_4[case_id] = any(
                bool(per_seed[s]["checks"]["deterministic_core_success"])
                for s in seeds)
        else:
            incomplete.append(case_id)
    return at_1, at_4, incomplete


def arm_summary(scores, *, k: int) -> dict:
    """Every number one arm contributes, over one stratum."""
    at_1, at_4, incomplete = case_success_flags(scores, k=k)
    seconds = [s["seconds"] for s in scores]
    terminations: dict[str, int] = {t: 0 for t in TERMINATIONS}
    for s in scores:
        terminations[s["termination"]] = terminations.get(
            s["termination"], 0) + 1
    failures = sum(1 for s in scores if not s["checks"]["parse_success"])
    masked = [s["gate_masked_candidates"] for s in scores
              if s["gate_masked_candidates"] is not None]
    deferrals = [s["eos_deferrals"] for s in scores
                 if s["eos_deferrals"] is not None]
    by_slot: dict[str, int] = {}
    for s in scores:
        for slot, n in (s["gate_masked_by_slot"] or {}).items():
            by_slot[slot] = by_slot.get(slot, 0) + int(n)
    return {
        "draws": len(scores),
        "cases": len({s["case_id"] for s in scores}),
        "rates": {name: _rate(scores, name)
                  for name in tuple(DRAW_CHECKS)
                  + ("deterministic_core_success",)},
        "core_success_at_1": {
            **wilson(sum(1 for v in at_1.values() if v), len(at_1)),
            "seed": SETTINGS.seeds[0]},
        "core_success_at_4": {
            **wilson(sum(1 for v in at_4.values() if v), len(at_4)),
            "k": k, "incomplete_cases": incomplete},
        "generation_failure_rate": {
            **wilson(failures, len(scores)),
            "definition": METRIC_SPEC["generation_failure_rate"]["definition"]},
        "termination_reasons": terminations,
        "seconds": {
            "n": len(seconds),
            "total": sum(seconds),
            "mean": (sum(seconds) / len(seconds)) if seconds else None,
            "min": min(seconds) if seconds else None,
            "max": max(seconds) if seconds else None,
            "quantiles": quantiles(seconds),
        },
        "gate": {
            "masked_candidates_total": sum(masked) if masked else 0,
            "masked_candidates_mean": (sum(masked) / len(masked)
                                       if masked else None),
            "masked_by_slot": dict(sorted(by_slot.items())),
            "eos_deferrals_total": sum(deferrals) if deferrals else 0,
            "draws_with_a_deferral": sum(1 for d in deferrals if d),
            "unimplemented_counters": acceptance.unimplemented_counters(),
            "note": METRIC_SPEC["gate_masked_candidates"]["definition"],
        },
    }


def _stratum_scores(scores, kind: str, value):
    if kind == "overall":
        return list(scores)
    return [s for s in scores if s[kind] == value]


def strata_keys() -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = [("overall", None)]
    out.extend(("role", r) for r in ROLES)
    out.extend(("variant", v) for v in VARIANTS)
    return out


def stratum_key(kind: str, value) -> str:
    return kind if value is None else f"{kind}={value}"


def contrast(scores_a, scores_b, *, k: int) -> dict:
    """The paired comparison of two arms over one stratum."""
    a1, a4, _ = case_success_flags(scores_a, k=k)
    b1, b4, _ = case_success_flags(scores_b, k=k)
    shared4 = sorted(set(a4) & set(b4))
    shared1 = sorted(set(a1) & set(b1))

    def draw_delta(name: str) -> dict:
        ra, rb = _rate(scores_a, name), _rate(scores_b, name)
        return {"a": ra, "b": rb,
                "delta": (None if ra["value"] is None or rb["value"] is None
                          else ra["value"] - rb["value"])}

    per_case_a = _case_means(scores_a)
    per_case_b = _case_means(scores_b)
    shared_cases = sorted(set(per_case_a) & set(per_case_b))

    out = {
        "n_cases_paired": len(shared4),
        "draw_rate_deltas": {name: draw_delta(name)
                             for name in tuple(DRAW_CHECKS)
                             + ("deterministic_core_success",)},
        "core_success_at_1": {
            "a": wilson(sum(1 for c in shared1 if a1[c]), len(shared1)),
            "b": wilson(sum(1 for c in shared1 if b1[c]), len(shared1)),
            "bootstrap": paired_bootstrap([1.0 if a1[c] else 0.0
                                           for c in shared1],
                                          [1.0 if b1[c] else 0.0
                                           for c in shared1]),
            "discordant": discordant([a1[c] for c in shared1],
                                     [b1[c] for c in shared1]),
        },
        "core_success_at_4": {
            "a": wilson(sum(1 for c in shared4 if a4[c]), len(shared4)),
            "b": wilson(sum(1 for c in shared4 if b4[c]), len(shared4)),
            "bootstrap": paired_bootstrap([1.0 if a4[c] else 0.0
                                           for c in shared4],
                                          [1.0 if b4[c] else 0.0
                                           for c in shared4]),
            "discordant": discordant([a4[c] for c in shared4],
                                     [b4[c] for c in shared4]),
        },
        "paired_check_deltas": {
            name: paired_bootstrap(
                [per_case_a[c][name] for c in shared_cases],
                [per_case_b[c][name] for c in shared_cases])
            for name in tuple(DRAW_CHECKS) + ("deterministic_core_success",)
        },
        "paired_seconds": paired_bootstrap(
            [per_case_a[c]["seconds"] for c in shared_cases],
            [per_case_b[c]["seconds"] for c in shared_cases]),
    }
    return out


def _case_means(scores) -> dict[str, dict]:
    """Per-case means of every draw boolean and of seconds.

    The bootstrap resamples cases, so a case has to be one number per
    quantity; the mean over its K seeds is that number.
    """
    by_case = _by_case(scores)
    out: dict[str, dict] = {}
    for case_id, per_seed in by_case.items():
        draws = list(per_seed.values())
        entry = {name: sum(1 for d in draws if d["checks"][name]) / len(draws)
                 for name in tuple(DRAW_CHECKS)
                 + ("deterministic_core_success",)}
        entry["seconds"] = sum(d["seconds"] for d in draws) / len(draws)
        out[case_id] = entry
    return out


def per_case_records(scores_by_arm: dict, *, k: int) -> list[dict]:
    """One record per case: every arm's verdict, and the primary contrast."""
    seeds = list(SETTINGS.seeds[:k])
    case_ids = sorted({s["case_id"] for arm_scores in scores_by_arm.values()
                       for s in arm_scores})
    flags = {name: case_success_flags(scores_by_arm[name], k=k)
             for name in scores_by_arm}
    by_arm_case = {name: _by_case(scores_by_arm[name])
                   for name in scores_by_arm}
    a, b = PRIMARY_CONTRAST
    out = []
    for case_id in case_ids:
        record: dict = {"case_id": case_id, "arms": {}}
        for name in ARM_ORDER:
            if name not in by_arm_case:
                continue
            per_seed = by_arm_case[name].get(case_id, {})
            any_draw = next(iter(per_seed.values()), None)
            if any_draw is not None:
                record.setdefault("pair_id", any_draw["pair_id"])
                record.setdefault("role", any_draw["role"])
                record.setdefault("variant", any_draw["variant"])
            record["arms"][name] = {
                "seeds": {str(s): {
                    "checks": per_seed[s]["checks"],
                    "termination": per_seed[s]["termination"],
                    "n_bricks": per_seed[s]["parse"]["n_bricks"],
                    "seconds": per_seed[s]["seconds"],
                    "gate_masked_candidates":
                        per_seed[s]["gate_masked_candidates"],
                    "eos_deferrals": per_seed[s]["eos_deferrals"],
                } for s in seeds if s in per_seed},
                "core_success_at_1": flags[name][0].get(case_id),
                "core_success_at_4": flags[name][1].get(case_id),
            }
        if a in record["arms"] and b in record["arms"]:
            av = record["arms"][a]["core_success_at_4"]
            bv = record["arms"][b]["core_success_at_4"]
            record["primary_contrast"] = {
                "contrast": contrast_name(a, b),
                a: av, b: bv,
                "verdict": (None if av is None or bv is None else
                            "same" if av == bv else
                            f"{a} only" if av else f"{b} only"),
            }
        out.append(record)
    return out


def score_record(scores_by_arm: dict, *, plan: dict, k: int,
                 root=None) -> dict:
    """Everything the report quotes, derived here and nowhere else."""
    per_arm: dict = {}
    for name in ARM_ORDER:
        arm_scores = scores_by_arm.get(name, [])
        per_arm[name] = {
            stratum_key(kind, value): arm_summary(
                _stratum_scores(arm_scores, kind, value), k=k)
            for kind, value in strata_keys()}
    contrasts: dict = {}
    for a, b in CONTRASTS:
        contrasts[contrast_name(a, b)] = {
            stratum_key(kind, value): contrast(
                _stratum_scores(scores_by_arm.get(a, []), kind, value),
                _stratum_scores(scores_by_arm.get(b, []), kind, value), k=k)
            for kind, value in strata_keys()}
    record = {
        "kind": SCORES_KIND,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan["plan_digest"],
        "case_membership_digest": plan["case_membership_digest"],
        "audit_digest": plan["audit_digest"],
        "arms": list(ARM_ORDER),
        "primary_contrast": list(PRIMARY_CONTRAST),
        "k": k,
        "seeds": list(SETTINGS.seeds[:k]),
        "draws": sum(len(v) for v in scores_by_arm.values()),
        "cases": len({s["case_id"] for v in scores_by_arm.values()
                      for s in v}),
        "scorer_source_manifest": scorer_manifest(root),
        "scorer_source_manifest_digest": scorer_manifest_digest(root),
        "plan_scorer_source_manifest_digest":
            plan["scorer_source_manifest_digest"],
        "per_arm": per_arm,
        "contrasts": contrasts,
        "per_case": per_case_records(scores_by_arm, k=k),
        "uncertainty": {
            "level": CI_LEVEL,
            "wilson": WILSON_METHOD,
            "bootstrap": BOOTSTRAP_METHOD,
            "significance_test": None,
        },
        "note": ("collision_free is 1.0 in arm C and inventory_valid is 1.0 "
                 "in arms B and C by construction, not by improvement. "
                 "stud_only_connected and unsupported_brick_count are "
                 "geometric and are not support, stability or physics."),
    }
    record["scores_digest"] = digest_obj(
        {k: v for k, v in record.items() if k != "scores_digest"})
    return record


# ---------------------------------------------------------------------------
# The execution authorization
# ---------------------------------------------------------------------------
#
# The gap this closes. Until this existed the execution manifest *recorded* a
# ``pack_digest`` and a ``dependency_digest`` and nothing ever compared them
# to anything: any two well-formed hex strings were accepted, on the node and
# again on the Mac. Worse, nothing bound the **gate source** at all -- the
# whole question this phase asks is what ``InventoryPlacementGate`` does, and
# a run could have been produced by an edited copy of it with every digest in
# the bundle still agreeing with every other.
#
# So the order is now: materialise the plan, build the pack, read the two
# digests off that build, and write an authorization that names them together
# with the digest of every module the generation path can reach. The node
# refuses to start unless its own recomputation of all of them equals what
# the authorization says, and the Mac refuses to score unless the manifest
# the node wrote carries the authorization's digest and repeats its fields.
#
# The authorization is write-once and lives beside the plan, not inside the
# run directory: a document that authorises a directory cannot live in the
# directory it authorises.

AUTHORIZATION_KIND = "brickagain.phase3c_execution_authorization"

#: The node's entry point. Its import closure is what "the generation source"
#: means, computed rather than listed so a new module reached by the decode
#: path cannot quietly escape the digest.
GENERATION_ENTRY_POINT = "scripts/59_phase3c.py"

#: The two gate implementations, named separately from the closure as well.
#: They are already inside it; naming them is what makes a test able to say
#: "the layer under test is pinned" without depending on the closure's
#: contents staying the same shape.
GATE_SOURCES: tuple[str, ...] = (
    "src/constraints/inventory_decode.py",
    "src/constraints/placement_decode.py",
)


def generation_sources(root=None) -> tuple[str, ...]:
    """Every file the node can reach from the entry point, plus the entry.

    Computed by static import closure rather than listed by hand. A list
    would be a promise that somebody remembered to update it; this is a
    measurement, and :func:`generation_source_problems` fails closed when it
    stops matching.
    """
    from src.training import pack as pack_module

    base = Path(root or ROOT)
    closure = pack_module.import_closure(
        base, entry_points=(GENERATION_ENTRY_POINT,))
    return tuple(sorted(set(closure) | {GENERATION_ENTRY_POINT}))


def generation_source_manifest(root=None) -> dict:
    base = Path(root or ROOT)
    return {name: sha256_file(base / name)
            for name in generation_sources(base)}


def generation_source_manifest_digest(root=None) -> str:
    return digest_obj(generation_source_manifest(root))


def build_authorization(*, plan: dict, pack_digest: str,
                        dependency_digest: str, pack_evidence_digest: str,
                        authorized_at: str, root=None) -> dict:
    """Bind, before anything runs, what may run and what it may run against.

    ``pack_digest`` and ``dependency_digest`` come from the Mac's own
    ``18_gpu_pack.py`` build and ``--dependencies`` read. They are not
    invented here and they are not read back out of the pack: the point is
    that the value in this file arrived from the tool that computed it, and
    the node recomputes it independently.
    """
    root = Path(root or ROOT)
    for label, value in (("pack_digest", pack_digest),
                         ("dependency_digest", dependency_digest),
                         ("pack_evidence_digest", pack_evidence_digest)):
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value.lower()):
            raise PlanRefused(
                f"{label} must be a 64-character SHA-256; got {value!r}")
    sources = generation_source_manifest(root)
    body = {
        "kind": AUTHORIZATION_KIND,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan["plan_digest"],
        "settings_digest": plan["settings_digest"],
        "case_membership_digest": plan["case_membership_digest"],
        "audit_digest": plan["audit_digest"],
        "scorer_source_manifest_digest":
            plan["scorer_source_manifest_digest"],
        "pack_digest": pack_digest,
        # The evidence for the value above, bound *into* the grant rather
        # than merely archived beside it. gen03 archived the file table and
        # nothing tied the two together, so a grant and an evidence file
        # could describe different packs and each hold on its own.
        "pack_evidence_digest": pack_evidence_digest,
        "dependency_digest": dependency_digest,
        "adapter_sha256": dict(FINAL_ADAPTER_SHA256),
        "final_model": final_model_document(),
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "gate_sources": {name: sources[name] for name in GATE_SOURCES},
        "generation_entry_point": GENERATION_ENTRY_POINT,
        "generation_source_manifest": sources,
        "generation_source_manifest_digest": digest_obj(sources),
        "execution_environment": EXECUTION_ENVIRONMENT,
        "pinned_environment_fields": list(PINNED_ENVIRONMENT_FIELDS),
        "authorized_at": authorized_at,
        "note": ("The node may start only if its own recomputation of the "
                 "pack digest, the dependency digest and every source digest "
                 "here agrees field by field. The Mac may score only if the "
                 "execution manifest carries this authorization's digest and "
                 "repeats these fields. pack_evidence_digest names the "
                 "archived file table the pack digest was taken over, so "
                 "the binding can be rechecked with no pack present."),
    }
    body["authorization_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "authorization_digest"})
    return body


def generation_source_problems(recorded, root=None) -> list[str]:
    """Whether the generation path on this machine is the authorised one."""
    problems: list[str] = []
    if not isinstance(recorded, dict):
        return [f"the generation source manifest is a "
                f"{type(recorded).__name__}"]
    actual = generation_source_manifest(root)
    for name in sorted(set(recorded) | set(actual)):
        want, got = recorded.get(name), actual.get(name)
        if want is None:
            problems.append(f"{name} is on this machine and is not in the "
                            "authorised generation source")
        elif got is None:
            problems.append(f"{name} is authorised and is not on this "
                            "machine")
        elif want != got:
            problems.append(
                f"{name} hashes to {got[:16]}... and the authorization says "
                f"{want[:16]}...")
    return problems


def authorization_problems(auth, plan: dict, *, root=None,
                           check_sources: bool = True) -> list[str]:
    """Everything wrong with an grant, against the plan and the tree.

    ``check_sources`` is False only where the caller is on a machine that is
    deliberately not the generation machine and is checking the document's
    internal consistency -- never as a way of accepting an unverified source
    manifest on the node.
    """
    problems: list[str] = []
    if not isinstance(auth, dict):
        return [f"the authorization is a {type(auth).__name__}"]
    if auth.get("kind") != AUTHORIZATION_KIND:
        problems.append(f"kind is {auth.get('kind')!r}")
    for field in ("contract_digest", "plan_digest", "settings_digest",
                  "case_membership_digest", "audit_digest",
                  "scorer_source_manifest_digest"):
        if auth.get(field) != plan.get(field):
            problems.append(f"{field} does not match the plan")
    if auth.get("adapter_sha256") != dict(FINAL_ADAPTER_SHA256):
        problems.append("the adapter digests are not the contract's")
    if auth.get("execution_environment") != EXECUTION_ENVIRONMENT:
        problems.append("the authorised environment is not the contract's")
    if auth.get("arms") != {name: ARMS[name].as_dict()
                            for name in ARM_ORDER}:
        problems.append("the authorised arms are not the contract's")
    if auth.get("generation_entry_point") != GENERATION_ENTRY_POINT:
        problems.append("the authorised entry point is not "
                        f"{GENERATION_ENTRY_POINT}")
    for field in ("pack_digest", "dependency_digest", "pack_evidence_digest"):
        if not _is_digest(auth.get(field)):
            problems.append(
                f"{field} is {str(auth.get(field))[:24]!r}, not a SHA-256")

    sources = auth.get("generation_source_manifest")
    if not isinstance(sources, dict):
        problems.append("the generation source manifest is missing")
    else:
        if auth.get("generation_source_manifest_digest") != \
                digest_obj(sources):
            problems.append("generation_source_manifest_digest does not "
                            "cover the generation source manifest")
        gates = auth.get("gate_sources") or {}
        for name in GATE_SOURCES:
            if gates.get(name) != sources.get(name):
                problems.append(
                    f"{name} is pinned twice in this authorization and the "
                    "two values disagree")
        if check_sources:
            problems.extend(generation_source_problems(sources, root))

    body = {k: v for k, v in auth.items() if k != "authorization_digest"}
    if auth.get("authorization_digest") != digest_obj(body):
        problems.append("authorization_digest does not cover the "
                        "authorization")
    return problems


def carried_digest_problems(auth, *, pack_digest, dependency_digest
                            ) -> list[str]:
    """The two values the operator carried, against the authorised ones.

    This is where the old design failed open. It accepted any well-formed
    digest because nothing held a value to compare against; now the
    authorization holds them, and a pack that is not the authorised pack
    stops the run before a weight loads.
    """
    problems = []
    for label, carried in (("pack_digest", pack_digest),
                           ("dependency_digest", dependency_digest)):
        want = (auth or {}).get(label)
        if not carried:
            problems.append(f"--expected-{label.replace('_', '-')} was not "
                            "given, and the authorization names one")
        elif carried != want:
            problems.append(
                f"{label} carried here is {str(carried)[:16]}... and the "
                f"authorization names {str(want)[:16]}...")
    return problems


def read_authorization(path, plan: dict, *, root=None,
                       check_sources: bool = True) -> dict:
    path = Path(path)
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; a run is authorised before "
                          "it starts, not after")
    body = json.loads(path.read_text())
    problems = authorization_problems(body, plan, root=root,
                                      check_sources=check_sources)
    if problems:
        raise PlanRefused("this authorization does not hold:\n  - "
                          + "\n  - ".join(problems))
    return body


# ---------------------------------------------------------------------------
# The execution manifest, the seal and the receipt
# ---------------------------------------------------------------------------

def environment_problems(observed: dict) -> list[str]:
    """Whether the node is the machine this contract pinned, field by field."""
    problems = []
    for field in PINNED_ENVIRONMENT_FIELDS:
        want = EXECUTION_ENVIRONMENT[field]
        got = (observed or {}).get(field)
        if got != want:
            problems.append(f"{field} is {got!r} and the contract pins "
                            f"{want!r}")
    return problems


def build_execution_manifest(*, plan: dict, grant: dict,
                             observed: dict, started_at: str,
                             recomputed_sources: dict | None = None) -> dict:
    """What ran, where, against what. Written once, before the first cell.

    Every digest here is copied from the authorization rather than from an
    argument, so the manifest cannot name a pack the run was not authorised
    for. ``recomputed_sources`` is the node's *own* reading of the generation
    source, stored beside the authorised one: a reader who does not trust
    either machine can compare them without re-running anything.
    """
    body = {
        "kind": MANIFEST_KIND,
        "contract_version": CONTRACT_VERSION,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan["plan_digest"],
        "settings_digest": plan["settings_digest"],
        "case_membership_digest": plan["case_membership_digest"],
        "audit_digest": plan["audit_digest"],
        "authorization_digest": grant["authorization_digest"],
        "arms": {name: ARMS[name].as_dict() for name in ARM_ORDER},
        "settings": SETTINGS.as_dict(),
        "final_model": final_model_document(),
        "adapter_sha256": dict(grant["adapter_sha256"]),
        "pack_digest": grant["pack_digest"],
        "dependency_digest": grant["dependency_digest"],
        "gate_sources": dict(grant["gate_sources"]),
        "generation_source_manifest_digest":
            grant["generation_source_manifest_digest"],
        "generation_source_manifest_recomputed_here":
            dict(recomputed_sources or grant[
                "generation_source_manifest"]),
        "environment_pinned": EXECUTION_ENVIRONMENT,
        "environment_observed": observed,
        "steps": [{"step_index": i, "group": g, "arm": a,
                   "member": samples_member(i)}
                  for i, (g, a) in enumerate(STEP_ORDER)],
        "started_at": started_at,
    }
    body["manifest_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "manifest_digest"})
    return body


def manifest_problems(manifest, plan: dict, grant=None) -> list[str]:
    """Whether this manifest describes a run this plan and grant allow.

    ``authorization`` is optional only so a caller can check a manifest's
    internal shape in isolation. Every path that decides whether a *result*
    may be quoted passes it, and omitting it there is what the fail-open
    this replaces looked like.
    """
    problems = []
    if not isinstance(manifest, dict):
        return [f"the manifest is a {type(manifest).__name__}"]
    if manifest.get("kind") != MANIFEST_KIND:
        problems.append(f"kind is {manifest.get('kind')!r}")
    for field in ("contract_digest", "plan_digest", "settings_digest",
                  "case_membership_digest", "audit_digest"):
        if manifest.get(field) != plan.get(field):
            problems.append(f"{field} does not match the plan")
    if manifest.get("adapter_sha256") != dict(FINAL_ADAPTER_SHA256):
        problems.append("the adapter digests are not the contract's")
    problems.extend(environment_problems(
        manifest.get("environment_observed")))

    if grant is not None:
        for field in ("authorization_digest", "pack_digest",
                      "dependency_digest", "adapter_sha256", "gate_sources",
                      "generation_source_manifest_digest"):
            want = grant.get(field)
            if manifest.get(field) != want:
                problems.append(
                    f"{field} in the manifest is not the authorised value")
        recomputed = manifest.get(
            "generation_source_manifest_recomputed_here")
        if recomputed != grant.get("generation_source_manifest"):
            problems.append(
                "the generation source the node recomputed is not the one "
                "the authorization pins")

    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != digest_obj(body):
        problems.append("manifest_digest does not cover the manifest")
    return problems


def build_seal(out_dir: Path, manifest: dict) -> dict:
    """Size and SHA-256 of every published sample file, plus the manifest."""
    out_dir = Path(out_dir)
    members = {}
    for index in range(N_STEPS):
        member = samples_member(index)
        path = out_dir / member
        if not path.is_file():
            raise PlanRefused(f"{member} is not here; a seal names what "
                              "exists")
        members[member] = {"size": path.stat().st_size,
                           "sha256": sha256_file(path),
                           "rows": len(read_rows(path))}
    manifest_path = out_dir / "execution_manifest.json"
    body = {
        "kind": SEAL_KIND,
        "contract_digest": manifest["contract_digest"],
        "plan_digest": manifest["plan_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "arms": list(ARM_ORDER),
        "n_cases": N_CASES,
        "n_steps": N_STEPS,
        "expected_rows": N_CASES * SETTINGS.k * len(ARM_ORDER),
        "members": dict(sorted(members.items())),
        "execution_manifest": {
            "size": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
    }
    body["seal_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "seal_digest"})
    return body


def seal_problems(seal, plan: dict) -> list[str]:
    problems = []
    if not isinstance(seal, dict):
        return [f"the seal is a {type(seal).__name__}"]
    if seal.get("kind") != SEAL_KIND:
        problems.append(f"kind is {seal.get('kind')!r}")
    if seal.get("plan_digest") != plan["plan_digest"]:
        problems.append("the seal is for a different plan")
    if seal.get("contract_digest") != plan["contract_digest"]:
        problems.append("the seal is for a different contract")
    members = seal.get("members") or {}
    want = {samples_member(i) for i in range(N_STEPS)}
    if set(members) != want:
        problems.append(f"the seal names {sorted(set(members) - want)} and "
                        f"omits {sorted(want - set(members))}")
    total = sum(int(m.get("rows") or 0) for m in members.values())
    expected = N_CASES * SETTINGS.k * len(ARM_ORDER)
    if total != expected:
        problems.append(f"the seal accounts for {total} rows, not {expected}")
    body = {k: v for k, v in seal.items() if k != "seal_digest"}
    if seal.get("seal_digest") != digest_obj(body):
        problems.append("seal_digest does not cover the seal")
    return problems


#: What a sealed run directory may hold besides its samples: the four
#: run documents, and the four a published report adds. Named once, so the
#: stray check, the manifest check and the report cannot drift apart.
#:
#: The report members are here because ``--report`` writes into the run
#: directory, so a directory that has been reported over would otherwise
#: fail its own re-verification for holding files the seal does not name.
#: They are write-once and are re-derived by ``phase3c_report.verify``, so
#: tolerating them here does not tolerate an unchecked file.
NON_SAMPLE_FILES: tuple[str, ...] = (MANIFEST_NAME, SEAL_NAME, RECEIPT_NAME,
                                     SCORES_NAME) + PUBLISHED_NAMES


def rehash(out_dir: Path, seal: dict) -> tuple[dict, list[str]]:
    """Re-hash every sealed member here, *and the execution manifest*.

    The manifest used to be merely tolerated: it was excluded from the stray
    list and then never looked at, so the one file that says which pack,
    which weights and which gate produced the samples was the one file whose
    bytes nobody checked. The seal records its size and SHA-256; this is
    where those are spent.
    """
    out_dir = Path(out_dir)
    found, problems = {}, []
    for member, expected in sorted((seal.get("members") or {}).items()):
        path = out_dir / member
        if not path.is_file():
            problems.append(f"{member} is sealed and is not here")
            continue
        size, digest = path.stat().st_size, sha256_file(path)
        found[member] = {"size": size, "sha256": digest}
        if size != expected.get("size"):
            problems.append(f"{member} is {size} bytes and the seal says "
                            f"{expected.get('size')}")
        if digest != expected.get("sha256"):
            problems.append(f"{member} hashes to {digest[:16]}... and the "
                            f"seal says {str(expected.get('sha256'))[:16]}...")

    sealed_manifest = seal.get("execution_manifest") or {}
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        problems.append(f"{MANIFEST_NAME} is sealed and is not here")
    elif not sealed_manifest:
        problems.append(f"the seal does not record {MANIFEST_NAME}")
    else:
        size = manifest_path.stat().st_size
        digest = sha256_file(manifest_path)
        found[MANIFEST_NAME] = {"size": size, "sha256": digest}
        if size != sealed_manifest.get("size"):
            problems.append(
                f"{MANIFEST_NAME} is {size} bytes and the seal says "
                f"{sealed_manifest.get('size')}")
        if digest != sealed_manifest.get("sha256"):
            problems.append(
                f"{MANIFEST_NAME} hashes to {digest[:16]}... and the seal "
                f"says {str(sealed_manifest.get('sha256'))[:16]}...")

    strays = sorted(p.relative_to(out_dir).as_posix()
                    for p in out_dir.rglob("*")
                    if p.is_file()
                    and p.relative_to(out_dir).as_posix() not in
                    set(seal.get("members") or {})
                    and p.name not in NON_SAMPLE_FILES)
    if strays:
        problems.append(f"files returned that the seal does not name: "
                        f"{strays}")
    return found, problems


def read_execution_manifest(out_dir) -> dict:
    """The manifest in a run directory, from its bytes, or a refusal.

    The digest it returns under ``manifest_digest`` is the *recomputed* one:
    the stated field is compared to a digest taken over the rest of the
    document, and a disagreement is a refusal rather than a preference for
    one of the two. Callers that bind rows to a manifest bind them to this,
    never to a string an argument carried in.
    """
    path = Path(out_dir) / MANIFEST_NAME
    if not path.is_file():
        raise PlanRefused(
            f"{MANIFEST_NAME} is not in {out_dir}; there is nothing saying "
            "which pack, weights or gate produced these samples")
    try:
        body = json.loads(path.read_text())
    except ValueError as exc:
        raise PlanRefused(f"{MANIFEST_NAME} is not valid JSON ({exc})"
                          ) from exc
    if not isinstance(body, dict):
        raise PlanRefused(f"{MANIFEST_NAME} is a {type(body).__name__}")
    recomputed = digest_obj({k: v for k, v in body.items()
                             if k != "manifest_digest"})
    if body.get("manifest_digest") != recomputed:
        raise PlanRefused(
            f"{MANIFEST_NAME} states manifest_digest "
            f"{str(body.get('manifest_digest'))[:16]}... and its own "
            f"contents digest to {recomputed[:16]}...")
    return body


def row_binding_problems(out_dir, plan: dict, *, manifest_digest: str
                         ) -> list[str]:
    """Every stored row names *this* execution manifest, counted not listed.

    A per-row report of 1,920 identical sentences is unreadable, so this
    counts and names the first. It is deliberately a whole-directory sweep
    rather than a per-step one: a row moved from one member to another still
    has to name the manifest that is here.
    """
    problems: list[str] = []
    for index in range(N_STEPS):
        path = Path(out_dir) / samples_member(index)
        if not path.is_file():
            problems.append(f"{samples_member(index)} is not here, so its "
                            "rows cannot be bound to the execution manifest")
            continue
        try:
            rows = read_rows(path)
        except PlanRefused as exc:
            problems.append(f"{samples_member(index)}: {exc}")
            continue
        wrong = [r for r in rows
                 if not isinstance(r, dict)
                 or r.get("manifest_digest") != manifest_digest]
        if wrong:
            first = wrong[0] if isinstance(wrong[0], dict) else {}
            problems.append(
                f"{samples_member(index)}: {len(wrong)} of {len(rows)} rows "
                f"name an execution manifest other than the one in this "
                f"directory, first "
                f"{(first.get('case_id'), first.get('arm'), first.get('seed'))}"
                f" naming {str(first.get('manifest_digest'))[:16]}...")
    return problems


def manifest_chain_problems(out_dir: Path, seal: dict, *, plan: dict,
                            grant: dict, check_rows: bool = True
                            ) -> list[str]:
    """The execution manifest, read and judged rather than assumed present.

    Five separate things, because they fail separately: the file is there,
    the seal's record of its size and SHA-256 matches the bytes on disk
    (``rehash``), the document parses and its stated digest is the digest of
    its own contents, it holds against the plan *and* the grant, the seal's
    ``manifest_digest`` is that same value, and **every stored sample row
    names it**.

    The last one is new and is the fail-open it closes: ``manifest_digest``
    was a required field on every row and nothing compared it to the
    manifest actually present, so a bundle whose rows pointed at a different
    execution -- or at nothing -- passed the receipt and was scored.
    """
    problems: list[str] = []
    try:
        manifest = read_execution_manifest(out_dir)
    except PlanRefused as exc:
        return [str(exc)]

    problems.extend(manifest_problems(manifest, plan, grant))
    if seal.get("manifest_digest") != manifest.get("manifest_digest"):
        problems.append(
            "the seal covers a different execution manifest than the one in "
            "this directory")
    if check_rows:
        problems.extend(row_binding_problems(
            out_dir, plan, manifest_digest=manifest["manifest_digest"]))
    return problems


#: What a receipt says that is a *judgement about the run*, as opposed to a
#: note about when the judgement happened. Two verifications of an unchanged
#: directory differ only in ``verified_at``, so identity excludes it -- and
#: ``receipt_digest``, which covers the timestamp and therefore moves with it.
#:
#: The gap this closes: ``--verify`` compared stored and fresh receipts by
#: ``receipt_digest``. Run twice across a clock tick, the second run refused
#: its own unchanged conclusion as "a different receipt".
RECEIPT_VOLATILE_FIELDS: tuple[str, ...] = ("verified_at", "receipt_digest")


def receipt_identity(receipt) -> dict:
    """What makes two verifications the same verification."""
    if not isinstance(receipt, dict):
        return {"not_a_receipt": type(receipt).__name__}
    return {k: v for k, v in receipt.items()
            if k not in RECEIPT_VOLATILE_FIELDS}


def expected_receipt(out_dir: Path, seal: dict, carried_seal_digest: str, *,
                     plan: dict, grant: dict) -> tuple[dict, list[str]]:
    """The receipt this directory *must* produce, and why it might not.

    Everything here is rebuilt from the four inputs a verifier is allowed to
    trust -- the plan, the grant, the seal and the bytes on disk -- and from
    nothing the run directory asserts about itself. The returned body is the
    canonical expected record: :func:`receipt_problems` compares a stored
    receipt to it field by field rather than re-deriving a chosen subset,
    which is what let a receipt keep a field nobody looked at.

    ``members`` is the *actual* rehash, so a stored receipt whose member
    table was edited fails on that field even if every digest it carries was
    recomputed to agree with the edit.
    """
    problems = list(seal_problems(seal, plan))
    if seal.get("seal_digest") != carried_seal_digest:
        problems.append(
            f"the seal in the directory digests to "
            f"{str(seal.get('seal_digest'))[:16]}... and the value carried "
            f"here separately was {str(carried_seal_digest)[:16]}...")
    problems.extend(authorization_problems(grant, plan,
                                           check_sources=False))
    problems.extend(manifest_chain_problems(out_dir, seal, plan=plan,
                                            grant=grant))
    found, rehash_problems = rehash(out_dir, seal)
    problems.extend(rehash_problems)
    body = {
        "kind": RECEIPT_KIND,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan["plan_digest"],
        "authorization_digest": grant.get("authorization_digest"),
        "pack_digest": grant.get("pack_digest"),
        "pack_evidence_digest": grant.get("pack_evidence_digest"),
        "dependency_digest": grant.get("dependency_digest"),
        "adapter_sha256": dict(grant.get("adapter_sha256") or {}),
        "gate_sources": dict(grant.get("gate_sources") or {}),
        "generation_source_manifest_digest":
            grant.get("generation_source_manifest_digest"),
        "seal_digest": seal.get("seal_digest"),
        "carried_seal_digest": carried_seal_digest,
        "manifest_digest": seal.get("manifest_digest"),
        "members": found,
        "problems": problems,
        "verified": not problems,
        "verified_on": "Mac",
    }
    return body, problems


def build_receipt(out_dir: Path, seal: dict, carried_seal_digest: str, *,
                  plan: dict, grant: dict, verified_at: str) -> dict:
    """The Mac's independent statement about what came back.

    Independent in two ways that matter. The digest it checks the seal
    against is :paramref:`carried_seal_digest`, which reached this machine by
    a route other than the directory it authenticates. And the grant it
    checks the manifest against is :paramref:`grant`, which was written
    before the run started and lives outside the run directory -- so a
    bundle whose every internal digest agrees is still refused unless it is
    the bundle somebody authorised.
    """
    body, _problems = expected_receipt(out_dir, seal, carried_seal_digest,
                                       plan=plan, grant=grant)
    body["verified_at"] = verified_at
    body["receipt_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "receipt_digest"})
    return body


def receipt_problems(receipt, out_dir: Path, *, plan: dict, seal: dict,
                     grant: dict) -> list[str]:
    """Whether a stored receipt is *the* receipt this run must produce.

    Not a re-check of a chosen subset any more. The expected record is
    rebuilt in full from the plan, the grant, the seal, the execution
    manifest and the rehashed bytes, and the stored receipt is compared to
    it field by field. A field nobody thought to check is therefore not a
    field an attacker can move: adding one to the record adds it to the
    comparison automatically, and a stored receipt carrying a field the
    expected record does not have is itself a problem.

    The stored receipt must also be *clean*: ``problems`` empty and
    ``verified`` true. Those two used to be able to disagree with each other
    -- a stored ``verified: true`` beside a non-empty list passed, because
    only the flag was read.

    ``verified_at`` is excluded, because when the Mac looked is not a claim
    about the run; see :data:`RECEIPT_VOLATILE_FIELDS`.
    """
    if not isinstance(receipt, dict):
        return [f"the receipt is a {type(receipt).__name__}"]

    problems: list[str] = []
    carried = receipt.get("carried_seal_digest")
    if not _is_digest(carried):
        problems.append(
            f"the receipt's carried_seal_digest is {str(carried)[:24]!r}, "
            "which is not a SHA-256")
        carried = ""
    want, chain_problems = expected_receipt(
        out_dir, seal, carried, plan=plan, grant=grant)
    problems.extend(chain_problems)

    stored = receipt_identity(receipt)
    extra = sorted(set(stored) - set(want))
    missing = sorted(set(want) - set(stored))
    if extra:
        problems.append(f"the receipt carries {extra}, which a receipt for "
                        "this run does not have")
    if missing:
        problems.append(f"the receipt is missing {missing}")
    for field in sorted(set(want) & set(stored)):
        if stored[field] != want[field]:
            problems.append(
                f"the receipt's {field} is not what this run derives")

    if receipt.get("problems"):
        problems.append(
            f"the receipt is not clean: {receipt.get('problems')}")
    if receipt.get("verified") is not True:
        problems.append("the receipt does not claim to be verified")
    if not _is_digest(receipt.get("receipt_digest")):
        problems.append("receipt_digest is not a SHA-256")
    else:
        body = {k: v for k, v in receipt.items() if k != "receipt_digest"}
        if receipt["receipt_digest"] != digest_obj(body):
            problems.append("receipt_digest does not cover the receipt")
    return problems


# ---------------------------------------------------------------------------
# Writing a generation
# ---------------------------------------------------------------------------

def write_plan_set(out_dir, plan: dict, membership: dict, audit: dict,
                   *, root=None) -> dict:
    """Publish a whole generation, or publish none of it.

    Write-once, and *all four or nothing*: a plan on disk beside a missing
    audit is a case list whose independence nobody can check, and the failure
    mode of writing them one at a time is that the first three survive a
    crash in the fourth and look like a complete set.

    The contract snapshot travels with them. The contract is reconstructible
    from this module, but only from *this* version of it, and the whole point
    of an archive is to be readable when the module has moved on.
    """
    from src.training.session import write_once_json

    out_dir = Path(out_dir)
    paths = {
        "plan": out_dir / Path(PLAN_PATH).name,
        "membership": out_dir / Path(MEMBERSHIP_PATH).name,
        "audit": out_dir / Path(AUDIT_PATH).name,
        "contract": out_dir / Path(CONTRACT_SNAPSHOT_PATH).name,
        # Beside the archives rather than inside one, so recording that an
        # older generation is dead never touches a byte that generation
        # wrote. Written here rather than by the CLI because it is derived
        # entirely from this module: a generation without it is a generation
        # nothing says may run.
        "supersession": out_dir.parent / Path(SUPERSESSION_PATH).name,
    }
    # The supersession record is shared across generations in the same root
    # and is identical for a given generation, so an existing one that says
    # the same thing is accepted rather than treated as a collision.
    if paths["supersession"].exists():
        stored = json.loads(paths["supersession"].read_text())
        if stored != build_supersession_record():
            raise PlanRefused(
                f"{paths['supersession']} already records a different "
                "supersession for this generation")
        paths.pop("supersession")
    existing = sorted(k for k, p in paths.items() if p.exists())
    if existing:
        raise PlanRefused(
            f"{existing} already exist in {out_dir}. A generation is written "
            "once; a second materialisation would replace the case list a "
            "result was produced against. Start a new generation instead.")
    bodies = {
        "plan": plan, "membership": membership, "audit": audit,
        "contract": build_contract_snapshot(plan, root=root),
        "supersession": build_supersession_record(),
    }
    written: dict = {}
    try:
        # The plan is written last and is what every later stage names, so a
        # half-written generation never has the file that makes it usable.
        for role in ("supersession", "contract", "audit", "membership",
                     "plan"):
            if role not in paths:
                continue
            write_once_json(paths[role], bodies[role])
            written[role] = paths[role]
    except Exception:
        for path in written.values():
            path.unlink(missing_ok=True)
        raise
    return written


#: The two archived documents the plan names by digest, as
#: ``filename -> (the document's own digest field, the plan's field)``.
#: Each carries a self-digest over everything but that field, so there are
#: two things to check and they fail separately: whether the digest covers
#: the document, and whether it is the one the plan was built against.
ARCHIVE_MEMBERS: dict = {
    "case_membership.json": ("membership_digest", "case_membership_digest"),
    "isolation_audit.json": ("audit_digest", "audit_digest"),
}


def archive_problems(plan: dict, archive_dir=None, *, root=None,
                     require_grant: bool = True) -> list[str]:
    """Whether the generation on disk is the whole generation this plan names.

    The closure is all six documents, not the three the plan happens to
    name by digest. A report could be published over an archive with no
    grant, no pack evidence and no supersession record, because nothing
    asked for them -- and the grant is the only thing binding the pack, the
    evidence is the only thing making its digest re-derivable, and the
    supersession record is the only thing saying this generation is the one
    that may run.

    ``require_grant`` is False only where the caller is checking a plan set
    that has been materialised and not yet authorised, which is a real state
    between ``--materialize`` and ``--authorize`` and nowhere else.
    """
    root = Path(root or ROOT)
    directory = Path(archive_dir) if archive_dir else root / ARCHIVE_DIR
    problems: list[str] = []
    if not directory.is_dir():
        return [f"{directory} is not a generation archive"]

    plan_path = directory / Path(PLAN_PATH).name
    if not plan_path.is_file():
        problems.append(f"{plan_path.name} is not in {directory}")
    else:
        stored = json.loads(plan_path.read_text())
        if stored.get("plan_digest") != plan.get("plan_digest"):
            problems.append(
                f"{plan_path.name} in the archive digests to "
                f"{str(stored.get('plan_digest'))[:16]}... and this plan is "
                f"{str(plan.get('plan_digest'))[:16]}...")

    for name, (own, field) in sorted(ARCHIVE_MEMBERS.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is not in {directory}; the plan names "
                            f"its {field} and nothing here can produce it")
            continue
        try:
            body = json.loads(path.read_text())
        except ValueError as exc:
            problems.append(f"{name} is not valid JSON ({exc})")
            continue
        recomputed = digest_obj({k: v for k, v in body.items() if k != own})
        if body.get(own) != recomputed:
            problems.append(
                f"{name} states {own} {str(body.get(own))[:16]}... and its "
                f"contents digest to {recomputed[:16]}...")
        elif recomputed != plan.get(field):
            problems.append(
                f"{name} digests to {recomputed[:16]}... and the plan's "
                f"{field} is {str(plan.get(field))[:16]}...")

    snapshot_path = directory / Path(CONTRACT_SNAPSHOT_PATH).name
    if not snapshot_path.is_file():
        problems.append(f"{snapshot_path.name} is not in {directory}")
    else:
        try:
            snapshot = json.loads(snapshot_path.read_text())
        except ValueError as exc:
            problems.append(f"{snapshot_path.name} is not valid JSON ({exc})")
        else:
            problems.extend(f"{snapshot_path.name}: {p}" for p in
                            contract_snapshot_problems(snapshot, plan))

    supersession = directory.parent / Path(SUPERSESSION_PATH).name
    if not supersession.is_file():
        problems.append(
            f"{supersession.name} is not beside {directory}; nothing says "
            "this generation is the one that may run")
    else:
        try:
            record = json.loads(supersession.read_text())
        except ValueError as exc:
            problems.append(f"{supersession.name} is not valid JSON ({exc})")
        else:
            problems.extend(f"{supersession.name}: {p}" for p in
                            supersession_problems(record, root=root))

    if not require_grant:
        return problems

    grant_path = directory / Path(AUTHORIZATION_PATH).name
    evidence_path = directory / Path(PACK_EVIDENCE_PATH).name
    if not grant_path.is_file():
        problems.append(f"{grant_path.name} is not in {directory}; no run "
                        "may be authorised by a document that is not there")
        return problems
    grant = json.loads(grant_path.read_text())
    problems.extend(f"{grant_path.name}: {p}" for p in
                    authorization_problems(grant, plan, root=root,
                                           check_sources=False))
    if not evidence_path.is_file():
        problems.append(
            f"{evidence_path.name} is not in {directory}; the pack digest "
            "the grant names would have no path back to any bytes")
        return problems
    try:
        evidence = json.loads(evidence_path.read_text())
    except ValueError as exc:
        problems.append(f"{evidence_path.name} is not valid JSON ({exc})")
        return problems
    problems.extend(f"{evidence_path.name}: {p}" for p in
                    pack_evidence_problems(evidence,
                                           pack_digest=grant.get(
                                               "pack_digest")))
    if evidence.get("evidence_digest") != grant.get("pack_evidence_digest"):
        problems.append(
            f"the grant names pack evidence "
            f"{str(grant.get('pack_evidence_digest'))[:16]}... and the "
            f"archive holds {str(evidence.get('evidence_digest'))[:16]}...")
    return problems


def supersession_problems(record, *, root=None) -> list[str]:
    """Whether a supersession record is this generation's, and holds.

    An entry that names an attempted execution is checked by **reading that
    attempt**, not by comparing path strings. A supersession record could
    otherwise say "gen04 attempted this and produced nothing" while the
    directory it names held a sealed run, or no longer existed at all, and
    every digest in the record would still agree with itself.
    """
    problems: list[str] = []
    if not isinstance(record, dict):
        return [f"the supersession record is a {type(record).__name__}"]
    if record.get("kind") != "brickagain.phase3c_supersession":
        problems.append(f"kind is {record.get('kind')!r}")
    if record.get("current") != GENERATION:
        problems.append(
            f"it names {record.get('current')!r} as current and this is "
            f"{GENERATION!r}")
    named = record.get("not_executable")
    want = [dict(entry) for entry in SUPERSEDED]
    if named != want:
        problems.append(
            "its not-executable entries are not what this module records")
    body = {k: v for k, v in record.items() if k != "supersession_digest"}
    if record.get("supersession_digest") != digest_obj(body):
        problems.append("supersession_digest does not cover the record")
    problems += attempted_execution_problems(record, root=root)
    return problems


def attempted_execution_problems(record, *, root=None) -> list[str]:
    """Read every attempted execution a supersession record names.

    The binding is the index's own SHA-256, recorded in the entry. So the
    entry names a directory, the directory's index has to be *that* file,
    and :func:`failed_attempt_problems` then has to hold over it. Renaming a
    directory, swapping its index, or pointing the entry at a different
    attempt each fail here rather than passing as a matching string.
    """
    base = Path(root or ROOT)
    problems: list[str] = []
    for entry in (record or {}).get("not_executable") or []:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("attempted_execution")
        want = entry.get("attempted_execution_index_sha256")
        generation = entry.get("generation")
        if rel is None:
            if want is not None:
                problems.append(
                    f"{generation} records an index digest and names no "
                    "attempted execution to find it in")
            continue
        if not isinstance(want, str) or len(want) != 64:
            problems.append(
                f"{generation} names an attempted execution at {rel} and no "
                "SHA-256 for its index; a path is not a binding")
            continue
        directory = base / rel
        index_path = directory / "index.json"
        if not index_path.is_file():
            problems.append(
                f"{generation} names an attempted execution at {rel} and "
                "there is no index.json there")
            continue
        actual = sha256_file(index_path)
        if actual != want:
            problems.append(
                f"{rel}/index.json digests to {actual[:16]}... and "
                f"{generation}'s entry binds {want[:16]}...")
            continue
        for problem in failed_attempt_problems(directory, root=base):
            problems.append(f"{rel}: {problem}")

        # Against what the attempt's own record says, not against zero.
        # Requiring zero here was correct while gen04 was the only attempt
        # and wrong the moment gen08 reached a node, wrote step 0's 320
        # cells and stopped: a supersession entry has to state what actually
        # happened, and this is where that statement is held to the
        # evidence. ``failed_attempt_problems`` has just checked that number
        # against the bytes on disk, the read-only listing and the frozen
        # plan, so agreeing with it is agreeing with all three.
        try:
            attempted = json.loads(index_path.read_text(encoding="utf-8"))
        except ValueError:
            problems.append(f"{rel}/index.json is not valid JSON")
            continue
        produced = failed_attempt_cells(attempted)
        if produced is None:
            problems.append(
                f"{rel}/index.json states no cell count, so {generation}'s "
                f"cells_produced={entry.get('cells_produced')!r} is bound to "
                "nothing")
        elif entry.get("cells_produced") != produced:
            problems.append(
                f"{generation} records {entry.get('cells_produced')!r} cells "
                f"produced and the attempt it names recorded {produced}")

        # A seal, unlike a cell count, is zero for every failed attempt: a
        # seal over a partial member set would certify a run the plan does
        # not describe.
        if entry.get("sealed") is not False:
            problems.append(f"{generation} records a seal for an attempt "
                            "that was not sealed")
    return problems


def _listing_blocks(text: str) -> list[dict]:
    """The observation listing, split into the commands that produced it.

    Parsed rather than pattern-matched over the whole file so a claim can be
    tied to the command that produced it: "the samples directory held zero
    entries" means nothing without knowing which command was asked.
    """
    blocks: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        if line.startswith("### COMMAND: "):
            current = {"command": line[len("### COMMAND: "):],
                       "exit_code": None, "body": []}
            blocks.append(current)
        elif current is None:
            continue
        elif line.startswith("### EXIT_CODE: "):
            try:
                current["exit_code"] = int(line[len("### EXIT_CODE: "):])
            except ValueError:
                current["exit_code"] = None
        elif line.startswith("### "):
            continue
        else:
            current["body"].append(line)
    return blocks


def _member_states_from_listing(text: str) -> dict[str, str]:
    """``{member stem: state}`` as the node reported it, member by member."""
    out: dict[str, str] = {}
    for block in _listing_blocks(text):
        for line in block["body"]:
            parts = line.split()
            if len(parts) == 3 and parts[0].startswith("step_") \
                    and parts[1] in ("ABSENT", "ZERO_BYTE", "POPULATED"):
                out[parts[0]] = parts[1].lower()
    return out


def failed_attempt_problems(directory, *, root=None) -> list[str]:
    """Everything wrong with a failed-attempt record, in one list.

    ``root`` is the tree the attempt's frozen generation is read from, so a
    test can point the whole check at a temporary copy without touching a
    frozen byte. The checks, in order:

    * the index is this kind, this schema, ``outcome: failed`` and not
      citable;
    * its own digest covers everything else in it;
    * every file it names is here at exactly the size and SHA-256 it names,
      and no file is here that it does not name;
    * the execution manifest's own ``manifest_digest`` recomputes, and its
      plan, authorization, pack, dependency and source-closure identity are
      the ones the index declares -- and, where the generation is still in
      the tree, the ones that generation froze;
    * the listing found all six sample members **absent**, not zero-byte and
      not populated, and the index says the same;
    * no cell, no seal, no receipt, no scores and no report, in the index,
      in the listing and on disk.
    """
    base = Path(root or ROOT)
    directory = Path(directory)
    index_path = directory / "index.json"
    if not index_path.is_file():
        return [f"{directory} has no index.json; an attempt that records "
                "nothing is not evidence of anything"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"index.json is not valid JSON ({exc})"]
    if not isinstance(index, dict):
        return [f"index.json is a {type(index).__name__}, not an object"]

    problems: list[str] = []
    if index.get("kind") != FAILED_ATTEMPT_KIND:
        problems.append(f"kind is {index.get('kind')!r}, not "
                        f"{FAILED_ATTEMPT_KIND!r}")
    version = index.get("schema_version")
    if version not in FAILED_ATTEMPT_SCHEMA_VERSIONS:
        problems.append(f"schema_version is {version!r}, not one of "
                        f"{list(FAILED_ATTEMPT_SCHEMA_VERSIONS)}")
    if index.get("outcome") != "failed":
        problems.append(f"outcome is {index.get('outcome')!r}, not 'failed'; "
                        "this directory is only ever a failure")
    if index.get("citable_as_a_result") is not False:
        problems.append(
            "citable_as_a_result is not False. A failed attempt that says it "
            "may be cited is the one sentence this file exists to prevent.")

    # Its own digest, over everything else. An index edited to agree with
    # itself still has to agree with this.
    recorded = index.get("index_digest")
    body = {k: v for k, v in index.items() if k != "index_digest"}
    recomputed = hashlib.sha256(
        canonical_json(body).encode("utf-8")).hexdigest()
    if recorded != recomputed:
        problems.append(
            f"index_digest is {str(recorded)[:16]}... and the record digests "
            f"to {recomputed[:16]}...")

    # Every file it names, and no file it does not.
    table = index.get("evidence_files")
    if not isinstance(table, list):
        problems.append("evidence_files is not a list")
        table = []
    named = set()
    for entry in table:
        if not isinstance(entry, dict) or "path" not in entry:
            problems.append(f"an evidence entry is not a record: {entry!r}")
            continue
        rel = entry["path"]
        named.add(rel)
        path = directory / rel
        if not path.is_file():
            problems.append(f"{rel} is named by the index and is not here")
            continue
        blob = path.read_bytes()
        if len(blob) != entry.get("bytes"):
            problems.append(f"{rel} is {len(blob)} bytes and the index says "
                            f"{entry.get('bytes')!r}")
        actual = hashlib.sha256(blob).hexdigest()
        if actual != entry.get("sha256"):
            problems.append(f"{rel} digests to {actual[:16]}... and the "
                            f"index says {str(entry.get('sha256'))[:16]}...")
    on_disk = {str(p.relative_to(directory)) for p in directory.rglob("*")
               if p.is_file() and p.name != "index.json"}
    for rel in sorted(on_disk - named):
        problems.append(f"{rel} is here and the index does not name it")

    problems += _failed_attempt_manifest_problems(directory, index, base)

    # One outcome block, matching the schema the record declares. A record
    # carrying both would be checked by whichever branch ran and could state
    # a different outcome in the other.
    has_zero, has_partial = ZERO_CELL_PROOF in index, PARTIAL_PROOF in index
    if has_zero and has_partial:
        problems.append(
            f"the record carries both {ZERO_CELL_PROOF} and {PARTIAL_PROOF}; "
            "an attempt has one outcome")
    elif version == 2:
        if has_partial:
            problems.append(f"schema 2 records {ZERO_CELL_PROOF} and this "
                            f"one carries {PARTIAL_PROOF}")
        problems += _failed_attempt_emptiness_problems(directory, index)
    elif version == 3:
        if has_zero:
            problems.append(f"schema 3 records {PARTIAL_PROOF} and this one "
                            f"carries {ZERO_CELL_PROOF}")
        problems += _failed_attempt_partial_problems(directory, index, base)
    return problems


def failed_attempt_cells(index) -> int | None:
    """How many cells the record says the attempt produced, or ``None``.

    Read from whichever outcome block the schema names, so a caller binding
    a supersession entry to an attempt compares against the number this
    record actually claims rather than against a constant.
    """
    if not isinstance(index, dict):
        return None
    for block_name in (ZERO_CELL_PROOF, PARTIAL_PROOF):
        block = index.get(block_name)
        if isinstance(block, dict) and "total_cells_written" in block:
            written = block["total_cells_written"]
            if isinstance(written, int) and not isinstance(written, bool):
                return written
            return None
    return None


def _failed_attempt_manifest_problems(directory: Path, index: dict,
                                      base: Path) -> list[str]:
    """The execution manifest, checked against itself and its generation."""
    problems: list[str] = []
    declared = index.get("digests_this_attempt_ran_under")
    if not isinstance(declared, dict):
        return ["digests_this_attempt_ran_under is not a record, so the "
                "attempt does not say what it ran under"]

    path = directory / "node" / "execution_manifest.json"
    if not path.is_file():
        # An attempt can legitimately have none -- it may have stopped before
        # the manifest was written -- but then it must not claim one.
        if declared.get("execution_manifest_digest") is not None:
            problems.append(
                "the index names an execution_manifest_digest and no "
                "execution manifest is here")
        return problems
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"the execution manifest is not valid JSON ({exc})"]

    recomputed = digest_obj({k: v for k, v in manifest.items()
                             if k != "manifest_digest"})
    if manifest.get("manifest_digest") != recomputed:
        problems.append(
            f"the execution manifest carries manifest_digest "
            f"{str(manifest.get('manifest_digest'))[:16]}... and digests to "
            f"{recomputed[:16]}...")
    if declared.get("execution_manifest_digest") != \
            manifest.get("manifest_digest"):
        problems.append(
            "the index's execution_manifest_digest is not the digest in the "
            "manifest beside it")

    fields = ("plan_digest", "contract_digest", "settings_digest",
              "case_membership_digest", "audit_digest",
              "authorization_digest", "pack_digest", "dependency_digest",
              "generation_source_manifest_digest", "adapter_sha256",
              "gate_sources")
    for field in fields:
        if declared.get(field) != manifest.get(field):
            problems.append(
                f"the index says {field} was {str(declared.get(field))[:16]}"
                f"... and the execution manifest says "
                f"{str(manifest.get(field))[:16]}...")

    # And against the generation it ran under, where that is still here.
    generation = index.get("generation")
    archive = base / ARCHIVE_ROOT / str(generation)
    plan_path = archive / "plan.json"
    grant_path = archive / "execution_authorization.json"
    evidence_path = archive / "pack_manifest.json"
    if plan_path.is_file():
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        if stored.get("plan_digest") != manifest.get("plan_digest"):
            problems.append(
                f"{generation}'s archived plan digests to "
                f"{str(stored.get('plan_digest'))[:16]}... and the attempt "
                "ran under another")
        if stored.get("contract_digest") != manifest.get("contract_digest"):
            problems.append(f"{generation}'s archived contract digest is not "
                            "the one the attempt ran under")
    if grant_path.is_file():
        grant = json.loads(grant_path.read_text(encoding="utf-8"))
        for field in ("authorization_digest", "pack_digest",
                      "dependency_digest",
                      "generation_source_manifest_digest"):
            if grant.get(field) != manifest.get(field):
                problems.append(
                    f"{generation}'s archived grant says {field} "
                    f"{str(grant.get(field))[:16]}... and the attempt ran "
                    f"under {str(manifest.get(field))[:16]}...")
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if declared.get("pack_evidence_digest") != \
                evidence.get("evidence_digest"):
            problems.append(
                f"the index's pack_evidence_digest is not {generation}'s "
                "archived pack evidence digest")
    return problems


def _failed_attempt_emptiness_problems(directory: Path,
                                       index: dict) -> list[str]:
    """Zero cells, six absent members, and no sealed anything.

    The index's claim and the listing's observation are checked against each
    other, because either alone is a sentence somebody wrote.
    """
    problems: list[str] = []
    proof = index.get("proof_no_cell_was_written")
    if not isinstance(proof, dict):
        return ["proof_no_cell_was_written is not a record"]

    if proof.get("total_cells_written") != 0:
        problems.append(
            f"the index says {proof.get('total_cells_written')!r} cells were "
            "written; a failed attempt kept here produced none")
    if proof.get("entries_in_samples_dir") != 0:
        problems.append(
            f"the index says the samples directory held "
            f"{proof.get('entries_in_samples_dir')!r} entries")
    if proof.get("observed_state") != "absent":
        problems.append(
            f"the index's observed_state is {proof.get('observed_state')!r}; "
            "the members are absent, which is not the same as empty")

    expected = proof.get("expected_members")
    if not isinstance(expected, list) or len(expected) != N_STEPS:
        problems.append(f"expected_members is not {N_STEPS} entries")
        expected = []
    claimed: dict[str, str] = {}
    for entry in expected:
        if not isinstance(entry, dict):
            problems.append("an expected_members entry is not a record")
            continue
        member = str(entry.get("member", ""))
        state = entry.get("state")
        if state not in MEMBER_STATES:
            problems.append(f"{member} has state {state!r}, which is not one "
                            f"of {list(MEMBER_STATES)}")
        elif state != "absent":
            problems.append(
                f"{member} is recorded as {state!r}. A zero-byte or populated "
                "member is not the zero-cell result this directory claims.")
        if entry.get("bytes") is not None or entry.get("sha256") is not None:
            problems.append(f"{member} is absent and carries a size or a "
                            "digest, which an absent file does not have")
        stem = Path(member).stem
        claimed[stem] = state if state in MEMBER_STATES else "unknown"

    # The listing, read from the file rather than from the index's summary.
    observation = index.get("node", {}).get("observation") or {}
    listing_rel = observation.get("listing")
    if not listing_rel:
        problems.append("the index records no read-only listing of the node "
                        "directory, so its claims rest on nothing observed")
        return problems
    listing_path = directory / listing_rel
    if not listing_path.is_file():
        problems.append(f"{listing_rel} is named as the listing and is not "
                        "here")
        return problems
    text = listing_path.read_text(encoding="utf-8", errors="replace")
    observed = _member_states_from_listing(text)
    if len(observed) != N_STEPS:
        problems.append(
            f"the listing probes {len(observed)} sample member(s), not "
            f"{N_STEPS}; it does not answer the question it is here for")
    for stem, state in sorted(observed.items()):
        if state != "absent":
            problems.append(
                f"the listing found {stem} {state}, not absent")
        if stem in claimed and claimed[stem] != state:
            problems.append(
                f"the index calls {stem} {claimed[stem]!r} and the listing "
                f"found it {state!r}")
    for stem in sorted(set(claimed) - set(observed)):
        problems.append(f"the index names {stem} and the listing never "
                        "probed it")

    for block in _listing_blocks(text):
        if block.get("exit_code") not in (0, None):
            problems.append(
                f"an observation command exited {block['exit_code']}: "
                f"{block['command'][:60]}")
        joined = " ".join(block["body"]).strip()
        if "seal.json" in block["command"] and joined not in ("0", ""):
            problems.append(
                f"the listing found {joined} seal/receipt/scores/report "
                "match(es) in a directory that must hold none")

    created = index.get("not_created_for_this_attempt")
    if not isinstance(created, dict):
        problems.append("not_created_for_this_attempt is not a record")
    else:
        for name in ("seal", "receipt", "scores", "report"):
            if created.get(name) is not False:
                problems.append(
                    f"the index says a {name} exists for this attempt; a "
                    "failed attempt has none, and one built over zero cells "
                    "would certify nothing")
    for name in ("seal.json", "receipt.json", "scores.json"):
        for found in directory.rglob(name):
            problems.append(f"{found.relative_to(directory)} is in a failed "
                            "attempt's evidence and must not exist")
    return problems


def _partial_member_shape_problems(entry: dict, directory: Path) -> list[str]:
    """One member entry, against its own declared state and the bytes here.

    ``absent`` carries nothing, because an absent file has no size, no
    digest and no copy to keep. ``zero_byte`` and ``populated`` both carry a
    file, and the distinction between them is the one the whole record rests
    on: a zero-byte member is something a process created and left empty,
    and inventing one to stand for an absent member would destroy exactly
    the evidence it pretended to supply.
    """
    member = str(entry.get("member", ""))
    state = entry.get("state")
    rel = entry.get("evidence_path")
    size, digest, cells = (entry.get("bytes"), entry.get("sha256"),
                           entry.get("cells"))
    problems: list[str] = []

    if state == "absent":
        for field, value in (("bytes", size), ("sha256", digest),
                             ("cells", cells), ("evidence_path", rel)):
            if value is not None:
                problems.append(
                    f"{member} is absent and carries {field}={value!r}; an "
                    "absent file has none of these")
        return problems

    if not rel:
        problems.append(f"{member} is {state!r} and names no copy of itself "
                        "in this evidence; the state rests on nothing")
        return problems
    path = directory / rel
    if not path.is_file():
        problems.append(f"{member} names its copy at {rel} and it is not "
                        "here")
        return problems

    blob = path.read_bytes()
    if size != len(blob):
        problems.append(f"{member} says {size!r} bytes and {rel} is "
                        f"{len(blob)}")
    actual = hashlib.sha256(blob).hexdigest()
    if digest != actual:
        problems.append(f"{member} says sha256 {str(digest)[:16]}... and "
                        f"{rel} digests to {actual[:16]}...")

    if state == "zero_byte":
        if blob:
            problems.append(f"{member} is recorded zero_byte and {rel} holds "
                            f"{len(blob)} bytes")
        if cells != 0:
            problems.append(f"{member} is zero_byte and records {cells!r} "
                            "cells")
    elif state == "populated":
        if not blob:
            problems.append(f"{member} is recorded populated and {rel} is "
                            "empty; that is zero_byte, which is a different "
                            "claim")
        if not isinstance(cells, int) or isinstance(cells, bool) or cells < 1:
            problems.append(f"{member} is populated and records {cells!r} "
                            "cells")
    return problems


def _partial_member_cell_problems(entry: dict, directory: Path,
                                  plan: dict | None,
                                  manifest_digest: str | None) -> list[str]:
    """The rows of a populated member, against the plan that predetermined
    them.

    This is the check that makes a partial attempt evidence rather than an
    assertion. The member is read from bytes and every row is judged by the
    same :func:`step_problems` a real run is judged by: the cells must be
    exactly the ones this step was to produce, each bound to this attempt's
    execution manifest -- so a duplicated, missing, extra, altered or
    corrupt row is a refusal, and so is a member of the right length made of
    the wrong cells.
    """
    member = str(entry.get("member", ""))
    index = entry.get("step_index")
    if not isinstance(index, int) or isinstance(index, bool) \
            or not 0 <= index < N_STEPS:
        return [f"{member} records step_index {index!r}"]
    if member != samples_member(index):
        return [f"{member} is recorded at step {index}, which is "
                f"{samples_member(index)}"]
    if plan is None:
        return [f"{member} is populated and its generation's plan is not in "
                "this tree, so its rows cannot be checked against the cells "
                "they were meant to be"]
    if manifest_digest is None:
        return [f"{member} is populated and no execution manifest digest was "
                "read, so its rows are bound to nothing"]

    path = directory / str(entry.get("evidence_path") or "")
    try:
        rows = read_rows(path)
    except (PlanRefused, ValueError, OSError) as exc:
        return [f"{member} could not be read as rows ({type(exc).__name__}: "
                f"{exc})"]

    problems = [f"{member}: {p}" for p in step_problems(
        rows, plan, index, manifest_digest=manifest_digest)]
    if entry.get("cells") != len(rows):
        problems.append(f"{member} records {entry.get('cells')!r} cells and "
                        f"holds {len(rows)} rows")
    return problems


def _failed_attempt_partial_problems(directory: Path, index: dict,
                                     base: Path) -> list[str]:
    """An attempt that produced some cells and then stopped.

    gen08 is the case: step 0 wrote its 320 cells and exited 0, step 1
    refused before its first cell, and steps 2 to 5 never started. The
    record has to survive being read by somebody who assumes it is lying, so
    the index's claim, the read-only listing, the bytes on disk and the
    frozen plan are each checked against the others.
    """
    problems: list[str] = []
    proof = index.get(PARTIAL_PROOF)
    if not isinstance(proof, dict):
        return [f"{PARTIAL_PROOF} is not a record"]

    # The plan and the manifest this attempt ran under, both read here
    # rather than taken from the index's own summary of them.
    generation = index.get("generation")
    plan_path = base / ARCHIVE_ROOT / str(generation) / "plan.json"
    plan = None
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except ValueError:
            problems.append(f"{generation}'s archived plan is not valid JSON")
    manifest_path = directory / "node" / "execution_manifest.json"
    manifest_digest = None
    if manifest_path.is_file():
        try:
            manifest_digest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("manifest_digest")
        except ValueError:
            problems.append("the execution manifest is not valid JSON")

    members = proof.get("members")
    if not isinstance(members, list) or len(members) != N_STEPS:
        return problems + [f"members is not {N_STEPS} entries"]

    claimed: dict[str, str] = {}
    counted = 0
    populated = 0
    present = 0
    for position, entry in enumerate(members):
        if not isinstance(entry, dict):
            problems.append("a members entry is not a record")
            continue
        member = str(entry.get("member", ""))
        # In schedule order, so a record cannot quietly omit one step by
        # listing another twice.
        if member != samples_member(position):
            problems.append(f"members[{position}] is {member!r} and the "
                            f"schedule has {samples_member(position)}")
        state = entry.get("state")
        if state not in MEMBER_STATES:
            problems.append(f"{member} has state {state!r}, which is not one "
                            f"of {list(MEMBER_STATES)}")
            continue
        claimed[Path(member).stem] = state
        problems += _partial_member_shape_problems(entry, directory)
        if state != "absent":
            present += 1
        if state == "populated":
            populated += 1
            counted += entry.get("cells") if isinstance(
                entry.get("cells"), int) else 0
            problems += _partial_member_cell_problems(
                entry, directory, plan, manifest_digest)

    # Exactly the members the record claims, and no other sample file.
    # Listing a planted file in the evidence table is enough to satisfy
    # "no file is here the index does not name", so without this a
    # fabricated member could sit in the evidence unclaimed -- named as a
    # file, attributed to no step, and checked by nothing.
    kept = {str(p.relative_to(directory))
            for p in (directory / "node" / "samples").rglob("*")
            if p.is_file()}
    claimed_paths = {str(e.get("evidence_path")) for e in members
                     if isinstance(e, dict) and e.get("evidence_path")}
    for rel in sorted(kept - claimed_paths):
        problems.append(f"{rel} is a sample file that no member entry "
                        "accounts for")

    if not populated:
        problems.append(
            f"no member is populated; an attempt that wrote no cell is "
            f"schema 2 and states it in {ZERO_CELL_PROOF}")
    if proof.get("total_cells_written") != counted:
        problems.append(
            f"{PARTIAL_PROOF} says {proof.get('total_cells_written')!r} cells "
            f"were written and its members account for {counted}")
    if proof.get("entries_in_samples_dir") != present:
        problems.append(
            f"the index says the samples directory held "
            f"{proof.get('entries_in_samples_dir')!r} entries and "
            f"{present} member(s) are recorded as present")
    if proof.get("total_cells_planned") != \
            N_CASES * SETTINGS.k * len(ARM_ORDER):
        problems.append(
            f"total_cells_planned is {proof.get('total_cells_planned')!r}")

    problems += _partial_listing_problems(directory, index, claimed)
    problems += _no_published_artefact_problems(directory, index)
    return problems


def _partial_listing_problems(directory: Path, index: dict,
                              claimed: dict[str, str]) -> list[str]:
    """The read-only listing, against what the index says it found."""
    problems: list[str] = []
    observation = index.get("node", {}).get("observation") or {}
    listing_rel = observation.get("listing")
    if not listing_rel:
        problems.append("the index records no read-only listing of the node "
                        "directory, so its claims rest on nothing observed")
        return problems
    listing_path = directory / listing_rel
    if not listing_path.is_file():
        problems.append(f"{listing_rel} is named as the listing and is not "
                        "here")
        return problems
    text = listing_path.read_text(encoding="utf-8", errors="replace")
    observed = _member_states_from_listing(text)
    if len(observed) != N_STEPS:
        problems.append(
            f"the listing probes {len(observed)} sample member(s), not "
            f"{N_STEPS}; it does not answer the question it is here for")
    for stem, state in sorted(observed.items()):
        if stem in claimed and claimed[stem] != state:
            problems.append(
                f"the index calls {stem} {claimed[stem]!r} and the listing "
                f"found it {state!r}")
    for stem in sorted(set(claimed) - set(observed)):
        problems.append(f"the index names {stem} and the listing never "
                        "probed it")
    for block in _listing_blocks(text):
        if block.get("exit_code") not in (0, None):
            problems.append(
                f"an observation command exited {block['exit_code']}: "
                f"{block['command'][:60]}")
        joined = " ".join(block["body"]).strip()
        if "seal.json" in block["command"] and joined not in ("0", ""):
            problems.append(
                f"the listing found {joined} seal/receipt/scores/report "
                "match(es) in a directory that must hold none")
    return problems


def _no_published_artefact_problems(directory: Path, index: dict) -> list[str]:
    """No seal, receipt, scores or report -- claimed, and on disk.

    True of every failed attempt whatever it produced. A seal over a partial
    run would certify a member set the plan does not have, which is the one
    way 320 cells could be mistaken for a result.
    """
    problems: list[str] = []
    created = index.get("not_created_for_this_attempt")
    if not isinstance(created, dict):
        problems.append("not_created_for_this_attempt is not a record")
    else:
        for name in ("seal", "receipt", "scores", "report"):
            if created.get(name) is not False:
                problems.append(
                    f"the index says a {name} exists for this attempt; a "
                    "failed attempt has none, and one built over a partial "
                    "member set would certify a run that did not happen")
    for name in ("seal.json", "receipt.json", "scores.json"):
        for found in directory.rglob(name):
            problems.append(f"{found.relative_to(directory)} is in a failed "
                            "attempt's evidence and must not exist")
    return problems


def build_supersession_record(*, current: str = GENERATION) -> dict:
    """Which generations may not be executed, and why, in one document.

    Written beside the archives rather than inside any of them, and versioned
    by the generation it was written for, so recording that gen02 is dead
    does not require touching a byte gen02 wrote.
    """
    body = {
        "kind": "brickagain.phase3c_supersession",
        "contract_version": CONTRACT_VERSION,
        "current": current,
        "current_archive": ARCHIVE_DIR,
        "current_staged_plan": NODE_PLAN_PATH,
        "not_executable": [dict(entry) for entry in SUPERSEDED],
        "rule": ("Only the current generation may authorise a run. A "
                 "superseded generation's files are kept unmodified as the "
                 "record of what was frozen and why it was replaced; no "
                 "result may cite one."),
    }
    body["supersession_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "supersession_digest"})
    return body


def build_pack_evidence(pack_manifest: dict, *, pack_digest: str) -> dict:
    """The file table the authorised pack digest was taken over.

    ``pack_manifest`` is what ``pack.build`` wrote into the pack. Its
    ``created_at`` is dropped: it is not in ``pack.pack_digest`` and keeping
    a second-precision stamp in an archived file is the thing the public
    audit refuses. What stays is every included path with its own SHA-256
    and byte count, and the two intermediate digests, which is exactly what
    is needed to recompute the value with no pack present.
    """
    from src.training import pack as pack_module

    keep = ("schema_version", "kind", "root_relative", "files",
            "files_digest", "data_requirements", "data_digest", "pack_digest")
    table = {k: pack_manifest[k] for k in keep if k in pack_manifest}
    recomputed = pack_module.pack_digest(table)
    if recomputed != pack_digest or table.get("pack_digest") != pack_digest:
        raise PlanRefused(
            f"this pack manifest recomputes to {recomputed[:16]}... and the "
            f"digest being authorised is {str(pack_digest)[:16]}...; the "
            "evidence must be the table the value was taken over")
    body = {
        "kind": "brickagain.phase3c_pack_evidence",
        "generation": GENERATION,
        "pack_digest": pack_digest,
        "files_digest": table.get("files_digest"),
        "data_digest": table.get("data_digest"),
        "n_files": len(table.get("files") or {}),
        "how_to_recheck": (
            "src.training.pack.pack_digest over {schema_version, kind, "
            "files_digest, data_digest} reproduces pack_digest from this "
            "document alone; src.training.session.manifest_digest over "
            "{'files': files} reproduces files_digest; and every row of "
            "files can be checked against the tree at the commit that froze "
            "this generation."),
        "pack_manifest": table,
    }
    body["evidence_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "evidence_digest"})
    return body


def pack_evidence_problems(evidence, *, pack_digest=None) -> list[str]:
    """Whether archived pack evidence still recomputes the digest it names."""
    from src.training import pack as pack_module

    if not isinstance(evidence, dict):
        return [f"the pack evidence is a {type(evidence).__name__}"]
    problems: list[str] = []
    if evidence.get("kind") != "brickagain.phase3c_pack_evidence":
        problems.append(f"kind is {evidence.get('kind')!r}")
    table = evidence.get("pack_manifest")
    if not isinstance(table, dict):
        return problems + ["the pack manifest table is missing"]

    files = table.get("files")
    if not isinstance(files, dict) or not files:
        problems.append("the pack manifest table names no files")
    else:
        from src.training.session import manifest_digest as _files_digest

        if _files_digest({"files": files}) != table.get("files_digest"):
            problems.append("files_digest does not cover the file table")
        if len(files) != evidence.get("n_files"):
            problems.append(
                f"the table holds {len(files)} files and the evidence says "
                f"{evidence.get('n_files')}")
    recomputed = pack_module.pack_digest(table)
    if recomputed != evidence.get("pack_digest"):
        problems.append(
            f"the table recomputes to {recomputed[:16]}... and this evidence "
            f"names {str(evidence.get('pack_digest'))[:16]}...")
    if pack_digest is not None and evidence.get("pack_digest") != pack_digest:
        problems.append(
            f"this evidence is for pack {str(evidence.get('pack_digest'))[:16]}"
            f"... and the value being checked is {str(pack_digest)[:16]}...")
    body = {k: v for k, v in evidence.items() if k != "evidence_digest"}
    if evidence.get("evidence_digest") != digest_obj(body):
        problems.append("evidence_digest does not cover the evidence")
    return problems


CONTRACT_SNAPSHOT_KIND = "brickagain.phase3c_contract_snapshot"


def build_contract_snapshot(plan: dict, *, root=None) -> dict:
    """The contract as it stood at the freeze, with its own digest.

    The gap this closes: the snapshot carried ``contract_digest`` copied
    from the plan and a ``contract`` payload, and nothing checked that the
    two described each other. A snapshot could name one contract and hold
    another, and the only thing that would have noticed is a reader.
    """
    document = contract_document(root)
    body = {
        "kind": CONTRACT_SNAPSHOT_KIND,
        "contract_version": CONTRACT_VERSION,
        "generation": GENERATION,
        "contract_digest": plan["contract_digest"],
        "payload_digest": digest_obj(document),
        "superseded": [dict(entry) for entry in SUPERSEDED],
        "contract": document,
    }
    body["snapshot_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "snapshot_digest"})
    return body


def contract_snapshot_problems(snapshot, plan: dict) -> list[str]:
    """Whether an archived snapshot is the contract this plan was built on.

    Three separate things: the payload digests to what the snapshot says it
    digests to, that value is the plan's ``contract_digest``, and the
    snapshot's own digest covers the whole document. A snapshot that named
    the right contract while holding a different one used to pass.
    """
    problems: list[str] = []
    if not isinstance(snapshot, dict):
        return [f"the contract snapshot is a {type(snapshot).__name__}"]
    if snapshot.get("kind") != CONTRACT_SNAPSHOT_KIND:
        problems.append(f"kind is {snapshot.get('kind')!r}")
    payload = snapshot.get("contract")
    if not isinstance(payload, dict):
        problems.append("the contract snapshot carries no contract")
    else:
        actual = digest_obj(payload)
        if snapshot.get("payload_digest") != actual:
            problems.append(
                f"the snapshot says its contract digests to "
                f"{str(snapshot.get('payload_digest'))[:16]}... and the "
                f"contract it holds digests to {actual[:16]}...")
        elif actual != plan.get("contract_digest"):
            problems.append(
                f"the archived contract digests to {actual[:16]}... and the "
                f"plan was built on {str(plan.get('contract_digest'))[:16]}"
                "...")
    if snapshot.get("contract_digest") != plan.get("contract_digest"):
        problems.append(
            "the snapshot's contract_digest is not the plan's")
    body = {k: v for k, v in snapshot.items() if k != "snapshot_digest"}
    if snapshot.get("snapshot_digest") != digest_obj(body):
        problems.append("snapshot_digest does not cover the snapshot")
    return problems


def stage_for_node(root=None, *, archive_dir=None) -> dict:
    """Copy the plan out of the archive for the pack, write-once.

    The archive under ``data/`` is the record and stays put; a pack may not
    carry anything from there, so this copy exists and nothing else does.
    ``copy_once`` refuses to overwrite, and :func:`staged_copy_problems`
    proves the copy is still the archive's bytes rather than a second,
    drifting truth.

    The authorization is not copied here. See :data:`NODE_PLAN_PATH`.
    """
    from src.training.session import copy_once

    root = Path(root or ROOT)
    archive = Path(archive_dir) if archive_dir else root / ARCHIVE_DIR
    pairs = ((archive / Path(PLAN_PATH).name, root / NODE_PLAN_PATH),)
    missing = [str(src) for src, _dst in pairs if not src.is_file()]
    if missing:
        raise PlanRefused(
            f"nothing to stage: {missing} are not in the archive. The plan "
            "is materialised and authorised before anything is staged.")
    out: dict = {}
    for src, dst in pairs:
        if dst.exists():
            if sha256_file(dst) != sha256_file(src):
                raise PlanRefused(
                    f"{dst} already exists and is not the archive's bytes. "
                    "A staged copy is never rewritten; start a generation.")
            out[dst.name] = {"path": str(dst), "sha256": sha256_file(dst),
                             "already_present": True}
            continue
        copy_once(src, dst)
        out[dst.name] = {"path": str(dst), "sha256": sha256_file(dst),
                         "already_present": False}
    out[Path(pack_module.PHASE3C_STAGED_POINTER).name] = \
        write_staged_pointer(root)
    return out


def staged_pointer_document(root=None) -> dict:
    """Which staged plan the pack may carry, as a document rather than a
    literal in :mod:`src.training.pack`.

    ``pack.py`` is inside the V1 visual run's import closure, so naming the
    generation there made every Phase 3C bump invalidate V1's frozen runs --
    and re-freezing V1 to repair that invalidated the Phase 3C pack, because
    the pack carries ``src/eval/visual_stress.py``. gen05 was frozen into
    that cycle. The generation lives here and travels as data, so a bump
    rewrites this file and leaves ``pack.py`` untouched.

    The plan's own ``plan_digest`` is deliberately *not* what binds it:
    gen03, gen04 and gen05 share one, because it excludes the scorer
    manifest. The file's SHA-256 is what tells the generations apart, so
    that is what the pointer declares.
    """
    root = Path(root or ROOT)
    plan = root / NODE_PLAN_PATH
    if not plan.is_file():
        raise PlanRefused(
            f"{NODE_PLAN_PATH} has not been staged, so there is nothing to "
            "point at")
    return {
        "kind": pack_module.PHASE3C_POINTER_KIND,
        "generation": GENERATION,
        "staged_plan": NODE_PLAN_PATH,
        "plan_sha256": sha256_file(plan),
        "note": ("The pack carries exactly the plan named here and no other "
                 "staged generation. src.training.pack.staged_phase3c_plan "
                 "refuses unless the generation, the name and the bytes "
                 "agree, and pack.build refuses if any other staged plan "
                 "reaches the include list."),
    }


def write_staged_pointer(root=None) -> dict:
    """Write the pointer for this generation. Rewritten on a bump, by design.

    Not write-once: it is the one mutable thing in the chain, because it is
    the thing that says which generation is current. Everything it names is
    write-once, and it is rewritten only from those files.
    """
    root = Path(root or ROOT)
    body = staged_pointer_document(root)
    path = root / pack_module.PHASE3C_STAGED_POINTER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(body) + "\n", encoding="utf-8")
    resolved, problems = pack_module.staged_phase3c_plan(root)
    if problems or resolved != NODE_PLAN_PATH:
        raise PlanRefused(
            "the pointer just written does not resolve to this generation's "
            "staged plan: " + "; ".join(problems or [str(resolved)]))
    return {"path": str(path), "sha256": sha256_file(path),
            "generation": GENERATION, "staged_plan": NODE_PLAN_PATH}


def staged_pointer_problems(root=None) -> list[str]:
    """Whether the pointer on disk is this generation's, and correct."""
    root = Path(root or ROOT)
    path = root / pack_module.PHASE3C_STAGED_POINTER
    if not path.is_file():
        return [f"{pack_module.PHASE3C_STAGED_POINTER} is not here; the pack "
                "would carry no plan"]
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{pack_module.PHASE3C_STAGED_POINTER} is not readable JSON "
                f"({exc})"]
    problems = []
    if body.get("generation") != GENERATION:
        problems.append(
            f"{pack_module.PHASE3C_STAGED_POINTER} names generation "
            f"{body.get('generation')!r} and this is {GENERATION!r}")
    if body.get("staged_plan") != NODE_PLAN_PATH:
        problems.append(
            f"{pack_module.PHASE3C_STAGED_POINTER} names "
            f"{body.get('staged_plan')!r}, not {NODE_PLAN_PATH!r}")
    resolved, refusals = pack_module.staged_phase3c_plan(root)
    problems += refusals
    if not refusals and resolved != NODE_PLAN_PATH:
        problems.append(f"the pointer resolves to {resolved!r}, not "
                        f"{NODE_PLAN_PATH!r}")
    return problems


def pack_carries_this_generation_problems(evidence) -> list[str]:
    """Whether the pack that was built carries this generation's plan alone.

    The pointer is data, and data can be stale. This is the check that does
    not depend on it: the archived file table is read directly, and the one
    Phase 3C plan in it has to be this generation's staged path carrying the
    archive's own bytes. A stale pointer that shipped a superseded plan
    cannot be authorised, whatever the pointer said at build time.
    """
    table = ((evidence or {}).get("pack_manifest") or {}).get("files")
    if not isinstance(table, dict):
        return ["the pack evidence carries no file table"]
    carried = sorted(
        rel for rel in table
        if rel.startswith("gpu_plans/phase3c_") and rel.endswith("_plan.json"))
    if carried != [NODE_PLAN_PATH]:
        return [f"the pack carries {carried} as Phase 3C plans; this "
                f"generation stages {NODE_PLAN_PATH!r} and only that one may "
                "travel"]
    archived = Path(ROOT) / PLAN_PATH
    if not archived.is_file():
        return [f"{PLAN_PATH} is not in the archive, so the carried plan "
                "cannot be checked against it"]
    want = sha256_file(archived)
    got = (table[NODE_PLAN_PATH] or {}).get("sha256")
    if got != want:
        return [f"the pack carries {NODE_PLAN_PATH} at {str(got)[:16]}..., "
                f"and the archive's plan is {want[:16]}.... The node would "
                "run against a document nobody archived."]
    return []


def staged_copy_problems(root=None, *, archive_dir=None) -> list[str]:
    """Whether the staged copies are still byte-identical to the archive."""
    root = Path(root or ROOT)
    archive = Path(archive_dir) if archive_dir else root / ARCHIVE_DIR
    problems: list[str] = []
    for name, staged in ((Path(PLAN_PATH).name, NODE_PLAN_PATH),):
        source, copy = archive / name, root / staged
        if not source.is_file():
            problems.append(f"{source} is not in the archive")
            continue
        if not copy.is_file():
            problems.append(f"{staged} has not been staged")
            continue
        if sha256_file(source) != sha256_file(copy):
            problems.append(
                f"{staged} is not the archive's bytes; the node would run "
                "against a document nobody archived")
    problems += staged_pointer_problems(root)
    return problems


def observed_environment(probe: dict, *, versions=None) -> dict:
    """The node's own reading, in the shape the contract pins.

    ``versions`` is injectable so a test can state a machine rather than be
    one. Read on the node it imports four packages; a version that cannot be
    read stays ``None`` and fails :func:`environment_problems` loudly, which
    is the point -- an unreadable check has not been satisfied.
    """
    if versions is None:
        def _version(name: str):
            try:
                return __import__(name).__version__
            except Exception:
                return None

        versions = {
            "python": _platform_python(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "accelerate": _version("accelerate"),
        }
    probe = probe or {}
    return {
        "os_system": probe.get("os_system"),
        "wsl2": probe.get("wsl2"),
        "device": SETTINGS.device,
        "dtype": SETTINGS.dtype,
        "gpu_name": probe.get("gpu_name"),
        "torch_cuda_build": probe.get("torch_cuda_build"),
        "python": versions.get("python"),
        "torch": versions.get("torch"),
        "transformers": versions.get("transformers"),
        "peft": versions.get("peft"),
        "accelerate": versions.get("accelerate"),
        "offline_env": dict(probe.get("offline_env") or {}),
        "local_files_only": True,
        "vram_total_gb": probe.get("vram_total_gb"),
        "system_ram_gb": probe.get("system_ram_gb"),
    }


def _platform_python() -> str:
    import platform

    return platform.python_version()


# ---------------------------------------------------------------------------
# Appending cells
# ---------------------------------------------------------------------------

def known_keys(path) -> set:
    """The cells already on disk. Raises on a damaged file, deliberately."""
    path = Path(path)
    if not path.exists():
        return set()
    return {cell_key(r) for r in read_rows(path)}


def append_cell(path, row: dict, *, known: set | None = None) -> None:
    """Append one cell, or refuse. There is no third outcome.

    Same discipline as Phase 2's: a cell is measured once, and replacing one
    would swap a measurement for another with nothing in the file saying so.
    ``known`` is the caller's own set, seeded from the file and updated here,
    so a 320-cell step stays linear instead of re-reading everything per row.
    """
    path = Path(path)
    missing = [f for f in RESULT_FIELDS if f not in row]
    if missing:
        raise PlanRefused(f"a sample row without {missing} is not a "
                          "measurement of anything")
    key = cell_key(row)
    seen = known if known is not None else known_keys(path)
    if key in seen:
        raise PlanRefused(
            f"{key} is already recorded. A cell is measured once; replacing "
            "it would swap one measurement for another silently.")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json({f: row[f] for f in RESULT_FIELDS}) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    if known is not None:
        known.add(key)


# ---------------------------------------------------------------------------
# The Mac's side: preconditions, then scoring from the bytes
# ---------------------------------------------------------------------------

def score_preconditions(out_dir, plan: dict, seal: dict, *,
                        carried_seal_digest: str, grant: dict,
                        root=None) -> list[str]:
    """Everything that must hold before a number is derived from this run.

    Deliberately re-derived rather than read. The receipt on disk lives
    inside the directory it authenticates, so a scorer that trusted it would
    be trusting a file an attacker with write access to the samples also had
    write access to. So: the seal is re-checked, every member and the
    execution manifest are re-hashed, the carried seal digest is compared
    again, the manifest is judged against both the plan and the
    grant, and the scorer's own source manifest is recomputed and
    matched against what the plan was approved with.

    The authorization is required, not optional. Without it this function
    once accepted a manifest naming any well-formed pack digest, which meant
    a run produced against an edited gate would have scored clean. This is
    the existing-result path as well as the first-run path: there is no
    branch that skips any of it because a receipt already exists.
    """
    out_dir = Path(out_dir)
    problems = list(seal_problems(seal, plan))
    if seal.get("seal_digest") != carried_seal_digest:
        problems.append(
            f"the seal in the directory digests to "
            f"{str(seal.get('seal_digest'))[:16]}... and the value carried "
            f"here separately was {str(carried_seal_digest)[:16]}...")

    problems.extend(authorization_problems(grant, plan, root=root,
                                           check_sources=False))
    problems.extend(manifest_chain_problems(out_dir, seal, plan=plan,
                                            grant=grant))

    _found, rehash_problems = rehash(out_dir, seal)
    problems.extend(rehash_problems)
    problems.extend(scorer_manifest_problems(
        plan.get("scorer_source_manifest"), root))
    if (scorer_manifest_digest(root)
            != plan.get("scorer_source_manifest_digest")):
        problems.append(
            "the scorer on this machine is not the scorer this plan was "
            "approved with, so a number derived here would not be the "
            "number the plan describes")
    return problems


def rescore(out_dir, plan: dict) -> dict:
    """``arm -> [scored draw]``, derived from the stored bytes only.

    The node writes no summary this reads. Every rate the report quotes is
    recomputed here from ``raw_text`` by the scorer the plan pins, so a node
    that reported a figure it did not measure changes nothing.

    The execution manifest is read *here*, from the directory being scored,
    and every row is required to name it. No caller passes the value in:
    a scorer that accepted a digest as an argument would score whatever the
    caller believed rather than what the directory says.
    """
    out_dir = Path(out_dir)
    manifest_digest = read_execution_manifest(out_dir)["manifest_digest"]
    cases = case_index(plan)
    scores_by_arm: dict = {name: [] for name in ARM_ORDER}
    seen: set = set()
    for index in range(N_STEPS):
        path = out_dir / samples_member(index)
        rows = read_rows(path)
        problems = step_problems(rows, plan, index,
                                 manifest_digest=manifest_digest)
        if problems:
            raise PlanRefused(
                f"step {index} does not hold and will not be scored:\n  - "
                + "\n  - ".join(problems[:20]))
        for row in rows:
            key = cell_key(row)
            if key in seen:
                raise PlanRefused(f"{key} appears in more than one step")
            seen.add(key)
            scores_by_arm[row["arm"]].append(
                score_row(row, cases[row["case_id"]]))
    expected = N_CASES * SETTINGS.k
    for name in ARM_ORDER:
        if len(scores_by_arm[name]) != expected:
            raise PlanRefused(f"arm {name} has {len(scores_by_arm[name])} "
                              f"draws, not {expected}")
    return scores_by_arm


def scores_identity(record) -> dict:
    """What makes two score records the same result: everything.

    Deliberately no longer a chosen subset. It was eleven named fields, and
    the ones left out -- ``seeds``, ``arms``, ``primary_contrast``,
    ``uncertainty``, the two scorer manifest digests, the note -- are
    exactly the ones a rewrite would target: a record claiming a different
    resampling rule, a different primary contrast or a different scorer
    compared equal to the real one.

    Only ``scores_digest`` is excluded, because it is the digest *of* this
    mapping and including it would make the comparison circular.
    """
    if not isinstance(record, dict):
        return {"not_a_record": str(type(record).__name__)}
    return {k: v for k, v in record.items() if k != "scores_digest"}


def scores_problems(stored, rebuilt) -> list[str]:
    """Field-by-field, both directions, plus the record's own digest.

    ``scores_identity`` compares two whole mappings; this says *which* field
    disagrees, which is the difference between a refusal somebody can act on
    and one they have to bisect.
    """
    problems: list[str] = []
    if not isinstance(stored, dict):
        return [f"the score record is a {type(stored).__name__}"]
    mine, theirs = scores_identity(stored), scores_identity(rebuilt)
    extra = sorted(set(mine) - set(theirs))
    missing = sorted(set(theirs) - set(mine))
    if extra:
        problems.append(f"the stored scores carry {extra}, which a record "
                        "derived from these samples does not have")
    if missing:
        problems.append(f"the stored scores are missing {missing}")
    for field in sorted(set(mine) & set(theirs)):
        if mine[field] != theirs[field]:
            problems.append(
                f"the stored scores' {field} is not what these samples "
                "derive")
    if not _is_digest(stored.get("scores_digest")):
        problems.append("scores_digest is not a SHA-256")
    elif stored["scores_digest"] != digest_obj(mine):
        problems.append("scores_digest does not cover the score record")
    return problems


__all__ = [
    "ARM_ORDER", "ARMS", "CONTRASTS", "PRIMARY_CONTRAST", "SETTINGS",
    "N_CASES", "N_PAIRS", "N_STEPS", "STEP_ORDER", "TERMINATIONS",
    "ACCEPTED_TERMINATIONS", "EOS_TERMINATIONS", "EXECUTION_ENVIRONMENT",
    "arm", "contract_digest", "contract_document", "contrast_name",
    "correct_token_accounting", "expected_complete_bricks", "isolation_audit",
    "materialize_plan", "plan_problems", "read_plan", "case_index",
    "step", "step_cells", "step_cases", "step_name", "samples_member",
    "default_loaders", "build_interface", "warm_up", "adapter_digests",
    "warm_up_problems", "expected_warm_up", "WARMUP_FIELDS",
    "failed_attempt_problems", "attempted_execution_problems",
    "failed_attempt_cells",
    "FAILED_ATTEMPT_ROOT", "FAILED_ATTEMPT_KIND", "MEMBER_STATES",
    "FAILED_ATTEMPT_SCHEMA_VERSION", "FAILED_ATTEMPT_SCHEMA_VERSIONS",
    "ZERO_CELL_PROOF", "PARTIAL_PROOF",
    "gate_class", "observed_gate", "gate_ledger",
    "observed_model_identity", "loaded_identity_problems",
    "verified_model_identity", "model_identity",
    "run_case", "row_problems", "read_rows", "step_problems", "score_row",
    "score_record", "wilson", "paired_bootstrap", "discordant",
    "build_execution_manifest", "manifest_problems", "build_seal",
    "seal_problems", "build_receipt", "receipt_problems", "rehash",
    "environment_problems", "membership_document", "scorer_manifest",
    "scorer_manifest_digest", "scorer_manifest_problems",
    "PLAN_PATH", "MEMBERSHIP_PATH", "AUDIT_PATH", "MANIFEST_NAME",
    "SEAL_NAME", "RECEIPT_NAME", "SCORES_NAME", "write_plan_set",
    "REPORT_NAME", "SUCCESS_INDEX_NAME", "FAILURE_INDEX_NAME",
    "REPRODUCE_NAME", "PUBLISHED_NAMES",
    "observed_environment", "known_keys", "append_cell",
    "score_preconditions", "rescore", "scores_identity", "scores_problems",
    "receipt_identity", "expected_receipt", "RECEIPT_VOLATILE_FIELDS",
    "GENERATION", "ARCHIVE_ROOT", "ARCHIVE_DIR", "AUTHORIZATION_PATH",
    "NODE_PLAN_PATH", "stage_for_node",
    "staged_copy_problems", "staged_pointer_document", "write_staged_pointer",
    "staged_pointer_problems", "pack_carries_this_generation_problems",
    "CONTRACT_SNAPSHOT_PATH", "SUPERSEDED", "NON_SAMPLE_FILES",
    "AUTHORIZATION_KIND", "GENERATION_ENTRY_POINT", "GATE_SOURCES",
    "generation_sources", "generation_source_manifest",
    "generation_source_manifest_digest", "generation_source_problems",
    "build_authorization", "authorization_problems", "read_authorization",
    "carried_digest_problems", "manifest_chain_problems",
    "read_execution_manifest", "row_binding_problems",
    "PACK_EVIDENCE_PATH", "SUPERSESSION_PATH", "ARCHIVE_MEMBERS",
    "archive_problems", "build_supersession_record", "build_pack_evidence",
    "pack_evidence_problems", "supersession_problems",
    "CONTRACT_SNAPSHOT_KIND", "build_contract_snapshot",
    "contract_snapshot_problems",
]
