"""The Mac scorer: what a raw generation is, once somebody reads it properly.

Everything here is deterministic and none of it needs a GPU, a model or a
network. That is the point of the split: the execution node produces text and
a termination reason, and every claim made about that text is made on the
machine whose numbers the report quotes, by code that can be re-run over the
stored text at any time and must give the same answer.

Every formula is in :data:`src.eval.acceptance.METRIC_SPEC`, which is inside
the contract digest. This module reads the check names and the quantile
function from there rather than restating them, so a metric cannot be
specified in one place and computed in another.

Four things this deliberately does not do.

**It does not call anything stability.** ``unsupported_bricks`` counts bricks
with nothing directly beneath them, which is a description of the geometry and
not a physical claim -- most corpus structures contain some, and a brick with
no support below can still be held by one above. Real stability analysis needs
the physics the project does not have (Gurobi, still unlicensed). So the count
is reported under its own name with a note attached.

**It does not invent a semantic score.** No CLIP threshold has been calibrated
and nothing is rendered, so there is no semantic success and therefore no full
success either.

**It does not report zero for something it did not count.** Candidate
rejections, brick retries, previous-brick backtracks and physics rollbacks all
belong to a rejection layer that does not exist yet. They come through as
``null`` with ``implemented: false``.

**It does not report a ratio, a percentage change or a significance claim.**
A contrast is ``value(a) - value(b)`` with both raw values and both
denominators printed beside it. This run has one sample of each cell; nothing
here estimates a variance, so nothing here may imply one.

Connectivity is stud coupling alone -- ``ground=False``. Two bricks side by
side in a layer touch and do not connect, and joining everything that rests on
``z == 0`` would pass an assembly that falls into pieces when it is lifted off
the baseplate. Whether the model is anchored at all is a separate question and
is reported separately, as ``touches_ground`` -- which core success does
require, because a structure floating in the air is not one anybody can build.
"""

from __future__ import annotations

from collections import Counter

from src.data.bricks import (PART_INDEX, connected_components,
                             find_collisions, is_connected, required_inventory,
                             touches_ground, unsupported_bricks)
from src.eval.acceptance import (ACCEPTED_TERMINATIONS, ARM_ORDER, CONTRASTS,
                                 CORE_SUCCESS_CHECKS, QUANTILE_METHOD,
                                 QUANTILE_PROBS, ROLES, VARIANTS,
                                 contrast_name, quantile, quantiles,
                                 scorer_manifest, scorer_manifest_digest,
                                 unimplemented_counters)
from src.generation.brickgpt import TOKENS_PER_BRICK, parse_output
from src.rendering.ldr import to_ldr

#: A generation that ended on EOS spends one token on it, at slot 0. Every
#: other stop lands on a multiple of ten because the two budgets coincide.
EOS_TOKENS = 1

#: The per-draw booleans reported as arm rates and compared across arms: the
#: core-success conjuncts, plus the conjunction itself.
DRAW_BOOLEANS: tuple[str, ...] = CORE_SUCCESS_CHECKS + (
    "deterministic_core_success",)

__all__ = [
    "expected_complete_bricks", "count_overflow", "score_generation",
    "score_row", "core_success_at_k", "per_arm_summary",
    "paired_comparisons", "seconds_summary", "score_record",
]


def expected_complete_bricks(n_tokens: int, termination: str
                             ) -> tuple[int | None, str | None]:
    """How many whole bricks that many tokens can be, or why they cannot.

    EOS is only ever offered at slot 0, so a run that ended on it spent
    ``10 * bricks + 1`` tokens and a run that ended on a budget spent a
    multiple of ten. A count that is neither means the decoder stopped inside
    a brick, which the grammar is supposed to make impossible -- so it is
    reported rather than rounded away.
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) \
            or n_tokens < 1:
        return None, f"n_tokens is {n_tokens!r}"
    if termination in ACCEPTED_TERMINATIONS:
        body = n_tokens - EOS_TOKENS
        if body % TOKENS_PER_BRICK:
            return None, (f"{n_tokens} tokens ending on EOS is not "
                          f"{TOKENS_PER_BRICK}n + 1; the run stopped inside a "
                          "brick")
        return body // TOKENS_PER_BRICK, None
    if n_tokens % TOKENS_PER_BRICK:
        return None, (f"{n_tokens} tokens ending on {termination} is not a "
                      f"multiple of {TOKENS_PER_BRICK}; the run stopped "
                      "inside a brick")
    return n_tokens // TOKENS_PER_BRICK, None


def count_overflow(used: Counter, inventory: dict) -> dict:
    """The workflow's definition, both halves of it.

    ``amount`` is the total number of bricks drawn beyond stock and ``rate``
    is that over the total drawn, with the denominator floored at one so an
    empty generation is 0.0 rather than undefined.
    """
    over = {p: n - inventory.get(p, 0)
            for p, n in used.items() if n > inventory.get(p, 0)}
    amount = sum(over.values())
    total = sum(used.values())
    return {
        "overdrawn": dict(sorted(over.items())),
        "count_overflow_amount": amount,
        "count_overflow_rate": amount / max(1, total),
        "used_total": total,
    }


def score_generation(raw_text: str, *, inventory: dict, n_tokens: int,
                     termination: str) -> dict:
    """Every deterministic check, on one generation. No model, no network."""
    bricks, unparsed = parse_output(raw_text)
    expected, token_note = expected_complete_bricks(n_tokens, termination)
    tokens_consistent = expected is not None and expected == len(bricks)

    parse_success = bool(bricks) and not unparsed and tokens_consistent

    unknown = sorted({b.part for b in bricks if b.part not in PART_INDEX})
    known_parts = not unknown

    used = required_inventory(bricks)
    stocked = {p for p, n in inventory.items() if n > 0}
    type_violations = sorted({p for p in used if p not in stocked})
    overflow = count_overflow(used, inventory)
    inventory_valid = not type_violations and \
        overflow["count_overflow_amount"] == 0

    out_of_bounds = [i for i, b in enumerate(bricks) if not b.in_bounds()]
    collisions = find_collisions(bricks)

    components = connected_components(bricks, ground=False)
    connected = is_connected(bricks, ground=False)
    grounded = touches_ground(bricks)

    unsupported = unsupported_bricks(bricks)

    ldraw_error = None
    try:
        to_ldr(bricks)
    except Exception as exc:                 # an unknown part has no LDraw id
        ldraw_error = f"{type(exc).__name__}: {exc}"
    ldraw_serializable = ldraw_error is None and bool(bricks)

    # One flat table keyed by the contract's own check names. The conjunction
    # below iterates it rather than repeating the list, so the specification
    # and the computation cannot come apart.
    checks = {
        "parse_success": parse_success,
        "known_parts": known_parts,
        "type_compliance": not type_violations,
        "inventory_valid": inventory_valid,
        "in_bounds": not out_of_bounds,
        "collision_free": not collisions,
        "stud_only_connected": connected,
        "touches_ground": grounded,
        "ldraw_serializable": ldraw_serializable,
        "termination_accepted": termination in ACCEPTED_TERMINATIONS,
    }
    missing = [name for name in CORE_SUCCESS_CHECKS if name not in checks]
    if missing:
        raise RuntimeError(
            f"the contract names core-success checks this scorer does not "
            f"compute: {missing}")
    core = all(bool(checks[name]) for name in CORE_SUCCESS_CHECKS)
    checks["deterministic_core_success"] = core

    return {
        "checks": checks,
        "parse": {
            "n_bricks": len(bricks),
            "n_unparsed_lines": len(unparsed),
            "unparsed_lines": unparsed,
            "expected_complete_bricks": expected,
            "token_brick_note": token_note,
            "tokens_match_complete_bricks": tokens_consistent,
            "parse_success": parse_success,
        },
        "parts": {
            "known_parts": known_parts,
            "unknown_parts": unknown,
        },
        "inventory": {
            "initial": dict(sorted(inventory.items())),
            "used": dict(sorted(used.items())),
            "type_compliance": not type_violations,
            "type_violations": type_violations,
            **overflow,
            "inventory_valid": inventory_valid,
        },
        "geometry": {
            "in_bounds": not out_of_bounds,
            "out_of_bounds_indices": out_of_bounds,
            "collision_free": not collisions,
            "colliding_pairs": [list(p) for p in collisions],
        },
        "connectivity": {
            "criterion": "stud coupling between adjacent layers; ground=False",
            "stud_only_connected": connected,
            "n_components": len(components),
            "touches_ground": grounded,
        },
        "support": {
            "unsupported_brick_count": len(unsupported),
            "unsupported_brick_rate": len(unsupported) / max(1, len(bricks)),
            "note": ("descriptive only: bricks with nothing directly beneath "
                     "them. Not a physics result and not a claim about "
                     "whether the model stands up."),
        },
        "ldraw": {
            "serializable": ldraw_serializable,
            "error": ldraw_error,
        },
        "termination": termination,
        "termination_accepted": checks["termination_accepted"],
        "counters": unimplemented_counters(),
        "deterministic_core_success": core,
    }


def score_row(row: dict, case: dict) -> dict:
    """One stored result cell, scored against the case it belongs to."""
    if row["case_id"] != case["case_id"]:
        raise ValueError(f"row {row['case_id']!r} scored against case "
                         f"{case['case_id']!r}")
    scored = score_generation(row["raw_text"], inventory=case["inventory"],
                              n_tokens=row["n_tokens"],
                              termination=row["termination"])
    return {
        "case_id": row["case_id"],
        "pair_id": case["pair_id"],
        "role": case["role"],
        "variant": case["variant"],
        "arm": row["arm"],
        "seed": row["seed"],
        "step_index": row.get("step_index"),
        "group": row.get("group"),
        # Exactly as measured and stored. Rounding here would make every
        # paired difference below a difference between rounded numbers.
        "seconds": row["seconds"],
        "n_tokens": row["n_tokens"],
        **scored,
    }


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def strata() -> list[tuple[str, str | None]]:
    """Every stratum a contrast is reported over, in a fixed order."""
    out: list[tuple[str, str | None]] = [("overall", None)]
    out += [("role", role) for role in ROLES]
    out += [("variant", variant) for variant in VARIANTS]
    return out


def _in_stratum(score: dict, kind: str, value) -> bool:
    if kind == "overall":
        return True
    return score.get(kind) == value


def select(scores, kind: str, value) -> list[dict]:
    return [s for s in scores if _in_stratum(s, kind, value)]


# ---------------------------------------------------------------------------
# Per-arm figures
# ---------------------------------------------------------------------------

def seconds_summary(values) -> dict:
    """n, total, mean, min, max and the frozen quantiles. One algorithm."""
    xs = [float(v) for v in values]
    return {
        "n": len(xs),
        "total": sum(xs) if xs else 0.0,
        "mean": sum(xs) / len(xs) if xs else None,
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "quantiles": quantiles(xs),
        "quantile_method": QUANTILE_METHOD,
    }


def boolean_rate(scores, name: str) -> dict:
    """A rate with its numerator and its denominator, never just a rate."""
    n = len(scores)
    hits = sum(1 for s in scores if s["checks"][name])
    return {"numerator": hits, "denominator": n,
            "value": hits / n if n else None}


def core_success_at_k(scores, *, k: int, arms=ARM_ORDER) -> dict:
    """``Core Success@K``: one case counts if any of its K seeds succeeded.

    A case whose seeds are not all present is reported as incomplete rather
    than folded in. "At least one of the four" said over three seeds is a
    different quantity wearing the same name.
    """
    per_arm = {}
    for name in arms:
        per_arm[name] = {}
        for kind, value in strata():
            per_arm[name][_stratum_key(kind, value)] = _core_at_k(
                select([s for s in scores if s["arm"] == name], kind, value),
                k=k)
    return {"k": k, "metric": "Core Success@K", "strata": list(STRATUM_KEYS()),
            "by_arm": per_arm}


def _stratum_key(kind: str, value) -> str:
    return kind if value is None else f"{kind}={value}"


def STRATUM_KEYS():
    return [_stratum_key(kind, value) for kind, value in strata()]


def _core_at_k(scores, *, k: int) -> dict:
    by_case: dict[str, list[dict]] = {}
    for s in scores:
        by_case.setdefault(s["case_id"], []).append(s)

    complete, successes, incomplete = 0, 0, []
    for case_id in sorted(by_case):
        got = by_case[case_id]
        if len({s["seed"] for s in got}) != k or len(got) != k:
            incomplete.append(case_id)
            continue
        complete += 1
        if any(s["deterministic_core_success"] for s in got):
            successes += 1
    return {
        "k": k,
        "cases_seen": len(by_case),
        "numerator": successes,
        "denominator": complete,
        "value": successes / complete if complete else None,
        "incomplete_cases": incomplete,
    }


def per_arm_summary(scores, *, arms=ARM_ORDER, k: int) -> dict:
    """Everything reported about one arm on its own, per stratum."""
    out = {}
    for name in arms:
        mine = [s for s in scores if s["arm"] == name]
        out[name] = {}
        for kind, value in strata():
            rows = select(mine, kind, value)
            overflow_amount = sum(
                s["inventory"]["count_overflow_amount"] for s in rows)
            used_total = sum(s["inventory"]["used_total"] for s in rows)
            out[name][_stratum_key(kind, value)] = {
                "draws": len(rows),
                "cases": len({s["case_id"] for s in rows}),
                "rates": {b: boolean_rate(rows, b) for b in DRAW_BOOLEANS},
                "core_success_at_k": _core_at_k(rows, k=k),
                "macro_count_overflow_rate": {
                    "numerator": sum(s["inventory"]["count_overflow_rate"]
                                     for s in rows),
                    "denominator": len(rows),
                    "value": (sum(s["inventory"]["count_overflow_rate"]
                                  for s in rows) / len(rows)
                              if rows else None),
                },
                "micro_count_overflow_rate": {
                    "numerator": overflow_amount,
                    "denominator": max(1, used_total),
                    "bricks_used": used_total,
                    "value": (overflow_amount / max(1, used_total)
                              if rows else None),
                },
                "count_overflow_amount_total": overflow_amount,
                "unsupported_brick_rate_mean": (
                    sum(s["support"]["unsupported_brick_rate"] for s in rows)
                    / len(rows) if rows else None),
                "termination_reasons": dict(sorted(
                    Counter(s["termination"] for s in rows).items())),
                "seconds": seconds_summary(s["seconds"] for s in rows),
            }
    return out


# ---------------------------------------------------------------------------
# The two contrasts
# ---------------------------------------------------------------------------

def _delta(a: dict, b: dict) -> dict:
    """One comparison line: both raw values, both denominators, a - b.

    Absolute, not relative. A ratio over a rate that can be zero is a number
    that stops existing exactly where the interesting cases are, and a
    percentage change invites a significance reading this run cannot support.
    """
    av, bv = a.get("value"), b.get("value")
    return {
        "a_value": av,
        "b_value": bv,
        "delta": (av - bv) if (av is not None and bv is not None) else None,
        "a_numerator": a.get("numerator"),
        "a_denominator": a.get("denominator"),
        "b_numerator": b.get("numerator"),
        "b_denominator": b.get("denominator"),
        "denominators_match": a.get("denominator") == b.get("denominator"),
    }


def paired_seconds(scores, a: str, b: str, kind: str, value) -> dict:
    """``seconds(a) - seconds(b)`` for every cell both arms measured.

    Paired rather than a difference of means: the same 160 captions and
    inventories go to both arms, and the case-to-case spread is far larger
    than the effect. Cells one arm has and the other does not are excluded
    and counted, because a paired difference needs both halves.
    """
    def by_cell(name):
        return {(s["case_id"], s["seed"]): s["seconds"]
                for s in select(scores, kind, value) if s["arm"] == name}

    left, right = by_cell(a), by_cell(b)
    shared = sorted(set(left) & set(right))
    deltas = [left[c] - right[c] for c in shared]
    return {
        "pairs_compared": len(shared),
        "a_draws": len(left),
        "b_draws": len(right),
        "unpaired_a": len(set(left) - set(right)),
        "unpaired_b": len(set(right) - set(left)),
        "mean": sum(deltas) / len(deltas) if deltas else None,
        "min": min(deltas) if deltas else None,
        "max": max(deltas) if deltas else None,
        "median": quantile(deltas, 0.5),
        "quantiles": quantiles(deltas),
        "quantile_method": QUANTILE_METHOD,
        "definition": "seconds(a) - seconds(b), per (case_id, seed)",
    }


def paired_comparisons(scores, *, k: int, contrasts=CONTRASTS) -> dict:
    """B - C and D - E, over the whole set and over every role and variant.

    The scorer produces these directly rather than leaving a reader to
    subtract two tables. Subtracting them by hand is where a stratum gets
    compared against the wrong stratum, and where a denominator that differed
    between the two arms goes unnoticed.
    """
    out = []
    for a, b in contrasts:
        rows = []
        for kind, value in strata():
            left = select([s for s in scores if s["arm"] == a], kind, value)
            right = select([s for s in scores if s["arm"] == b], kind, value)
            metrics = {
                name: _delta(boolean_rate(left, name),
                             boolean_rate(right, name))
                for name in DRAW_BOOLEANS
            }
            metrics["core_success_at_k"] = _delta(_core_at_k(left, k=k),
                                                  _core_at_k(right, k=k))
            for label, getter in (
                    ("macro_count_overflow_rate", _macro_overflow),
                    ("micro_count_overflow_rate", _micro_overflow),
                    ("mean_seconds", _mean_seconds)):
                metrics[label] = _delta(getter(left), getter(right))
            rows.append({
                "stratum": {"kind": kind, "value": value,
                            "key": _stratum_key(kind, value)},
                "a_draws": len(left),
                "b_draws": len(right),
                "a_cases": len({s["case_id"] for s in left}),
                "b_cases": len({s["case_id"] for s in right}),
                "metrics": metrics,
                "paired_seconds_delta": paired_seconds(scores, a, b, kind,
                                                       value),
            })
        out.append({
            "contrast": contrast_name(a, b),
            "a": a, "b": b,
            "direction": f"every delta is value({a}) - value({b})",
            "strata": rows,
        })
    return {
        "contrasts": out,
        "note": ("absolute differences only. No ratio, no percentage change "
                 "and no significance claim: this run has one sample of each "
                 "cell and nothing here estimates a variance."),
    }


def _macro_overflow(scores) -> dict:
    total = sum(s["inventory"]["count_overflow_rate"] for s in scores)
    return {"numerator": total, "denominator": len(scores),
            "value": total / len(scores) if scores else None}


def _micro_overflow(scores) -> dict:
    over = sum(s["inventory"]["count_overflow_amount"] for s in scores)
    used = sum(s["inventory"]["used_total"] for s in scores)
    return {"numerator": over, "denominator": max(1, used),
            "value": over / max(1, used) if scores else None}


def _mean_seconds(scores) -> dict:
    xs = [s["seconds"] for s in scores]
    return {"numerator": sum(xs) if xs else 0.0, "denominator": len(xs),
            "value": sum(xs) / len(xs) if xs else None}


# ---------------------------------------------------------------------------
# The whole record
# ---------------------------------------------------------------------------

def score_record(scores, *, k: int, arms=ARM_ORDER, contrasts=CONTRASTS,
                 root=None) -> dict:
    """Per-draw scores, per-arm summaries and both contrasts, in one object.

    The scorer's own source manifest goes in here. A record that does not say
    which parser, which checkers and which LDraw writer produced it is a
    record whose numbers cannot be reproduced, and re-scoring stored text
    under changed code is legitimate exactly as long as the record says so.
    """
    named = tuple(arms)
    usable = [c for c in contrasts if c[0] in named and c[1] in named]
    return {
        "kind": "core_eval_scores",
        "arms": list(named),
        "k": k,
        "draws": len(scores),
        "cases": len({s["case_id"] for s in scores}),
        "strata": STRATUM_KEYS(),
        "quantiles": {"probabilities": list(QUANTILE_PROBS),
                      "method": QUANTILE_METHOD},
        "scorer_source_manifest": scorer_manifest(root),
        "scorer_source_manifest_digest": scorer_manifest_digest(root),
        "core_success_at_k": core_success_at_k(scores, k=k, arms=named),
        "per_arm": per_arm_summary(scores, arms=named, k=k),
        "contrasts": paired_comparisons(scores, k=k, contrasts=usable),
        "per_draw": scores,
        "note": ("Deterministic checks only. No stability analysis, no "
                 "semantic score and nothing rendered; the four rejection "
                 "and rollback counters are null because no such layer "
                 "exists in this run."),
    }
