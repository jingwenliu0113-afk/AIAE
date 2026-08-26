#!/usr/bin/env python3
"""The project demonstration, from the command line.

    inventory + brief -> brick list -> checks -> LDraw + plan/3-D preview

The brick list comes from exactly one of three places, the modes are mutually
exclusive, and the report always names the one that produced it:

    --sample NAME     a stored brief that ships with the code, used whole. No
                      model, no network, no GPU. Nothing about it may be
                      overridden -- a changed sample is not that sample.
    --variant-of NAME the same stored brick text with the operator's own
                      caption, inventory and termination. Reported as
                      supplied text labelled a variant, never as the sample.
    --bricks FILE     brick text from anywhere; ``-`` reads standard input.
    --generate        decode with real weights. The only mode that loads a
                      model, and the only one where the token count and the
                      termination are measured rather than derived.

Flags that do not apply to the chosen mode are **refused, not ignored**: a
sampling temperature quietly dropped on a stored brief is a report that looks
like it honoured a setting it never saw.

What this is not: a measurement. Nothing it prints is a metric, none of it is
comparable to the frozen Phase 2 evaluation, and ``--placement`` turning a
check green is not evidence that anything improved -- that gate has never been
formally evaluated. The module docstring of :mod:`src.demo.showcase` states
the limits in full, and every report repeats the short form.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constraints.placement_decode import CONNECTIVITY_MODES
from src.demo.showcase import (INVENTORY_HELP, MODELS, MODEL_PROJECT, SAMPLES,
                               TERMINATIONS, ShowcaseError, format_report,
                               inspect_sample, inspect_supplied, parse_inventory,
                               passed, sample, write_ldraw)

#: Which flags each mode accepts. Anything else the operator passed is
#: refused by name: silently ignoring a flag is how a report ends up looking
#: like it honoured a setting nothing ever read.
ALLOWED: dict[str, set[str]] = {
    "sample": set(),
    "variant_of": {"caption", "inventory", "termination"},
    "bricks": {"caption", "inventory", "termination"},
    "generate": {"caption", "inventory", "model", "device", "seed",
                 "temperature", "max_bricks", "max_tokens", "placement",
                 "connectivity"},
}

#: Allowed in every mode: they choose how the report is printed, not what it
#: says about where anything came from.
OUTPUT_FLAGS = {"ldr", "preview", "json", "prompt", "no_plan", "list"}

#: Defaults applied after validation, so "the operator passed it" and "it has
#: a value" stay different questions.
DECODE_DEFAULTS = {"model": MODEL_PROJECT, "device": "mps", "seed": 0,
                   "temperature": 0.6, "max_bricks": 80, "max_tokens": 800,
                   "placement": False, "connectivity": "off"}

EXIT_OK, EXIT_CHECK_FAILED, EXIT_REFUSED, EXIT_UNDECIDED = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrickAgain demonstration. Measures nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stored briefs:\n" + "\n".join(
            f"  {s.name:<12}{s.shows}" for s in SAMPLES.values()))

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", choices=sorted(SAMPLES),
                     help="run a stored brief whole; no model needed")
    src.add_argument("--variant-of", choices=sorted(SAMPLES), metavar="NAME",
                     help="a stored brief's text with your own caption, "
                          "inventory and termination; reported as a variant")
    src.add_argument("--bricks", metavar="FILE",
                     help="read brick text from FILE, or '-' for stdin")
    src.add_argument("--generate", action="store_true",
                     help="decode with real weights (requires them locally)")
    src.add_argument("--list", action="store_true",
                     help="list the stored briefs and exit")

    p.add_argument("--caption", help="the text brief")
    p.add_argument("--inventory", help=INVENTORY_HELP)
    p.add_argument("--termination", choices=sorted(TERMINATIONS),
                   help="why the run that produced this text stopped. Leave "
                        "it out and the report records it as unavailable; it "
                        "is never assumed and never called measured.")

    p.add_argument("--placement", action="store_true", default=None,
                   help="opt in to the collision/connectivity gate "
                        "(--generate only; never formally evaluated)")
    p.add_argument("--connectivity", choices=CONNECTIVITY_MODES,
                   help="connectivity mode for the placement gate")
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-bricks", type=int)
    p.add_argument("--max-tokens", type=int)

    p.add_argument("--ldr", metavar="FILE", help="write the LDraw file here")
    p.add_argument("--preview", metavar="FILE",
                   help="write a CPU-only 3-D geometric preview (.png/.svg)")
    p.add_argument("--json", action="store_true",
                   help="print the report as JSON instead of text")
    p.add_argument("--prompt", action="store_true",
                   help="also print the prompt the model would be given")
    p.add_argument("--no-plan", action="store_true",
                   help="omit the plan view")
    return p


def chosen_mode(args) -> str:
    for name in ("sample", "variant_of", "bricks"):
        if getattr(args, name) is not None:
            return name
    return "generate"


def given_flags(args) -> set[str]:
    """Which optional flags the operator actually passed.

    Reads ``None`` as absent, which is why every one of them defaults to
    ``None`` rather than to a usable value.
    """
    return {name for name in vars(args)
            if name not in OUTPUT_FLAGS
            and name not in ("sample", "variant_of", "bricks", "generate")
            and getattr(args, name) is not None}


#: Flags that describe the brief itself rather than a decode. Refused on
#: --sample for a different reason from the rest, and the hint says which.
BRIEF_FLAGS = {"caption", "inventory", "termination"}


def refuse_inapplicable(args, mode: str) -> None:
    extra = sorted(given_flags(args) - ALLOWED[mode])
    if not extra:
        return
    flags = ", ".join("--" + f.replace("_", "-") for f in extra)
    if mode == "generate":
        hint = ("A decode measures its own token count and termination; "
                "neither is the operator's to state.")
    elif set(extra) <= BRIEF_FLAGS:
        hint = ("A stored brief is used whole -- its caption, inventory and "
                "termination belong with its text. To supply your own, use "
                f"--variant-of {args.sample}, which reports the result as "
                "supplied text labelled a variant.")
    else:
        hint = ("The weights, the sampling settings and the placement gate "
                "describe a decode that happened; use --generate.")
    raise ShowcaseError(
        f"{flags} does not apply to --{mode.replace('_', '-')} and will not "
        f"be ignored. {hint}")


#: Pairs that parse, apply to the mode, and still change nothing. Refused
#: for the same reason an inapplicable flag is: a flag that was read and had
#: no effect leaves a report looking like it honoured a setting.
def refuse_ineffective(args) -> None:
    if args.connectivity is not None and not args.placement:
        raise ShowcaseError(
            "--connectivity configures the placement gate and the placement "
            "gate was not asked for, so it would change nothing. Add "
            "--placement, or drop it.")
    for flag in ("prompt", "no_plan"):
        if getattr(args, flag):
            if args.json:
                raise ShowcaseError(
                    f"--{flag.replace('_', '-')} changes the printed report, "
                    "and --json prints none. The JSON always carries the "
                    "prompt and the plan view, so this would change nothing.")


def require(args, *names) -> None:
    missing = [f"--{n}" for n in names if getattr(args, n) is None]
    if missing:
        raise ShowcaseError(
            f"{', '.join(missing)} is required with "
            f"--{chosen_mode(args).replace('_', '-')}")


def read_bricks(where: str) -> str:
    if where == "-":
        return sys.stdin.read()
    path = Path(where)
    if not path.is_file():
        raise ShowcaseError(f"{where} is not a file")
    return path.read_text(encoding="utf-8")


def refuse_same_output_path(args) -> None:
    """Do not let the second writer overwrite the first output."""
    if args.ldr is None or args.preview is None:
        return
    ldraw = Path(args.ldr).expanduser().resolve(strict=False)
    preview = Path(args.preview).expanduser().resolve(strict=False)
    if ldraw == preview:
        raise ShowcaseError(
            "--ldr and --preview resolve to the same output path; use two "
            "different files so neither output overwrites the other")


def report_for(args) -> dict:
    """One report, from whichever mode was named."""
    mode = chosen_mode(args)
    refuse_inapplicable(args, mode)
    refuse_ineffective(args)

    if mode == "sample":
        return inspect_sample(args.sample)

    if mode == "variant_of":
        require(args, "caption", "inventory")
        return inspect_supplied(
            args.caption, parse_inventory(args.inventory),
            sample(args.variant_of).text,
            origin=f"sample-variant:{args.variant_of}",
            termination=args.termination, variant_of=args.variant_of)

    if mode == "bricks":
        require(args, "caption", "inventory")
        origin = "stdin" if args.bricks == "-" else f"file:{args.bricks}"
        return inspect_supplied(
            args.caption, parse_inventory(args.inventory),
            read_bricks(args.bricks), origin=origin,
            termination=args.termination)

    require(args, "caption", "inventory")
    settings = {k: (v if (v := getattr(args, k)) is not None else default)
                for k, default in DECODE_DEFAULTS.items()}
    # Imported here, not at the top: this script has to run, and its tests
    # have to pass, on a machine with no torch and no weights.
    from src.demo.showcase import generate

    return generate(args.caption, parse_inventory(args.inventory), **settings)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        # ``--list`` used to return before validation, so flags such as
        # ``--seed``, ``--json`` and ``--ldr`` were silently ignored.  Listing
        # is an operation in its own right and accepts no modifiers.
        extras = sorted(
            name for name, value in vars(args).items()
            if name != "list" and value is not None and value is not False)
        if extras:
            flags = ", ".join("--" + name.replace("_", "-")
                              for name in extras)
            print(f"refused: {flags} does not apply to --list and will not "
                  "be ignored.", file=sys.stderr)
            return EXIT_REFUSED
        for name, s in sorted(SAMPLES.items()):
            print(f"{name}\n  brief      : {s.caption}\n  inventory  : "
                  + ", ".join(f"{p}:{n}" for p, n in s.inventory.items())
                  + f"\n  termination: {s.termination} (stated by the "
                    f"fixture, not measured)\n  shows      : {s.shows}\n")
        print("These are hand-written fixtures. None came out of a model, "
              "and no result computed from one is evidence about any model. "
              "They are used whole: to change one, use --variant-of.")
        return EXIT_OK

    try:
        refuse_same_output_path(args)
        if args.preview:
            from src.rendering.preview import (PreviewError,
                                               validate_preview_path)
            try:
                validate_preview_path(args.preview)
            except PreviewError as exc:
                raise ShowcaseError(str(exc)) from exc
        report = report_for(args)
        if args.ldr:
            report["ldraw_written_to"] = str(write_ldraw(report, args.ldr))
        if args.preview:
            from src.generation.brickgpt import parse_output
            from src.rendering.preview import PreviewError, write_preview
            bricks, unparsed = parse_output(report["result"]["text"])
            if unparsed:
                raise ShowcaseError(
                    "the 3-D preview refuses unparsed brick lines; inspect "
                    "the text report for them")
            try:
                report["preview_written_to"] = str(write_preview(
                    args.preview, bricks,
                    title=report["request"]["caption"]))
            except PreviewError as exc:
                raise ShowcaseError(str(exc)) from exc
    except ShowcaseError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(format_report(report, show_prompt=args.prompt,
                            show_plan=not args.no_plan))
        if args.ldr:
            print(f"LDraw written to {report['ldraw_written_to']}")
        if args.preview:
            print(f"3-D preview written to {report['preview_written_to']}")

    verdict = passed(report)
    if verdict is None:
        return EXIT_UNDECIDED
    return EXIT_OK if verdict else EXIT_CHECK_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
