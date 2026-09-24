#!/usr/bin/env python3
"""Regenerate every figure in `reports/figures/` from the published reports.

    ./.venv/bin/python scripts/73_figures.py            # draw, and write figures.json
    ./.venv/bin/python scripts/73_figures.py --check    # are the committed ones current?

Each figure is a view of a JSON report already in this repository, so nothing
here is a number typed by hand: change the report, rerun this, and the figure
follows. The report path each figure came from is printed into the figure's
own caption, so a reader looking at a PNG can find the file behind it.

Figures are written deterministically -- fixed metadata, no timestamps -- so a
rebuild that changes nothing produces the same bytes *on the machine that drew
them*. Not on another platform: on ubuntu-latest a rebuild of unchanged
reports came out different in all four PNGs (CI run 36012796552, 2026-09-24)
although matplotlib is pinned. So "is this figure current?" cannot be asked by
drawing again and comparing bytes, which is what CI used to do.

It is asked of ``figures.json`` instead, written beside the PNGs on every
draw: the SHA-256 of this script, of every report each figure actually read,
and of each PNG as drawn. ``--check`` recomputes the first two from the tree
and the third from the committed PNGs, draws nothing, and so gives the same
answer on any machine. A report that moved without a redraw, a script edited
without one, and a PNG that is not the one the manifest describes all fail.
Where the PNG digests were written -- ``drawn_with`` names the platform --
``tests/test_figures.py`` draws again and holds them to the bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
OUT = ROOT / "reports" / "figures"

INK = "#17191F"
ACCENT = "#C8461E"
MUTED = "#5E6672"
SOFT = "#C3CAD3"

# Matplotlib stamps a creation date into PNG metadata by default, which makes
# an unchanged rebuild produce different bytes.
SAVE = {"dpi": 150, "bbox_inches": "tight", "metadata": {"Software": None}}

#: Every figure this script writes. `tests/test_public_snapshot.py` reads this
#: to decide which PNGs may be published: the boundary forbids images in
#: general, and the exception is "a chart this script drew from a committed
#: report", not "anything under reports/figures/".
FIGURES = (
    "01_part_distribution.png",
    "02_variant_breakdown.png",
    "03_retile_feasibility.png",
    "04_stagger_ablation.png",
)

#: Written beside the PNGs on every draw; what ``--check`` reads.
MANIFEST = "figures.json"

#: Repository-relative paths of the reports the figure being drawn has read.
#: Filled by :func:`load` rather than declared, so the manifest records what
#: the code opened and not what somebody remembered it opens.
_READ: set[str] = set()


def load(name: str) -> dict:
    path = REPORTS / name
    _READ.add(path.relative_to(ROOT).as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(ax, title: str, source: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom")
    ax.text(0, -0.16, f"source: {source}", transform=ax.transAxes,
            fontsize=7.5, color=SOFT, va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SOFT)
    ax.tick_params(colors=MUTED, labelsize=9)


def part_distribution(out: Path) -> str:
    eda = load("01_eda.json")
    parts = dict(sorted(eda["canonical_parts"].items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(list(parts), [v / 1e6 for v in parts.values()], color=ACCENT, width=0.62)
    ax.set_ylabel("bricks placed (millions)", fontsize=9.5, color=MUTED)
    frame(ax, "Eight canonical part types carry 5.1M placements",
          "data/reports/01_eda.json",
          f"{len(eda['raw_spellings'])} raw spellings normalise to "
          f"{len(parts)} inventory items; 1x2 and 2x1 are the same part")
    fig.savefig(out / "01_part_distribution.png", **SAVE)
    plt.close(fig)
    return "01_part_distribution.png"


def variant_breakdown(out: Path) -> str:
    eda = load("01_eda.json")
    rows = [("identical inventory", eda["variants_identical_inventory"]),
            ("differ in counts only", eda["variants_differ_counts_only"]),
            ("differ in part types", eda["variants_differ_types"])]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [SOFT, MUTED, ACCENT]
    ax.barh(labels[::-1], values[::-1], color=colours[::-1], height=0.55)
    for n, value in enumerate(values[::-1]):
        ax.text(value + max(values) * 0.015, n, f"{value:,}", va="center",
                fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("objects", fontsize=9.5, color=MUTED)
    frame(ax, "Why counterfactual re-tiling had to be generated",
          "data/reports/01_eda.json",
          f"of {eda['multi_structure_objects']:,} objects with more than one "
          f"variant, only {eda['variants_differ_types']:,} use different parts")
    fig.savefig(out / "02_variant_breakdown.png", **SAVE)
    plt.close(fig)
    return "02_variant_breakdown.png"


def retile_feasibility(out: Path) -> str:
    retile = load("02_retile.json")
    by_part = retile["by_part"]
    order = sorted(by_part, key=lambda p: -by_part[p]["feasible_rate"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(order, [by_part[p]["feasible_rate"] * 100 for p in order],
           color=ACCENT, width=0.62)
    ax.axhline(retile["overall_feasible"] * 100, color=MUTED, linewidth=1,
               linestyle="--")
    ax.text(len(order) - 0.4, retile["overall_feasible"] * 100 + 2,
            f"overall {retile['overall_feasible'] * 100:.1f}%", ha="right",
            fontsize=9, color=MUTED)
    ax.set_ylabel("feasible re-tilings (%)", fontsize=9.5, color=MUTED)
    frame(ax, "CP-SAT feasibility by the part the inventory is restricted to",
          "data/reports/02_retile.json",
          f"{retile['n_structures']} structures, {retile['time_limit']:.0f}s "
          f"limit, {retile['verify_failures']} verification failures")
    fig.savefig(out / "03_retile_feasibility.png", **SAVE)
    plt.close(fig)
    return "03_retile_feasibility.png"


def stagger_ablation(out: Path) -> str:
    results = load("09_stagger_ablation.json")["results"]
    arms = list(results)
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    left.bar(arms, [results[a]["rate"] * 100 for a in arms], color=ACCENT,
             width=0.55)
    left.set_ylabel("stud-connected (%)", fontsize=9.5, color=MUTED)
    frame(left, "Connectivity", "data/reports/09_stagger_ablation.json")
    right.bar(arms, [results[a]["seconds"] for a in arms], color=SOFT, width=0.55)
    right.set_yscale("log")
    right.set_ylabel("solve time (s, log)", fontsize=9.5, color=MUTED)
    frame(right, "Cost", "data/reports/09_stagger_ablation.json")
    for axis in (left, right):
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(arms, fontsize=8.5, rotation=12, ha="right")
    fig.suptitle("Staggering cost 140x the solve time and lost connectivity",
                 x=0.09, ha="left", fontsize=13, color=INK)
    fig.subplots_adjust(top=0.80, wspace=0.35)
    fig.savefig(out / "04_stagger_ablation.png", **SAVE)
    plt.close(fig)
    return "04_stagger_ablation.png"


BUILDS = (part_distribution, variant_breakdown, retile_feasibility,
          stagger_ablation)


def draw(out: Path = OUT) -> dict:
    """Draw every figure into ``out`` and write the manifest beside them."""
    out.mkdir(parents=True, exist_ok=True)
    figures = {}
    for build in BUILDS:
        _READ.clear()
        name = build(out)
        figures[name] = {
            "sources": {rel: _sha256(ROOT / rel) for rel in sorted(_READ)},
            "png_sha256": _sha256(out / name),
        }
        print(f"  {name}  {(out / name).stat().st_size:,} bytes")
    manifest = {
        "kind": "brickagain.figures",
        "note": ("Written by scripts/73_figures.py beside the PNGs. --check "
                 "compares it to the script, the reports and the committed "
                 "PNGs without drawing, so it answers the same on any "
                 "machine; PNG bytes are only reproducible where drawn_with "
                 "says they were drawn."),
        "generator": {"path": "scripts/73_figures.py",
                      "sha256": _sha256(Path(__file__).resolve())},
        "drawn_with": {"matplotlib": matplotlib.__version__,
                       "platform": f"{platform.system()} {platform.machine()}"},
        "figures": figures,
    }
    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def check(root: Path = ROOT) -> list[str]:
    """Why the committed figures are not current, or ``[]``. Draws nothing."""
    figures_dir = root / "reports" / "figures"
    path = figures_dir / MANIFEST
    if not path.is_file():
        return [f"reports/figures/{MANIFEST} does not exist"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    script = root / "scripts" / "73_figures.py"
    if manifest.get("generator", {}).get("sha256") != _sha256(script):
        problems.append("scripts/73_figures.py changed since the figures were drawn")
    recorded = manifest.get("figures") or {}
    if sorted(recorded) != sorted(FIGURES):
        problems.append(f"{MANIFEST} names {sorted(recorded)}, "
                        f"the script draws {sorted(FIGURES)}")
    for name, entry in sorted(recorded.items()):
        sources = entry.get("sources") or {}
        if not sources:
            problems.append(f"{name}: no source report is recorded")
        for rel, digest in sorted(sources.items()):
            if not (root / rel).is_file():
                problems.append(f"{name}: {rel} is gone")
            elif _sha256(root / rel) != digest:
                problems.append(f"{name}: {rel} changed since it was drawn")
        png = figures_dir / name
        if not png.is_file():
            problems.append(f"{name} is missing")
        elif _sha256(png) != entry.get("png_sha256"):
            problems.append(f"{name} is not the PNG {MANIFEST} describes")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report whether the committed figures are current, "
                             "without drawing")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(ROOT)
        for problem in problems:
            print(f"STALE: {problem}")
        if problems:
            print(f"\n{len(problems)} problem(s): run `make figures` and "
                  "commit reports/figures/", file=sys.stderr)
            return 1
        print(f"figures are current: {len(FIGURES)} figures, the generator and "
              "every report they read unchanged")
        return 0
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
