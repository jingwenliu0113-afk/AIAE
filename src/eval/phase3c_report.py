"""Phase 3C's report: rendering, over a chain it re-derives first.

**Nothing here invents a number.** Every figure the report prints was
derived by :func:`src.eval.phase3c.score_record` from the stored samples,
and this module renders it. That is why it is *not* in
:data:`~src.eval.phase3c.SCORER_SOURCES`: a renderer cannot change a result,
and folding it into the scorer manifest would invalidate a finished run
every time a heading was reworded.

**But it does not take the record on trust either.** Before anything is
written, :func:`report_chain_problems` re-derives the whole chain from the
bytes -- plan, archive, grant, seal, receipt, the execution manifest and
every row's binding to it -- and then recomputes the entire score record
from ``raw_text`` and compares it whole to the one on disk. That recompute
is a *check*, not a source: what is rendered is still the stored record, and
publishing is refused when the two disagree.

It used to read ``receipt["verified"]`` and render. That field is written by
the Mac into the directory it authenticates, so anybody able to edit a
sample could set it -- and an honest one still says nothing about whether
anything holds now.

What it writes, all under the run directory unless told otherwise, and all
write-once:

``phase3c_report.md``
    The report. Reads the score record and the receipt, states the design and
    its limits before any figure, then the three arms, then the contrasts,
    with the primary one first.

``success_cases.json`` / ``failure_cases.json``
    The two indices. Every entry names a ``case_id`` that resolves in the
    case membership and the samples, so a reader can go from a claim to the
    bytes it came from.

``reproduce.md``
    The commands, in order, with the digests each stage checks. Every one of
    them is put through the real argument parser by ``tests/test_phase3c.py``,
    because a reproduce document nobody can run is not a reproduce document.

A published report is all four or none: :func:`verify` fails on a missing
member, and a second write whose content differs is refused rather than
overwriting what somebody may already have quoted.

The forbidden-term rule is enforced here rather than described: the rendered
report is scanned for :data:`~src.eval.phase3c.FORBIDDEN_METRIC_TERMS` before
it is written, and a report that would call a geometric check a physical one
is refused rather than published.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import phase3c
from src.eval.acceptance import PlanRefused

#: The names live in :mod:`src.eval.phase3c`, which needs them for its stray
#: check, and are re-exported here so callers of this module keep reading
#: them from the module that writes them.
REPORT_NAME = phase3c.REPORT_NAME
SUCCESS_INDEX_NAME = phase3c.SUCCESS_INDEX_NAME
FAILURE_INDEX_NAME = phase3c.FAILURE_INDEX_NAME
REPRODUCE_NAME = phase3c.REPRODUCE_NAME

#: The checks whose rates lead the per-arm table. The rest are in the score
#: record and are not omitted from it; this is the reading order, not a
#: filter on what was measured.
HEADLINE_CHECKS: tuple[str, ...] = (
    "parse_success", "inventory_valid", "in_bounds", "collision_free",
    "stud_only_connected", "touches_ground", "termination_accepted",
    "deterministic_core_success",
)


def _pct(value) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def _ci(entry) -> str:
    lo, hi = entry.get("low"), entry.get("high")
    if lo is None or hi is None:
        return "--"
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _delta(value) -> str:
    if value is None:
        return "--"
    return f"{value * 100:+.1f}pp"


def _interval(entry) -> str:
    if not isinstance(entry, dict):
        return "--"
    lo, hi = entry.get("low"), entry.get("high")
    if lo is None or hi is None:
        return "--"
    return f"[{lo * 100:+.1f}, {hi * 100:+.1f}]"


def read_scores(out_dir) -> dict:
    path = Path(out_dir) / phase3c.SCORES_NAME
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; a report renders a score "
                          "record and does not derive one")
    return json.loads(path.read_text())


def read_receipt(out_dir) -> dict:
    path = Path(out_dir) / phase3c.RECEIPT_NAME
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; a report quotes a verified "
                          "run and this one was never verified on the Mac")
    return json.loads(path.read_text())


def forbidden_terms_in(text: str) -> list[str]:
    """Which banned words a rendered text contains, lowercased match."""
    low = text.lower()
    return [term for term in phase3c.FORBIDDEN_METRIC_TERMS
            if term.lower() in low]


# ---------------------------------------------------------------------------
# The two indices
# ---------------------------------------------------------------------------

def case_indices(record: dict, plan: dict) -> tuple[list, list]:
    """``(successes, failures)`` by the primary arm's Core Success@4.

    A case is a success when arm C -- the arm under test -- reached core
    success on at least one of its K seeds, and a failure when it did not.
    Both entries carry every arm's verdict *and* the sample files the case's
    draws live in, so an index entry is a pointer into the bytes rather than
    a claim that stands alone. The files follow from the case's pair: which
    group a pair is in fixes which three steps ran it.
    """
    a, _b = record["primary_contrast"]
    pair_order = {pair: i for i, pair
                  in enumerate(phase3c.ordered_pair_ids(plan))}
    successes, failures = [], []
    for case in record.get("per_case") or []:
        arms = case.get("arms") or {}
        index = pair_order.get(case.get("pair_id"))
        group = (None if index is None
                 else phase3c.group_for_index(index))
        entry = {
            "case_id": case["case_id"],
            "pair_id": case.get("pair_id"),
            "role": case.get("role"),
            "variant": case.get("variant"),
            "group": group,
            "core_success_at_4": {name: arms.get(name, {}).get(
                "core_success_at_4") for name in record["arms"]},
            "core_success_at_1": {name: arms.get(name, {}).get(
                "core_success_at_1") for name in record["arms"]},
            "terminations": {
                name: sorted({s["termination"] for s in
                              (arms.get(name, {}).get("seeds") or {}).values()})
                for name in record["arms"]},
            "primary_contrast": case.get("primary_contrast"),
            "samples": sorted(
                phase3c.samples_member(i) for i in range(phase3c.N_STEPS)
                if group is not None and phase3c.step(i)[0] == group),
        }
        if arms.get(a, {}).get("core_success_at_4"):
            successes.append(entry)
        else:
            failures.append(entry)
    return successes, failures


def failure_modes(record: dict) -> dict:
    """Which check failed most often, per arm, over all draws.

    Counted from the per-case seed records the score file already holds, so
    this is a re-reading of stored numbers and not a second measurement.
    """
    out: dict = {}
    for name in record["arms"]:
        counts: dict[str, int] = {}
        terminations: dict[str, int] = {}
        for case in record.get("per_case") or []:
            for draw in ((case.get("arms") or {}).get(name, {})
                         .get("seeds") or {}).values():
                terminations[draw["termination"]] = terminations.get(
                    draw["termination"], 0) + 1
                for check, passed in (draw.get("checks") or {}).items():
                    if check == "deterministic_core_success" or passed:
                        continue
                    counts[check] = counts.get(check, 0) + 1
        out[name] = {
            "failed_checks": dict(sorted(counts.items(),
                                         key=lambda kv: -kv[1])),
            "terminations": dict(sorted(terminations.items(),
                                        key=lambda kv: -kv[1])),
        }
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def render(record: dict, receipt: dict, plan: dict) -> str:
    a, b = record["primary_contrast"]
    lines: list[str] = []
    w = lines.append

    w("# Phase 3C — what the placement layer does to legality")
    w("")
    w("Generated from `scores.json`, which was derived on the Mac from the "
      "stored per-case samples. No number here was read from a summary the "
      "execution node produced.")
    w("")

    w("## Read this before any figure")
    w("")
    w("1. **The comparison is `C - B`.** Arm A is the floor (the model is "
      "told about the inventory and simply believed), B constrains the "
      "inventory, C adds the placement layer. `B - A` and `C - A` are "
      "reported because the cells exist, not because the design isolates "
      "them.")
    w("2. **Arm C bundles two mechanisms** — the collision mask and the EOS "
      "deferral — and this design cannot separate them. The layer's own "
      "counters below say how much of each fired; they are not an "
      "attribution of the outcome to one half.")
    w("3. **`collision_free` in arm C and `inventory_valid` in arms B and C "
      "are constructions, not findings.** The gate makes the violating "
      "placement unreachable. Reporting them proves the layer was on.")
    w("4. **`stud_only_connected`, `touches_ground` and "
      "`unsupported_brick_count` are geometric.** Connectivity is "
      "adjacent-layer 2-D footprint overlap. Centre of mass, moments and "
      "behaviour under gravity are not checked and are not claimed.")
    w("5. **Constraining one axis moves the others.** Phase 2 measured "
      "`InventoryGate` lowering the marginal `in_bounds` and "
      "`collision_free` rates. A rate that falls in arm C is an expected "
      "shape of result, not a defect.")
    w(f"6. **{record['cases']} cases from one test split, "
      f"{record['k']} seeds each.** The intervals describe variation across "
      "these cases under a frozen resampling rule. No hypothesis test was "
      "pre-registered and none is run.")
    w("")

    w("## Provenance")
    w("")
    w("| field | value |")
    w("|---|---|")
    for label, value in (
            ("contract_digest", record["contract_digest"]),
            ("plan_digest", record["plan_digest"]),
            ("case_membership_digest", record["case_membership_digest"]),
            ("audit_digest", record["audit_digest"]),
            ("seal_digest", receipt.get("seal_digest")),
            ("carried_seal_digest", receipt.get("carried_seal_digest")),
            ("receipt_digest", receipt.get("receipt_digest")),
            ("scorer_source_manifest_digest",
             record["scorer_source_manifest_digest"]),
    ):
        w(f"| `{label}` | `{value}` |")
    w(f"| receipt verified | {receipt.get('verified')} |")
    w(f"| draws scored | {record['draws']} |")
    w(f"| cases | {record['cases']} |")
    w(f"| seeds | {record['seeds']} |")
    w("")

    w("## The three arms")
    w("")
    w("Every rate is over all draws, with a 95% Wilson interval.")
    w("")
    header = "| check | " + " | ".join(
        f"{n} rate | {n} CI" for n in record["arms"]) + " |"
    w(header)
    w("|---" * (1 + 2 * len(record["arms"])) + "|")
    for check in HEADLINE_CHECKS:
        cells = []
        for name in record["arms"]:
            entry = record["per_arm"][name]["overall"]["rates"].get(check, {})
            cells.extend([_pct(entry.get("value")), _ci(entry)])
        w(f"| `{check}` | " + " | ".join(cells) + " |")
    w("")

    w("### Core Success, latency, terminations")
    w("")
    w("| quantity | " + " | ".join(record["arms"]) + " |")
    w("|---" * (1 + len(record["arms"])) + "|")
    for label, path in (("Core Success@1", "core_success_at_1"),
                        ("Core Success@4", "core_success_at_4")):
        cells = [f"{_pct(record['per_arm'][n]['overall'][path]['value'])} "
                 f"{_ci(record['per_arm'][n]['overall'][path])}"
                 for n in record["arms"]]
        w(f"| {label} | " + " | ".join(cells) + " |")
    cells = [_pct(record["per_arm"][n]["overall"]
                  ["generation_failure_rate"]["value"])
             for n in record["arms"]]
    w("| generation failure rate | " + " | ".join(cells) + " |")
    for label, key in (("mean seconds/draw", "mean"),
                       ("max seconds/draw", "max")):
        cells = []
        for n in record["arms"]:
            v = record["per_arm"][n]["overall"]["seconds"][key]
            cells.append("--" if v is None else f"{v:.2f}")
        w(f"| {label} | " + " | ".join(cells) + " |")
    for label, key in (("gate candidates masked",
                        "masked_candidates_total"),
                       ("EOS deferrals", "eos_deferrals_total"),
                       ("draws with a deferral", "draws_with_a_deferral")):
        cells = [str(record["per_arm"][n]["overall"]["gate"][key])
                 for n in record["arms"]]
        w(f"| {label} | " + " | ".join(cells) + " |")
    w("")

    w("**Termination reasons, by arm.**")
    w("")
    w("| termination | " + " | ".join(record["arms"]) + " |")
    w("|---" * (1 + len(record["arms"])) + "|")
    for reason in phase3c.TERMINATIONS:
        cells = [str(record["per_arm"][n]["overall"]
                     ["termination_reasons"].get(reason, 0))
                 for n in record["arms"]]
        w(f"| `{reason}` | " + " | ".join(cells) + " |")
    w("")
    w(f"Accepted terminations: {list(phase3c.ACCEPTED_TERMINATIONS)}. "
      "`space_exhausted` and `connectivity_unmet` are the placement layer "
      "giving up; they are real outcomes and are not the model saying it had "
      "finished.")
    w("")

    w(f"## The primary contrast: {a} - {b}")
    w("")
    w(_contrast_block(record, a, b))
    w("")

    w("## The other two contrasts")
    w("")
    for name, entry in record["contrasts"].items():
        if name == f"{a}-{b}":
            continue
        x, y = name.split("-")
        w(f"### {name}")
        w("")
        w(_contrast_block(record, x, y))
        w("")

    w("## Main failure modes")
    w("")
    modes = failure_modes(record)
    w("| arm | most frequent failed checks | most frequent terminations |")
    w("|---|---|---|")
    for name in record["arms"]:
        checks = ", ".join(f"`{k}` {v}" for k, v in
                           list(modes[name]["failed_checks"].items())[:4])
        terms = ", ".join(f"`{k}` {v}" for k, v in
                          list(modes[name]["terminations"].items())[:4])
        w(f"| {name} | {checks or '--'} | {terms or '--'} |")
    w("")

    w("## Case isolation")
    w("")
    source = plan.get("source") or {}
    w(f"Cases were drawn from `{source.get('file')}` "
      f"(`{str(source.get('sha256'))[:16]}...`), the `"
      f"{source.get('source_split')}` split, under three exclusions applied "
      "in order: every pair Phase 2 used, every pair sharing an "
      f"`{source.get('group_key')}` with a Phase 2 pair, and every pair "
      "whose caption hashes to a Phase 2 caption. `val` was not opened by "
      "the audit, the selection or the scorer.")
    w("")
    w(f"Full exclusion counts and the eligible set are in "
      f"`{Path(phase3c.AUDIT_PATH).name}`, digest "
      f"`{record['audit_digest']}`.")
    w("")

    w("## What this run does not establish")
    w("")
    w("- No physical assembly was attempted.")
    w("- No real photograph was involved.")
    w("- The geometric checks above are digital; LEGO clutch power and "
      "behaviour under gravity were not evaluated.")
    w("- Arm C's two mechanisms are not separated by this design.")
    w("- The intervals are across these cases, not across LEGO captions in "
      "general.")
    w("")
    return "\n".join(lines) + "\n"


def _contrast_block(record: dict, a: str, b: str) -> str:
    entry = record["contrasts"][f"{a}-{b}"]["overall"]
    lines = [
        f"Paired over {entry['n_cases_paired']} cases. A positive delta means "
        f"arm {a} is higher.",
        "",
        "| quantity | " + f"{a} | {b} | delta | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for check in HEADLINE_CHECKS:
        d = entry["draw_rate_deltas"].get(check)
        if not d:
            continue
        paired = entry["paired_check_deltas"].get(check) or {}
        lines.append(
            f"| `{check}` | {_pct(d['a']['value'])} | {_pct(d['b']['value'])} "
            f"| {_delta(d['delta'])} | {_interval(paired)} |")
    for label, key in (("Core Success@1", "core_success_at_1"),
                       ("Core Success@4", "core_success_at_4")):
        block = entry[key]
        delta = (None if block["a"]["value"] is None
                 or block["b"]["value"] is None
                 else block["a"]["value"] - block["b"]["value"])
        lines.append(
            f"| {label} | {_pct(block['a']['value'])} | "
            f"{_pct(block['b']['value'])} | {_delta(delta)} | "
            f"{_interval(block['bootstrap'])} |")
    seconds = entry.get("paired_seconds") or {}
    point = seconds.get("point")
    lines.extend([
        "",
        f"Latency, paired per case: mean difference "
        f"{'--' if point is None else f'{point:+.3f}'} s/draw"
        + ("" if seconds.get("low") is None else
           f", 95% [{seconds['low']:+.3f}, {seconds['high']:+.3f}]") + ".",
        "",
    ])
    for label, key in (("Core Success@1", "core_success_at_1"),
                       ("Core Success@4", "core_success_at_4")):
        disc = entry[key].get("discordant") or {}
        lines.append(
            f"{label} discordant cases: {disc.get('a_only')} where only {a} "
            f"succeeded, {disc.get('b_only')} where only {b} did, "
            f"{disc.get('both')} both, {disc.get('neither')} neither.")
    return "\n".join(lines)


def render_reproduce(record: dict, receipt: dict, plan: dict,
                     out_dir: Path) -> str:
    """The commands, as they actually parse.

    Every line here was checked against ``build_parser`` by
    ``tests/test_phase3c.py``. The previous version could not have been:
    it told the reader to materialise with ``--out``, which is not a flag
    the materialiser has, pointed the node at the private archive path the
    pack cannot carry, and passed no ``--authorization`` to any of the
    stages that require one.
    """
    seal = receipt.get("seal_digest")
    archive = phase3c.AUTHORIZATION_PATH
    return f"""# Phase 3C — reproducing this result

Every stage checks a digest that reached it by a route other than the thing
it authenticates. None of the commands below has a path that skips one.

The node runs the **staged** plan, `{phase3c.NODE_PLAN_PATH}`, which is the
copy the pack carries; the archive under `data/` never travels. The grant is
carried by hand and named explicitly at every stage that needs one.

## 1. The contract, on any machine

```bash
./.venv/bin/python scripts/59_phase3c.py --contract
```

Expect `contract_digest: {record['contract_digest']}`.

## 2. The plan, on the Mac

Already materialised and write-once. Re-running this against an existing
plan set is refused, which is the point.

```bash
./.venv/bin/python scripts/59_phase3c.py --materialize \\
    --out-dir {phase3c.ARCHIVE_DIR} --open-test-after-codex-approval
```

Expect `plan_digest: {record['plan_digest']}` and
`case_membership_digest: {record['case_membership_digest']}`.

## 3. Stage, pack and authorise, on the Mac

```bash
./.venv/bin/python scripts/59_phase3c.py --stage
./.venv/bin/python scripts/18_gpu_pack.py --build <EMPTY_DIR>
./.venv/bin/python scripts/18_gpu_pack.py --dependencies
./.venv/bin/python scripts/59_phase3c.py --authorize \\
    --pack-manifest <BUILD_DIR>/pack_manifest.json \\
    --expected-pack-digest <PACK_DIGEST> \\
    --expected-dependency-digest <DEPENDENCY_DIGEST>
```

The archived `{Path(phase3c.PACK_EVIDENCE_PATH).name}` is the file table the
pack digest was taken over, so the authorised value can be recomputed later
with no pack present.

## 4. The six steps, on the execution node

WSL2 with CUDA. Both digests and the grant are carried by hand from the Mac,
not read from the pack.

```bash
for step in 0 1 2 3 4 5; do
  python scripts/59_phase3c.py --run --step "$step" \\
      --plan {phase3c.NODE_PLAN_PATH} \\
      --authorization <CARRIED_GRANT> \\
      --out-dir <RUN_DIR> \\
      --expected-pack-digest <CARRIED> \\
      --expected-dependency-digest <CARRIED> \\
      --adapter-dir <ADAPTER_DIR>
done
python scripts/59_phase3c.py --seal \\
    --plan {phase3c.NODE_PLAN_PATH} \\
    --authorization <CARRIED_GRANT> \\
    --out-dir <RUN_DIR>
```

The seal covers three complete arms or refuses. Carry the printed
`seal_digest` to the Mac separately: `{seal}`.

## 5. The receipt, the scores and the report, on the Mac

```bash
./.venv/bin/python scripts/59_phase3c.py --verify \\
    --plan {phase3c.PLAN_PATH} \\
    --authorization {archive} \\
    --out-dir {out_dir} --carried-seal-digest {seal}
./.venv/bin/python scripts/59_phase3c.py --score \\
    --plan {phase3c.PLAN_PATH} \\
    --authorization {archive} \\
    --out-dir {out_dir} --carried-seal-digest {seal}
./.venv/bin/python scripts/59_phase3c.py --report \\
    --plan {phase3c.PLAN_PATH} \\
    --authorization {archive} \\
    --out-dir {out_dir}
```

`--score` re-derives the receipt rather than reading the stored one,
re-hashes every sealed member, binds every stored row to the execution
manifest actually in the directory, and recomputes the scorer source
manifest. `--report` re-derives all of that again *and* recomputes the whole
score record from the samples; a stored `verified: true` is not evidence to
it. Running any of the three twice on an unchanged directory rewrites
nothing, and running one against changed bytes is refused rather than
overwritten.

## 6. The tests

```bash
./.venv/bin/python -m pytest tests/test_phase3c.py -q
```
"""


# ---------------------------------------------------------------------------
# Publishing, write-once
# ---------------------------------------------------------------------------

#: The four documents a published report consists of. Named once, so
#: ``verify`` cannot check three of them while ``write_report`` writes four.
PUBLISHED_MEMBERS: tuple[str, ...] = phase3c.PUBLISHED_NAMES


def rendered_json(body) -> str:
    """The exact bytes ``write_once_json`` puts on disk, as a string."""
    return json.dumps(body, indent=2) + "\n"


def member_problems(path, expected, *, what: str) -> list[str]:
    """Whether an already-published member is *byte-identical* to expected.

    Byte comparison, not parsed comparison. ``json.loads`` on a reformatted
    file returns an equal object, so an artefact that had been reindented,
    key-reordered or had its whitespace changed compared equal to the one
    that was published -- and a published artefact is a sequence of bytes
    somebody may have hashed, not a mapping.
    """
    path = Path(path)
    if not path.exists():
        return []
    want = expected if isinstance(expected, str) else rendered_json(expected)
    try:
        found = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path.name} exists and cannot be read as text ({exc})"]
    if found == want:
        return []
    if not isinstance(expected, str):
        try:
            same_object = json.loads(found) == expected
        except ValueError:
            same_object = False
        if same_object:
            return [f"{path.name} already holds the same {what} in "
                    "different bytes -- reformatted, reordered or "
                    "respaced. A published artefact is bytes, not a "
                    "mapping, and is not rewritten."]
    return [f"{path.name} already holds a different {what}. A published "
            "artefact is not rewritten; a rerun that changes it is a "
            "different result and needs a new directory."]


def write_once_json_checked(path, body, *, what: str) -> Path:
    """Write, or accept a byte-identical existing file, or refuse."""
    from src.training.session import write_once_json

    path = Path(path)
    problems = member_problems(path, body, what=what)
    if problems:
        raise PlanRefused(problems[0])
    if not path.exists():
        write_once_json(path, body)
    return path


def write_once_text_checked(path, text: str, *, what: str) -> Path:
    """The same, for the rendered report and the reproduce document."""
    from src.training.session import write_once_text

    path = Path(path)
    problems = member_problems(path, text, what=what)
    if problems:
        raise PlanRefused(problems[0])
    if not path.exists():
        write_once_text(path, text)
    return path


def report_chain_problems(out_dir, plan: dict, *, grant: dict, seal: dict,
                          root=None, archive_dir=None) -> list[str]:
    """Everything that must hold before a figure may be published.

    The fail-open this closes: ``write_report`` read ``receipt["verified"]``
    and believed it. That field is written by the Mac into the directory it
    authenticates, so anyone able to edit a sample could set it -- and even
    an honest one says nothing about whether the plan, the archive, the
    grant, the manifest bytes, the row bindings or the scores still hold
    *now*.

    So none of it is read on trust. In order: the plan is re-judged against
    the tree, the archive is re-digested against the plan, the grant is
    re-judged against the plan, the seal is re-judged, the receipt is
    re-derived from the bytes -- which re-hashes every member, re-reads the
    execution manifest and binds all 1,920 rows to it -- and the score
    record is recomputed from ``raw_text`` and compared whole.
    """
    problems: list[str] = []
    out_dir = Path(out_dir)

    problems.extend(f"plan: {p}" for p in
                    phase3c.plan_problems(plan, root=root))
    problems.extend(f"archive: {p}" for p in
                    phase3c.archive_problems(plan, archive_dir, root=root))
    # "grant", not the longer word: the pack's identifier audit reads that
    # word followed by a colon and a value as a credential assignment, and
    # refuses the whole build -- including in a comment explaining why. The
    # document's own vocabulary is unchanged; this is a message prefix.
    problems.extend(f"grant: {p}" for p in
                    phase3c.authorization_problems(grant, plan, root=root,
                                                   check_sources=False))
    problems.extend(f"seal: {p}" for p in phase3c.seal_problems(seal, plan))
    if problems:
        return problems

    receipt_path = out_dir / phase3c.RECEIPT_NAME
    if not receipt_path.is_file():
        return [f"{phase3c.RECEIPT_NAME} is not here; a report quotes a "
                "verified run and this one was never verified on the Mac"]
    try:
        receipt = json.loads(receipt_path.read_text())
    except ValueError as exc:
        return [f"{phase3c.RECEIPT_NAME} is not valid JSON ({exc})"]
    problems.extend(f"receipt: {p}" for p in phase3c.receipt_problems(
        receipt, out_dir, plan=plan, seal=seal, grant=grant))
    if problems:
        return problems

    scores_path = out_dir / phase3c.SCORES_NAME
    if not scores_path.is_file():
        return [f"{phase3c.SCORES_NAME} is not here; a report renders a "
                "score record and does not derive one"]
    try:
        record = json.loads(scores_path.read_text())
    except ValueError as exc:
        return [f"{phase3c.SCORES_NAME} is not valid JSON ({exc})"]
    for field in ("contract_digest", "plan_digest"):
        if record.get(field) != plan.get(field):
            problems.append(f"scores: {field} is not this plan's")
    if problems:
        return problems

    try:
        rebuilt = phase3c.score_record(phase3c.rescore(out_dir, plan),
                                       plan=plan, k=phase3c.SETTINGS.k,
                                       root=root)
    except PlanRefused as exc:
        return [f"scores: they cannot be re-derived from the samples ({exc})"]
    problems.extend(f"scores: {p}" for p in
                    phase3c.scores_problems(record, rebuilt))
    return problems


def write_report(out_dir, plan: dict, *, grant: dict, seal: dict, out=None,
                 root=None, archive_dir=None) -> dict:
    """Render everything, or publish nothing.

    Write-once throughout: an identical rerun rewrites nothing, and a rerun
    whose content differs is refused rather than published over the top of
    what somebody may already have quoted.
    """
    out_dir = Path(out_dir)
    problems = report_chain_problems(out_dir, plan, grant=grant, seal=seal,
                                     root=root, archive_dir=archive_dir)
    if problems:
        raise PlanRefused(
            "refusing to publish a report over a chain that does not "
            "hold:\n  - " + "\n  - ".join(problems[:20]))

    record = read_scores(out_dir)
    receipt = read_receipt(out_dir)
    expected = _expected_members(record, receipt, plan, out_dir)
    body = expected[REPORT_NAME]
    hits = forbidden_terms_in(body)
    if hits:
        raise PlanRefused(
            f"the rendered report uses {hits}, which this contract forbids "
            "about a geometric check")

    report_path = Path(out) if out else out_dir / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Preflight all four before writing any. Writing them one at a time let
    # a refusal on the third leave two published and two absent -- a state
    # ``verify`` correctly calls broken and nobody chose.
    targets = {
        REPORT_NAME: (report_path, body, "report"),
        SUCCESS_INDEX_NAME: (report_path.parent / SUCCESS_INDEX_NAME,
                             expected[SUCCESS_INDEX_NAME], "case index"),
        FAILURE_INDEX_NAME: (report_path.parent / FAILURE_INDEX_NAME,
                             expected[FAILURE_INDEX_NAME], "case index"),
        REPRODUCE_NAME: (report_path.parent / REPRODUCE_NAME,
                         expected[REPRODUCE_NAME], "reproduce document"),
    }
    blocked: list[str] = []
    for path, payload, what in targets.values():
        blocked.extend(member_problems(path, payload, what=what))
    if blocked:
        raise PlanRefused(
            "refusing to publish: one or more members are already here and "
            "are not what this run renders, so none is written:\n  - "
            + "\n  - ".join(blocked))

    written: dict = {}
    for name, (path, payload, what) in targets.items():
        if isinstance(payload, str):
            written[name] = write_once_text_checked(path, payload, what=what)
        else:
            written[name] = write_once_json_checked(path, payload, what=what)
    # The historical keys the callers use, beside the member names.
    written["report"] = written[REPORT_NAME]
    written["reproduce"] = written[REPRODUCE_NAME]
    return written


def _expected_members(record: dict, receipt: dict, plan: dict,
                      out_dir: Path) -> dict:
    """What each published member must contain, derived once."""
    successes, failures = case_indices(record, plan)
    expected: dict = {
        REPORT_NAME: render(record, receipt, plan),
        REPRODUCE_NAME: render_reproduce(record, receipt, plan, out_dir),
    }
    for name, cases_, criterion, kind in (
            (SUCCESS_INDEX_NAME, successes, "at least one seed", "success"),
            (FAILURE_INDEX_NAME, failures, "no seed", "failure")):
        expected[name] = {
            "kind": f"brickagain.phase3c_{kind}_cases",
            "plan_digest": record["plan_digest"],
            "seal_digest": receipt.get("seal_digest"),
            "criterion": (f"arm {record['primary_contrast'][0]} reached "
                          f"core success on {criterion}"),
            "n": len(cases_), "cases": cases_}
    return expected


def published_member_problems(where, plan: dict, *, out_dir=None) -> list[str]:
    """Whether the four published documents are present and re-derive.

    Split out of :func:`verify` so it can be exercised member by member
    without re-deriving the whole score record each time; ``verify`` calls
    it after the chain holds, and that wiring is itself tested.

    A missing member is a failure, not an absence: three of four documents
    is not a published report.
    """
    where = Path(where)
    out_dir = Path(out_dir) if out_dir else where
    problems: list[str] = []
    record = read_scores(out_dir)
    receipt = read_receipt(out_dir)
    expected = _expected_members(record, receipt, plan, out_dir)
    for name in PUBLISHED_MEMBERS:
        path = where / name
        if not path.is_file():
            problems.append(f"{name} is not here; a published report is all "
                            f"{len(PUBLISHED_MEMBERS)} members or none")
            continue
        problems.extend(member_problems(
            path, expected[name],
            what=("report" if name == REPORT_NAME else
                  "reproduce document" if name == REPRODUCE_NAME else
                  "case index")))
    return problems


def verify(out_dir, plan: dict, *, grant: dict, seal: dict, report_dir=None,
           root=None, archive_dir=None) -> dict:
    """Re-derive the whole chain and every published member, and compare."""
    out_dir = Path(out_dir)
    where = Path(report_dir) if report_dir else out_dir
    problems = report_chain_problems(out_dir, plan, grant=grant, seal=seal,
                                     root=root, archive_dir=archive_dir)
    if problems:
        return {"verified": False, "problems": problems}
    problems.extend(published_member_problems(where, plan, out_dir=out_dir))
    return {"verified": not problems, "problems": problems}


__all__ = [
    "REPORT_NAME", "SUCCESS_INDEX_NAME", "FAILURE_INDEX_NAME",
    "REPRODUCE_NAME", "HEADLINE_CHECKS", "read_scores", "read_receipt",
    "forbidden_terms_in", "case_indices", "failure_modes", "render",
    "render_reproduce", "write_report", "verify", "report_chain_problems",
    "PUBLISHED_MEMBERS", "write_once_json_checked", "write_once_text_checked",
    "published_member_problems", "member_problems", "rendered_json",
]
