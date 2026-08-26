# BrickAgain — running the demonstration

```text
brick list  ->  every deterministic check  ->  LDraw + plan view + optional 3-D preview
```

**Read this first.** The demonstration measures nothing. No number it prints is
a metric, none of it is comparable to the frozen evaluation the project ran
once, and nothing here is evidence that any component helps. It exists to make
the pipeline legible and runnable, not to make a case.

**The CPU path does not generate.** It checks, exports and draws a brick list
that came from somewhere else. Only `--generate` runs a model, and it needs the
weights present locally.

---

## The one-line version

```bash
./.venv/bin/python scripts/26_showcase.py --sample tower
```

No model, no network, no GPU. It prints where the bricks came from, the brief,
the inventory, the brick list, every deterministic check, and a plan view.

List what ships with it:

```bash
./.venv/bin/python scripts/26_showcase.py --list
```

`--list` accepts no output or decode flags. For example, `--list --json`,
`--list --seed 7` and `--list --ldr FILE` are refused rather than silently
ignoring the extra setting.

---

## Three modes, and every report names its own

The modes are mutually exclusive. The provenance panel is printed first,
before any result, because where a brick list came from decides what the
checks below it are worth.

### `sample` — a stored brief, used whole

```bash
./.venv/bin/python scripts/26_showcase.py --sample collision
```

Four briefs ship with the code, chosen so each lights up a different failure:

| brief | what it shows |
|---|---|
| `tower` | every check passes: payable, in bounds, no overlap, one component |
| `overdrawn` | three bricks against a stock of two — the only failing check is the stock one, and the remaining count goes negative rather than clamping at zero |
| `collision` | two bricks claiming the same cells — the plan view marks them `*` and the check names the pair |
| `in-pieces` | two towers that never touch — legal bricks, nothing overlapping, and connectivity false |

**These are hand-written fixtures.** Not one came out of a model. They exercise
the checking, export and plan-view path; no result computed from one is
evidence about any model.

**They are used whole.** The caption, inventory, brick text and termination
belong together — a report on this text against a different inventory is a
report about something else. So `--caption`, `--inventory` and `--termination`
are *refused* here, not applied. To change one, use `--variant-of`.

Each fixture *states* its termination. That is the fixture declaring its own
shape, not a measurement, and the report says so.

### `variant-of` — the same text, your brief

```bash
./.venv/bin/python scripts/26_showcase.py \
  --variant-of tower --caption "my own brief" --inventory "2x4:1"
```

Takes a stored brief's brick text and nothing else. The report is a
`supplied_bricks` report labelled a variant, and it records which of the
fixture's fields you replaced — so it can never be read as the brief it
started from.

### `supplied_bricks` — brick text from anywhere

Any text in the brick grammar — paste a model's output, or write one by hand.
`-` reads standard input.

```bash
printf '2x4 (0,0,0)\n2x4 (0,0,1)\n' \
  | ./.venv/bin/python scripts/26_showcase.py \
      --bricks - --caption "a stack" --inventory "2x4:2" \
      --termination normal_eos
```

The inventory is entered by hand as `part:count`, comma separated. Either
spelling of a rotated part names the same stock, so `4x1:3` and `1x4:3` are the
same request — and giving both is refused rather than summed, because which of
the two counts was meant is not the parser's guess to make.

**`--termination` is never assumed.** Leave it out and the report records the
termination as *unavailable*, marks the two checks that read it `n/a`, and
exits 3. State it and the report records it as *operator-supplied* — your
claim, not a measurement. Either way nothing is called measured.

### `decoded` — a decode this process ran

```bash
./.venv/bin/python scripts/26_showcase.py \
  --generate --caption "a small chair" --inventory "2x4:10,2x2:6,1x2:8"
```

The only mode that loads a model, and the only one where the token count and
the termination are **measured**. It is **not run in the offline test suite**:
the suite covers the wiring, the refusals and everything downstream of the
brick list, and stops where weights would be required.

A decoded report carries the weights, the adapter path, the device, the seed,
the temperature, both budgets, and the gate configuration — a decoded result
nobody can re-run is a screenshot.

`--model project` (the default) reads `runs/project_model.json`, which is not
published; a public checkout does not have it and the refusal says so.
`--model published` uses the published BrickGPT weights instead.

The loading order is the project's one correct order — base, published adapter,
merge, then the local adapter — reached through the same two loaders the frozen
evaluation used, because a second spelling of that order is how a demonstration
ends up showing a model nobody trained. It still declares no arm identity,
writes no result row and produces no metric.

---

## Flags that do not apply are refused

Not ignored. A sampling temperature quietly dropped on a stored brief is a
report that looks like it honoured a setting nothing ever read.

| mode | accepts |
|---|---|
| `--sample` | nothing but the output flags |
| `--variant-of` | `--caption`, `--inventory`, `--termination` |
| `--bricks` | `--caption`, `--inventory`, `--termination` |
| `--generate` | `--caption`, `--inventory`, `--model`, `--device`, `--seed`, `--temperature`, `--max-bricks`, `--max-tokens`, `--placement`, `--connectivity` |

`--ldr` and `--json` apply everywhere. `--prompt` and `--no-plan` change the
printed report, so they apply wherever one is printed.

`--termination` is refused on `--generate` for the same reason `--placement` is
refused on `--bricks`: a decode measures its own termination, and supplied text
has no gate to describe.

**A flag that would change nothing is refused too**, on the same principle:
read and ignored leaves a report looking like it honoured a setting.

- `--connectivity` without `--placement` configures a gate that was not asked
  for. Refused even when it names the default, `off`.
- `--prompt` or `--no-plan` with `--json`. The JSON always carries the prompt
  and the plan view, so neither flag has anything to change.

---

## Reading the output

**Provenance.** Printed first: the mode, where the brick text came from, where
the caption and inventory came from, whether the token count was measured or
derived, and whether the termination was measured, stated by a fixture,
supplied by you, or unavailable.

**Checks.** Ten of them, then their conjunction, computed by
`src.eval.scoring.score_generation` — imported and called, not reimplemented.
A second copy of "what counts as a collision" is how a demonstration and an
evaluation come to disagree while both look right.

| check | what it means |
|---|---|
| `parse_success` | every line is a brick line, and the token count agrees with the brick count |
| `known_parts` | every part is one of the eight |
| `type_compliance` | no part was used that was never stocked |
| `inventory_valid` | no part was used that was never stocked, **and** nothing was overdrawn |
| `in_bounds` | every brick fits inside the 20×20×20 world |
| `collision_free` | no two bricks claim the same cell |
| `stud_only_connected` | one component under adjacent-layer footprint overlap |
| `touches_ground` | something sits at `z = 0` — generic showcase 只回報；最低交付層另把它列為必要靜態條件 |
| `ldraw_serializable` | the structure writes as LDraw |
| `termination_accepted` | the run ended on EOS or on running out of stock |

**`n/a` is not `FAIL`.** Two checks read the termination. With none available
they come back as `null` — in `--json` as well as in the printed report, so a
machine reader gets the same three answers a person does: passed, failed, or
nobody can say. The verdict is withheld and the exit code is 3. Writing `false`
there and a caveat elsewhere would hand every consumer a failure that never
happened, in the direction that happens to look cautious.

**Inventory.** Stocked, used and left, part by part. A negative remainder is
printed as negative and flagged `OVERDRAWN`.

**Plan view.** One grid per layer, lowest first, over the model's bounding box.
Each brick gets a symbol; a cell claimed twice prints `*`. It is not the 3-D
preview — it is the occupancy, drawn.

**LDraw.** `--ldr FILE` writes a real `.ldr` for an LDraw viewer, using the
exporter pinned against the BrickGPT reference vectors.

**3-D preview.** `--preview FILE.png` (or `.svg`) draws the parsed bricks as
axis-aligned cuboids with Matplotlib Agg on the CPU. Colliding bricks are
coloured magenta instead of being hidden. It is a geometric inspection aid,
not a photorealistic render and not a physics or stability analysis. Empty,
unparsed, unknown-part or out-of-bounds structures are refused. An invalid
preview suffix is refused before an accompanying LDraw is written. Long
captions are wrapped to at most two title lines with an ellipsis; the report
still retains the exact caption.

---

## What this does not show

- **No metric.** Nothing printed is comparable to the frozen evaluation. That
  comparison ran once, on 160 cases frozen in advance, and this demonstration
  does not touch them.
- **`--placement` is opt-in, unevaluated, and reachable only by decoding.** It
  can only be true on a decoded report, because the gate is a property of a
  decode that happened — claiming it over text that arrived some other way
  would be claiming a decode nobody ran. Its rules and implementation have
  passed review and its collision masking is exact by construction, and it has
  **never been formally evaluated**: the evaluation phase for it is not
  authorised, no metric has ever been computed with it on, and turning it on
  here says nothing about the success rate in either direction. The one
  relevant precedent points the other way: the inventory gate *lowered* the
  marginal `in_bounds` and `collision_free` rates in the frozen evaluation,
  because constraining one axis moves the others.
- **Connectivity is not support and not physics.** `stud_only_connected` is
  2-D footprint overlap between adjacent layers. It does not check centre of
  mass, moments, or whether a model stands up. The separate `unsupported`
  count is the scorer's own descriptive count of bricks with nothing directly
  beneath them, printed with the scorer's own words, and it is not a stability
  result either. Real analysis of that kind needs a solver this project does
  not have.
- **A derived token count checks nothing.** Text nothing decoded here has no
  measured token count, so one is derived from the grammar — which makes the
  token/brick agreement true by construction. On a decoded result the count is
  the one the loop actually spent, and the check is real. Passing a measured
  count in a supplied mode is refused, not relabelled.

---

## Exit codes

| code | meaning |
|---|---|
| 0 | ran, and every check passed |
| 1 | ran, and at least one check failed |
| 2 | refused — bad inventory, missing file, or a flag that does not apply |
| 3 | ran, but the verdict could not be decided (no termination available) |

A failing check is a result, not an error: `collision` is *supposed* to exit 1.
