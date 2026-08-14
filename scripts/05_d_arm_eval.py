"""D arm: generation under a hard inventory gate, multi-seed.

Records for every run: environment, seed, caption, starting inventory, raw
output, termination reason, and parse / type / count compliance, plus a
grammar audit of every sampled token against the slot that produced it.

The prompt carries the same ``### Available Parts`` block the gate enforces:
this arm is the conditioned prompt *plus* the hard gate, so measuring it
against the unconditioned prompt would be measuring a configuration no arm
uses. Both come from one snapshot, via ``generate_with_inventory``.

This is a compliance measurement of the constraint layer on the *base*
checkpoint. It is not an A/B/C/D/E comparison -- those need a fixed prompt
set and matched physics settings across arms.

Writes data/reports/05_d_arm.md and .json, and .ldr files under artifacts/.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints.inventory_decode import (  # noqa: E402
    InventoryGate,
    generate_with_inventory,
)
from src.data.bricks import (  # noqa: E402
    connected_components,
    find_collisions,
    is_connected,
    required_inventory,
    unsupported_bricks,
)
from src.generation.brickgpt import TOKENS_PER_BRICK, BrickGPT  # noqa: E402
from src.inventory.engine import Inventory  # noqa: E402
from src.rendering.ldr import write_ldr  # noqa: E402

REPORT_DIR = ROOT / "data" / "reports"
LDR_DIR = ROOT / "artifacts" / "ldraw" / "d_arm"
SEEDS = range(8)
TEMPERATURE = 0.6
MAX_BRICKS = 40

CASES = [
    ("A simple chair.", {"2x4": 10, "1x2": 8, "2x2": 6}),
    ("A small car.", {"1x1": 3, "2x6": 4, "1x4": 2}),
    ("A table.", {"2x2": 20, "2x4": 20, "1x8": 10, "1x2": 20}),
]


class AuditGate(InventoryGate):
    """Inventory gate that also checks every sampled token against its slot."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tokens = 0
        self.violations: list[tuple[int, int]] = []

    def allowed(self, slot, out):
        if out:
            prev_slot = (len(out) - 1) % TOKENS_PER_BRICK
            legal = set(self._legal(prev_slot, out[:-1]))
            self.tokens += 1
            if out[-1] not in legal:
                self.violations.append((prev_slot, out[-1]))
        return super().allowed(slot, out)

    def _legal(self, slot, out):
        s = self.slots
        return {
            0: s.dims + [s.eos], 1: [s.literal_x], 2: s.dims, 3: [s.literal_open],
            4: s.posns, 5: [s.literal_comma], 6: s.posns, 7: [s.literal_comma],
            8: s.posns, 9: [s.literal_close],
        }[slot]


def main() -> None:
    gpt = BrickGPT()
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": gpt.device,
        "dtype": str(next(gpt.model.parameters()).dtype),
        "base": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter": "AvaLovelace/BrickGPT",
        "temperature": TEMPERATURE,
        "max_bricks": MAX_BRICKS,
        "prompt": "inventory-conditioned (the B-E form), matching the gate",
    }
    LDR_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    t0 = time.time()
    for caption, budget in CASES:
        for seed in SEEDS:
            inv = Inventory.from_parts(budget)
            gen, gate = generate_with_inventory(
                gpt, caption, inv, gate_cls=AuditGate, max_bricks=MAX_BRICKS,
                temperature=TEMPERATURE, seed=seed)
            used = required_inventory(gen.bricks)
            type_viol = sorted(p for p in used if p not in budget)
            count_viol = {p: [used[p], budget.get(p, 0)]
                          for p in used if used[p] > budget.get(p, 0)}
            name = f"{caption[:10].strip().replace(' ', '_')}_s{seed}"
            write_ldr(LDR_DIR / f"{name}.ldr", gen.bricks)
            runs.append({
                "caption": caption,
                "seed": seed,
                "initial_inventory": budget,
                # What the prompt listed. Recorded, not assumed: the gate and
                # the block are only useful as evidence if they demonstrably
                # started from the same numbers.
                "prompt_inventory": gate.opening_inventory,
                "prompt_matches_gate": gate.opening_inventory == budget,
                "raw_output": gen.text,
                "n_tokens": gen.n_tokens,
                "seconds": round(gen.seconds, 2),
                "termination": gate.stop_reason,
                "n_bricks": len(gen.bricks),
                "accepted": gate.accepted,
                "used": dict(used),
                "remaining": inv.as_dict(),
                "parse_ok": not gen.unparsed,
                "unparsed": gen.unparsed,
                "grammar_tokens_checked": gate.tokens,
                "grammar_violations": gate.violations,
                "type_violations": type_viol,
                "count_violations": count_viol,
                "bricks_match_accepted": len(gen.bricks) == len(gate.accepted),
                "collisions": len(find_collisions(gen.bricks)),
                "components": len(connected_components(gen.bricks, ground=True)),
                "connected": is_connected(gen.bricks, ground=True),
                "unsupported": len(unsupported_bricks(gen.bricks)),
                "ldr": str((LDR_DIR / f"{name}.ldr").relative_to(ROOT)),
            })
            print(f"  {caption[:14]:16s} seed {seed}: {len(gen.bricks):3d} bricks, "
                  f"{gate.stop_reason:22s} grammar_viol={len(gate.violations)}")

    n = len(runs)
    tok = sum(r["grammar_tokens_checked"] for r in runs)
    gv = sum(len(r["grammar_violations"]) for r in runs)
    tv = sum(1 for r in runs if r["type_violations"])
    cv = sum(1 for r in runs if r["count_violations"])
    pv = sum(1 for r in runs if r["parse_ok"])
    mv = sum(1 for r in runs if r["bricks_match_accepted"])
    pm = sum(1 for r in runs if r["prompt_matches_gate"])

    L = ["# D arm: hard inventory gate", ""]
    L.append("Compliance measurement of the constraint layer on the base "
             "checkpoint. **Not** an A/B/C/D/E comparison -- that needs a fixed "
             "prompt set, matched physics settings and multiple samples per arm.")
    L += ["", "The prompt states the same inventory the gate enforces, both "
          "taken from one snapshot before decoding starts. An earlier version "
          "of this script built the gate by hand and left the block off, so it "
          "measured a hard gate over an unconditioned prompt -- a configuration "
          "no arm uses. The counts below were produced after that was fixed."]
    L += ["", "## Environment", ""]
    for k, v in env.items():
        L.append(f"- `{k}`: {v}")
    L += ["", "## Compliance", "",
          f"- runs: {n} ({len(CASES)} inventories x {len(SEEDS)} seeds)",
          f"- tokens audited against their slot: {tok}",
          f"- **grammar violations: {gv}**",
          f"- **type-compliance failures: {tv}/{n}**",
          f"- **count-compliance failures: {cv}/{n}**",
          f"- parse rate: {pv}/{n}",
          f"- parsed bricks == gate ledger: {mv}/{n}",
          f"- prompt block == gate opening ledger: {pm}/{n}",
          f"- wall clock: {time.time()-t0:.0f}s", ""]
    L += ["## Per run", "",
          "| caption | seed | bricks | termination | remaining | coll | comps | unsup |",
          "|---|---:|---:|---|---|---:|---:|---:|"]
    for r in runs:
        L.append(f"| {r['caption']} | {r['seed']} | {r['n_bricks']} | "
                 f"{r['termination']} | {r['remaining'] or '(empty)'} | "
                 f"{r['collisions']} | {r['components']} | {r['unsupported']} |")
    L += ["", "Collisions, components and unsupported bricks are *not* gated by "
          "this layer -- placement legality belongs to the rejection stage, which "
          "is not implemented yet. They are recorded to show what the inventory "
          "gate alone does and does not buy.", ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "05_d_arm.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "05_d_arm.json").write_text(
        json.dumps({"env": env, "runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\n" + "\n".join(L[:40]))


if __name__ == "__main__":
    main()
