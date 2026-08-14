"""Reproducer: sparse-distribution sampling on MPS draws outside the support.

Constrained decoding masks all but a handful of a 128k-entry vocabulary. If the
sampler can return an index whose probability is zero, the grammar and the
inventory gate are both unsound -- so this measures whether it does, per device.

Scope: the result below describes *this* machine, torch build and dtype only.
It is not a claim about Apple Silicon in general, about other torch versions,
or about other sampling paths. Re-run it after any torch upgrade.

Writes data/reports/06_mps_multinomial.md.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports"

VOCAB = 128_256          # Llama-3.2 vocabulary
K = 20                   # live candidates after masking, as in a coordinate slot
TRIALS = 4000
SEED = 0


def env() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }


def run(device: str, dtype: torch.dtype) -> dict:
    g = torch.Generator().manual_seed(SEED)
    allowed = torch.randperm(VOCAB, generator=g)[:K]
    values = torch.randn(K, generator=g)

    scores = torch.full((1, VOCAB), float("-inf"), dtype=dtype, device=device)
    scores[0, allowed] = values.to(device).to(dtype)
    probs = torch.softmax(scores.float(), dim=-1)

    allowed_set = set(allowed.tolist())
    outside = 0
    for _ in range(TRIALS):
        if int(torch.multinomial(probs, 1).item()) not in allowed_set:
            outside += 1
    return {
        "device": device,
        "dtype": str(dtype),
        "trials": TRIALS,
        "outside_support": outside,
        "rate": outside / TRIALS,
    }


def run_restricted(device: str, dtype: torch.dtype) -> dict:
    """The fix: normalise over the candidates only, and draw on CPU."""
    sys.path.insert(0, str(ROOT))
    from src.generation.brickgpt import sample

    g = torch.Generator().manual_seed(SEED)
    allowed = torch.randperm(VOCAB, generator=g)[:K].tolist()
    logits = torch.randn(VOCAB, generator=g).to(device).to(dtype)
    outside = sum(
        1 for _ in range(TRIALS) if sample(logits, allowed, 0.6) not in set(allowed)
    )
    return {
        "device": device,
        "dtype": str(dtype),
        "trials": TRIALS,
        "outside_support": outside,
        "rate": outside / TRIALS,
        "method": "restricted-then-normalise (CPU draw)",
    }


def main() -> None:
    e = env()
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    masked = [run(d, torch.bfloat16) for d in devices]
    fixed = [run_restricted(d, torch.bfloat16) for d in devices]

    L = ["# Sparse-distribution sampling by device", ""]
    L.append("## Environment")
    L.append("")
    for k, v in e.items():
        L.append(f"- `{k}`: {v}")
    L += ["", f"## Setup", "",
          f"- vocabulary {VOCAB}, {K} candidates left finite after masking",
          f"- {TRIALS} draws per device, seed {SEED}, dtype bfloat16",
          ""]
    L += ["## Result: mask the full row, then sample (what `generate` does)", "",
          "| device | draws | outside support | rate |", "|---|---:|---:|---:|"]
    for r in masked:
        L.append(f"| `{r['device']}` | {r['trials']} | **{r['outside_support']}** | "
                 f"{r['rate']:.2%} |")
    L += ["", "## Result: restrict to candidates, then normalise (what this project does)",
          "", "| device | draws | outside support | rate |", "|---|---:|---:|---:|"]
    for r in fixed:
        L.append(f"| `{r['device']}` | {r['trials']} | **{r['outside_support']}** | "
                 f"{r['rate']:.2%} |")
    L += ["", "## Reading this", "",
          "A brick is ten tokens. At the masked-row rate measured above, a "
          "60-brick structure draws 600 times, so even a fraction of a percent "
          "corrupts a large share of generations; the observed symptom was a "
          "coordinate slot emitting a word.",
          "",
          "This measures the environment listed above and nothing else. It is "
          "not a general statement about Apple Silicon, other torch releases, "
          "or other samplers. Re-run after upgrading torch.",
          ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "06_mps_multinomial.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "06_mps_multinomial.json").write_text(
        json.dumps({"env": e, "masked_row": masked, "restricted": fixed}, indent=2),
        encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
