"""BrickGPT inference with syntax-constrained decoding.

BrickGPT is published as a LoRA adapter over ``meta-llama/Llama-3.2-1B-Instruct``
(gated -- the base model needs an accepted licence and ``hf auth login``).

The brick grammar is enforced during decoding: at each step only the tokens
that can legally continue the current brick are considered, and the sample is
drawn from that short list rather than from a masked full vocabulary.  The
reference implementation masks logits too, but calls ``generate`` afresh for
every token; this keeps one KV cache across the whole structure.

Every brick is exactly ten tokens::

    <dim> x <dim> ' (' <pos> , <pos> , <pos> ')\\n'

which holds because each dimension (1-8), each coordinate (0-19) and each
literal is a single token in this tokenizer -- verified in
``tests/test_decoding.py``.  EOS is only offered at slot 0, so a brick can never
be cut in half.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.bricks import WORLD, Brick, parse_bricks
from src.generation.prompt import INSTRUCTION, build_prompt

# Model identities and pinned revisions come from src.model_ids, shared with
# the training path. Defining them here as well is how arms A/B/D and C/E would
# drift onto different base weights while each side looked internally correct.
from src.model_ids import (  # noqa: E402
    ADAPTER,
    ADAPTER_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    LOCAL_ADAPTER_MANIFEST as _LOCAL_ADAPTER_MANIFEST,
    TOKENIZER,
    TOKENIZER_REVISION,
)

MAX_DIM = 8
TOKENS_PER_BRICK = 10

# Re-exported so callers keep importing them from here; the definitions live
# in src.generation.prompt so training data and inference cannot drift apart.
__all__ = ["INSTRUCTION", "build_prompt", "BrickGPT", "BrickGate", "Slots",
           "TOKENS_PER_BRICK", "sample", "parse_output", "Generation",
           "RawGeneration"]


def _refuse_locally_trained_adapter(adapter: str) -> None:
    """Stop a locally trained adapter from being applied to bare Llama.

    Our adapters are fitted on top of the *merged* BrickGPT weights. This
    constructor stacks an adapter straight onto ``base``, which for such a
    checkpoint silently produces a wrong model -- it loads, it generates, and
    only the numbers look off. Detected by the manifest the trainer writes.
    """
    import os

    if os.path.isdir(adapter) and os.path.exists(
        os.path.join(adapter, _LOCAL_ADAPTER_MANIFEST)
    ):
        raise ValueError(
            f"{adapter} is a locally trained adapter fitted on top of the "
            "merged BrickGPT weights. BrickGPT(adapter=...) would apply it to "
            "the bare base model instead, which fails silently. Use "
            "src.training.lora.load_finetuned() for these checkpoints."
        )


#: The three files a fast tokenizer needs, and the only ones this loader asks
#: for. ``config.json`` is deliberately not among them -- see below.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json")


def _tokenizer_config(name: str, revision: str | None, *,
                      local_files_only: bool) -> dict:
    """Read the repo's own ``tokenizer_config.json``, locally if required."""
    import json

    local = os.path.join(name, "tokenizer_config.json")
    if os.path.isdir(name):
        if not os.path.exists(local):
            raise OSError(f"{name} has no tokenizer_config.json")
        with open(local, encoding="utf-8") as fh:
            return json.load(fh)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(name, "tokenizer_config.json", revision=revision,
                           local_files_only=local_files_only)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def declared_tokenizer_class(name: str = TOKENIZER,
                             revision: str | None = TOKENIZER_REVISION, *,
                             local_files_only: bool = False):
    """The tokenizer class the repo declares for itself.

    ``AutoTokenizer`` works this out by first fetching the *model's*
    ``config.json``. BrickGPT is an adapter-only repo and has never published
    one, so online that request 404s and Auto quietly falls back -- and
    offline it cannot tell a 404 from an unreachable server, so it raises.
    That is what killed report 16's first measured run 51 seconds in, with the
    boot already spent.

    The repo does publish ``tokenizer_config.json``, and it names its own
    class. Reading that is both cheaper and honest: nothing is inferred from a
    file the repo does not have.
    """
    import transformers

    declared = _tokenizer_config(name, revision,
                                 local_files_only=local_files_only)
    named = declared.get("tokenizer_class") or "PreTrainedTokenizerFast"
    cls = getattr(transformers, named, None)
    if cls is None:
        raise OSError(f"{name} declares tokenizer_class {named!r}, which this "
                      "transformers does not provide")
    return cls


def load_tokenizer(name: str = TOKENIZER,
                   revision: str | None = TOKENIZER_REVISION, *,
                   local_files_only: bool = False):
    """Resolve the tokenizer without ever asking for a model ``config.json``.

    ``local_files_only=True`` is the contract report 16's measured child runs
    under: every byte must already be in the local cache, and no request may
    leave the process. It is passed explicitly rather than read from the
    environment, because "did the operator remember to export HF_HUB_OFFLINE"
    is not a property a measurement should depend on.

    ``local_files_only=False`` is the ordinary path for everything else, and
    may download on first use.
    """
    cls = declared_tokenizer_class(name, revision,
                                   local_files_only=local_files_only)
    kwargs = {"local_files_only": local_files_only}
    if revision is not None and not os.path.isdir(name):
        kwargs["revision"] = revision
    try:
        return cls.from_pretrained(name, **kwargs)
    except Exception as e:
        if local_files_only:
            raise OSError(
                f"tokenizer {name!r} @ {revision or 'main'} is not fully in "
                f"the local cache; strict offline loading needs "
                f"{', '.join(TOKENIZER_FILES)}. Populate the cache once with "
                "an online run, or pass a local tokenizer directory."
            ) from e
        raise


@dataclass
class Slots:
    """Token ids allowed at each of the ten slots of a brick line."""

    dims: list[int]
    posns: list[int]
    literal_x: int
    literal_open: int
    literal_comma: int
    literal_close: int
    eos: int

    @classmethod
    def build(cls, tok) -> "Slots":
        def one(s: str) -> int:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{s!r} is not a single token ({len(ids)})")
            return ids[0]

        return cls(
            dims=[one(str(i)) for i in range(1, MAX_DIM + 1)],
            posns=[one(str(i)) for i in range(WORLD)],
            literal_x=one("x"),
            literal_open=one(" ("),
            literal_comma=one(","),
            literal_close=one(")\n"),
            eos=tok.eos_token_id,
        )

    def allowed(self, slot: int) -> list[int]:
        match slot:
            case 0:
                return self.dims + [self.eos]
            case 1:
                return [self.literal_x]
            case 2:
                return self.dims
            case 3:
                return [self.literal_open]
            case 4 | 6 | 8:
                return self.posns
            case 5 | 7:
                return [self.literal_comma]
            case 9:
                return [self.literal_close]
        raise ValueError(slot)


def sample(logits: torch.Tensor, allowed: list[int], temperature: float) -> int:
    """Sample one token from ``allowed`` only.

    Restricting first and normalising over the ~20 survivors, rather than
    masking a 128k-wide row and sampling from that, is a correctness
    requirement and not an optimisation: on the measured configuration,
    ``torch.multinomial`` on MPS draws outside the support of a sparse
    distribution a small but non-zero fraction of the time, while CPU is exact.
    At ten tokens a brick that is enough to corrupt a large share of
    generations. The draw itself runs on CPU. Current rates, and the exact
    environment they apply to, are in scripts/06_mps_multinomial_repro.py.
    """
    sub = logits[allowed].float() / max(temperature, 1e-6)
    probs = torch.softmax(sub, dim=-1).cpu()
    return allowed[int(torch.multinomial(probs, 1).item())]


class BrickGate:
    """Decides which tokens may follow, one slot at a time.

    Subclasses add semantics on top of the grammar; ``on_brick`` fires once a
    brick's ten tokens are complete.
    """

    #: Termination reasons, per the workflow definition.
    STOP_REASONS = ("normal_eos", "inventory_exhausted", "max_bricks", "max_tokens")

    def __init__(self, slots: Slots):
        self.slots = slots
        self.stop_reason = "running"

    def allowed(self, slot: int, out: list[int]) -> list[int]:
        """Legal token ids at ``slot``; ``out`` is everything generated so far."""
        return self.slots.allowed(slot)

    def on_brick(self, h: int, w: int) -> None:
        """Called once a brick's ten tokens are complete."""
        return None


@dataclass
class RawGeneration:
    """What the decoder produced, before anything interprets it.

    The two layers exist because they run on different machines. Decoding
    needs the GPU; the parser, the checkers and the LDraw writer need none of
    it, and the project's rule is that a result is scored on the Mac. So the
    execution node produces exactly this -- the text, how many tokens it cost,
    how long it took, why it stopped -- and forms no opinion about whether any
    of it is a brick.

    Keeping the parse out of the node is not tidiness. A parse that happens
    there happens under a different transformers, a different Python and a
    different filesystem from the one that will be quoted in the report, and
    the difference would be invisible: both sides produce a brick list.
    """

    text: str
    n_tokens: int
    seconds: float
    truncated: bool
    termination: str


@dataclass
class Generation:
    text: str
    bricks: list[Brick]
    n_tokens: int
    seconds: float
    truncated: bool
    unparsed: list[str] = field(default_factory=list)


class BrickGPT:
    def __init__(
        self,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        adapter: str | None = ADAPTER,
        base: str = BASE_MODEL,
        base_revision: str | None = BASE_REVISION,
        adapter_revision: str | None = ADAPTER_REVISION,
        tokenizer: str = TOKENIZER,
        tokenizer_revision: str | None = TOKENIZER_REVISION,
        local_files_only: bool = False,
    ):
        """Three sources, three revisions, pinned independently.

        ``base``/``base_revision`` must match what the training path loads, or
        arms A/B/D and C/E sit on different weights and every comparison
        between them measures that too. Both sides take the value from
        :mod:`src.model_ids`.

        ``adapter``/``adapter_revision`` and ``tokenizer``/``tokenizer_revision``
        are independent on purpose. A local checkpoint directory is a valid
        adapter but carries no tokenizer files and has no revision to pin, so
        it is passed as ``adapter=<path>, adapter_revision=None`` while the
        tokenizer keeps resolving from the pinned published repo.
        """
        if device is None:
            device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device
        self.tokenizer = load_tokenizer(tokenizer, tokenizer_revision,
                                        local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=base_revision, dtype=dtype,
            local_files_only=local_files_only)
        if adapter:
            _refuse_locally_trained_adapter(adapter)
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, adapter, revision=adapter_revision,
                local_files_only=local_files_only)
        self.model = model.to(device).eval()
        self.slots = Slots.build(self.tokenizer)

    @classmethod
    def from_loaded(cls, model, tokenizer, *, device: str | None = None
                    ) -> "BrickGPT":
        """Wrap weights somebody else resolved, rather than resolving any.

        Arms C and E run a locally trained adapter, and the only correct way
        to build one is ``src.training.lora.load_finetuned``: base, then the
        published adapter, then the merge, then the local delta. The
        constructor above cannot express that -- it stacks an adapter onto the
        bare base -- and reaching for ``BrickGPT(adapter=<local path>)``
        because it is the obvious thing produces a model that loads, generates
        and is wrong. ``_refuse_locally_trained_adapter`` catches that one
        spelling; this is the route that makes the mistake unnecessary.

        The same door serves arms B and D, which take
        ``load_merged_brickgpt()``. That matters more than it looks: B/D and
        C/E then differ by the local delta alone, rather than by the delta
        *and* whether the published adapter was merged or left as a PEFT
        wrapper.

        Nothing is downloaded, moved between devices or cast here. ``device``
        is read from the weights unless the caller states it, and a stated
        value that disagrees is a refusal -- a tensor built on the wrong
        device is a silent copy, not an error.
        """
        obj = cls.__new__(cls)
        try:
            actual = str(next(model.parameters()).device)
        except StopIteration:                    # a stand-in with no weights
            actual = None
        if device is not None and actual is not None \
                and not actual.startswith(device):
            raise ValueError(
                f"the model is on {actual!r} and the caller asked for "
                f"{device!r}; refusing to generate on a device the weights "
                "are not on")
        obj.device = device or actual or "cpu"
        obj.tokenizer = tokenizer
        obj.model = model.eval() if hasattr(model, "eval") else model
        obj.slots = Slots.build(tokenizer)
        return obj

    def encode(
        self, caption: str, inventory: dict[str, int] | None = None
    ) -> torch.Tensor:
        """Tokenise the prompt. With ``inventory`` this is the B-E prompt, and
        it is byte-identical to what the training data contains."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(caption, inventory)}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return enc["input_ids"].to(self.device)

    def generate_raw(
        self,
        caption: str,
        *,
        inventory: dict[str, int] | None = None,
        max_bricks: int = 120,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        seed: int | None = None,
        gate: BrickGate | None = None,
    ) -> RawGeneration:
        """Decode under ``gate``, one token at a time with a KV cache.

        Written as an explicit loop rather than ``model.generate`` with a
        ``LogitsProcessor``: the framework path samples from the full masked
        vocabulary, which is not reliable on MPS (see :func:`sample`), and the
        loop is also what the per-brick rejection layer will hook into.
        """
        import time

        if seed is not None:
            torch.manual_seed(seed)
        gate = gate or BrickGate(self.slots)

        ids = self.encode(caption, inventory)
        eos = self.slots.eos
        dim_value = {tid: i + 1 for i, tid in enumerate(self.slots.dims)}

        brick_budget = max_bricks * TOKENS_PER_BRICK
        budget = brick_budget if max_tokens is None else min(brick_budget, max_tokens)

        out: list[int] = []
        cur = ids
        past = None
        # A monotonic clock, not the wall clock. ``time.time()`` follows the
        # system clock, and the system clock is adjusted: NTP steps it, and a
        # WSL2 guest has it re-synchronised against the Windows host often
        # enough that a decode can finish before it started. That is not a
        # hypothetical -- one cell of one measured step came back with
        # ``seconds = -0.5653038024902344`` and the sealer refused the step,
        # correctly. ``perf_counter()`` cannot go backwards, so the duration
        # is the duration whatever happens to the calendar mid-decode.
        t = time.perf_counter()

        with torch.no_grad():
            for step in range(budget):
                res = self.model(input_ids=cur, past_key_values=past, use_cache=True)
                past = res.past_key_values
                logits = res.logits[0, -1, :]

                slot = step % TOKENS_PER_BRICK
                allowed = gate.allowed(slot, out)
                tok = sample(logits, allowed, temperature)
                out.append(tok)
                if tok == eos:
                    if gate.stop_reason == "running":
                        gate.stop_reason = "normal_eos"
                    break
                # Account the brick as soon as its tenth token lands, not at
                # the next slot 0: hitting the token budget mid-structure used
                # to leave the final brick unbilled, so accepted/used and the
                # parsed output disagreed by one.
                if slot == TOKENS_PER_BRICK - 1:
                    gate.on_brick(dim_value[out[-TOKENS_PER_BRICK]],
                                  dim_value[out[-TOKENS_PER_BRICK + 2]])
                cur = torch.tensor([[tok]], device=self.device)

        seconds = time.perf_counter() - t
        if gate.stop_reason == "running":
            # Which ceiling was hit matters: max_bricks means the structure
            # budget ran out, max_tokens that a raw token cap cut it short.
            gate.stop_reason = (
                "max_bricks" if budget == brick_budget else "max_tokens"
            )

        text = self.tokenizer.decode(out, skip_special_tokens=True)
        return RawGeneration(
            text=text,
            n_tokens=len(out),
            seconds=seconds,
            truncated=len(out) >= budget,
            termination=gate.stop_reason,
        )

    def generate(
        self,
        caption: str,
        *,
        inventory: dict[str, int] | None = None,
        max_bricks: int = 120,
        max_tokens: int | None = None,
        temperature: float = 0.6,
        seed: int | None = None,
        gate: BrickGate | None = None,
    ) -> Generation:
        """Decode and parse, exactly as before.

        The signature, the return type and every field of it are unchanged:
        this is :meth:`generate_raw` followed by :func:`parse_output`, which
        is what the body used to do inline. Callers that want the two halves
        on two machines take the halves; callers that want a brick list keep
        calling this.
        """
        raw = self.generate_raw(
            caption, inventory=inventory, max_bricks=max_bricks,
            max_tokens=max_tokens, temperature=temperature, seed=seed,
            gate=gate)
        bricks, unparsed = parse_output(raw.text)
        return Generation(
            text=raw.text,
            bricks=bricks,
            n_tokens=raw.n_tokens,
            seconds=raw.seconds,
            truncated=raw.truncated,
            unparsed=unparsed,
        )


_BRICK_RE = re.compile(r"\d+x\d+\s*\(\d+,\d+,\d+\)")


def parse_output(text: str) -> tuple[list[Brick], list[str]]:
    """Salvage every well-formed brick line; report the rest.

    Unconstrained baselines emit prose and malformed lines, and the parse rate
    is itself a reported metric, so failures are collected rather than raised.
    """
    bricks: list[Brick] = []
    unparsed: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _BRICK_RE.fullmatch(line)
        if m:
            bricks.extend(parse_bricks(line, strict=False))
        else:
            unparsed.append(line)
    return bricks, unparsed
