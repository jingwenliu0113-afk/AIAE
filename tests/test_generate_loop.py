"""Integration tests for the decode loop, without the base model.

A scripted stand-in supplies logits, so the loop, the gate and the bookkeeping
are exercised end to end while only the (small, ungated) tokenizer is loaded.
The point is that ``accepted``/``used``/``remaining`` and the text that comes
out of the parser agree on every termination path -- especially the token
budget, where the last brick used to go unbilled.
"""

import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constraints.inventory_decode import InventoryGate
from src.data.bricks import WORLD, required_inventory
from src.generation.brickgpt import TOKENS_PER_BRICK, BrickGPT, Slots
from src.generation.prompt import build_prompt
from src.inventory.engine import Inventory

VOCAB = 128_256


class ScriptedModel:
    """Returns logits that strongly prefer the next token in ``script``."""

    def __init__(self, script: list[int]):
        self.script = list(script)
        self.calls = 0

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        want = self.script[self.calls] if self.calls < len(self.script) else None
        self.calls += 1
        logits = torch.full((1, 1, VOCAB), -1e4)
        if want is not None:
            logits[0, 0, want] = 1e4
        return type("Out", (), {"logits": logits, "past_key_values": None})()


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("AvaLovelace/BrickGPT")
    except Exception as e:                      # offline with a cold cache
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")


@pytest.fixture(scope="module")
def slots(tok):
    return Slots.build(tok)


def make_gpt(tok, slots, script) -> BrickGPT:
    gpt = object.__new__(BrickGPT)
    gpt.device = "cpu"
    gpt.tokenizer = tok
    gpt.slots = slots
    gpt.model = ScriptedModel(script)
    return gpt


def brick_tokens(slots: Slots, h: int, w: int, x: int, y: int, z: int) -> list[int]:
    return [
        slots.dims[h - 1], slots.literal_x, slots.dims[w - 1], slots.literal_open,
        slots.posns[x], slots.literal_comma, slots.posns[y],
        slots.literal_comma, slots.posns[z], slots.literal_close,
    ]


def check_consistent(gen, gate, initial: dict[str, int]) -> None:
    """Parsed output, the gate's ledger and the inventory must all agree."""
    parsed = required_inventory(gen.bricks)
    assert len(gen.bricks) == len(gate.accepted), "parsed bricks vs accepted"
    assert parsed == Counter(gate.accepted), "parsed parts vs accepted parts"
    for part, n0 in initial.items():
        assert gate.inventory.available(part) == n0 - parsed.get(part, 0), part
    assert gen.unparsed == [], gen.unparsed


class TestNormalEos:
    def test_stops_and_bills_every_brick(self, tok, slots):
        script = (
            brick_tokens(slots, 2, 4, 0, 0, 0)
            + brick_tokens(slots, 2, 4, 4, 0, 0)
            + [slots.eos]
        )
        initial = {"2x4": 5}
        gate = InventoryGate(slots, Inventory.from_parts(initial))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=10, temperature=0.6
        )
        assert gate.stop_reason == "normal_eos"
        assert gate.accepted == ["2x4", "2x4"]
        assert gate.inventory.as_dict() == {"2x4": 3}
        check_consistent(gen, gate, initial)


class TestInventoryExhausted:
    def test_budget_spent_exactly(self, tok, slots):
        script = (
            brick_tokens(slots, 2, 4, 0, 0, 0)
            + brick_tokens(slots, 4, 2, 4, 0, 0)     # rotated spelling
        )
        initial = {"2x4": 2}
        gate = InventoryGate(slots, Inventory.from_parts(initial))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=10, temperature=0.6
        )
        assert gate.stop_reason == "inventory_exhausted"
        assert gate.accepted == ["2x4", "2x4"]
        assert gate.inventory.total() == 0
        check_consistent(gen, gate, initial)

    def test_eos_is_forced_once_empty(self, tok, slots):
        """The script keeps trying to build; the gate must refuse."""
        script = [
            t for _ in range(4) for t in brick_tokens(slots, 1, 1, 0, 0, 0)
        ]
        gate = InventoryGate(slots, Inventory.from_parts({"1x1": 1}))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=10, temperature=0.6
        )
        assert len(gen.bricks) == 1
        assert gate.stop_reason == "inventory_exhausted"


class TestTokenBudget:
    def test_last_brick_is_billed(self, tok, slots):
        """Regression: the final brick went unbilled when the budget ran out.

        Accounting happened at the following slot 0, which never arrives when
        the loop stops on ``max_bricks``.
        """
        script = (
            brick_tokens(slots, 2, 4, 0, 0, 0)
            + brick_tokens(slots, 2, 4, 4, 0, 0)
        )
        initial = {"2x4": 9}
        gate = InventoryGate(slots, Inventory.from_parts(initial))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=2, temperature=0.6
        )
        assert gate.stop_reason == "max_bricks"
        assert gen.truncated
        assert len(gen.bricks) == 2
        assert gate.accepted == ["2x4", "2x4"], "final brick must be accounted"
        assert gate.inventory.as_dict() == {"2x4": 7}
        check_consistent(gen, gate, initial)

    def test_single_brick_budget(self, tok, slots):
        script = brick_tokens(slots, 1, 2, 0, 0, 0)
        initial = {"1x2": 3}
        gate = InventoryGate(slots, Inventory.from_parts(initial))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=1, temperature=0.6
        )
        assert len(gen.bricks) == 1 and gate.accepted == ["1x2"]
        check_consistent(gen, gate, initial)


class TestGrammarHolds:
    def test_output_parses_on_every_path(self, tok, slots):
        for max_bricks, script in [
            (10, brick_tokens(slots, 2, 2, 0, 0, 0) + [slots.eos]),
            (1, brick_tokens(slots, 2, 2, 0, 0, 0)),
        ]:
            gate = InventoryGate(slots, Inventory.from_parts({"2x2": 4}))
            gen = make_gpt(tok, slots, script).generate(
                "x", gate=gate, max_bricks=max_bricks, temperature=0.6
            )
            assert gen.unparsed == []
            assert gen.n_tokens % TOKENS_PER_BRICK in (0, 1)  # +1 only for EOS

    def test_rotated_output_bills_canonical_part(self, tok, slots):
        script = brick_tokens(slots, 8, 1, 0, 0, 0) + [slots.eos]
        gate = InventoryGate(slots, Inventory.from_parts({"1x8": 2}))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=5, temperature=0.6
        )
        assert gate.accepted == ["1x8"]
        assert gen.bricks[0].h == 8 and gen.bricks[0].w == 1
        assert gate.inventory.as_dict() == {"1x8": 1}


class TestTerminationReasons:
    """The four reasons are distinct; max_bricks must not be logged as
    max_tokens. Hitting the structure budget and hitting a raw token cap are
    different events and the reports separate them."""

    def test_reason_vocabulary(self):
        from src.generation.brickgpt import BrickGate

        assert BrickGate.STOP_REASONS == (
            "normal_eos", "inventory_exhausted", "max_bricks", "max_tokens",
        )

    def test_brick_budget_reports_max_bricks(self, tok, slots):
        script = brick_tokens(slots, 2, 2, 0, 0, 0) * 2
        gate = InventoryGate(slots, Inventory.from_parts({"2x2": 9}))
        make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=2, temperature=0.6
        )
        assert gate.stop_reason == "max_bricks"

    def test_token_cap_reports_max_tokens(self, tok, slots):
        """A tighter token cap bites first and must be named as such."""
        script = brick_tokens(slots, 2, 2, 0, 0, 0) * 3
        gate = InventoryGate(slots, Inventory.from_parts({"2x2": 9}))
        gen = make_gpt(tok, slots, script).generate(
            "x", gate=gate, max_bricks=10, max_tokens=15, temperature=0.6
        )
        assert gate.stop_reason == "max_tokens"
        assert gen.n_tokens == 15

    def test_every_reason_is_in_the_vocabulary(self, tok, slots):
        from src.generation.brickgpt import BrickGate

        seen = set()
        for kw, script, stock in [
            (dict(max_bricks=5), brick_tokens(slots, 2, 2, 0, 0, 0) + [slots.eos], 4),
            (dict(max_bricks=1), brick_tokens(slots, 2, 2, 0, 0, 0), 4),
            (dict(max_bricks=5, max_tokens=12), brick_tokens(slots, 2, 2, 0, 0, 0) * 2, 4),
            (dict(max_bricks=5), brick_tokens(slots, 2, 2, 0, 0, 0) * 3, 1),
        ]:
            gate = InventoryGate(slots, Inventory.from_parts({"2x2": stock}))
            make_gpt(tok, slots, script).generate("x", gate=gate, temperature=0.6, **kw)
            seen.add(gate.stop_reason)
        assert seen == set(BrickGate.STOP_REASONS)


class TestPromptMatchesTheGate:
    """The stock the model is told about must be the stock the gate enforces.

    generate_with_inventory previously built the prompt without an inventory
    block at all, so the constrained arm was decoding against an arm-A prompt:
    the gate refused illegal bricks, but the model was never told what it had.
    """

    def test_prompt_lists_exactly_the_gate_opening_ledger(self, tok, slots):
        from src.constraints.inventory_decode import generate_with_inventory
        from src.generation.prompt import INVENTORY_HEADER

        stock = {"2x4": 3, "1x2": 5, "1x8": 2}
        gpt = make_gpt(tok, slots, brick_tokens(slots, 2, 4, 0, 0, 0) + [slots.eos])
        seen = {}
        real_encode = gpt.encode

        def spy(caption, inventory=None):
            seen["inventory"] = inventory
            return real_encode(caption, inventory)

        gpt.encode = spy
        _gen, gate = generate_with_inventory(
            gpt, "A chair.", Inventory.from_parts(stock),
            max_bricks=5, temperature=0.6,
        )
        assert seen["inventory"] == stock
        assert gate.opening_inventory == stock
        prompt = build_prompt("A chair.", seen["inventory"])
        assert INVENTORY_HEADER in prompt
        for part, n in stock.items():
            assert f"{part}: {n}" in prompt

    def test_snapshot_is_taken_before_the_gate_spends_anything(self, tok, slots):
        """The gate consumes stock as it decodes; the prompt must predate that."""
        from src.constraints.inventory_decode import generate_with_inventory

        stock = {"2x4": 2}
        script = brick_tokens(slots, 2, 4, 0, 0, 0) + brick_tokens(slots, 4, 2, 4, 0, 0)
        gpt = make_gpt(tok, slots, script)
        seen = {}
        real_encode = gpt.encode
        gpt.encode = lambda c, inventory=None: (
            seen.setdefault("inventory", inventory), real_encode(c, inventory)
        )[1]
        _gen, gate = generate_with_inventory(
            gpt, "A chair.", Inventory.from_parts(stock),
            max_bricks=5, temperature=0.6,
        )
        assert seen["inventory"] == {"2x4": 2}      # opening position
        assert gate.inventory.total() == 0          # fully spent by the end
        assert gate.opening_inventory == {"2x4": 2}

    def test_arm_a_has_no_block_and_b_to_e_do(self, tok, slots):
        from src.generation.prompt import INVENTORY_HEADER

        gpt = make_gpt(tok, slots, [slots.eos])
        a = gpt.encode("A chair.")
        b = gpt.encode("A chair.", {"2x4": 3})
        assert INVENTORY_HEADER not in build_prompt("A chair.")
        assert INVENTORY_HEADER in build_prompt("A chair.", {"2x4": 3})
        assert b.shape[1] > a.shape[1]

    def test_a_gate_subclass_still_gets_the_block(self, tok, slots):
        """The D-arm eval audits tokens with an InventoryGate subclass.

        It used to construct that subclass itself and call ``generate``
        directly, which is how it ended up decoding against an arm-A prompt
        even after ``generate_with_inventory`` was fixed. Subclasses go through
        the same path now, so the pairing cannot be forgotten again.
        """
        from src.constraints.inventory_decode import (
            InventoryGate,
            generate_with_inventory,
        )

        class AuditGate(InventoryGate):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.slots_seen = 0

            def allowed(self, slot, out):
                self.slots_seen += 1
                return super().allowed(slot, out)

        stock = {"2x4": 3, "1x2": 5}
        gpt = make_gpt(tok, slots, brick_tokens(slots, 2, 4, 0, 0, 0) + [slots.eos])
        seen = {}
        real_encode = gpt.encode
        gpt.encode = lambda c, inventory=None: (
            seen.setdefault("inventory", inventory), real_encode(c, inventory)
        )[1]
        _gen, gate = generate_with_inventory(
            gpt, "A chair.", Inventory.from_parts(stock),
            gate_cls=AuditGate, max_bricks=5, temperature=0.6,
        )
        assert isinstance(gate, AuditGate) and gate.slots_seen > 0
        assert seen["inventory"] == stock
        assert gate.opening_inventory == stock


class TestTokenizerSourceIsSeparate:
    """A locally trained adapter is saved without tokenizer files.

    Loading the tokenizer from the adapter directory would therefore fail for
    exactly the checkpoints this project is about to produce, so the two are
    addressed independently.
    """

    def test_constructor_takes_both_sources(self):
        import inspect

        params = inspect.signature(BrickGPT.__init__).parameters
        for name in ("adapter", "adapter_revision", "tokenizer",
                     "tokenizer_revision"):
            assert name in params, name
        # A bare ``revision`` would be ambiguous about which source it pins.
        assert "revision" not in params

    def test_tokenizer_defaults_do_not_follow_a_custom_adapter(self, monkeypatch):
        import src.generation.brickgpt as B

        seen = {}

        class FakeTok:
            @staticmethod
            def from_pretrained(name, revision=None, **kw):
                seen["tokenizer"] = (name, revision)
                return _StubTokenizer()

        class _StubTokenizer:
            eos_token_id = 0

            def encode(self, s, add_special_tokens=False):
                return [1]

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None, **kw):
                seen["base"] = name
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        # The seam moved: `load_tokenizer` resolves the class the repo
        # declares for itself instead of going through AutoTokenizer, which
        # would first ask an adapter repo for a model config.json it has
        # never published. What this test is about -- the tokenizer source is
        # addressed independently of the adapter -- is unchanged.
        monkeypatch.setattr(B, "declared_tokenizer_class",
                            lambda *a, **k: FakeTok)
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        B.BrickGPT(device="cpu", adapter=None,
                   tokenizer="AvaLovelace/BrickGPT", tokenizer_revision="abc123")
        assert seen["tokenizer"] == ("AvaLovelace/BrickGPT", "abc123")

    def test_local_adapter_without_tokenizer_files_still_resolves(
        self, monkeypatch, tmp_path
    ):
        """A real local adapter is loaded while the tokenizer comes from the hub.

        The earlier version of this test passed ``adapter=None``, so the adapter
        branch never ran and the separation it claimed to check was never
        exercised. Here the adapter is an actual directory holding only weights
        -- what ``save_pretrained`` on a LoRA produces -- and ``AutoTokenizer``
        raises if anyone points it there.
        """
        import src.generation.brickgpt as B

        adapter_dir = tmp_path / "checkpoints" / "lora-smoke"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
        assert not list(adapter_dir.glob("tokenizer*")), "fixture must be bare"

        calls = {}

        class FakeTok:
            @staticmethod
            def from_pretrained(name, revision=None, **kw):
                if str(name).startswith(str(tmp_path)):
                    raise OSError("no tokenizer files in a local adapter dir")
                calls["tokenizer"] = (name, revision)
                return type("T", (), {"eos_token_id": 0})()

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None, **kw):
                calls["base"] = name
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        class FakePeft:
            @staticmethod
            def from_pretrained(model, adapter, revision=None, **kw):
                calls["adapter"] = (adapter, revision)
                return model

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeft
        monkeypatch.setitem(sys.modules, "peft", peft)
        monkeypatch.setattr(B, "declared_tokenizer_class",
                            lambda *a, **k: FakeTok)
        monkeypatch.setattr(B, "AutoModelForCausalLM", FakeModel)
        monkeypatch.setattr(B.Slots, "build", classmethod(lambda cls, t: "slots"))

        B.BrickGPT(
            device="cpu",
            adapter=str(adapter_dir),
            adapter_revision=None,          # a local path has nothing to pin
            tokenizer="AvaLovelace/BrickGPT",
            tokenizer_revision=B.TOKENIZER_REVISION,
        )

        # The adapter really was loaded, from the local path, unpinned.
        assert calls["adapter"] == (str(adapter_dir), None)
        # ...and the tokenizer came from the hub at its own pinned revision.
        assert calls["tokenizer"] == ("AvaLovelace/BrickGPT", B.TOKENIZER_REVISION)
        assert calls["tokenizer"][0] != calls["adapter"][0]


class TestOfflineTokenizerFailsClearly:
    """Strict-local loading, and what it says when the cache is cold.

    **Rewritten after report 16's exp001.** These tests used to patch
    ``AutoTokenizer`` and to read ``HF_HUB_OFFLINE`` from the environment.
    Both were the problem, not the contract:

    * ``AutoTokenizer.from_pretrained`` first asks for the *model's*
      ``config.json``. BrickGPT is an adapter repo and has never published
      one, so online that 404s harmlessly and offline it is indistinguishable
      from an unreachable server. That is what killed exp001's b1, 51 seconds
      into a boot it had already spent.
    * whether a measurement may reach the network is not something to infer
      from the shell that started it. It is now an explicit argument.

    So the loader resolves the class the repo declares for itself and takes
    ``local_files_only`` from its caller, and these tests patch that.
    """

    @staticmethod
    def _declared(monkeypatch, cls):
        import src.generation.brickgpt as B
        monkeypatch.setattr(B, "declared_tokenizer_class",
                            lambda *a, **k: cls)

    def test_a_local_only_miss_names_the_repo_the_revision_and_the_cache(
            self, monkeypatch):
        import src.generation.brickgpt as B

        class Boom:
            @staticmethod
            def from_pretrained(name, **kw):
                raise OSError("Connection error, and we cannot find the file")

        self._declared(monkeypatch, Boom)
        with pytest.raises(OSError) as e:
            B.load_tokenizer("AvaLovelace/BrickGPT", "deadbeef",
                             local_files_only=True)
        msg = str(e.value)
        assert "cache" in msg
        assert "AvaLovelace/BrickGPT" in msg and "deadbeef" in msg

    def test_an_online_error_is_not_reworded(self, monkeypatch):
        """Not strictly local: the original exception passes through intact."""
        import src.generation.brickgpt as B

        class Boom:
            @staticmethod
            def from_pretrained(name, **kw):
                raise ValueError("gated repo: accept the licence")

        self._declared(monkeypatch, Boom)
        with pytest.raises(ValueError, match="gated repo"):
            B.load_tokenizer(local_files_only=False)

    def test_the_environment_does_not_decide_this(self, monkeypatch):
        """HF_HUB_OFFLINE must not change what the loader does.

        The argument decides. An environment variable that could flip a
        measured run between local and networked loading is exactly the
        dependency exp001 tripped over.
        """
        import src.generation.brickgpt as B

        seen = []

        class Spy:
            @staticmethod
            def from_pretrained(name, **kw):
                seen.append(kw.get("local_files_only"))
                return object()

        self._declared(monkeypatch, Spy)
        for value in ("0", "1"):
            monkeypatch.setenv("HF_HUB_OFFLINE", value)
            B.load_tokenizer(local_files_only=True)
            B.load_tokenizer(local_files_only=False)
        assert seen == [True, False, True, False]

    def test_a_cached_tokenizer_is_returned_unchanged(self, monkeypatch):
        import src.generation.brickgpt as B

        sentinel = object()

        class Ok:
            @staticmethod
            def from_pretrained(name, **kw):
                return sentinel

        self._declared(monkeypatch, Ok)
        assert B.load_tokenizer(local_files_only=True) is sentinel

    def test_the_loader_never_asks_for_a_model_config(self, monkeypatch):
        """The single fact exp001 turned into a terminal experiment."""
        import src.generation.brickgpt as B

        asked = []

        class Trap:
            @staticmethod
            def from_pretrained(*a, **k):
                asked.append("AutoTokenizer")
                raise AssertionError("AutoTokenizer probes config.json")

        class Ok:
            @staticmethod
            def from_pretrained(name, **kw):
                return object()

        monkeypatch.setattr(B, "AutoTokenizer", Trap)
        self._declared(monkeypatch, Ok)
        B.load_tokenizer(local_files_only=True)
        assert asked == []


class StubTokenizer:
    """Enough tokenizer for the decode loop, and no download.

    The clock tests must run on every machine, including one with a cold
    Hugging Face cache, or the thing they prove is proved nowhere. Nothing
    here touches the real vocabulary: the loop only needs ``encode`` to hand
    it a prompt tensor and ``decode`` to turn ids back into a string.
    """

    eos_token_id = 2

    def apply_chat_template(self, messages, **kw):
        return {"input_ids": torch.tensor([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def stub_slots() -> Slots:
    """Distinct ids per slot, which is all the gate and the loop require."""
    return Slots(
        dims=list(range(10, 10 + 8)),
        posns=list(range(100, 100 + WORLD)),
        literal_x=200, literal_open=201, literal_comma=202,
        literal_close=203, eos=StubTokenizer.eos_token_id,
    )


class TestDecodeIsTimedByAMonotonicClock:
    """The wall clock is adjusted; a duration measured against it is not one.

    Not a hypothetical failure mode. One cell of one measured step on the
    WSL2 execution node came back with ``seconds = -0.5653038024902344`` and
    the sealer refused to close the step, which is what a frozen validator is
    for. That machine's wall clock is re-synchronised against the Windows
    host, so a decode could finish before it started. ``perf_counter()`` is
    monotonic and cannot.
    """

    def decode(self, **kw):
        slots = stub_slots()
        script = brick_tokens(slots, 2, 4, 0, 0, 0) + [slots.eos]
        gpt = object.__new__(BrickGPT)
        gpt.device = "cpu"
        gpt.tokenizer = StubTokenizer()
        gpt.slots = slots
        gpt.model = ScriptedModel(script)
        gate = InventoryGate(slots, Inventory.from_parts({"2x4": 5}))
        return gpt.generate_raw("x", gate=gate, max_bricks=10,
                                temperature=0.6, **kw)

    def test_seconds_is_the_perf_counter_span(self, monkeypatch):
        """The number written is the difference of the two readings, as taken."""
        import time

        readings = iter([100.0, 100.25])
        monkeypatch.setattr(time, "perf_counter", lambda: next(readings))
        assert self.decode().seconds == pytest.approx(0.25)

    def test_a_wall_clock_that_jumps_backwards_cannot_make_seconds_negative(
            self, monkeypatch):
        """The exact shape of the failure that stopped the batch."""
        import time

        # Backwards by more than half a second: what produced -0.5653.
        wall = iter([1_787_000_000.0, 1_786_999_999.4])
        monkeypatch.setattr(time, "time", lambda: next(wall))
        perf = iter([500.0, 500.4])
        monkeypatch.setattr(time, "perf_counter", lambda: next(perf))

        raw = self.decode()
        assert raw.seconds > 0
        assert raw.seconds == pytest.approx(0.4)

    def test_the_decode_loop_never_reads_the_wall_clock(self, monkeypatch):
        """Not "it survives a jump" but "it never asks the clock that jumps"."""
        import time

        called = []
        monkeypatch.setattr(time, "time",
                            lambda: called.append(1) or 0.0)
        self.decode()
        assert called == []

    def test_the_source_names_only_the_monotonic_clock(self):
        """A second call site added later would reintroduce the bug."""
        import inspect

        src = inspect.getsource(BrickGPT.generate_raw)
        body = "\n".join(line for line in src.splitlines()
                         if not line.lstrip().startswith("#"))
        assert "time.perf_counter()" in body
        assert "time.time()" not in body

    def test_text_tokens_and_termination_are_unchanged_by_the_clock(
            self, monkeypatch):
        """Timing is measurement, not behaviour: same seed, same output."""
        import time

        plain = self.decode(seed=7)

        wall = iter([1_787_000_000.0, 1_786_999_990.0])
        monkeypatch.setattr(time, "time", lambda: next(wall))
        perf = iter([9.0, 9.75])
        monkeypatch.setattr(time, "perf_counter", lambda: next(perf))
        skewed = self.decode(seed=7)

        assert skewed.text == plain.text
        assert skewed.n_tokens == plain.n_tokens
        assert skewed.termination == plain.termination
        assert skewed.truncated == plain.truncated
        assert skewed.seconds == pytest.approx(0.75)
