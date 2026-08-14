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
from src.data.bricks import required_inventory
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
            def from_pretrained(name, revision=None):
                seen["tokenizer"] = (name, revision)
                return _StubTokenizer()

        class _StubTokenizer:
            eos_token_id = 0

            def encode(self, s, add_special_tokens=False):
                return [1]

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                seen["base"] = name
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        monkeypatch.setattr(B, "AutoTokenizer", FakeTok)
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
            def from_pretrained(name, revision=None):
                if str(name).startswith(str(tmp_path)):
                    raise OSError("no tokenizer files in a local adapter dir")
                calls["tokenizer"] = (name, revision)
                return type("T", (), {"eos_token_id": 0})()

        class FakeModel:
            @staticmethod
            def from_pretrained(name, revision=None, dtype=None):
                calls["base"] = name
                return FakeModel()

            def to(self, d):
                return self

            def eval(self):
                return self

        class FakePeft:
            @staticmethod
            def from_pretrained(model, adapter, revision=None):
                calls["adapter"] = (adapter, revision)
                return model

        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeft
        monkeypatch.setitem(sys.modules, "peft", peft)
        monkeypatch.setattr(B, "AutoTokenizer", FakeTok)
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
    """Offline with a cold cache is the common way this goes wrong.

    The hub's own error arrives after its retry and fallback path and talks
    about connections; the actionable fact is that nothing is cached. These
    tests pin the fast, specific failure rather than the generic one.
    """

    def test_offline_miss_names_the_cache_and_the_flag(self, monkeypatch):
        import src.generation.brickgpt as B

        class Boom:
            @staticmethod
            def from_pretrained(name, revision=None):
                raise OSError("Connection error, and we cannot find the file")

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(B, "AutoTokenizer", Boom)

        with pytest.raises(OSError) as e:
            B.load_tokenizer("AvaLovelace/BrickGPT", "deadbeef")
        msg = str(e.value)
        assert "HF_HUB_OFFLINE" in msg
        assert "cache" in msg
        assert "AvaLovelace/BrickGPT" in msg and "deadbeef" in msg

    def test_online_error_is_not_reworded(self, monkeypatch):
        """Without the flag the original exception must pass through intact."""
        import src.generation.brickgpt as B

        class Boom:
            @staticmethod
            def from_pretrained(name, revision=None):
                raise ValueError("gated repo: accept the licence")

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(B, "AutoTokenizer", Boom)

        with pytest.raises(ValueError, match="gated repo"):
            B.load_tokenizer()

    def test_offline_zero_counts_as_online(self, monkeypatch):
        import src.generation.brickgpt as B

        class Boom:
            @staticmethod
            def from_pretrained(name, revision=None):
                raise ValueError("some other problem")

        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setattr(B, "AutoTokenizer", Boom)

        with pytest.raises(ValueError, match="some other problem"):
            B.load_tokenizer()

    def test_a_cached_tokenizer_is_returned_unchanged(self, monkeypatch):
        import src.generation.brickgpt as B

        sentinel = object()

        class Ok:
            @staticmethod
            def from_pretrained(name, revision=None):
                return sentinel

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(B, "AutoTokenizer", Ok)
        assert B.load_tokenizer() is sentinel
